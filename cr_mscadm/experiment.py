from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata

from repro.data import DataBundle, SplitData
from repro.diffusion import GaussianDiffusion
from repro.metrics import all_scores, crps, interval_scores
from repro.sampling import save_scenarios

from .calibration import CopulaPITCalibrator, count_rank_inversions, pit_values
from .training import load_cr_checkpoint, seed_everything


@torch.no_grad()
def generate_cr_scenarios(
    checkpoint: str | Path,
    data: DataBundle,
    *,
    split_name: str,
    scenarios: int,
    steps: int,
    eta: float,
    day_batch: int = 8,
    seed: int = 0,
    device: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload, model = load_cr_checkpoint(checkpoint, device=selected_device)
    diffusion = GaussianDiffusion(**payload["config"]["diffusion"]).to(selected_device)
    split: SplitData = getattr(data, split_name)
    seed_everything(seed)
    chunks: list[np.ndarray] = []
    for start in range(0, len(split), day_batch):
        condition = torch.from_numpy(split.condition[start : start + day_batch]).to(selected_device)
        repeated = condition[:, None].expand(-1, scenarios, -1, -1).reshape(
            -1, condition.shape[1], condition.shape[2]
        )
        residual = diffusion.sample_ddim(model, repeated, steps=steps, eta=eta)
        reconstructed = model.reconstruct(residual, repeated)
        standard = reconstructed.squeeze(-1).cpu().numpy().reshape(len(condition), scenarios, 24)
        chunks.append(standard)
    standardized = np.concatenate(chunks, axis=0)
    generated = np.clip(data.target_standardizer.inverse(standardized), 0.0, 1.0)
    metadata = {
        "model": "cr_mscadm",
        "variant": payload["variant"],
        "training_seed": int(payload["seed"]),
        "sampling_seed": int(seed),
        "split": split_name,
        "scenarios": scenarios,
        "sampling_steps": steps,
        "sampler": "ddim",
        "eta": eta,
        "checkpoint": str(Path(checkpoint).resolve()),
    }
    return generated.astype(np.float32), metadata


def calibration_error(
    scenarios: np.ndarray,
    observations: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.8, 0.9),
) -> float:
    intervals = interval_scores(scenarios, observations, np.asarray(levels))
    return float(np.mean(np.abs(intervals["ace"])))


def select_calibration_strength(
    scenarios: np.ndarray,
    observations: np.ndarray,
    *,
    strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    folds: int = 5,
    penalty_weight: float = 0.1,
) -> tuple[float, list[dict[str, float]]]:
    """Select one global shrinkage strength by validation-only cross-fitting."""

    assignments = np.arange(len(scenarios)) % folds
    records: list[dict[str, float]] = []
    for strength in strengths:
        cross_fitted = np.empty_like(scenarios)
        for fold in range(folds):
            train = assignments != fold
            held_out = ~train
            calibrator = CopulaPITCalibrator.fit(
                scenarios[train], observations[train], strength=strength
            )
            cross_fitted[held_out] = calibrator.transform(scenarios[held_out])
        score_crps = crps(cross_fitted, observations)
        score_calibration = calibration_error(cross_fitted, observations)
        records.append(
            {
                "strength": float(strength),
                "crossfit_CRPS": score_crps,
                "crossfit_calibration_error": score_calibration,
                "objective": score_crps + penalty_weight * score_calibration,
            }
        )
    selected = min(records, key=lambda item: (item["objective"], item["strength"]))
    return selected["strength"], records


def pit_ks_distance(scenarios: np.ndarray, observations: np.ndarray) -> float:
    pits = np.sort(pit_values(scenarios, observations), axis=0)
    n = len(pits)
    lower = np.arange(n, dtype=np.float64)[:, None] / n
    upper = np.arange(1, n + 1, dtype=np.float64)[:, None] / n
    distance = np.maximum(np.max(np.abs(pits - lower), axis=0), np.max(np.abs(upper - pits), axis=0))
    return float(np.mean(distance))


def ramp_crps(scenarios: np.ndarray, observations: np.ndarray) -> float:
    return crps(np.diff(scenarios, axis=2), np.diff(observations, axis=1))


def _rank_correlation(values: np.ndarray) -> np.ndarray:
    ranked = np.apply_along_axis(rankdata, 0, values)
    return np.corrcoef(ranked, rowvar=False)


def temporal_dependence_error(scenarios: np.ndarray, observations: np.ndarray) -> float:
    observed = _rank_correlation(observations)
    generated = _rank_correlation(scenarios.reshape(-1, scenarios.shape[-1]))
    mask = ~np.eye(observed.shape[0], dtype=bool)
    return float(np.sqrt(np.mean((observed[mask] - generated[mask]) ** 2)))


def extended_scores(scenarios: np.ndarray, observations: np.ndarray) -> dict[str, float]:
    scores = all_scores(scenarios, observations)
    intervals = interval_scores(scenarios, observations, np.asarray([0.5, 0.8, 0.9]))
    for level, coverage, width in zip(
        intervals["confidence"], intervals["coverage"], intervals["piaw"]
    ):
        suffix = int(round(level * 100))
        scores[f"coverage_{suffix}"] = float(coverage)
        scores[f"width_{suffix}"] = float(width)
    scores["calibration_error"] = calibration_error(scenarios, observations)
    scores["PIT_KS"] = pit_ks_distance(scenarios, observations)
    scores["ramp_CRPS"] = ramp_crps(scenarios, observations)
    scores["temporal_rank_RMSE"] = temporal_dependence_error(scenarios, observations)
    return scores


def per_day_scores(scenarios: np.ndarray, observations: np.ndarray) -> dict[str, np.ndarray]:
    mean = scenarios.mean(axis=1)
    first = np.abs(scenarios - observations[:, None]).mean(axis=1)
    pairwise = np.abs(scenarios[:, :, None] - scenarios[:, None, :]).mean(axis=(1, 2))
    day_crps = np.mean(first - pairwise / 2, axis=1)
    lower = np.quantile(scenarios, 0.05, axis=1)
    upper = np.quantile(scenarios, 0.95, axis=1)
    coverage = np.mean((observations >= lower) & (observations <= upper), axis=1)
    truth = np.abs(observations[:, :, None] - observations[:, None, :]) ** 0.5
    sampled = np.abs(scenarios[:, :, :, None] - scenarios[:, :, None, :]) ** 0.5
    variogram = np.sum((truth - sampled.mean(axis=1)) ** 2, axis=(1, 2))
    return {
        "MAE": np.mean(np.abs(mean - observations), axis=1),
        "CRPS": day_crps,
        "coverage_90": coverage,
        "VS": variogram,
    }


def paired_bootstrap(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    seed: int = 2026,
    replicates: int = 5_000,
    higher_is_better: bool = False,
) -> dict[str, float]:
    if baseline.shape != method.shape:
        raise ValueError("paired score vectors must have the same shape")
    difference = method - baseline if higher_is_better else baseline - method
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
    samples = difference[indices].mean(axis=1)
    return {
        "improvement": float(difference.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "probability_improvement": float(np.mean(samples > 0.0)),
        "replicates": int(replicates),
    }


def calibrate_and_save(
    validation_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    calibration_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    validation = np.load(validation_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    strength, selection = select_calibration_strength(
        validation["scenarios"], validation["observations"]
    )
    calibrator = CopulaPITCalibrator.fit(
        validation["scenarios"], validation["observations"], strength=strength
    )
    calibrated = calibrator.transform(test["scenarios"])
    inversions = count_rank_inversions(test["scenarios"], calibrated)
    metadata = {
        "model": "cr_mscadm_calibrated",
        "source": str(Path(test_path).resolve()),
        "calibration_split": "validation_only",
        "strength": strength,
        "selection": selection,
        "rank_inversions": inversions,
        "copula_statement": "zero strict pairwise ensemble-order reversals at every day/hour",
    }
    split = SplitData(
        condition=np.empty((len(calibrated), 0), dtype=np.float32),
        flat_condition=np.empty((len(calibrated), 0), dtype=np.float32),
        target=test["observations"],
        target_standard=np.empty((len(calibrated), 0), dtype=np.float32),
        zone=test["zone"],
        day=test["day"],
    )
    save_scenarios(output_path, calibrated, split, metadata)
    Path(calibration_path).parent.mkdir(parents=True, exist_ok=True)
    Path(calibration_path).write_text(
        json.dumps({"calibrator": calibrator.to_dict(), "selection": selection}, indent=2),
        encoding="utf-8",
    )
    return calibrated, metadata


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "generate_cr_scenarios",
    "select_calibration_strength",
    "extended_scores",
    "per_day_scores",
    "paired_bootstrap",
    "calibrate_and_save",
    "sha256",
]
