from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stgf_flow.data import build_stgf_confirmation_gefcom2014
from stgf_flow.graph import SpectralArtifacts
from stgf_flow.metrics import evaluate_stgf
from stgf_flow.sampling_moment import sample_stgf_moment_anchored
from stgf_flow.training import load_stgf_checkpoint
from stgf_flow.training_v2 import PhysicalCenterSTGFTrainer


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    config = json.loads(source.read_text(encoding="utf-8"))
    lock_path = (ROOT / config["development_selection_lock"]).resolve()
    if sha256_file(lock_path) != config["development_selection_lock_sha256"]:
        raise RuntimeError("development selection lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "stgf_development_selection_lock_v2":
        raise RuntimeError("unexpected development selection lock")
    selected = config["locked_selection"]
    checks = {
        "transform_mode": lock["selected_transform_mode"],
        "physical_mean_anchor": lock["selected_physical_mean_anchor"],
        "residual_temperature": lock["selected_residual_temperature"],
    }
    if selected != checks:
        raise RuntimeError("confirmation configuration does not match lock")
    if config.get("test_access_policy") != (
        "selection lock required before any frozen test archive is generated"
    ):
        raise RuntimeError("missing frozen-test access policy")
    return config


def build_data(config: dict[str, Any], outer: int):
    return build_stgf_confirmation_gefcom2014(
        ROOT / config["data_dir"],
        outer=outer,
        split_path=ROOT / config["split_registry"],
    )


def run_dir(
    config: dict[str, Any], outer: int, seed: int, transform_mode: str
) -> Path:
    return (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "runs"
        / transform_mode
        / f"seed{seed}"
    )


def scenario_path(
    config: dict[str, Any], outer: int, seed: int, transform_mode: str
) -> Path:
    return (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "scenarios"
        / f"{transform_mode}_seed{seed}_test.npz"
    )


def train_one(
    config: dict[str, Any],
    *,
    outer: int,
    seed: int,
    transform_mode: str,
    device: str | None,
) -> Path:
    destination = run_dir(config, outer, seed, transform_mode)
    final = destination / "final.pt"
    if final.exists():
        payload = torch.load(final, map_location="cpu", weights_only=False)
        if (
            payload.get("config") != config
            or payload.get("transform_mode") != transform_mode
            or int(payload.get("seed", -1)) != seed
        ):
            raise RuntimeError(f"checkpoint/config mismatch: {final}")
        return final
    data = build_data(config, outer)
    return PhysicalCenterSTGFTrainer(
        config,
        data,
        destination,
        seed=seed,
        transform_mode=transform_mode,
        device=device,
    ).fit()


@torch.no_grad()
def test_one(
    config: dict[str, Any],
    *,
    outer: int,
    seed: int,
    transform_mode: str,
    device: str | None,
) -> Path:
    data = build_data(config, outer)
    checkpoint = run_dir(config, outer, seed, transform_mode) / "final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    destination = scenario_path(config, outer, seed, transform_mode)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = destination.with_suffix(".metrics.json")
    sampling = config["sampling"]
    selection = config["locked_selection"]
    selected_device = torch.device(
        device
        or (
            config.get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
    )
    payload, model = load_stgf_checkpoint(
        checkpoint, device=selected_device
    )
    artifacts = SpectralArtifacts.from_torch_payload(payload["artifacts"])
    if not destination.exists():
        chunks: list[np.ndarray] = []
        for start in range(
            0, len(data.test), int(sampling["day_batch"])
        ):
            condition = torch.from_numpy(
                data.test.condition[
                    start : start + int(sampling["day_batch"])
                ]
            ).to(selected_device)
            generated = sample_stgf_moment_anchored(
                model,
                condition,
                members=int(sampling["members"]),
                steps=int(sampling["flow_steps"]),
                member_chunk=int(sampling["member_chunk"]),
                seed=2_100_000
                + outer * 100_000
                + seed * 10_000
                + config["transform_modes"].index(transform_mode) * 1_000
                + start,
                residual_temperature=float(
                    selection["residual_temperature"]
                ),
                physical_mean_anchor=float(
                    selection["physical_mean_anchor"]
                ),
            )
            chunks.append(generated.cpu().numpy())
        scenarios = np.concatenate(chunks, axis=0).astype(np.float32)
        metadata = {
            "name": "STGF-Flow",
            "transform_mode": transform_mode,
            "seed": seed,
            "outer": outer,
            "split": "new_frozen_test",
            "members": int(sampling["members"]),
            "flow_steps": int(sampling["flow_steps"]),
            "integrator": "heun",
            "physical_mean_anchor": float(
                selection["physical_mean_anchor"]
            ),
            "residual_temperature": float(
                selection["residual_temperature"]
            ),
            "checkpoint": str(checkpoint.resolve()),
            "protocol_sha256": data.protocol["protocol_sha256"],
            "selection_lock_sha256": config[
                "development_selection_lock_sha256"
            ],
            "evidence_scope": config["evidence_label"],
        }
        np.savez_compressed(
            destination,
            scenarios=scenarios,
            observations=data.test.target.astype(np.float32),
            day=data.test.day,
            zones=data.test.zones,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    if not metrics_path.exists():
        with np.load(destination, allow_pickle=False) as stored:
            metrics = evaluate_stgf(
                stored["scenarios"], data.test, artifacts
            )
        metrics_path.write_text(
            json.dumps(
                {
                    "outer": outer,
                    "seed": seed,
                    "model": "STGF-Flow",
                    "transform_mode": transform_mode,
                    "selection_role": (
                        "primary"
                        if transform_mode
                        == selection["transform_mode"]
                        else "predeclared representation ablation"
                    ),
                    "split": "new_frozen_test",
                    "protocol_sha256": data.protocol["protocol_sha256"],
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return metrics_path


def experiment_matrix(
    config: dict[str, Any],
    requested_mode: str | None,
    requested_seed: int | None,
) -> list[tuple[str, int]]:
    selected = config["locked_selection"]["transform_mode"]
    modes = (
        config["transform_modes"]
        if requested_mode is None
        else [requested_mode]
    )
    matrix: list[tuple[str, int]] = []
    for mode in modes:
        if mode not in config["transform_modes"]:
            raise ValueError(f"unregistered transform mode: {mode}")
        allowed = (
            config["primary_model_seeds"]
            if mode == selected
            else config["ablation_model_seeds"]
        )
        seeds = allowed if requested_seed is None else [requested_seed]
        for seed in seeds:
            if seed not in allowed:
                raise ValueError(
                    f"seed {seed} is not predeclared for mode {mode}"
                )
            matrix.append((mode, int(seed)))
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-lock STGF-Flow frozen confirmation"
    )
    parser.add_argument(
        "--config", default="repro_configs/stgf_confirmation_v1.json"
    )
    parser.add_argument(
        "--phase", required=True, choices=["train", "test", "all"]
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--transform-mode")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    outers = (
        config["outer_splits"] if args.outer is None else [args.outer]
    )
    matrix = experiment_matrix(
        config, args.transform_mode, args.seed
    )
    for outer in outers:
        if outer not in config["outer_splits"]:
            raise ValueError(f"unregistered outer: {outer}")
        for mode, seed in matrix:
            if args.phase in {"train", "all"}:
                trained = train_one(
                    config,
                    outer=int(outer),
                    seed=seed,
                    transform_mode=mode,
                    device=args.device,
                )
                print(
                    json.dumps(
                        {
                            "trained": str(trained),
                            "outer": outer,
                            "seed": seed,
                            "transform_mode": mode,
                        }
                    ),
                    flush=True,
                )
            if args.phase in {"test", "all"}:
                evaluated = test_one(
                    config,
                    outer=int(outer),
                    seed=seed,
                    transform_mode=mode,
                    device=args.device,
                )
                print(
                    json.dumps(
                        {
                            "evaluated": str(evaluated),
                            "outer": outer,
                            "seed": seed,
                            "transform_mode": mode,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
