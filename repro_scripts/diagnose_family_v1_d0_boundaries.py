#!/usr/bin/env python3
"""Read-only diagnostic for D0 interior-state boundary saturation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "repro_configs" / "architecture_v1_family_v1.json"
DATA_CONFIG = ROOT / "repro_configs" / "architecture_v1_formal_v2_2_1.json"
OUTPUT = (
    ROOT
    / "outputs"
    / "architecture_v1_family_v1"
    / "formal_evaluation"
    / "diagnostics"
    / "D0_seed3_sampling21000_boundary_diagnostic.json"
)

from architecture_v1.data import build_architecture_v1_fit_data
from architecture_v1.model import T0StableSourceRectifiedFlow
from repro_scripts.run_architecture_v1_family_v1 import (
    _atomic_json,
    _configure_cuda,
    _load_config,
    _load_shared_ea,
    _model_kwargs,
)
from repro_scripts.run_architecture_v1_family_v1_formal import _make_trainer


def main() -> int:
    config, model_config = _load_config(CONFIG)
    completion_path = (
        ROOT
        / "outputs"
        / "architecture_v1_family_v1"
        / "formal_training"
        / "runs"
        / "D0"
        / "seed3"
        / "completion.json"
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    device, runtime = _configure_cuda()
    bundle = build_architecture_v1_fit_data(config_path=DATA_CONFIG)
    kwargs = _model_kwargs(config, model_config)
    shared = _load_shared_ea(config, kwargs, torch.device("cpu"))
    torch.manual_seed(12003)
    torch.cuda.manual_seed_all(12003)
    model = T0StableSourceRectifiedFlow(**kwargs).to(device)
    model.load_shared_from(shared, freeze=True)
    trainer = _make_trainer(model, family="D0", config=config)
    trainer.load_checkpoint(
        Path(completion["best_checkpoint"]),
        expected_identity=completion["identity"],
        restore_rng=False,
    )
    sampling_seed = 21000
    thresholds = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
    positive = {str(value): 0 for value in thresholds}
    negative = {str(value): 0 for value in thresholds}
    active_total = exact_zero = exact_one = 0
    affected_days: list[dict[str, int]] = []
    latent_min = float("inf")
    latent_max = float("-inf")
    all_finite = True
    with trainer.ema_weights() as ema_model:
        ema_model.eval()
        for day_index in range(len(bundle.validation)):
            condition = torch.from_numpy(
                np.ascontiguousarray(
                    bundle.validation.condition[day_index : day_index + 1],
                    dtype=np.float32,
                )
            ).to(device)
            scenario = trainer.diffusion.sample_ddim(
                ema_model,
                condition,
                members=100,
                steps=31,
                eta=0.0,
                seed=sampling_seed + day_index * 1009,
                member_chunk=10,
            )
            active = scenario.active_mask
            latent = scenario.interior_latent[active]
            values = scenario.values[active]
            day_zero = int((values == 0.0).sum().item())
            day_one = int((values == 1.0).sum().item())
            if day_zero or day_one:
                affected_days.append(
                    {
                        "day_index": day_index,
                        "exact_zero": day_zero,
                        "exact_one": day_one,
                    }
                )
            active_total += int(active.sum().item())
            exact_zero += day_zero
            exact_one += day_one
            latent_min = min(latent_min, float(latent.min().cpu()))
            latent_max = max(latent_max, float(latent.max().cpu()))
            all_finite = all_finite and bool(torch.isfinite(latent).all())
            for threshold in thresholds:
                positive[str(threshold)] += int((latent >= threshold).sum().item())
                negative[str(threshold)] += int((latent <= -threshold).sum().item())
            print(
                f"[D0 boundary diagnostic] day {day_index + 1}/50; "
                f"zero={day_zero}; one={day_one}",
                flush=True,
            )
    report = {
        "schema": "architecture_v1_family_v1_D0_boundary_diagnostic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "post_failure_read_only_diagnostic_not_candidate_selection",
        "family": "D0",
        "training_seed": 3,
        "sampling_seed": sampling_seed,
        "checkpoint": completion["best_checkpoint"],
        "checkpoint_sha256": completion["best_checkpoint_sha256"],
        "runtime": runtime,
        "active_interior_coordinates": active_total,
        "interior_decoded_exact_zero": exact_zero,
        "interior_decoded_exact_one": exact_one,
        "interior_boundary_total": exact_zero + exact_one,
        "interior_boundary_fraction": (exact_zero + exact_one) / active_total,
        "affected_day_count": len(affected_days),
        "affected_days": affected_days,
        "active_latent_min": latent_min,
        "active_latent_max": latent_max,
        "latent_count_ge_positive_threshold": positive,
        "latent_count_le_negative_threshold": negative,
        "all_latents_finite": all_finite,
        "selection_target_accessed": False,
        "calibration_target_accessed": False,
        "conclusion": (
            "formal_D0_archive_contract_failed_interior_state_strict_boundary"
        ),
    }
    digest = _atomic_json(OUTPUT, report)
    print(json.dumps({**report, "file_sha256": digest}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
