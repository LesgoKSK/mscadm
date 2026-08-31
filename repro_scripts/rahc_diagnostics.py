"""Consolidate RAHC parameter, gate, and drift diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaincinv

from rahc.v2_calibration import RAHCalibratorV2, rank_intervals


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "outputs" / "rahc_cr_mscadm"
CR_ROOT = WORKSPACE / "outputs" / "cr_mscadm" / "scenarios"


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _parameter_audit(output: Path) -> None:
    calibrator, metadata = RAHCalibratorV2.load(ROOT / "calibrators" / "C6_RAHC.pt")
    desired = (np.arange(100, dtype=np.float64) + 0.5) / 100.0
    splits: dict[str, object] = {}
    for split in ("validation", "test"):
        alpha_values: list[np.ndarray] = []
        beta_values: list[np.ndarray] = []
        tail_values: list[np.ndarray] = []
        tied = total = 0
        for seed in range(3):
            with np.load(CR_ROOT / f"full_seed{seed}_{split}_raw.npz", allow_pickle=False) as archive:
                alpha, beta = calibrator.predict_shapes(
                    archive["scenarios"], archive["zone"], strength=1.0
                )
                probability = betaincinv(
                    alpha[:, None, :], beta[:, None, :], desired[None, :, None]
                )
                lower, upper = rank_intervals(archive["scenarios"], archive["observations"])
                tied += int(np.sum(lower != upper))
                total += int(lower.size)
                alpha_values.append(alpha.reshape(-1))
                beta_values.append(beta.reshape(-1))
                tail_values.append(probability.reshape(-1))
        alpha_all = np.concatenate(alpha_values)
        beta_all = np.concatenate(beta_values)
        probability_all = np.concatenate(tail_values)
        quantiles = [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
        splits[split] = {
            "alpha_quantiles": dict(zip(map(str, quantiles), np.quantile(alpha_all, quantiles).tolist())),
            "beta_quantiles": dict(zip(map(str, quantiles), np.quantile(beta_all, quantiles).tolist())),
            "probability_below_0.001_fraction": float(np.mean(probability_all < 0.001)),
            "probability_above_0.999_fraction": float(np.mean(probability_all > 0.999)),
            "tie_censored_cell_fraction": tied / total,
            "cells_across_three_seed_archives": total,
        }
    parameters = {
        name: value.detach().cpu().numpy().tolist()
        for name, value in calibrator.model.named_parameters()
    }
    _json(
        output / "parameter_audit.json",
        {
            "metadata": metadata,
            "fit_summary": calibrator.fit_summary.to_dict(),
            "parameters": parameters,
            "split_shape_and_tail_audit": splits,
            "interpretation": {
                "global_dispersion_log_effect": float(parameters["global_effect"][1]),
                "log_spread_to_dispersion_effect": float(parameters["regime_effect"][1][1]),
                "statement": (
                    "negative global dispersion makes the rank distribution U-shaped; "
                    "positive log-spread dispersion means low-spread states receive stronger widening"
                ),
            },
        },
    )


def _success_gates(output: Path) -> None:
    overall = pd.read_csv(ROOT / "metrics" / "overall_metrics.csv")
    means = overall.groupby("method").mean(numeric_only=True)
    groups = pd.read_csv(ROOT / "metrics" / "group_metrics.csv")
    ranks = pd.read_csv(ROOT / "metrics" / "copula_rank_audit.csv")
    bootstrap = json.loads(
        (ROOT / "statistics" / "paired_calendar_day_bootstrap.json").read_text(encoding="utf-8")
    )["legacy_G0_vs_C6_RAHC"]["metrics"]
    legacy = means.loc["legacy_G0"]
    rahc = means.loc["C6_RAHC"]
    group_mean = (
        groups[groups["method"] == "C6_RAHC"]
        .groupby(["family", "group"], as_index=False)
        .mean(numeric_only=True)
    )
    legacy_group_summary = pd.read_csv(ROOT / "metrics" / "group_summary.csv")
    legacy_ace = legacy_group_summary.loc[
        legacy_group_summary["method"] == "legacy_G0", "family_equal_ACE_90"
    ].mean()
    rahc_ace = legacy_group_summary.loc[
        legacy_group_summary["method"] == "C6_RAHC", "family_equal_ACE_90"
    ].mean()
    rank_rows = ranks[ranks["method"] == "C6_RAHC"]
    per_seed_crps_improved = int(
        np.sum(
            overall[overall["method"] == "C6_RAHC"].sort_values("seed")["CRPS"].to_numpy()
            < overall[overall["method"] == "legacy_G0"].sort_values("seed")["CRPS"].to_numpy()
        )
    )
    zone_min = float(group_mean[group_mean["family"] == "zone"]["coverage_90"].min())
    interaction_min = float(
        group_mean[group_mean["family"] == "zone_x_spread"]["coverage_90"].min()
    )
    rows = [
        {
            "gate": "CRPS >=2% improvement and 95% CI excludes zero",
            "value": (legacy["CRPS"] - rahc["CRPS"]) / legacy["CRPS"],
            "threshold": 0.02,
            "passed": bool(
                (legacy["CRPS"] - rahc["CRPS"]) / legacy["CRPS"] >= 0.02
                and bootstrap["crps_improvement"]["ci_low"] > 0.0
            ),
        },
        {
            "gate": "mean 90% coverage in [0.88,0.92]",
            "value": rahc["coverage_90"],
            "threshold": "[0.88,0.92]",
            "passed": bool(0.88 <= rahc["coverage_90"] <= 0.92),
        },
        {
            "gate": "conditional ACE reduction >=30%",
            "value": (legacy_ace - rahc_ace) / legacy_ace,
            "threshold": 0.30,
            "passed": bool((legacy_ace - rahc_ace) / legacy_ace >= 0.30),
        },
        {
            "gate": "worst mean-seed zone coverage >=0.85",
            "value": zone_min,
            "threshold": 0.85,
            "passed": bool(zone_min >= 0.85),
        },
        {
            "gate": "worst mean-seed zone x spread coverage >=0.80",
            "value": interaction_min,
            "threshold": 0.80,
            "passed": bool(interaction_min >= 0.80),
        },
        {
            "gate": "MAE degradation <=1%",
            "value": rahc["MAE"] / legacy["MAE"] - 1.0,
            "threshold": 0.01,
            "passed": bool(rahc["MAE"] / legacy["MAE"] - 1.0 <= 0.01),
        },
        {
            "gate": "VS degradation <=1%",
            "value": rahc["VS"] / legacy["VS"] - 1.0,
            "threshold": 0.01,
            "passed": bool(rahc["VS"] / legacy["VS"] - 1.0 <= 0.01),
        },
        {
            "gate": "ramp-CRPS degradation <=1%",
            "value": rahc["ramp_CRPS"] / legacy["ramp_CRPS"] - 1.0,
            "threshold": 0.01,
            "passed": bool(rahc["ramp_CRPS"] / legacy["ramp_CRPS"] - 1.0 <= 0.01),
        },
        {
            "gate": "CRPS improvement in all 3 model seeds",
            "value": per_seed_crps_improved,
            "threshold": 3,
            "passed": bool(per_seed_crps_improved == 3),
        },
        {
            "gate": "zero strict rank reversals",
            "value": int(rank_rows["strict_reversals"].sum()),
            "threshold": 0,
            "passed": bool(rank_rows["strict_reversals"].sum() == 0),
        },
        {
            "gate": "stable ordinal rank fraction 1.0",
            "value": float(rank_rows["stable_ordinal_rank_fraction"].mean()),
            "threshold": 1.0,
            "passed": bool(np.all(rank_rows["stable_ordinal_rank_fraction"] == 1.0)),
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "success_gates.csv", index=False)
    _json(
        output / "success_gates.json",
        {
            "primary_method": "C6_RAHC bounded-tail",
            "passed": int(frame["passed"].sum()),
            "total": int(len(frame)),
            "overall_success": bool(frame["passed"].all()),
            "gates": rows,
        },
    )


def _tail_and_groups(output: Path) -> None:
    overall = pd.read_csv(ROOT / "metrics" / "overall_metrics.csv")
    selected = overall[overall["method"].isin(
        ["legacy_G0", "C6_RAHC", "C6_RAHC_linear", "C6_RAHC_clamp"]
    )]
    selected.groupby("method").agg(["mean", "std"]).to_csv(output / "tail_sensitivity.csv")
    groups = pd.read_csv(ROOT / "metrics" / "group_metrics.csv")
    key = groups[
        groups["method"].isin(["legacy_G0", "C6_RAHC"])
        & groups["family"].isin(["zone", "spread_quintile", "zone_x_spread"])
    ]
    key.groupby(["method", "family", "group"], as_index=False).mean(numeric_only=True).to_csv(
        output / "key_group_metrics.csv", index=False
    )


def main() -> None:
    output = ROOT / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    _parameter_audit(output)
    _success_gates(output)
    _tail_and_groups(output)
    print(f"wrote diagnostics to {output}")


if __name__ == "__main__":
    main()
