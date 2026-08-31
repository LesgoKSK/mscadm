"""Generate publication-ready diagnostic figures from completed RAHC artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "outputs" / "rahc_cr_mscadm"
FIGURES = ROOT / "figures"


COLORS = {
    "legacy_G0": "#6b7280",
    "C0_empirical": "#94a3b8",
    "C1_global": "#60a5fa",
    "C2_hour": "#3b82f6",
    "C3_hour_zone": "#2563eb",
    "C4_hour_regime": "#a78bfa",
    "C5_additive": "#8b5cf6",
    "C6_RAHC": "#dc2626",
    "C7_no_shrink": "#f97316",
    "raw": "#111827",
}


def _save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def validation_selection() -> None:
    frame = pd.read_csv(ROOT / "cv" / "selection_table.csv")
    selected = frame.sort_values(["objective", "strength", "regularization"], ascending=[True, True, False])
    selected = selected.groupby("method", as_index=False).first().sort_values("objective")
    figure, axis = plt.subplots(figsize=(8.2, 4.3))
    axis.barh(
        selected["method"],
        selected["objective"],
        color=[COLORS.get(value, "#64748b") for value in selected["method"]],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Validation OOF selection objective (lower is better)")
    axis.set_title("RAHC ablation: grouped-date cross-validation")
    axis.grid(axis="x", alpha=0.25)
    for index, value in enumerate(selected["objective"]):
        axis.text(value + 0.00005, index, f"{value:.5f}", va="center", fontsize=8)
    _save(figure, "validation_ablation_objective")


def test_tradeoff() -> None:
    frame = pd.read_csv(ROOT / "metrics" / "overall_seed_summary.csv")
    selected_names = [
        "raw", "legacy_G0", "C0_empirical", "C1_global", "C2_hour", "C3_hour_zone",
        "C4_hour_regime", "C5_additive", "C6_RAHC", "C7_no_shrink",
    ]
    frame = frame[frame["method"].isin(selected_names)]
    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    for _, row in frame.iterrows():
        method = row["method"]
        axis.scatter(
            row["coverage_90_mean"], row["CRPS_mean"], s=65,
            color=COLORS.get(method, "#64748b"), edgecolor="white", linewidth=0.7, zorder=3,
        )
        axis.annotate(method, (row["coverage_90_mean"], row["CRPS_mean"]), xytext=(5, 4),
                      textcoords="offset points", fontsize=7.5)
    axis.axvspan(0.88, 0.92, color="#22c55e", alpha=0.10, label="pre-registered 90% coverage band")
    axis.axhline(
        float(frame.loc[frame["method"] == "legacy_G0", "CRPS_mean"].iloc[0]),
        color="#6b7280", linestyle="--", linewidth=1.0, label="legacy G0 CRPS",
    )
    axis.set_xlabel("90% interval coverage")
    axis.set_ylabel("CRPS (lower is better)")
    axis.set_title("Test trade-off: reliability versus sharp probabilistic score")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="upper right")
    _save(figure, "test_crps_coverage_tradeoff")


def conditional_heatmaps() -> None:
    frame = pd.read_csv(ROOT / "metrics" / "group_metrics.csv")
    frame = frame[frame["family"] == "zone_x_spread"]
    methods = ("legacy_G0", "C6_RAHC")
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    matrices = {}
    for method in methods:
        values = (
            frame[frame["method"] == method]
            .groupby("group_code")["coverage_90"].mean()
            .sort_index().to_numpy().reshape(10, 5)
        )
        matrices[method] = values
    for axis, method in zip(axes[:2], methods):
        image = axis.imshow(matrices[method], vmin=0.65, vmax=1.0, cmap="RdYlGn", aspect="auto")
        axis.set_title(method)
        axis.set_xlabel("Within-zone raw-spread quintile")
        axis.set_ylabel("Zone")
        axis.set_xticks(range(5), range(1, 6))
        axis.set_yticks(range(10), range(1, 11))
    difference = matrices["C6_RAHC"] - matrices["legacy_G0"]
    image_difference = axes[2].imshow(difference, vmin=-0.08, vmax=0.22, cmap="coolwarm", aspect="auto")
    axes[2].set_title("C6 − legacy coverage")
    axes[2].set_xlabel("Within-zone raw-spread quintile")
    axes[2].set_ylabel("Zone")
    axes[2].set_xticks(range(5), range(1, 6))
    axes[2].set_yticks(range(10), range(1, 11))
    figure.colorbar(image, ax=axes[:2], shrink=0.8, label="90% coverage")
    figure.colorbar(image_difference, ax=axes[2], shrink=0.8, label="coverage change")
    figure.suptitle("Frozen validation-defined Zone × spread groups (three-seed mean)")
    _save(figure, "conditional_zone_spread_heatmaps")


def bootstrap_intervals() -> None:
    payload = json.loads(
        (ROOT / "statistics" / "paired_calendar_day_bootstrap.json").read_text(encoding="utf-8")
    )
    metrics = payload["legacy_G0_vs_C6_RAHC"]["metrics"]
    names = [
        "crps_improvement",
        "winkler_90_improvement",
        "coverage_absolute_error_90_improvement",
        "conditional_family_equal_ACE_90_improvement",
        "conditional_worst_undercoverage_90_improvement",
    ]
    labels = ["CRPS", "Winkler-90", "coverage |error|", "conditional ACE", "worst undercoverage"]
    point = np.asarray([metrics[name]["point_estimate"] for name in names])
    low = np.asarray([metrics[name]["ci_low"] for name in names])
    high = np.asarray([metrics[name]["ci_high"] for name in names])
    figure, axis = plt.subplots(figsize=(8.0, 4.3))
    y = np.arange(len(names))
    axis.errorbar(point, y, xerr=np.vstack((point - low, high - point)), fmt="o", color="#dc2626",
                  ecolor="#991b1b", capsize=4)
    axis.axvline(0.0, color="#111827", linewidth=1.0)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Legacy G0 minus C6 improvement (positive is favorable)")
    axis.set_title("10,000 paired bootstrap replicates clustered by 50 calendar dates")
    axis.grid(axis="x", alpha=0.25)
    _save(figure, "bootstrap_primary_comparison")


def suc_summary() -> None:
    path = ROOT / "suc" / "summary.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path).set_index("model")
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    colors = [COLORS.get(index, "#64748b") for index in frame.index]
    axes[0].bar(frame.index, frame["mean_realized_total_cost"], color=colors)
    axes[0].set_ylabel("Mean realized total cost")
    axes[0].set_title("Seven zone-date cases")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(frame.index, frame["mean_realized_load_shedding"], color=colors)
    axes[1].set_ylabel("Mean realized load shedding")
    axes[1].set_title("Only five unique dates; descriptive")
    axes[1].tick_params(axis="x", rotation=15)
    figure.suptitle("RTS-24 single-zone 1200 MW proxy sensitivity")
    _save(figure, "suc_descriptive_summary")


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    validation_selection()
    test_tradeoff()
    conditional_heatmaps()
    bootstrap_intervals()
    suc_summary()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
