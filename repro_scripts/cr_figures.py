from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cr_mscadm.calibration import pit_values
from repro.metrics import interval_scores


ROOT = Path("outputs/cr_mscadm")
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(ROOT / "scenarios" / name, allow_pickle=False)
    return archive["scenarios"], archive["observations"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 220,
        }
    )


def reliability() -> None:
    levels = np.arange(0.1, 1.0, 0.1)
    sources = {
        "MS-CADM raw": load("baseline_mscadm_test_raw.npz"),
        "MS-CADM calibrated": load("baseline_mscadm_test_calibrated.npz"),
        "CR raw": load("full_seed0_test_raw.npz"),
        "CR calibrated": load("full_seed0_test_calibrated.npz"),
    }
    ddpm = np.load("outputs/full_reproduction/scenarios/ddpm_test_250steps.npz", allow_pickle=False)
    sources["DDPM"] = (ddpm["scenarios"], ddpm["observations"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(levels, levels, "k--", linewidth=1.2, label="ideal")
    for name, (scenarios, observations) in sources.items():
        coverage = interval_scores(scenarios, observations, levels)["coverage"]
        ax.plot(levels, coverage, marker="o", linewidth=1.8, label=name)
    ax.set(xlabel="Nominal central interval", ylabel="Empirical coverage", xlim=(0.08, 0.92), ylim=(0.05, 0.95))
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_reliability.png")
    plt.close(fig)


def ablation() -> None:
    frame = pd.read_csv(ROOT / "tables" / "ablation.csv")
    labels = ["MS raw", "Fixed residual", "Hetero", "+ CRPS", "+ calibration", "MS + calibration"]
    colors = ["#777777", "#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3))
    for ax, metric, title in zip(
        axes,
        ["CRPS", "coverage_90", "VS"],
        ["CRPS (lower is better)", "90% coverage", "Variogram score (lower is better)"],
    ):
        values = frame[metric].to_numpy()
        ax.bar(np.arange(len(values)), values, color=colors)
        if metric == "coverage_90":
            ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_ablation.png")
    plt.close(fig)


def pit_histograms() -> None:
    sources = {
        "MS-CADM raw": load("baseline_mscadm_test_raw.npz"),
        "CR-MS-CADM raw": load("full_seed0_test_raw.npz"),
        "CR-MS-CADM calibrated": load("full_seed0_test_calibrated.npz"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8), sharey=True)
    for ax, (name, (scenarios, observations)) in zip(axes, sources.items()):
        pits = pit_values(scenarios, observations).ravel()
        ax.hist(pits, bins=np.linspace(0, 1, 11), density=True, color="#4C78A8", edgecolor="white")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set(title=name, xlabel="Finite-ensemble PIT", xlim=(0, 1))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3_pit.png")
    plt.close(fig)


def scale_mechanism() -> None:
    frame = pd.read_csv(ROOT / "tables" / "scale_quintiles.csv")
    selected = ["fixed seed0", "hetero no CRPS seed0", "full seed0"]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for name in selected:
        part = frame[frame["model"] == name]
        ax.plot(
            part["mean_predicted_scale"],
            part["mean_absolute_standardized_error"],
            marker="o",
            linewidth=2,
            label=name,
        )
        for _, row in part.iterrows():
            ax.annotate(str(int(row["scale_quintile"])), (row["mean_predicted_scale"], row["mean_absolute_standardized_error"]), fontsize=8)
    ax.set(xlabel="Mean predicted standardized scale", ylabel="Observed absolute standardized error")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_scale_mechanism.png")
    plt.close(fig)


def seed_robustness() -> None:
    frame = pd.read_csv(ROOT / "tables" / "all_metrics.csv")
    raw = frame[frame["method"].str.match(r"CR full raw seed")]
    calibrated = frame[frame["method"].str.match(r"CR full calibrated seed")]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    for ax, metric in zip(axes, ["CRPS", "coverage_90", "VS"]):
        for index, (name, values, color) in enumerate(
            [("raw", raw[metric], "#4C78A8"), ("calibrated", calibrated[metric], "#E15759")]
        ):
            jitter = np.linspace(-0.06, 0.06, len(values))
            ax.scatter(np.full(len(values), index) + jitter, values, color=color, s=40, zorder=3)
            ax.errorbar(index, values.mean(), yerr=values.std(ddof=1), color="black", marker="D", capsize=4)
        ax.set_xticks([0, 1], ["raw", "calibrated"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_seed_robustness.png")
    plt.close(fig)


def example_day() -> None:
    raw, observations = load("full_seed0_test_raw.npz")
    calibrated, _ = load("full_seed0_test_calibrated.npz")
    raw_low, raw_high = np.quantile(raw, [0.05, 0.95], axis=1)
    cal_low, cal_high = np.quantile(calibrated, [0.05, 0.95], axis=1)
    raw_cov = np.mean((observations >= raw_low) & (observations <= raw_high), axis=1)
    cal_cov = np.mean((observations >= cal_low) & (observations <= cal_high), axis=1)
    improvement = cal_cov - raw_cov
    day = int(np.argsort(improvement)[len(improvement) // 2])
    hour = np.arange(1, 25)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True, sharey=True)
    for ax, scenarios, title in [
        (axes[0], raw, "CR-MS-CADM raw"),
        (axes[1], calibrated, "CR-MS-CADM calibrated"),
    ]:
        lower, upper = np.quantile(scenarios[day], [0.05, 0.95], axis=0)
        mean = scenarios[day].mean(axis=0)
        ax.fill_between(hour, lower, upper, color="#4C78A8", alpha=0.25, label="90% interval")
        ax.plot(hour, mean, color="#4C78A8", linewidth=1.8, label="ensemble mean")
        ax.plot(hour, observations[day], color="black", linewidth=1.6, label="observation")
        ax.set_title(f"{title} — test index {day}")
        ax.set_ylabel("Normalized wind power")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_xlabel("Hour")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure6_example_day.png")
    (FIGURES / "figure6_selection.json").write_text(
        json.dumps({"test_index": day, "rule": "median per-day 90% coverage improvement"}, indent=2),
        encoding="utf-8",
    )
    plt.close(fig)


def calibration_sensitivity() -> None:
    frame = pd.read_csv(ROOT / "tables" / "calibration_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for name, part in frame.groupby("model"):
        axes[0].plot(part["strength"], part["objective"], marker="o", label=name)
        axes[1].plot(part["strength"], part["crossfit_calibration_error"], marker="o", label=name)
    axes[0].set(title="Validation objective", xlabel="Calibration strength", ylabel="CRPS + 0.1 × calibration error")
    axes[1].set(title="Cross-fitted calibration error", xlabel="Calibration strength", ylabel="Mean |coverage - nominal|")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure7_calibration_sensitivity.png")
    plt.close(fig)


def suc() -> None:
    frame = pd.read_csv(ROOT / "suc" / "summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
    for ax, metric, title in zip(
        axes,
        ["total_cost", "load_shedding", "wind_curtailment"],
        ["Realized total cost", "Load shedding (MWh)", "Wind curtailment (MWh)"],
    ):
        ax.bar(np.arange(len(frame)), frame[metric], color=["#777777", "#B07AA1", "#4C78A8", "#E15759"])
        ax.set_xticks(np.arange(len(frame)), frame["model"], rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure8_suc.png")
    plt.close(fig)


def temporal_correlation() -> None:
    raw, observations = load("full_seed0_test_raw.npz")
    calibrated, _ = load("full_seed0_test_calibrated.npz")

    def corr(values: np.ndarray) -> np.ndarray:
        ranks = np.apply_along_axis(rankdata, 0, values)
        return np.corrcoef(ranks, rowvar=False)

    matrices = [
        ("Observed", corr(observations)),
        ("CR raw", corr(raw.reshape(-1, 24))),
        ("CR calibrated", corr(calibrated.reshape(-1, 24))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    for ax, (title, matrix) in zip(axes, matrices):
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set(title=title, xlabel="Hour", ylabel="Hour")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Spearman correlation")
    fig.savefig(FIGURES / "figure9_temporal_rank_correlation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style()
    reliability()
    ablation()
    pit_histograms()
    scale_mechanism()
    seed_robustness()
    example_day()
    calibration_sensitivity()
    suc()
    temporal_correlation()
    print(f"generated {len(list(FIGURES.glob('*.png')))} figures in {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
