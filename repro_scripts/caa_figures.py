"""Generate the frozen CAA-RAHC result figures and a hash manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "caa_rahc"
FIG = OUT / "figures"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIG / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(OUT / "metrics" / "overall_metrics.csv")
    pooled = metrics[metrics["outer"].astype(str).eq("pooled")]
    summary = pooled.groupby("method", sort=False).mean(numeric_only=True)
    bootstrap = json.loads(
        (OUT / "statistics" / "paired_calendar_day_bootstrap.json").read_text(
            encoding="utf-8"
        )
    )
    consistency = json.loads(
        (OUT / "statistics" / "outer_consistency.json").read_text(encoding="utf-8")
    )

    colors = {"A0": "#667085", "A2": "#2E90FA", "A3": "#F79009",
              "A4": "#12B76A", "A6": "#7A5AF8"}
    methods = ["A0", "A2", "A3", "A4", "A6"]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method in methods:
        row = summary.loc[method]
        ax.scatter(row["conditional_ACE90"], row["CRPS"], s=90,
                   color=colors[method], edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(method, (row["conditional_ACE90"], row["CRPS"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Conditional ACE90 (lower is better)")
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title("CAA-RAHC ablation trade-off on sealed outer tests")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    files = [_save(fig, "ablation_tradeoff.png")]

    names = ["Conditional ACE90", "CRPS", "Winkler90", "Coverage error"]
    keys = ["conditional_family_equal_ACE_90_improvement", "crps_improvement",
            "winkler_90_improvement", "coverage_absolute_error_90_improvement"]
    a4 = bootstrap["comparisons"]["A4"]["metrics"]
    points = [a4[key]["point_estimate"] for key in keys]
    lows = [a4[key]["ci_low"] for key in keys]
    highs = [a4[key]["ci_high"] for key in keys]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    for ax, label, point, low, high in zip(axes, names, points, lows, highs):
        ax.errorbar([0], [point], yerr=[[point-low], [high-point]], fmt="o",
                    color=colors["A4"], capsize=5, linewidth=2)
        ax.axhline(0, color="#667085", linewidth=1, linestyle="--")
        ax.set_xticks([])
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("A4 improvement over A0: paired calendar-day bootstrap (95% CI)")
    fig.tight_layout()
    files.append(_save(fig, "a4_paired_bootstrap.png"))

    per_seed = pd.DataFrame(consistency["per_outer_seed"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for seed, group in per_seed.groupby("seed"):
        axes[0].plot(group["outer"], -100 * group["CRPS_relative_degradation"],
                     marker="o", label=f"seed {seed}")
        axes[1].plot(group["outer"], -group["conditional_ACE90_delta"],
                     marker="o", label=f"seed {seed}")
    axes[0].axhline(0, color="#667085", linewidth=1, linestyle="--")
    axes[1].axhline(0, color="#667085", linewidth=1, linestyle="--")
    axes[0].set_title("CRPS improvement (%)")
    axes[1].set_title("Conditional ACE90 improvement")
    for ax in axes:
        ax.set_xlabel("Outer split")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("A4 versus A0 across all 3 outer splits and 3 seeds")
    fig.tight_layout()
    files.append(_save(fig, "outer_seed_consistency.png"))

    manifest = {
        "schema": "caa_rahc_figure_manifest_v1",
        "inputs": {
            "overall_metrics_sha256": _sha256(OUT / "metrics" / "overall_metrics.csv"),
            "paired_bootstrap_sha256": _sha256(
                OUT / "statistics" / "paired_calendar_day_bootstrap.json"
            ),
            "outer_consistency_sha256": _sha256(
                OUT / "statistics" / "outer_consistency.json"
            ),
        },
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        ],
    }
    (FIG / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
