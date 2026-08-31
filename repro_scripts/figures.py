from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from repro.metrics import interval_scores


def archives(root: Path) -> dict[str, np.lib.npyio.NpzFile]:
    return {path.stem: np.load(path, allow_pickle=False) for path in sorted(root.glob("*_test*.npz"))}


def scenario_panel(name: str, archive: np.lib.npyio.NpzFile, index: int, path: Path) -> None:
    sample, observation = archive["scenarios"][index], archive["observations"][index]
    hour = np.arange(1, 25)
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(hour, sample.T, color="0.75", alpha=0.12, linewidth=0.6)
    axis.plot(hour, np.quantile(sample, 0.1, axis=0), color="#2474b5", label="10%")
    axis.plot(hour, np.quantile(sample, 0.9, axis=0), color="#2f9e44", label="90%")
    axis.plot(hour, sample.mean(axis=0), color="#e3a008", label="mean")
    axis.plot(hour, observation, color="#d62728", linewidth=1.8, label="observation")
    axis.set(xlabel="Hour", ylabel="Normalized wind power", title=name)
    axis.set_xlim(1, 24); axis.set_ylim(0, 1); axis.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-style Figures 4-7")
    parser.add_argument("--root", default="outputs/full_reproduction")
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--zone", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.root); source = root / "scenarios"; output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    data = archives(source)
    for name, archive in data.items():
        scenario_panel(name, archive, args.day_index, output / f"figure4_{name}.png")
    if data:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for name, archive in data.items():
            score = interval_scores(archive["scenarios"], archive["observations"])
            axes[0].plot(score["confidence"], score["piaw"], marker="o", ms=3, label=name)
            axes[1].plot(score["confidence"], score["ace"], marker="o", ms=3, label=name)
        axes[0].set(xlabel="Nominal coverage", ylabel="PIAW")
        axes[1].set(xlabel="Nominal coverage", ylabel="ACE")
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(output / "figure5_intervals.png", dpi=200); plt.close(fig)
        fig, axis = plt.subplots(figsize=(7.2, 3.8))
        reference = next(iter(data.values()))["observations"].ravel()
        axis.hist(reference, bins=40, density=True, histtype="step", linewidth=2, color="black", label="observed")
        for name, archive in data.items():
            axis.hist(archive["scenarios"].ravel(), bins=40, range=(0, 1), density=True, histtype="step", label=name)
        axis.set(xlabel="Normalized wind power", ylabel="Density"); axis.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(output / "figure7_distribution.png", dpi=200); plt.close(fig)
    mscadm = next((archive for name, archive in data.items() if name.startswith("mscadm")), None)
    if mscadm is not None:
        indices = np.flatnonzero(mscadm["zone"] == args.zone)[:3]
        for number, index in enumerate(indices, 1):
            scenario_panel(f"MS-CADM Zone {args.zone}, day {number}", mscadm, int(index), output / f"figure6_zone{args.zone}_day{number}.png")


if __name__ == "__main__":
    main()
