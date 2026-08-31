from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stgf_flow.data import build_stgf_development_gefcom2014
from stgf_flow.graph import SpectralArtifacts
from stgf_flow.metrics import evaluate_stgf
from stgf_flow.sampling_moment import sample_stgf_moment_anchored
from stgf_flow.training import load_stgf_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate calibration-only STGF representation candidates"
    )
    parser.add_argument(
        "--config", default="repro_configs/stgf_development_v2.json"
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--transform-mode")
    parser.add_argument("--device")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    modes = (
        config["transform_modes"]
        if args.transform_mode is None
        else [args.transform_mode]
    )
    seed = int(config["model_seeds"][0])
    temperatures = config["sampling_selection"]["residual_temperature"]
    anchors = config["sampling_selection"]["physical_mean_anchor"]
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    for outer in outers:
        data = build_stgf_development_gefcom2014(
            ROOT / config["data_dir"], outer=int(outer)
        )
        for mode in modes:
            checkpoint = (
                ROOT
                / config["output_root"]
                / f"outer{outer}"
                / "runs"
                / mode
                / f"seed{seed}"
                / "final.pt"
            )
            payload, model = load_stgf_checkpoint(
                checkpoint, device=device
            )
            artifacts = SpectralArtifacts.from_torch_payload(
                payload["artifacts"]
            )
            directory = (
                ROOT
                / config["output_root"]
                / f"outer{outer}"
                / "candidates_v2"
            )
            directory.mkdir(parents=True, exist_ok=True)
            for anchor in anchors:
                for temperature in temperatures:
                    identity = (
                        f"{mode}_pa{float(anchor):.2f}_"
                        f"t{float(temperature):.2f}"
                    )
                    archive = directory / f"{identity}_calibration.npz"
                    metrics_path = archive.with_suffix(".metrics.json")
                    if not archive.exists():
                        chunks: list[np.ndarray] = []
                        sampling = config["sampling"]
                        for start in range(
                            0,
                            len(data.calibration),
                            int(sampling["day_batch"]),
                        ):
                            condition = torch.from_numpy(
                                data.calibration.condition[
                                    start : start
                                    + int(sampling["day_batch"])
                                ]
                            ).to(device)
                            scenarios = sample_stgf_moment_anchored(
                                model,
                                condition,
                                members=int(sampling["members"]),
                                steps=int(sampling["flow_steps"]),
                                member_chunk=int(
                                    sampling["member_chunk"]
                                ),
                                seed=1_800_000
                                + int(outer) * 10_000
                                + config["transform_modes"].index(mode) * 1_000
                                + int(round(float(anchor) * 100))
                                + int(round(float(temperature) * 10)),
                                residual_temperature=float(temperature),
                                physical_mean_anchor=float(anchor),
                            )
                            chunks.append(scenarios.cpu().numpy())
                        values = np.concatenate(chunks, axis=0).astype(
                            np.float32
                        )
                        np.savez_compressed(
                            archive,
                            scenarios=values,
                            observations=data.calibration.target.astype(
                                np.float32
                            ),
                            day=data.calibration.day,
                            zones=data.calibration.zones,
                            metadata=np.asarray(
                                json.dumps(
                                    {
                                        "split": "calibration_only",
                                        "transform_mode": mode,
                                        "physical_mean_anchor": float(anchor),
                                        "residual_temperature": float(
                                            temperature
                                        ),
                                        "checkpoint": str(
                                            checkpoint.resolve()
                                        ),
                                        "config": str(config_path),
                                    },
                                    sort_keys=True,
                                )
                            ),
                        )
                    if not metrics_path.exists():
                        with np.load(
                            archive, allow_pickle=False
                        ) as stored:
                            metrics = evaluate_stgf(
                                stored["scenarios"],
                                data.calibration,
                                artifacts,
                            )
                        metrics_path.write_text(
                            json.dumps(
                                {
                                    "outer": int(outer),
                                    "seed": seed,
                                    "split": "calibration_only",
                                    "transform_mode": mode,
                                    "physical_mean_anchor": float(anchor),
                                    "residual_temperature": float(
                                        temperature
                                    ),
                                    "metrics": metrics,
                                },
                                indent=2,
                                sort_keys=True,
                            ),
                            encoding="utf-8",
                        )
                    print(
                        json.dumps(
                            {
                                "candidate": identity,
                                "outer": outer,
                                "metrics": str(metrics_path),
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
