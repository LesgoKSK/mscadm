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
        description="Calibration-only physical moment-anchor pilot"
    )
    parser.add_argument(
        "--config", default="repro_configs/stgf_development_v1.json"
    )
    parser.add_argument("--outer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transform-mode", default="stgf")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    data = build_stgf_development_gefcom2014(
        ROOT / config["data_dir"], outer=args.outer
    )
    checkpoint = (
        ROOT
        / config["output_root"]
        / f"outer{args.outer}"
        / "runs"
        / args.transform_mode
        / f"seed{args.seed}"
        / "final.pt"
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    payload, model = load_stgf_checkpoint(checkpoint, device=device)
    artifacts = SpectralArtifacts.from_torch_payload(payload["artifacts"])
    output = (
        ROOT
        / config["output_root"]
        / f"outer{args.outer}"
        / "moment_pilot"
    )
    output.mkdir(parents=True, exist_ok=True)
    temperatures = [0.45, 0.60, 0.75, 0.90, 1.05]
    anchors = [0.5, 1.0]
    for anchor in anchors:
        for temperature in temperatures:
            name = f"{args.transform_mode}_pa{anchor:.2f}_t{temperature:.2f}"
            metrics_path = output / f"{name}.metrics.json"
            if metrics_path.exists():
                continue
            chunks: list[np.ndarray] = []
            sampling = config["sampling"]
            for start in range(0, len(data.calibration), int(sampling["day_batch"])):
                condition = torch.from_numpy(
                    data.calibration.condition[
                        start : start + int(sampling["day_batch"])
                    ]
                ).to(device)
                scenarios = sample_stgf_moment_anchored(
                    model,
                    condition,
                    members=int(sampling["members"]),
                    steps=int(sampling["flow_steps"]),
                    member_chunk=int(sampling["member_chunk"]),
                    seed=1_700_000 + args.outer * 10_000 + start * 100,
                    residual_temperature=temperature,
                    physical_mean_anchor=anchor,
                )
                chunks.append(scenarios.cpu().numpy())
            values = np.concatenate(chunks, axis=0)
            metrics = evaluate_stgf(values, data.calibration, artifacts)
            record = {
                "split": "calibration_only",
                "outer": args.outer,
                "seed": args.seed,
                "transform_mode": args.transform_mode,
                "physical_mean_anchor": anchor,
                "residual_temperature": temperature,
                "metrics": metrics,
            }
            metrics_path.write_text(
                json.dumps(record, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "candidate": name,
                        "CRPS": metrics["CRPS"],
                        "MAE": metrics["MAE"],
                        "coverage_90": metrics["coverage_90"],
                        "width_90": metrics["width_90"],
                        "joint_ES_240": metrics["joint_ES_240"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
