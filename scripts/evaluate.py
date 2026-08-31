from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mscadm.data import build_datasets
from mscadm.diffusion import GaussianDiffusion
from mscadm.metrics import evaluate_all
from mscadm.model import MSCADM


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data_config = config["data"]
    _, _, test_set, _ = build_datasets(
        data_config["data_dir"],
        validation_days=int(data_config["validation_days"]),
        target_scale=data_config["target_scale"],
        zones=tuple(data_config["zones"]),
    )
    model_config = dict(config["model"])
    model_config["learn_variance"] = bool(config["diffusion"]["learn_variance"])
    model = MSCADM(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion(**config["diffusion"]).to(device)
    evaluation = config["evaluation"]
    loader = DataLoader(test_set, batch_size=int(evaluation["batch_size"]), shuffle=False)

    generated_batches = []
    observation_batches = []
    mask_batches = []
    scenarios = int(evaluation["scenarios"])
    scenario_chunk = int(evaluation.get("scenario_chunk", min(scenarios, 10)))
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        condition = batch["condition"].to(device)
        batch_scenarios = []
        remaining = scenarios
        while remaining:
            count = min(scenario_chunk, remaining)
            repeated = condition[:, None].expand(-1, count, -1, -1)
            repeated = repeated.reshape(-1, condition.shape[1], condition.shape[2])
            samples = diffusion.sample(model, repeated)
            samples = test_set.inverse_target(samples).squeeze(-1)
            samples = samples.reshape(condition.shape[0], count, condition.shape[1])
            batch_scenarios.append(samples.cpu().numpy())
            remaining -= count
        generated_batches.append(np.concatenate(batch_scenarios, axis=1))
        target = test_set.inverse_target(batch["target"]).squeeze(-1)
        observation_batches.append(target.numpy())
        mask_batches.append(batch["target_mask"].squeeze(-1).numpy())
        print(json.dumps({"generated_batch": batch_index + 1}))

    generated = np.concatenate(generated_batches)
    observations = np.concatenate(observation_batches)
    mask = np.concatenate(mask_batches)
    quantiles = [float(value) for value in evaluation.get("quantiles", [0.1, 0.5, 0.9])]
    results = evaluate_all(generated, observations, mask, quantiles)
    results.update({
        "checkpoint_step": int(checkpoint["step"]),
        "sequences": int(len(generated)),
        "scenarios": scenarios,
    })
    output_path = Path(args.checkpoint).parent / "evaluation.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

