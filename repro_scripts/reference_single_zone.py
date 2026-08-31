from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from repro.configuration import load_config, model_config
from repro.data import build_gefcom2014
from repro.models.reference_vae import DumasReferenceVAE
from repro.sampling import generate_torch_scenarios, save_scenarios
from repro.torch_data import WindScenarioDataset
from repro.training import ExperimentTrainer, atomic_torch_save, infinite, move, seed_everything


def train_vae(config: dict, data, root: Path, device: torch.device) -> Path:
    seed = int(config.get("seed", 0))
    model = DumasReferenceVAE(condition_dim=250, target_dim=24, latent_dim=20, hidden_dim=200).to(device)
    learning_rate = 10 ** -3.4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10 ** -3.4)
    # The public baseline uses 10 batches per epoch.  A single-zone training set
    # contains 631 days, so the corresponding batch size is floor(631/10)=63.
    steps, batch_size = 2_000, 63
    loader = DataLoader(
        WindScenarioDataset(data.train, flat_condition=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batches, history = infinite(loader), []
    model.train()
    for step in range(1, steps + 1):
        batch = move(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(batch["target"].squeeze(-1), batch["condition"])
        losses["loss"].backward()
        optimizer.step()
        if step % 100 == 0:
            record = {"step": step, **{key: float(value.detach()) for key, value in losses.items()}}
            history.append(record)
            print(json.dumps(record), flush=True)
    checkpoint_config = {
        "seed": seed,
        "device": str(device),
        "model": {"condition_dim": 250, "target_dim": 24, "latent_dim": 20, "hidden_dim": 200},
        "training": {
            "steps": steps,
            "epochs": 200,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 10 ** -3.4,
            "reparameterization": "exp(log_sigma), behavior-compatible with public reference",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "final.pt"
    atomic_torch_save(
        {
            "name": "vae",
            "config": checkpoint_config,
            "step": steps,
            "model": model.state_dict(),
            "optimizers": {"model": optimizer.state_dict()},
            "history": history,
        },
        checkpoint,
    )
    return checkpoint


def train_wgan(config: dict, data, root: Path, device: str | None) -> Path:
    selected = deepcopy(model_config(config, "wgan"))
    selected["training"].update({"steps": 3_000, "batch_size": 63, "resume": False})
    return ExperimentTrainer("wgan", selected, data, root, device=device).fit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a public-reference baseline on one GEFCom zone")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--model", required=True, choices=["vae", "wgan"])
    parser.add_argument("--zone", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scenarios", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    data = build_gefcom2014(
        config.get("data_dir", "Data"), zones=(args.zone,), seed=int(config.get("split_seed", 0))
    )
    output_root = Path(config.get("output_root", "outputs/full_reproduction"))
    run_root = output_root / "single_zone" / f"zone{args.zone}" / f"{args.model}_reference"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.model == "vae":
        checkpoint = train_vae(config, data, run_root, device)
    else:
        checkpoint = train_wgan(config, data, run_root, str(device))
    scenarios, metadata = generate_torch_scenarios(
        checkpoint, data, scenarios=args.scenarios, day_batch=64, device=str(device)
    )
    metadata.update({
        "baseline_variant": "dumas_reference_behavior" if args.model == "vae" else "dumas_reference_schedule",
        "zone": args.zone,
        "training_scope": "single_zone",
        "scenarios": args.scenarios,
    })
    archive = output_root / "scenarios" / "single_zone" / f"zone{args.zone}_{args.model}_reference.npz"
    save_scenarios(archive, scenarios, data.test, metadata)
    print(archive.resolve())


if __name__ == "__main__":
    main()
