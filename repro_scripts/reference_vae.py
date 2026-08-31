from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from repro.configuration import load_config
from repro.data import build_gefcom2014
from repro.models.reference_vae import DumasReferenceVAE
from repro.sampling import generate_torch_scenarios, save_scenarios
from repro.torch_data import WindScenarioDataset
from repro.training import atomic_torch_save, infinite, move, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the behavior-compatible public Dumas VAE baseline")
    parser.add_argument("--config", default="repro_configs/paper.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scenarios", type=int, default=100)
    args = parser.parse_args()
    root_config = load_config(args.config)
    seed = int(root_config.get("seed", 0)); seed_everything(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = build_gefcom2014(root_config.get("data_dir", "Data"), seed=int(root_config.get("split_seed", 0)))
    model = DumasReferenceVAE(condition_dim=250, target_dim=24, latent_dim=20, hidden_dim=200).to(device)
    learning_rate = 10 ** -3.4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10 ** -3.4)
    # Public code: batch = 10% of 6310 and 200 epochs, i.e. 10 updates/epoch.
    steps, batch_size = 2_000, 631
    loader = DataLoader(
        WindScenarioDataset(data.train, flat_condition=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    batches = infinite(loader); history = []
    model.train()
    for step in range(1, steps + 1):
        batch = move(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(batch["target"].squeeze(-1), batch["condition"])
        losses["loss"].backward(); optimizer.step()
        if step % 100 == 0:
            record = {"step": step, **{key: float(value.detach()) for key, value in losses.items()}}
            history.append(record); print(json.dumps(record), flush=True)
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
            "reparameterization": "exp(log_sigma), behavior-compatible with public reference"
        },
    }
    root = Path(root_config.get("output_root", "outputs/full_reproduction")) / "vae_reference"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "final.pt"
    atomic_torch_save(
        {"name": "vae", "config": checkpoint_config, "step": steps, "model": model.state_dict(), "optimizers": {"model": optimizer.state_dict()}, "history": history},
        checkpoint,
    )
    scenarios, metadata = generate_torch_scenarios(checkpoint, data, scenarios=args.scenarios, device=str(device))
    metadata["baseline_variant"] = "dumas_reference_behavior"
    archive = Path(root_config.get("output_root", "outputs/full_reproduction")) / "scenarios" / "vae_reference_test.npz"
    save_scenarios(archive, scenarios, data.test, metadata)
    print(archive.resolve())


if __name__ == "__main__":
    main()
