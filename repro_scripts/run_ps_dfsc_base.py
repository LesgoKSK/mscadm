from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mm_jdwind.experiment import generate_joint_scenarios, save_joint_archive
from mm_jdwind.training_v2 import StableMMTrainer
from ps_dfsc.data import build_ps_dfsc_outer_data


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("development_train", "development_validation", "confirmation_test")


def load_config(path: str | Path) -> tuple[Path, dict]:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / source
    return source.resolve(), json.loads(source.read_text(encoding="utf-8"))


def run_directory(config: dict, outer: int, seed: int) -> Path:
    return (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "runs"
        / f"seed{seed}"
    )


def build_data(config: dict, outer: int):
    return build_ps_dfsc_outer_data(
        ROOT / config["data_dir"],
        outer=outer,
        split_registry=ROOT / config["split_registry"],
    )


def train(config: dict, outer: int, seed: int, device: str | None) -> Path:
    destination = run_directory(config, outer, seed)
    final = destination / "final.pt"
    if final.exists():
        payload = torch.load(final, map_location="cpu", weights_only=False)
        if int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"checkpoint seed mismatch: {final}")
        return final
    data = build_data(config, outer)
    return StableMMTrainer(
        config,
        data.base_generator,
        destination,
        seed=seed,
        device=device,
    ).fit()


def _role_split(data, role: str):
    if role == "development_train":
        return data.development_train
    if role == "development_validation":
        return data.development_validation
    if role == "confirmation_test":
        return data.confirmation_test
    raise ValueError(role)


def generate(
    config: dict,
    outer: int,
    seed: int,
    role: str,
    device: str | None,
) -> Path:
    data = build_data(config, outer)
    selected = _role_split(data, role)
    bundle = replace(
        data.base_generator,
        test=selected,
        protocol={**data.protocol, "generation_role": role},
    )
    destination = (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "scenarios"
        / f"seed{seed}_{role}.npz"
    )
    if destination.exists():
        return destination
    sampling = config["sampling"]
    arrays, metadata = generate_joint_scenarios(
        run_directory(config, outer, seed) / "flow_best.pt",
        bundle,
        split_name="test",
        members=int(sampling["members_per_seed"]),
        jump_steps=int(sampling["jump_steps"]),
        flow_steps=int(sampling["flow_steps"]),
        state_mode=sampling["state_mode"],
        day_batch=int(sampling["day_batch"]),
        member_chunk=int(sampling["member_chunk"]),
        seed=2_000_000
        + outer * 100_000
        + seed * 10_000
        + ROLES.index(role) * 1000,
        device=device,
    )
    metadata.update(
        {
            "schema": "ps_dfsc_base_seed_archive_v1",
            "outer": outer,
            "role": role,
            "base_training_excludes_all_ps_development": True,
            "base_training_excludes_current_outer_test": True,
        }
    )
    save_joint_archive(destination, arrays, metadata)
    return destination


def pool(config: dict, outer: int, role: str) -> Path:
    seeds = list(map(int, config["model_seeds"]))
    counts = list(map(int, config["sampling"]["pooled_counts_by_seed"]))
    if len(seeds) != len(counts) or sum(counts) != int(
        config["sampling"]["pooled_members"]
    ):
        raise ValueError("invalid pooled seed counts")
    archives = []
    for seed in seeds:
        path = (
            ROOT
            / config["output_root"]
            / f"outer{outer}"
            / "scenarios"
            / f"seed{seed}_{role}.npz"
        )
        archives.append(np.load(path, allow_pickle=False))
    days = archives[0]["day"]
    observations = archives[0]["observations"]
    if any(not np.array_equal(item["day"], days) for item in archives[1:]):
        raise ValueError("seed archives do not share day order")
    if any(
        not np.array_equal(item["observations"], observations)
        for item in archives[1:]
    ):
        raise ValueError("seed archives do not share observations")
    scenario_parts = []
    state_parts = []
    for seed, count, archive in zip(seeds, counts, archives):
        rng = np.random.default_rng(3_000_000 + outer * 10_000 + seed * 100)
        members = rng.permutation(archive["scenarios"].shape[1])[:count]
        scenario_parts.append(archive["scenarios"][:, members])
        state_parts.append(archive["states"][:, members])
    total = float(sum(counts))
    zero_probability = sum(
        count * archive["zero_probability"]
        for count, archive in zip(counts, archives)
    ) / total
    one_probability = sum(
        count * archive["one_probability"]
        for count, archive in zip(counts, archives)
    ) / total
    destination = (
        ROOT
        / config["output_root"]
        / f"outer{outer}"
        / "pooled"
        / f"{role}_M{int(total)}.npz"
    )
    arrays = {
        "scenarios": np.concatenate(scenario_parts, axis=1).astype(np.float32),
        "observations": observations.astype(np.float32),
        "zero_probability": zero_probability.astype(np.float32),
        "one_probability": one_probability.astype(np.float32),
        "states": np.concatenate(state_parts, axis=1).astype(np.int8),
        "day": days,
        "zones": archives[0]["zones"],
    }
    metadata = {
        "schema": "ps_dfsc_frozen_base_pool_v1",
        "outer": outer,
        "role": role,
        "seeds": seeds,
        "counts": counts,
        "members": int(total),
        "pooling": "deterministic member subsample, equal-as-possible seed allocation",
    }
    save_joint_archive(destination, arrays, metadata)
    for archive in archives:
        archive.close()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen MM-JDWind PS-DFSC bases")
    parser.add_argument(
        "--config", default="repro_configs/ps_dfsc_base.json"
    )
    parser.add_argument(
        "--phase", choices=["train", "generate", "pool", "all"], required=True
    )
    parser.add_argument("--outer", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--device")
    args = parser.parse_args()
    _, config = load_config(args.config)
    outers = config["outer_splits"] if args.outer is None else [args.outer]
    seeds = config["model_seeds"] if args.seed is None else [args.seed]
    roles = ROLES if args.role is None else (args.role,)
    for outer in map(int, outers):
        if args.phase in {"train", "all"}:
            for seed in map(int, seeds):
                path = train(config, outer, seed, args.device)
                print(json.dumps({"trained": str(path), "outer": outer, "seed": seed}))
        if args.phase in {"generate", "all"}:
            for seed in map(int, seeds):
                for role in roles:
                    path = generate(config, outer, seed, role, args.device)
                    print(
                        json.dumps(
                            {
                                "generated": str(path),
                                "outer": outer,
                                "seed": seed,
                                "role": role,
                            }
                        )
                    )
        if args.phase in {"pool", "all"}:
            for role in roles:
                path = pool(config, outer, role)
                print(json.dumps({"pooled": str(path), "outer": outer, "role": role}))


if __name__ == "__main__":
    main()
