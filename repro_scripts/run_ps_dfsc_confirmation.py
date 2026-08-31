from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import repro_scripts.run_ps_dfsc as pipeline
from ps_dfsc.manifest import verify_lock_manifest
from ps_dfsc.mapping import WindFarmMapping
from ps_dfsc.reduction import fit_fixed_assignments, weighted_cluster_reduction
from repro_scripts.run_ps_dfsc_base import generate, load_config, pool


ROOT = Path(__file__).resolve().parents[1]


def _verified_lock(path: str | Path, outer: int) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_lock_manifest(value)
    if int(value["outer"]) != outer:
        raise ValueError("lock manifest outer does not match requested outer")
    return value


def _load_mapping(path: str | Path) -> WindFarmMapping:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return WindFarmMapping(
        groups=tuple(tuple(group) for group in value["groups"]),
        wind_buses=tuple(value["wind_buses"]),
        zone_capacity_mw=float(value["zone_capacity_mw"]),
    )


def _identity_archive(
    pooled_path: Path,
    mapping_path: str | Path,
    output: Path,
    clusters: int,
) -> None:
    source = np.load(pooled_path, allow_pickle=False)
    scenarios = source["scenarios"]
    members = scenarios.shape[1]
    uniform = np.full(members, 1.0 / members)
    mapping = _load_mapping(mapping_path)
    reduced = []
    probability = []
    for values in scenarios:
        mapped = mapping.transform(values)
        labels = fit_fixed_assignments(mapped, clusters)
        current, mass = weighted_cluster_reduction(mapped, uniform, labels)
        reduced.append(current)
        probability.append(mass)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        full_scenarios=scenarios,
        probabilities=np.broadcast_to(uniform, scenarios.shape[:2]).copy(),
        suc_scenarios=np.stack(reduced),
        suc_probabilities=np.stack(probability),
        ess=np.full(len(scenarios), members),
        entropy=np.full(len(scenarios), np.log(members)),
        transport_cost=np.zeros(len(scenarios)),
        used_fallback=np.zeros(len(scenarios), dtype=bool),
        fallback_reason=np.full(len(scenarios), "", dtype="<U1"),
        days=source["day"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locked PS-DFSC confirmation pipeline"
    )
    parser.add_argument(
        "--base-config", default="repro_configs/ps_dfsc_base.json"
    )
    parser.add_argument("--lock", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument(
        "--phase",
        choices=["generate", "calibrate", "evaluate", "all"],
        required=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--regime-reference")
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--output-root", default="outputs/ps_dfsc/confirmation")
    args = parser.parse_args()
    _verified_lock(args.lock, args.outer)
    _, config = load_config(args.base_config)
    pooled_path = (
        ROOT
        / config["output_root"]
        / f"outer{args.outer}"
        / "pooled"
        / f"confirmation_test_M{config['sampling']['pooled_members']}.npz"
    )
    output_root = ROOT / args.output_root / f"outer{args.outer}"
    calibrated_path = output_root / "ps_dfsc_test.npz"
    identity_path = output_root / "identity_test.npz"
    if args.phase in {"generate", "all"}:
        for seed in map(int, config["model_seeds"]):
            generated = generate(
                config,
                args.outer,
                seed,
                "confirmation_test",
                args.device,
            )
            print(json.dumps({"generated_after_lock": str(generated)}))
        pooled_path = pool(config, args.outer, "confirmation_test")
        print(json.dumps({"pooled_after_lock": str(pooled_path)}))
    if args.phase in {"calibrate", "all"}:
        if not pooled_path.exists():
            raise FileNotFoundError("locked confirmation pool does not exist")
        pipeline.calibrate_archive(
            SimpleNamespace(
                input=str(pooled_path),
                scenario_key="scenarios",
                checkpoint=args.checkpoint,
                mapping=args.mapping,
                mapping_training_archive=None,
                mapping_training_key="observations",
                clusters=args.clusters,
                ess_floor=50.0,
                entropy_floor=0.85,
                transport_budget=0.02,
                device=args.device,
                output=str(calibrated_path),
            )
        )
        _identity_archive(
            pooled_path, args.mapping, identity_path, args.clusters
        )
    if args.phase in {"evaluate", "all"}:
        _verified_lock(args.lock, args.outer)
        if args.regime_reference is None:
            raise ValueError("--regime-reference is required for evaluation")
        if not calibrated_path.exists() or not identity_path.exists():
            raise FileNotFoundError("calibrated/identity confirmation archives missing")
        result_csv = output_root / "exact_ps_dfsc.csv"
        pipeline.exact_evaluate(
            SimpleNamespace(
                calibrated=str(calibrated_path),
                truth=str(pooled_path),
                observation_key="observations",
                mapping=args.mapping,
                mapping_training_archive=None,
                mapping_training_key="observations",
                outer=args.outer,
                method="PS-DFSC",
                mip_gap=args.mip_gap,
                time_limit=args.time_limit,
                output=str(result_csv),
            )
        )
        pipeline.exact_evaluate(
            SimpleNamespace(
                calibrated=str(identity_path),
                truth=str(pooled_path),
                observation_key="observations",
                mapping=args.mapping,
                mapping_training_archive=None,
                mapping_training_key="observations",
                outer=args.outer,
                method="MM-JDWind identity",
                mip_gap=args.mip_gap,
                time_limit=args.time_limit,
                output=str(output_root / "exact_identity.csv"),
            )
        )
        pipeline.safety_gate(
            SimpleNamespace(
                candidate=str(calibrated_path),
                baseline=str(identity_path),
                truth=str(pooled_path),
                regime_reference=args.regime_reference,
                observation_key="observations",
                ess_floor=50.0,
                entropy_floor=0.85,
                transport_budget=0.02,
                bootstrap_samples=10_000,
                seed=10_000 + args.outer,
                output=str(output_root / "confirmation_safety.json"),
            )
        )
    print(
        json.dumps(
            {
                "outer": args.outer,
                "phase": args.phase,
                "lock_verified_before_test_access": True,
                "output_root": str(output_root.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
