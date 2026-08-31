"""Publication-grade identity baseline preparation for PS-DFSC.

The runner is restartable, records exact-solver provenance per day, and only
reuses legacy caches when their saved MIP gap certifies the requested target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ps_dfsc.exact_suc import evaluate_realized, solve_two_stage_suc
from ps_dfsc.manifest import file_sha256
from ps_dfsc.mapping import WindFarmMapping
from ps_dfsc.reduction import fit_fixed_assignments, weighted_cluster_reduction


def load_mapping(path: str | Path) -> WindFarmMapping:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return WindFarmMapping(
        groups=tuple(tuple(group) for group in value["groups"]),
        wind_buses=tuple(value.get("wind_buses", (3, 5, 7, 16, 21, 23))),
        zone_capacity_mw=float(value.get("zone_capacity_mw", 120.0)),
    )


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _scalar(archive, key: str, default=None):
    if key not in archive:
        return default
    return archive[key].item()


def _legacy_cache_is_certified(archive, requested_gap: float) -> bool:
    gap = float(_scalar(archive, "planned_mip_gap", np.nan))
    return bool(
        np.isfinite(gap)
        and gap <= requested_gap * (1.0 + 1e-6) + 1e-12
        and "commitment" in archive
        and "realized_total_cost" in archive
    )


def _load_cache(
    path: Path,
    *,
    input_sha256: str,
    day_sha256: str,
    mapping_sha256: str,
    requested_gap: float,
) -> dict | None:
    if not path.exists():
        return None
    archive = np.load(path, allow_pickle=False)
    if "schema" in archive:
        if str(_scalar(archive, "schema")) != "ps_dfsc_identity_day_cache_v2":
            return None
        if str(_scalar(archive, "input_sha256")) != input_sha256:
            return None
        if str(_scalar(archive, "day_sha256")) != day_sha256:
            return None
        if str(_scalar(archive, "mapping_sha256")) != mapping_sha256:
            return None
        return {key: archive[key] for key in archive.files}
    if not _legacy_cache_is_certified(archive, requested_gap):
        return None
    return {
        "commitment": archive["commitment"],
        "realized_total_cost": np.asarray(
            float(archive["realized_total_cost"])
        ),
        "planned_total_cost": np.asarray(
            float(_scalar(archive, "planned_total_cost", np.nan))
        ),
        "planned_mip_gap": np.asarray(
            float(archive["planned_mip_gap"])
        ),
        "planned_solve_time": np.asarray(
            float(_scalar(archive, "planned_solve_time", np.nan))
        ),
        "planned_success": np.asarray(True),
        "planned_status": np.asarray("legacy_cache_gap_certified"),
        "planned_dual_bound": np.asarray(np.nan),
        "planned_node_count": np.asarray(-1),
        "realized_mip_gap": np.asarray(
            float(_scalar(archive, "realized_mip_gap", np.nan))
        ),
        "realized_solve_time": np.asarray(
            float(_scalar(archive, "realized_solve_time", np.nan))
        ),
        "realized_success": np.asarray(True),
        "realized_status": np.asarray("legacy_cache_solution_available"),
        "realized_dual_bound": np.asarray(np.nan),
        "realized_node_count": np.asarray(-1),
        "cache_origin": np.asarray("legacy_gap_certified"),
    }


def _solve_day(
    reduced: np.ndarray,
    probability: np.ndarray,
    observed: np.ndarray,
    mapping: WindFarmMapping,
    *,
    mip_gap: float,
    time_limit: float,
) -> dict:
    planned = solve_two_stage_suc(
        reduced,
        probability,
        mip_gap=mip_gap,
        time_limit=time_limit,
    )
    realized = evaluate_realized(
        planned.first_stage,
        mapping.transform(observed),
        mip_gap=mip_gap,
        time_limit=time_limit,
    )
    return {
        "commitment": planned.first_stage.commitment,
        "realized_total_cost": np.asarray(realized.total_cost),
        "planned_total_cost": np.asarray(planned.total_cost),
        "planned_mip_gap": np.asarray(planned.mip_gap),
        "planned_solve_time": np.asarray(planned.solve_time_seconds),
        "planned_success": np.asarray(planned.success),
        "planned_status": np.asarray(planned.status),
        "planned_dual_bound": np.asarray(planned.mip_dual_bound),
        "planned_node_count": np.asarray(planned.mip_node_count),
        "realized_mip_gap": np.asarray(realized.mip_gap),
        "realized_solve_time": np.asarray(realized.solve_time_seconds),
        "realized_success": np.asarray(realized.success),
        "realized_status": np.asarray(realized.status),
        "realized_dual_bound": np.asarray(realized.mip_dual_bound),
        "realized_node_count": np.asarray(realized.mip_node_count),
        "cache_origin": np.asarray("solved"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare audited identity baseline and commitment caches"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--scenario-key", default="scenarios")
    parser.add_argument("--observation-key", default="observations")
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--identity-output", required=True)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=False)
    scenarios = source[args.scenario_key].astype(np.float64)
    observations = source[args.observation_key].astype(np.float64)
    days = source["day"] if "day" in source else (
        source["days"] if "days" in source else np.arange(len(scenarios))
    )
    if scenarios.ndim != 4 or scenarios.shape[2:] != (10, 24):
        raise ValueError("input scenarios must have shape [day,member,10,24]")
    if observations.shape != (len(scenarios), 10, 24):
        raise ValueError("observations do not align with scenarios")

    input_sha256 = file_sha256(args.input)
    mapping_sha256 = file_sha256(args.mapping)
    mapping = load_mapping(args.mapping)
    members = scenarios.shape[1]
    uniform = np.full(members, 1.0 / members)
    assignments = []
    reduced_scenarios = []
    reduced_probabilities = []
    records = []
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    for day in range(len(scenarios)):
        mapped = mapping.transform(scenarios[day])
        labels = fit_fixed_assignments(mapped, args.clusters)
        reduced, probability = weighted_cluster_reduction(
            mapped, uniform, labels
        )
        assignments.append(labels)
        reduced_scenarios.append(reduced)
        reduced_probabilities.append(probability)
        day_sha256 = _array_sha256(
            scenarios[day], observations[day], labels, reduced, probability
        )
        cache_path = cache / f"day_{day:03d}.npz"
        record = _load_cache(
            cache_path,
            input_sha256=input_sha256,
            day_sha256=day_sha256,
            mapping_sha256=mapping_sha256,
            requested_gap=args.mip_gap,
        )
        if record is None:
            record = _solve_day(
                reduced,
                probability,
                observations[day],
                mapping,
                mip_gap=args.mip_gap,
                time_limit=args.time_limit,
            )
        enriched = {
            "schema": np.asarray("ps_dfsc_identity_day_cache_v2"),
            "input_sha256": np.asarray(input_sha256),
            "day_sha256": np.asarray(day_sha256),
            "mapping_sha256": np.asarray(mapping_sha256),
            "requested_mip_gap": np.asarray(args.mip_gap),
            "time_limit_seconds": np.asarray(args.time_limit),
            **record,
        }
        np.savez_compressed(cache_path, **enriched)
        records.append(enriched)
        print(
            json.dumps(
                {
                    "day_index": day,
                    "days": len(scenarios),
                    "planned_mip_gap": float(enriched["planned_mip_gap"]),
                    "planned_success": bool(enriched["planned_success"]),
                    "planned_status": str(enriched["planned_status"]),
                    "cache_origin": str(enriched["cache_origin"]),
                }
            ),
            flush=True,
        )

    def vector(key, dtype=None):
        value = np.asarray([record[key].item() for record in records])
        return value if dtype is None else value.astype(dtype)

    commitments = np.stack([record["commitment"] for record in records])
    baseline_costs = vector("realized_total_cost", np.float64)
    planned_gaps = vector("planned_mip_gap", np.float64)
    training_output = Path(args.training_output)
    training_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        training_output,
        base_scenarios=scenarios.astype(np.float32),
        observations=observations.astype(np.float32),
        assignments=np.stack(assignments).astype(np.int64),
        commitments=commitments.astype(np.float32),
        baseline_realized_cost=baseline_costs,
        planned_total_cost=vector("planned_total_cost", np.float64),
        planned_mip_gap=planned_gaps,
        planned_solve_time=vector("planned_solve_time", np.float64),
        planned_success=vector("planned_success", bool),
        planned_status=vector("planned_status", str),
        planned_dual_bound=vector("planned_dual_bound", np.float64),
        planned_node_count=vector("planned_node_count", np.int64),
        realized_mip_gap=vector("realized_mip_gap", np.float64),
        realized_solve_time=vector("realized_solve_time", np.float64),
        realized_success=vector("realized_success", bool),
        realized_status=vector("realized_status", str),
        realized_dual_bound=vector("realized_dual_bound", np.float64),
        realized_node_count=vector("realized_node_count", np.int64),
        cache_origin=vector("cache_origin", str),
        input_sha256=np.asarray(input_sha256),
        mapping_sha256=np.asarray(mapping_sha256),
        requested_mip_gap=np.asarray(args.mip_gap),
        time_limit_seconds=np.asarray(args.time_limit),
        days=days,
    )
    identity_output = Path(args.identity_output)
    identity_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        identity_output,
        full_scenarios=scenarios.astype(np.float32),
        probabilities=np.broadcast_to(uniform, scenarios.shape[:2]).copy(),
        suc_scenarios=np.stack(reduced_scenarios),
        suc_probabilities=np.stack(reduced_probabilities),
        ess=np.full(len(scenarios), members, dtype=np.float64),
        entropy=np.full(len(scenarios), np.log(members), dtype=np.float64),
        transport_cost=np.zeros(len(scenarios), dtype=np.float64),
        used_fallback=np.zeros(len(scenarios), dtype=bool),
        fallback_reason=np.full(len(scenarios), "", dtype="<U1"),
        days=days,
        input_sha256=np.asarray(input_sha256),
        mapping_sha256=np.asarray(mapping_sha256),
    )
    print(
        json.dumps(
            {
                "training_output": str(training_output.resolve()),
                "identity_output": str(identity_output.resolve()),
                "days": len(scenarios),
                "baseline_mean_cost": float(np.mean(baseline_costs)),
                "planned_target_gap_pass_rate": float(
                    np.mean(planned_gaps <= args.mip_gap * (1.0 + 1e-6) + 1e-12)
                ),
                "planned_solver_success_rate": float(
                    np.mean(vector("planned_success", bool))
                ),
                "input_sha256": input_sha256,
                "mapping_sha256": mapping_sha256,
            }
        )
    )


if __name__ == "__main__":
    main()
