"""Parallel helper for the fixed-model midpoint commitment cache.

This helper is deliberately separate from the canonical trainer.  It fills
only missing per-day cache files for a model whose resume state is already
frozen at the midpoint.  The canonical trainer remains the authority: it
validates cache hashes before reuse and can continue serially if this helper
is stopped or fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from ps_dfsc.inference import calibrate
from ps_dfsc.mapping import WindFarmMapping
from ps_dfsc.midpoint_suc_recovery import solve_midpoint_suc_with_recovery
from ps_dfsc.model import PSDFSCNetwork
from ps_dfsc.manifest import file_sha256


_MODEL: PSDFSCNetwork | None = None
_MAPPING: WindFarmMapping | None = None
_DEVICE: str = "cuda"
_CLUSTERS: int = 20
_MIP_GAP: float = 0.001
_TIME_LIMIT: float = 600.0
_MODEL_SHA: str = ""
_MAPPING_SHA: str = ""


def _model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().view(np.uint8))
    return digest.hexdigest()


def _day_sha256(scenarios: np.ndarray, assignments: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (scenarios, assignments):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _init_worker(
    model_state: dict[str, torch.Tensor],
    mapping_payload: dict,
    device: str,
    clusters: int,
    mip_gap: float,
    time_limit: float,
    model_sha: str,
    mapping_sha: str,
) -> None:
    global _MODEL, _MAPPING, _DEVICE, _CLUSTERS, _MIP_GAP, _TIME_LIMIT
    global _MODEL_SHA, _MAPPING_SHA
    torch.set_num_threads(1)
    _DEVICE = device
    _CLUSTERS = int(clusters)
    _MIP_GAP = float(mip_gap)
    _TIME_LIMIT = float(time_limit)
    _MODEL_SHA = model_sha
    _MAPPING_SHA = mapping_sha
    _MAPPING = WindFarmMapping(
        groups=tuple(tuple(int(zone) for zone in group) for group in mapping_payload["groups"]),
        wind_buses=tuple(int(bus) for bus in mapping_payload["wind_buses"]),
        zone_capacity_mw=float(mapping_payload["zone_capacity_mw"]),
    )
    _MODEL = PSDFSCNetwork()
    _MODEL.load_state_dict(model_state)
    _MODEL.to(device)
    _MODEL.eval()


def _solve_day(task: tuple[int, np.ndarray, np.ndarray]) -> dict:
    if _MODEL is None or _MAPPING is None:
        raise RuntimeError("parallel midpoint worker was not initialized")
    day, values, assignments = task
    day_hash = _day_sha256(values, assignments)
    distribution = calibrate(
        _MODEL,
        values,
        _MAPPING,
        clusters=_CLUSTERS,
        assignments=assignments,
        device=_DEVICE,
    )
    solved = solve_midpoint_suc_with_recovery(
        distribution.suc_scenarios,
        distribution.suc_probabilities,
        mip_gap=_MIP_GAP,
        time_limit=_TIME_LIMIT,
    )
    return {
        "day": int(day),
        "day_sha256": day_hash,
        "commitment": solved.first_stage.commitment,
        "mip_gap": float(solved.mip_gap),
        "solver_success": bool(solved.success),
        "solver_status": str(solved.status),
        "dual_bound": float(solved.mip_dual_bound),
        "node_count": int(solved.mip_node_count),
        "solve_time_seconds": float(solved.solve_time_seconds),
    }


def _write_cache(
    path: Path,
    result: dict,
    *,
    mapping_sha: str,
    mip_gap: float,
    time_limit: float,
) -> bool:
    if path.exists():
        return False
    payload = {
        "schema": np.asarray("ps_dfsc_midpoint_commitment_cache_v1"),
        "midpoint_model_sha256": np.asarray(_MODEL_SHA),
        "day_sha256": np.asarray(result["day_sha256"]),
        "mapping_sha256": np.asarray(mapping_sha),
        "requested_mip_gap": np.asarray(mip_gap),
        "time_limit_seconds": np.asarray(time_limit),
        "commitment": result["commitment"],
        "mip_gap": np.asarray(result["mip_gap"]),
        "solver_success": np.asarray(result["solver_success"]),
        "solver_status": np.asarray(result["solver_status"]),
        "dual_bound": np.asarray(result["dual_bound"]),
        "node_count": np.asarray(result["node_count"]),
        "solve_time_seconds": np.asarray(result["solve_time_seconds"]),
        "cache_origin": np.asarray("parallel_solved"),
    }
    temporary = path.with_name(path.stem + ".parallel.tmp.npz")
    np.savez_compressed(temporary, **payload)
    try:
        temporary.replace(path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-day", type=int, default=7)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")
    if not 0 <= args.start_day < 100:
        raise ValueError("start-day must be in [0,99]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reproducible parallel calibration")

    source = np.load(args.input, allow_pickle=False)
    scenarios = source["base_scenarios"]
    assignments = source["assignments"]
    resume = torch.load(args.resume, map_location="cpu", weights_only=False)
    state = resume["state"]
    if int(state["next_epoch"]) != 35 or bool(resume.get("midpoint_refresh_completed")):
        raise RuntimeError("resume state is not the frozen pre-refresh midpoint")
    model = PSDFSCNetwork()
    model.load_state_dict(state["model_state"])
    model_sha = _model_sha256(model)
    mapping_payload = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    mapping_sha = file_sha256(args.mapping)
    cache_dir = Path(args.output).with_suffix(".refresh_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for day in range(args.start_day, len(scenarios)):
        path = cache_dir / f"day_{day:03d}.npz"
        if not path.exists():
            tasks.append((day, scenarios[day], assignments[day]))
    print(json.dumps({"tasks": len(tasks), "workers": args.workers, "model_sha256": model_sha}), flush=True)
    if not tasks:
        return

    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(
            state["model_state"],
            mapping_payload,
            args.device,
            args.clusters,
            args.mip_gap,
            args.time_limit,
            model_sha,
            mapping_sha,
        ),
    ) as pool:
        futures = [pool.submit(_solve_day, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            path = cache_dir / f"day_{result['day']:03d}.npz"
            wrote = _write_cache(
                path,
                result,
                mapping_sha=mapping_sha,
                mip_gap=args.mip_gap,
                time_limit=args.time_limit,
            )
            print(
                json.dumps(
                    {
                        "phase": "parallel_commitment_refresh",
                        "day_index": result["day"],
                        "wrote": wrote,
                        "mip_gap": result["mip_gap"],
                        "solver_success": result["solver_success"],
                        "solve_time_seconds": result["solve_time_seconds"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
