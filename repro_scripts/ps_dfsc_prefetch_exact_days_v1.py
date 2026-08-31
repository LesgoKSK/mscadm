"""Prefetch independent confirmation days into the audited exact cache.

The locked confirmation driver evaluates days in date order.  Each day is an
independent exact MILP evaluation, so this helper can safely solve a disjoint
future range ahead of the driver.  It writes the same per-day cache metadata
and row schema as ``run_ps_dfsc_exact_validation_cached_v1``; the canonical
driver later validates the metadata and reuses the solutions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ps_dfsc.evaluation_publication as evaluation
from ps_dfsc.exact_suc_publication_v3 import evaluate_realized, solve_two_stage_suc
from ps_dfsc.manifest import file_sha256
from repro_scripts.run_ps_dfsc import _load_distributions, _load_mapping


evaluation.solve_two_stage_suc = solve_two_stage_suc
evaluation.evaluate_realized = evaluate_realized


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _row_dict(frame):
    return {
        str(key): _native(value)
        for key, value in frame.iloc[0].to_dict().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--calibrated", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--start-day", type=int, required=True)
    parser.add_argument("--end-day", type=int, required=True)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--per-day-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.start_day < 0 or args.end_day < args.start_day:
        raise ValueError("invalid day range")

    calibrated_path = Path(args.calibrated).resolve()
    truth_path = Path(args.truth).resolve()
    mapping_path = Path(args.mapping).resolve()
    calibrated = np.load(calibrated_path, allow_pickle=False)
    truth = np.load(truth_path, allow_pickle=False)
    distributions = _load_distributions(calibrated)
    dates = truth["days"] if "days" in truth else truth["day"]
    mapping = _load_mapping(mapping_path)
    farm_truth = mapping.transform(truth["observations"])
    if args.end_day >= len(distributions) or len(dates) != len(distributions):
        raise ValueError("day range is outside aligned confirmation archive")

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "calibrated_sha256": file_sha256(calibrated_path),
        "truth_sha256": file_sha256(truth_path),
        "mapping_sha256": file_sha256(mapping_path),
    }
    solved = 0
    reused = 0
    failed = 0
    for day in range(args.start_day, args.end_day + 1):
        date = str(np.asarray(dates[day]).astype("datetime64[D]"))
        metadata = {
            "schema": "ps_dfsc_exact_validation_day_v1",
            "outer": int(args.outer),
            "method": args.method,
            "day_index": int(day),
            "date": date,
            "requested_mip_gap": float(args.mip_gap),
            "time_limit_seconds": float(args.time_limit),
            **hashes,
        }
        cache_path = cache_dir / f"day_{day:03d}.json"
        if cache_path.is_file():
            try:
                current = json.loads(cache_path.read_text(encoding="utf-8"))
                if (
                    current.get("metadata") == metadata
                    and current.get("row", {}).get("status")
                    == "solution_available"
                ):
                    reused += 1
                    print(json.dumps({"day_index": day, "status": "reused"}), flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        last_row = None
        for attempt in range(1, args.per_day_attempts + 1):
            frame = evaluation.run_exact_cases(
                [distributions[day]],
                farm_truth[day : day + 1],
                outer=args.outer,
                method=args.method,
                dates=np.asarray(dates[day : day + 1]),
                mip_gap=args.mip_gap,
                time_limit=args.time_limit,
            )
            last_row = _row_dict(frame)
            if last_row["status"] == "solution_available":
                payload = {
                    "metadata": metadata,
                    "row": last_row,
                    "solved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "attempt": int(attempt),
                    "prefetched_by": "ps_dfsc_prefetch_exact_days_v1",
                }
                temporary = cache_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
                solved += 1
                print(json.dumps({"day_index": day, "status": "solution_available"}), flush=True)
                break
            failure = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **metadata,
                "attempt": int(attempt),
                "row": last_row,
                "prefetched_by": "ps_dfsc_prefetch_exact_days_v1",
            }
            with (cache_dir / "failed_attempts_prefetch.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, sort_keys=True) + "\n")
        else:
            failed += 1
            print(json.dumps({"day_index": day, "status": "failed_no_solution"}), flush=True)

    print(
        json.dumps(
            {
                "outer": int(args.outer),
                "start_day": int(args.start_day),
                "end_day": int(args.end_day),
                "solved": solved,
                "reused": reused,
                "failed": failed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
