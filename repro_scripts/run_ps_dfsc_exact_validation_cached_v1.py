"""Restartable per-day strict exact evaluation for validation selection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ps_dfsc.evaluation_publication as evaluation
from ps_dfsc.exact_suc_publication_v3 import (
    evaluate_realized,
    solve_two_stage_suc,
)
from ps_dfsc.manifest import file_sha256
from repro_scripts.run_ps_dfsc import (
    _load_distributions,
    _load_mapping,
)


evaluation.solve_two_stage_suc = solve_two_stage_suc
evaluation.evaluate_realized = evaluate_realized


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _row_dict(frame: pd.DataFrame) -> dict:
    return {
        str(key): _native(value)
        for key, value in frame.iloc[0].to_dict().items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrated", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--per-day-attempts", type=int, default=3)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    calibrated_path = Path(args.calibrated)
    truth_path = Path(args.truth)
    mapping_path = Path(args.mapping)
    calibrated = np.load(calibrated_path, allow_pickle=False)
    truth = np.load(truth_path, allow_pickle=False)
    distributions = _load_distributions(calibrated)
    observations = truth["observations"]
    dates = (
        truth["days"]
        if "days" in truth
        else truth["day"]
        if "day" in truth
        else None
    )
    if dates is None or len(dates) != len(distributions):
        raise ValueError("validation truth lacks aligned date labels")
    mapping = _load_mapping(mapping_path)
    farm_truth = mapping.transform(observations)
    hashes = {
        "calibrated_sha256": file_sha256(calibrated_path),
        "truth_sha256": file_sha256(truth_path),
        "mapping_sha256": file_sha256(mapping_path),
    }
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    attempt_log = cache_dir / "failed_attempts.jsonl"
    rows = []
    reused = 0
    newly_solved = 0
    for day, distribution in enumerate(distributions):
        date = str(np.asarray(dates[day]).astype("datetime64[D]"))
        metadata = {
            "schema": "ps_dfsc_exact_validation_day_v1",
            "outer": int(args.outer),
            "method": args.method,
            "day_index": day,
            "date": date,
            "requested_mip_gap": float(args.mip_gap),
            "time_limit_seconds": float(args.time_limit),
            **hashes,
        }
        cache_path = cache_dir / f"day_{day:03d}.json"
        cached = None
        if cache_path.is_file():
            candidate = json.loads(
                cache_path.read_text(encoding="utf-8")
            )
            if (
                candidate.get("metadata") == metadata
                and candidate.get("row", {}).get("status")
                == "solution_available"
            ):
                cached = candidate["row"]
        if cached is not None:
            rows.append(cached)
            reused += 1
            print(
                json.dumps(
                    {
                        "day_index": day,
                        "days": len(distributions),
                        "status": "reused_solution",
                    }
                ),
                flush=True,
            )
            continue
        last_row = None
        for attempt in range(1, args.per_day_attempts + 1):
            frame = evaluation.run_exact_cases(
                [distribution],
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
                    "solved_at_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "attempt": attempt,
                }
                temporary = cache_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
                newly_solved += 1
                break
            failure = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **metadata,
                "attempt": attempt,
                "row": last_row,
            }
            with attempt_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, sort_keys=True) + "\n")
        rows.append(last_row)
        print(
            json.dumps(
                {
                    "day_index": day,
                    "days": len(distributions),
                    "status": last_row["status"],
                }
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "schema": "ps_dfsc_cached_exact_validation_v1",
        **evaluation.summarize_exact_cases(frame),
        "outer": int(args.outer),
        "method": args.method,
        "cache_reused_cases": reused,
        "newly_solved_cases": newly_solved,
        "per_day_attempts": int(args.per_day_attempts),
        "strict_exact_wrapper": "exact_suc_publication_v3",
        "test_truth_accessed": False,
        **hashes,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output.resolve()), **summary}))
    if int(summary["successful_cases"]) != len(distributions):
        raise RuntimeError(
            "exact validation remains incomplete after registered retries"
        )


if __name__ == "__main__":
    main()
