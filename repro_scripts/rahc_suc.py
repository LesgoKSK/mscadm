"""Descriptive RTS-24 SUC comparison for legacy G0 and seed-0 RAHC.

This script deliberately reuses the seven fixed zone-date cases from the
CR-MS-CADM reconstruction.  They span only five unique calendar dates, so the
result is a sensitivity study rather than a statistical comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from repro.suc import reduce_scenarios, solve_suc


DEFAULT_DAYS = [37, 20, 134, 253, 153, 315, 420]
MODEL_SEED = 0
WIND_CAPACITY_MW = 1200.0


def _archive_candidates(root: Path, model: str) -> list[Path]:
    """Return deterministic candidates used by current and legacy layouts."""

    scenario_root = root / "scenarios"
    candidates = [
        scenario_root / f"{model}_seed{MODEL_SEED}_test.npz",
        scenario_root / f"seed{MODEL_SEED}_{model}_test.npz",
        scenario_root / f"{model}_test_seed{MODEL_SEED}.npz",
        scenario_root / "test" / f"{model}_seed{MODEL_SEED}.npz",
        scenario_root / f"seed{MODEL_SEED}" / f"{model}_test.npz",
        scenario_root / model / f"seed{MODEL_SEED}_test.npz",
    ]
    if model == "legacy_G0":
        # The legacy comparator is an immutable CR-MS-CADM artifact and need
        # not be copied into the RAHC output tree.
        candidates.extend(
            [
                Path("outputs/cr_mscadm/scenarios/full_seed0_test_calibrated.npz"),
                root.parent / "cr_mscadm" / "scenarios" / "full_seed0_test_calibrated.npz",
            ]
        )
    return candidates


def _resolve_archive(root: Path, model: str) -> Path:
    for candidate in _archive_candidates(root, model):
        if candidate.is_file():
            return candidate

    # Keep alternate experiment layouts usable without silently choosing
    # between ambiguous files.  Both the model label, seed, and test split
    # must be present in the path.
    model_token = model.casefold()
    matches = sorted(
        path
        for path in root.rglob("*.npz")
        if model_token in str(path).casefold()
        and f"seed{MODEL_SEED}" in str(path).casefold()
        and "test" in str(path).casefold()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        rendered = "\n  - ".join(str(path) for path in matches)
        raise RuntimeError(f"ambiguous {model} seed-0 test archives:\n  - {rendered}")
    expected = "\n  - ".join(str(path) for path in _archive_candidates(root, model))
    raise FileNotFoundError(f"no {model} seed-0 test archive found; checked:\n  - {expected}")


def _validate_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"scenarios", "observations", "zone", "day"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing archive fields: {missing}")
        loaded = {name: np.asarray(archive[name]) for name in required}

    scenarios = loaded["scenarios"]
    observations = loaded["observations"]
    zones = loaded["zone"]
    days = loaded["day"]
    cases = len(scenarios)
    if scenarios.ndim != 3 or scenarios.shape[2] != 24:
        raise ValueError(f"{path}: scenarios must have shape [case, member, 24]")
    if observations.shape != (cases, 24):
        raise ValueError(f"{path}: observations must have shape [case, 24]")
    if zones.shape != (cases,) or days.shape != (cases,):
        raise ValueError(f"{path}: zone and day must each have shape [case]")
    if max(DEFAULT_DAYS) >= cases:
        raise ValueError(f"{path}: fixed case index exceeds archive length {cases}")
    if not np.all(np.isfinite(scenarios)) or not np.all(np.isfinite(observations)):
        raise ValueError(f"{path}: scenarios and observations must be finite")
    return loaded


def _calendar_text(value: np.generic | object) -> str:
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="D")
    return str(value)


def _check_common_cases(archives: dict[str, dict[str, np.ndarray]]) -> list[dict[str, int | str]]:
    reference_name = next(iter(archives))
    reference = archives[reference_name]
    selected = np.asarray(DEFAULT_DAYS, dtype=int)
    for model, archive in archives.items():
        if not np.array_equal(archive["zone"][selected], reference["zone"][selected]):
            raise ValueError(f"zone metadata differs between {reference_name} and {model}")
        if not np.array_equal(archive["day"][selected], reference["day"][selected]):
            raise ValueError(f"calendar-day metadata differs between {reference_name} and {model}")
        if not np.allclose(
            archive["observations"][selected],
            reference["observations"][selected],
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"realized observations differ between {reference_name} and {model}")

    cases = [
        {
            "day_index": int(index),
            "zone": int(reference["zone"][index]),
            "calendar_date": _calendar_text(reference["day"][index]),
        }
        for index in DEFAULT_DAYS
    ]
    unique_dates = {case["calendar_date"] for case in cases}
    if len(cases) != 7 or len(unique_dates) != 5:
        raise ValueError(
            "the frozen SUC protocol must contain seven zone-date cases "
            "spanning exactly five unique calendar dates"
        )
    return cases


def _record(
    *,
    model: str,
    case: dict[str, int | str],
    planned: dict[str, np.ndarray | float | str],
    realized: dict[str, np.ndarray | float | str],
) -> dict[str, float | int | str]:
    record: dict[str, float | int | str] = {
        "model": model,
        "model_seed": MODEL_SEED,
        **case,
        "planned_status": str(planned["status"]),
        "realized_status": str(realized["status"]),
    }
    for stage, result in (("planned", planned), ("realized", realized)):
        for metric in (
            "total_cost",
            "startup_cost",
            "energy_cost",
            "penalty_cost",
            "wind_curtailment",
            "load_shedding",
        ):
            record[f"{stage}_{metric}"] = float(result[metric])
    return record


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "planned_total_cost",
        "planned_startup_cost",
        "planned_energy_cost",
        "planned_penalty_cost",
        "planned_wind_curtailment",
        "planned_load_shedding",
        "realized_total_cost",
        "realized_startup_cost",
        "realized_energy_cost",
        "realized_penalty_cost",
        "realized_wind_curtailment",
        "realized_load_shedding",
    ]
    summary = frame.groupby("model", sort=False)[numeric].mean()
    summary.columns = [f"mean_{name}" for name in summary.columns]
    summary.insert(0, "zone_date_cases", frame.groupby("model", sort=False).size())
    summary.insert(
        1,
        "unique_calendar_dates",
        frame.groupby("model", sort=False)["calendar_date"].nunique(),
    )
    return summary.reset_index()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the descriptive legacy-G0 versus C6-RAHC seed-0 RTS-24 SUC comparison"
    )
    parser.add_argument("--root", type=Path, default=Path("outputs/rahc_cr_mscadm"))
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.clusters < 1:
        raise ValueError("--clusters must be positive")
    if args.time_limit <= 0:
        raise ValueError("--time-limit must be positive")

    sources = {
        model: _resolve_archive(args.root, model)
        for model in ("legacy_G0", "C6_RAHC")
    }
    archives = {model: _validate_archive(path) for model, path in sources.items()}
    cases = _check_common_cases(archives)

    records: list[dict[str, float | int | str]] = []
    for model, archive in archives.items():
        for case in cases:
            day_index = int(case["day_index"])
            reduced, probabilities = reduce_scenarios(
                archive["scenarios"][day_index],
                args.clusters,
                capacity=WIND_CAPACITY_MW,
                seed=day_index,
            )
            planned = solve_suc(
                reduced,
                probabilities,
                time_limit=args.time_limit,
            )
            realized = solve_suc(
                archive["observations"][day_index][None, :] * WIND_CAPACITY_MW,
                np.ones(1, dtype=float),
                fixed_commitment=np.asarray(planned["commitment"]),
                time_limit=args.time_limit,
            )
            record = _record(model=model, case=case, planned=planned, realized=realized)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    output = args.root / "suc"
    output.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(records)
    summary = _summarize(daily)
    daily.to_csv(output / "daily_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    unique_dates = sorted({str(case["calendar_date"]) for case in cases})
    protocol = {
        "experiment": "RAHC-CR-MS-CADM descriptive SUC sensitivity",
        "models": ["legacy_G0", "C6_RAHC"],
        "model_seed": MODEL_SEED,
        "common_day_indices": DEFAULT_DAYS,
        "zone_date_cases": cases,
        "zone_date_case_count": len(cases),
        "unique_calendar_dates": unique_dates,
        "unique_calendar_date_count": len(unique_dates),
        "clusters": args.clusters,
        "kmeans_seed_rule": "day_index; identical for both forecast methods",
        "time_limit_seconds_per_solve": args.time_limit,
        "wind_capacity_mw": int(WIND_CAPACITY_MW),
        "capacity_interpretation": (
            "Each selected archive row is one normalized GEFCom zone forecast; "
            "it is scaled to a single-zone 1200 MW system-level proxy. This is "
            "not an aggregation of all ten GEFCom zones."
        ),
        "two_stage_protocol": (
            "K-means-reduced forecast scenarios determine the planned unit commitment; "
            "the common realized observation is then solved with that commitment fixed."
        ),
        "archives": {model: str(path.resolve()) for model, path in sources.items()},
        "fairness": (
            "Both methods use identical zone-date indices, realized observations, "
            "1200 MW proxy capacity, cluster count, K-means seed rule, solver, and penalties."
        ),
        "statistical_scope": (
            "Descriptive comparison only, not a statistical or inferential test: the seven "
            "fixed zone-date cases contain only five unique calendar dates."
        ),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
