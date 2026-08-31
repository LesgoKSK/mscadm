from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from repro_scripts import caa_suc


def _lock() -> dict[str, Any]:
    return {
        "schema": "caa_rahc_selection_lock_v1",
        "selected": {
            "main_A4": {
                "selected": "A4_test",
                "fallback": False,
                "config": {"width_cap_reference": "atom_only"},
            },
            "A2": {
                "selected": "A2_test",
                "fallback": False,
                "config": {"width_cap_reference": "atom_only"},
            },
        },
    }


def _install_locked_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "outputs"
    root.mkdir()
    lock = _lock()
    (root / caa_suc.SELECTION_LOCK_NAME).write_text(
        json.dumps(lock), encoding="utf-8"
    )
    lock_sha256 = caa_suc._sha256_file(root / caa_suc.SELECTION_LOCK_NAME)
    context = SimpleNamespace(
        root=root,
        config={
            "outer_splits": [1, 2, 3],
            "model_seeds": [0, 1, 2],
            "sampling": {"scenarios": 4},
        },
    )
    monkeypatch.setattr(caa_suc, "load_context", lambda _config: context)
    monkeypatch.setattr(caa_suc, "validate_caa_lock", lambda _context: lock)

    families = ("A0", "A4", "A2")
    members = 4
    hours = 24
    for outer in (1, 2, 3):
        directory = root / f"outer{outer}" / "scenarios"
        directory.mkdir(parents=True)
        # Deliberately reverse the rows within each date.  Selection must sort
        # by date/zone metadata, not use archive position or method benefit.
        day_values: list[np.datetime64] = []
        zones: list[int] = []
        for day_offset in (4, 3, 2, 1, 0):
            for zone in (3, 1, 2):
                day_values.append(
                    np.datetime64(f"202{outer}-01-01") + np.timedelta64(day_offset, "D")
                )
                zones.append(zone)
        day = np.asarray(day_values, dtype="datetime64[D]")
        zone = np.asarray(zones, dtype=np.int64)
        cases = len(day)
        observations = np.broadcast_to(
            np.linspace(0.15, 0.75, hours, dtype=np.float64), (cases, hours)
        ).copy()
        observations += np.arange(cases, dtype=np.float64)[:, None] * 1e-4
        registry: list[dict[str, Any]] = []
        for family_index, family in enumerate(families):
            for seed in (0, 1, 2):
                if family == "A0":
                    level = 0.10
                elif family == "A4":
                    # Its values cannot influence the representative cases.
                    level = 0.70
                else:
                    level = 0.40
                scenarios = np.full(
                    (cases, members, hours),
                    level + 0.01 * seed + 0.001 * family_index,
                    dtype=np.float64,
                )
                metadata = {
                    "schema": "caa_rahc_locked_test_scenarios_v1",
                    "outer": outer,
                    "model_seed": seed,
                    "family": family,
                    "selection_lock_sha256": lock_sha256,
                }
                path = directory / f"{family}_seed{seed}_test_final.npz"
                np.savez(
                    path,
                    scenarios=scenarios,
                    observations=observations,
                    zone=zone,
                    day=day,
                    metadata=np.asarray(json.dumps(metadata)),
                )
                registry.append(
                    {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": caa_suc._sha256_file(path),
                    }
                )
        manifest = {
            "schema": "caa_rahc_locked_test_outputs_v1",
            "outer": outer,
            "selection_lock_sha256": lock_sha256,
            "families": list(families),
            "files": registry,
        }
        (directory / caa_suc.OUTER_TEST_MANIFEST_NAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return root, lock


def _install_fake_optimization(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_high_wind_plans: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[object]]:
    reductions: list[dict[str, Any]] = []
    solves: list[dict[str, Any]] = []
    systems: list[object] = []

    def fake_system() -> object:
        system = object()
        systems.append(system)
        return system

    def fake_reduce(
        scenarios: np.ndarray,
        clusters: int,
        *,
        capacity: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        reductions.append(
            {
                "scenarios": np.asarray(scenarios).copy(),
                "clusters": clusters,
                "capacity": capacity,
                "seed": seed,
            }
        )
        return (
            np.asarray(scenarios[:clusters]) * capacity,
            np.full(clusters, 1.0 / clusters, dtype=np.float64),
        )

    def fake_solve(
        wind: np.ndarray,
        probabilities: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        call = {
            "wind": np.asarray(wind).copy(),
            "probabilities": np.asarray(probabilities).copy(),
            **kwargs,
        }
        solves.append(call)
        planned = kwargs["fixed_commitment"] is None
        if fail_high_wind_plans and planned and float(np.mean(wind)) > 500.0:
            raise RuntimeError("synthetic planned failure")
        value = float(np.mean(wind))
        return {
            "status": "optimal",
            "commitment": np.ones((12, 24), dtype=np.float64),
            "total_cost": value + 6.0,
            "startup_cost": value + 1.0,
            "energy_cost": value + 2.0,
            "penalty_cost": value + 3.0,
            "wind_curtailment": value + 4.0,
            "load_shedding": value + 5.0,
        }

    monkeypatch.setattr(caa_suc, "rts24", fake_system)
    monkeypatch.setattr(caa_suc, "reduce_scenarios", fake_reduce)
    monkeypatch.setattr(caa_suc, "solve_suc", fake_solve)
    return reductions, solves, systems


def test_lock_guard_runs_before_any_outer_archive_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    context = SimpleNamespace(
        root=root,
        config={
            "outer_splits": [1, 2, 3],
            "model_seeds": [0, 1, 2],
            "sampling": {"scenarios": 4},
        },
    )
    monkeypatch.setattr(caa_suc, "load_context", lambda _config: context)
    monkeypatch.setattr(caa_suc, "validate_caa_lock", lambda _context: _lock())
    touched = False

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal touched
        touched = True
        raise AssertionError("outer archive was touched before the lock guard")

    monkeypatch.setattr(caa_suc, "_load_outer_archives", forbidden)
    with pytest.raises(RuntimeError, match="missing selection lock"):
        caa_suc.run_sensitivity("unused.json", clusters=2)
    assert not touched


def test_fixed_cases_equal_seed_pooling_and_fair_solver_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _install_locked_outputs(tmp_path, monkeypatch)
    reductions, solves, systems = _install_fake_optimization(monkeypatch)

    result = caa_suc.run_sensitivity(
        "unused.json",
        clusters=2,
        time_limit=7.5,
        reduction_seed=101,
        environment_seed=202,
    )
    daily = result["daily"]

    assert len(daily) == 18
    assert set(daily["family"]) == {"A0", "A4"}
    assert len(result["protocol"]["representative_case_selection"]["cases"]) == 9
    assert not result["protocol"]["representative_case_selection"][
        "uses_method_performance"
    ]
    assert result["protocol"]["success_gate_role"] == "none"
    assert not result["protocol"]["paired_design"]["environment_seed_consumed"]

    for (_outer, _slot), paired in daily.groupby(
        ["outer", "representative_slot"], sort=False
    ):
        assert paired["calendar_date"].nunique() == 1
        assert paired["zone"].tolist() == [2, 2]
        assert paired["archive_index"].nunique() == 1
        assert paired["pooled_scenario_count"].tolist() == [12, 12]
        assert paired["reduced_scenario_count"].tolist() == [2, 2]
        assert paired["reduction_seed"].nunique() == 1
        assert paired["environment_seed"].nunique() == 1
        assert paired["time_limit_seconds"].tolist() == [7.5, 7.5]

    assert len(reductions) == 18
    for first, second in zip(reductions[0::2], reductions[1::2]):
        assert first["scenarios"].shape == second["scenarios"].shape == (12, 24)
        assert first["clusters"] == second["clusters"] == 2
        assert first["capacity"] == second["capacity"] == 1200.0
        assert first["seed"] == second["seed"]

    assert len(systems) == 9
    assert len(solves) == 36
    for system in systems:
        paired_calls = [call for call in solves if call["system"] is system]
        assert len(paired_calls) == 4
        planned = [call for call in paired_calls if call["fixed_commitment"] is None]
        realized = [call for call in paired_calls if call["fixed_commitment"] is not None]
        assert len(planned) == len(realized) == 2
        assert np.array_equal(realized[0]["wind"], realized[1]["wind"])
        for call in paired_calls:
            assert call["time_limit"] == 7.5
            assert call["mip_gap"] == 0.01
            assert call["shedding_penalty"] == 1000.0
            assert call["curtailment_penalty"] == 80.0

    daily_path = root / "suc" / "daily_results.csv"
    summary_path = root / "suc" / "summary.csv"
    protocol_path = root / "suc" / "protocol.json"
    assert daily_path.is_file() and summary_path.is_file() and protocol_path.is_file()
    assert len(pd.read_csv(daily_path)) == 18
    assert set(pd.read_csv(summary_path)["family"]) == {"A0", "A4"}
    assert json.loads(protocol_path.read_text(encoding="utf-8"))[
        "statistical_scope"
    ].startswith("Descriptive sensitivity only")


def test_optional_a2_uses_the_same_nine_paired_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_locked_outputs(tmp_path, monkeypatch)
    _install_fake_optimization(monkeypatch)

    result = caa_suc.run_sensitivity(
        "unused.json", include_a2=True, clusters=2
    )

    assert len(result["daily"]) == 27
    assert set(result["daily"]["family"]) == {"A0", "A4", "A2"}
    assert list(result["summary"]["family"]) == ["A0", "A4", "A2"]
    assert set(result["protocol"]["locked_decisions"]) == {"A0", "A4", "A2"}


def test_solver_failure_is_recorded_and_does_not_disappear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_locked_outputs(tmp_path, monkeypatch)
    _install_fake_optimization(monkeypatch, fail_high_wind_plans=True)

    result = caa_suc.run_sensitivity("unused.json", clusters=2)
    daily = result["daily"]
    failed = daily.loc[daily["family"] == "A4"]

    assert len(failed) == 9
    assert not failed["planned_success"].any()
    assert set(failed["planned_status"]) == {"ERROR"}
    assert set(failed["planned_error_type"]) == {"RuntimeError"}
    assert failed["planned_error_message"].str.contains("synthetic planned failure").all()
    assert set(failed["realized_status"]) == {"NOT_RUN_PLANNED_FAILED"}
    assert failed["planned_total_cost"].isna().all()
    summary = result["summary"].set_index("family")
    assert summary.loc["A4", "planned_failed_cases"] == 9
    assert summary.loc["A0", "planned_failed_cases"] == 0
    assert result["protocol"]["solver"]["failed_method_cases"] == 9
    assert "WARNING: 9 method-case" in capsys.readouterr().err
