from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from caa_rahc.candidates_nested import FAMILIES
from caa_rahc.selection import CandidateRecord, METRIC_KEYS
from repro_scripts import run_caa_calibration as runner


def _record(name: str, family: str, outer: int, *, days: int = 50) -> CandidateRecord:
    base = np.arange(3 * days, dtype=np.float64).reshape(3, days) + 1.0 + outer
    policy = "baseline" if family == "A0" else (
        "unconstrained" if family == "A5" else "constrained"
    )
    return CandidateRecord(
        name=name,
        metrics={metric: base + index for index, metric in enumerate(METRIC_KEYS)},
        conditional_ace90=np.full(3, 0.01 * outer),
        selection_policy=policy,
        metadata={"family": family, "config": {"outer_invariant": True}},
    )


class _FakeCandidateResult:
    def __init__(self, outer: int) -> None:
        self.audit = {"outer": outer}
        self._records = {
            family: [_record("A0" if family == "A0" else f"{family}_candidate", family, outer)]
            for family in FAMILIES
        }

    def records_for_family(self, family: str) -> list[CandidateRecord]:
        return list(self._records[family])


def test_outer_catalog_aggregation_retains_dates_and_stacks_nine_replicates() -> None:
    aggregated = runner._aggregate_catalog(
        {outer: _FakeCandidateResult(outer) for outer in (1, 2, 3)}
    )
    assert set(aggregated) == set(FAMILIES)
    for family in FAMILIES:
        assert len(aggregated[family]) == 1
        record = aggregated[family][0]
        for metric in METRIC_KEYS:
            assert np.asarray(record.metrics[metric]).shape == (9, 50)
        assert np.asarray(record.conditional_ace90).shape == (9,)


def test_exact_a0_fallback_returns_copy_without_calling_transform(monkeypatch) -> None:
    baseline = np.linspace(0.0, 1.0, 2 * 8 * 24).reshape(2, 8, 24)

    def forbidden(*args, **kwargs):  # pragma: no cover - failure sentinel
        raise AssertionError("fallback must not call the CAA transformation")

    monkeypatch.setattr(runner, "atom_aware_logit_hinge_tail", forbidden)
    output, audit = runner.apply_locked_family(
        "A4",
        {"selected": "A0", "fallback": True, "config": {"definition": "exact A0"}},
        baseline,
    )
    assert np.array_equal(output, baseline)
    assert output is not baseline
    assert audit["exact_A0"] is True
    assert audit["transform_called"] is False


def test_registered_file_same_size_hash_drift_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "gate.pt"
    artifact.write_bytes(b"abcd")
    record = {
        "path": artifact.name,
        "bytes": 4,
        "sha256": runner.outer_runner.sha256_file(artifact),
    }
    runner.validate_registered_files(tmp_path, [record])
    artifact.write_bytes(b"abce")
    with pytest.raises(RuntimeError, match="hash drift"):
        runner.validate_registered_files(tmp_path, [record])


def test_apply_validates_every_final_before_requesting_test_raw(monkeypatch, tmp_path: Path) -> None:
    context = runner.ExperimentContext(
        config_path=tmp_path / "config.json",
        config={"outer_splits": [1, 2, 3]},
        manifest={},
        root=tmp_path,
        bundles={1: object(), 2: object(), 3: object()},
    )
    test_loader_called = False

    monkeypatch.setattr(runner, "load_context", lambda config: context)
    monkeypatch.setattr(runner, "validate_caa_lock", lambda context: {"selected": {}})

    def hash_drift(context, lock, *, outer):
        raise RuntimeError(f"outer{outer} final calibrator hash drift")

    def forbidden_test_loader(context, *, outer, split):  # pragma: no cover
        nonlocal test_loader_called
        test_loader_called = True
        raise AssertionError("test loader ran before final validation")

    monkeypatch.setattr(runner, "load_final_outer", hash_drift)
    monkeypatch.setattr(runner, "load_raw_inputs", forbidden_test_loader)
    with pytest.raises(RuntimeError, match="hash drift"):
        runner.apply_test(tmp_path / "config.json")
    assert test_loader_called is False


def test_select_and_lock_requests_calibration_only(monkeypatch, tmp_path: Path) -> None:
    context = runner.ExperimentContext(
        config_path=tmp_path / "config.json",
        config={"device": "cpu", "baseline": {"calibration_strength": 1.0}},
        manifest={},
        root=tmp_path,
        bundles={1: object(), 2: object(), 3: object()},
    )
    (tmp_path / runner.outer_runner.FROZEN_MANIFEST_NAME).write_text(
        "{}", encoding="utf-8"
    )
    requested_splits: list[str] = []

    def fake_raw(context, *, outer, split):
        requested_splits.append(split)
        if split == "test":  # pragma: no cover - explicit leakage sentinel
            raise AssertionError("selection touched a test archive")
        return SimpleNamespace(
            archive_sha256_by_seed={0: f"{outer}-0", 1: f"{outer}-1", 2: f"{outer}-2"}
        )

    fake_results = {outer: _FakeCandidateResult(outer) for outer in (1, 2, 3)}
    fake_run = runner.SelectionRun(
        groupings={outer: {"mean_edges": [], "spread_edges": []} for outer in (1, 2, 3)},
        grouping_sha256={outer: f"group-{outer}" for outer in (1, 2, 3)},
        outer_results=fake_results,
        aggregated_records={family: [] for family in FAMILIES},
        selector_audits={},
        selected={
            key: {
                "family": "A4" if key == "main_A4" else key,
                "selected": "A0",
                "fallback": True,
                "config": {"definition": "exact A0"},
                "decision_key": "constrained_A4",
            }
            for key in runner.LOCK_SELECTION_KEYS
        },
        atom_score_rows=[],
    )

    class FakeGrid:
        def to_dict(self):
            return {"width_cap_reference": "atom_only"}

    monkeypatch.setattr(runner, "load_context", lambda config: context)
    monkeypatch.setattr(runner, "load_raw_inputs", fake_raw)
    monkeypatch.setattr(
        runner, "run_calibration_selection", lambda context, raw, device: fake_run
    )
    monkeypatch.setattr(
        runner, "candidate_grid_from_config", lambda config, device: FakeGrid()
    )
    monkeypatch.setattr(
        runner,
        "_selection_audit_payload",
        lambda context, raw, run, grid: {"test_access": "none"},
    )
    monkeypatch.setattr(runner, "_selection_summary_rows", lambda run: [])
    monkeypatch.setattr(
        runner,
        "analysis_code_fingerprint",
        lambda: {"algorithm": "sha256", "combined_sha256": "analysis", "files": []},
    )
    monkeypatch.setattr(runner, "_final_fit_protocol", lambda config: {"fit": "calibration"})
    monkeypatch.setattr(
        runner.outer_runner,
        "selection_lock_identity",
        lambda root, manifest: {
            "config_sha256": "config",
            "base_code_sha256": "base",
            "code_sha256": "selection",
            "manifest_sha256": "manifest",
            "protocol_sha256": {"outer1": "p1", "outer2": "p2", "outer3": "p3"},
        },
    )

    lock = runner.select_and_lock(tmp_path / "config.json", device="cpu")
    assert requested_splits == ["calibration", "calibration", "calibration"]
    assert lock["test_archives_accessed"] is False
    assert lock["authoritative_module"] == runner.AUTHORITATIVE_MODULE
    assert lock["full_code_sha256"] == "analysis"
    assert lock["analysis_sha256"] == lock["selection_audit_sha256"]
    assert set(lock["selected_configs"]) == set(runner.APPLIED_FAMILIES)
    assert isinstance(lock["locked_at_utc"], str)
    lock_path = tmp_path / runner.outer_runner.SELECTION_LOCK_NAME
    assert lock_path.is_file()
    audit = runner.json.loads(
        (tmp_path / runner.SELECTION_DIR_NAME / "selection.audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["authoritative_module"] == runner.AUTHORITATIVE_MODULE
    assert audit["full_code_sha256"] == "analysis"
    assert set(audit["selected_configs"]) == set(runner.APPLIED_FAMILIES)


def test_analysis_registry_includes_authoritative_nested_and_locking_code() -> None:
    assert "caa_rahc/candidates_nested.py" in runner.ANALYSIS_CODE_PATHS
    assert "caa_rahc/aggregate.py" in runner.ANALYSIS_CODE_PATHS
    assert "repro_scripts/run_caa_calibration.py" in runner.ANALYSIS_CODE_PATHS





