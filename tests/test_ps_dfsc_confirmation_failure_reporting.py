from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

import repro_scripts.run_ps_dfsc_exact_cached_v3 as wrapper


def test_locked_confirmation_retains_explicit_failed_rows(
    tmp_path, monkeypatch
):
    output = tmp_path / "exact.csv"
    summary = output.with_suffix(".summary.json")

    def incomplete_v2():
        pd.DataFrame(
            [
                {"status": "solution_available"},
                {"status": "failed_no_solution"},
            ]
        ).to_csv(output, index=False)
        summary.write_text(
            json.dumps(
                {
                    "schema": "ps_dfsc_cached_exact_evaluation_v2",
                    "cases": 2,
                    "successful_cases": 1,
                }
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            "exact validation remains incomplete after registered retries"
        )

    monkeypatch.setattr(wrapper.exact_v2, "main", incomplete_v2)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--output", str(output)],
    )
    wrapper.main()
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["failed_cases"] == 1
    assert result["incomplete_exact_evaluation_reported"] is True
    assert (
        result["confirmation_failure_policy"]
        == "failed days retained explicitly; no replacement"
    )


def test_unrelated_runtime_error_is_not_suppressed(
    tmp_path, monkeypatch
):
    output = tmp_path / "exact.csv"

    def unrelated_error():
        raise RuntimeError("unexpected corruption")

    monkeypatch.setattr(wrapper.exact_v2, "main", unrelated_error)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--output", str(output)],
    )
    with pytest.raises(RuntimeError, match="unexpected corruption"):
        wrapper.main()
