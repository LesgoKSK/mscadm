import json
import sys

import repro_scripts.run_ps_dfsc_exact_cached_v2 as wrapper


def test_validation_provenance_does_not_claim_test_access(
    monkeypatch, tmp_path
):
    output = tmp_path / "exact.csv"
    summary = output.with_suffix(".summary.json")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    day = cache_dir / "day_000.json"

    def fake_main():
        output.write_text("status\nsolution_available\n", encoding="utf-8")
        summary.write_text(
            json.dumps({"schema": "old"}), encoding="utf-8"
        )
        day.write_text(
            json.dumps({"metadata": {}, "row": {}}), encoding="utf-8"
        )

    monkeypatch.setattr(wrapper.cached, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrapper",
            "--split-role",
            "validation",
            "--outer",
            "3",
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(output),
        ],
    )
    wrapper.main()
    payload = json.loads(summary.read_text(encoding="utf-8"))
    cached = json.loads(day.read_text(encoding="utf-8"))
    assert payload["test_truth_accessed"] is False
    assert payload["split_role"] == "validation"
    assert cached["evaluation_provenance"]["test_truth_accessed"] is False
