import json
import sys

import numpy as np
import pandas as pd

import repro_scripts.run_ps_dfsc_exact_validation_cached_v1 as cached


def test_successful_exact_day_is_reused(monkeypatch, tmp_path):
    calibrated = tmp_path / "calibrated.npz"
    truth = tmp_path / "truth.npz"
    mapping = tmp_path / "mapping.json"
    output = tmp_path / "exact.csv"
    cache_dir = tmp_path / "cache"
    np.savez_compressed(
        calibrated,
        full_scenarios=np.zeros((1, 2, 10, 24)),
        probabilities=np.full((1, 2), 0.5),
        suc_scenarios=np.zeros((1, 1, 6, 24)),
        suc_probabilities=np.ones((1, 1)),
        ess=np.asarray([2.0]),
        entropy=np.asarray([np.log(2.0)]),
        transport_cost=np.asarray([0.0]),
        used_fallback=np.asarray([False]),
        fallback_reason=np.asarray([""]),
    )
    np.savez_compressed(
        truth,
        observations=np.zeros((1, 10, 24)),
        days=np.asarray(["2013-01-01"], dtype="datetime64[D]"),
    )
    mapping.write_text(
        json.dumps(
            {
                "groups": [[0, 1], [2, 3], [4, 5], [6, 7], [8], [9]],
                "wind_buses": [3, 5, 7, 16, 21, 23],
                "zone_capacity_mw": 120.0,
            }
        ),
        encoding="utf-8",
    )
    calls = {"count": 0}

    def fake_run(*_args, **_kwargs):
        calls["count"] += 1
        return pd.DataFrame(
            [
                {
                    "outer": 3,
                    "date": "2013-01-01",
                    "method": "candidate",
                    "status": "solution_available",
                }
            ]
        )

    monkeypatch.setattr(cached.evaluation, "run_exact_cases", fake_run)
    monkeypatch.setattr(
        cached.evaluation,
        "summarize_exact_cases",
        lambda frame: {
            "cases": len(frame),
            "successful_cases": int(
                (frame["status"] == "solution_available").sum()
            ),
        },
    )
    argv = [
        "cached",
        "--calibrated",
        str(calibrated),
        "--truth",
        str(truth),
        "--mapping",
        str(mapping),
        "--outer",
        "3",
        "--method",
        "candidate",
        "--cache-dir",
        str(cache_dir),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cached.main()
    cached.main()
    assert calls["count"] == 1
    assert (cache_dir / "day_000.json").is_file()
