"""Fast decision-contract tests for the family-v1 validation evaluator."""

from __future__ import annotations

import json

import numpy as np

from repro_scripts.run_architecture_v1_family_v1_evaluation import (
    DEFAULT_CONFIG,
    _directional_family_gate,
    _equivalence_gate,
    _seed_stability,
    _validate_training,
)


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _daily(level: float) -> dict[str, np.ndarray]:
    return {
        "level_CRPS": np.full(50, level),
        "ramp_CRPS": np.full(50, level / 2.0),
        "normalized_joint_ES": np.full(50, level * 1.2),
        "lagged_increment_variogram_score": np.full(50, level * 0.8),
        "coverage90": np.full(50, 0.9),
        "width90": np.full(50, 0.5),
    }


def test_six_training_artifacts_are_complete_and_hash_verified() -> None:
    result, completions = _validate_training()
    assert result["payload"]["status"] == "SIX_OF_SIX_TRAINING_COMPLETE"
    assert len(completions) == 6


def test_lower_proper_scores_have_the_registered_superiority_direction() -> None:
    better = _daily(0.08)
    worse = _daily(0.10)
    gate = _directional_family_gate(
        "better", "worse", better, worse, config=_config(), seed=81_000
    )
    assert gate["noninferior_on_all"] is True
    assert gate["superior_on_at_least_one_proper_score"] is True
    assert gate["endpoint"]["level_CRPS"]["superiority"]["estimate"] > 0.0


def test_identical_daily_metrics_pass_equivalence() -> None:
    values = _daily(0.10)
    gate = _equivalence_gate(values, values, config=_config())
    assert gate["passed"] is True
    assert all(gate["checks"].values())


def test_registered_seed_stability_rejects_large_level_spread() -> None:
    values = {
        3: {"level_CRPS": 0.10, "ramp_CRPS": 0.05, "normalized_joint_ES": 0.15, "coverage90": 0.90},
        4: {"level_CRPS": 0.101, "ramp_CRPS": 0.05, "normalized_joint_ES": 0.15, "coverage90": 0.90},
        5: {"level_CRPS": 0.104, "ramp_CRPS": 0.05, "normalized_joint_ES": 0.15, "coverage90": 0.90},
    }
    gate = _seed_stability(values, _config()["three_seed_stability"])
    assert gate["checks"]["level_CRPS"] is False
    assert gate["passed"] is False


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")


if __name__ == "__main__":
    main()
