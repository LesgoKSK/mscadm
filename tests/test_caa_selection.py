import json

import numpy as np

from caa_rahc.selection import CandidateRecord, METRIC_KEYS, select_candidates


def _daily_metrics(
    *,
    seeds: int = 3,
    days: int = 20,
    multiplier: float | np.ndarray = 1.0,
    width_change: float = 0.0,
) -> dict[str, np.ndarray]:
    """Deterministic positive daily scores with mild paired date variation."""

    date = np.linspace(-0.04, 0.04, days)[None, :]
    seed = np.linspace(-0.01, 0.01, seeds)[:, None]
    base = 1.0 + date + seed
    factor = np.asarray(multiplier, dtype=np.float64)
    if factor.ndim == 0:
        factor = np.full((seeds, 1), float(factor))
    elif factor.shape == (seeds,):
        factor = factor[:, None]
    else:
        raise ValueError("multiplier must be scalar or one value per seed")
    result = {
        "CRPS": base * factor,
        "MAE": 1.10 * base * factor,
        "VS": 1.20 * base * factor,
        "ramp_CRPS": 0.80 * base * factor,
        "winkler_90": 1.40 * base * factor,
        "width_90": 0.40 + 0.05 * date + 0.01 * seed + width_change,
    }
    assert set(result) == set(METRIC_KEYS)
    return result


def _record(
    name: str,
    *,
    multiplier: float | np.ndarray = 1.0,
    width_change: float = 0.0,
    ace: object = 0.05,
    policy: str = "constrained",
    gate: float = 0.5,
    cap: float = 0.05,
    shrinkage: float = 0.1,
) -> CandidateRecord:
    return CandidateRecord(
        name=name,
        metrics=_daily_metrics(multiplier=multiplier, width_change=width_change),
        conditional_ace90=ace,
        gate_maximum=gate,
        width_cap=cap,
        shrinkage=shrinkage,
        selection_policy=policy,
        metadata={"synthetic": True, "array": np.asarray([1, 2])},
    )


def _candidate(result: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in result["candidates"] if item["name"] == name)


def test_constrained_A4_and_unconstrained_A5_are_separate() -> None:
    baseline = _record("A0", ace=np.asarray([0.08, 0.08, 0.08]), policy="baseline")
    feasible = _record(
        "A4_safe",
        multiplier=0.998,
        width_change=0.02,
        ace=np.asarray([[0.040, 0.050], [0.045, 0.055], [0.035, 0.045]]),
        policy="constrained",
        gate=0.5,
        cap=0.05,
        shrinkage=0.2,
    )
    unsafe_ablation = _record(
        "A5_unconstrained",
        multiplier=1.03,
        width_change=0.09,
        ace=np.asarray([0.01, 0.012, 0.011]),
        policy="unconstrained",
        gate=1.0,
        cap=0.10,
        shrinkage=0.0,
    )
    result = select_candidates(
        [unsafe_ablation, baseline, feasible],
        day_ids=np.arange("2020-01-01", "2020-01-21", dtype="datetime64[D]"),
        bootstrap_replicates=500,
        bootstrap_seed=17,
    )

    assert result["constrained_A4"]["selected"] == "A4_safe"
    assert result["constrained_A4"]["fallback"] is False
    assert result["unconstrained_A5"]["selected"] == "A5_unconstrained"
    assert _candidate(result, "A5_unconstrained")["eligible_for_constrained_A4"] is False
    assert _candidate(result, "A4_safe")["eligible_for_unconstrained_A5"] is False
    assert _candidate(result, "A4_safe")["conditional_ace90"][
        "per_model_replicate_mean"
    ] == [0.045, 0.05, 0.04]
    json.dumps(result, sort_keys=True)


def test_rejected_candidates_produce_exact_A0_fallback() -> None:
    baseline = _record("A0", ace=0.07, policy="baseline")
    crps_bad = _record(
        "A4_crps_bad",
        multiplier=1.02,
        width_change=0.0,
        ace=0.01,
        policy="constrained",
    )
    width_bad = _record(
        "A4_width_bad",
        multiplier=1.0,
        width_change=0.06,
        ace=0.02,
        policy="constrained",
    )
    result = select_candidates(
        [crps_bad, width_bad, baseline],
        bootstrap_replicates=300,
        bootstrap_seed=3,
    )

    assert result["constrained_A4"] == {
        "selected": "A0",
        "fallback": True,
        "feasible_candidates": ["A0"],
        "selection_key": result["constrained_A4"]["selection_key"],
    }
    baseline_audit = _candidate(result, "A0")
    assert baseline_audit["all_constraints_passed"] is True
    for summary in baseline_audit["bootstrap_differences_vs_A0"].values():
        assert summary["q05"] == summary["median"] == summary["upper"] == 0.0
    for constraint in baseline_audit["constraints"].values():
        assert constraint["passed"] is True
        if isinstance(constraint["value"], list):
            assert constraint["value"] == [0.0, 0.0, 0.0]
        else:
            assert constraint["value"] == 0.0
    assert _candidate(result, "A4_crps_bad")["constraints"][
        "CRPS_bootstrap_upper_relative"
    ]["passed"] is False
    assert _candidate(result, "A4_width_bad")["constraints"][
        "width_90_point_absolute_change"
    ]["passed"] is False


def test_every_model_replicate_must_pass_CRPS_point_margin() -> None:
    baseline = _record("A0", ace=0.08, policy="baseline")
    # The cross-seed mean degradation is favorable, but seed 0 is 0.6% worse.
    mixed = _record(
        "A4_one_seed_bad",
        multiplier=np.asarray([1.006, 0.996, 0.996]),
        width_change=0.0,
        ace=0.02,
        policy="constrained",
    )
    result = select_candidates(
        [baseline, mixed], bootstrap_replicates=250, bootstrap_seed=11
    )

    audit = _candidate(result, "A4_one_seed_bad")
    replicate_constraint = audit["constraints"][
        "CRPS_each_model_replicate_point_relative"
    ]
    assert replicate_constraint["value"][0] > 0.005
    assert audit["constraints"]["CRPS_bootstrap_upper_relative"]["passed"] is True
    assert replicate_constraint["passed"] is False
    assert audit["all_constraints_passed"] is False
    assert result["constrained_A4"]["selected"] == "A0"
    assert result["constrained_A4"]["fallback"] is True


def test_selection_is_deterministic_and_tie_break_is_stable() -> None:
    baseline = _record("A0", ace=0.09, policy="baseline")
    higher_gate = _record(
        "A4_higher_gate",
        multiplier=0.999,
        width_change=0.01,
        ace=0.03,
        gate=0.75,
        cap=0.05,
        shrinkage=0.3,
    )
    lower_gate = _record(
        "A4_lower_gate",
        multiplier=0.999,
        width_change=0.01,
        ace=0.03,
        gate=0.25,
        cap=0.05,
        shrinkage=0.1,
    )
    first = select_candidates(
        [higher_gate, baseline, lower_gate],
        bootstrap_replicates=400,
        bootstrap_seed=29,
    )
    second = select_candidates(
        [lower_gate, higher_gate, baseline],
        bootstrap_replicates=400,
        bootstrap_seed=29,
    )

    assert first == second
    assert first["constrained_A4"]["selected"] == "A4_lower_gate"
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

