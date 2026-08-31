import numpy as np

from caa_rahc.aggregate import aggregate_outer_records
from caa_rahc.selection import CandidateRecord, METRIC_KEYS


def _record(name, offset):
    metrics = {key: np.full((3, 5), 1.0 + offset) for key in METRIC_KEYS}
    return CandidateRecord(
        name=name,
        metrics=metrics,
        conditional_ace90=np.full(3, 0.1 + offset),
        metadata={"offset": offset},
    )


def test_aggregate_stacks_model_replicates_not_dates():
    result = aggregate_outer_records({1: [_record("A0", 0)], 2: [_record("A0", 0.1)]})
    assert len(result) == 1
    assert result[0].metrics["CRPS"].shape == (6, 5)
    assert np.asarray(result[0].conditional_ace90).shape == (6,)


def test_aggregate_rejects_catalog_drift():
    try:
        aggregate_outer_records({1: [_record("A0", 0)], 2: [_record("A1", 0)]})
    except ValueError as error:
        assert "catalog differs" in str(error)
    else:
        raise AssertionError("catalog drift was accepted")
