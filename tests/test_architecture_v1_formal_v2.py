from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_v1.atom import INTERIOR_STATE, ONE_STATE, ZERO_STATE
from architecture_v1.data import (
    ArchitectureFitDataBundle,
    build_architecture_v1_fit_data,
)
from architecture_v1.model import R0JointRectifiedFlow
from architecture_v1.protocol import ALL_ROLES, build_architecture_protocol

CONFIG = ROOT / "repro_configs" / "architecture_v1_formal_v2.json"
SMOKE_QUARANTINE = (
    ROOT / "repro_configs" / "architecture_v1_smoke_quarantine.json"
)
DATA = ROOT / "Data"
AVAILABLE_DAYS = np.arange(
    np.datetime64("2012-01-01"),
    np.datetime64("2014-01-01"),
    dtype="datetime64[D]",
)


@contextmanager
def _raises(exception: type[BaseException], pattern: str):
    try:
        yield
    except exception as error:
        assert re.search(pattern, str(error), flags=re.IGNORECASE), str(error)
    else:
        raise AssertionError(f"expected {exception.__name__}: {pattern}")


def _tiny_formal_r0(*, fixed_one_probability: float) -> R0JointRectifiedFlow:
    torch.manual_seed(7102)
    return R0JointRectifiedFlow(
        encoder_dim=4,
        encoder_depth=0,
        flow_dim=4,
        flow_depth=0,
        heads=1,
        ff_multiplier=1,
        dropout=0.0,
        atom_hidden_dim=4,
        atom_initial_probabilities=(0.08, 0.919, 0.001),
        atom_fixed_one_probability=fixed_one_probability,
        atom_location_auxiliary_weight=0.25,
        atom_location_mean=0.0,
        atom_location_std=1.0,
        atom_location_smooth_l1_beta=1.0,
        logit_epsilon=1e-4,
    )


def test_formal_v2_partition_has_314_quarantine_and_267_50_50_50() -> None:
    protocol = build_architecture_protocol(
        CONFIG,
        available_days=AVAILABLE_DAYS,
        smoke=False,
    )
    expected = {
        "train": 267,
        "validation": 50,
        "calibration": 50,
        "selection": 50,
        "r_seen": 314,
        "final": 0,
    }
    assert {
        role: len(protocol.role_dates(role, full=True)) for role in ALL_ROLES
    } == expected

    local = np.concatenate(
        [
            protocol.role_dates(role, full=True)
            for role in ALL_ROLES
            if role != "final"
        ]
    )
    assert len(local) == len(AVAILABLE_DAYS) == 731
    assert len(np.unique(local)) == 731
    for role in ("train", "validation", "calibration", "selection"):
        assert not np.intersect1d(
            protocol.role_dates("r_seen", full=True),
            protocol.role_dates(role, full=True),
        ).size


def test_all_14_v1_smoke_dates_are_quarantined_from_every_formal_role() -> None:
    registry = json.loads(SMOKE_QUARANTINE.read_text(encoding="utf-8"))
    assert registry["schema"] == "architecture_v1_smoke_quarantine_v1"
    assert {role: len(values) for role, values in registry["dates_by_role"].items()} == {
        "train": 8,
        "validation": 2,
        "calibration": 2,
        "selection": 2,
    }
    smoke_dates = np.asarray(
        [
            value
            for values in registry["dates_by_role"].values()
            for value in values
        ],
        dtype="datetime64[D]",
    )
    assert len(smoke_dates) == registry["all_dates_count"] == 14
    assert len(np.unique(smoke_dates)) == 14

    protocol = build_architecture_protocol(
        CONFIG,
        available_days=AVAILABLE_DAYS,
        smoke=False,
    )
    r_seen = protocol.role_dates("r_seen", full=True)
    np.testing.assert_array_equal(np.intersect1d(smoke_dates, r_seen), np.sort(smoke_dates))
    for role in ("train", "validation", "calibration", "selection", "final"):
        assert not np.intersect1d(
            smoke_dates, protocol.role_dates(role, full=True)
        ).size


def test_formal_v2_protocol_refuses_every_smoke_subset_view() -> None:
    with _raises(RuntimeError, r"formal-v2.*forbids.*smoke|forbids smoke"):
        build_architecture_protocol(
            CONFIG,
            available_days=AVAILABLE_DAYS,
            smoke=True,
        )


def test_formal_fit_bundle_materializes_only_train_and_validation() -> None:
    bundle = build_architecture_v1_fit_data(DATA, config_path=CONFIG)
    assert isinstance(bundle, ArchitectureFitDataBundle)
    assert bundle.materialized_roles == ("train", "validation")
    assert len(bundle.train) == 267
    assert len(bundle.validation) == 50
    assert bundle.train.role == "train"
    assert bundle.validation.role == "validation"
    np.testing.assert_array_equal(
        bundle.train.day,
        bundle.protocol.role_dates("train", full=True),
    )
    np.testing.assert_array_equal(
        bundle.validation.day,
        bundle.protocol.role_dates("validation", full=True),
    )

    access = bundle.manifest["formal_fit_target_access"]
    assert access["materialized_roles"] == ["train", "validation"]
    assert access["materialized_date_count"] == 317
    assert access["forbidden_roles"] == [
        "calibration",
        "selection",
        "r_seen",
        "final",
    ]
    assert access["forbidden_target_arrays_materialized"] is False
    assert set(bundle.manifest["data_audit"]["split_array_sha256"]) == {
        "train",
        "validation",
    }

    assert not hasattr(bundle, "selection")
    with _raises(AttributeError, "selection"):
        getattr(bundle, "selection")
    with _raises(RuntimeError, r"does not materialize.*selection|selection.*sealed"):
        bundle.role("selection")
    for role in ("calibration", "r_seen", "final"):
        with _raises(RuntimeError, r"does not materialize|sealed"):
            bundle.role(role)


def test_fixed_upper_one_probability_is_context_and_training_invariant() -> None:
    fixed = 0.0125
    model = _tiny_formal_r0(fixed_one_probability=fixed)
    first_condition = torch.randn(2, 10, 24, 20)
    second_condition = torch.randn(2, 10, 24, 20) * 11.0 + 7.0

    first = model.atom_statistics(first_condition).one_probability
    second = model.atom_statistics(second_condition).one_probability
    torch.testing.assert_close(first, torch.full_like(first, fixed), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(second, torch.full_like(second, fixed), rtol=1e-6, atol=1e-7)

    target = torch.full((2, 10, 24), 0.4, dtype=torch.float32)
    states = torch.full_like(target, INTERIOR_STATE, dtype=torch.long)
    states[:, 0, 0] = ZERO_STATE
    states[:, 0, 1] = ONE_STATE
    observed = torch.ones_like(target, dtype=torch.bool)
    optimizer = torch.optim.SGD(model.atom.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    losses = model.atom_loss(
        first_condition,
        states=states,
        observed_mask=observed,
        location_target=target,
    )
    losses["loss"].backward()
    optimizer.step()

    after_update = model.atom_statistics(first_condition).one_probability
    torch.testing.assert_close(
        after_update,
        torch.full_like(after_update, fixed),
        rtol=1e-6,
        atol=1e-7,
    )


def test_location_auxiliary_uses_only_observed_interior_and_reaches_encoder() -> None:
    model = _tiny_formal_r0(fixed_one_probability=0.001)
    # The production head starts at the constant-location null.  Give its final
    # projection a deterministic nonzero value so this unit test can verify the
    # complete auxiliary -> encoder gradient path in a single backward pass.
    with torch.no_grad():
        model.atom.head.location[-1].weight.copy_(
            torch.tensor([[0.05, 0.10, 0.15, 0.20]])
        )

    condition = torch.randn(1, 10, 24, 20, requires_grad=True)
    target = torch.full((1, 10, 24), 0.4, dtype=torch.float32)
    states = torch.full_like(target, INTERIOR_STATE, dtype=torch.long)
    observed = torch.ones_like(target, dtype=torch.bool)
    states[0, 0, 0] = ZERO_STATE
    target[0, 0, 0] = 0.0
    states[0, 0, 1] = ONE_STATE
    target[0, 0, 1] = 1.0
    observed[0, 0, 2] = False
    target[0, 0, 2] = 0.2

    baseline = model.atom_loss(
        condition,
        states=states,
        observed_mask=observed,
        location_target=target,
    )
    assert int(baseline["interior_location_count"].item()) == 237

    excluded_changed = target.clone()
    excluded_changed[0, 0, 0] = 0.73
    excluded_changed[0, 0, 1] = 0.27
    excluded_changed[0, 0, 2] = 0.99
    excluded = model.atom_loss(
        condition,
        states=states,
        observed_mask=observed,
        location_target=excluded_changed,
    )
    torch.testing.assert_close(
        baseline["interior_location_smooth_l1"],
        excluded["interior_location_smooth_l1"],
        rtol=0.0,
        atol=0.0,
    )

    included_changed = target.clone()
    included_changed[0, 0, 3] = 0.8
    included = model.atom_loss(
        condition,
        states=states,
        observed_mask=observed,
        location_target=included_changed,
    )
    assert not torch.equal(
        baseline["interior_location_smooth_l1"],
        included["interior_location_smooth_l1"],
    )

    model.zero_grad(set_to_none=True)
    baseline["interior_location_smooth_l1"].backward()
    location_gradient = model.atom.head.location[-1].weight.grad
    assert location_gradient is not None
    assert bool(torch.isfinite(location_gradient).all())
    assert float(location_gradient.norm()) > 0.0
    encoder_gradients = [
        parameter.grad
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_gradients
    assert all(bool(torch.isfinite(value).all()) for value in encoder_gradients)
    assert sum(float(value.square().sum()) for value in encoder_gradients) > 0.0

    assert condition.grad is not None
    # With encoder_depth=0 every cell is pointwise, so excluded cells must not
    # receive a gradient from the isolated location loss.
    assert torch.equal(condition.grad[0, 0, 0], torch.zeros(20))
    assert torch.equal(condition.grad[0, 0, 1], torch.zeros(20))
    assert torch.equal(condition.grad[0, 0, 2], torch.zeros(20))
    assert bool((condition.grad[0, 0, 3].abs() > 0.0).any())


def main() -> None:
    tests = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and inspect.isfunction(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
