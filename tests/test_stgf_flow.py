from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mm_jdwind.data import JointSplitData
from stgf_flow.data import (
    build_stgf_confirmation_gefcom2014,
    load_stgf_confirmation_splits,
)
from stgf_flow.graph import (
    decode_field,
    encode_field,
    fit_spectral_artifacts,
)
from stgf_flow.model import STGFFlow
from stgf_flow.sampling import sample_stgf


ROOT = Path(__file__).resolve().parents[1]


def synthetic_split() -> JointSplitData:
    generator = np.random.default_rng(4)
    target = generator.uniform(0.01, 0.99, size=(12, 10, 24)).astype(
        np.float32
    )
    shared = generator.normal(size=(12, 1, 24)).astype(np.float32)
    target = np.clip(target + 0.05 * shared, 0.01, 0.99)
    return JointSplitData(
        condition=generator.normal(size=(12, 10, 24, 20)).astype(np.float32),
        target=target,
        day=np.arange(
            np.datetime64("2012-01-01"),
            np.datetime64("2012-01-13"),
            dtype="datetime64[D]",
        ),
        zones=np.arange(1, 11, dtype=np.int64),
    )


def test_all_spectral_transforms_roundtrip() -> None:
    split = synthetic_split()
    values = torch.randn(3, 10, 24)
    for mode in (
        "time_domain",
        "graph_only",
        "time_frequency",
        "stgf",
    ):
        artifacts = fit_spectral_artifacts(
            split, transform_mode=mode, neighbors=2
        )
        encoded = encode_field(
            values,
            torch.from_numpy(artifacts.graph_basis),
            torch.from_numpy(artifacts.temporal_basis),
        )
        decoded = decode_field(
            encoded,
            torch.from_numpy(artifacts.graph_basis),
            torch.from_numpy(artifacts.temporal_basis),
        )
        assert torch.allclose(values, decoded, atol=1e-5)


def test_model_loss_and_sampling_are_finite_and_bounded() -> None:
    split = synthetic_split()
    artifacts = fit_spectral_artifacts(
        split, transform_mode="stgf", neighbors=2
    )
    model = STGFFlow(
        artifacts,
        center_dim=32,
        center_depth=1,
        flow_dim=32,
        flow_depth=1,
        heads=4,
        ff_multiplier=2,
    )
    condition = torch.from_numpy(split.condition[:2])
    target = torch.from_numpy(split.target[:2])
    assert torch.isfinite(model.center_loss(condition, target)["loss"])
    assert torch.isfinite(model.flow_loss(condition, target)["loss"])
    scenarios = sample_stgf(
        model,
        condition,
        members=3,
        steps=2,
        member_chunk=2,
        seed=9,
    )
    assert scenarios.shape == (2, 3, 10, 24)
    assert torch.isfinite(scenarios).all()
    assert float(scenarios.min()) >= 0.0
    assert float(scenarios.max()) <= 1.0


def test_confirmation_registry_excludes_prior_tests() -> None:
    registry = load_stgf_confirmation_splits(
        ROOT / "repro_configs" / "stgf_confirmation_splits.json"
    )
    blocks = registry["_blocks"]
    assert len(np.unique(np.concatenate(tuple(blocks.values())))) == 150
    assert registry["calendar_day_counts"]["train"] == 481


def test_confirmation_builder_roles_and_hash() -> None:
    bundle = build_stgf_confirmation_gefcom2014(
        ROOT / "Data",
        outer=1,
        split_path=ROOT
        / "repro_configs"
        / "stgf_confirmation_splits.json",
    )
    assert (len(bundle.train), len(bundle.validation)) == (481, 50)
    assert (len(bundle.calibration), len(bundle.test)) == (50, 50)
    assert len(bundle.protocol["protocol_sha256"]) == 64
    train_days = np.unique(bundle.train.day)
    registry = json.loads(
        (
            ROOT / "repro_configs" / "stgf_confirmation_splits.json"
        ).read_text(encoding="utf-8")
    )
    all_test = np.asarray(
        [
            date
            for dates in registry["outer_test_dates"].values()
            for date in dates
        ],
        dtype="datetime64[D]",
    )
    assert np.intersect1d(train_days, all_test).size == 0
