"""Self-contained semantic tests for the variance-preserving v3.1 source."""

from __future__ import annotations

import math

import torch

from architecture_v1.model import (
    T0StableSourceRectifiedFlow,
    T1FeatureStableSourceRectifiedFlow,
    T1SourceStableRectifiedFlow,
    T1StableShuffleRectifiedFlow,
    VariancePreservingARSource,
)
from architecture_v1.training import configure_stage, tensor_state_sha256


ORDER = (17, 2, 12, 0, 15, 18, 1, 11, 16, 20, 13, 7, 5, 4, 9, 23, 22, 10, 3, 8, 6, 21, 19, 14)


def _kwargs() -> dict[str, object]:
    return {
        "condition_dim": 4,
        "zones": 2,
        "hours": 24,
        "encoder_dim": 8,
        "encoder_depth": 1,
        "flow_dim": 8,
        "flow_depth": 1,
        "heads": 2,
        "ff_multiplier": 2,
        "dropout": 0.0,
        "atom_hidden_dim": 8,
        "stable_source_rho_max": 0.95,
    }


def _models() -> list[torch.nn.Module]:
    kwargs = _kwargs()
    torch.manual_seed(901)
    reference = T0StableSourceRectifiedFlow(**kwargs)
    state = reference.state_dict()
    models = [
        reference,
        T1FeatureStableSourceRectifiedFlow(**kwargs),
        T1SourceStableRectifiedFlow(**kwargs),
        T1StableShuffleRectifiedFlow(
            temporal_hour_order=ORDER, shuffle_target="feature", **kwargs
        ),
        T1StableShuffleRectifiedFlow(
            temporal_hour_order=ORDER, shuffle_target="source", **kwargs
        ),
    ]
    for model in models:
        model.load_state_dict(state, strict=True)
    return models


def test_common_topology_and_initial_tensor_bank() -> None:
    models = _models()
    keys = [tuple(model.state_dict()) for model in models]
    shapes = [tuple(tuple(value.shape) for value in model.state_dict().values()) for model in models]
    hashes = [tensor_state_sha256(model.state_dict()) for model in models]
    assert len(set(keys)) == 1
    assert len(set(shapes)) == 1
    assert len(set(hashes)) == 1
    assert len({model.parameter_count() for model in models}) == 1


def test_t0_and_feature_sources_are_exact_masked_iid() -> None:
    t0, feature = _models()[:2]
    generator = torch.Generator().manual_seed(902)
    condition = torch.randn(3, 2, 24, 4, generator=generator)
    iid = torch.randn(3, 2, 24, generator=generator)
    active = torch.rand(3, 2, 24, generator=generator) > 0.2
    expected = torch.where(active, iid, torch.zeros_like(iid))
    for model in (t0, feature):
        encoded = model.encode_condition(condition)
        actual = model.prepare_source_noise(iid, encoded, active)
        assert torch.equal(actual, expected)


def test_ar_source_preserves_unit_marginal_variance() -> None:
    source = VariancePreservingARSource(4, rho_max=0.95)
    target_rho = 0.7
    with torch.no_grad():
        source.rho_head.bias.fill_(math.atanh(target_rho / source.rho_max))
    generator = torch.Generator().manual_seed(903)
    iid = torch.randn(20000, 1, 24, generator=generator)
    encoded = torch.zeros(20000, 1, 24, 4)
    active = torch.ones_like(iid, dtype=torch.bool)
    result = source(iid, encoded, active, use_correlation=True)
    assert abs(float(result.detach().mean())) < 0.02
    variance_by_hour = result.detach().var(dim=0, unbiased=True)[0]
    assert float(variance_by_hour.min()) > 0.96
    assert float(variance_by_hour.max()) < 1.04
    detached = result.detach()
    correlation = torch.corrcoef(torch.stack((detached[:, 0, :-1].reshape(-1), detached[:, 0, 1:].reshape(-1))))[0, 1]
    assert abs(float(correlation) - target_rho) < 0.02
    assert float(source.correlation(encoded[:1]).detach().abs().max()) <= 0.95


def test_atom_breaks_ar_adjacency_and_has_zero_latent() -> None:
    source = VariancePreservingARSource(4, rho_max=0.95)
    with torch.no_grad():
        source.rho_head.bias.fill_(math.atanh(0.8 / source.rho_max))
    iid_a = torch.randn(2, 1, 24, generator=torch.Generator().manual_seed(904))
    iid_b = iid_a.clone()
    iid_b[:, :, :8] += 100.0
    active = torch.ones_like(iid_a, dtype=torch.bool)
    active[:, :, 8] = False
    encoded = torch.zeros(2, 1, 24, 4)
    out_a = source(iid_a, encoded, active, use_correlation=True)
    out_b = source(iid_b, encoded, active, use_correlation=True)
    assert torch.equal(out_a[:, :, 8], torch.zeros_like(out_a[:, :, 8]))
    assert torch.equal(out_a[:, :, 9:], out_b[:, :, 9:])


def test_shuffle_changes_source_adjacency_and_restores_slots() -> None:
    chronological = T1SourceStableRectifiedFlow(**_kwargs())
    shuffled = T1StableShuffleRectifiedFlow(
        temporal_hour_order=ORDER, shuffle_target="source", **_kwargs()
    )
    shuffled.load_state_dict(chronological.state_dict(), strict=True)
    with torch.no_grad():
        chronological.stable_source.rho_head.bias.fill_(math.atanh(0.6 / 0.95))
        shuffled.stable_source.rho_head.bias.copy_(chronological.stable_source.rho_head.bias)
    condition = torch.randn(2, 2, 24, 4, generator=torch.Generator().manual_seed(905))
    iid = torch.randn(2, 2, 24, generator=torch.Generator().manual_seed(906))
    active = torch.ones_like(iid, dtype=torch.bool)
    first = chronological.prepare_source_noise(iid, chronological.encode_condition(condition), active)
    second = shuffled.prepare_source_noise(iid, shuffled.encode_condition(condition), active)
    assert first.shape == second.shape == iid.shape
    assert not torch.equal(first, second)


def test_flow_optimizer_owns_stable_source_parameters() -> None:
    model = T1SourceStableRectifiedFlow(**_kwargs())
    parameters = configure_stage(model, "flow")
    selected = {id(parameter) for parameter in parameters}
    assert all(id(parameter) in selected for parameter in model.stable_source.parameters())
    assert all(parameter.requires_grad for parameter in model.stable_source.parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.atom.parameters())


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")


if __name__ == "__main__":
    main()
