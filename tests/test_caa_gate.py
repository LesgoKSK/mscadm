import numpy as np

from caa_rahc.features import build_features
from caa_rahc.gate import FittedTailGate, fit_tail_gate


def _toy(seed: int = 7):
    rng = np.random.default_rng(seed)
    n, m = 12, 21
    center = rng.uniform(0.2, 0.8, size=(n, 24))
    scenarios = np.clip(center[:, None, :] + rng.normal(0, 0.04, size=(n, m, 24)), 0, 1)
    observations = np.clip(center + rng.normal(0, 0.10, size=(n, 24)), 0, 1)
    features = build_features(scenarios, scenarios)
    zone = np.arange(n) % 10 + 1
    return scenarios, observations, features, zone


def test_gate_fit_predict_and_roundtrip(tmp_path):
    scenarios, observations, features, zone = _toy()
    gate = fit_tail_gate(
        [scenarios], observations, [features], zone, steps=3, batch_cells=128, random_seed=4
    )
    predicted = gate.predict(features, zone, strength=0.5)
    assert predicted.shape == (len(zone), 24, 2)
    assert np.isfinite(predicted).all()
    assert np.all((predicted >= 0.0) & (predicted <= 0.5))
    path = gate.save(tmp_path / "gate.pt", {"fold": 2})
    loaded, metadata = FittedTailGate.load(path)
    np.testing.assert_allclose(loaded.predict(features, zone, strength=0.5), predicted)
    assert metadata == {"fold": 2}


def test_zero_strength_is_exact_zero_gate():
    scenarios, observations, features, zone = _toy()
    gate = fit_tail_gate([scenarios], observations, [features], zone, steps=1, batch_cells=64)
    assert np.count_nonzero(gate.predict(features, zone, strength=0.0)) == 0


def test_gate_rejects_bad_shapes():
    scenarios, observations, features, zone = _toy()
    try:
        fit_tail_gate([scenarios], observations[:, :-1], [features], zone, steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned truth was accepted")
