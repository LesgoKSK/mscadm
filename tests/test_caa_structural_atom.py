import numpy as np

from caa_rahc.structural_atom import (
    StructuralZeroModel,
    blend_zero_probability,
    zero_model_features,
)


def _toy():
    rng = np.random.default_rng(3)
    cases = 20
    condition = rng.normal(size=(cases, 24, 20)).astype(np.float32)
    condition[..., 10:] = 0.0
    for case in range(cases):
        condition[case, :, 10 + case % 10] = 1.0
    target = rng.uniform(0.02, 0.8, size=(cases, 24)).astype(np.float32)
    target[:, :3] = 0.0
    return condition, target


def test_structural_zero_fit_predict_and_roundtrip(tmp_path):
    condition, target = _toy()
    assert zero_model_features(condition).shape == (condition.shape[0] * 24, 32)
    model = StructuralZeroModel.fit(condition, target, regularization_c=0.1)
    probability = model.predict_zero(condition)
    assert probability.shape == target.shape
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert np.isfinite(list(model.analytic_scores(condition, target).values())).all()
    path = model.save(tmp_path / "zero.pt", {"outer": 1})
    loaded, metadata = StructuralZeroModel.load(path)
    np.testing.assert_allclose(loaded.predict_zero(condition), probability)
    assert metadata == {"outer": 1}


def test_blend_endpoints():
    condition, target = _toy()
    structural = StructuralZeroModel.fit(condition, target).predict_zero(condition)
    baseline = np.linspace(0.0, 1.0, 21)[None, :, None]
    baseline = np.broadcast_to(baseline, (len(target), 21, 24)).copy()
    weak = blend_zero_probability(baseline, structural, 0.0)
    strong = blend_zero_probability(baseline, structural, 1.0)
    np.testing.assert_allclose(strong, structural, rtol=1e-6, atol=1e-7)
    assert np.isfinite(weak).all()
