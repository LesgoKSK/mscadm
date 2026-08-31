from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import repro_scripts.ps_dfsc_publication_runtime_v2 as runtime
from ps_dfsc.types import CalibratedDistribution


class _Mapping:
    groups = ((0, 1), (2, 3), (4, 5), (6, 7), (8,), (9,))
    zone_capacity_mw = 120.0

    def transform(self, values):
        array = np.asarray(values)
        return np.stack(
            [
                array[..., group, :].sum(axis=-2) * self.zone_capacity_mw
                for group in self.groups
            ],
            axis=-2,
        )


def test_calibration_uses_pretransport_assignments_and_preserves_day(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(4)
    scenarios = rng.uniform(size=(1, 4, 10, 24)).astype(np.float32)
    source = tmp_path / "source.npz"
    day = np.array(["2020-05-01"], dtype="datetime64[D]")
    np.savez_compressed(source, scenarios=scenarios, day=day)
    captured = []

    monkeypatch.setattr(
        runtime.pipeline, "_mapping_from_args", lambda args: _Mapping()
    )
    monkeypatch.setattr(
        runtime.pipeline, "_load_model", lambda checkpoint, device: object()
    )

    def fake_calibrate(model, values, mapping, **kwargs):
        captured.append(np.asarray(kwargs["assignments"]).copy())
        return CalibratedDistribution(
            full_scenarios=values,
            probabilities=np.full(4, 0.25),
            suc_scenarios=np.zeros((2, 6, 24)),
            suc_probabilities=np.full(2, 0.5),
            ess=4.0,
            entropy=np.log(4.0),
            transport_cost=0.0,
        )

    monkeypatch.setattr(runtime, "calibrate", fake_calibrate)
    output = tmp_path / "calibrated.npz"
    runtime.calibrate_archive(
        SimpleNamespace(
            input=str(source),
            scenario_key="scenarios",
            checkpoint="candidate.pt",
            device="cpu",
            clusters=2,
            ess_floor=1.0,
            entropy_floor=0.0,
            transport_budget=1.0,
            output=str(output),
        )
    )
    archive = np.load(output, allow_pickle=False)
    assert np.array_equal(archive["days"], day)
    assert str(archive["assignment_source"]) == "untransported_base_scenarios"
    assert np.array_equal(archive["assignments"][0], captured[0])
    assert set(np.unique(captured[0])) == {0, 1}
