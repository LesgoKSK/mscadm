from types import SimpleNamespace

import torch

import ps_dfsc.resumable_training_runtime as resumable


class _Model:
    def __init__(self):
        self.loaded = []

    def load_state_dict(self, state):
        self.loaded.append(state)


def _args(tmp_path):
    source = tmp_path / "development_train.npz"
    mapping = tmp_path / "mapping.json"
    source.write_bytes(b"registered development archive")
    mapping.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        output=str(tmp_path / "beta025.pt"),
        input=str(source),
        mapping=str(mapping),
        beta=0.25,
        epochs=50,
        warmup_epochs=20,
        batch_size=4,
        learning_rate=1.0e-3,
        strong_convexity=1.0e-4,
        seed=30025,
        clusters=20,
    )


def test_epoch_resume_is_atomic_and_rebuilds_completed_refresh(monkeypatch, tmp_path):
    args = _args(tmp_path)
    calls = {"refresh": 0, "resume": []}

    def fake_publication_train_command(_args):
        model = _Model()

        def refresh(_model):
            calls["refresh"] += 1
            return "commitments"

        resumable.runtime.train_calibrator(
            model,
            "base",
            commitment_refresh=refresh,
            device="cpu",
        )
        calls["models"] = calls.get("models", []) + [model]

    def fake_trainer(
        model,
        *_values,
        commitment_refresh=None,
        resume_state=None,
        epoch_callback=None,
        **_kwargs,
    ):
        calls["resume"].append(resume_state)
        if resume_state is None:
            commitment_refresh(model)
            epoch_callback(
                {
                    "next_epoch": 36,
                    "model_state": {"weight": torch.tensor([1.0])},
                }
            )
        return "history"

    monkeypatch.setattr(resumable.runtime, "train_command", fake_publication_train_command)
    monkeypatch.setattr(resumable, "train_calibrator", fake_trainer)

    resumable.train_command(args)
    resume_path = tmp_path / "beta025.epoch_resume.pt"
    temporary_path = tmp_path / "beta025.epoch_resume.tmp"
    assert resume_path.is_file()
    assert not temporary_path.exists()
    first = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert first["midpoint_refresh_completed"] is True
    assert first["state"]["next_epoch"] == 36
    assert calls["refresh"] == 1
    assert calls["resume"] == [None]

    resumable.train_command(args)
    assert calls["refresh"] == 2
    assert calls["resume"][1]["next_epoch"] == 36
    assert calls["models"][1].loaded[0]["weight"].item() == 1.0


def test_resume_metadata_mismatch_fails_closed(monkeypatch, tmp_path):
    args = _args(tmp_path)
    resume_path = tmp_path / "beta025.epoch_resume.pt"
    torch.save(
        {
            "metadata": {"schema": "wrong-run"},
            "midpoint_refresh_completed": False,
            "state": {},
        },
        resume_path,
    )

    def fake_publication_train_command(_args):
        resumable.runtime.train_calibrator(
            _Model(),
            "base",
            commitment_refresh=lambda _model: None,
            device="cpu",
        )

    monkeypatch.setattr(resumable.runtime, "train_command", fake_publication_train_command)

    try:
        resumable.train_command(args)
    except ValueError as error:
        assert "metadata does not match" in str(error)
    else:
        raise AssertionError("mismatched resume metadata was accepted")
