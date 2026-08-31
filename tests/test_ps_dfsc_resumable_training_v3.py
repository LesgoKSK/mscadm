from types import SimpleNamespace

import torch

import ps_dfsc.resumable_training_runtime_v3 as resumable


def _args(tmp_path):
    source = tmp_path / "train.npz"
    mapping = tmp_path / "mapping.json"
    source.write_bytes(b"train")
    mapping.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        output=str(tmp_path / "beta000.pt"),
        input=str(source),
        mapping=str(mapping),
        beta=0.0,
        epochs=50,
        warmup_epochs=20,
        batch_size=4,
        learning_rate=1.0e-3,
        strong_convexity=1.0e-4,
        seed=30000,
        clusters=20,
    )


def test_cuda_target_resume_keeps_cpu_rng_state(monkeypatch, tmp_path):
    args = _args(tmp_path)
    observed = []

    def fake_publication_command(_args):
        resumable.runtime.train_calibrator(
            object(),
            "base",
            commitment_refresh=None,
            device="cuda",
        )

    def fake_train(
        _model,
        *_values,
        resume_state=None,
        epoch_callback=None,
        **_kwargs,
    ):
        observed.append(resume_state)
        if resume_state is None:
            epoch_callback(
                {
                    "next_epoch": 20,
                    "model_state": {"weight": torch.ones(1)},
                    "torch_rng_state": torch.get_rng_state(),
                }
            )
        else:
            assert resume_state["torch_rng_state"].device.type == "cpu"
        return "history"

    monkeypatch.setattr(
        resumable.runtime, "train_command", fake_publication_command
    )
    monkeypatch.setattr(resumable, "train_calibrator", fake_train)
    resumable.train_command(args)
    resumable.train_command(args)
    assert observed[0] is None
    assert observed[1]["next_epoch"] == 20
