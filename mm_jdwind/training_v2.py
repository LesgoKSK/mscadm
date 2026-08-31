from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .training import MMTrainer, _atomic_torch_save, _infinite


class StableMMTrainer(MMTrainer):
    """Finite-guarded trainer with validation early stopping.

    The proper-score branch is intentionally disabled in the locked v2
    development protocol after the v1 pilot produced non-finite parameters.
    """

    def fit_stage(self, stage: str) -> Path:
        if stage not in self.STAGES:
            raise ValueError(stage)
        training = self.config["training"]
        steps = int(training[f"{stage}_steps"])
        if steps == 0:
            self.history[stage].append(
                {"step": 0, "status": "disabled_by_protocol"}
            )
            return self.output_dir / f"{stage}_skipped.pt"
        self._set_stage(stage)
        parameters = list(self._parameters(stage))
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(training[f"{stage}_learning_rate"]),
            weight_decay=float(training.get(f"{stage}_weight_decay", 0.0)),
        )
        loader = _infinite(
            self._loader(self.data.train, shuffle=True, seed_offset=100 + len(stage))
        )
        log_every = int(training.get("log_every", 100))
        validate_every = int(training.get("validate_every", 250))
        patience = int(training.get("early_stopping_patience_validations", 8))
        minimum_delta = float(training.get("early_stopping_minimum_delta", 0.0))
        best = float("inf")
        best_step = 0
        validations_without_improvement = 0
        started = time.perf_counter()
        rolling: dict[str, float] = {}
        rolling_count = 0
        best_path = self.output_dir / f"{stage}_best.pt"
        latest_path = self.output_dir / f"{stage}_latest.pt"
        stopped_step = steps
        for step in range(1, steps + 1):
            condition, observation, state = self._move(next(loader))
            optimizer.zero_grad(set_to_none=True)
            losses = self._stage_loss(stage, condition, observation, state)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(
                    f"non-finite {stage} loss at step {step}; checkpoint rejected"
                )
            losses["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(training.get("gradient_clip", 1.0))
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite {stage} gradient at step {step}; checkpoint rejected"
                )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise FloatingPointError(
                    f"non-finite {stage} parameter at step {step}; checkpoint rejected"
                )
            rolling_count += 1
            for key, value in losses.items():
                rolling[key] = rolling.get(key, 0.0) + float(value.detach())
            if step % log_every == 0 or step == steps:
                record: dict[str, Any] = {
                    "step": step,
                    "seconds": time.perf_counter() - started,
                    **{key: value / rolling_count for key, value in rolling.items()},
                }
                self.history[stage].append(record)
                print(json.dumps({"stage": stage, **record}), flush=True)
                rolling, rolling_count = {}, 0
            if step % validate_every == 0 or step == steps:
                validation = self._validate(stage)
                score = float(validation["val_loss"])
                if not torch.isfinite(torch.tensor(score)):
                    raise FloatingPointError(
                        f"non-finite {stage} validation at step {step}"
                    )
                payload = self._payload(stage=stage, step=step)
                payload["validation"] = validation
                payload["optimizer"] = optimizer.state_dict()
                _atomic_torch_save(payload, latest_path)
                if score < best - minimum_delta:
                    best = score
                    best_step = step
                    validations_without_improvement = 0
                    _atomic_torch_save(payload, best_path)
                else:
                    validations_without_improvement += 1
                print(
                    json.dumps(
                        {
                            "stage": f"{stage}_validation",
                            "step": step,
                            "best_step": best_step,
                            "patience_used": validations_without_improvement,
                            **validation,
                        }
                    ),
                    flush=True,
                )
                self._set_stage(stage)
                if validations_without_improvement >= patience:
                    stopped_step = step
                    break
        best_payload = torch.load(
            best_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(best_payload["model"])
        self.history[stage].append(
            {
                "step": stopped_step,
                "status": "early_stopped" if stopped_step < steps else "max_steps",
                "best_step": best_step,
                "best_validation_loss": best,
            }
        )
        return best_path

    def fit(self) -> Path:
        for stage in ("head", "jump", "flow"):
            self.fit_stage(stage)
        if int(self.config["training"].get("proper_steps", 0)) != 0:
            raise RuntimeError(
                "StableMMTrainer v2 protocol requires proper_steps=0; "
                "the v1 proper-score branch is a recorded failed ablation"
            )
        self.fit_stage("proper")
        final = self.output_dir / "final.pt"
        flow_best = torch.load(
            self.output_dir / "flow_best.pt",
            map_location=self.device,
            weights_only=False,
        )
        payload = self._payload(stage="flow_validation_selected", step=flow_best["step"])
        payload["validation"] = flow_best.get("validation", {})
        payload["protocol_note"] = (
            "Proper-score fine-tuning disabled after pre-lock v1 pilot failed "
            "the finite-parameter gate; final equals validation-selected flow."
        )
        _atomic_torch_save(payload, final)
        self.output_dir.joinpath("history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        return final


__all__ = ["StableMMTrainer"]
