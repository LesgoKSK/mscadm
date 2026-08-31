from __future__ import annotations

import torch

from .training import STGFTrainer


class PhysicalCenterSTGFTrainer(STGFTrainer):
    """STGF trainer whose center targets the physical conditional mean."""

    def _loss(
        self,
        stage: str,
        condition: torch.Tensor,
        observation: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if stage != "center":
            return super()._loss(
                stage,
                condition,
                observation,
                generator=generator,
            )
        target_logit = self.model.standardized_logit(observation)
        prediction_logit = self.model.center(condition)
        prediction = self.model.physical_from_standardized(prediction_logit)
        physical_mse = (prediction - observation).square().mean()
        physical_mae = (prediction - observation).abs().mean()
        logit_mse = (prediction_logit - target_logit).square().mean()
        training = self.config["training"]
        loss = (
            float(training.get("center_physical_weight", 1.0)) * physical_mse
            + float(training.get("center_logit_weight", 0.05)) * logit_mse
        )
        return {
            "loss": loss,
            "physical_mse": physical_mse,
            "physical_mae": physical_mae,
            "logit_mse": logit_mse,
        }


__all__ = ["PhysicalCenterSTGFTrainer"]
