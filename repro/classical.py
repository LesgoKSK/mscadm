from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import rankdata
from xgboost import XGBRegressor

from .data import SplitData


def random_scenarios(
    evaluation: SplitData,
    *,
    scenarios: int = 100,
    source: SplitData | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Reproduce Dumas RAND, or use a separate leakage-free source split.

    The published reference passes the evaluation split itself as ``source``.
    Passing the training split instead gives the scientifically cleaner control.
    """
    population = evaluation if source is None else source
    generator = np.random.default_rng(seed)
    result = np.empty((len(evaluation), scenarios, evaluation.target.shape[1]), dtype=np.float32)
    for zone in np.unique(evaluation.zone):
        output_indices = np.flatnonzero(evaluation.zone == zone)
        candidates = population.target[population.zone == zone]
        if len(candidates) == 0:
            raise ValueError(f"No RAND source observations for zone {zone}")
        chosen = generator.integers(0, len(candidates), size=(len(output_indices), scenarios))
        result[output_indices] = candidates[chosen]
    return result


@dataclass
class QRGBMConfig:
    quantiles: int = 99
    estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.04
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 3.0
    tree_method: str = "hist"
    seed: int = 0


class QuantileRegressionGBM:
    """Hourly conditional quantile GBMs with ensemble-copula coupling.

    The article only specifies conditional quantile GBM and does not disclose
    hyperparameters or how marginal quantiles become 24-hour trajectories. This
    implementation makes that missing step explicit: marginal XGBoost quantiles
    are rearranged using same-zone training residual ranks (ECC/Schaake shuffle).
    """

    def __init__(self, config: QRGBMConfig | None = None) -> None:
        self.config = QRGBMConfig() if config is None else config
        self.levels = np.arange(1, self.config.quantiles + 1, dtype=np.float32) / (
            self.config.quantiles + 1
        )
        self.models: list[XGBRegressor] = []
        self.residual_rank_templates: dict[int, np.ndarray] = {}

    def _new_model(self, hour: int) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=self.levels,
            n_estimators=self.config.estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            tree_method=self.config.tree_method,
            random_state=self.config.seed + hour,
            n_jobs=-1,
        )

    def fit(self, split: SplitData) -> "QuantileRegressionGBM":
        self.models = []
        median_predictions = np.empty_like(split.target, dtype=np.float32)
        median_index = int(np.argmin(np.abs(self.levels - 0.5)))
        for hour in range(split.target.shape[1]):
            model = self._new_model(hour)
            model.fit(split.flat_condition, split.target[:, hour], verbose=False)
            prediction = np.asarray(model.predict(split.flat_condition), dtype=np.float32)
            if prediction.ndim == 1:
                prediction = prediction[:, None]
            median_predictions[:, hour] = prediction[:, median_index]
            self.models.append(model)
        residuals = split.target - median_predictions
        self.residual_rank_templates = {}
        for zone in np.unique(split.zone):
            zone_residuals = residuals[split.zone == zone]
            ranks = np.stack(
                [rankdata(row, method="average") / (len(row) + 1) for row in zone_residuals]
            ).astype(np.float32)
            self.residual_rank_templates[int(zone)] = ranks
        return self

    def predict_quantiles(self, condition: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("QRGBM has not been fitted")
        hourly = []
        for model in self.models:
            value = np.asarray(model.predict(condition), dtype=np.float32)
            hourly.append(value[:, None] if value.ndim == 1 else value)
        quantiles = np.stack(hourly, axis=1)
        return np.maximum.accumulate(quantiles, axis=-1)

    @staticmethod
    def _interpolate_levels(values: np.ndarray, levels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        extended_levels = np.concatenate(([0.0], levels, [1.0]))
        extended_values = np.concatenate((values[:1], values, values[-1:]))
        return np.interp(probabilities, extended_levels, extended_values)

    def sample(
        self,
        split: SplitData,
        *,
        scenarios: int = 100,
        seed: int = 0,
        coupling: Literal["ecc", "independent"] = "ecc",
    ) -> np.ndarray:
        quantiles = self.predict_quantiles(split.flat_condition)
        generator = np.random.default_rng(seed)
        result = np.empty((len(split), scenarios, split.target.shape[1]), dtype=np.float32)
        for index, zone in enumerate(split.zone):
            if coupling == "ecc":
                templates = self.residual_rank_templates[int(zone)]
                selected = templates[generator.integers(0, len(templates), scenarios)]
                # Randomize scenario order without destroying each 24-hour rank trajectory.
                probabilities = (selected + generator.uniform(-0.5 / 24, 0.5 / 24, selected.shape)).clip(0, 1)
            elif coupling == "independent":
                probabilities = generator.random((scenarios, split.target.shape[1]))
            else:
                raise ValueError(f"Unknown coupling: {coupling}")
            for hour in range(split.target.shape[1]):
                result[index, :, hour] = self._interpolate_levels(
                    quantiles[index, hour], self.levels, probabilities[:, hour]
                )
        return np.clip(result, 0.0, 1.0)


__all__ = ["random_scenarios", "QRGBMConfig", "QuantileRegressionGBM"]
