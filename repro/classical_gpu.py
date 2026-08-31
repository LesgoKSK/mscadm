from __future__ import annotations

from xgboost import XGBRegressor

from .classical import QuantileRegressionGBM


class GPUQuantileRegressionGBM(QuantileRegressionGBM):
    """CUDA training variant with the same model and scenario semantics."""

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
            device="cuda",
            random_state=self.config.seed + hour,
            n_jobs=-1,
        )


__all__ = ["GPUQuantileRegressionGBM"]
