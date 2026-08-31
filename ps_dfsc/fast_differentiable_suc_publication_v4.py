"""Batched Clarabel per-unit QP for publication training."""

from __future__ import annotations

from .differentiable_suc import RealizedOutput
from .fast_differentiable_suc_publication_v3 import (
    ClarabelPublicationDifferentiableSUC,
)


class BatchedClarabelPublicationDifferentiableSUC(
    ClarabelPublicationDifferentiableSUC
):
    """Support leading-day batches and parallel diffcp workers."""

    @staticmethod
    def _solver_arguments(solver_args):
        return {
            "solve_method": "Clarabel",
            "max_iter": 500,
            "tol_gap_abs": 1e-7,
            "tol_gap_rel": 1e-7,
            "tol_feas": 1e-7,
            "equilibrate_enable": True,
            "n_jobs_forward": 4,
            "n_jobs_backward": 4,
            **({} if solver_args is None else solver_args),
        }

    def realize(
        self,
        observed_wind,
        commitment,
        startup,
        shutdown,
        day_ahead_dispatch,
        reserve_up,
        reserve_down,
        *,
        solver_args: dict | None = None,
    ) -> RealizedOutput:
        if observed_wind.ndim == 2:
            wind = observed_wind[None]
        elif observed_wind.ndim == 3:
            wind = observed_wind[:, None]
        else:
            wind = observed_wind
        values = self._realized_layer(
            wind / self.power_scale,
            commitment,
            startup,
            shutdown,
            day_ahead_dispatch / self.power_scale,
            reserve_up / self.power_scale,
            reserve_down / self.power_scale,
            solver_args=self._solver_arguments(solver_args),
        )
        return RealizedOutput(
            *(value * self.power_scale for value in values)
        )


__all__ = ["BatchedClarabelPublicationDifferentiableSUC"]
