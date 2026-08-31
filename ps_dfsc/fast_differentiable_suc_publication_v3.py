"""Clarabel-backed per-unit QP for robust publication training."""

from __future__ import annotations

from .fast_differentiable_suc import stable_soft_cvar
from .fast_differentiable_suc_publication_v2 import (
    ScaledPublicationDifferentiableSUC,
)


class ClarabelPublicationDifferentiableSUC(
    ScaledPublicationDifferentiableSUC
):
    """Use diffcp's Clarabel path instead of numerically fragile SCS."""

    @staticmethod
    def _solver_arguments(solver_args):
        return {
            "solve_method": "Clarabel",
            "max_iter": 500,
            "tol_gap_abs": 1e-7,
            "tol_gap_rel": 1e-7,
            "tol_feas": 1e-7,
            "equilibrate_enable": True,
            **({} if solver_args is None else solver_args),
        }


def activate_clarabel_publication_training_layer() -> None:
    from . import training_v2

    training_v2.DifferentiableSUC = ClarabelPublicationDifferentiableSUC
    training_v2.soft_cvar = stable_soft_cvar


__all__ = [
    "ClarabelPublicationDifferentiableSUC",
    "activate_clarabel_publication_training_layer",
]
