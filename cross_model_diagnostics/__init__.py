"""Architecture-agnostic diagnostics for saved wind-scenario archives.

The package deliberately depends only on NumPy for metric computation.  It
does not import the training packages, so saved confirmation archives can be
audited in a lightweight CPU environment without loading a model checkpoint.
"""

from .core import (
    atom_duration_summary,
    classify_states,
    cross_lag_summary,
    empirical_crps,
    generated_state_attribution,
    masked_per_day_scores,
    per_day_cross_lag_error,
    per_day_metrics,
    stratified_paired_bootstrap,
)
from .advanced import lag1_increment_state_decomposition

__all__ = [
    "atom_duration_summary",
    "classify_states",
    "cross_lag_summary",
    "empirical_crps",
    "generated_state_attribution",
    "lag1_increment_state_decomposition",
    "masked_per_day_scores",
    "per_day_cross_lag_error",
    "per_day_metrics",
    "stratified_paired_bootstrap",
]
