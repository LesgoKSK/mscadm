"""Date-clustered validation utilities."""

from __future__ import annotations

import numpy as np


def grouped_day_folds(day: np.ndarray, folds: int = 5, seed: int = 20260718) -> np.ndarray:
    """Assign every zone from the same calendar date to the same fold."""

    dates = np.asarray(day).astype("datetime64[D]")
    unique = np.unique(dates)
    if len(unique) < folds:
        raise ValueError("fewer unique dates than folds")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    date_to_fold = {value: index % folds for index, value in enumerate(shuffled)}
    assignments = np.asarray([date_to_fold[value] for value in dates], dtype=np.int64)
    for value in unique:
        if len(np.unique(assignments[dates == value])) != 1:
            raise AssertionError("calendar date leaked across folds")
    return assignments


def fold_table(day: np.ndarray, assignments: np.ndarray) -> list[dict[str, object]]:
    dates = np.asarray(day).astype("datetime64[D]")
    result: list[dict[str, object]] = []
    for value in np.unique(dates):
        mask = dates == value
        result.append(
            {
                "day": str(value),
                "fold": int(np.unique(assignments[mask]).item()),
                "zone_days": int(mask.sum()),
            }
        )
    return result


__all__ = ["grouped_day_folds", "fold_table"]
