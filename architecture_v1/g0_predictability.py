"""Train-only utilities for the G0-A NWP predictability audit.

The module deliberately contains no diffusion model.  It constructs six fixed
spatio-temporal projector groups, cross-fits a nuisance conditional mean, and
fits small Gaussian variance-score regressions.  Exact atoms and missing cells
are represented only by an explicit active mask; their numeric values never
enter the continuous residual geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ModeGroup:
    """One fixed spatial-by-temporal orthogonal projector group."""

    name: str
    temporal_label: str
    spatial_label: str
    spatial_projector: np.ndarray
    temporal_projector: np.ndarray
    leverage: np.ndarray
    full_rank: float


@dataclass(frozen=True)
class CellwiseRidge:
    """Independent ridge regressions for the 10 x 24 target coordinates."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    intercept: np.ndarray
    coefficient: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float64)
        if value.ndim != 4 or value.shape[1:3] != (10, 24):
            raise ValueError("cellwise features must have shape [day,10,24,feature]")
        if value.shape[-1] != self.coefficient.shape[-1]:
            raise ValueError("cellwise feature dimension drifted")
        standardized = (value - self.feature_mean[None]) / self.feature_std[None]
        prediction = self.intercept[None] + np.sum(
            standardized * self.coefficient[None], axis=-1
        )
        if not np.isfinite(prediction).all():
            raise FloatingPointError("cellwise ridge produced non-finite predictions")
        return prediction


@dataclass(frozen=True)
class LogVarianceModel:
    """Ridge-regularized log-variance GLM fitted by damped Newton steps."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    coefficient: np.ndarray
    variance_clip: tuple[float, float]

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != len(self.feature_mean):
            raise ValueError("variance-model feature shape drifted")
        standardized = (value - self.feature_mean) / self.feature_std
        design = np.concatenate(
            [np.ones((len(value), 1), dtype=np.float64), standardized], axis=1
        )
        eta = np.clip(design @ self.coefficient, -40.0, 40.0)
        prediction = np.exp(eta)
        prediction = np.clip(prediction, *self.variance_clip)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("variance model produced non-finite values")
        return prediction


def orthonormal_dct_ii(length: int) -> np.ndarray:
    """Return an explicitly normalized DCT-II basis without SciPy."""

    if length < 2:
        raise ValueError("DCT length must be at least two")
    position = np.arange(length, dtype=np.float64)[:, None]
    frequency = np.arange(length, dtype=np.float64)[None, :]
    basis = np.sqrt(2.0 / length) * np.cos(
        np.pi * (position + 0.5) * frequency / length
    )
    basis[:, 0] = np.sqrt(1.0 / length)
    if not np.allclose(basis.T @ basis, np.eye(length), atol=1e-12, rtol=0.0):
        raise RuntimeError("DCT-II construction lost orthonormality")
    return basis


def build_mode_groups(
    temporal_ranges: dict[str, Sequence[int]],
    ordered_names: Sequence[str],
) -> tuple[ModeGroup, ...]:
    """Construct the frozen common/local x low/mid/high registry."""

    dct = orthonormal_dct_ii(24)
    common = np.ones((10, 10), dtype=np.float64) / 10.0
    local = np.eye(10, dtype=np.float64) - common
    spatial = {"common": common, "local": local}
    temporal: dict[str, np.ndarray] = {}
    for label, bounds in temporal_ranges.items():
        if len(bounds) != 2:
            raise ValueError(f"temporal range {label!r} must contain two bounds")
        start, stop = map(int, bounds)
        if not 0 <= start < stop <= 24:
            raise ValueError(f"invalid temporal range {label!r}")
        subspace = dct[:, start:stop]
        temporal[label] = subspace @ subspace.T

    if not np.allclose(
        sum(temporal.values()), np.eye(24), atol=1e-12, rtol=0.0
    ):
        raise ValueError("temporal bands must partition the complete DCT basis")

    groups: list[ModeGroup] = []
    for name in ordered_names:
        pieces = str(name).split("_")
        if len(pieces) != 2:
            raise ValueError(f"mode group name must be temporal_spatial: {name!r}")
        temporal_label, spatial_label = pieces
        if temporal_label not in temporal or spatial_label not in spatial:
            raise ValueError(f"unknown mode group {name!r}")
        ps = spatial[spatial_label]
        pt = temporal[temporal_label]
        leverage = np.outer(np.diag(ps), np.diag(pt))
        full_rank = float(np.trace(ps) * np.trace(pt))
        groups.append(
            ModeGroup(
                name=str(name),
                temporal_label=temporal_label,
                spatial_label=spatial_label,
                spatial_projector=ps.copy(),
                temporal_projector=pt.copy(),
                leverage=leverage,
                full_rank=full_rank,
            )
        )

    if len(groups) != 6 or len({group.name for group in groups}) != 6:
        raise ValueError("G0-A requires exactly six unique mode groups")
    return tuple(groups)


def continuous_logit_target(
    target: np.ndarray,
    state: np.ndarray,
    observed_mask: np.ndarray,
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the D0-v continuous latent and its exact active mask."""

    value = np.asarray(target, dtype=np.float64)
    states = np.asarray(state)
    observed = np.asarray(observed_mask, dtype=bool)
    if value.shape != states.shape or value.shape != observed.shape:
        raise ValueError("target, state, and observed mask must align")
    if value.ndim != 3 or value.shape[1:] != (10, 24):
        raise ValueError("joint target must have shape [day,10,24]")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("logit epsilon must lie in (0,0.5)")
    active = observed & (states == 1)
    safe = np.where(active, np.clip(value, epsilon, 1.0 - epsilon), 0.5)
    latent = np.log(safe) - np.log1p(-safe)
    latent = np.where(active, latent, 0.0)
    if not np.isfinite(latent).all() or np.any(latent[~active] != 0.0):
        raise FloatingPointError("continuous latent violated mask semantics")
    return latent, active


def masked_band_energies(
    residual: np.ndarray,
    active_mask: np.ndarray,
    groups: Sequence[ModeGroup],
    *,
    minimum_effective_rank: float,
    energy_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mask-normalized projected residual energy for every day/group."""

    value = np.asarray(residual, dtype=np.float64)
    mask = np.asarray(active_mask, dtype=bool)
    if value.shape != mask.shape or value.ndim != 3 or value.shape[1:] != (10, 24):
        raise ValueError("residual and active mask must align as [day,10,24]")
    if minimum_effective_rank <= 0.0 or energy_floor <= 0.0:
        raise ValueError("effective-rank and energy floors must be positive")
    masked = np.where(mask, value, 0.0)
    energies = np.empty((len(value), len(groups)), dtype=np.float64)
    effective = np.empty_like(energies)
    for index, group in enumerate(groups):
        projected = np.einsum(
            "ij,djk,kl->dil",
            group.spatial_projector,
            masked,
            group.temporal_projector,
            optimize=True,
        )
        raw_energy = np.square(projected).sum(axis=(1, 2))
        rank = np.einsum("dzh,zh->d", mask, group.leverage, optimize=True)
        if np.any(rank < minimum_effective_rank):
            bad = int(np.flatnonzero(rank < minimum_effective_rank)[0])
            raise RuntimeError(
                f"mode group {group.name} has insufficient effective rank on day {bad}"
            )
        energies[:, index] = np.maximum(raw_energy / rank, energy_floor)
        effective[:, index] = rank
    if not np.isfinite(energies).all() or not np.isfinite(effective).all():
        raise FloatingPointError("band energy audit produced non-finite values")
    return energies, effective


def mask_nuisance_features(
    active_mask: np.ndarray,
    effective_rank: np.ndarray,
    groups: Sequence[ModeGroup],
) -> np.ndarray:
    """Return the seven frozen mask-only nuisance summaries."""

    mask = np.asarray(active_mask, dtype=bool)
    effective = np.asarray(effective_rank, dtype=np.float64)
    if mask.ndim != 3 or mask.shape[1:] != (10, 24):
        raise ValueError("active mask must have shape [day,10,24]")
    if effective.shape != (len(mask), len(groups)):
        raise ValueError("effective rank does not align with mode groups")
    fractions = np.column_stack(
        [
            mask.mean(axis=(1, 2)),
            *[
                effective[:, index] / group.full_rank
                for index, group in enumerate(groups)
            ],
        ]
    )
    if fractions.shape[1] != 7 or not np.isfinite(fractions).all():
        raise RuntimeError("mask nuisance feature contract failed")
    return fractions


def nwp_summary_features(
    raw_condition: np.ndarray,
    groups: Sequence[ModeGroup],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the frozen 34-dimensional day-level NWP summaries."""

    value = np.asarray(raw_condition, dtype=np.float64)
    if value.ndim != 4 or value.shape[1:] != (10, 24, 10):
        raise ValueError("raw NWP must have shape [day,10,24,10]")
    if not np.isfinite(value).all():
        raise ValueError("raw NWP contains non-finite values")
    channel_names = (
        "U10",
        "V10",
        "U100",
        "V100",
        "WS10",
        "WS100",
        "WE10",
        "WE100",
    )
    columns: list[np.ndarray] = []
    names: list[str] = []
    for channel_index, channel_name in enumerate(channel_names):
        field = value[..., channel_index]
        columns.extend([field.mean(axis=(1, 2)), field.std(axis=(1, 2), ddof=0)])
        names.extend([f"{channel_name}_mean", f"{channel_name}_std"])

    for channel_index, channel_name in ((2, "U100"), (3, "V100"), (5, "WS100")):
        field = value[..., channel_index]
        for group in groups:
            projected = np.einsum(
                "ij,djk,kl->dil",
                group.spatial_projector,
                field,
                group.temporal_projector,
                optimize=True,
            )
            rms = np.sqrt(np.square(projected).sum(axis=(1, 2)) / group.full_rank)
            columns.append(rms)
            names.append(f"{channel_name}_{group.name}_rms")
    features = np.column_stack(columns)
    if features.shape != (len(value), 34):
        raise RuntimeError("frozen NWP summary dimension must be exactly 34")
    if not np.isfinite(features).all():
        raise FloatingPointError("NWP summaries contain non-finite values")
    return features, tuple(names)


def chronological_folds(indices: Iterable[int], folds: int) -> tuple[np.ndarray, ...]:
    """Split already date-sorted indices into contiguous, exhaustive folds."""

    value = np.asarray(list(indices), dtype=np.int64)
    if value.ndim != 1 or len(value) < folds or folds < 2:
        raise ValueError("invalid chronological fold request")
    if len(np.unique(value)) != len(value):
        raise ValueError("fold indices contain duplicates")
    result = tuple(np.asarray(part, dtype=np.int64) for part in np.array_split(value, folds))
    if any(len(part) == 0 for part in result):
        raise RuntimeError("a chronological fold is empty")
    combined = np.concatenate(result)
    if not np.array_equal(combined, value):
        raise RuntimeError("chronological folds changed index ordering")
    return result


def _complement(universe: np.ndarray, held_out: np.ndarray) -> np.ndarray:
    keep = ~np.isin(universe, held_out)
    result = universe[keep]
    if len(result) + len(held_out) != len(universe):
        raise RuntimeError("fold complement is not disjoint and exhaustive")
    return result


def fit_cellwise_ridge(
    features: np.ndarray,
    target: np.ndarray,
    active_mask: np.ndarray,
    fit_indices: Sequence[int],
    *,
    alpha: float,
    minimum_observations: int,
) -> CellwiseRidge:
    """Fit 240 small local-NWP conditional-mean regressions."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    mask = np.asarray(active_mask, dtype=bool)
    selected = np.asarray(fit_indices, dtype=np.int64)
    if x.ndim != 4 or x.shape[1:3] != (10, 24):
        raise ValueError("mean-model features must have shape [day,10,24,feature]")
    if y.shape != x.shape[:3] or mask.shape != y.shape:
        raise ValueError("mean-model arrays do not align")
    if alpha < 0.0 or minimum_observations < 2:
        raise ValueError("invalid ridge fitting contract")
    feature_count = x.shape[-1]
    feature_mean = np.empty((10, 24, feature_count), dtype=np.float64)
    feature_std = np.empty_like(feature_mean)
    intercept = np.empty((10, 24), dtype=np.float64)
    coefficient = np.empty_like(feature_mean)
    identity = np.eye(feature_count, dtype=np.float64)
    for zone in range(10):
        for hour in range(24):
            usable = selected[mask[selected, zone, hour]]
            if len(usable) < minimum_observations:
                raise RuntimeError(
                    f"zone {zone + 1}, hour {hour} has only {len(usable)} fit observations"
                )
            local_x = x[usable, zone, hour]
            local_y = y[usable, zone, hour]
            mean = local_x.mean(axis=0)
            std = local_x.std(axis=0, ddof=0)
            std = np.where(std < 1e-10, 1.0, std)
            standardized = (local_x - mean) / std
            y_mean = float(local_y.mean())
            gram = standardized.T @ standardized + float(alpha) * identity
            rhs = standardized.T @ (local_y - y_mean)
            try:
                beta = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
            feature_mean[zone, hour] = mean
            feature_std[zone, hour] = std
            intercept[zone, hour] = y_mean
            coefficient[zone, hour] = beta
    model = CellwiseRidge(feature_mean, feature_std, intercept, coefficient)
    for array in (
        model.feature_mean,
        model.feature_std,
        model.intercept,
        model.coefficient,
    ):
        if not np.isfinite(array).all():
            raise FloatingPointError("cellwise ridge contains non-finite parameters")
    return model


def masked_mse(
    prediction: np.ndarray, target: np.ndarray, active_mask: np.ndarray
) -> tuple[float, int]:
    """Return SSE and count rather than a prematurely averaged fold loss."""

    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    mask = np.asarray(active_mask, dtype=bool)
    if pred.shape != truth.shape or pred.shape != mask.shape:
        raise ValueError("masked MSE inputs do not align")
    error = np.square(pred - truth)
    count = int(mask.sum())
    if count == 0:
        raise RuntimeError("masked MSE received no active cells")
    return float(error[mask].sum()), count


def select_cellwise_alpha(
    features: np.ndarray,
    target: np.ndarray,
    active_mask: np.ndarray,
    universe: Sequence[int],
    validation_folds: Sequence[np.ndarray],
    alphas: Sequence[float],
    *,
    minimum_observations: int,
) -> tuple[float, dict[str, float]]:
    """Choose one shared mean-model alpha by nested masked MSE."""

    all_indices = np.asarray(universe, dtype=np.int64)
    losses: dict[str, float] = {}
    for alpha in alphas:
        sse = 0.0
        count = 0
        for held_out in validation_folds:
            fit = _complement(all_indices, np.asarray(held_out, dtype=np.int64))
            model = fit_cellwise_ridge(
                features,
                target,
                active_mask,
                fit,
                alpha=float(alpha),
                minimum_observations=minimum_observations,
            )
            prediction = model.predict(np.asarray(features)[held_out])
            part_sse, part_count = masked_mse(
                prediction,
                np.asarray(target)[held_out],
                np.asarray(active_mask)[held_out],
            )
            sse += part_sse
            count += part_count
        losses[str(float(alpha))] = sse / count
    selected = min((loss, float(alpha)) for alpha, loss in ((a, losses[str(float(a))]) for a in alphas))[1]
    return selected, losses


def cross_fitted_cellwise_predictions(
    features: np.ndarray,
    target: np.ndarray,
    active_mask: np.ndarray,
    universe: Sequence[int],
    folds: Sequence[np.ndarray],
    *,
    alpha: float,
    minimum_observations: int,
) -> np.ndarray:
    """Predict each universe day with a model that excluded its fold."""

    all_indices = np.asarray(universe, dtype=np.int64)
    prediction = np.full(np.asarray(target).shape, np.nan, dtype=np.float64)
    seen = np.zeros(len(target), dtype=bool)
    for held_out in folds:
        held = np.asarray(held_out, dtype=np.int64)
        fit = _complement(all_indices, held)
        model = fit_cellwise_ridge(
            features,
            target,
            active_mask,
            fit,
            alpha=alpha,
            minimum_observations=minimum_observations,
        )
        prediction[held] = model.predict(np.asarray(features)[held])
        seen[held] = True
    if not bool(seen[all_indices].all()) or bool(seen[_complement(np.arange(len(seen)), all_indices)].any()):
        raise RuntimeError("cross-fitted prediction coverage is invalid")
    return prediction


def gaussian_variance_score(energy: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Gaussian variance log score without the model-independent constant."""

    observed = np.asarray(energy, dtype=np.float64)
    predicted = np.asarray(variance, dtype=np.float64)
    if observed.shape != predicted.shape:
        raise ValueError("energy and variance predictions must align")
    if np.any(observed <= 0.0) or np.any(predicted <= 0.0):
        raise ValueError("variance score requires positive inputs")
    score = np.log(predicted) + observed / predicted
    if not np.isfinite(score).all():
        raise FloatingPointError("variance score is non-finite")
    return score


def fit_static_variance(
    energy: np.ndarray,
    effective_rank: np.ndarray,
    *,
    variance_clip: tuple[float, float],
) -> float:
    """Fit the Gaussian-score-optimal weighted constant variance."""

    observed = np.asarray(energy, dtype=np.float64)
    weight = np.asarray(effective_rank, dtype=np.float64)
    if observed.ndim != 1 or observed.shape != weight.shape:
        raise ValueError("static variance inputs must be aligned vectors")
    value = float(np.sum(weight * observed) / np.sum(weight))
    return float(np.clip(value, *variance_clip))


def _variance_objective(
    design: np.ndarray,
    energy: np.ndarray,
    weight: np.ndarray,
    coefficient: np.ndarray,
    alpha: float,
) -> float:
    eta = np.clip(design @ coefficient, -40.0, 40.0)
    likelihood = np.sum(weight * (eta + energy * np.exp(-eta))) / np.sum(weight)
    penalty = 0.5 * float(alpha) * float(np.square(coefficient[1:]).sum())
    return float(likelihood + penalty)


def fit_log_variance_glm(
    features: np.ndarray,
    energy: np.ndarray,
    effective_rank: np.ndarray,
    *,
    alpha: float,
    variance_clip: tuple[float, float],
    maximum_iterations: int,
    tolerance: float,
) -> LogVarianceModel:
    """Fit the frozen convex log-variance model with damped Newton steps."""

    x = np.asarray(features, dtype=np.float64)
    observed = np.asarray(energy, dtype=np.float64)
    weight = np.asarray(effective_rank, dtype=np.float64)
    if x.ndim != 2 or observed.shape != (len(x),) or weight.shape != observed.shape:
        raise ValueError("log-variance fitting arrays do not align")
    if np.any(observed <= 0.0) or np.any(weight <= 0.0):
        raise ValueError("log-variance fitting requires positive energy and weights")
    if alpha < 0.0 or maximum_iterations < 1 or tolerance <= 0.0:
        raise ValueError("invalid log-variance optimizer settings")
    mean = x.mean(axis=0) if x.shape[1] else np.empty(0, dtype=np.float64)
    std = x.std(axis=0, ddof=0) if x.shape[1] else np.empty(0, dtype=np.float64)
    std = np.where(std < 1e-10, 1.0, std)
    standardized = (x - mean) / std if x.shape[1] else x.copy()
    design = np.concatenate(
        [np.ones((len(x), 1), dtype=np.float64), standardized], axis=1
    )
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    coefficient[0] = np.log(np.sum(weight * observed) / np.sum(weight))
    penalty = np.zeros(design.shape[1], dtype=np.float64)
    penalty[1:] = float(alpha)
    normalizer = float(np.sum(weight))

    for _ in range(maximum_iterations):
        eta = np.clip(design @ coefficient, -40.0, 40.0)
        scaled = observed * np.exp(-eta)
        gradient = design.T @ (weight * (1.0 - scaled)) / normalizer
        gradient += penalty * coefficient
        if float(np.max(np.abs(gradient))) < tolerance:
            break
        hessian = (design.T * (weight * scaled)) @ design / normalizer
        hessian += np.diag(penalty)
        hessian += np.eye(len(coefficient), dtype=np.float64) * 1e-10
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        # For a positive-definite Hessian, ``gradient @ step`` is positive
        # and subtracting the Newton step is a descent direction.  Extremely
        # ill-conditioned nuisance designs can lose that property at machine
        # precision; falling back to the gradient preserves the frozen
        # objective while keeping the solver fail-closed on real divergence.
        if not np.isfinite(step).all():
            raise FloatingPointError("log-variance Newton step is non-finite")
        if float(np.dot(gradient, step)) <= 0.0:
            step = gradient.copy()
        old_objective = _variance_objective(
            design, observed, weight, coefficient, alpha
        )
        scale = 1.0
        accepted = False
        while scale >= 2.0 ** -20:
            candidate = coefficient - scale * step
            candidate_objective = _variance_objective(
                design, observed, weight, candidate, alpha
            )
            if np.isfinite(candidate_objective) and candidate_objective <= old_objective:
                coefficient = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            # At the optimum, every representable trial can round to the
            # same or an infinitesimally larger objective.  Treat only such a
            # sub-tolerance step as convergence; material failures still stop.
            if float(np.max(np.abs(scale * step))) < tolerance:
                break
            raise FloatingPointError("log-variance Newton line search failed")
        if float(np.max(np.abs(scale * step))) < tolerance:
            break
    else:
        raise RuntimeError("log-variance Newton optimizer did not converge")

    model = LogVarianceModel(mean, std, coefficient, variance_clip)
    model.predict(x)
    return model


def select_variance_alpha(
    features: np.ndarray,
    energies: np.ndarray,
    effective_rank: np.ndarray,
    folds: Sequence[np.ndarray],
    alphas: Sequence[float],
    *,
    variance_clip: tuple[float, float],
    maximum_iterations: int,
    tolerance: float,
) -> tuple[float, dict[str, float]]:
    """Choose one ridge strength by six-band macro validation score."""

    x = np.asarray(features, dtype=np.float64)
    energy = np.asarray(energies, dtype=np.float64)
    rank = np.asarray(effective_rank, dtype=np.float64)
    if energy.shape != rank.shape or energy.shape[0] != len(x):
        raise ValueError("variance alpha-selection arrays do not align")
    universe = np.arange(len(x), dtype=np.int64)
    losses: dict[str, float] = {}
    for alpha in alphas:
        fold_scores: list[np.ndarray] = []
        for held_out in folds:
            held = np.asarray(held_out, dtype=np.int64)
            fit = _complement(universe, held)
            score = np.empty((len(held), energy.shape[1]), dtype=np.float64)
            for band in range(energy.shape[1]):
                model = fit_log_variance_glm(
                    x[fit],
                    energy[fit, band],
                    rank[fit, band],
                    alpha=float(alpha),
                    variance_clip=variance_clip,
                    maximum_iterations=maximum_iterations,
                    tolerance=tolerance,
                )
                predicted = model.predict(x[held])
                score[:, band] = gaussian_variance_score(
                    energy[held, band], predicted
                )
            fold_scores.append(score)
        losses[str(float(alpha))] = float(np.concatenate(fold_scores).mean())
    selected = min((loss, float(alpha)) for alpha, loss in ((a, losses[str(float(a))]) for a in alphas))[1]
    return selected, losses


def derangement(size: int, seed: int) -> np.ndarray:
    """Return a deterministic PCG64 derangement, failing on impossible sizes."""

    if size < 2:
        raise ValueError("a derangement requires at least two items")
    generator = np.random.default_rng(int(seed))
    identity = np.arange(size, dtype=np.int64)
    for _ in range(10000):
        candidate = generator.permutation(size)
        if not bool(np.any(candidate == identity)):
            return candidate.astype(np.int64, copy=False)
    raise RuntimeError("failed to construct a derangement after 10000 attempts")


def month_cluster_bootstrap(
    values: np.ndarray,
    days: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    """Paired bootstrap that resamples complete calendar-month clusters."""

    sample = np.asarray(values, dtype=np.float64)
    date = np.asarray(days, dtype="datetime64[D]")
    if sample.shape != (len(date),) or not np.isfinite(sample).all():
        raise ValueError("cluster-bootstrap inputs do not align")
    if replicates < 100 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid cluster-bootstrap contract")
    months = date.astype("datetime64[M]")
    unique = np.unique(months)
    if len(unique) < 12:
        raise RuntimeError("fewer than twelve calendar-month clusters")
    members = [np.flatnonzero(months == month) for month in unique]
    generator = np.random.default_rng(int(seed))
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = generator.integers(0, len(unique), size=len(unique))
        numerator = 0.0
        denominator = 0
        for cluster_index in selected:
            block = members[int(cluster_index)]
            numerator += float(sample[block].sum())
            denominator += len(block)
        draws[index] = numerator / denominator
    tail = (1.0 - confidence_level) / 2.0
    return {
        "estimate": float(sample.mean()),
        "lower": float(np.quantile(draws, tail)),
        "upper": float(np.quantile(draws, 1.0 - tail)),
        "probability_positive": float(np.mean(draws > 0.0)),
        "month_clusters": int(len(unique)),
        "replicates": int(replicates),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks used by the dependency-free Spearman diagnostic."""

    value = np.asarray(values, dtype=np.float64)
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        stop = start + 1
        while stop < len(value) and value[order[stop]] == value[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return Spearman correlation, or zero for a constant diagnostic vector."""

    x = rankdata(np.asarray(left, dtype=np.float64))
    y = rankdata(np.asarray(right, dtype=np.float64))
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x, y) / denominator)


__all__ = [
    "CellwiseRidge",
    "LogVarianceModel",
    "ModeGroup",
    "build_mode_groups",
    "chronological_folds",
    "continuous_logit_target",
    "cross_fitted_cellwise_predictions",
    "derangement",
    "fit_cellwise_ridge",
    "fit_log_variance_glm",
    "fit_static_variance",
    "gaussian_variance_score",
    "masked_band_energies",
    "masked_mse",
    "mask_nuisance_features",
    "month_cluster_bootstrap",
    "nwp_summary_features",
    "orthonormal_dct_ii",
    "select_cellwise_alpha",
    "select_variance_alpha",
    "spearman_correlation",
]
