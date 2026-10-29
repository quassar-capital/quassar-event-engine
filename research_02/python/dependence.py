
"""Dependence estimation for high-dimensional financial event studies.

Research 02 — Quassar Capital.

Implements covariance shrinkage, stable covariance diagnostics,
and deterministic event-window overlap analysis.

No causal identification is implied by these estimators.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matrix(
    values: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    """Validate a finite, nonempty two-dimensional numeric matrix."""
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if result.ndim != 2 or min(result.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix")

    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")

    return result


def sample_covariance(
    residuals: ArrayLike,
) -> NDArray[np.float64]:
    """Estimate unbiased cross-sectional residual covariance.

    Input shape: (n_observations, n_instruments).

    Assumes rows are aligned observations and the residual
    vectors are suitable for covariance estimation.
    """
    x = _matrix(residuals, "residuals")
    n, p = x.shape

    if n < 2:
        raise ValueError("At least two observations are required")

    centered = x - np.mean(x, axis=0)
    covariance = (centered.T @ centered) / (n - 1)

    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance calculation overflowed")

    return 0.5 * (covariance + covariance.T)


def shrinkage_covariance(
    residuals: ArrayLike,
    intensity: float = 0.1,
    target: str = "diagonal",
) -> NDArray[np.float64]:
    """Regularize sample covariance toward a specified target.

    Sigma_hat = (1 - intensity) * S + intensity * T.

    Supported targets:
      diagonal: retain individual sample variances
      spherical: common variance equal to trace(S) / p

    The shrinkage intensity is user-specified, not estimated
    optimally. Positive definiteness is not guaranteed when
    sample variances are zero.
    """
    try:
        weight = float(intensity)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("intensity must be numeric") from exc

    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("intensity must lie in [0, 1]")

    covariance = sample_covariance(residuals)
    p = covariance.shape[0]

    if target == "diagonal":
        regularization_target = np.diag(
            np.diag(covariance)
        )
    elif target == "spherical":
        mean_variance = np.trace(covariance) / p
        regularization_target = (
            np.eye(p, dtype=np.float64) * mean_variance
        )
    else:
        raise ValueError("Unsupported shrinkage target")

    result = (
        (1.0 - weight) * covariance
        + weight * regularization_target
    )

    result = 0.5 * (result + result.T)

    if not np.all(np.isfinite(result)):
        raise ValueError("Regularized covariance is nonfinite")

    return result


def covariance_diagnostics(
    covariance: ArrayLike,
) -> dict[str, float | int | bool]:
    """Report symmetry, rank, spectrum, and conditioning.

    A singular or indefinite covariance matrix is flagged
    rather than silently modified.
    """
    sigma = _matrix(covariance, "covariance")

    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Covariance matrix must be square")

    symmetric = bool(
        np.allclose(
            sigma,
            sigma.T,
            rtol=1e-12,
            atol=1e-12,
        )
    )

    if not symmetric:
        raise ValueError("Covariance matrix must be symmetric")

    eigenvalues = np.linalg.eigvalsh(sigma)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    positive_definite = minimum > 0.0

    condition_number = (
        float(maximum / minimum)
        if positive_definite
        else float("inf")
    )

    return {
        "dimension": int(sigma.shape[0]),
        "rank": int(np.linalg.matrix_rank(sigma)),
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "positive_definite": bool(positive_definite),
        "condition_number": condition_number,
    }


def event_window_overlap(
    windows: ArrayLike,
) -> NDArray[np.bool_]:
    """Identify overlapping half-open event windows.

    Input shape: (n_events, 2).
    Each row contains [start, stop), with start < stop.

    Two windows overlap when they share at least one
    observation position. Diagonal entries are False.
    """
    try:
        raw = np.asarray(windows)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid event windows") from exc

    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("windows must have shape (n_events, 2)")

    if raw.shape[0] == 0:
        raise ValueError("At least one event is required")

    if raw.dtype.kind not in "iu":
        raise ValueError("Event boundaries must be integers")

    starts = raw[:, 0]
    stops = raw[:, 1]

    if np.any(starts >= stops):
        raise ValueError("Each event window must have start < stop")

    overlap = (
        (starts[:, None] < stops[None, :])
        & (starts[None, :] < stops[:, None])
    )

    np.fill_diagonal(overlap, False)

    return overlap
