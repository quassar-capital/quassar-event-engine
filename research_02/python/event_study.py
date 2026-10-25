
"""Numerical foundations for high-dimensional financial event studies.

Research 02 — Quassar Capital.

This module provides deterministic estimation primitives. Statistical
inference, covariance-estimation uncertainty, and causal identification
require additional procedures and assumptions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_array(
    values: ArrayLike,
    name: str,
    ndim: int,
) -> NDArray[np.float64]:
    """Validate and convert an array to finite float64 values."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")

    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must not be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")

    return array


def abnormal_returns(
    observed: ArrayLike,
    expected: ArrayLike,
) -> NDArray[np.float64]:
    """Compute abnormal returns for aligned instrument-time panels.

    Both inputs must have shape (n_instruments, n_observations).
    """
    actual = _finite_array(observed, "observed", 2)
    baseline = _finite_array(expected, "expected", 2)

    if actual.shape != baseline.shape:
        raise ValueError("Observed and expected panels must match")

    result = actual - baseline

    if not np.all(np.isfinite(result)):
        raise ValueError("Abnormal-return calculation overflowed")

    return result


def cumulative_abnormal_returns(
    abnormal: ArrayLike,
    start: int,
    stop: int,
) -> NDArray[np.float64]:
    """Calculate CAR over the half-open event window [start, stop).

    The event window is indexed by observation position. Missing
    observations must be handled before invoking this function.
    """
    panel = _finite_array(abnormal, "abnormal", 2)
    n_observations = panel.shape[1]

    if (
        isinstance(start, (bool, np.bool_))
        or isinstance(stop, (bool, np.bool_))
        or not isinstance(start, (int, np.integer))
        or not isinstance(stop, (int, np.integer))
    ):
        raise ValueError("Event-window boundaries must be integers")

    if not 0 <= start < stop <= n_observations:
        raise ValueError("Invalid event-window boundaries")

    result = np.sum(panel[:, start:stop], axis=1)

    if not np.all(np.isfinite(result)):
        raise ValueError("CAR calculation overflowed")

    return result


def estimate_expected_returns(
    estimation_returns: ArrayLike,
    estimation_factors: ArrayLike,
    event_factors: ArrayLike,
) -> NDArray[np.float64]:
    """Fit separate factor models and predict event-window returns.

    estimation_returns: (n_instruments, n_estimation_observations)
    estimation_factors: (n_estimation_observations, n_factors)
    event_factors: (n_event_observations, n_factors)

    An intercept is included automatically. The estimation design
    must have full column rank. This function does not quantify
    parameter-estimation uncertainty.
    """
    returns = _finite_array(
        estimation_returns, "estimation_returns", 2
    )
    factors = _finite_array(
        estimation_factors, "estimation_factors", 2
    )
    future = _finite_array(event_factors, "event_factors", 2)

    if returns.shape[1] != factors.shape[0]:
        raise ValueError("Estimation observation counts must match")

    if factors.shape[1] != future.shape[1]:
        raise ValueError("Factor dimensions must match")

    design = np.column_stack(
        (np.ones(factors.shape[0]), factors)
    )
    future_design = np.column_stack(
        (np.ones(future.shape[0]), future)
    )

    if design.shape[0] <= design.shape[1]:
        raise ValueError("Insufficient estimation observations")

    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Estimation design is rank deficient")

    coefficients, _, _, _ = np.linalg.lstsq(
        design, returns.T, rcond=None
    )
    predictions = (future_design @ coefficients).T

    if not np.all(np.isfinite(predictions)):
        raise ValueError("Expected-return calculation overflowed")

    return predictions


def generalized_least_squares(
    design: ArrayLike,
    response: ArrayLike,
    covariance: ArrayLike,
) -> NDArray[np.float64]:
    """Estimate GLS coefficients using Cholesky whitening.

    The covariance matrix is treated as known and must be symmetric
    positive definite. The design matrix must have full column rank.

    Returned coefficients are point estimates, not inferential tests.
    """
    x = _finite_array(design, "design", 2)
    y = _finite_array(response, "response", 1)
    sigma = _finite_array(covariance, "covariance", 2)

    n_observations, n_parameters = x.shape

    if y.shape[0] != n_observations:
        raise ValueError("Response and design dimensions differ")

    if sigma.shape != (n_observations, n_observations):
        raise ValueError("Covariance dimensions do not match")

    if n_observations < n_parameters:
        raise ValueError("Underdetermined GLS system")

    if not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Covariance matrix must be symmetric")

    try:
        chol = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Covariance matrix must be positive definite"
        ) from exc

    whitened_x = np.linalg.solve(chol, x)
    whitened_y = np.linalg.solve(chol, y)

    if np.linalg.matrix_rank(whitened_x) != n_parameters:
        raise ValueError("GLS design is rank deficient")

    coefficients, _, _, _ = np.linalg.lstsq(
        whitened_x, whitened_y, rcond=None
    )

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("GLS calculation produced nonfinite values")

    return coefficients


def gls_inference(
    design: ArrayLike,
    response: ArrayLike,
    covariance: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """GLS coefficient estimates and known-covariance standard errors.

    The covariance matrix is assumed known, symmetric, and positive
    definite. These standard errors do not account for uncertainty
    from estimating the covariance matrix.
    """
    x = _finite_array(design, "design", 2)
    y = _finite_array(response, "response", 1)
    sigma = _finite_array(covariance, "covariance", 2)

    n, p = x.shape

    if y.size != n or sigma.shape != (n, n):
        raise ValueError("Incompatible GLS dimensions")

    if not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-12):
        raise ValueError("Covariance matrix must be symmetric")

    try:
        chol = np.linalg.cholesky(sigma)
        wx = np.linalg.solve(chol, x)
        wy = np.linalg.solve(chol, y)

        if np.linalg.matrix_rank(wx) != p:
            raise ValueError("GLS design is rank deficient")

        coefficients, _, _, _ = np.linalg.lstsq(
            wx, wy, rcond=None
        )

        information = wx.T @ wx
        parameter_covariance = np.linalg.solve(
            information, np.eye(p)
        )

    except np.linalg.LinAlgError as exc:
        raise ValueError("Invalid or singular GLS system") from exc

    variances = np.diag(parameter_covariance)

    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        raise ValueError("Invalid parameter variances")

    standard_errors = np.sqrt(variances)
    z_statistics = coefficients / standard_errors

    if not all(
        np.all(np.isfinite(result))
        for result in (
            coefficients,
            standard_errors,
            z_statistics,
            parameter_covariance,
        )
    ):
        raise ValueError("Nonfinite GLS inference results")

    return {
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "z_statistics": z_statistics,
        "covariance": parameter_covariance,
    }


def _validate_pvalues(
    pvalues: ArrayLike,
) -> NDArray[np.float64]:
    """Validate a nonempty vector of probability values."""
    values = _finite_array(pvalues, "pvalues", 1)

    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("P-values must lie within [0, 1]")

    return values


def holm_adjustment(
    pvalues: ArrayLike,
) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values.

    Controls family-wise error rate when the input p-values
    are individually valid, without requiring independence.
    """
    values = _validate_pvalues(pvalues)
    m = values.size

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]

    multipliers = np.arange(m, 0, -1, dtype=np.float64)

    adjusted_sorted = np.maximum.accumulate(
        sorted_values * multipliers
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted

    return adjusted


def benjamini_hochberg_adjustment(
    pvalues: ArrayLike,
) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values.

    Classical FDR guarantees require appropriate assumptions,
    such as independence or certain positive-dependence
    structures. Arbitrary dependence is not covered.
    """
    values = _validate_pvalues(pvalues)
    m = values.size

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]

    ranks = np.arange(1, m + 1, dtype=np.float64)

    raw_adjusted = sorted_values * (m / ranks)

    adjusted_sorted = np.minimum.accumulate(
        raw_adjusted[::-1]
    )[::-1]

    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted

    return adjusted


def multiple_testing_decisions(
    pvalues: ArrayLike,
    alpha: float = 0.05,
    method: str = "holm",
) -> dict[str, NDArray]:
    """Apply a multiple-testing correction to a hypothesis family."""
    values = _validate_pvalues(pvalues)

    if (
        not np.isfinite(alpha)
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("alpha must lie strictly between 0 and 1")

    procedures = {
        "holm": holm_adjustment,
        "benjamini-hochberg": benjamini_hochberg_adjustment,
    }

    if method not in procedures:
        raise ValueError("Unsupported multiple-testing procedure")

    adjusted = procedures[method](values)

    return {
        "adjusted_pvalues": adjusted,
        "reject": adjusted <= alpha,
    }
