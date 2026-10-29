
"""Joint multiple-hypothesis inference for Research 02.

Single-step Max-T inference using a circular block bootstrap
and Bartlett-kernel HAC studentization.

The same time indices are resampled across instruments to
preserve contemporaneous cross-sectional dependence.

The procedure is an asymptotic approximation. Its calibration
depends on stationarity, dependence strength, bandwidth choice,
block length, sample size, and bootstrap regularity conditions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from research_02.python.bootstrap import circular_block_indices


def _validate_panel(
    observations: ArrayLike,
) -> NDArray[np.float64]:
    try:
        panel = np.asarray(observations, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "observations must be numeric"
        ) from exc

    if panel.ndim != 2:
        raise ValueError(
            "observations must be two-dimensional"
        )

    if panel.shape[0] < 2 or panel.shape[1] < 1:
        raise ValueError(
            "At least two observations and one hypothesis required"
        )

    if not np.all(np.isfinite(panel)):
        raise ValueError(
            "observations must be finite"
        )

    return panel


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ) or value < 1:
        raise ValueError(
            f"{name} must be a positive integer"
        )

    return int(value)


def _hac_standard_errors(
    observations: NDArray[np.float64],
    bandwidth: int,
) -> NDArray[np.float64]:
    """Bartlett HAC standard errors for instrument-wise means.

    Long-run variance is estimated using centered observations
    and weighted autocovariances through the chosen bandwidth.

    The estimator uses normalization by n. The bandwidth is
    specified externally rather than selected optimally.
    """
    n = observations.shape[0]

    if not 0 <= bandwidth < n:
        raise ValueError(
            "HAC bandwidth must be between 0 and n - 1"
        )

    centered = observations - observations.mean(
        axis=0,
        keepdims=True,
    )

    long_run_variance = np.mean(
        centered * centered,
        axis=0,
    )

    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)

        autocovariance = np.sum(
            centered[lag:] * centered[:-lag],
            axis=0,
        ) / n

        long_run_variance += (
            2.0 * weight * autocovariance
        )

    if (
        not np.all(np.isfinite(long_run_variance))
        or np.any(long_run_variance <= 0.0)
    ):
        raise ValueError(
            "HAC long-run variance is nonpositive or nonfinite"
        )

    return np.sqrt(long_run_variance / n)


def max_t_inference(
    observations: ArrayLike,
    block_length: int,
    n_resamples: int = 1999,
    alpha: float = 0.05,
    seed: int = 2026,
) -> dict[str, NDArray | float | int | str]:
    """Single-step Max-T adjustment for zero-mean hypotheses.

    H0_j: E[X_t,j] = 0.

    Uses Bartlett HAC standard errors with bandwidth equal
    to block_length - 1 for observed and bootstrap statistics.

    Bootstrap observations are centered at their original
    sample means to impose the complete-null reference model.

    Adjusted p-values use the maximum absolute bootstrap
    statistic and the finite-resample plus-one correction.

    Empirical weak and strong FWER behavior must be assessed
    under explicitly stated data-generating processes.
    """
    panel = _validate_panel(observations)

    length = _positive_integer(
        block_length,
        "block_length",
    )
    count = _positive_integer(
        n_resamples,
        "n_resamples",
    )

    n, p = panel.shape

    if length > n:
        raise ValueError(
            "block_length exceeds sample size"
        )

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError(
            "seed must be a nonnegative integer"
        )

    try:
        threshold = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "alpha must be numeric"
        ) from exc

    if (
        not np.isfinite(threshold)
        or not 0.0 < threshold < 1.0
    ):
        raise ValueError(
            "alpha must lie in (0, 1)"
        )

    bandwidth = length - 1

    means = panel.mean(axis=0)
    centered = panel - means

    standard_errors = _hac_standard_errors(
        panel,
        bandwidth,
    )

    observed_statistics = means / standard_errors

    if not np.all(np.isfinite(observed_statistics)):
        raise ValueError(
            "Observed statistics are nonfinite"
        )

    rng = np.random.default_rng(int(seed))

    bootstrap_statistics = np.empty(
        (count, p),
        dtype=np.float64,
    )

    for replicate in range(count):
        indices = circular_block_indices(
            n,
            length,
            rng,
        )

        sampled = centered[indices]

        sampled_standard_errors = _hac_standard_errors(
            sampled,
            bandwidth,
        )

        bootstrap_statistics[replicate] = (
            sampled.mean(axis=0)
            / sampled_standard_errors
        )

    maximum_statistics = np.max(
        np.abs(bootstrap_statistics),
        axis=1,
    )

    absolute_observed = np.abs(
        observed_statistics
    )

    adjusted_p_values = (
        1.0
        + np.sum(
            maximum_statistics[:, None]
            >= absolute_observed[None, :],
            axis=0,
        )
    ) / (count + 1.0)

    marginal_p_values = (
        1.0
        + np.sum(
            np.abs(bootstrap_statistics)
            >= absolute_observed[None, :],
            axis=0,
        )
    ) / (count + 1.0)

    return {
        "method": "single_step_max_t_block_bootstrap",
        "studentization": "bartlett_hac",
        "hac_bandwidth": bandwidth,
        "observed_means": means,
        "observed_standard_errors": standard_errors,
        "observed_statistics": observed_statistics,
        "marginal_p_values": marginal_p_values,
        "adjusted_p_values": adjusted_p_values,
        "reject": adjusted_p_values <= threshold,
        "alpha": threshold,
        "block_length": length,
        "n_resamples": count,
        "n_hypotheses": p,
    }
