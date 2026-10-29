
"""Joint multiple-event inference for Research 02.

Single-step studentized Max-T circular block bootstrap.

Resampling uses common time indices across instruments to
retain contemporaneous dependence. Centering imposes the
joint zero-mean null for the bootstrap reference distribution.

Validity depends on suitable stationarity, weak dependence,
variance estimation, and bootstrap approximation conditions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from research_02.python.bootstrap import (
    circular_block_indices,
)


def _validate_panel(
    observations: ArrayLike,
) -> NDArray[np.float64]:
    try:
        panel = np.asarray(observations, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("observations must be numeric") from exc

    if panel.ndim != 2:
        raise ValueError("observations must be two-dimensional")

    if panel.shape[0] < 2 or panel.shape[1] < 1:
        raise ValueError(
            "At least two observations and one hypothesis required"
        )

    if not np.all(np.isfinite(panel)):
        raise ValueError("observations must be finite")

    return panel


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ) or value < 1:
        raise ValueError(f"{name} must be a positive integer")

    return int(value)


def max_t_inference(
    observations: ArrayLike,
    block_length: int,
    n_resamples: int = 1999,
    alpha: float = 0.05,
    seed: int = 2026,
) -> dict[str, NDArray | float | int | str]:
    """Single-step Max-T adjustment for multiple mean hypotheses.

    Null family: H0_j: E[X_t,j] = 0, for every instrument j.

    Uses sample standard deviations for studentization.
    These are not HAC standard errors; temporal dependence
    is approximated through circular block resampling.

    Constant columns are rejected because their estimated
    standard deviations are zero.

    The adjusted p-value for hypothesis j is the fraction
    of bootstrap maximum absolute statistics at least as
    large as its observed absolute statistic, with the
    standard finite-resample plus-one correction.
    """
    panel = _validate_panel(observations)

    length = _positive_integer(block_length, "block_length")
    count = _positive_integer(n_resamples, "n_resamples")

    n, p = panel.shape

    if length > n:
        raise ValueError("block_length exceeds sample size")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    try:
        threshold = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("alpha must be numeric") from exc

    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    means = panel.mean(axis=0)
    centered = panel - means

    standard_deviations = np.std(
        panel,
        axis=0,
        ddof=1,
    )

    if np.any(standard_deviations <= 0.0):
        raise ValueError(
            "All hypothesis columns must have positive variance"
        )

    if not np.all(np.isfinite(standard_deviations)):
        raise ValueError("Standard deviations are nonfinite")

    scale = standard_deviations / np.sqrt(n)

    observed_statistics = means / scale

    if not np.all(np.isfinite(observed_statistics)):
        raise ValueError("Observed statistics are nonfinite")

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

        sampled_scale = (
            np.std(sampled, axis=0, ddof=1) / np.sqrt(n)
        )

        # A degenerate resample cannot be studentized.
        # Reject rather than silently replacing its scale.
        if (
            np.any(sampled_scale <= 0.0)
            or not np.all(np.isfinite(sampled_scale))
        ):
            raise ValueError(
                "Degenerate bootstrap resample; "
                "revise the data or block configuration"
            )

        bootstrap_statistics[replicate] = (
            sampled.mean(axis=0) / sampled_scale
        )

    maximum_statistics = np.max(
        np.abs(bootstrap_statistics),
        axis=1,
    )

    adjusted_p_values = (
        1.0
        + np.sum(
            maximum_statistics[:, None]
            >= np.abs(observed_statistics)[None, :],
            axis=0,
        )
    ) / (count + 1.0)

    marginal_p_values = (
        1.0
        + np.sum(
            np.abs(bootstrap_statistics)
            >= np.abs(observed_statistics)[None, :],
            axis=0,
        )
    ) / (count + 1.0)

    return {
        "method": "single_step_max_t_block_bootstrap",
        "observed_means": means,
        "observed_statistics": observed_statistics,
        "marginal_p_values": marginal_p_values,
        "adjusted_p_values": adjusted_p_values,
        "reject": adjusted_p_values <= threshold,
        "alpha": threshold,
        "block_length": length,
        "n_resamples": count,
        "n_hypotheses": p,
    }
