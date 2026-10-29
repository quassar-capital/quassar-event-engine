
"""Dependence-aware block bootstrap for Research 02.

Resamples complete cross-sectional observations in circular
time blocks. Suitable for sensitivity analysis under temporal
weak-dependence assumptions; not a universal finite-sample
inference guarantee.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _panel(values: ArrayLike) -> NDArray[np.float64]:
    try:
        panel = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Panel must be numeric") from exc

    if panel.ndim != 2 or min(panel.shape) == 0:
        raise ValueError("Panel must be a nonempty 2D array")

    if not np.all(np.isfinite(panel)):
        raise ValueError("Panel must contain finite values")

    return panel


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")

    if value < 1:
        raise ValueError(f"{name} must be positive")

    return int(value)


def circular_block_indices(
    n_observations: int,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate n circular-block resampling indices."""
    n = _positive_integer(n_observations, "n_observations")
    length = _positive_integer(block_length, "block_length")

    if length > n:
        raise ValueError(
            "block_length cannot exceed n_observations"
        )

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")

    n_blocks = (n + length - 1) // length
    starts = rng.integers(0, n, size=n_blocks)

    indices = (
        starts[:, None] + np.arange(length)[None, :]
    ) % n

    return indices.reshape(-1)[:n].astype(np.int64)


def block_bootstrap_means(
    observations: ArrayLike,
    block_length: int,
    n_resamples: int = 1999,
    seed: int = 2026,
    null_center: bool = False,
) -> NDArray[np.float64]:
    """Bootstrap instrument-wise sample means.

    Input: observations of shape (time, instruments).
    Output: bootstrap means of shape (resamples, instruments).

    The same sampled time indices are applied to every
    instrument, preserving contemporaneous co-movement.

    With null_center=True, each instrument is centered at
    its observed mean before resampling, imposing a zero
    mean null for the bootstrap reference distribution.
    """
    panel = _panel(observations)
    n = panel.shape[0]

    length = _positive_integer(block_length, "block_length")
    count = _positive_integer(n_resamples, "n_resamples")

    if length > n:
        raise ValueError(
            "block_length cannot exceed the sample size"
        )

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    rng = np.random.default_rng(int(seed))

    working = (
        panel - np.mean(panel, axis=0, keepdims=True)
        if null_center
        else panel
    )

    result = np.empty(
        (count, panel.shape[1]),
        dtype=np.float64,
    )

    for replicate in range(count):
        indices = circular_block_indices(n, length, rng)
        result[replicate] = np.mean(
            working[indices],
            axis=0,
        )

    if not np.all(np.isfinite(result)):
        raise ValueError("Bootstrap calculation is nonfinite")

    return result


def bootstrap_mean_inference(
    observations: ArrayLike,
    block_length: int,
    n_resamples: int = 1999,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> dict[str, NDArray[np.float64] | float | int]:
    """Compute percentile intervals and centered-null p-values.

    Confidence intervals use the percentile distribution of
    uncentered block-bootstrap sample means.

    Two-sided p-values compare absolute observed means with
    a separately null-centered bootstrap distribution.

    These procedures target stationary mean inference under
    appropriate weak-dependence and block-length conditions.
    They are not exact randomization tests.
    """
    panel = _panel(observations)

    try:
        level = float(confidence_level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confidence_level must be numeric") from exc

    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    observed_means = np.mean(panel, axis=0)

    bootstrap_means = block_bootstrap_means(
        panel,
        block_length=block_length,
        n_resamples=n_resamples,
        seed=seed,
        null_center=False,
    )

    null_means = block_bootstrap_means(
        panel,
        block_length=block_length,
        n_resamples=n_resamples,
        seed=seed,
        null_center=True,
    )

    alpha = 1.0 - level

    lower = np.quantile(
        bootstrap_means,
        alpha / 2.0,
        axis=0,
    )
    upper = np.quantile(
        bootstrap_means,
        1.0 - alpha / 2.0,
        axis=0,
    )

    exceedances = np.sum(
        np.abs(null_means) >= np.abs(observed_means),
        axis=0,
    )

    p_values = (
        1.0 + exceedances
    ) / (n_resamples + 1.0)

    return {
        "observed_means": observed_means,
        "confidence_interval_lower": lower,
        "confidence_interval_upper": upper,
        "p_values": p_values,
        "bootstrap_standard_errors": np.std(
            bootstrap_means,
            axis=0,
            ddof=1 if n_resamples > 1 else 0,
        ),
        "block_length": int(block_length),
        "n_resamples": int(n_resamples),
        "confidence_level": level,
    }
