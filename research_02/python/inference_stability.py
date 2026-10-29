
"""Inference sensitivity diagnostics for Research 02.

Evaluates how single-step Max-T inference changes across
prespecified circular block-bootstrap lengths.

This is a sensitivity analysis, not a proof of robustness,
validity, or statistical coverage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from research_02.python.joint_inference import max_t_inference


def block_length_sensitivity(
    observations: ArrayLike,
    block_lengths: Sequence[int],
    *,
    n_resamples: int = 399,
    alpha: float = 0.05,
    seed: int = 2026,
) -> dict:
    """Compare joint inference across bootstrap block lengths.

    The same initial random seed is used for each block
    length. Results remain stochastic and should not be
    interpreted as independent across specifications.

    A stable rejection decision does not establish that
    the underlying test is correctly calibrated.
    """
    try:
        panel = np.asarray(observations, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid observation panel") from exc

    if panel.ndim != 2:
        raise ValueError("observations must be a 2D array")

    n, p = panel.shape

    if n < 3 or p < 1:
        raise ValueError("Insufficient panel dimensions")

    if not np.all(np.isfinite(panel)):
        raise ValueError("Observations must be finite")

    if isinstance(block_lengths, (str, bytes)):
        raise ValueError("block_lengths must be integers")

    try:
        lengths = tuple(block_lengths)
    except TypeError as exc:
        raise ValueError(
            "block_lengths must be iterable"
        ) from exc

    if not lengths:
        raise ValueError("At least one block length is required")

    for length in lengths:
        if isinstance(length, (bool, np.bool_)) or not isinstance(
            length, (int, np.integer)
        ):
            raise ValueError("Block lengths must be integers")

        if not 1 <= length <= n:
            raise ValueError("Block length outside panel range")

    if len(set(lengths)) != len(lengths):
        raise ValueError("Block lengths must be distinct")

    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, (int, np.integer)
    ) or n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    results = []

    for length in lengths:
        inference = max_t_inference(
            panel,
            block_length=int(length),
            n_resamples=int(n_resamples),
            alpha=float(alpha),
            seed=int(seed),
        )

        adjusted = np.asarray(
            inference["adjusted_p_values"],
            dtype=float,
        )

        if adjusted.shape != (p,):
            raise RuntimeError(
                "Unexpected adjusted p-value dimensions"
            )

        if not np.all(np.isfinite(adjusted)):
            raise RuntimeError(
                "Nonfinite adjusted p-values"
            )

        decisions = adjusted <= alpha

        results.append({
            "block_length": int(length),
            "adjusted_p_values": adjusted.copy(),
            "rejections": decisions.copy(),
            "n_rejections": int(np.sum(decisions)),
        })

    rejection_matrix = np.stack(
        [item["rejections"] for item in results],
        axis=0,
    )

    p_value_matrix = np.stack(
        [item["adjusted_p_values"] for item in results],
        axis=0,
    )

    stable_rejections = np.all(
        rejection_matrix,
        axis=0,
    )

    stable_nonrejections = ~np.any(
        rejection_matrix,
        axis=0,
    )

    decision_changes = ~(
        stable_rejections | stable_nonrejections
    )

    return {
        "method": "single_step_max_t_block_bootstrap",
        "n_observations": n,
        "n_instruments": p,
        "n_resamples": int(n_resamples),
        "alpha": float(alpha),
        "seed": int(seed),
        "block_lengths": tuple(int(x) for x in lengths),
        "specifications": results,
        "adjusted_p_value_range": (
            np.max(p_value_matrix, axis=0)
            - np.min(p_value_matrix, axis=0)
        ),
        "stable_rejections": stable_rejections,
        "stable_nonrejections": stable_nonrejections,
        "decision_changes": decision_changes,
        "n_decision_changes": int(np.sum(decision_changes)),
    }
