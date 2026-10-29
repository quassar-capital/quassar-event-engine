
"""Partial-null family-wise error experiments for Research 02.

Evaluates false rejections among true null hypotheses while
other instruments have nonzero means.

This is a design-specific empirical assessment of strong
FWER control, not a mathematical guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from research_02.python.joint_inference import max_t_inference
from research_02.python.monte_carlo import (
    SimulationConfig,
    simulate_panel,
    validate_config,
)


@dataclass(frozen=True)
class PartialNullConfig:
    n_observations: int = 120
    n_instruments: int = 8
    n_alternatives: int = 3
    effect_size: float = 0.5
    n_simulations: int = 200
    n_resamples: int = 399
    block_length: int = 6
    temporal_correlation: float = 0.3
    cross_correlation: float = 0.4
    alpha: float = 0.05
    seed: int = 2026


def _validate(config: PartialNullConfig) -> SimulationConfig:
    if not isinstance(config, PartialNullConfig):
        raise ValueError("config must be PartialNullConfig")

    if isinstance(config.n_alternatives, (bool, np.bool_)) or (
        not isinstance(config.n_alternatives, (int, np.integer))
    ):
        raise ValueError("n_alternatives must be an integer")

    if not 0 < config.n_alternatives < config.n_instruments:
        raise ValueError(
            "At least one true null and one alternative are required"
        )

    try:
        effect = float(config.effect_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("effect_size must be numeric") from exc

    if not np.isfinite(effect) or effect == 0.0:
        raise ValueError("effect_size must be finite and nonzero")

    simulation = SimulationConfig(
        n_observations=config.n_observations,
        n_instruments=config.n_instruments,
        n_simulations=config.n_simulations,
        n_resamples=config.n_resamples,
        block_length=config.block_length,
        temporal_correlation=config.temporal_correlation,
        cross_correlation=config.cross_correlation,
        effect_size=0.0,
        alpha=config.alpha,
        seed=config.seed,
    )

    validate_config(simulation)
    return simulation


def run_partial_null_experiment(
    config: PartialNullConfig,
) -> dict[str, float | int | str]:
    """Estimate FWER over true nulls and power over alternatives.

    The first n_alternatives instruments receive a common
    nonzero mean shift. Remaining instruments retain zero
    means and constitute the true-null family.

    The Max-T procedure is applied to the entire family,
    including both true and false null hypotheses.

    FWER counts any rejection among the true nulls.
    Power is the average rejection proportion among
    instruments under the alternative.
    """
    simulation = _validate(config)
    rng = np.random.default_rng(int(config.seed))

    false_family_rejections = 0
    marginal_false_family_rejections = 0
    alternative_rejections = 0

    k = config.n_alternatives
    p = config.n_instruments

    for _ in range(config.n_simulations):
        panel = simulate_panel(simulation, rng)

        panel[:, :k] += config.effect_size

        bootstrap_seed = int(
            rng.integers(
                0,
                np.iinfo(np.int64).max,
            )
        )

        result = max_t_inference(
            panel,
            block_length=config.block_length,
            n_resamples=config.n_resamples,
            alpha=config.alpha,
            seed=bootstrap_seed,
        )

        adjusted = result["adjusted_p_values"]
        marginal = result["marginal_p_values"]

        false_family_rejections += int(
            np.any(adjusted[k:] <= config.alpha)
        )

        marginal_false_family_rejections += int(
            np.any(marginal[k:] <= config.alpha)
        )

        alternative_rejections += int(
            np.sum(adjusted[:k] <= config.alpha)
        )

    n = config.n_simulations

    fwer = false_family_rejections / n
    marginal_fwer = marginal_false_family_rejections / n
    power = alternative_rejections / (n * k)

    return {
        "design": "stationary_gaussian_ar1_partial_null",
        "method": "single_step_max_t_block_bootstrap",
        "scope": "empirical_strong_fwer_partial_null_design",
        "n_observations": config.n_observations,
        "n_instruments": p,
        "n_alternatives": k,
        "n_true_nulls": p - k,
        "effect_size": config.effect_size,
        "n_simulations": n,
        "n_resamples": config.n_resamples,
        "block_length": config.block_length,
        "temporal_correlation": config.temporal_correlation,
        "cross_correlation": config.cross_correlation,
        "alpha": config.alpha,
        "seed": int(config.seed),
        "max_t_fwer": fwer,
        "max_t_fwer_mc_se": sqrt(
            fwer * (1.0 - fwer) / n
        ),
        "marginal_fwer": marginal_fwer,
        "marginal_fwer_mc_se": sqrt(
            marginal_fwer * (1.0 - marginal_fwer) / n
        ),
        "average_alternative_power": power,
        "alternative_rejection_count": alternative_rejections,
        "true_null_family_rejection_count": false_family_rejections,
    }
