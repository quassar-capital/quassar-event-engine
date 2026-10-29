
"""Monte Carlo family-wise error evaluation for Research 02.

Evaluates single-step Max-T against unadjusted marginal
bootstrap testing under a joint zero-mean null.

All hypotheses are true in this design. Therefore, the
probability of at least one rejection estimates FWER.

Results are design-specific and include Monte Carlo
standard errors. No universal FWER guarantee is implied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import numpy as np

from research_02.python.joint_inference import max_t_inference
from research_02.python.monte_carlo import (
    SimulationConfig,
    simulate_panel,
    validate_config,
)


@dataclass(frozen=True)
class FWERConfig:
    n_observations: int = 120
    n_instruments: int = 8
    n_simulations: int = 200
    n_resamples: int = 399
    block_length: int = 6
    temporal_correlation: float = 0.3
    cross_correlation: float = 0.4
    alpha: float = 0.05
    seed: int = 2026


def _simulation_config(config: FWERConfig) -> SimulationConfig:
    if not isinstance(config, FWERConfig):
        raise ValueError("config must be FWERConfig")

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


def run_fwer_experiment(
    config: FWERConfig,
) -> dict[str, float | int | str | bool]:
    """Estimate complete-null family-wise rejection rates.

    For each independent Monte Carlo replication:
      1. Simulate a stationary, equicorrelated AR(1) panel.
      2. Apply the joint Max-T block-bootstrap procedure.
      3. Record any rejection with and without adjustment.

    Monte Carlo standard errors reflect variability across
    independent simulated panels, not bootstrap uncertainty.

    This evaluates weak FWER control under the complete null.
    Strong control under partial nulls is not established.
    """
    simulation = _simulation_config(config)

    rng = np.random.default_rng(int(config.seed))

    max_t_any_rejections = 0
    marginal_any_rejections = 0

    for _ in range(config.n_simulations):
        panel = simulate_panel(simulation, rng)

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

        max_t_any_rejections += int(
            np.any(result["adjusted_p_values"] <= config.alpha)
        )

        marginal_any_rejections += int(
            np.any(result["marginal_p_values"] <= config.alpha)
        )

    count = config.n_simulations

    adjusted_rate = max_t_any_rejections / count
    marginal_rate = marginal_any_rejections / count

    adjusted_se = sqrt(
        adjusted_rate * (1.0 - adjusted_rate) / count
    )

    marginal_se = sqrt(
        marginal_rate * (1.0 - marginal_rate) / count
    )

    return {
        "design": "stationary_gaussian_ar1_complete_null",
        "method": "single_step_max_t_block_bootstrap",
        "n_observations": config.n_observations,
        "n_instruments": config.n_instruments,
        "n_simulations": count,
        "n_resamples": config.n_resamples,
        "block_length": config.block_length,
        "temporal_correlation": config.temporal_correlation,
        "cross_correlation": config.cross_correlation,
        "alpha": config.alpha,
        "seed": int(config.seed),
        "max_t_fwer": adjusted_rate,
        "max_t_mc_se": adjusted_se,
        "marginal_family_rejection_rate": marginal_rate,
        "marginal_mc_se": marginal_se,
        "max_t_rejection_count": max_t_any_rejections,
        "marginal_rejection_count": marginal_any_rejections,
        "max_t_within_nominal": bool(
            adjusted_rate <= config.alpha
        ),
        "scope": "weak_fwer_complete_null_only",
    }


def compare_dependence_regimes(
    config: FWERConfig,
    correlations: tuple[float, ...] = (0.0, 0.4, 0.8),
) -> list[dict[str, float | int | str | bool]]:
    """Evaluate FWER across specified cross-correlations.

    The same initial seed is used for each regime,
    enabling reproducible paired-design comparisons.
    Results are not statistically independent across regimes.
    """
    _simulation_config(config)

    if not correlations:
        raise ValueError("At least one correlation is required")

    results = []

    for correlation in correlations:
        candidate = replace(
            config,
            cross_correlation=correlation,
        )
        _simulation_config(candidate)
        results.append(run_fwer_experiment(candidate))

    return results
