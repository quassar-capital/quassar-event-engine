
"""Reproducible Monte Carlo evaluation for Research 02.

Evaluates block-bootstrap mean inference under stationary
Gaussian AR(1) panels with equicorrelated innovations.

Simulation results are descriptive finite-design evidence,
not universal guarantees of test validity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from research_02.python.bootstrap import bootstrap_mean_inference


@dataclass(frozen=True)
class SimulationConfig:
    n_observations: int = 120
    n_instruments: int = 5
    n_simulations: int = 200
    n_resamples: int = 399
    block_length: int = 6
    temporal_correlation: float = 0.3
    cross_correlation: float = 0.4
    effect_size: float = 0.0
    alpha: float = 0.05
    seed: int = 2026


def validate_config(config: SimulationConfig) -> None:
    """Validate the stationary equicorrelated simulation design."""
    if not isinstance(config, SimulationConfig):
        raise ValueError("config must be SimulationConfig")

    for name in (
        "n_observations",
        "n_instruments",
        "n_simulations",
        "n_resamples",
        "block_length",
    ):
        value = getattr(config, name)

        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    if config.n_observations < 2:
        raise ValueError("n_observations must be at least two")

    if config.block_length > config.n_observations:
        raise ValueError("block_length exceeds sample size")

    if isinstance(config.seed, (bool, np.bool_)) or not isinstance(
        config.seed, (int, np.integer)
    ) or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    for name in (
        "temporal_correlation",
        "cross_correlation",
        "effect_size",
        "alpha",
    ):
        try:
            value = float(getattr(config, name))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be numeric") from exc

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if abs(config.temporal_correlation) >= 1.0:
        raise ValueError(
            "Stationarity requires absolute temporal correlation < 1"
        )

    rho = config.cross_correlation
    p = config.n_instruments

    lower_bound = -1.0 / (p - 1) if p > 1 else -1.0

    if not lower_bound < rho < 1.0:
        raise ValueError(
            "Cross-correlation must yield positive-definite covariance"
        )

    if not 0.0 < config.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")


def simulate_panel(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a stationary Gaussian AR(1) return panel.

    Each instrument has marginal variance one.
    Innovations share an equicorrelation structure.
    The unconditional instrument mean is effect_size.
    """
    validate_config(config)

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")

    n = config.n_observations
    p = config.n_instruments
    phi = config.temporal_correlation
    rho = config.cross_correlation

    correlation = (
        (1.0 - rho) * np.eye(p)
        + rho * np.ones((p, p))
    )

    factor = np.linalg.cholesky(correlation)

    innovations = (
        rng.standard_normal((n, p)) @ factor.T
    ) * np.sqrt(1.0 - phi**2)

    panel = np.empty((n, p), dtype=np.float64)

    panel[0] = (
        rng.standard_normal(p) @ factor.T
    )

    for t in range(1, n):
        panel[t] = phi * panel[t - 1] + innovations[t]

    return panel + config.effect_size


def _binomial_standard_error(
    proportion: float,
    count: int,
) -> float:
    """Monte Carlo standard error of a Bernoulli proportion."""
    return float(
        np.sqrt(proportion * (1.0 - proportion) / count)
    )


def run_monte_carlo(
    config: SimulationConfig,
) -> dict[str, float | int | str]:
    """Evaluate rejection and interval coverage for instrument zero.

    Under effect_size == 0, rejection_rate estimates
    empirical type-I error at the configured alpha.

    Under effect_size != 0, rejection_rate estimates
    power against the zero-mean null.

    Coverage measures inclusion of the true mean in
    the percentile bootstrap confidence interval.

    Monte Carlo uncertainty is reported for both rates.
    """
    validate_config(config)

    rng = np.random.default_rng(int(config.seed))

    rejections = 0
    covered = 0

    for _ in range(config.n_simulations):
        panel = simulate_panel(config, rng)

        bootstrap_seed = int(
            rng.integers(0, np.iinfo(np.int64).max)
        )

        result = bootstrap_mean_inference(
            panel,
            block_length=config.block_length,
            n_resamples=config.n_resamples,
            confidence_level=1.0 - config.alpha,
            seed=bootstrap_seed,
        )

        p_value = float(result["p_values"][0])
        lower = float(
            result["confidence_interval_lower"][0]
        )
        upper = float(
            result["confidence_interval_upper"][0]
        )

        rejections += int(p_value <= config.alpha)

        covered += int(
            lower <= config.effect_size <= upper
        )

    count = config.n_simulations
    rejection_rate = rejections / count
    coverage_rate = covered / count

    return {
        "design": "stationary_gaussian_ar1_equicorrelated",
        "inference": "circular_block_bootstrap",
        "estimand": "instrument_0_mean",
        "evaluation": (
            "type_i_error"
            if config.effect_size == 0.0
            else "power"
        ),
        "n_simulations": count,
        "n_resamples": config.n_resamples,
        "n_observations": config.n_observations,
        "n_instruments": config.n_instruments,
        "block_length": config.block_length,
        "temporal_correlation": config.temporal_correlation,
        "cross_correlation": config.cross_correlation,
        "effect_size": config.effect_size,
        "alpha": config.alpha,
        "seed": int(config.seed),
        "rejection_rate": rejection_rate,
        "rejection_mc_se": _binomial_standard_error(
            rejection_rate, count
        ),
        "coverage_rate": coverage_rate,
        "coverage_mc_se": _binomial_standard_error(
            coverage_rate, count
        ),
    }
