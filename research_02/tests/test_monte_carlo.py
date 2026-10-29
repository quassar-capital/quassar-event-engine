
"""Validation of Research 02 Monte Carlo simulation engine."""

from dataclasses import replace

import numpy as np
import pytest

from research_02.python.monte_carlo import (
    SimulationConfig,
    validate_config,
    simulate_panel,
    run_monte_carlo,
)


def test_default_configuration_is_valid():
    validate_config(SimulationConfig())


def test_invalid_configuration_rejected():
    baseline = SimulationConfig()

    invalid = [
        {"n_observations": 1},
        {"n_instruments": 0},
        {"n_simulations": 0},
        {"n_resamples": 0},
        {"block_length": 0},
        {"block_length": 121},
        {"temporal_correlation": 1.0},
        {"temporal_correlation": -1.0},
        {"cross_correlation": 1.0},
        {"cross_correlation": -0.3},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"effect_size": np.nan},
        {"seed": -1},
        {"seed": True},
    ]

    for changes in invalid:
        with pytest.raises(ValueError):
            validate_config(
                replace(baseline, **changes)
            )


def test_single_instrument_configuration():
    config = SimulationConfig(
        n_instruments=1,
        cross_correlation=0.0,
    )

    validate_config(config)


def test_simulation_is_reproducible():
    config = SimulationConfig(
        n_observations=60,
        n_instruments=4,
    )

    first = simulate_panel(
        config,
        np.random.default_rng(77),
    )

    second = simulate_panel(
        config,
        np.random.default_rng(77),
    )

    np.testing.assert_array_equal(first, second)

    assert first.shape == (60, 4)


def test_simulated_panel_is_finite():
    config = SimulationConfig(
        n_observations=80,
        n_instruments=6,
        temporal_correlation=0.6,
        cross_correlation=0.5,
    )

    panel = simulate_panel(
        config,
        np.random.default_rng(20),
    )

    assert panel.shape == (80, 6)
    assert np.all(np.isfinite(panel))


def test_simulated_mean_matches_design():
    config = SimulationConfig(
        n_observations=12000,
        n_instruments=3,
        temporal_correlation=0.3,
        cross_correlation=0.4,
        effect_size=0.75,
    )

    panel = simulate_panel(
        config,
        np.random.default_rng(900),
    )

    np.testing.assert_allclose(
        panel.mean(axis=0),
        np.full(3, 0.75),
        atol=0.06,
    )


def test_simulated_dependence_matches_design():
    config = SimulationConfig(
        n_observations=15000,
        n_instruments=3,
        temporal_correlation=0.5,
        cross_correlation=0.6,
    )

    panel = simulate_panel(
        config,
        np.random.default_rng(100),
    )

    temporal = np.corrcoef(
        panel[:-1, 0],
        panel[1:, 0],
    )[0, 1]

    cross_sectional = np.corrcoef(
        panel[:, 0],
        panel[:, 1],
    )[0, 1]

    assert temporal == pytest.approx(0.5, abs=0.04)
    assert cross_sectional == pytest.approx(0.6, abs=0.04)


def test_monte_carlo_reproducibility():
    config = SimulationConfig(
        n_observations=40,
        n_instruments=3,
        n_simulations=12,
        n_resamples=49,
        block_length=4,
        seed=25,
    )

    first = run_monte_carlo(config)
    second = run_monte_carlo(config)

    assert first == second


def test_null_design_reports_type_i_error():
    config = SimulationConfig(
        n_observations=40,
        n_instruments=3,
        n_simulations=12,
        n_resamples=49,
        block_length=4,
        effect_size=0.0,
    )

    result = run_monte_carlo(config)

    assert result["evaluation"] == "type_i_error"
    assert result["n_simulations"] == 12

    assert 0.0 <= result["rejection_rate"] <= 1.0
    assert 0.0 <= result["coverage_rate"] <= 1.0

    assert result["rejection_mc_se"] >= 0.0
    assert result["coverage_mc_se"] >= 0.0


def test_alternative_design_reports_power():
    config = SimulationConfig(
        n_observations=40,
        n_instruments=3,
        n_simulations=12,
        n_resamples=49,
        block_length=4,
        effect_size=0.5,
    )

    result = run_monte_carlo(config)

    assert result["evaluation"] == "power"
    assert result["effect_size"] == 0.5


def test_monte_carlo_uncertainty_formula():
    config = SimulationConfig(
        n_observations=35,
        n_instruments=2,
        n_simulations=15,
        n_resamples=39,
        block_length=3,
    )

    result = run_monte_carlo(config)

    rejection = result["rejection_rate"]
    coverage = result["coverage_rate"]
    count = result["n_simulations"]

    assert result["rejection_mc_se"] == pytest.approx(
        np.sqrt(rejection * (1.0 - rejection) / count)
    )

    assert result["coverage_mc_se"] == pytest.approx(
        np.sqrt(coverage * (1.0 - coverage) / count)
    )
