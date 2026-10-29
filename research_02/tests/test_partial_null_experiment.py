
"""Validation of Research 02 partial-null FWER experiments."""

from dataclasses import replace

import numpy as np
import pytest

from research_02.python.partial_null_experiment import (
    PartialNullConfig,
    run_partial_null_experiment,
)


def _small_config():
    return PartialNullConfig(
        n_observations=48,
        n_instruments=4,
        n_alternatives=2,
        effect_size=0.6,
        n_simulations=8,
        n_resamples=49,
        block_length=4,
        temporal_correlation=0.2,
        cross_correlation=0.3,
        alpha=0.05,
        seed=2026,
    )


def test_partial_null_output_structure():
    result = run_partial_null_experiment(_small_config())

    assert result["design"] == (
        "stationary_gaussian_ar1_partial_null"
    )
    assert result["method"] == (
        "single_step_max_t_block_bootstrap"
    )
    assert result["scope"] == (
        "empirical_strong_fwer_partial_null_design"
    )

    assert result["n_instruments"] == 4
    assert result["n_alternatives"] == 2
    assert result["n_true_nulls"] == 2
    assert result["n_simulations"] == 8


def test_partial_null_reproducibility():
    config = _small_config()

    first = run_partial_null_experiment(config)
    second = run_partial_null_experiment(config)

    assert first == second


def test_partial_null_rates_are_valid():
    result = run_partial_null_experiment(_small_config())

    for key in (
        "max_t_fwer",
        "marginal_fwer",
        "average_alternative_power",
    ):
        assert 0.0 <= result[key] <= 1.0

    for key in (
        "max_t_fwer_mc_se",
        "marginal_fwer_mc_se",
    ):
        assert result[key] >= 0.0


def test_adjusted_fwer_not_above_marginal():
    result = run_partial_null_experiment(_small_config())

    assert result["max_t_fwer"] <= result["marginal_fwer"]


def test_rejection_counts_match_rates():
    result = run_partial_null_experiment(_small_config())
    count = result["n_simulations"]

    assert result["max_t_fwer"] == pytest.approx(
        result["true_null_family_rejection_count"] / count
    )

    assert result[
        "average_alternative_power"
    ] == pytest.approx(
        result["alternative_rejection_count"]
        / (count * result["n_alternatives"])
    )


def test_partial_null_mc_standard_error():
    result = run_partial_null_experiment(_small_config())
    count = result["n_simulations"]

    for rate_key, se_key in (
        ("max_t_fwer", "max_t_fwer_mc_se"),
        ("marginal_fwer", "marginal_fwer_mc_se"),
    ):
        rate = result[rate_key]

        expected = np.sqrt(
            rate * (1.0 - rate) / count
        )

        assert result[se_key] == pytest.approx(expected)


def test_invalid_partial_null_configuration():
    baseline = _small_config()

    invalid = [
        {"n_observations": 1},
        {"n_instruments": 0},
        {"n_alternatives": 0},
        {"n_alternatives": 4},
        {"n_alternatives": True},
        {"effect_size": 0.0},
        {"effect_size": np.nan},
        {"n_simulations": 0},
        {"n_resamples": 0},
        {"block_length": 0},
        {"block_length": 49},
        {"temporal_correlation": 1.0},
        {"cross_correlation": 1.0},
        {"cross_correlation": -0.5},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"seed": -1},
        {"seed": True},
    ]

    for changes in invalid:
        with pytest.raises(ValueError):
            run_partial_null_experiment(
                replace(baseline, **changes)
            )


def test_single_true_null_configuration():
    config = replace(
        _small_config(),
        n_alternatives=3,
    )

    result = run_partial_null_experiment(config)

    assert result["n_true_nulls"] == 1
    assert 0.0 <= result["max_t_fwer"] <= 1.0


def test_single_alternative_configuration():
    config = replace(
        _small_config(),
        n_alternatives=1,
    )

    result = run_partial_null_experiment(config)

    assert result["n_alternatives"] == 1
    assert result["n_true_nulls"] == 3
