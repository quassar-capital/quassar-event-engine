
"""Validation of Research 02 complete-null FWER experiments."""

from dataclasses import replace

import numpy as np
import pytest

from research_02.python.fwer_experiment import (
    FWERConfig,
    run_fwer_experiment,
    compare_dependence_regimes,
)


def _small_config():
    return FWERConfig(
        n_observations=40,
        n_instruments=3,
        n_simulations=8,
        n_resamples=39,
        block_length=4,
        temporal_correlation=0.2,
        cross_correlation=0.3,
        alpha=0.05,
        seed=2026,
    )


def test_fwer_output_structure():
    config = _small_config()
    result = run_fwer_experiment(config)

    assert result["design"] == (
        "stationary_gaussian_ar1_complete_null"
    )
    assert result["method"] == (
        "single_step_max_t_block_bootstrap"
    )
    assert result["scope"] == "weak_fwer_complete_null_only"

    assert result["n_simulations"] == 8
    assert result["n_instruments"] == 3
    assert result["alpha"] == 0.05


def test_fwer_reproducibility():
    config = _small_config()

    first = run_fwer_experiment(config)
    second = run_fwer_experiment(config)

    assert first == second


def test_fwer_rates_are_valid_probabilities():
    result = run_fwer_experiment(_small_config())

    for key in (
        "max_t_fwer",
        "marginal_family_rejection_rate",
    ):
        assert 0.0 <= result[key] <= 1.0

    for key in (
        "max_t_mc_se",
        "marginal_mc_se",
    ):
        assert result[key] >= 0.0


def test_fwer_counts_match_reported_rates():
    result = run_fwer_experiment(_small_config())
    count = result["n_simulations"]

    assert result["max_t_fwer"] == pytest.approx(
        result["max_t_rejection_count"] / count
    )

    assert result[
        "marginal_family_rejection_rate"
    ] == pytest.approx(
        result["marginal_rejection_count"] / count
    )


def test_max_t_cannot_reject_more_families_than_marginal():
    result = run_fwer_experiment(_small_config())

    assert (
        result["max_t_rejection_count"]
        <= result["marginal_rejection_count"]
    )


def test_monte_carlo_standard_errors():
    result = run_fwer_experiment(_small_config())
    count = result["n_simulations"]

    for rate_key, se_key in (
        ("max_t_fwer", "max_t_mc_se"),
        (
            "marginal_family_rejection_rate",
            "marginal_mc_se",
        ),
    ):
        rate = result[rate_key]

        expected = np.sqrt(
            rate * (1.0 - rate) / count
        )

        assert result[se_key] == pytest.approx(expected)


def test_nominal_flag_matches_estimate():
    result = run_fwer_experiment(_small_config())

    assert result["max_t_within_nominal"] == (
        result["max_t_fwer"] <= result["alpha"]
    )


def test_compare_dependence_regimes():
    config = _small_config()

    results = compare_dependence_regimes(
        config,
        correlations=(0.0, 0.4, 0.7),
    )

    assert len(results) == 3

    assert [
        item["cross_correlation"]
        for item in results
    ] == [0.0, 0.4, 0.7]

    assert all(
        item["n_simulations"] == 8
        for item in results
    )


def test_dependence_comparison_reproducibility():
    config = _small_config()

    first = compare_dependence_regimes(
        config,
        correlations=(0.0, 0.5),
    )

    second = compare_dependence_regimes(
        config,
        correlations=(0.0, 0.5),
    )

    assert first == second


def test_invalid_configuration_rejected():
    baseline = _small_config()

    invalid = [
        {"n_observations": 1},
        {"n_instruments": 0},
        {"n_simulations": 0},
        {"n_resamples": 0},
        {"block_length": 0},
        {"block_length": 41},
        {"temporal_correlation": 1.0},
        {"cross_correlation": 1.0},
        {"cross_correlation": -0.6},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"seed": -1},
        {"seed": True},
    ]

    for changes in invalid:
        with pytest.raises(ValueError):
            run_fwer_experiment(
                replace(baseline, **changes)
            )


def test_invalid_dependence_regimes_rejected():
    config = _small_config()

    with pytest.raises(ValueError):
        compare_dependence_regimes(
            config,
            correlations=(),
        )

    with pytest.raises(ValueError):
        compare_dependence_regimes(
            config,
            correlations=(0.0, 1.0),
        )

    with pytest.raises(ValueError):
        compare_dependence_regimes(
            config,
            correlations=(0.0, -0.6),
        )
