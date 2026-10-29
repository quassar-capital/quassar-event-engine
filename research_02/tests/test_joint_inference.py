
"""Validation of Research 02 joint Max-T inference."""

import numpy as np
import pytest

from research_02.python.joint_inference import max_t_inference


def _panel(seed=2026, n=90, p=4):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n, 1))
    noise = rng.normal(size=(n, p))
    return 0.5 * common + noise


def test_output_structure_and_dimensions():
    observations = _panel()

    result = max_t_inference(
        observations,
        block_length=5,
        n_resamples=199,
        seed=42,
    )

    assert result["method"] == (
        "single_step_max_t_block_bootstrap"
    )
    assert result["n_hypotheses"] == 4
    assert result["n_resamples"] == 199

    for name in (
        "observed_means",
        "observed_statistics",
        "marginal_p_values",
        "adjusted_p_values",
        "reject",
    ):
        assert result[name].shape == (4,)


def test_reproducibility():
    observations = _panel(seed=99)

    first = max_t_inference(
        observations, 6, 149, seed=21
    )
    second = max_t_inference(
        observations, 6, 149, seed=21
    )

    for name in (
        "observed_means",
        "observed_statistics",
        "marginal_p_values",
        "adjusted_p_values",
        "reject",
    ):
        np.testing.assert_array_equal(
            first[name], second[name]
        )


def test_adjusted_p_values_are_not_smaller():
    observations = _panel(seed=301, p=6)

    result = max_t_inference(
        observations,
        block_length=4,
        n_resamples=299,
        seed=31,
    )

    assert np.all(
        result["adjusted_p_values"]
        >= result["marginal_p_values"]
    )

    assert np.all(
        (result["adjusted_p_values"] > 0.0)
        & (result["adjusted_p_values"] <= 1.0)
    )


def test_single_hypothesis_equals_marginal_test():
    observations = _panel(p=1)

    result = max_t_inference(
        observations,
        block_length=5,
        n_resamples=149,
        seed=13,
    )

    np.testing.assert_array_equal(
        result["adjusted_p_values"],
        result["marginal_p_values"],
    )


def test_identical_hypotheses_have_identical_p_values():
    rng = np.random.default_rng(200)
    series = rng.normal(size=100)
    observations = np.column_stack([
        series,
        series,
        series,
    ])

    result = max_t_inference(
        observations,
        block_length=5,
        n_resamples=199,
        seed=44,
    )

    np.testing.assert_array_equal(
        result["adjusted_p_values"],
        np.repeat(result["adjusted_p_values"][0], 3),
    )


def test_strong_signal_is_detected():
    rng = np.random.default_rng(900)
    observations = rng.normal(
        loc=1.0,
        scale=0.5,
        size=(120, 3),
    )

    result = max_t_inference(
        observations,
        block_length=5,
        n_resamples=399,
        alpha=0.05,
        seed=55,
    )

    assert np.all(result["reject"])


def test_zero_mean_null_design_is_well_formed():
    observations = _panel(seed=72)

    result = max_t_inference(
        observations,
        block_length=5,
        n_resamples=149,
        seed=71,
    )

    assert result["reject"].dtype == np.bool_

    np.testing.assert_allclose(
        result["observed_means"],
        observations.mean(axis=0),
        atol=1e-12,
    )


def test_invalid_inputs():
    observations = _panel()

    invalid_parameters = [
        {"block_length": 0},
        {"block_length": 91},
        {"n_resamples": 0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"alpha": np.nan},
        {"seed": -1},
        {"seed": True},
    ]

    for changes in invalid_parameters:
        parameters = {
            "block_length": 5,
            "n_resamples": 49,
            "alpha": 0.05,
            "seed": 20,
        }
        parameters.update(changes)

        with pytest.raises(ValueError):
            max_t_inference(
                observations,
                **parameters,
            )

    with pytest.raises(ValueError):
        max_t_inference(
            np.ones((30, 2)),
            block_length=3,
            n_resamples=20,
        )

    with pytest.raises(ValueError):
        max_t_inference(
            [[1.0, np.nan], [2.0, 3.0]],
            block_length=1,
            n_resamples=20,
        )

    with pytest.raises(ValueError):
        max_t_inference(
            [1.0, 2.0, 3.0],
            block_length=1,
            n_resamples=20,
        )
