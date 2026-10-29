
"""Regression tests for Research 02 temporal block bootstrap."""

import numpy as np
import pytest

from research_02.python.bootstrap import (
    circular_block_indices,
    block_bootstrap_means,
    bootstrap_mean_inference,
)


def test_circular_indices_are_reproducible():
    first = circular_block_indices(
        20, 4, np.random.default_rng(123)
    )
    second = circular_block_indices(
        20, 4, np.random.default_rng(123)
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (20,)
    assert np.all((first >= 0) & (first < 20))


def test_circular_blocks_preserve_adjacency():
    indices = circular_block_indices(
        25, 5, np.random.default_rng(77)
    )

    blocks = indices.reshape(5, 5)

    for block in blocks:
        np.testing.assert_array_equal(
            block[1:],
            (block[:-1] + 1) % 25,
        )


def test_block_length_one_produces_valid_indices():
    indices = circular_block_indices(
        12, 1, np.random.default_rng(10)
    )

    assert indices.shape == (12,)
    assert np.all(indices < 12)


def test_bootstrap_means_are_reproducible():
    rng = np.random.default_rng(2026)
    observations = rng.normal(size=(80, 6))

    first = block_bootstrap_means(
        observations, 5, 100, seed=42
    )
    second = block_bootstrap_means(
        observations, 5, 100, seed=42
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (100, 6)


def test_joint_resampling_preserves_identical_columns():
    rng = np.random.default_rng(200)
    series = rng.normal(size=100)
    observations = np.column_stack(
        [series, series, 3.0 * series]
    )

    samples = block_bootstrap_means(
        observations, 8, 200, seed=51
    )

    np.testing.assert_allclose(
        samples[:, 0],
        samples[:, 1],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        samples[:, 2],
        3.0 * samples[:, 0],
        atol=1e-12,
    )


def test_null_centered_bootstrap_has_zero_grand_mean():
    rng = np.random.default_rng(300)
    observations = rng.normal(
        loc=2.0,
        scale=1.0,
        size=(100, 3),
    )

    samples = block_bootstrap_means(
        observations,
        6,
        2000,
        seed=90,
        null_center=True,
    )

    assert np.all(
        np.abs(samples.mean(axis=0)) < 0.05
    )


def test_constant_panel_bootstrap():
    observations = np.full((40, 3), 2.5)

    samples = block_bootstrap_means(
        observations, 5, 100, seed=12
    )

    np.testing.assert_allclose(
        samples,
        np.full((100, 3), 2.5),
        atol=1e-12,
    )


def test_bootstrap_inference_output():
    rng = np.random.default_rng(501)
    observations = rng.normal(
        size=(120, 4)
    )

    result = bootstrap_mean_inference(
        observations,
        block_length=6,
        n_resamples=299,
        confidence_level=0.95,
        seed=2026,
    )

    assert result["observed_means"].shape == (4,)
    assert result["confidence_interval_lower"].shape == (4,)
    assert result["confidence_interval_upper"].shape == (4,)
    assert result["p_values"].shape == (4,)

    assert np.all(
        result["confidence_interval_lower"]
        <= result["confidence_interval_upper"]
    )

    assert np.all(
        (result["p_values"] > 0.0)
        & (result["p_values"] <= 1.0)
    )

    assert np.all(
        result["bootstrap_standard_errors"] >= 0.0
    )


def test_strong_signal_produces_small_p_value():
    rng = np.random.default_rng(800)
    observations = rng.normal(
        loc=1.0,
        scale=0.5,
        size=(150, 2),
    )

    result = bootstrap_mean_inference(
        observations,
        block_length=5,
        n_resamples=399,
        seed=33,
    )

    assert np.all(result["p_values"] < 0.05)


def test_constant_nonzero_signal():
    observations = np.full((30, 2), 3.0)

    result = bootstrap_mean_inference(
        observations,
        block_length=3,
        n_resamples=99,
        seed=5,
    )

    np.testing.assert_allclose(
        result["observed_means"],
        [3.0, 3.0],
    )

    np.testing.assert_allclose(
        result["confidence_interval_lower"],
        [3.0, 3.0],
    )

    np.testing.assert_allclose(
        result["confidence_interval_upper"],
        [3.0, 3.0],
    )

    np.testing.assert_allclose(
        result["p_values"],
        [0.01, 0.01],
    )


def test_invalid_block_parameters():
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError):
        circular_block_indices(10, 0, rng)

    with pytest.raises(ValueError):
        circular_block_indices(10, 11, rng)

    with pytest.raises(ValueError):
        circular_block_indices(0, 1, rng)

    with pytest.raises(ValueError):
        circular_block_indices(10, 2, None)


def test_invalid_bootstrap_inputs():
    observations = np.ones((20, 2))

    with pytest.raises(ValueError):
        block_bootstrap_means(
            observations, 3, n_resamples=0
        )

    with pytest.raises(ValueError):
        block_bootstrap_means(
            observations, 3, seed=-1
        )

    with pytest.raises(ValueError):
        block_bootstrap_means(
            observations, 3, seed=True
        )

    with pytest.raises(ValueError):
        block_bootstrap_means(
            observations, 21
        )

    with pytest.raises(ValueError):
        block_bootstrap_means(
            [[1.0, np.nan]], 1
        )


def test_invalid_confidence_levels():
    observations = np.ones((20, 2))

    for level in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            bootstrap_mean_inference(
                observations,
                block_length=3,
                n_resamples=20,
                confidence_level=level,
            )
