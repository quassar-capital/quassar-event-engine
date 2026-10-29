
"""Regression tests for Research 02 dependence estimation."""

import numpy as np
import pytest

from research_02.python.dependence import (
    sample_covariance,
    shrinkage_covariance,
    covariance_diagnostics,
    event_window_overlap,
)


def test_sample_covariance_reference_values():
    residuals = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    actual = sample_covariance(residuals)

    np.testing.assert_allclose(
        actual,
        [[1.0, 2.0], [2.0, 4.0]],
        atol=1e-12,
    )


def test_sample_covariance_matches_numpy():
    rng = np.random.default_rng(202602)
    residuals = rng.normal(size=(120, 8))

    actual = sample_covariance(residuals)
    expected = np.cov(residuals, rowvar=False)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_sample_covariance_rejects_invalid_data():
    with pytest.raises(ValueError):
        sample_covariance([[1.0, 2.0]])

    with pytest.raises(ValueError):
        sample_covariance([[1.0, np.nan], [2.0, 3.0]])

    with pytest.raises(ValueError):
        sample_covariance([1.0, 2.0, 3.0])


def test_diagonal_shrinkage_reference_values():
    residuals = [
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ]

    actual = shrinkage_covariance(
        residuals,
        intensity=0.5,
        target="diagonal",
    )

    np.testing.assert_allclose(
        actual,
        [[1.0, 1.0], [1.0, 4.0]],
        atol=1e-12,
    )


def test_spherical_shrinkage_reference_values():
    residuals = [
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ]

    actual = shrinkage_covariance(
        residuals,
        intensity=0.5,
        target="spherical",
    )

    np.testing.assert_allclose(
        actual,
        [[1.75, 1.0], [1.0, 3.25]],
        atol=1e-12,
    )


def test_shrinkage_boundary_values():
    rng = np.random.default_rng(902)
    residuals = rng.normal(size=(60, 5))
    sample = sample_covariance(residuals)

    np.testing.assert_allclose(
        shrinkage_covariance(
            residuals,
            intensity=0.0,
        ),
        sample,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        shrinkage_covariance(
            residuals,
            intensity=1.0,
            target="diagonal",
        ),
        np.diag(np.diag(sample)),
        atol=1e-12,
    )


def test_shrinkage_rejects_invalid_parameters():
    residuals = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ]

    for intensity in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            shrinkage_covariance(
                residuals,
                intensity=intensity,
            )

    with pytest.raises(ValueError):
        shrinkage_covariance(
            residuals,
            target="unsupported",
        )


def test_positive_definite_covariance_diagnostics():
    covariance = np.array([
        [2.0, 0.0],
        [0.0, 8.0],
    ])

    result = covariance_diagnostics(covariance)

    assert result["dimension"] == 2
    assert result["rank"] == 2
    assert result["positive_definite"] is True

    assert result["minimum_eigenvalue"] == pytest.approx(2.0)
    assert result["maximum_eigenvalue"] == pytest.approx(8.0)
    assert result["condition_number"] == pytest.approx(4.0)


def test_singular_covariance_diagnostics():
    covariance = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]

    result = covariance_diagnostics(covariance)

    assert result["rank"] == 1
    assert result["positive_definite"] is False
    assert np.isinf(result["condition_number"])


def test_indefinite_covariance_diagnostics():
    result = covariance_diagnostics([
        [1.0, 0.0],
        [0.0, -1.0],
    ])

    assert result["positive_definite"] is False
    assert result["minimum_eigenvalue"] == pytest.approx(-1.0)
    assert np.isinf(result["condition_number"])


def test_covariance_diagnostics_rejects_invalid_shape():
    with pytest.raises(ValueError):
        covariance_diagnostics([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

    with pytest.raises(ValueError):
        covariance_diagnostics([
            [1.0, 2.0],
            [0.0, 1.0],
        ])


def test_event_window_overlap_reference_values():
    windows = np.array([
        [0, 5],
        [4, 8],
        [8, 12],
        [2, 3],
    ])

    actual = event_window_overlap(windows)

    expected = np.array([
        [False, True, False, True],
        [True, False, False, False],
        [False, False, False, False],
        [True, False, False, False],
    ])

    np.testing.assert_array_equal(actual, expected)

    assert actual.dtype == np.bool_
    assert np.array_equal(actual, actual.T)
    assert not np.any(np.diag(actual))


def test_event_window_overlap_boundary_exclusion():
    actual = event_window_overlap([
        [0, 5],
        [5, 10],
    ])

    assert not np.any(actual)


def test_event_window_overlap_rejects_invalid_data():
    invalid_windows = [
        [[2, 2]],
        [[5, 3]],
        [[0.5, 2.5]],
        [[1, 2, 3]],
        [],
    ]

    for windows in invalid_windows:
        with pytest.raises(ValueError):
            event_window_overlap(windows)
