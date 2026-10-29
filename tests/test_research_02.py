
"""Regression tests for Research 02 numerical foundations."""

import numpy as np
import pytest

from research_02.python.event_study import (
    abnormal_returns,
    cumulative_abnormal_returns,
    estimate_expected_returns,
    generalized_least_squares,
    gls_inference,
    holm_adjustment,
    benjamini_hochberg_adjustment,
    multiple_testing_decisions,
)


def test_abnormal_returns_and_car():
    observed = np.array([
        [0.03, 0.02, -0.01],
        [0.02, 0.01, 0.04],
    ])
    expected = np.full((2, 3), 0.01)

    abnormal = abnormal_returns(observed, expected)

    np.testing.assert_allclose(
        abnormal,
        [[0.02, 0.01, -0.02], [0.01, 0.0, 0.03]],
    )
    np.testing.assert_allclose(
        cumulative_abnormal_returns(abnormal, 0, 2),
        [0.03, 0.01],
    )


def test_abnormal_returns_rejects_bad_input():
    with pytest.raises(ValueError):
        abnormal_returns([[0.1]], [[0.1, 0.2]])

    with pytest.raises(ValueError):
        cumulative_abnormal_returns([[0.1]], 0, 2)


def test_expected_returns_factor_model():
    factors = np.arange(6, dtype=float).reshape(-1, 1)

    returns = np.vstack([
        1.0 + 2.0 * factors[:, 0],
        -1.0 + 0.5 * factors[:, 0],
    ])

    predicted = estimate_expected_returns(
        returns,
        factors,
        [[6.0], [7.0]],
    )

    np.testing.assert_allclose(
        predicted,
        [[13.0, 15.0], [2.0, 2.5]],
        atol=1e-12,
    )


def test_gls_known_covariance():
    x = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    y = np.array([2.0, 3.0, 4.0, 5.0])
    sigma = np.diag([1.0, 2.0, 1.0, 2.0])

    expected = np.array([2.0, 1.0])

    np.testing.assert_allclose(
        generalized_least_squares(x, y, sigma),
        expected,
        atol=1e-12,
    )

    result = gls_inference(x, y, sigma)

    np.testing.assert_allclose(
        result["coefficients"],
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["covariance"],
        np.linalg.inv(x.T @ np.linalg.solve(sigma, x)),
        atol=1e-12,
    )

    assert np.all(result["standard_errors"] > 0)


def test_gls_rejects_invalid_covariance():
    with pytest.raises(ValueError):
        generalized_least_squares(
            [[1.0], [1.0]],
            [1.0, 2.0],
            [[1.0, 0.0], [0.0, -1.0]],
        )


def test_multiple_testing_reference_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])

    np.testing.assert_allclose(
        holm_adjustment(p),
        [0.04, 0.09, 0.09, 0.20],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        benjamini_hochberg_adjustment(p),
        [
            0.04,
            0.05333333333333334,
            0.05333333333333334,
            0.20,
        ],
        atol=1e-12,
    )

    decisions = multiple_testing_decisions(
        p,
        alpha=0.05,
        method="holm",
    )

    assert decisions["reject"].tolist() == [
        True, False, False, False
    ]


def test_multiple_testing_validation():
    with pytest.raises(ValueError):
        holm_adjustment([-0.01, 0.5])

    with pytest.raises(ValueError):
        multiple_testing_decisions(
            [0.01],
            method="unsupported",
        )
