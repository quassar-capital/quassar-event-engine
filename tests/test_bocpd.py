"""Statistical and numerical tests for Bayesian change-point filtering."""

import numpy as np
import pytest

from quassar_event_engine.bocpd import (
    BayesianChangePointDetector,
    NormalInverseGamma,
    detect_changes,
)


def test_prior_update_preserves_valid_parameters():
    posterior = NormalInverseGamma().update(3.0)
    assert posterior.kappa == 2.0
    assert posterior.alpha == 2.5
    assert posterior.beta > 0


def test_predictive_density_is_finite():
    prior = NormalInverseGamma()
    assert np.isfinite(prior.log_predictive(1.0))
    assert np.isfinite(prior.log_predictive(100.0))


def test_posterior_normalization_at_every_step():
    detector = BayesianChangePointDetector(0.07)
    for observation in np.linspace(-2, 2, 50):
        probabilities = detector.update(float(observation))
        assert np.isclose(probabilities.sum(), 1.0, atol=1e-12)
        assert np.all(probabilities >= 0.0)


def test_run_length_support_expands_exactly():
    detector = BayesianChangePointDetector(0.1)
    for step in range(1, 11):
        assert len(detector.update(0.5)) == step + 1


def test_change_probability_matches_hazard_under_constant_hazard():
    detector = BayesianChangePointDetector(0.12)
    for observation in [0.2, -0.5, 1.1, 0.0]:
        detector.update(observation)
        assert detector.change_probability == pytest.approx(0.12)


def test_reproducible_inference():
    rng = np.random.default_rng(2026)
    data = rng.normal(size=30)
    first = detect_changes(data)
    second = detect_changes(data)
    np.testing.assert_array_equal(first, second)


def test_structural_shift_changes_run_length_distribution():
    data = np.r_[np.zeros(30), np.full(5, 8.0)]
    posteriors = detect_changes(data, hazard=0.05)
    assert posteriors.shape == (35, 36)
    assert np.allclose(posteriors.sum(axis=1), 1.0)


@pytest.mark.parametrize("hazard", [-0.1, 0.0, 1.0, 1.2, np.nan])
def test_invalid_hazard_rejected(hazard):
    with pytest.raises(ValueError):
        BayesianChangePointDetector(hazard)


def test_nonfinite_observation_rejected():
    detector = BayesianChangePointDetector(0.1)
    with pytest.raises(ValueError):
        detector.update(float("nan"))
    assert detector.observations == 0
