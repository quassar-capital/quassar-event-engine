"""Bayesian online change-point detection with a conjugate Gaussian model."""

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log, log1p, pi
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NormalInverseGamma:
    """Prior for an unknown Gaussian mean and variance."""

    mu: float = 0.0
    kappa: float = 1.0
    alpha: float = 2.0
    beta: float = 1.0

    def __post_init__(self):
        values = (self.mu, self.kappa, self.alpha, self.beta)
        if not all(isfinite(v) for v in values):
            raise ValueError("Prior parameters must be finite")
        if min(self.kappa, self.alpha, self.beta) <= 0:
            raise ValueError("Scale and shape parameters must be positive")

    def update(self, x: float) -> "NormalInverseGamma":
        if not isfinite(x):
            raise ValueError("Observations must be finite")
        kappa_new = self.kappa + 1.0
        mu_new = (self.kappa * self.mu + x) / kappa_new
        alpha_new = self.alpha + 0.5
        beta_new = self.beta + (
            self.kappa * (x - self.mu) ** 2 / (2.0 * kappa_new)
        )
        return NormalInverseGamma(mu_new, kappa_new, alpha_new, beta_new)

    def log_predictive(self, x: float) -> float:
        """Student-t log predictive density, integrating mean and variance."""
        if not isfinite(x):
            raise ValueError("Observations must be finite")
        nu = 2.0 * self.alpha
        scale2 = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        z2 = (x - self.mu) ** 2 / (nu * scale2)
        return (
            lgamma((nu + 1.0) / 2.0)
            - lgamma(nu / 2.0)
            - 0.5 * log(nu * pi * scale2)
            - ((nu + 1.0) / 2.0) * log1p(z2)
        )


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if maximum == -float("inf"):
        raise ArithmeticError("Posterior has zero total probability")
    return maximum + log(float(np.exp(values - maximum).sum()))


class BayesianChangePointDetector:
    """Exact run-length filtering under a constant hazard probability.

    The state retains one conjugate posterior for every possible run length.
    Memory and update cost grow linearly with the number of observations.
    """

    def __init__(
        self,
        hazard: float,
        prior: NormalInverseGamma | None = None,
    ):
        if not isfinite(hazard) or not 0.0 < hazard < 1.0:
            raise ValueError("Hazard must be strictly between zero and one")
        self.hazard = hazard
        self.prior = prior or NormalInverseGamma()
        self.log_run_probabilities = np.array([0.0])
        self.parameters = [self.prior]
        self.observations = 0

    def update(self, observation: float) -> np.ndarray:
        """Assimilate an observation and return run-length probabilities."""
        x = float(observation)
        if not isfinite(x):
            raise ValueError("Observation must be finite")

        predictive = np.array(
            [state.log_predictive(x) for state in self.parameters]
        )

        growth = (
            self.log_run_probabilities
            + predictive
            + log1p(-self.hazard)
        )

        reset = _logsumexp(
            self.log_run_probabilities + predictive + log(self.hazard)
        )

        next_log = np.empty(len(growth) + 1)
        next_log[0] = reset
        next_log[1:] = growth
        next_log -= _logsumexp(next_log)

        self.parameters = [
            self.prior.update(x),
            *[state.update(x) for state in self.parameters],
        ]
        self.log_run_probabilities = next_log
        self.observations += 1

        return np.exp(next_log).copy()

    @property
    def change_probability(self) -> float:
        """Posterior probability of a new regime at the latest observation."""
        if self.observations == 0:
            return 0.0
        return exp(float(self.log_run_probabilities[0]))


def detect_changes(
    observations: Iterable[float],
    hazard: float = 0.05,
    prior: NormalInverseGamma | None = None,
) -> np.ndarray:
    """Return the sequential run-length posterior as a padded matrix."""
    data = list(observations)
    detector = BayesianChangePointDetector(hazard, prior)
    result = np.zeros((len(data), len(data) + 1))

    for index, observation in enumerate(data):
        posterior = detector.update(observation)
        result[index, : len(posterior)] = posterior

    return result
