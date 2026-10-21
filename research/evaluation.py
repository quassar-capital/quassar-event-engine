"""Finite-horizon evaluation of sequential change-detection alarms."""

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class AlarmMetrics:
    """Empirical metrics under a stationary Gaussian null hypothesis.

    The restricted mean first-alarm time includes right-censored paths
    at the evaluation horizon; it is not an infinite-horizon ARL.
    """

    pathwise_false_alarm_probability: float
    observation_alarm_rate: float
    average_run_length: float
    trajectories: int


def evaluate_null_alarms(
    detector,
    *,
    replications=200,
    length=200,
    burn_in=20,
    threshold=0.80,
    seed=2026,
):
    """Estimate finite-horizon alarm behavior on independent null paths."""
    if isinstance(replications, bool) or not isinstance(replications, int):
        raise ValueError("Replications must be an integer")
    if isinstance(length, bool) or not isinstance(length, int):
        raise ValueError("Length must be an integer")
    if isinstance(burn_in, bool) or not isinstance(burn_in, int):
        raise ValueError("Burn-in must be an integer")
    if replications < 1 or burn_in < 0 or length <= burn_in:
        raise ValueError("Invalid simulation dimensions")
    if not isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must lie strictly between zero and one")
    if not callable(detector):
        raise TypeError("Detector must be callable")

    rng = np.random.default_rng(seed)
    horizon = length - burn_in
    paths_with_alarm = 0
    total_alarms = 0
    first_alarm_times = []

    for _ in range(replications):
        observations = rng.normal(size=length)
        scores = np.asarray(detector(observations), dtype=float)

        if scores.shape != (length,) or not np.all(np.isfinite(scores)):
            raise ValueError(
                "Detector must return one finite score per observation"
            )

        alarms = scores[burn_in:] >= threshold
        count = int(np.count_nonzero(alarms))
        total_alarms += count

        if count:
            paths_with_alarm += 1
            first_alarm_times.append(int(np.flatnonzero(alarms)[0]) + 1)
        else:
            first_alarm_times.append(horizon)

    return AlarmMetrics(
        pathwise_false_alarm_probability=paths_with_alarm / replications,
        observation_alarm_rate=total_alarms / (replications * horizon),
        average_run_length=float(np.mean(first_alarm_times)),
        trajectories=replications,
    )
