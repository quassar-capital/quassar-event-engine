"""Tests for finite-horizon alarm evaluation."""

import numpy as np
import pytest

from research.evaluation import evaluate_null_alarms


def test_no_alarms():
    metrics = evaluate_null_alarms(
        lambda x: np.zeros(len(x)),
        replications=10,
        length=30,
        burn_in=5,
    )
    assert metrics.pathwise_false_alarm_probability == 0.0
    assert metrics.observation_alarm_rate == 0.0
    assert metrics.average_run_length == 25.0


def test_alarm_on_first_evaluated_observation():
    def detector(x):
        scores = np.zeros(len(x))
        scores[5] = 1.0
        return scores

    metrics = evaluate_null_alarms(
        detector, replications=10, length=30, burn_in=5
    )
    assert metrics.pathwise_false_alarm_probability == 1.0
    assert metrics.observation_alarm_rate == pytest.approx(1 / 25)
    assert metrics.average_run_length == 1.0


def test_alarm_on_last_observation_is_counted():
    def detector(x):
        scores = np.zeros(len(x))
        scores[-1] = 1.0
        return scores

    metrics = evaluate_null_alarms(
        detector, replications=10, length=30, burn_in=5
    )
    assert metrics.pathwise_false_alarm_probability == 1.0
    assert metrics.average_run_length == 25.0


def test_repeated_alarms_are_counted_per_observation():
    metrics = evaluate_null_alarms(
        lambda x: np.ones(len(x)),
        replications=10,
        length=30,
        burn_in=5,
    )
    assert metrics.pathwise_false_alarm_probability == 1.0
    assert metrics.observation_alarm_rate == 1.0
    assert metrics.average_run_length == 1.0


def test_reproducibility():
    detector = lambda x: np.abs(x) / (1.0 + np.abs(x))
    kwargs = dict(replications=30, length=50, seed=2026)
    assert evaluate_null_alarms(detector, **kwargs) == (
        evaluate_null_alarms(detector, **kwargs)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replications": 0},
        {"length": 20, "burn_in": 20},
        {"burn_in": -1},
        {"threshold": 0.0},
        {"threshold": 1.0},
        {"threshold": float("nan")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        evaluate_null_alarms(lambda x: x, **kwargs)


def test_invalid_detector_output():
    with pytest.raises(ValueError):
        evaluate_null_alarms(
            lambda x: np.array([1.0]),
            replications=2,
            length=30,
        )
