"""Reproducible Monte Carlo evaluation of Bayesian regime detection."""

import numpy as np

from quassar_event_engine.bocpd import detect_changes


def recent_mass(data, hazard=0.02, window=10):
    posterior = detect_changes(data, hazard=hazard)
    return posterior[:, : window + 1].sum(axis=1)


def experiment(seed=2026, replications=100, length=100, shift=5.0):
    if replications <= 0 or length <= 20 or shift <= 0:
        raise ValueError("Invalid experimental configuration")

    rng = np.random.default_rng(seed)
    detections = []
    false_alarms = []
    delays = []

    threshold = 0.80

    for _ in range(replications):
        baseline = rng.normal(0.0, 1.0, length)
        changed = np.concatenate([
            rng.normal(0.0, 1.0, length),
            rng.normal(shift, 1.0, length),
        ])

        null_mass = recent_mass(baseline)
        alternative_mass = recent_mass(changed)

        # Exclude the initialization period from false-alarm measurement.
        false_alarms.append(bool(np.any(null_mass[20:] >= threshold)))

        detection_window = alternative_mass[length : length + 10]
        crossing = np.flatnonzero(detection_window >= threshold)
        detected = len(crossing) > 0

        detections.append(detected)
        if detected:
            delays.append(int(crossing[0]))

    return {
        "replications": replications,
        "detection_rate": float(np.mean(detections)),
        "false_alarm_probability": float(np.mean(false_alarms)),
        "mean_detection_delay": (
            float(np.mean(delays)) if delays else float("nan")
        ),
    }


if __name__ == "__main__":
    results = experiment()
    for name, value in results.items():
        print(f"{name}: {value}")

    assert 0.0 <= results["detection_rate"] <= 1.0
    assert 0.0 <= results["false_alarm_probability"] <= 1.0
    print("MONTE CARLO EVALUATION COMPLETED")
