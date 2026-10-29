
"""Event-window sensitivity diagnostics for Research 02.

Compares cumulative abnormal returns across prespecified
event windows on a common integer observation calendar.

The output describes specification sensitivity. It does not
establish causal attribution or statistical significance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def event_window_sensitivity(
    abnormal_returns: ArrayLike,
    windows: Sequence[tuple[int, int]],
    *,
    event_index: int,
) -> dict:
    """Compute CAR sensitivity across half-open event windows.

    Each window is expressed relative to event_index, with
    start included and stop excluded. For example, (-2, 3)
    includes five observations centered on the event.

    All windows must lie within the observed sample.
    """
    try:
        returns = np.asarray(
            abnormal_returns,
            dtype=float,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid abnormal returns") from exc

    if returns.ndim != 2:
        raise ValueError(
            "abnormal_returns must have shape (time, instruments)"
        )

    n, p = returns.shape

    if n < 1 or p < 1:
        raise ValueError("Empty observation panel")

    if not np.all(np.isfinite(returns)):
        raise ValueError(
            "Abnormal returns must be finite"
        )

    if isinstance(event_index, (bool, np.bool_)) or not isinstance(
        event_index, (int, np.integer)
    ):
        raise ValueError("event_index must be an integer")

    if not 0 <= event_index < n:
        raise ValueError("event_index outside sample")

    try:
        specifications = tuple(windows)
    except TypeError as exc:
        raise ValueError(
            "windows must be iterable"
        ) from exc

    if not specifications:
        raise ValueError(
            "At least one event window is required"
        )

    validated = []

    for window in specifications:
        if not isinstance(window, (tuple, list)) or len(window) != 2:
            raise ValueError(
                "Each window must contain (start, stop)"
            )

        start, stop = window

        for boundary in (start, stop):
            if isinstance(boundary, (bool, np.bool_)) or not isinstance(
                boundary, (int, np.integer)
            ):
                raise ValueError(
                    "Window boundaries must be integers"
                )

        if start >= stop:
            raise ValueError(
                "Window start must precede stop"
            )

        absolute_start = int(event_index) + int(start)
        absolute_stop = int(event_index) + int(stop)

        if absolute_start < 0 or absolute_stop > n:
            raise ValueError(
                "Event window extends beyond observations"
            )

        validated.append(
            (int(start), int(stop))
        )

    if len(set(validated)) != len(validated):
        raise ValueError(
            "Event windows must be distinct"
        )

    car_matrix = np.stack(
        [
            np.sum(
                returns[
                    event_index + start:
                    event_index + stop,
                    :
                ],
                axis=0,
            )
            for start, stop in validated
        ],
        axis=0,
    )

    car_min = np.min(car_matrix, axis=0)
    car_max = np.max(car_matrix, axis=0)

    sign_positive = car_matrix > 0
    sign_negative = car_matrix < 0

    stable_positive = np.all(
        sign_positive,
        axis=0,
    )

    stable_negative = np.all(
        sign_negative,
        axis=0,
    )

    sign_instability = ~(
        stable_positive | stable_negative
    )

    return {
        "method": "event_window_car_sensitivity",
        "n_observations": n,
        "n_instruments": p,
        "event_index": int(event_index),
        "windows": tuple(validated),
        "car_matrix": car_matrix,
        "car_min": car_min,
        "car_max": car_max,
        "car_range": car_max - car_min,
        "stable_positive": stable_positive,
        "stable_negative": stable_negative,
        "sign_instability": sign_instability,
        "n_sign_instability": int(
            np.sum(sign_instability)
        ),
    }
