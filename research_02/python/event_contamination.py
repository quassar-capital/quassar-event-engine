
"""Event-window contamination diagnostics for Research 02.

Detects overlapping event windows, concurrent announcements,
and estimation-window exposure to other event windows.

Intervals are half-open [start, stop) and use a shared
integer observation calendar. Findings indicate potential
contamination, not causal effects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _intervals(
    values: ArrayLike,
    name: str,
) -> NDArray[np.int64]:
    """Validate nonempty integer half-open intervals."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} contains invalid intervals"
        ) from exc

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (n, 2)"
        )

    if array.shape[0] == 0:
        raise ValueError(
            f"{name} cannot be empty"
        )

    if array.dtype.kind not in "iu":
        raise ValueError(
            f"{name} boundaries must be integers"
        )

    if array.dtype.kind == "u":
        maximum = np.iinfo(np.int64).max
        if np.any(array > maximum):
            raise ValueError(
                f"{name} boundaries exceed int64"
            )

    result = array.astype(np.int64)

    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError(
            f"{name} requires start < stop for every interval"
        )

    return result


def overlap_matrix(
    windows: ArrayLike,
) -> NDArray[np.bool_]:
    """Return pairwise overlap excluding self-overlap."""
    intervals = _intervals(windows, "windows")

    starts = intervals[:, 0]
    stops = intervals[:, 1]

    result = (
        (starts[:, None] < stops[None, :])
        & (starts[None, :] < stops[:, None])
    )

    np.fill_diagonal(result, False)

    return result


def concurrent_announcements(
    announcement_times: ArrayLike,
) -> NDArray[np.bool_]:
    """Flag events announced at identical calendar positions."""
    try:
        raw = np.asarray(announcement_times)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid announcement times"
        ) from exc

    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(
            "announcement_times must be a nonempty vector"
        )

    if raw.dtype.kind not in "iu":
        raise ValueError(
            "Announcement times must be integers"
        )

    result = raw[:, None] == raw[None, :]
    np.fill_diagonal(result, False)

    return result


def estimation_contamination(
    estimation_windows: ArrayLike,
    event_windows: ArrayLike,
) -> NDArray[np.bool_]:
    """Find estimation periods overlapping other event windows.

    Output [i, j] is True if estimation window i overlaps
    event window j. The diagonal is excluded: an event's own
    estimation window is not counted as external contamination.

    The two input arrays must have equal numbers of rows,
    representing aligned events.
    """
    estimation = _intervals(
        estimation_windows,
        "estimation_windows",
    )

    events = _intervals(
        event_windows,
        "event_windows",
    )

    if estimation.shape[0] != events.shape[0]:
        raise ValueError(
            "Estimation and event window counts must match"
        )

    result = (
        (estimation[:, 0, None] < events[None, :, 1])
        & (events[None, :, 0] < estimation[:, 1, None])
    )

    np.fill_diagonal(result, False)

    return result


def contamination_report(
    event_windows: ArrayLike,
    estimation_windows: ArrayLike,
    announcement_times: ArrayLike,
) -> dict[str, NDArray | int]:
    """Summarize observable timing-based contamination.

    Reports pairwise overlap, concurrency, and exposure of
    estimation periods to other events. It does not detect
    unobserved announcements or establish identification.
    """
    events = _intervals(
        event_windows,
        "event_windows",
    )

    estimation = _intervals(
        estimation_windows,
        "estimation_windows",
    )

    n = events.shape[0]

    if estimation.shape[0] != n:
        raise ValueError(
            "Window counts must match"
        )

    concurrency = concurrent_announcements(
        announcement_times
    )

    if concurrency.shape != (n, n):
        raise ValueError(
            "Announcement count must match events"
        )

    overlap = overlap_matrix(events)

    contamination = estimation_contamination(
        estimation,
        events,
    )

    flags = (
        np.any(overlap, axis=1)
        | np.any(concurrency, axis=1)
        | np.any(contamination, axis=1)
    )

    return {
        "n_events": n,
        "event_overlap": overlap,
        "concurrent_announcements": concurrency,
        "estimation_contamination": contamination,
        "potentially_contaminated": flags,
        "n_flagged_events": int(np.sum(flags)),
    }
