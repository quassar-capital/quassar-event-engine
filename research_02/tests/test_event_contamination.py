
"""Tests for Research 02 event contamination diagnostics."""

import numpy as np
import pytest

from research_02.python.event_contamination import (
    overlap_matrix,
    concurrent_announcements,
    estimation_contamination,
    contamination_report,
)


def test_overlapping_event_windows():
    windows = np.array([
        [10, 20],
        [15, 25],
        [30, 40],
    ])

    result = overlap_matrix(windows)

    expected = np.array([
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ])

    np.testing.assert_array_equal(result, expected)


def test_adjacent_windows_do_not_overlap():
    windows = np.array([
        [0, 5],
        [5, 10],
        [10, 15],
    ])

    assert not np.any(overlap_matrix(windows))


def test_overlap_matrix_is_symmetric():
    windows = np.array([
        [0, 10],
        [3, 8],
        [9, 14],
        [20, 30],
    ])

    result = overlap_matrix(windows)

    np.testing.assert_array_equal(result, result.T)
    assert not np.any(np.diag(result))


def test_single_event_has_no_pairwise_overlap():
    result = overlap_matrix([[10, 20]])

    assert result.shape == (1, 1)
    assert not result[0, 0]


def test_concurrent_announcements():
    times = np.array([10, 20, 10, 30])

    result = concurrent_announcements(times)

    expected = np.array([
        [False, False, True, False],
        [False, False, False, False],
        [True, False, False, False],
        [False, False, False, False],
    ])

    np.testing.assert_array_equal(result, expected)


def test_unique_announcements():
    times = np.array([1, 2, 3, 4])

    assert not np.any(concurrent_announcements(times))


def test_estimation_window_contamination():
    estimation = np.array([
        [0, 10],
        [10, 20],
        [20, 30],
    ])

    events = np.array([
        [30, 35],
        [5, 12],
        [40, 45],
    ])

    result = estimation_contamination(
        estimation,
        events,
    )

    expected = np.array([
        [False, True, False],
        [False, False, False],
        [False, False, False],
    ])

    np.testing.assert_array_equal(result, expected)


def test_estimation_contamination_excludes_diagonal():
    estimation = np.array([
        [0, 20],
        [20, 40],
    ])

    events = np.array([
        [10, 15],
        [25, 30],
    ])

    result = estimation_contamination(
        estimation,
        events,
    )

    assert not np.any(result)


def test_contamination_report_flags():
    events = np.array([
        [10, 20],
        [15, 25],
        [40, 50],
    ])

    estimation = np.array([
        [0, 8],
        [0, 8],
        [25, 35],
    ])

    times = np.array([12, 16, 45])

    report = contamination_report(
        events,
        estimation,
        times,
    )

    assert report["n_events"] == 3
    assert report["n_flagged_events"] == 2

    np.testing.assert_array_equal(
        report["potentially_contaminated"],
        [True, True, False],
    )


def test_clean_event_design():
    events = np.array([
        [10, 15],
        [30, 35],
        [50, 55],
    ])

    estimation = np.array([
        [0, 5],
        [20, 25],
        [40, 45],
    ])

    times = np.array([10, 30, 50])

    report = contamination_report(
        events,
        estimation,
        times,
    )

    assert report["n_flagged_events"] == 0
    assert not np.any(
        report["potentially_contaminated"]
    )


@pytest.mark.parametrize(
    "windows",
    [
        [],
        [[1, 1]],
        [[5, 2]],
        [[1.5, 3.5]],
        [[1, 2, 3]],
        [1, 2, 3],
        [[True, False]],
    ],
)
def test_invalid_windows_rejected(windows):
    with pytest.raises(ValueError):
        overlap_matrix(windows)


def test_invalid_announcement_times_rejected():
    invalid = [
        [],
        [[1, 2]],
        [1.5, 2.5],
        ["a", "b"],
    ]

    for values in invalid:
        with pytest.raises(ValueError):
            concurrent_announcements(values)


def test_mismatched_window_counts_rejected():
    with pytest.raises(ValueError):
        estimation_contamination(
            [[0, 5]],
            [[10, 15], [20, 25]],
        )


def test_mismatched_announcement_counts_rejected():
    with pytest.raises(ValueError):
        contamination_report(
            [[10, 15], [20, 25]],
            [[0, 5], [5, 10]],
            [10],
        )


def test_unsigned_overflow_rejected():
    windows = np.array(
        [[0, 2**63]],
        dtype=np.uint64,
    )

    with pytest.raises(ValueError):
        overlap_matrix(windows)


def test_negative_calendar_positions_supported():
    windows = np.array([
        [-10, -2],
        [-5, 3],
    ])

    result = overlap_matrix(windows)

    assert result[0, 1]
    assert result[1, 0]
