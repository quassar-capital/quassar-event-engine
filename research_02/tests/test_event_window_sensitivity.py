
"""Tests for Research 02 event-window sensitivity."""

import numpy as np
import pytest

from research_02.python.event_window_sensitivity import (
    event_window_sensitivity,
)


def _panel():
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, -1.0, 0.5],
        [2.0, -2.0, -0.5],
        [3.0, -3.0, 1.0],
        [4.0, -4.0, -1.0],
        [5.0, -5.0, 0.5],
        [6.0, -6.0, -0.5],
    ])


def _run(windows=((-1, 1), (-2, 2), (0, 2))):
    return event_window_sensitivity(
        _panel(),
        windows,
        event_index=3,
    )


def test_output_structure():
    result = _run()

    assert result["method"] == (
        "event_window_car_sensitivity"
    )
    assert result["n_observations"] == 7
    assert result["n_instruments"] == 3
    assert result["event_index"] == 3
    assert len(result["windows"]) == 3
    assert result["car_matrix"].shape == (3, 3)


def test_car_calculation():
    result = _run()

    expected = np.array([
        [5.0, -5.0, 0.5],
        [10.0, -10.0, 0.0],
        [7.0, -7.0, 0.0],
    ])

    np.testing.assert_allclose(
        result["car_matrix"],
        expected,
    )


def test_car_range():
    result = _run()

    np.testing.assert_allclose(
        result["car_min"],
        [5.0, -10.0, 0.0],
    )

    np.testing.assert_allclose(
        result["car_max"],
        [10.0, -5.0, 0.5],
    )

    np.testing.assert_allclose(
        result["car_range"],
        [5.0, 5.0, 0.5],
    )


def test_sign_stability():
    result = _run()

    np.testing.assert_array_equal(
        result["stable_positive"],
        [True, False, False],
    )

    np.testing.assert_array_equal(
        result["stable_negative"],
        [False, True, False],
    )

    np.testing.assert_array_equal(
        result["sign_instability"],
        [False, False, True],
    )

    assert result["n_sign_instability"] == 1


def test_single_window_has_zero_range():
    result = _run(windows=((-1, 1),))

    np.testing.assert_array_equal(
        result["car_range"],
        np.zeros(3),
    )


def test_reproducibility():
    first = _run()
    second = _run()

    for key in (
        "car_matrix",
        "car_min",
        "car_max",
        "car_range",
        "stable_positive",
        "stable_negative",
        "sign_instability",
    ):
        np.testing.assert_array_equal(
            first[key],
            second[key],
        )


def test_input_panel_not_modified():
    panel = _panel()
    original = panel.copy()

    event_window_sensitivity(
        panel,
        ((-1, 1), (0, 2)),
        event_index=3,
    )

    np.testing.assert_array_equal(
        panel,
        original,
    )


def test_boundary_window():
    result = event_window_sensitivity(
        _panel(),
        ((-3, 4),),
        event_index=3,
    )

    np.testing.assert_allclose(
        result["car_matrix"],
        np.sum(_panel(), axis=0)[None, :],
    )


@pytest.mark.parametrize(
    "windows",
    [
        (),
        ((0, 0),),
        ((2, 1),),
        ((-4, 1),),
        ((0, 5),),
        ((-1, 1), (-1, 1)),
        ((-1.5, 2),),
        ((True, 2),),
        ((0, 1, 2),),
    ],
)
def test_invalid_windows(windows):
    with pytest.raises(ValueError):
        event_window_sensitivity(
            _panel(),
            windows,
            event_index=3,
        )


@pytest.mark.parametrize(
    "event_index",
    [
        -1,
        7,
        1.5,
        True,
    ],
)
def test_invalid_event_index(event_index):
    with pytest.raises(ValueError):
        event_window_sensitivity(
            _panel(),
            ((0, 1),),
            event_index=event_index,
        )


@pytest.mark.parametrize(
    "panel",
    [
        np.ones(5),
        np.ones((0, 3)),
        np.ones((5, 0)),
        np.full((5, 3), np.nan),
        np.full((5, 3), np.inf),
    ],
)
def test_invalid_panels(panel):
    with pytest.raises(ValueError):
        event_window_sensitivity(
            panel,
            ((0, 1),),
            event_index=0,
        )
