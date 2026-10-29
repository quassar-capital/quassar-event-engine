
"""Tests for Research 02 inference stability diagnostics."""

import numpy as np
import pytest

from research_02.python.inference_stability import (
    block_length_sensitivity,
)


def _panel():
    rng = np.random.default_rng(2026)
    return rng.normal(size=(60, 4))


def _run(panel=None, lengths=(3, 5, 8)):
    if panel is None:
        panel = _panel()

    return block_length_sensitivity(
        panel,
        lengths,
        n_resamples=49,
        alpha=0.05,
        seed=77,
    )


def test_output_structure():
    result = _run()

    assert result["method"] == (
        "single_step_max_t_block_bootstrap"
    )
    assert result["n_observations"] == 60
    assert result["n_instruments"] == 4
    assert result["block_lengths"] == (3, 5, 8)
    assert len(result["specifications"]) == 3


def test_reproducibility():
    first = _run()
    second = _run()

    assert first["n_decision_changes"] == (
        second["n_decision_changes"]
    )

    for key in (
        "adjusted_p_value_range",
        "stable_rejections",
        "stable_nonrejections",
        "decision_changes",
    ):
        np.testing.assert_array_equal(
            first[key],
            second[key],
        )

    for a, b in zip(
        first["specifications"],
        second["specifications"],
    ):
        np.testing.assert_array_equal(
            a["adjusted_p_values"],
            b["adjusted_p_values"],
        )
        np.testing.assert_array_equal(
            a["rejections"],
            b["rejections"],
        )


def test_specification_results_are_consistent():
    result = _run()

    for specification in result["specifications"]:
        p_values = specification["adjusted_p_values"]
        decisions = specification["rejections"]

        assert p_values.shape == (4,)
        assert decisions.shape == (4,)

        assert np.all(
            (p_values >= 0.0) & (p_values <= 1.0)
        )

        np.testing.assert_array_equal(
            decisions,
            p_values <= result["alpha"],
        )

        assert specification["n_rejections"] == int(
            np.sum(decisions)
        )


def test_decision_categories_partition_instruments():
    result = _run()

    stable_yes = result["stable_rejections"]
    stable_no = result["stable_nonrejections"]
    changes = result["decision_changes"]

    assert stable_yes.shape == (4,)
    assert stable_no.shape == (4,)
    assert changes.shape == (4,)

    assert not np.any(stable_yes & stable_no)
    assert not np.any(stable_yes & changes)
    assert not np.any(stable_no & changes)

    np.testing.assert_array_equal(
        stable_yes | stable_no | changes,
        np.ones(4, dtype=bool),
    )

    assert result["n_decision_changes"] == int(
        np.sum(changes)
    )


def test_p_value_range_matches_specifications():
    result = _run()

    values = np.stack(
        [
            item["adjusted_p_values"]
            for item in result["specifications"]
        ],
        axis=0,
    )

    expected = (
        np.max(values, axis=0)
        - np.min(values, axis=0)
    )

    np.testing.assert_allclose(
        result["adjusted_p_value_range"],
        expected,
    )


def test_single_block_length_is_trivially_stable():
    result = _run(lengths=(5,))

    assert result["n_decision_changes"] == 0

    np.testing.assert_array_equal(
        result["adjusted_p_value_range"],
        np.zeros(4),
    )


def test_input_panel_is_not_modified():
    panel = _panel()
    original = panel.copy()

    _run(panel=panel)

    np.testing.assert_array_equal(panel, original)


@pytest.mark.parametrize(
    "lengths",
    [
        (),
        (0,),
        (61,),
        (3, 3),
        (2.5,),
        (True,),
        ("3",),
    ],
)
def test_invalid_block_lengths(lengths):
    with pytest.raises(ValueError):
        _run(lengths=lengths)


@pytest.mark.parametrize(
    "panel",
    [
        np.ones(5),
        np.ones((2, 3)),
        np.ones((10, 0)),
        np.full((10, 3), np.nan),
        np.full((10, 3), np.inf),
    ],
)
def test_invalid_panel(panel):
    with pytest.raises(ValueError):
        _run(panel=panel, lengths=(2,))


@pytest.mark.parametrize(
    "changes",
    [
        {"n_resamples": 0},
        {"n_resamples": True},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_inference_settings(changes):
    options = {
        "n_resamples": 49,
        "alpha": 0.05,
        "seed": 77,
    }
    options.update(changes)

    with pytest.raises(ValueError):
        block_length_sensitivity(
            _panel(),
            (3, 5),
            **options,
        )
