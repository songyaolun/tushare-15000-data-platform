"""Verify mature-label gates, exact equal-weight compatibility and session decay."""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.training_weights import daily_equal_training_weights


def test_equal_weights_reproduce_original_operations_exactly():
    """Uneven daily row counts preserve the original bitwise weights and equal day mass."""
    days = np.array([0, 0, 1, 2, 2, 2])
    calendar = ["20251231", "20260105", "20260106"]
    weights, audit = daily_equal_training_weights(
        days, np.full(6, 2.), calendar, cutoff=calendar[-1],
    )
    _, inverse, counts = np.unique(days, return_inverse=True, return_counts=True)
    original = 1 / counts[inverse].astype(float)
    original /= original.mean()
    np.testing.assert_array_equal(weights, original)
    assert audit["method"] == "daily_equal"
    assert audit["day_effective_sample_size"] == pytest.approx(3)
    assert audit["annual_weights"]["2025"]["weight_share"] == pytest.approx(1 / 3)
    assert audit["annual_weights"]["2026"]["rows"] == 4
    assert audit["last_label_exit_date"] == "20260106"
    assert audit["immature_fit_rows"] == 0


def test_decay_uses_calendar_sessions_and_keeps_total_weight():
    """A missing fit day still ages samples; one session halves daily mass across year end."""
    calendar = ["20251230", "20251231", "20260105"]
    weights, audit = daily_equal_training_weights(
        np.array([0, 0, 2]), np.array([2., 2., 2.]), calendar,
        cutoff=calendar[-1], half_life_sessions=1,
    )
    np.testing.assert_allclose(weights, [.3, .3, 2.4], rtol=1e-15)
    assert weights.mean() == pytest.approx(1)
    assert audit["day_effective_sample_size"] == pytest.approx(25 / 17)
    assert audit["row_effective_sample_size"] == pytest.approx(50 / 33)
    assert audit["annual_weights"]["2025"]["weight_share"] == pytest.approx(.2)
    assert audit["method"] == "daily_equal_exponential"


@pytest.mark.parametrize("half_life", [126, 252, 504])
def test_frozen_half_lives_are_positive_and_report_exact_parameter(half_life):
    """All registered arms give the newest day higher mass and retain mean-one scale."""
    weights, audit = daily_equal_training_weights(
        np.array([0, 1]), np.array([1., 1.]), ["20260105", "20260106"],
        cutoff="20260106", half_life_sessions=half_life,
    )
    assert weights[1] / weights[0] == pytest.approx(2 ** (1 / half_life))
    assert audit["half_life_sessions"] == half_life
    assert audit["weight_sum"] == pytest.approx(2)


@pytest.mark.parametrize("calendar,cutoff", [
    ([], "20260105"),
    (["20260106", "20260105"], "20260105"),
    (["20260105", "20260105"], "20260105"),
    (["202615"], "202615"),
    (["abcdefgh"], "abcdefgh"),
    (["20260105"], "20260106"),
])
def test_invalid_calendar_or_absent_cutoff_fails_closed(calendar, cutoff):
    """Empty, unordered, duplicate, malformed or unrepresented date axes cannot be aged."""
    with pytest.raises(ValueError, match="invalid_training_calendar_or_cutoff"):
        daily_equal_training_weights(np.array([0]), np.array([0.]), calendar, cutoff=cutoff)


@pytest.mark.parametrize("half_life", [0, -1, True, 1., "126", float("nan")])
def test_invalid_half_life_fails_closed(half_life):
    """Zero, negative, bool and non-integer half-lives are never silently coerced."""
    with pytest.raises(ValueError, match="invalid_training_half_life"):
        daily_equal_training_weights(
            np.array([0]), np.array([0.]), ["20260105"],
            cutoff="20260105", half_life_sessions=half_life,
        )


@pytest.mark.parametrize("days,exits", [
    ([[0]], [0.]), ([], []), ([0.], [0.]), ([0], []), ([0], [np.nan]),
    ([0], [np.inf]), ([0], [.5]), ([-1], [0.]), ([2], [2.]),
    ([1], [0.]), ([0], [2.]),
])
def test_invalid_or_immature_labels_fail_without_dropping_rows(days, exits):
    """Malformed axes, future signals and future exits reject the entire fit before training."""
    with pytest.raises(ValueError, match="invalid_or_immature_training_labels"):
        daily_equal_training_weights(
            np.asarray(days), np.asarray(exits), ["20260105", "20260106"], cutoff="20260106",
        )


def test_cutoff_can_precede_panel_end_only_when_every_label_is_known():
    """An explicit earlier cutoff accepts its same-day exit but rejects an exit one day later."""
    calendar = ["20260105", "20260106"]
    weights, audit = daily_equal_training_weights(
        np.array([0]), np.array([0.]), calendar, cutoff="20260105",
    )
    np.testing.assert_array_equal(weights, [1.])
    assert audit["cutoff_index"] == 0
    with pytest.raises(ValueError, match="invalid_or_immature_training_labels"):
        daily_equal_training_weights(np.array([0]), np.array([1.]), calendar, cutoff="20260105")


def test_underflow_rejects_zero_weight_instead_of_erasing_old_samples():
    """A half-life too short for a long calendar fails instead of discarding old fit rows."""
    calendar = np.arange("2020-01-01", "2025-01-01", dtype="datetime64[D]")
    dates = [str(day).replace("-", "") for day in calendar]
    with pytest.raises(ValueError, match="nonpositive_training_weights"):
        daily_equal_training_weights(
            np.array([0]), np.array([0.]), dates, cutoff=dates[-1], half_life_sessions=1,
        )
