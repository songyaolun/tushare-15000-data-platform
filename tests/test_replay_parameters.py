"""Execution search rejects invalid inputs before reading any live data."""

from dataclasses import asdict

import pytest

from tushare_entitlement_hub.strategy_discovery.replay_parameters import ReplayParameters


def test_defaults_reproduce_frozen_execution_contract():
    """The default configuration exactly preserves the established baseline."""
    assert asdict(ReplayParameters()) == {
        "take_profit": .045, "stop_loss": .08, "maximum_holding": 30,
        "position_weight": .05, "replacement_sessions": 10, "candidate_count": 10,
    }


@pytest.mark.parametrize("field", ["take_profit", "stop_loss", "position_weight"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1, 1, 1.01, True])
def test_invalid_fraction_fails_closed(field, value):
    """Non-finite, non-positive, Boolean and levered fractions fail before replay."""
    with pytest.raises(ValueError, match="invalid_replay_fraction"):
        ReplayParameters(**{field: value})


@pytest.mark.parametrize("field", ["maximum_holding", "replacement_sessions"])
@pytest.mark.parametrize("value", [0, 31, 1.5, True])
def test_invalid_sessions_fail_closed(field, value):
    """Session counts must be actual integers within the preregistered bounds."""
    with pytest.raises(ValueError, match="invalid_replay_sessions"):
        ReplayParameters(**{field: value})


def test_valid_boundaries_and_single_parameter_change():
    """Valid extremes survive validation and a profit change preserves other controls."""
    assert ReplayParameters(.99, .99, 1, .4, 30).maximum_holding == 1
    parameters = ReplayParameters(take_profit=.09)
    assert parameters.take_profit == .09
    assert parameters.position_weight == .05


def test_position_weight_cannot_exceed_preserved_sector_limit():
    """The fixed 40% sector limit remains binding even without leverage."""
    with pytest.raises(ValueError, match="invalid_replay_sector_weight"):
        ReplayParameters(position_weight=.41)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "20", None])
def test_invalid_candidate_count_fails_before_replay(value):
    """Candidate limits reject nonpositive counts and non-integers before data loading."""
    with pytest.raises(ValueError, match="invalid_replay_candidate_count"):
        ReplayParameters(candidate_count=value)


@pytest.mark.parametrize("count", [1, 10, 20, 30])
def test_candidate_count_preserves_other_execution_controls(count):
    """A positive candidate count changes candidate supply without changing budgets or exits."""
    parameters = ReplayParameters(candidate_count=count)
    assert parameters.candidate_count == count
    assert parameters.position_weight == .05
    assert parameters.take_profit == .045
