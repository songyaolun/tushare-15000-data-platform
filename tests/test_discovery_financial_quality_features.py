"""Offline disclosure, report identity, cumulative TTM and unit-independent feature tests."""

from typing import Any

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.financial_quality_features import (
    candidate_report_values,
    earnings_yield,
    finite_number,
    quality_features,
    valid_date,
)

CALENDAR = ["20240425", "20240426", "20240429", "20240829", "20240830", "20240902"]


def report(period: str, announcement: str, value: Any, **changes: Any) -> dict[str, Any]:
    """Build a consolidated cumulative row; report('20231231','20240425',100) yields x=100."""
    return {
        "ts_code": "A", "end_date": period, "ann_date": announcement,
        "f_ann_date": announcement, "report_type": "1", "comp_type": "1",
        "update_flag": "0", "x": value, **changes,
    }


def candidates(records: list[dict[str, Any]], **options: Any) -> tuple[Any, Any, Any]:
    """Evaluate six candidates in original day order; no options means cumulative x TTM."""
    arguments = {
        "calendar": CALENDAR, "codes": ["A"], "candidate_days": np.arange(6),
        "candidate_securities": np.zeros(6, dtype=np.int64), "fields": ("x",),
        "cumulative": True,
    }
    arguments.update(options)
    return candidate_report_values(records, **arguments)


@pytest.mark.parametrize("value,expected", [(12, 12), (1.5, 1.5), (None, np.nan),
                                           ("12", np.nan), (True, np.nan), (np.inf, np.nan)])
def test_only_finite_numeric_values_are_facts(value: Any, expected: float) -> None:
    """Text, booleans, missing and infinite values stay unknown instead of being coerced."""
    np.testing.assert_equal(finite_number(value), expected)


@pytest.mark.parametrize("value,expected", [("20240229", True), ("20240230", False),
                                           ("20241301", False), (None, False),
                                           ("2024-01-01", False), ("aaaaaaaa", False)])
def test_dates_are_real_exact_calendar_dates(value: Any, expected: bool) -> None:
    """Invalid or non-eight-digit dates fail; a real leap day remains accepted."""
    assert valid_date(value) is expected


def test_disclosure_delay_ttm_and_future_prefix() -> None:
    """Report-day values are unavailable; H1 TTM uses three already visible cumulative rows."""
    records = [report("20230630", "20230829", 50), report("20231231", "20240425", 100),
               report("20240630", "20240829", 80)]
    values, reasons, audit = candidates(records)
    np.testing.assert_equal(values[:, 0], [np.nan, 100, 100, 100, 130, 130])
    assert reasons[0, 0] == 4  # Latest H1 before annual visibility lacks 2022 components.
    assert audit["disclosure_identities"] == 3
    future = records + [report("20240630", "20240902", 500)]
    np.testing.assert_equal(candidates(future)[0], values)
    prefix = candidates(records, calendar=CALENDAR[:5], candidate_days=np.arange(5),
                        candidate_securities=np.zeros(5, dtype=np.int64))[0]
    np.testing.assert_equal(prefix, values[:5])


def test_weekend_actual_announcement_and_unordered_candidate_axis() -> None:
    """Later actual disclosure controls availability; unsorted candidate rows stay unsorted."""
    rows = [report("20231231", "20240425", 100, f_ann_date="20240831")]
    values, _, _ = candidates(rows, candidate_days=np.array([5, 0, 4]),
                              candidate_securities=np.zeros(3, dtype=np.int64))
    np.testing.assert_equal(values[:, 0], [100, np.nan, np.nan])


def test_later_old_period_revision_cannot_replace_latest_report() -> None:
    """A disclosed revision updates a TTM component but cannot replace a newer report period."""
    rows = [report("20230630", "20230829", 50), report("20231231", "20240425", 100),
            report("20240630", "20240829", 80), report("20230630", "20240830", 60)]
    np.testing.assert_equal(candidates(rows)[0][:, 0], [np.nan, 100, 100, 100, 130, 120])


def test_latest_missing_and_same_day_conflict_never_fall_back_or_pick_update_flag() -> None:
    """A conflicting annual report is unknown regardless of update flag or input ordering."""
    rows = [report("20231231", "20240425", 100),
            report("20231231", "20240425", 101, update_flag="1")]
    values, reasons, audit = candidates(rows)
    assert np.isnan(values).all()
    assert reasons[-1, 0] == 3
    assert audit["field_x_conflicting_values"] == 1
    np.testing.assert_equal(candidates(list(reversed(rows)))[0], values)
    rows = [report("20231231", "20240425", 100), report("20240630", "20240829", None)]
    assert np.isnan(candidates(rows)[0][-1, 0])


def test_conflicts_are_field_specific_and_exact_duplicates_do_not_conflict() -> None:
    """One conflicted field does not erase another exact matching direct indicator value."""
    row = report("20231231", "20240425", 30, y=10)
    values, reasons, audit = candidates([row, row, {**row, "y": 11}],
                                       fields=("x", "y"), cumulative=False)
    np.testing.assert_equal(values[-1], [30, np.nan])
    np.testing.assert_equal(reasons[-1], [0, 3])
    assert audit["source_rows"] == 3


@pytest.mark.parametrize("changes,counter", [
    ({"ts_code": "OUT"}, "outside_candidate_security_axis"),
    ({"ann_date": "broken"}, "invalid_date"),
    ({"f_ann_date": "20240230"}, "invalid_date"),
    ({"end_date": "20231230"}, "invalid_period_or_premature_disclosure"),
    ({"ann_date": "20230101", "f_ann_date": "20230101"},
     "invalid_period_or_premature_disclosure"),
    ({"report_type": "2"}, "noncumulative_or_nonconsolidated_report"),
    ({"report_type": "6"}, "noncumulative_or_nonconsolidated_report"),
])
def test_bad_source_identity_is_audited(changes: dict[str, Any], counter: str) -> None:
    """Malformed, premature, single-quarter and parent-only reports never enter cumulative TTM."""
    values, reasons, audit = candidates([report("20231231", "20240425", 10, **changes)])
    assert np.isnan(values).all() and np.all(reasons == 1)
    assert audit[counter] == 1


def test_missing_optional_actual_date_and_missing_value() -> None:
    """An empty optional actual date uses announcement date; absent numeric values stay missing."""
    values, reasons, _ = candidates([report("20231231", "20240425", None, f_ann_date="")])
    assert np.isnan(values).all() and reasons[-1, 0] == 2
    values, reasons, _ = candidates([report("20231231", "20240425", 100, f_ann_date=None)])
    assert values[-1, 0] == 100 and reasons[-1, 0] == 0


def test_missing_and_finite_versions_are_not_silently_reconciled() -> None:
    """Same-date missing/finite variants are ambiguous and never choose the nonempty row."""
    values, reasons, _ = candidates([report("20231231", "20240425", 100),
                                     report("20231231", "20240425", None)])
    assert np.isnan(values[-1, 0]) and reasons[-1, 0] == 3


def test_financial_cash_and_mixed_company_types_fail_closed() -> None:
    """Financial companies are excluded only for cash; TTM cannot mix industrial and bank scope."""
    rows = [report("20231231", "20240425", 100, comp_type="2")]
    assert candidates(rows, industrial_only=True)[1][-1, 0] == 6
    assert candidates(rows)[0][-1, 0] == 100
    rows += [report("20230630", "20230829", 50), report("20240630", "20240829", 80)]
    assert candidates(rows)[1][-1, 0] == 5
    assert candidates([report("20231231", "20240425", 100, comp_type="")])[1][-1, 0] == 5
    rows = [report("20231231", "20240425", 100),
            report("20231231", "20240425", 100, comp_type="2")]
    assert candidates(rows)[1][-1, 0] == 5


def test_ttm_overflow_is_unknown() -> None:
    """Finite source amounts with overflowing TTM arithmetic produce unknown rather than inf."""
    rows = [report("20230630", "20230829", -1e308),
            report("20231231", "20240425", 1e308), report("20240630", "20240829", 1e308)]
    assert candidates(rows)[1][-1, 0] == 7


@pytest.mark.parametrize("options", [
    {"calendar": []}, {"codes": []}, {"fields": ()}, {"fields": ("x", "x")},
    {"calendar": list(reversed(CALENDAR))}, {"calendar": ["20240430", "20240431"]},
    {"codes": ["A", "A"]}, {"candidate_days": np.array([[0]])},
    {"candidate_securities": np.array([0])}, {"candidate_days": np.array([0.5] * 6)},
    {"candidate_securities": np.array([0.] * 6)}, {"candidate_days": np.array([-1] * 6)},
    {"candidate_days": np.array([6] * 6)}, {"candidate_securities": np.array([-1] * 6)},
    {"candidate_securities": np.array([1] * 6)}, {"cumulative": False, "industrial_only": True},
])
def test_invalid_axes_fail_before_mapping(options: dict[str, Any]) -> None:
    """Shape, range, unique/date and incompatible scope errors fail before any output is trusted."""
    with pytest.raises(ValueError, match="financial feature"):
        candidates([], **options)


def test_quality_formula_partial_missing_losses_and_negative_cash() -> None:
    """Percentage growth and cash backing preserve negative CFO and reject NP<=0."""
    indicators = np.array([[30, 10], [20, np.nan], [np.nan, 10], [0, 0], [10, 0.]])
    cash = np.array([[150, 100], [-100, 100], [100, 0], [100, -10], [np.nan, 100]])
    np.testing.assert_allclose(quality_features(indicators, cash),
                               [[.3, .2, .2], [.2, np.nan, -1], [np.nan, np.nan, np.nan],
                                [0, 0, np.nan], [.1, .1, np.nan]], equal_nan=True)


def test_quality_overflow_missing_and_shape_rejection() -> None:
    """Overflow never becomes a finite extreme; malformed N×2 input raises explicitly."""
    result = quality_features(np.array([[1e308, -1e308], [np.inf, 1.]]),
                              np.array([[1e308, 1e308], [1, np.inf]]))
    assert np.isnan(result[0, 1]) and np.isnan(result[1]).all()
    for indicators, cash in [(np.ones(2), np.ones(2)), (np.ones((1, 3)), np.ones((1, 3))),
                             (np.ones((1, 2)), np.ones((2, 2)))]:
        with pytest.raises(ValueError, match="quality"):
            quality_features(indicators, cash)


def test_earnings_yield_keeps_losses_zero_and_rejects_bad_capitalization() -> None:
    """Parent earnings yield keeps losses/zero; invalid capitalization remains unknown."""
    result = earnings_yield(np.array([-10, 0, 20, 1, 1, 1, np.inf, 1e308]),
                            np.array([100, 100, 0, -1, np.nan, np.inf, 100, 1e-308]))
    np.testing.assert_equal(result[:, 0], [-.1, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    for profit, cap in [(np.ones((1, 1)), np.ones((1, 1))), (np.ones(1), np.ones(2))]:
        with pytest.raises(ValueError, match="earnings yield"):
            earnings_yield(profit, cap)
