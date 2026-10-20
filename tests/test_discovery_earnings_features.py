"""预告版本、19:30截止与候选映射的离线回归测试。"""

from datetime import date

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.earnings_features import (
    candidate_forecast_features,
    forecast_event_features,
)


def disclosure(day="20260105", lower=10, upper=20, **changes):
    """构造公开预告fixture；例disclosure()为1月5日增幅10–20%的年报预告。"""
    return {
        "ts_code": "000001.SZ", "ann_date": day, "first_ann_date": "20260105",
        "end_date": "20251231", "p_change_min": lower, "p_change_max": upper, **changes,
    }


def test_versions_delay_revision_and_duplicates():
    """同期间两次公告隔日生效，复制旧行不增加版本，修正只取同期间差。"""
    original = disclosure()
    rows = [original, original, disclosure("20260108", 20, 30),
            disclosure("20260109", 50, 60, end_date="20260331")]
    events = forecast_event_features(rows)
    assert len(events) == 3
    assert events[0]["available_date"] == date(2026, 1, 6)
    assert np.isnan(events[0]["features"][2])
    np.testing.assert_allclose(events[1]["features"], [.25, .1, .1])
    assert np.isnan(events[2]["features"][2])


@pytest.mark.parametrize("lower,upper", [(None, 1), (2, 1), (float("inf"), 2), ("bad", 2)])
def test_bad_intervals_break_revision_chain(lower, upper):
    """缺失、倒置和非有限增幅保留未知事件，下一版不能跳过未知前值计算修正。"""
    events = forecast_event_features([
        disclosure(), disclosure("20260106", lower, upper), disclosure("20260107", 30, 40)
    ])
    assert np.isnan(events[1]["features"]).all()
    assert np.isnan(events[2]["features"][2])


def test_conflict_invalid_date_missing_key_and_late_first_announcement():
    """同日数值冲突关闭特征，坏日期缺键跳过，较晚首次公告只能延后生效。"""
    events = forecast_event_features([
        {}, disclosure("bad"), disclosure(), disclosure(lower=30),
        disclosure("20260109", 20, 30, first_ann_date="20260110")
    ])
    assert len(events) == 2 and events[0]["conflict"]
    assert np.isnan(events[0]["features"]).all()
    assert events[1]["available_date"] == date(2026, 1, 11)
    assert np.isnan(events[1]["features"][2])


def test_mapping_next_session_period_priority_expiry_and_prefix():
    """周五公告周一可见，较旧期间的后续修订不覆盖新期间；未来事件不改历史。"""
    rows = [disclosure("20260109"), disclosure("20260112", 50, 60, end_date="20260331"),
            disclosure("20260113", 100, 120)]
    calendar = ["20260109", "20260112", "20260113", "20260114", "20261231"]
    days = np.array([0, 1, 2, 3, 4, 3])
    securities = np.array([0, 0, 0, 0, 0, 1])
    features = candidate_forecast_features(
        forecast_event_features(rows), calendar, ["000001.SZ", "000002.SZ"], days, securities
    )
    assert np.isnan(features[[0, 4, 5]]).all()
    np.testing.assert_allclose(features[1, :2], [.15, .1])
    np.testing.assert_allclose(features[2:4, 0], [.55, .55])
    earlier = candidate_forecast_features(
        forecast_event_features(rows[:2]), calendar[:3], ["000001.SZ"], days[:3], securities[:3]
    )
    np.testing.assert_array_equal(earlier, features[:3])


@pytest.mark.parametrize("days,securities,calendar,age", [
    ([-1], [0], ["20260105"], 180), ([1], [0], ["20260105"], 180),
    ([0], [-1], ["20260105"], 180), ([0], [1], ["20260105"], 180),
    ([0], [], ["20260105"], 180), ([[0]], [[0]], ["20260105"], 180),
    ([0], [0], ["20260105", "20260105"], 180), ([0], [0], ["20260105"], 0),
    ([0.5], [0], ["20260105"], 180), ([0], [0.5], ["20260105"], 180),
])
def test_invalid_candidate_axis_fails_closed(days, securities, calendar, age):
    """越界、负索引、形状不符、重复交易日或非法有效期必须拒绝。"""
    with pytest.raises(ValueError, match="invalid_forecast_candidate_axis"):
        candidate_forecast_features([], calendar, ["000001.SZ"], np.array(days),
                                    np.array(securities), maximum_age_days=age)
