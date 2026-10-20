"""价格突破三臂的结构等价、首次触碰、共同事件时钟与因果性测试。"""

from dataclasses import replace

import numpy as np
import pytest

from tushare_entitlement_hub.equity_research.right_side_policy import RightSidePolicy
from tushare_entitlement_hub.equity_research.right_side_structure import structure_at
from tushare_entitlement_hub.regime_research.accounting import Holding
from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery import breakout_retest as retest
from tushare_entitlement_hub.strategy_discovery.execution import additional_exit_reason


def breakout_fixture(days=70):
    """构造历史峰13、回调低8.9、整理上沿10/下沿9.7，60日10.1放量突破。

    输入70日，输出单证券panel/features；61/62日首次和再次触碰均恢复，源风险约4.44%。
    """
    shape = (days, 1)
    close = np.full(shape, 11.0)
    high, low = close + 0.1, close - 0.1
    high[0] = 13
    close[40:50], high[40:50], low[40:50] = 9.3, 9.4, 9.2
    close[40], high[40], low[40] = 9, 9.1, 8.9
    close[50:60], high[50:60], low[50:60] = 9.8, 10, 9.7
    close[60:], high[60:], low[60:] = 10.2, 10.3, 10.11
    close[60], high[60], low[60] = 10.1, 10.2, 10
    if days > 61:
        close[61], high[61], low[61] = 10.15, 10.2, 10.02
    if days > 62:
        close[62], high[62], low[62] = 10.18, 10.25, 10.01
    amount = np.full(shape, 100.0)
    amount[60] = 150
    amount[61:] = 90
    features = {
        "valid": np.ones(shape),
        "close": close,
        "high": high,
        "low": low,
        "raw_amount": amount,
        "ma20": close * 0.99,
        "return1": np.full(shape, 0.001),
        "sector_index": np.zeros(shape),
    }
    panel = Panel([f"day{day:03}" for day in range(days)], ["000001.SZ"], features, {}, {})
    return panel, features


def test_three_arms_share_frozen_source_score_and_execution_policy():
    """场景：60突破、61首触、62再次有效触碰；预期：三臂仅入场日期不同且共享风险/退出阵。"""
    panel, features = breakout_fixture()
    direct, first, repeated = retest.build_strategies(panel, features)
    assert [item.name for item in (direct, first, repeated)] == list(retest.STRATEGY_NAMES)
    assert np.flatnonzero(direct.entry_mask[:, 0]).tolist() == [60]
    assert np.flatnonzero(first.entry_mask[:, 0]).tolist() == [61]
    assert np.flatnonzero(repeated.entry_mask[:, 0]).tolist() == [61, 62]
    for item in (direct, first, repeated):
        assert item.scores is direct.scores
        assert item.exit_mask is direct.exit_mask
        assert item.entry_invalidation_mask is direct.entry_invalidation_mask
        assert replace(item.policy, name=direct.policy.name) == direct.policy
        assert (item.policy.take_profit, item.policy.stop_loss, item.policy.maximum_holding) == (
            0.08,
            0.08,
            30,
        )
        assert item.maximum_entry_gap == 0.02
        assert item.policy.allocations == (0.9,) * 5
        assert item.policy.position_weight == 0.05 and item.policy.sector_weight == 0.25
        assert (
            item.profit_protection is None and not item.intraday_exits and item.trailing_stop == 0
        )
    assert direct.scores[60, 0] == direct.scores[61, 0] == direct.scores[62, 0]
    event = retest.diagnose_events(panel, features)[0]
    assert event["source_day"] == 60 and event["pivot"] == 10
    assert event["support"] == 9.7
    assert event["first_confirmation_day"] == 61
    assert event["repeat_confirmation_days"] == [61, 62]
    assert event["status"] == "window_expired_after_retest"
    assert event["rearm_not_before_day"] == 69
    assert "return" not in event


def test_cheap_gate_never_excludes_exact_product_structure_sources():
    """场景：逐日纯结构检查与必要条件预筛；预期：所有精确合格日必在预筛内且锚值一致。"""
    panel, features = breakout_fixture()
    valid = retest._valid_quotes(panel, features)
    candidates = retest._source_candidates(features, valid)
    exact = []
    for day in range(60, len(panel.calendar)):
        bars = tuple(retest._bar(panel, features, prior, 0) for prior in range(day - 60, day + 1))
        observation = structure_at(bars, 60, RightSidePolicy())
        if observation.reason is None:
            exact.append(day)
            assert candidates[day, 0]
    assert exact == [60]
    event = retest.diagnose_events(panel, features)[0]
    bars = tuple(retest._bar(panel, features, day, 0) for day in range(61))
    observation = structure_at(bars, 60, RightSidePolicy())
    assert event["source_score"] == observation.values["structural_reward_risk"]
    assert event["pivot"] == observation.values["confirmation_price"]
    assert event["invalidation_price"] == observation.values["invalidation_price"]


def test_candidate_that_fails_original_risk_gate_does_not_become_source():
    """场景：突破金额充分但收盘追离超过3%；预期：cheap gate通过，原风险门仍拒绝事件。"""
    panel, features = breakout_fixture()
    features["close"][60], features["high"][60] = 10.4, 10.5
    assert retest._source_candidates(features, retest._valid_quotes(panel, features))[60, 0]
    assert retest.diagnose_events(panel, features) == []


@pytest.mark.parametrize("mutation", ["not_rising", "amount", "risk_position"])
def test_first_touch_failure_cannot_be_replaced_by_later_success(mutation):
    """场景：首触未恢复、金额超源或价格风险不合格，次日看似成功；预期：整个源注销。"""
    panel, features = breakout_fixture()
    if mutation == "not_rising":
        features["close"][61] = 10.05
    elif mutation == "amount":
        features["raw_amount"][61] = 151
    else:
        features["close"][61], features["high"][61] = 10.31, 10.4
    direct, first, repeated = retest.build_strategies(panel, features)
    assert direct.entry_mask[60, 0]
    assert not first.entry_mask.any() and not repeated.entry_mask.any()
    event = retest.diagnose_events(panel, features)[0]
    assert event["status"] == "first_touch_failed"
    assert event["first_touch_day"] == 61 and event["end_day"] == 61
    assert event["rearm_not_before_day"] == 69


@pytest.mark.parametrize(("field", "value"), [("low", 9.8), ("close", 9.99)])
def test_structure_failure_cancels_all_arms_without_selling_existing_positions(field, value):
    """场景：源后守位失败；预期：共同撤单阵为真，但既有持仓不因这个标记产生卖出。"""
    panel, features = breakout_fixture()
    features[field][61] = value
    if field == "close":
        features["low"][61] = 9.99
    definitions = retest.build_strategies(panel, features)
    event = retest.diagnose_events(panel, features)[0]
    assert event["status"] == "structure_failed"
    panel.values = features | {
        "raw_close": features["close"],
        "raw_adj_factor": np.ones_like(features["close"]),
    }
    holding = Holding(0, 60, 61, 100, 1010, 10.1, 10.1, 10.1, "breakout_retest", 0, 1)
    for definition in definitions:
        assert definition.entry_invalidation_mask[61, 0]
        assert not definition.exit_mask[61, 0]
        assert additional_exit_reason(panel, holding, 61, definition) == ""


def test_no_retest_remains_in_diagnostics_without_a_zero_return():
    """场景：突破后五日始终未触及源位；预期：direct保留、回踩臂无候选，诊断不捏造收益。"""
    panel, features = breakout_fixture()
    features["low"][61:] = 10.11
    definitions = retest.build_strategies(panel, features)
    assert definitions[0].entry_mask[60, 0]
    assert not definitions[1].entry_mask.any() and not definitions[2].entry_mask.any()
    event = retest.diagnose_events(panel, features)[0]
    assert event["status"] == "window_expired_without_touch"
    assert event["first_touch_day"] is None and event["end_day"] == 65
    assert not any("return" in key for key in event)


def test_last_window_day_can_confirm_but_tail_days_cannot_generate_new_signals():
    """场景：第5日才首触，第6日再次触碰；预期：仅第5日确认，尾部只监控待单。"""
    panel, features = breakout_fixture()
    features["low"][61:65] = 10.11
    features["low"][65:67] = 10.05
    features["close"][65:67] = np.array([[10.22], [10.25]])
    direct, first, repeated = retest.build_strategies(panel, features)
    assert direct.entry_mask[60, 0]
    assert np.flatnonzero(first.entry_mask[:, 0]).tolist() == [65]
    assert np.flatnonzero(repeated.entry_mask[:, 0]).tolist() == [65]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("valid", 0, "invalid_observation"),
        ("low", 9.8, "structure_failed"),
    ],
)
def test_tail_failure_cancels_old_order_without_starting_another_event(field, value, reason):
    """场景：五日确认窗口结束后旧挂单尾部失败；预期：保留窗口状态、单记尾部失败并撤单。"""
    panel, features = breakout_fixture()
    features[field][66] = value
    definitions = retest.build_strategies(panel, features)
    event = retest.diagnose_events(panel, features)[0]
    assert event["status"] == "window_expired_after_retest"
    assert event["tail_failure"] == {"day": 66, "reason": reason}
    assert definitions[0].entry_invalidation_mask[66, 0]
    assert not definitions[0].exit_mask[66, 0]
    assert not definitions[1].entry_mask[66:69].any()


@pytest.mark.parametrize("early_failure", [False, True])
def test_all_arms_reserve_same_five_plus_three_lifecycle_before_rearming(early_failure):
    """场景：66日出现合格新结构，69日再次出现；预期：不论旧事件早败与否最早69日重启。"""
    panel, features = breakout_fixture(72)
    # 69日的61日窗口仍保留阻力13，确保新结构通过原空间风险比，而非只通过cheap gate。
    features["high"][9] = 13
    if early_failure:
        features["raw_amount"][61] = 151
    features["close"][66], features["high"][66], features["raw_amount"][66] = 10.32, 10.35, 200
    features["close"][69], features["high"][69], features["low"][69] = 10.45, 10.5, 10.35
    features["raw_amount"][69] = 250
    candidate = retest._source_candidates(features, retest._valid_quotes(panel, features))
    assert candidate[66, 0] and candidate[69, 0]
    for source_day in (66, 69):
        bars = tuple(
            retest._bar(panel, features, day, 0) for day in range(source_day - 60, source_day + 1)
        )
        assert structure_at(bars, 60, RightSidePolicy()).reason is None
    events = retest.diagnose_events(panel, features)
    assert [event["source_day"] for event in events] == [60, 69]
    direct = retest.build_strategies(panel, features)[0]
    assert np.flatnonzero(direct.entry_mask[:, 0]).tolist() == [60, 69]


def test_later_unconfirmed_touch_does_not_rewrite_successful_first_touch():
    """场景：首触成功、第二触碰收盘未涨、第三次恢复；预期：first只一次，repeat接受后续有效触碰。"""
    panel, features = breakout_fixture()
    features["close"][62] = 10.14
    features["low"][63], features["close"][63] = 10.05, 10.19
    definitions = retest.build_strategies(panel, features)
    assert np.flatnonzero(definitions[1].entry_mask[:, 0]).tolist() == [61]
    assert np.flatnonzero(definitions[2].entry_mask[:, 0]).tolist() == [61, 63]
    touches = retest.diagnose_events(panel, features)[0]["touches"]
    assert [touch["confirmed"] for touch in touches] == [True, False, True]


@pytest.mark.parametrize(("field", "value"), [("valid", 0), ("close", np.nan), ("raw_amount", 0)])
def test_suspension_or_missing_quote_during_event_cancels_it(field, value):
    """场景：源后缺价、停牌或零额；预期：事件注销，未来恢复不得复活该源。"""
    panel, features = breakout_fixture()
    features[field][61] = value
    definitions = retest.build_strategies(panel, features)
    assert not definitions[1].entry_mask.any() and not definitions[2].entry_mask.any()
    assert retest.diagnose_events(panel, features)[0]["status"] == "invalid_observation"


def test_historical_missing_day_prevents_sixty_one_market_day_source():
    """场景：源窗口内停牌但供应商仍有占位价格；预期：不能压缩为61个观察日确认突破。"""
    panel, features = breakout_fixture()
    features["valid"][10] = 0
    assert retest.diagnose_events(panel, features) == []


def test_industry_veto_does_not_remove_price_source_from_opportunity_diagnostics(monkeypatch):
    """场景：价格突破合格但行业广度弱；预期：三臂共同否决，源事件仍完整记录且非市场50%门。"""
    panel, features = breakout_fixture()
    shape = features["close"].shape
    monkeypatch.setattr(
        retest,
        "_sector_features",
        lambda supplied: {
            "breadth": np.full(shape, 0.3),
            "momentum20": np.full(shape, 0.1),
        },
    )
    direct, first, repeated = retest.build_strategies(panel, features)
    assert direct.entry_mask[60, 0] and first.entry_mask[61, 0] and repeated.entry_mask[62, 0]
    assert direct.entry_invalidation_mask[60:63].all()
    assert not direct.exit_mask.any()
    assert retest.diagnose_events(panel, features)[0]["source_industry_blocked"]
    monkeypatch.setattr(
        retest,
        "_sector_features",
        lambda supplied: {
            "breadth": np.full(shape, 0.7),
            "momentum20": np.full(shape, -0.01),
        },
    )
    assert retest.build_strategies(panel, features)[0].exit_mask.all()


def test_future_perturbation_and_prefix_leave_historical_masks_scores_and_source_anchors_equal():
    """场景：64日后行情改写或截断；预期：此前信号/撤单/评分和源锚不变，截尾状态保留未知。"""
    panel, features = breakout_fixture()
    original = retest.build_strategies(panel, features)
    changed_features = {name: values.copy() for name, values in features.items()}
    for values in changed_features.values():
        values[64:] = 999
    changed = retest.build_strategies(panel, changed_features)
    prefix_features = {name: values[:64] for name, values in features.items()}
    prefix_panel = Panel(panel.calendar[:64], panel.codes, prefix_features, {}, {})
    prefix = retest.build_strategies(prefix_panel, prefix_features)
    for before, after, shortened in zip(original, changed, prefix, strict=True):
        for field in ("entry_mask", "scores", "exit_mask", "entry_invalidation_mask"):
            np.testing.assert_array_equal(getattr(before, field)[:64], getattr(after, field)[:64])
            np.testing.assert_array_equal(getattr(before, field)[:64], getattr(shortened, field))
    original_event = retest.diagnose_events(panel, features)[0]
    shorter_event = retest.diagnose_events(prefix_panel, prefix_features)[0]
    assert shorter_event["status"] == "right_censored"
    for field in ("source_day", "pivot", "support", "historical_resistance", "source_score"):
        assert original_event[field] == shorter_event[field]


def test_empty_panel_and_source_at_last_day_have_no_wrapping():
    """场景：空矩阵与末日才源突破；预期：空结果及未确认截尾，不从首日绕读未来行情。"""
    panel, features = breakout_fixture(61)
    definitions = retest.build_strategies(panel, features)
    assert definitions[0].entry_mask[60, 0]
    assert not definitions[1].entry_mask.any()
    assert retest.diagnose_events(panel, features)[0]["status"] == "right_censored"
    empty = {name: values[:0, :0] for name, values in features.items()}
    empty_panel = Panel([], [], empty, {}, {})
    assert retest.diagnose_events(empty_panel, empty) == []
    assert all(
        item.entry_mask.shape == (0, 0) for item in retest.build_strategies(empty_panel, empty)
    )


@pytest.mark.parametrize(
    ("field", "value"), [("high", 0), ("low", np.inf), ("low", 20), ("close", 30)]
)
def test_invalid_ohlc_never_qualifies_source(field, value):
    """场景：非正/非有限或互相矛盾的OHLC；预期：不构造源事件。"""
    panel, features = breakout_fixture()
    features[field][60] = value
    assert retest.diagnose_events(panel, features) == []


@pytest.mark.parametrize("mutation", ["missing", "shape", "calendar"])
def test_invalid_matrix_or_calendar_contract_fails(mutation):
    """场景：缺必需字段、矩阵错轴或重复市场日；预期：明确ValueError。"""
    panel, features = breakout_fixture()
    if mutation == "missing":
        del features["raw_amount"]
    elif mutation == "shape":
        features["close"] = np.ones((1, 70))
    else:
        panel.calendar[1] = panel.calendar[0]
    with pytest.raises(ValueError, match="breakout_retest_features_or_calendar_invalid"):
        retest.build_strategies(panel, features)


def test_input_arrays_remain_unchanged():
    """场景：生成三臂及诊断；预期：所有输入数组保持原值，诊断不写入行情对象。"""
    panel, features = breakout_fixture()
    before = {name: values.copy() for name, values in features.items()}
    retest.build_strategies(panel, features)
    retest.diagnose_events(panel, features)
    for name, values in features.items():
        np.testing.assert_array_equal(values, before[name])
