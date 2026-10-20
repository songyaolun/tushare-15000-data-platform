"""新量价与涨停策略的正负样本、前缀不变性和真实离线资金成交。"""

from datetime import date, timedelta

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery import limit_events, moving_support
from tushare_entitlement_hub.strategy_discovery.execution import simulate_strategy


def event_fixture():
    """构造80日单证券下降后回升序列；无输入，输出第48日三种托同时成立的面板。"""
    close = np.concatenate(
        (np.linspace(11, 10, 40), np.linspace(10, 11, 20), np.linspace(11, 10, 20))
    )[:, None]
    values = {
        "valid": np.ones_like(close),
        "close": close,
        "raw_close": close.copy(),
        "raw_open": close * 0.99,
        "raw_high": close * 1.01,
        "raw_low": close * 0.98,
        "raw_up_limit": close * 1.10,
        "raw_down_limit": close * 0.90,
        "raw_vol": close * 100,
        "raw_adj_factor": np.ones_like(close),
        "raw_amount": np.full_like(close, 100000),
        "amount_ratio": np.full_like(close, 2),
        "return1": np.full_like(close, 0.01),
        "return20": np.full_like(close, 0.05),
        "close_location": np.full_like(close, 0.8),
        "turnover": np.full_like(close, 5),
        "beta": np.ones_like(close),
        "capitalization": np.full_like(close, 100000),
        "sector_index": np.zeros_like(close),
    }
    calendar = [(date(2024, 1, 1) + timedelta(days=day)).strftime("%Y%m%d") for day in range(80)]
    return Panel(calendar, ["600000.SH"], values, {}, {})


def seal(panel, day, failed=False):
    """将指定日改为官方涨停或炸板；例如seal(panel,30)使第30日收盘等于上限。"""
    upper = panel.values["raw_up_limit"][day, 0]
    panel.values["raw_high"][day, 0] = upper
    panel.values["raw_close"][day, 0] = upper * (0.99 if failed else 1)


def test_support_positive_entries_and_exit_contracts():
    """场景：完整量价三叉在第48日形成；预期六策略仅该日发单，政策和退出独立。"""
    panel = event_fixture()
    definitions = moving_support.build_strategies(panel, panel.values)
    assert len(definitions) == 6
    for definition in definitions:
        assert np.flatnonzero(definition.entry_mask[:, 0]).tolist() == [48]
        assert definition.policy.allocations == (0.9,) * 5
        assert not definition.policy.switching and not definition.policy.force_rebalance
        assert definition.exit_mask[-1, 0]
        assert definition.maximum_entry_gap == 0.02
        assert definition.maximum_pending_sessions == 3
        assert np.isfinite(definition.scores[definition.entry_mask]).all()


@pytest.mark.parametrize(
    "field,value", [("valid", 0), ("raw_vol", 0), ("raw_adj_factor", 2), ("close", np.nan)]
)
def test_support_rejects_suspension_bad_volume_and_corporate_action(field, value):
    """场景：形态窗口出现停牌、缺价、无成交量或复权因子跳变；预期所有托失效。"""
    panel = event_fixture()
    panel.values[field][35, 0] = value
    assert all(
        not item.entry_mask.any() for item in moving_support.build_strategies(panel, panel.values)
    )


def test_price_and_volume_are_independent_sources():
    """场景：成交量恒定而价格形成托；预期只有价托发单，量托和双托均无信号。"""
    panel = event_fixture()
    panel.values["raw_vol"][:] = 1000
    for definition in moving_support.build_strategies(panel, panel.values):
        assert definition.entry_mask.any() == (definition.family == "price_support")


@pytest.mark.parametrize(
    "field,value",
    [("return1", 0), ("close_location", 0.5), ("close_location", 1.1), ("return1", np.nan)],
)
def test_support_quality_is_not_inferred_from_a_cross(field, value):
    """场景：金叉当天收阴或收盘弱、指标未知；预期技术交叉本身不强制产生买单。"""
    panel = event_fixture()
    panel.values[field][48] = value
    assert all(
        not item.entry_mask.any() for item in moving_support.build_strategies(panel, panel.values)
    )


def test_limit_first_second_and_failure_have_distinct_sources():
    """场景：第30日首板、31日二板、32日三板、45日炸板；预期事件不混淆。"""
    panel = event_fixture()
    for day in (30, 31, 32):
        seal(panel, day)
    seal(panel, 45, failed=True)
    expected = {"first_limit": [30], "second_limit": [31], "failed_limit": [45]}
    for definition in limit_events.build_strategies(panel, panel.values):
        assert np.flatnonzero(definition.entry_mask[:, 0]).tolist() == expected[definition.family]
        assert definition.maximum_pending_sessions == 1
        assert definition.maximum_entry_gap == 0.03
        assert definition.exit_mask is None


@pytest.mark.parametrize(
    "field,value", [("valid", 0), ("raw_up_limit", np.nan), ("raw_down_limit", 0), ("raw_low", -1)]
)
def test_limit_unknown_history_does_not_become_first_board(field, value):
    """场景：首板前十日中行情或官方限价未知；预期不能把未知历史当成未涨停。"""
    panel = event_fixture()
    seal(panel, 30)
    panel.values[field][25] = value
    assert all(
        not item.entry_mask.any() for item in limit_events.build_strategies(panel, panel.values)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("amount_ratio", 1),
        ("turnover", 0.5),
        ("turnover", 31),
        ("return20", np.nan),
        ("return1", -0.01),
        ("close_location", 0.4),
    ],
)
def test_limit_quality_filters_fail_closed(field, value):
    """场景：首板换手、成交额或收盘质量不足；预期不发入场信号。"""
    panel = event_fixture()
    seal(panel, 30)
    panel.values[field][30] = value
    assert all(
        not item.entry_mask.any() for item in limit_events.build_strategies(panel, panel.values)
    )


def test_locked_first_board_and_ten_day_recurrence_are_excluded():
    """场景：一字首板与十日内再次首板；预期两者不冒充可换手的首次涨停事件。"""
    panel = event_fixture()
    seal(panel, 25)
    seal(panel, 30)
    upper = panel.values["raw_up_limit"][25, 0]
    panel.values["raw_low"][25] = upper
    panel.values["raw_open"][25] = upper
    assert all(
        not item.entry_mask.any() for item in limit_events.build_strategies(panel, panel.values)
    )


@pytest.mark.parametrize("module", [limit_events, moving_support])
def test_builders_reject_missing_misaligned_and_handle_empty(module):
    """场景：字段缺失或日期错位；预期拒绝，零日历史则返回六个空定义。"""
    panel = event_fixture()
    incomplete = dict(panel.values)
    del incomplete[module.REQUIRED_FEATURES[0]]
    with pytest.raises(ValueError):
        module.build_strategies(panel, incomplete)
    incomplete = dict(panel.values)
    incomplete["valid"] = incomplete["valid"][:-1]
    with pytest.raises(ValueError):
        module.build_strategies(panel, incomplete)
    empty = Panel(
        [], panel.codes, {name: value[:0] for name, value in panel.values.items()}, {}, {}
    )
    assert all(
        item.entry_mask.shape == (0, 1) for item in module.build_strategies(empty, empty.values)
    )


@pytest.mark.parametrize("module", [limit_events, moving_support])
def test_builders_have_identical_prefix_without_future_data(module):
    """场景：截断样本或改写后段行情；预期全部入场、排序、退出及撤单历史不变。"""
    panel = event_fixture()
    seal(panel, 30)
    originals = module.build_strategies(panel, panel.values)
    prefix = Panel(
        panel.calendar[:55],
        panel.codes,
        {key: value[:55] for key, value in panel.values.items()},
        {},
        {},
    )
    shortened = module.build_strategies(prefix, prefix.values)
    changed_values = {key: value.copy() for key, value in panel.values.items()}
    for value in changed_values.values():
        value[55:] = np.nan
    changed = module.build_strategies(panel, changed_values)
    for original, shorter, altered in zip(originals, shortened, changed, strict=True):
        for field in ("entry_mask", "scores", "exit_mask", "entry_invalidation_mask"):
            value = getattr(original, field)
            if value is not None:
                np.testing.assert_array_equal(value[:55], getattr(shorter, field))
                np.testing.assert_array_equal(value[:55], getattr(altered, field)[:55])


@pytest.mark.parametrize("module", [limit_events, moving_support])
def test_every_new_configuration_executes_in_real_offline_account(module):
    """场景：无网络的完整形态与报价；预期全部12配置产生次日买单和T+1之后的平仓。"""
    panel = event_fixture()
    for day in (30, 31):
        seal(panel, day)
    seal(panel, 45, failed=True)
    for definition in module.build_strategies(panel, panel.values):
        result = simulate_strategy(
            panel, np.zeros(80, dtype=np.int8), [], definition, start=panel.calendar[0]
        )
        assert result["trades"], definition.name
        for trade in result["trades"]:
            assert trade["entry"] == trade["confirmation"] + 1
            assert trade["exit"] > trade["entry"]
            assert trade["cost"] > 0
