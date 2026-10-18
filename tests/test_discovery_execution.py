"""独立策略契约、共享账本与开盘成交的离线因果性测试。"""

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from unittest.mock import MagicMock

import numpy as np
import pytest
import test_regime_research

from tushare_entitlement_hub.regime_research.accounting import Distribution, Holding
from tushare_entitlement_hub.regime_research.panel import FIELDS, Panel
from tushare_entitlement_hub.regime_research.portfolio import (
    PortfolioPolicy,
    entry_candidates,
    simulate_portfolio,
)
from tushare_entitlement_hub.strategy_discovery import StrategyDefinition
from tushare_entitlement_hub.strategy_discovery.execution import (
    additional_exit_reason,
    entry_gap_allowed,
    intraday_fixed_exit,
    simulate_strategy,
    strategy_candidates,
)


@pytest.fixture
def legacy_panel():
    """输入无，复用旧测试的60日合成面板，输出与修改前快照相同的Panel。"""
    return test_regime_research.panel.__wrapped__()


@pytest.fixture
def discovery_panel():
    """输入无，输出8日12证券且全部满足共同资格的固定10元合成面板。"""
    shape = (8, 12)
    values = {name: np.ones(shape) for name in (*FIELDS, "beta", "capitalization")}
    for name in ("raw_open", "raw_close", "close", "raw_low", "raw_high"):
        values[name][:] = 10
    values["raw_up_limit"][:] = 11
    values["raw_down_limit"][:] = 9
    values["raw_amount"][:] = 20000
    values["sector_index"][:] = np.arange(12)
    values["capitalization"][:] = 100000
    signals = {"volume_reversal__sector_group": np.ones(shape, dtype=bool)}
    return Panel(
        [f"202401{day + 1:02d}" for day in range(8)],
        [f"{600000 + column}.SH" for column in range(12)],
        values,
        signals,
        {},
    )


def single_signal(panel, **changes):
    """输入面板，输出仅第0日第0证券入场的定义，可覆盖退出等字段。"""
    mask = np.zeros_like(panel.values["raw_close"], dtype=bool)
    mask[0, 0] = True
    return replace(StrategyDefinition("test", "independent", mask, np.ones(mask.shape)), **changes)


def run_strategy(panel, definition, events=None, stage=4):
    """输入策略和面板，输出从首日开始的独立账本结果；默认风险阶段仍可入场。"""
    return simulate_strategy(
        panel,
        np.full(len(panel.calendar), stage, dtype=np.int8),
        events or [],
        definition,
        start=panel.calendar[0],
    )


def test_contract_defaults_frozen_and_invalid_fields(discovery_panel):
    """默认仓位与固定字段符合约定，各种无效矩阵和配置应立即失败。"""
    definition = single_signal(discovery_panel)
    assert not definition.policy.switching
    assert not definition.policy.stage_allocation
    assert not definition.policy.force_rebalance
    assert definition.policy.position_weight == 0.05
    assert definition.policy.sector_weight == 0.25
    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"
    for changes in (
        {"name": ""},
        {"family": ""},
        {"entry_mask": np.zeros(8, dtype=bool)},
        {"entry_mask": np.zeros((8, 12))},
        {"scores": np.zeros((1, 1))},
        {"scores": np.ones((8, 12), dtype=int)},
        {"exit_mask": np.zeros((1, 1), dtype=bool)},
        {"exit_mask": np.zeros((8, 12))},
        {"policy": PortfolioPolicy()},
        {"policy": replace(definition.policy, stage_allocation=True)},
        {"policy": replace(definition.policy, force_rebalance=True)},
        {"trailing_stop": -0.1},
        {"trailing_stop": 1},
        {"trailing_stop": np.nan},
        {"intraday_exits": 1},
        {"maximum_entry_gap": -0.1},
        {"maximum_entry_gap": np.nan},
        {"maximum_entry_gap": np.inf},
    ):
        with pytest.raises(ValueError, match="invalid_strategy"):
            replace(definition, **changes)
    with pytest.raises(ValueError, match="shape"):
        run_strategy(
            discovery_panel,
            replace(definition, entry_mask=np.zeros((1, 1), dtype=bool), scores=np.ones((1, 1))),
        )


def test_candidates_common_eligibility_and_stable_score_ranking(discovery_panel):
    """每项共同资格独立排除，分数相同按证券列稳定排序，旧KDJ不影响排名。"""
    panel = discovery_panel
    definition = replace(single_signal(panel), entry_mask=np.ones((8, 12), dtype=bool))
    definition.scores[0] = [0, 0, 0, 0, 0, 0, np.nan, np.inf, 2, 3, 3, 1]
    panel.values["valid"][0, 0] = 0
    panel.values["beta"][0, 1] = np.nan
    panel.values["capitalization"][0, 2] = np.nan
    panel.values["sector_index"][0, 3] = -1
    panel.values["raw_amount"][0, 4] = 19999
    panel.values["raw_up_limit"][0, 5] = 10.5
    panel.values["raw_down_limit"][0, 5] = 9.5
    assert [row[0] for row in strategy_candidates(panel, 0, definition)] == [9, 10, 8, 11]
    panel.values["kdj_j"][:] = np.arange(12) * 1000
    assert [row[0] for row in strategy_candidates(panel, 0, definition)] == [9, 10, 8, 11]
    definition.entry_mask[:] = False
    assert strategy_candidates(panel, 0, definition) == []


@pytest.mark.parametrize("stage", range(5))
def test_trailing_in_every_stage_and_next_open_execution(discovery_panel, stage):
    """所有五个阶段均按复权峰值回撤退出，信号收盘之后的开盘才成交。"""
    panel = discovery_panel
    definition = single_signal(panel, trailing_stop=0.08)
    definition = replace(definition, policy=replace(definition.policy, take_profit=0.5))
    panel.values["raw_close"][2, 0] = 12
    panel.values["raw_close"][3, 0] = 11
    result = run_strategy(panel, definition, stage=stage)
    assert result["entry_days"] == [1]
    assert result["trades"][0]["exit"] == 4
    assert result["trades"][0]["reason"] == "trailing"
    assert abs(result["reconciliation_error"]) < 1e-8


@pytest.mark.parametrize(
    "closing,maximum_holding,expected",
    [(10, 20, "signal"), (8, 20, "stop"), (11, 20, "profit"), (10, 1, "time")],
)
def test_custom_exit_preserves_original_priority(
    discovery_panel, closing, maximum_holding, expected
):
    """额外退出与原规则同时命中时原规则优先，持仓当日触发仍须T+1卖出。"""
    panel = discovery_panel
    exits = np.zeros((8, 12), dtype=bool)
    exits[1, 0] = True
    definition = single_signal(panel, exit_mask=exits, trailing_stop=0.01)
    definition = replace(
        definition, policy=replace(definition.policy, maximum_holding=maximum_holding, trailing=0.3)
    )
    panel.values["raw_close"][1, 0] = closing
    result = run_strategy(panel, definition)
    assert result["trades"][0]["reason"] == expected
    assert result["trades"][0]["entry"] == 1
    assert result["trades"][0]["exit"] == 2


def test_extra_exit_handles_missing_close_and_no_signal(discovery_panel):
    """收盘缺失不伪造跟踪止损，明确退出信号仍可发单，无信号返回空原因。"""
    panel = discovery_panel
    definition = single_signal(panel, trailing_stop=0.05)
    holding = Holding(0, 0, 1, 100, 1000, 10, 10, 10, "independent", 4, 1)
    panel.values["raw_close"][2, 0] = np.nan
    assert additional_exit_reason(panel, holding, 2, definition) == ""
    exits = np.zeros((8, 12), dtype=bool)
    exits[2, 0] = True
    assert (
        additional_exit_reason(panel, holding, 2, replace(definition, exit_mask=exits)) == "signal"
    )


def test_entry_gap_uses_absolute_adjustment_and_cancels_orders(discovery_panel):
    """跨除权因子比较绝对复权价格，锚定close无影响，超过上限直接取消订单。"""
    panel = discovery_panel
    panel.values["raw_adj_factor"][0, 0] = 2
    panel.values["raw_adj_factor"][1:, 0] = 4
    panel.values["raw_open"][1, 0] = 5.1
    panel.values["raw_close"][1:, 0] = 5.1
    panel.values["close"][:] = 98765
    assert entry_gap_allowed(panel, 0, 1, 0, 0.03)
    assert run_strategy(panel, single_signal(panel, maximum_entry_gap=0.03))["entry_days"] == [1]
    panel.values["raw_open"][1, 0] = 5.2
    assert not entry_gap_allowed(panel, 0, 1, 0, 0.03)
    assert run_strategy(panel, single_signal(panel, maximum_entry_gap=0.03))["entry_days"] == []
    assert entry_gap_allowed(panel, 0, 1, 0, None)
    for bad in (np.nan, 0, -1):
        panel.values["raw_adj_factor"][1, 0] = bad
        assert not entry_gap_allowed(panel, 0, 1, 0, None)
    panel.values["raw_adj_factor"][1, 0] = 4
    for bad in (np.nan, 0, -1):
        panel.values["raw_close"][0, 0] = bad
        assert not entry_gap_allowed(panel, 0, 1, 0, 0.03)


def test_current_close_and_future_changes_cannot_rewrite_open_fill(discovery_panel):
    """修改当天收盘和有效标记不影响开盘成交，未来数据扰动不能回写历史。"""
    panel = discovery_panel
    definition = single_signal(panel, maximum_entry_gap=0.02)
    baseline = run_strategy(panel, definition)
    modified = deepcopy(panel)
    modified.values["raw_close"][1, 0] = 6
    modified.values["close"][1, 0] = 12345
    modified.values["valid"][1, 0] = 0
    changed = run_strategy(modified, definition)
    assert changed["entry_days"] == baseline["entry_days"] == [1]
    assert changed["trades"][0]["cost"] == baseline["open_cost"]
    assert changed["daily"][1]["open_exposure"] == baseline["daily"][1]["open_exposure"]
    assert changed["daily"][1]["open_nav"] == baseline["daily"][1]["open_nav"]
    assert baseline["daily"][0]["open_nav"] == baseline["capital"]
    assert 0 < baseline["daily"][1]["open_nav"] < baseline["capital"]
    future = deepcopy(panel)
    for name in ("raw_close", "raw_open", "close"):
        future.values[name][5:] *= 0.5
    definition.entry_mask[5:] = True
    future_result = run_strategy(future, definition)
    assert future_result["daily"][:5] == baseline["daily"][:5]


def test_open_filter_forbids_current_closing_data_reads(discovery_panel):
    """为成交日收盘访问设置会抛错的守卫，开盘过滤仍仅靠信号收盘和开盘完成。"""
    panel = discovery_panel
    original_close = panel.values["raw_close"]

    def read_signal_close(index):
        """只允许读取第0日确认价格；例(0,0)输出10，(1,0)抛AssertionError。"""
        assert index[0] == 0, "开盘过滤读取了成交日收盘"
        return original_close[index]

    guarded_close = MagicMock()
    guarded_close.__getitem__.side_effect = read_signal_close
    panel.values["raw_close"] = guarded_close
    for name in ("close", "valid"):
        panel.values[name] = MagicMock()
        panel.values[name].__getitem__.side_effect = AssertionError("开盘不应读取此字段")
    assert entry_gap_allowed(panel, 0, 1, 0, 0)
    assert entry_gap_allowed(panel, 0, 1, 0, None)


def test_hooks_preserve_limits_expiry_and_dividend_accounting(discovery_panel):
    """买单遇涨停延迟、三日过期，分红应收/支付沿用共享资金账本并正确核对。"""
    panel = discovery_panel
    definition = single_signal(panel)
    panel.values["raw_open"][1, 0] = 11
    result = run_strategy(panel, definition)
    assert result["entry_days"] == [2]
    panel.values["raw_open"][1:4, 0] = 11
    assert run_strategy(panel, definition)["entry_days"] == []
    panel.values["raw_open"][1:4, 0] = 10
    panel.values["raw_close"][3:, 0] = 9
    panel.values["raw_open"][3:, 0] = 9
    panel.values["raw_adj_factor"][3:, 0] = 10 / 9
    result = run_strategy(panel, definition, [Distribution(0, 2, 3, 5, 0, 1, 0)])
    assert result["daily"][3]["receivable"] == pytest.approx(4900 * 0.8)
    assert result["daily"][5]["receivable"] == 0
    assert abs(result["reconciliation_error"]) < 1e-8


@pytest.mark.parametrize("repeat_entry", [False, True])
def test_exit_signal_cancels_pending_without_same_close_reentry(discovery_panel, repeat_entry):
    """涨停等待期间明确退出会在收盘撤单，同日入场也不得重挂撤销的订单。"""
    panel = discovery_panel
    exits = np.zeros((8, 12), dtype=bool)
    exits[1, 0] = True
    definition = single_signal(panel, exit_mask=exits)
    definition.entry_mask[1, 0] = repeat_entry
    panel.values["raw_open"][1, 0] = panel.values["raw_up_limit"][1, 0]
    result = run_strategy(panel, definition)
    assert result["entry_days"] == []
    assert result["open_positions"] == 0


def test_pending_survives_non_rebalance_day_and_future_exit(discovery_panel):
    """非调仓日entry为False不撤旧买单，未来收盘退出不影响当前开盘成交。"""
    panel = discovery_panel
    exits = np.zeros((8, 12), dtype=bool)
    definition = single_signal(panel, exit_mask=exits)
    panel.values["raw_open"][1, 0] = panel.values["raw_up_limit"][1, 0]
    assert run_strategy(panel, definition)["entry_days"] == [2]
    exits[2, 0] = True
    changed = run_strategy(panel, definition)
    assert changed["entry_days"] == [2]
    assert changed["trades"][0]["exit"] == 3
    assert changed["trades"][0]["reason"] == "signal"


def test_pending_industry_change_obeys_current_industry_cap(discovery_panel):
    """待买证券转入已有五票行业后，新买预算按成交前一日统一行业核算至25%。"""
    panel = discovery_panel
    definition = single_signal(panel)
    definition.entry_mask[0, :6] = True
    panel.values["sector_index"][:, :5] = 1
    panel.values["sector_index"][0, 5] = 0
    panel.values["sector_index"][1:, 5] = 1
    panel.values["raw_open"][1, 5] = panel.values["raw_up_limit"][1, 5]
    result = run_strategy(panel, definition)
    assert result["entry_days"] == [1, 1, 1, 1, 1, 2]
    assert sum(result["daily"][2]["weights"].values()) <= 0.25
    assert result["daily"][2]["weights"][5] < 0.01
    # 成交当天才出现的行业变更不应回写当天开盘归属。
    panel.values["sector_index"][2:, 5] = 0
    changed = run_strategy(panel, definition)
    assert changed["daily"][2]["open_exposure"] == result["daily"][2]["open_exposure"]


@pytest.mark.parametrize("unknown_sector", [-1, np.nan, np.inf, -np.inf])
def test_pending_unknown_current_industry_fails_closed(discovery_panel, unknown_sector):
    """待买期间最新行业成为未知或非有限值，原行业已知也不得继续买入。"""
    panel = discovery_panel
    definition = single_signal(panel)
    panel.values["raw_open"][1, 0] = panel.values["raw_up_limit"][1, 0]
    panel.values["sector_index"][1:, 0] = unknown_sector
    assert run_strategy(panel, definition)["entry_days"] == []


def test_pending_budget_tracks_current_assets_after_drawdown(discovery_panel):
    """等待成交时账户因既有持仓下跌而缩水，新票仍最多按当前资产5%购买。"""
    panel = discovery_panel
    definition = single_signal(panel)
    definition.entry_mask[0, :6] = True
    panel.values["raw_open"][1, 5] = panel.values["raw_up_limit"][1, 5]
    panel.values["raw_open"][2:, :5] = 9.1
    panel.values["raw_close"][1:, :5] = 9.1
    result = run_strategy(panel, definition)
    assert result["entry_days"] == [1, 1, 1, 1, 1, 2]
    assert result["daily"][2]["weights"][5] <= definition.policy.position_weight


@pytest.mark.parametrize(
    "policy,expected",
    [
        (PortfolioPolicy(), "0cdcf48799204383d3257cd9d936ff51d9f48f6db389a12fc7e8f54b8eea462f"),
        (
            PortfolioPolicy(switching=False, stage_allocation=False),
            "1493a92b130505b385181cf74df8bd5da6ee6dbc36cf605dab34ddc3566eba8f",
        ),
    ],
)
def test_original_simulation_snapshot_and_explicit_default_hooks(legacy_panel, policy, expected):
    """新增open_nav外的历史合成账本快照一致，显式None与原候选函数不改变字段。"""
    states = np.full(len(legacy_panel.calendar), 2, dtype=np.int8)
    kwargs = {"start": legacy_panel.calendar[0]}
    baseline = simulate_portfolio(legacy_panel, states, [], policy, **kwargs)
    legacy_snapshot = deepcopy(baseline)
    for row in legacy_snapshot["daily"]:
        del row["open_nav"]
    assert (
        hashlib.sha256(json.dumps(legacy_snapshot, sort_keys=True).encode()).hexdigest() == expected
    )
    assert baseline == simulate_portfolio(
        legacy_panel,
        states,
        [],
        policy,
        candidate_provider=None,
        exit_provider=None,
        entry_filter=None,
        cancel_pending_provider=None,
        intraday_exit_provider=None,
        **kwargs,
    )
    assert baseline == simulate_portfolio(
        legacy_panel, states, [], policy, candidate_provider=entry_candidates, **kwargs
    )


@pytest.fixture
def intraday_position():
    """输入无，输出已持有一日且绝对复权成本20的持仓，因子2时原始成本10。"""
    return Holding(0, 0, 1, 100, 1000, 10, 20, 20, "intraday", 4, 1)


def set_intraday_bar(panel, day=2, *, opening=10, high=10, low=10, closing=10, factor=2):
    """设置宽限价带的完整合成日线；例因子2、开盘10对应绝对复权开盘20。"""
    for name, value in (
        ("raw_open", opening),
        ("raw_high", high),
        ("raw_low", low),
        ("raw_close", closing),
        ("raw_adj_factor", factor),
        ("raw_down_limit", 7),
        ("raw_up_limit", 13),
    ):
        panel.values[name][day, 0] = value


@pytest.mark.parametrize(
    "opening,high,low,closing,expected",
    [
        (8.8, 9, 8.5, 8.9, (8.8, "stop")),
        (10.6, 10.8, 8.8, 10, (10.6, "profit")),
        (10, 10.2, 8.9, 9.8, (9, "stop")),
        (10, 10.5, 9.5, 10.3, (10.4, "profit")),
        (10, 10.5, 8.9, 10, (9, "stop")),
        (9, 9.5, 8.9, 9, (9, "stop")),
        (10.4, 10.5, 10, 10.3, (10.4, "profit")),
        (10, 10.3, 9.1, 10, None),
    ],
)
def test_intraday_fixed_thresholds_and_conservative_double_touch(
    discovery_panel, intraday_position, opening, high, low, closing, expected
):
    """开盘越界按开盘，正常触线按阈值，双触先止损，无触线不成交。"""
    set_intraday_bar(discovery_panel, opening=opening, high=high, low=low, closing=closing)
    assert intraday_fixed_exit(discovery_panel, intraday_position, 2, PortfolioPolicy()) == expected


@pytest.mark.parametrize(
    "field",
    [
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_adj_factor",
        "raw_down_limit",
        "raw_up_limit",
    ],
)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0, -1])
def test_intraday_missing_or_invalid_market_fields_fail_closed(
    discovery_panel, intraday_position, field, invalid
):
    """任何OHLC、复权因子或限价字段缺失/非正/非有限，均不得伪造盘中成交。"""
    set_intraday_bar(discovery_panel, low=8.9)
    discovery_panel.values[field][2, 0] = invalid
    assert intraday_fixed_exit(discovery_panel, intraday_position, 2, PortfolioPolicy()) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"raw_down_limit": 13},
        {"raw_open": 14},
        {"raw_low": 10.1},
        {"raw_high": 9.9},
        {"raw_low": 6},
        {"raw_high": 14},
        {"raw_open": 7, "raw_low": 7},
    ],
)
def test_intraday_contradictory_quotes_and_opening_limit_down_fail_closed(
    discovery_panel, intraday_position, changes
):
    """矛盾OHLC或限价拒卖；即便随后开板，开盘跌停当日仍保守不成交。"""
    set_intraday_bar(discovery_panel)
    for field, value in changes.items():
        discovery_panel.values[field][2, 0] = value
    assert intraday_fixed_exit(discovery_panel, intraday_position, 2, PortfolioPolicy()) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"entry": 2},
        {"entry": 3},
        {"locked": 20},
        {"entry_adjusted": np.nan},
        {"entry_adjusted": 0},
    ],
)
def test_intraday_t_plus_one_locked_shares_and_invalid_entry_cost(
    discovery_panel, intraday_position, changes
):
    """当日新仓/未上市送股/无有效复权成本均不得参与盘中退出。"""
    set_intraday_bar(discovery_panel, low=8.9)
    position = replace(intraday_position, **changes)
    assert intraday_fixed_exit(discovery_panel, position, 2, PortfolioPolicy()) is None


def test_intraday_factor_scale_and_default_closed_bar_model(discovery_panel):
    """绝对复权成本跨因子变化仍正确触止损；关闭开关保留原完整结果。"""
    panel = discovery_panel
    definition = single_signal(panel)
    baseline = run_strategy(panel, definition)
    assert baseline == run_strategy(panel, replace(definition, intraday_exits=False))
    assert "execution_model" not in baseline
    # 买入日因子为4，次日因子变8、原价减半，盘中止损应以10.005/2为基础。
    panel.values["raw_adj_factor"][:] = 4
    set_intraday_bar(panel, opening=5, high=5.1, low=4.4, closing=5, factor=8)
    panel.values["raw_down_limit"][2, 0] = 4
    result = run_strategy(panel, replace(definition, intraday_exits=True))
    trade = result["trades"][0]
    reference = 10 * 1.0005 * 4 * (1 - definition.policy.stop_loss) / 8
    notional = reference * 0.9995 * trade["shares"]
    from tushare_entitlement_hub.regime_research.accounting import transaction_fee

    assert trade["exit"] == 2 and trade["reason"] == "stop"
    assert trade["proceeds"] == pytest.approx(
        notional - transaction_fee(notional, "20240103", True)
    )
    assert abs(result["reconciliation_error"]) < 1e-8
    assert result["execution_model"]["fixed_exits"] == "daily_ohlc_t_plus_one"


def test_intraday_entry_day_breach_does_not_queue_stale_fixed_exit(discovery_panel):
    """买入日触止损受T+1限制；翌日恢复后不执行陈旧收盘止损单，仍可按期限退出。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True)
    definition = replace(definition, policy=replace(definition.policy, maximum_holding=4))
    set_intraday_bar(panel, day=1, low=8, closing=8, factor=1)
    result = run_strategy(panel, definition)
    assert result["entry_days"] == [1]
    assert result["trades"][0]["exit"] == 5
    assert result["trades"][0]["reason"] == "time"


def test_intraday_double_touch_fees_open_metrics_and_no_sale_cash_reuse(discovery_panel):
    """盘中卖出不回流当日开盘预算；开盘资产在卖出前冻结，双触损失计入当天费用。"""
    panel = discovery_panel
    # 两组股票共享行业，使第二票次日开盘仍受已有持仓的行业额度限制。
    definition = single_signal(panel)
    definition.entry_mask[1, 1] = True
    policy = replace(definition.policy, position_weight=0.25, sector_weight=0.25)
    definition = replace(definition, policy=policy)
    panel.values["sector_index"][:, :2] = 0
    set_intraday_bar(panel, low=8.9, high=10.5, factor=1)
    baseline = run_strategy(panel, definition)
    intraday = run_strategy(panel, replace(definition, intraday_exits=True))
    assert intraday["entry_days"] == baseline["entry_days"]
    assert intraday["daily"][2]["open_nav"] == baseline["daily"][2]["open_nav"]
    assert intraday["daily"][2]["open_exposure"] == baseline["daily"][2]["open_exposure"]
    assert intraday["daily"][2]["exposure"] < intraday["daily"][2]["open_exposure"]
    assert intraday["daily"][2]["cost"] > baseline["daily"][2]["cost"]
    assert intraday["trades"][0]["exit"] == 2
    assert intraday["trades"][0]["reason"] == "stop"
    assert abs(intraday["reconciliation_error"]) < 1e-8


def test_intraday_structural_exit_retains_next_open_priority(discovery_panel):
    """前日结构退出在次日开盘先执行，当天盘中即使再触线也不能重复卖出。"""
    panel = discovery_panel
    exits = np.zeros((8, 12), dtype=bool)
    exits[1, 0] = True
    definition = single_signal(panel, intraday_exits=True, exit_mask=exits)
    set_intraday_bar(panel, low=8.9, factor=1)
    result = run_strategy(panel, definition)
    assert len(result["trades"]) == 1
    assert result["trades"][0]["exit"] == 2
    assert result["trades"][0]["reason"] == "signal"


def test_intraday_trailing_exit_remains_close_then_next_open(discovery_panel):
    """跟踪回撤仅看收盘，仍在下个开盘卖出，不使用当天低点提前触发。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True, trailing_stop=0.08)
    definition = replace(definition, policy=replace(definition.policy, take_profit=0.5))
    set_intraday_bar(panel, day=2, high=12, low=10, closing=12, factor=1)
    set_intraday_bar(panel, day=3, opening=12, high=12, low=10.5, closing=11, factor=1)
    result = run_strategy(panel, definition)
    assert result["trades"][0]["exit"] == 4
    assert result["trades"][0]["reason"] == "trailing"


def test_intraday_cash_dividend_and_stock_unlock_preserve_accounting(discovery_panel):
    """登记持仓保留现金股利，送股上市前整笔锁定，解锁后盘中卖出且资金对账。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True)
    # 第3日送股并锁定至第5日；报价等比除权，固定止损只在第4/5日触发。
    for day in range(3, 8):
        set_intraday_bar(panel, day=day, opening=8, high=8.2, low=8, closing=8, factor=1.25)
    panel.values["raw_low"][4:6, 0] = 7
    event = Distribution(0, 2, 3, 4, 5, 0.5, 0.2)
    result = run_strategy(panel, definition, [event])
    trade = result["trades"][0]
    assert trade["exit"] == 5
    assert trade["shares"] == pytest.approx(4900 * 1.2)
    assert trade["dividend"] == pytest.approx(4900 * 0.5 * 0.8)
    assert abs(result["reconciliation_error"]) < 1e-8


def test_intraday_future_bars_cannot_rewrite_past_accounting(discovery_panel):
    """未来日线触发强止损不得改变此前净值、开盘买入或历史平仓。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True)
    baseline = run_strategy(panel, definition)
    set_intraday_bar(panel, day=5, low=8, high=11, factor=1)
    changed = run_strategy(panel, definition)
    assert changed["daily"][:5] == baseline["daily"][:5]
    assert changed["entry_days"] == baseline["entry_days"]
    assert changed["trades"][0]["exit"] == 5


@pytest.mark.parametrize("gap_at_open", [False, True])
def test_intraday_profit_fills_at_threshold_or_gap_open(discovery_panel, gap_at_open):
    """盘中止盈按门槛，开盘已跳过止盈按开盘，两者均计原卖出滑点和费用。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True)
    opening = 10.8 if gap_at_open else 10
    set_intraday_bar(panel, opening=opening, high=11, low=10, closing=10.5, factor=1)
    result = run_strategy(panel, definition)
    trade = result["trades"][0]
    threshold = 10 * 1.0005 * (1 + definition.policy.take_profit)
    expected_notional = (opening if gap_at_open else threshold) * 0.9995 * trade["shares"]
    from tushare_entitlement_hub.regime_research.accounting import transaction_fee

    assert trade["exit"] == 2 and trade["reason"] == "profit"
    assert trade["proceeds"] == pytest.approx(
        expected_notional - transaction_fee(expected_notional, "20240103", True)
    )
    assert abs(result["reconciliation_error"]) < 1e-8


def test_intraday_open_limit_down_waits_and_sell_slippage_respects_lower_limit(discovery_panel):
    """开盘跌停即使盘中开板也整日不卖；以后触及跌停价止损时滑点不得越过跌停价。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True)
    set_intraday_bar(panel, opening=9, high=10, low=9, closing=10, factor=1)
    panel.values["raw_down_limit"][2, 0] = 9
    threshold = 10 * 1.0005 * (1 - definition.policy.stop_loss)
    set_intraday_bar(panel, day=3, low=threshold, factor=1)
    panel.values["raw_down_limit"][3, 0] = threshold
    result = run_strategy(panel, definition)
    trade = result["trades"][0]
    assert trade["exit"] == 3
    expected_notional = threshold * trade["shares"]
    from tushare_entitlement_hub.regime_research.accounting import transaction_fee

    assert trade["proceeds"] == pytest.approx(
        expected_notional - transaction_fee(expected_notional, "20240104", True)
    )


def test_intraday_missing_bar_does_not_create_fixed_close_exit(discovery_panel):
    """固定退出所需日线缺失时拒绝成交，不偷偷改成收盘触发的次开盘止损单。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True)
    set_intraday_bar(panel, low=8, closing=8, factor=1)
    panel.values["raw_high"][2, 0] = np.nan
    result = run_strategy(panel, definition)
    assert result["trades"] == []
    assert result["open_positions"] == 1


@pytest.mark.parametrize("opening,expected", [(8.8, (8.8, "stop")), (10.6, (10.6, "profit"))])
def test_intraday_open_gap_forbids_all_full_day_information_reads(
    discovery_panel, intraday_position, opening, expected
):
    """开盘跳空已触线时，仅凭开盘可用报价成交，不得读取任何全天行情或有效标记。"""
    panel = discovery_panel
    set_intraday_bar(panel, opening=opening)
    for name in ("raw_high", "raw_low", "raw_close", "valid", "raw_amount"):
        panel.values[name] = MagicMock()
        panel.values[name].__getitem__.side_effect = AssertionError("开盘跳空读取了全天信息")
    assert intraday_fixed_exit(panel, intraday_position, 2, PortfolioPolicy()) == expected


@pytest.mark.parametrize("closing", [np.nan, 20])
def test_intraday_gap_sale_is_unchanged_by_later_close(discovery_panel, closing):
    """开盘跳空止损已确定，后来收盘缺失或矛盾不得取消或改变该次卖出。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True)
    set_intraday_bar(panel, opening=8.8, high=9, low=8.5, closing=8.8, factor=1)
    baseline = run_strategy(panel, definition)
    panel.values["raw_close"][2, 0] = closing
    modified = run_strategy(panel, definition)
    assert baseline["trades"][0]["reason"] == "stop"
    assert baseline["trades"] == modified["trades"]
    assert baseline["daily"] == modified["daily"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("valid", 0),
        ("valid", np.nan),
        ("raw_amount", 0),
        ("raw_amount", -1),
        ("raw_amount", np.nan),
        ("raw_amount", np.inf),
    ],
)
def test_intraday_non_gap_suspended_placeholder_bar_cannot_trade(discovery_panel, field, value):
    """未跳空时停牌占位OHLC即使双触阈值，缺有效标记或正有限成交额仍不得卖出。"""
    panel = discovery_panel
    definition = single_signal(panel, intraday_exits=True)
    set_intraday_bar(panel, opening=10, high=10.5, low=8.9, closing=10, factor=1)
    panel.values[field][2, 0] = value
    result = run_strategy(panel, definition)
    assert result["trades"] == []
    assert result["open_positions"] == 1
