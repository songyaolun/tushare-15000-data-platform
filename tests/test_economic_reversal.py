"""固定短反实验的历史股票池、同行业匹配、延迟时点和因果性测试。"""

from copy import deepcopy
from dataclasses import fields, replace
from datetime import date, timedelta

import numpy as np
import pytest

from tushare_entitlement_hub.economic_research import reversal
from tushare_entitlement_hub.regime_research.panel import FIELDS, Panel
from tushare_entitlement_hub.regime_research.portfolio import simulate_portfolio


def _panel(days=35, securities=10):
    """生成完整历史沪深面板；例如第20日十股出现不同五日收益，前两股最弱。"""
    shape = days, securities
    values = {name: np.ones(shape) for name in (*FIELDS, "beta", "capitalization")}
    for name in ("raw_open", "raw_close", "raw_low", "raw_high", "close"):
        values[name][:] = 10
    values["capitalization"][:] = 100000
    values["raw_amount"][:] = 30000
    values["raw_up_limit"][:] = 11
    values["raw_down_limit"][:] = 9
    values["sector_index"][:] = np.arange(securities) // 5
    if securities == 10 and days > 20:
        changes = np.array([-0.06, -0.04, 0, 0.03, 0.04, -0.02, -0.01, 0, 0.01, 0.02])
        values["close"][20:] *= 1 + changes
    return Panel(
        [(date(2020, 1, 1) + timedelta(days=day)).strftime("%Y%m%d") for day in range(days)],
        [f"{column + 1:06d}.SZ" for column in range(securities)],
        values,
        {},
        {},
    )


def test_frozen_universe_relative_mean_selection_and_four_policies():
    """同业均值含自身、最差负收益20%入选；四组只作预声明变更，固定价格退出关闭。"""
    panel = _panel()
    features = reversal.build_features(panel)
    assert not features.eligibility[:20].any()
    assert features.universe[20].all()
    np.testing.assert_allclose(features.return5[20, :5], [-0.06, -0.04, 0, 0.03, 0.04])
    np.testing.assert_allclose(
        features.relative_return5[20, :5], [-0.054, -0.034, 0.006, 0.036, 0.046]
    )
    assert np.flatnonzero(features.entry_mask[20]).tolist() == [0, 1]
    strategies = reversal.build_strategies(panel, features)
    assert [item.name for item in strategies] == [
        "reversal",
        "delayed",
        "cost_stress",
        "matched_control",
    ]
    for strategy in strategies:
        assert strategy.policy.maximum_holding == 5
        assert not strategy.policy.fixed_price_exits
        assert not strategy.policy.switching and not strategy.policy.stage_allocation
        assert not strategy.policy.force_rebalance
        assert strategy.policy.position_weight == 0.05 and strategy.policy.sector_weight == 0.25
        assert strategy.policy.non_stage_allocation == 0.9
        assert strategy.policy.early_exit == strategy.policy.trailing == 0
        assert strategy.maximum_pending_sessions == 1
    assert strategies[2].entry_mask is strategies[0].entry_mask
    assert strategies[2].scores is strategies[0].scores
    assert strategies[0].policy.slippage_rate == 0.0005 and strategies[0].policy.fee_multiplier == 1
    assert strategies[2].policy.slippage_rate == 0.002 and strategies[2].policy.fee_multiplier == 2
    np.testing.assert_equal(strategies[0].source_days, np.arange(35))
    assert strategies[0].source_days.dtype == np.int64


@pytest.mark.parametrize(
    "field,value",
    [
        ("valid", 0),
        ("close", 0),
        ("close", np.nan),
        ("beta", np.nan),
        ("capitalization", 0),
        ("capitalization", np.inf),
        ("raw_amount", 19999),
        ("raw_amount", np.nan),
        ("raw_up_limit", 10.7),
        ("raw_down_limit", 0),
        ("sector_index", np.nan),
        ("sector_index", np.inf),
        ("sector_index", -1),
        ("sector_index", 0.5),
    ],
)
def test_common_eligibility_rejects_invalid_inputs(field, value):
    """任何历史资格缺失、量额不足、官方限价带不达标或未知行业均不占池名额。"""
    panel = _panel()
    if field == "raw_up_limit":
        panel.values["raw_down_limit"][20, 0] = 9.3
    panel.values[field][20, 0] = value
    features = reversal.build_features(panel)
    assert not features.eligibility[20, 0]
    assert not features.large_cap_pool[20, 0] and not features.universe[20, 0]


def test_five_market_day_return_and_complete_twenty_one_day_history():
    """5日收益直接用复权close而非旧return5，任一21日内停牌都暂停候选。"""
    panel = _panel()
    panel.values["return5"][:] = -999
    expected = panel.values["close"][20, 0] / panel.values["close"][15, 0] - 1
    assert reversal.build_features(panel).return5[20, 0] == pytest.approx(expected)
    panel.values["valid"][3, 0] = 0
    features = reversal.build_features(panel)
    assert not features.eligibility[20:24, 0].any()
    assert features.eligibility[24, 0]
    panel.codes[0] = "920001.BJ"
    assert not reversal.build_features(panel).eligibility[:, 0].any()


def test_average_amount_requires_twenty_positive_known_market_days():
    """均额要求完整20日，含当日、缺失中断且窗口离开缺失后恢复，未来值不回写。"""
    values = np.full((42, 3), 20000.0)
    values[5, 0] = np.nan
    values[5, 1] = 0
    values[5, 2] = np.inf
    average = reversal._average_amount20(values)
    assert np.isnan(average[:25]).all()
    np.testing.assert_equal(average[25:], 20000)
    np.testing.assert_equal(reversal._average_amount20(values[:30]), average[:30])


def test_top_thousand_market_cap_then_top_five_hundred_liquidity_are_historical():
    """先市值后流动性的两阶段筛选独立成立，最高流动性但市值1001名不能偷渡。"""
    panel = _panel(days=22, securities=1002)
    panel.values["sector_index"][:] = 0
    panel.values["capitalization"][:] = np.arange(1002, 0, -1)
    panel.values["raw_amount"][:] = 20000 + np.arange(1002)
    features = reversal.build_features(panel)
    assert np.flatnonzero(features.large_cap_pool[20]).tolist() == list(range(1000))
    assert np.flatnonzero(features.universe[20]).tolist() == list(range(500, 1000))
    panel.values["capitalization"][21, 1001] = 1000000
    changed = reversal.build_features(panel)
    np.testing.assert_equal(changed.universe[:21], features.universe[:21])
    assert changed.large_cap_pool[21, 1001] and changed.universe[21, 1001]


def test_unknown_industry_cannot_displace_known_names_at_pool_cutoff():
    """最高市值但行业未知的证券在基础资格阶段剔除，不能挤占有效前1000或前500名额。"""
    panel = _panel(days=21, securities=1001)
    panel.values["capitalization"][:] = np.arange(1001, 0, -1)
    panel.values["sector_index"][:] = 0
    panel.values["sector_index"][:, 0] = np.nan
    features = reversal.build_features(panel)
    assert np.flatnonzero(features.large_cap_pool[20]).tolist() == list(range(1, 1001))
    assert np.flatnonzero(features.universe[20]).tolist() == list(range(1, 501))


def test_same_value_ranking_ties_and_negative_only_quota():
    """市值/流动性/相对收益并列按列号，池少于5无名额，负值不足时不补正值。"""
    panel = _panel(days=21, securities=1001)
    features = reversal.build_features(panel)
    assert np.flatnonzero(features.large_cap_pool[20]).tolist() == list(range(1000))
    assert np.flatnonzero(features.universe[20]).tolist() == list(range(500))
    small = _panel()
    small.values["close"][20] = [9, 9, 9, 10, 10, 10, 10, 10, 10, 10]
    selected = reversal.build_features(small)
    assert np.flatnonzero(selected.entry_mask[20]).tolist() == [0, 1]
    small.values["close"][20] = [9, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    assert np.flatnonzero(reversal.build_features(small).entry_mask[20]).tolist() == [0]
    assert not reversal.build_features(_panel(days=21, securities=3)).entry_mask.any()


def test_industry_coverage_minimum_members_and_unknown_boundary():
    """同业均值至少3含自身且覆盖>=60%，3/5可比、3/6不可比、2成员及未知行业关闭。"""
    returns = np.array([[-0.04, 0, 0.01, np.nan, np.nan, np.nan]])
    universe = np.ones(returns.shape, bool)
    sector = np.zeros(returns.shape)
    assert np.isnan(reversal._relative_returns(returns, universe, sector)).all()
    universe[0, -1] = False
    result = reversal._relative_returns(returns, universe, sector)
    np.testing.assert_allclose(result[0, :3], [-0.03, 0.01, 0.02])
    universe[0, 2:] = False
    assert np.isnan(reversal._relative_returns(returns, universe, sector)).all()
    universe[:] = True
    sector[:] = [np.nan, -1, 0.5, np.inf, 0, 0]
    assert np.isnan(reversal._relative_returns(returns, universe, sector)).all()


def test_control_greedy_matching_respects_priority_distance_ties_and_no_reuse():
    """按处理股分数优先匹配最近市值/流动性同业；同距列序，不看对照收益且不重复。"""
    panel = _panel()
    panel.values["capitalization"][:, :5] = [100000, 100000, 100000, 100000, 200000]
    features = reversal.build_features(panel)
    mask, scores, pairs = reversal._matched_control(panel, features)
    assert [pair for pair in pairs if pair[0] == 20] == [(20, 0, 2), (20, 1, 3)]
    assert not np.any(mask & features.entry_mask)
    assert scores[20, 2] == features.scores[20, 0]
    assert scores[20, 3] == features.scores[20, 1]
    # 固定资格和处理股后改写对照收益排序，匹配仍只依赖市值和均额。
    modified = replace(features, relative_return5=-features.relative_return5)
    modified_mask, _, modified_pairs = reversal._matched_control(panel, modified)
    np.testing.assert_equal(modified_mask, mask)
    assert modified_pairs == pairs


def test_unmatched_treatment_is_not_filled_from_another_industry():
    """同业非入选股用完后留下缺额，不跨行业补对照，即使另一个行业还有候选。"""
    panel = _panel()
    features = reversal.build_features(panel)
    mask = np.zeros(features.entry_mask.shape, bool)
    mask[20, :4] = True
    scores = np.full(mask.shape, np.nan)
    scores[20, :4] = [4, 3, 2, 1]
    modified = replace(features, entry_mask=mask, scores=scores)
    control_mask, control_scores, pairs = reversal._matched_control(panel, modified)
    assert pairs == ((20, 0, 4),)
    assert control_mask[20].sum() == 1 and control_scores[20, 4] == 4


def test_delayed_dispatch_preserves_source_day_score_and_eligibility():
    """延迟日资格失效不重筛原信号，分数与source_days严格移动一天而不使用新日排名。"""
    panel = _panel()
    panel.values["beta"][21, :2] = np.nan
    features = reversal.build_features(panel)
    base, delayed, _, _ = reversal.build_strategies(panel, features)
    assert features.entry_mask[20, 0] and not features.eligibility[21, 0]
    assert reversal.candidate_orders(delayed, 21) == reversal.candidate_orders(base, 20)
    assert delayed.source_days[21] == 20
    assert delayed.source_days[0] == -1 and not delayed.entry_mask[0].any()
    np.testing.assert_equal(delayed.entry_mask[1:], base.entry_mask[:-1])
    np.testing.assert_equal(delayed.scores[1:], base.scores[:-1])


def _simulate_one_signal(panel, strategy, signal_day):
    """为软件测试接入一次冻结信号；例如源20日基础21日买、延迟22日买且只尝试一次开盘。"""
    mask = np.zeros(strategy.entry_mask.shape, bool)
    mask[signal_day] = strategy.entry_mask[signal_day]
    definition = replace(strategy, entry_mask=mask)

    def candidates(current_panel, day, stage, policy):
        """忽略当前资格并取冻结候选；例如延迟日beta未知仍按原信号提供订单。"""
        return reversal.candidate_orders(definition, day)

    def valid_open(current_panel, confirmation, day, security):
        """仅允许发单次日开盘；例如confirmation=20时day=21通过，22拒绝。"""
        return day == confirmation + 1

    def cancel_unfilled(current_panel, confirmation, day, security):
        """唯一允许开盘未成交后撤销；例如涨停后不顺延到第二天再买。"""
        return day >= confirmation + 1

    return simulate_portfolio(
        panel,
        np.zeros(len(panel.calendar), np.int8),
        [],
        definition.policy,
        start=panel.calendar[0],
        candidate_provider=candidates,
        entry_filter=valid_open,
        cancel_pending_provider=cancel_unfilled,
    )


def test_real_delayed_execution_holds_five_sessions_without_price_exit():
    """真实账本确认基础t+1与延迟t+2买入；超过旧止盈阈值仍只按真实入场后5日退出。"""
    panel = _panel()
    panel.values["beta"][21, :2] = np.nan
    base, delayed, _, _ = reversal.build_strategies(panel)
    panel.values["raw_close"][21:, :2] = 20
    baseline = _simulate_one_signal(panel, base, 20)
    later = _simulate_one_signal(panel, delayed, 21)
    assert baseline["entry_days"] == [21, 21]
    assert later["entry_days"] == [22, 22]
    assert {trade["exit"] for trade in baseline["trades"]} == {26}
    assert {trade["exit"] for trade in later["trades"]} == {27}
    assert all(trade["reason"] == "time" for trade in baseline["trades"] + later["trades"])
    panel.values["raw_open"][21, :2] = 11
    assert _simulate_one_signal(panel, base, 20)["entry_days"] == []


def test_future_perturbations_and_prefixes_preserve_all_past_features_and_signals():
    """更改未来行情、行业和资格不得改写历史；完整计算与截断前缀的四组定义一致。"""
    panel = _panel(days=45)
    original_features = reversal.build_features(panel)
    original = reversal.build_strategies(panel, original_features)
    prefix = Panel(
        panel.calendar[:30],
        panel.codes,
        {name: values[:30].copy() for name, values in panel.values.items()},
        {},
        {},
    )
    prefix_features = reversal.build_features(prefix)
    shortened = reversal.build_strategies(prefix, prefix_features)
    changed = deepcopy(panel)
    for values in changed.values.values():
        values[30:] = np.nan
    modified_features = reversal.build_features(changed)
    modified = reversal.build_strategies(changed, modified_features)
    for field in fields(original_features):
        expected = getattr(original_features, field.name)[:30]
        np.testing.assert_equal(expected, getattr(prefix_features, field.name))
        np.testing.assert_equal(expected, getattr(modified_features, field.name)[:30])
    for full, short, future in zip(original, shortened, modified, strict=True):
        for field in ("entry_mask", "scores", "source_days"):
            np.testing.assert_equal(getattr(full, field)[:30], getattr(short, field))
            np.testing.assert_equal(getattr(full, field)[:30], getattr(future, field)[:30])
        assert tuple(pair for pair in full.matched_pairs if pair[0] < 30) == short.matched_pairs


@pytest.mark.parametrize("change", ["missing", "shape", "calendar", "codes", "empty"])
def test_invalid_panel_contracts_rejected(change):
    """必需字段、形状或日期证券轴有误时拒绝计算，不隐式广播或套当前名单。"""
    panel = _panel()
    if change == "missing":
        del panel.values["capitalization"]
    elif change == "shape":
        panel.values["close"] = np.ones(35)
    elif change == "calendar":
        panel.calendar = panel.calendar[::-1]
    elif change == "codes":
        panel.codes[0] = panel.codes[1]
    else:
        panel.calendar = []
    with pytest.raises(ValueError, match="panel_fields_or_axes"):
        reversal.build_features(panel)


def test_invalid_feature_shape_and_candidate_day_rejected():
    """外传特征轴错位或候选日期越界时明确拒绝，禁止负索引意外读取最后一天。"""
    panel = _panel()
    features = reversal.build_features(panel)
    with pytest.raises(ValueError, match="feature_shape"):
        reversal.build_strategies(panel, replace(features, scores=np.ones((1, 1))))
    strategy = reversal.build_strategies(panel, features)[0]
    for day in (-1, len(panel.calendar), True):
        with pytest.raises(ValueError, match="candidate_day"):
            reversal.candidate_orders(strategy, day)
