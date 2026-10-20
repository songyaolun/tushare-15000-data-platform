"""层级共同研究域、同业自贡献、L1个股条件复现与L2独立事件测试。"""

from dataclasses import fields, replace

import numpy as np
import pytest
from test_discovery_sector_onset import _fixture as _source_fixture

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery import hierarchy_signals, sector_onset
from tushare_entitlement_hub.strategy_discovery.sector import _sector_features


def _fixture(days=80):
    """给真实源面板添加已知L2和市场广度；例如第30日股票6有原策略候选。"""
    panel, features = _source_fixture(days)
    features["sector_level2_index"] = features["sector_index"].copy()
    panel.market["breadth"] = np.full(days, 0.6)
    return panel, features


def _definitions(panel, features):
    """按层级短名返回四组定义；例如['matched_control']取得共同域原评分。"""
    return {
        item.name.removeprefix("hierarchy_"): item
        for item in hierarchy_signals.build_strategies(panel, features)
    }


def _wide_fixture():
    """构造L1持续强而一个L2第30日启动的24股面板；输出两层不同的事件日期。"""
    _, original = _fixture()
    features = {name: np.tile(values, (1, 2)) for name, values in original.items()}
    features["sector_index"][:] = 0
    features["sector_level2_index"][:, :6] = 0
    features["sector_level2_index"][:, 6:] = 1
    features["ma20"][:] = features["close"] * 0.99
    features["ma20"][:30, :4] = features["close"][:30, :4] * 1.01
    panel = Panel(
        [f"day{day:03}" for day in range(80)],
        [f"stock{i}" for i in range(24)],
        features,
        {},
        {"breadth": np.full(80, 0.6)},
    )
    return panel, features


def test_four_definitions_preserve_base_and_share_exact_ranking_candidates() -> None:
    """场景：L1/L2同样可得；预期排序三组同mask、风险退出不动，控制只受共同域限制。"""
    panel, features = _fixture()
    baseline = next(
        item
        for item in hierarchy_signals.build_structure_strategies(panel, features)
        if item.name == hierarchy_signals.BASE_NAME
    )
    definitions = _definitions(panel, features)
    assert set(definitions) == {
        "matched_control",
        "l1_peer_relative20",
        "l2_peer_relative20",
        "l2_onset",
    }
    control = definitions["matched_control"]
    np.testing.assert_array_equal(control.entry_mask, baseline.entry_mask)
    np.testing.assert_array_equal(control.scores, baseline.scores)
    assert control.entry_mask[30, 6]
    for item in definitions.values():
        for field in fields(baseline):
            if field.name in ("name", "entry_mask", "scores"):
                continue
            expected = getattr(baseline, field.name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(getattr(item, field.name), expected)
            else:
                assert getattr(item, field.name) == expected
    for label in ("l1_peer_relative20", "l2_peer_relative20"):
        item = definitions[label]
        assert item.entry_mask is control.entry_mask
        assert np.isfinite(item.scores[item.entry_mask]).all()
        assert (item.scores[item.entry_mask] == 0.5).all()
        assert np.isnan(item.scores[~item.entry_mask]).all()


def test_peer_values_exclude_self_and_equal_values_do_not_invent_ranks() -> None:
    """场景：六成员仅一股涨10%；预期其peer为0，其余差-.02，不同大小全等组均差0。"""
    values = np.array([[0.1, 0, 0, 0, 0, 0]])
    relative, comparable = hierarchy_signals._peer_statistics(
        values, np.zeros_like(values), np.ones(values.shape, bool)
    )
    np.testing.assert_allclose(relative, [[0.1, -0.02, -0.02, -0.02, -0.02, -0.02]])
    assert comparable.all()
    values = np.full((1, 13), 0.05)
    membership = np.array([[0] * 6 + [1] * 7], dtype=float)
    relative, comparable = hierarchy_signals._peer_statistics(
        values, membership, np.ones(values.shape, bool)
    )
    assert comparable.all() and (relative == 0).all()


@pytest.mark.parametrize(
    "members,observed,expected", [(5, 5, False), (6, 6, True), (11, 7, True), (12, 7, False)]
)
def test_peer_count_and_coverage_boundaries(members, observed, expected):
    """场景：4/5同业和60%覆盖上下边界；预期剔除本股后同时满足数量与覆盖才可得。"""
    eligible = np.zeros((1, members), bool)
    eligible[:, :observed] = True
    relative, comparable = hierarchy_signals._peer_statistics(
        np.ones((1, members)), np.zeros((1, members)), eligible
    )
    assert comparable[0, 0] == expected
    assert np.isfinite(relative[0, 0]) == expected
    assert not comparable[~eligible].any()


@pytest.mark.parametrize("identifier", [np.nan, np.inf, -0.1, 0.5])
def test_unknown_or_illegal_peer_membership_is_not_a_group(identifier):
    """场景：所有行业ID未知或非法；预期不形成合成组，也不把NaN成员计入分母。"""
    relative, comparable = hierarchy_signals._peer_statistics(
        np.ones((3, 6)), np.full((3, 6), identifier), np.ones((3, 6), bool)
    )
    assert np.isnan(relative).all() and not comparable.any()


def test_nonfinite_values_and_no_observed_peers_fail_closed() -> None:
    """场景：行业存在但无有效价格或一同业收益缺失；预期不把缺值填零制造五同业。"""
    for value in (np.nan, np.inf):
        values = np.ones((2, 6))
        values[0] = value
        values[1, 0] = value
        relative, comparable = hierarchy_signals._peer_statistics(
            values, np.zeros_like(values), np.ones(values.shape, bool)
        )
        assert np.isnan(relative).all() and not comparable.any()


@pytest.mark.parametrize(
    "values,membership,eligible",
    [
        (np.ones(6), np.ones(6), np.ones(6, bool)),
        (np.ones((1, 6)), np.ones((2, 6)), np.ones((1, 6), bool)),
        (np.ones((1, 6)), np.ones((1, 6)), np.ones((2, 6), bool)),
    ],
)
def test_peer_statistics_reject_misaligned_arrays(values, membership, eligible):
    """场景：维度不为2或三矩阵不同形；预期明确拒绝广播。"""
    with pytest.raises(ValueError, match="invalid_peer_statistics_shape"):
        hierarchy_signals._peer_statistics(values, membership, eligible)


@pytest.mark.parametrize("lookback", [0, -1, True, 1.0])
def test_stable_membership_rejects_invalid_window(lookback):
    """场景：非正整数或bool窗口；预期不截断或容忍错误参数。"""
    with pytest.raises(ValueError, match="invalid_stable_membership_history"):
        hierarchy_signals._stable_membership_history(
            np.ones((2, 1), bool), np.ones((2, 1)), lookback
        )


def test_stable_membership_shape_and_empty_history() -> None:
    """场景：错维与空日历；预期错维拒绝、合法空输入维持形状不产生历史。"""
    for valid, membership in (
        (np.ones(2, bool), np.ones(2)),
        (np.ones((2, 1), bool), np.ones((2, 2))),
    ):
        with pytest.raises(ValueError, match="invalid_stable_membership_history"):
            hierarchy_signals._stable_membership_history(valid, membership, 1)
    result = hierarchy_signals._stable_membership_history(
        np.empty((0, 2), bool), np.empty((0, 2)), 1
    )
    assert result.shape == (0, 2)


def test_stable_membership_detects_switch_return_and_intermediate_missing() -> None:
    """场景：A→B→A、途中无效、负/小数ID；预期相同端点不能伪造连续窗口。"""
    membership = np.ones((7, 5))
    membership[2, :4] = [2, np.nan, -1, 0.5]
    valid = np.ones((7, 5), bool)
    valid[2, 4] = False
    result = hierarchy_signals._stable_membership_history(valid, membership, 3)
    assert not result[:6].any()
    assert result[6].all()


@pytest.mark.parametrize("change", ["suspension", "switch", "missing_return"])
def test_relative_returns_require_twenty_one_continuous_days(change):
    """场景：六股中一股窗口停牌/换行业/收益未知；预期本股无资格，其余不够五同业。"""
    _, features = _fixture()
    if change == "suspension":
        features["valid"][25, 6] = 0
    elif change == "switch":
        features["sector_index"][25, 6] = 0
    else:
        features["return20"][30, 6] = np.nan
    relative = hierarchy_signals._relative_returns(features, features["sector_index"])
    assert np.isnan(relative[:20]).all()
    assert np.isnan(relative[30, 6:]).all()


def test_l1_and_l2_sorting_use_same_domain_and_same_scale(monkeypatch) -> None:
    """场景：L2未知减少研究域且两级相对收益不同；预期三组同候选，两排名均[0,1]。"""
    panel, features = _fixture()
    baseline = next(
        item
        for item in hierarchy_signals.build_structure_strategies(panel, features)
        if item.name == hierarchy_signals.BASE_NAME
    )
    entries = np.zeros_like(baseline.entry_mask)
    entries[30] = True
    baseline = replace(baseline, entry_mask=entries)
    features["return20"][30] = np.arange(12) * 0.01
    features["sector_level2_index"][:, [0, 6]] = 3

    def fixed_base(_panel, _features):
        """提供固定候选集合；例如第30日12只均候选，覆盖门独立决定减少多少。"""
        return [baseline]

    monkeypatch.setattr(hierarchy_signals, "build_structure_strategies", fixed_base)
    definitions = _definitions(panel, features)
    assert not definitions["matched_control"].entry_mask.any()
    features["sector_level2_index"][:] = 0
    definitions = _definitions(panel, features)
    assert definitions["matched_control"].entry_mask[30].all()
    first, second = definitions["l1_peer_relative20"], definitions["l2_peer_relative20"]
    assert first.entry_mask is second.entry_mask
    assert first.scores[30].min() >= 0 and first.scores[30].max() <= 1
    assert second.scores[30].min() == 0 and second.scores[30].max() == 1
    assert not np.array_equal(first.scores[30], second.scores[30])


@pytest.mark.parametrize(
    "field,value",
    [
        ("return20", -0.001),
        ("return20", 0.101),
        ("return1", -0.011),
        ("return1", 0.031),
        ("ma20", np.nan),
        ("return5", np.nan),
        ("turnover", np.nan),
        ("turnover", 100),
        ("valid", 0),
        ("close", 0),
    ],
)
def test_rebuilt_l1_stock_mask_exactly_matches_source_with_rejections(field, value):
    """场景：逐项触碰原个股资格或缺值；预期复建个股mask交原事件严格等于原源entry。"""
    panel, features = _fixture()
    features[field][30, 6] = value
    original = next(
        item
        for item in sector_onset.build_strategies(panel, features)
        if item.name == "sector_onset_selective_quiet_risk_tp8_sl8_h30"
    )
    industry = _sector_features(features)
    previous = sector_onset._previous_industry(industry["breadth"], features["sector_index"])
    event = (
        (previous < 0.65)
        & (industry["breadth"] >= 0.65)
        & (industry["diffusion"] >= 0.15)
        & (industry["momentum10"] > 0)
        & (industry["momentum10"] <= 0.08)
    )
    np.testing.assert_array_equal(
        hierarchy_signals._original_stock_mask(features) & event, original.entry_mask
    )


def test_l2_event_changes_dates_without_l2_stock_rank_or_management() -> None:
    """场景：L1一直强但L2在第30日启动；预期L2新增事件、保原L1分数和持仓管理。"""
    panel, features = _wide_fixture()
    definitions = _definitions(panel, features)
    assert not definitions["matched_control"].entry_mask.any()
    assert definitions["l2_onset"].entry_mask[30, 0]
    assert not definitions["l2_onset"].entry_mask[31:, 0].any()
    for field in ("scores", "exit_mask", "entry_invalidation_mask", "policy"):
        assert getattr(definitions["matched_control"], field) is getattr(
            definitions["l2_onset"], field
        )
    original_mask = hierarchy_signals._original_stock_mask(features)
    features["sector_level2_index"][:, 6:] = np.nan
    np.testing.assert_array_equal(hierarchy_signals._original_stock_mask(features), original_mask)


def test_l2_entry_does_not_eagerly_consume_original_invalidation(monkeypatch) -> None:
    """场景：L2事件日原入场否决为真；预期entry仍描述事件，原执行器负责否决。"""
    panel, features = _wide_fixture()
    baseline = next(
        item
        for item in hierarchy_signals.build_structure_strategies(panel, features)
        if item.name == hierarchy_signals.BASE_NAME
    )
    invalidation = np.ones_like(baseline.entry_mask)
    baseline = replace(baseline, entry_invalidation_mask=invalidation)

    def fixed_base(_panel, _features):
        """返回带原否决的L1基准；例如测试event与禁止执行分离。"""
        return [baseline]

    monkeypatch.setattr(hierarchy_signals, "build_structure_strategies", fixed_base)
    event = _definitions(panel, features)["l2_onset"]
    assert event.entry_mask[30, 0]
    assert event.entry_invalidation_mask is invalidation


@pytest.mark.parametrize("day", [25, 29, 30])
def test_l2_event_requires_enough_peers_today_yesterday_and_five_days_ago(day):
    """场景：三个比较时点任一同业MA未知使六股只剩五股可用；预期事件取消。"""
    _, features = _fixture()
    features["ma20"][day, 11] = np.nan
    event = hierarchy_signals._onset_event(features, features["sector_level2_index"])
    assert not event[30, 6]


def test_l2_event_rejects_unknown_market_and_switched_membership() -> None:
    """场景：市场广度未知/不达50%或窗口行业切换；预期L2新事件不能通过。"""
    for breadth in (np.nan, 0.49, 1.1):
        panel, features = _wide_fixture()
        panel.market["breadth"][30] = breadth
        assert not _definitions(panel, features)["l2_onset"].entry_mask[30, 0]
    panel, features = _wide_fixture()
    features["sector_level2_index"][28, 0] = 1
    assert not _definitions(panel, features)["l2_onset"].entry_mask[30, 0]


@pytest.mark.parametrize("name", hierarchy_signals.REQUIRED_FEATURES)
@pytest.mark.parametrize("missing", [True, False])
def test_required_feature_missing_or_shape_error_rejects(name, missing):
    """场景：L2或源字段缺失/错维；预期显式层级研究拒绝，不静默降为L1。"""
    panel, features = _fixture()
    if missing:
        del features[name]
    else:
        features[name] = np.ones(80)
    with pytest.raises(ValueError, match="hierarchy_features_missing_or_misaligned"):
        _definitions(panel, features)


def test_prefix_and_future_mutation_preserve_all_historical_outputs() -> None:
    """场景：截断60日前缀和改写全部未来行情/层级；预期四组历史入场评分退出完全相同。"""
    panel, features = _wide_fixture()
    full = _definitions(panel, features)
    shorter = {name: values[:60].copy() for name, values in features.items()}
    prefix_panel = Panel(
        calendar=panel.calendar[:60],
        codes=panel.codes,
        values=shorter,
        signals={},
        market={"breadth": panel.market["breadth"][:60].copy()},
    )
    prefix = _definitions(prefix_panel, shorter)
    for values in features.values():
        values[60:] = np.nan
    panel.market["breadth"][60:] = np.nan
    future = _definitions(panel, features)
    for name, original in full.items():
        for field in ("entry_mask", "scores", "exit_mask", "entry_invalidation_mask"):
            expected = getattr(original, field)[:60]
            np.testing.assert_array_equal(getattr(prefix[name], field), expected)
            np.testing.assert_array_equal(getattr(future[name], field)[:60], expected)
