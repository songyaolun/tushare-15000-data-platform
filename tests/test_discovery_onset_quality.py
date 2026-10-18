"""行业初期质量单因素的控制复现、边界、延迟取消与因果性测试。"""

from dataclasses import fields, replace

import numpy as np
import pytest
from test_discovery_sector_onset import _fixture

from tushare_entitlement_hub.strategy_discovery import onset_quality


def _definitions(panel, features):
    """按短名索引十三组策略；例如返回值['control']为原胜者复制。"""
    return {
        item.name.removeprefix("onset_quality_"): item
        for item in onset_quality.build_strategies(panel, features)
    }


def _controlled(monkeypatch):
    """构造全日有效源与可控行业指标，隔离单因素；例如momentum20=.02恰在门槛。"""
    panel, features = _fixture()
    base = next(
        item
        for item in onset_quality.build_sector_strategies(panel, features)
        if item.name == onset_quality.BASE_NAME
    )
    base = replace(
        base,
        entry_mask=np.ones((80, 12), dtype=bool),
        scores=np.ones((80, 12)),
        exit_mask=np.zeros((80, 12), dtype=bool),
    )
    industry = {
        name: np.full((80, 12), value)
        for name, value in (("momentum20", 0.02), ("momentum10", 0.01), ("breadth", 0.6))
    }

    def original(_panel, _features):
        """返回固定源定义；例如任意输入都输出[base]以独立检验附加规则。"""
        return [base]

    def sectors(_features):
        """返回可变行业矩阵；例如breadth[1,0]=.5会取消该窗口确认。"""
        return industry

    monkeypatch.setattr(onset_quality, "build_sector_strategies", original)
    monkeypatch.setattr(onset_quality, "_sector_features", sectors)
    return panel, features, base, industry


def test_control_exactly_reproduces_original_and_all_variants_preserve_policies() -> None:
    """场景：真实离线源面板；预期控制逐字段复现，13组评分、退出、执行政策不变。"""
    panel, features = _fixture()
    base = next(
        item
        for item in onset_quality.build_sector_strategies(panel, features)
        if item.name == onset_quality.BASE_NAME
    )
    definitions = _definitions(panel, features)
    assert len(definitions) == 13
    for item in definitions.values():
        for field in fields(base):
            if field.name in ("name", "entry_mask", "maximum_entry_gap"):
                continue
            expected, actual = getattr(base, field.name), getattr(item, field.name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
    np.testing.assert_array_equal(definitions["control"].entry_mask, base.entry_mask)
    assert definitions["control"].maximum_entry_gap == base.maximum_entry_gap
    for limit in (1, 2):
        np.testing.assert_array_equal(definitions[f"gap_{limit}pct"].entry_mask, base.entry_mask)
        assert definitions[f"gap_{limit}pct"].maximum_entry_gap == limit / 100
    assert definitions["control"].entry_mask[30, 6]
    assert definitions["confirm_breadth_1d"].entry_mask[31, 6]
    assert definitions["confirm_breadth_2d"].entry_mask[32, 6]
    assert not definitions["confirm_breadth_2d"].entry_mask[30:32, 6].any()


def test_individual_filters_are_inclusive_and_do_not_modify_source(monkeypatch) -> None:
    """场景：各门槛边界与缺失值；预期八项过滤各自生效，原数组/评分/policy保持同对象。"""
    panel, features, base, industry = _controlled(monkeypatch)
    industry["momentum20"][30, :3] = [0.019, 0.02, np.nan]
    industry["momentum10"][30, :3] = [0.009, 0.01, np.nan]
    features["ma20"][:] = np.arange(80)[:, None] + 10
    features["close"][:] = features["ma20"]
    features["ma20"][28, 2] = np.nan
    features["ma20"][30, 3] = features["ma20"][25, 3]
    features["close"][30, 4] -= 1
    features["kdj_j"] = np.tile(np.arange(80)[:, None], (1, 12)).astype(float)
    features["kdj_j"][30, :3] = [29, np.nan, 28]
    features["boll_position"] = np.full((80, 12), 0.4)
    features["boll_position"][30, :6] = [0.199, 0.2, 0.65, 0.651, np.nan, np.inf]
    panel.market["breadth"] = np.full(80, 0.5)
    panel.market["breadth"][:7] = [0.349, 0.35, 0.499, 0.5, np.nan, -1, 1.01]
    definitions = _definitions(panel, features)
    for name in ("sector_m20_ge2pct", "sector_m10_ge1pct"):
        assert definitions[name].entry_mask[30, :3].tolist() == [False, True, False]
    assert definitions["ma20_rising5"].entry_mask[30, :4].tolist() == [True, True, False, False]
    assert not definitions["ma20_rising5"].entry_mask[:5].any()
    assert definitions["close_above_ma20"].entry_mask[30, 0]
    assert not definitions["close_above_ma20"].entry_mask[30, 4]
    assert definitions["kdj_j_rising1"].entry_mask[30, :4].tolist() == [False, False, False, True]
    assert definitions["boll_20_65"].entry_mask[30, :6].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert definitions["market_breadth_35"].entry_mask[:7, 0].tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert definitions["market_breadth_50"].entry_mask[:7, 0].tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert base.entry_mask.all()
    for item in definitions.values():
        assert item.scores is base.scores and item.policy is base.policy
        assert item.exit_mask is base.exit_mask
    assert definitions["control"].entry_mask is base.entry_mask


def test_missing_optional_inputs_close_only_their_filters() -> None:
    """场景：源数据完整但无KDJ/BOLL/市场广度；预期四项过滤无信号，控制和延后仍有。"""
    panel, features = _fixture()
    definitions = _definitions(panel, features)
    for name in ("kdj_j_rising1", "boll_20_65", "market_breadth_35", "market_breadth_50"):
        assert not definitions[name].entry_mask.any()
    assert definitions["control"].entry_mask.any()
    assert definitions["confirm_breadth_2d"].entry_mask.any()


@pytest.mark.parametrize("name", ["kdj_j", "boll_position", "breadth"])
def test_misaligned_optional_inputs_raise(name: str) -> None:
    """场景：提供错维可选指标；预期明确拒绝，不能通过广播误用其他股票或未来日期。"""
    panel, features = _fixture()
    if name == "breadth":
        panel.market[name] = np.ones((80, 1))
    else:
        features[name] = np.ones(80)
    with pytest.raises(ValueError, match="onset_quality_.*misaligned"):
        _definitions(panel, features)


@pytest.mark.parametrize(
    "reason", ["source_exit", "later_exit", "invalid", "score", "close", "sector", "breadth"]
)
def test_delayed_source_and_intermediate_invalidations_cancel(monkeypatch, reason: str) -> None:
    """场景：源退出或途中退出/停牌/评分丢失/非法价/换行业/广度弱；预期严格两日确认取消。"""
    panel, features, base, industry = _controlled(monkeypatch)
    base.entry_mask[:] = False
    base.entry_mask[30, 6] = True
    if reason in ("source_exit", "later_exit"):
        base.exit_mask[30 if reason == "source_exit" else 31, 6] = True
    elif reason == "invalid":
        features["valid"][31, 6] = 0
    elif reason == "score":
        base.scores[31, 6] = np.nan
    elif reason == "close":
        features["close"][31, 6] = 0
    elif reason == "sector":
        features["sector_index"][31, 6] = 0
    else:
        industry["breadth"][31, 6] = 0.59
    definitions = _definitions(panel, features)
    assert definitions["control"].entry_mask[30, 6]
    assert not definitions["confirm_breadth_2d"].entry_mask[:, 6].any()
    assert not definitions["confirm_breadth_1d"].entry_mask[:, 6].any()


def test_delayed_confirmation_supports_absent_original_exit(monkeypatch) -> None:
    """场景：复用定义未设exit_mask；预期视为无禁止，照常在严格第二日确认。"""
    panel, features, base, _ = _controlled(monkeypatch)

    def original(_panel, _features):
        """移除可选退出矩阵；例如调用返回exit_mask=None的同策略。"""
        return [replace(base, exit_mask=None)]

    monkeypatch.setattr(onset_quality, "build_sector_strategies", original)
    definitions = _definitions(panel, features)
    assert definitions["confirm_breadth_2d"].entry_mask[2:].all()
    assert definitions["control"].exit_mask is None


@pytest.mark.parametrize("periods", [0, 3, True, 1.0])
def test_confirmation_rejects_invalid_periods(periods) -> None:
    """场景：0/3/布尔/浮点确认周期；预期只接受整数1或2。"""
    with pytest.raises(ValueError, match="invalid_onset_quality_confirmation_periods"):
        onset_quality._delayed_confirmation(
            np.ones((3, 1), bool), np.ones((3, 1), bool), np.ones((3, 1)), np.ones((3, 1)), periods
        )


@pytest.mark.parametrize("membership", [np.nan, np.inf, -1, 0.5])
def test_confirmation_rejects_invalid_source_sector(membership: float) -> None:
    """场景：源行业NaN/无穷/负数/非整数；预期全部延期信号拒绝。"""
    assert not onset_quality._delayed_confirmation(
        np.ones((4, 1), bool),
        np.ones((4, 1), bool),
        np.full((4, 1), membership),
        np.full((4, 1), 0.7),
        2,
    ).any()


@pytest.mark.parametrize("days", [0, 1, 2])
def test_confirmation_short_history_is_empty(days: int) -> None:
    """场景：不足两个后续交易日；预期全False且保持包括空输入的形状。"""
    result = onset_quality._delayed_confirmation(
        np.ones((days, 2), bool),
        np.ones((days, 2), bool),
        np.ones((days, 2)),
        np.ones((days, 2)),
        2,
    )
    assert result.shape == (days, 2) and not result.any()


def test_confirmation_does_not_wait_for_recovery_or_accept_unknown_breadth() -> None:
    """场景：中日NaN广度随后恢复、A→B→A、另一列完整窗口；预期只完整列在固定第2日触发。"""
    source = np.zeros((5, 3), bool)
    source[0] = True
    valid, membership, breadth = np.ones((5, 3), bool), np.ones((5, 3)), np.full((5, 3), 0.6)
    breadth[1, 0] = np.nan
    membership[1, 1] = 2
    result = onset_quality._delayed_confirmation(source, valid, membership, breadth, 2)
    expected = np.zeros((5, 3), bool)
    expected[2, 2] = True
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "values,periods",
    [(np.ones(2), 1), (np.ones((2, 1)), 0), (np.ones((2, 1)), True), (np.ones((2, 1)), 1.0)],
)
def test_lag_rejects_invalid_shape_and_periods(values, periods) -> None:
    """场景：非二维、非正整数或bool滞后；预期拒绝而非环绕/截断参数。"""
    with pytest.raises(ValueError, match="invalid_onset_quality_lag"):
        onset_quality._lag_market(values, periods)


def test_lag_empty_short_and_regular_inputs() -> None:
    """场景：空/短/普通行情；预期前端未知、无尾部环绕、保持形状。"""
    assert onset_quality._lag_market(np.empty((0, 2)), 1).shape == (0, 2)
    assert np.isnan(onset_quality._lag_market(np.ones((2, 1)), 2)).all()
    np.testing.assert_allclose(
        onset_quality._lag_market(np.array([[1.0], [2.0], [3.0]]), 2),
        [[np.nan], [np.nan], [1.0]],
        equal_nan=True,
    )


def test_all_variants_are_prefix_invariant_under_future_mutation() -> None:
    """场景：先算60日前缀，后续行情/行业/指标彻底变化；预期所有历史入场、评分、退出相同。"""
    panel, features = _fixture()
    features["kdj_j"] = np.broadcast_to(np.arange(80)[:, None], (80, 12)).astype(float)
    features["boll_position"] = np.full((80, 12), 0.4)
    panel.market["breadth"] = np.full(80, 0.5)
    initial = _definitions(panel, features)
    for values in features.values():
        values[60:] = np.nan
    panel.market["breadth"][60:] = np.nan
    mutated = _definitions(panel, features)
    for name, item in initial.items():
        np.testing.assert_array_equal(item.entry_mask[:60], mutated[name].entry_mask[:60])
        np.testing.assert_array_equal(item.scores[:60], mutated[name].scores[:60])
        np.testing.assert_array_equal(item.exit_mask[:60], mutated[name].exit_mask[:60])
