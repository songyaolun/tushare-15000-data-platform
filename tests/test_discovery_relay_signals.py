"""冻结五日前身份的三种排序及共同候选控制，验证不删候选、不读取未来。"""

from dataclasses import fields, replace

import numpy as np
import pytest
from test_discovery_hierarchy_signals import _fixture

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery import relay_signals


def _definitions(panel, features):
    """按relay短名索引四组；例如['current_strength']是不使用五日前身份的消融。"""
    return {
        item.name.removeprefix("relay_"): item
        for item in relay_signals.build_strategies(panel, features)
    }


def _controlled(monkeypatch, days=80):
    """固定源全部候选且原分数[0,1]；例如身份排序不能删除任何非优先股票。"""
    panel, features = _fixture(days)
    baseline = next(
        item
        for item in relay_signals.build_structure_strategies(panel, features)
        if item.name == relay_signals.BASE_NAME
    )
    baseline = replace(
        baseline,
        entry_mask=np.ones((days, 12), bool),
        scores=np.broadcast_to(np.linspace(0, 1, 12), (days, 12)).copy(),
    )

    def fixed_base(_panel, _features):
        """返回固定原定义；例如可在没有真实行业事件时验证身份优先级。"""
        return [baseline]

    monkeypatch.setattr(relay_signals, "build_structure_strategies", fixed_base)
    return panel, features, baseline


def test_four_variants_keep_candidates_and_all_management_fields() -> None:
    """场景：真实行业启动且身份历史完整；预期四组与原候选一致，退出和风险政策相同。"""
    panel, features = _fixture()
    baseline = next(
        item
        for item in relay_signals.build_structure_strategies(panel, features)
        if item.name == relay_signals.BASE_NAME
    )
    definitions = _definitions(panel, features)
    assert set(definitions) == {
        "matched_control",
        "previous_leaders",
        "new_participants",
        "current_strength",
    }
    for item in definitions.values():
        np.testing.assert_array_equal(item.entry_mask, baseline.entry_mask)
        for field in fields(baseline):
            if field.name in ("name", "entry_mask", "scores"):
                continue
            expected = getattr(baseline, field.name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(getattr(item, field.name), expected)
            else:
                assert getattr(item, field.name) == expected
    np.testing.assert_array_equal(definitions["matched_control"].scores, baseline.scores)
    assert definitions["matched_control"].entry_mask[30, 6]


def test_frozen_identity_priority_and_original_score_tiebreak(monkeypatch) -> None:
    """场景：老参与、新参与、两日均下方、过去等均线；预期身份主级原分次级且保留全部。"""
    panel, features, baseline = _controlled(monkeypatch)
    features["ma20"][25, :4] = features["close"][25, :4] * [0.99, 1.01, 1.01, 1]
    features["ma20"][25, 11] = features["close"][25, 11] * 1.01
    features["ma20"][30, :4] = features["close"][30, :4] * [0.99, 0.99, 1.01, 0.99]
    definitions = _definitions(panel, features)
    control = definitions["matched_control"]
    for item in definitions.values():
        assert item.entry_mask is control.entry_mask
        assert item.entry_mask[30].all()
        assert item.policy is baseline.policy
        assert item.exit_mask is baseline.exit_mask
        assert item.entry_invalidation_mask is baseline.entry_invalidation_mask
    old, new, current = (
        definitions[name].scores[30]
        for name in ("previous_leaders", "new_participants", "current_strength")
    )
    assert old[0] > old[1] and old[0] > old[11]
    assert new[1] > new[0] and new[3] > new[1] and new[1] > new[2]
    assert current[0] > current[2] and current[1] > current[0]
    assert control.scores is baseline.scores


def test_ties_are_not_broken_by_security_position_and_nan_is_not_a_score(monkeypatch) -> None:
    """场景：同身份同原分与原分NaN；预期同身份仍同分，未知分数不填成可交易优先分。"""
    panel, features, baseline = _controlled(monkeypatch)
    baseline.scores[:] = 0.5
    baseline.scores[30, 2] = np.nan
    for item in _definitions(panel, features).values():
        assert item.scores[30, 0] == item.scores[30, 1]
        assert np.isnan(item.scores[30, 2])


@pytest.mark.parametrize(
    "field,value",
    [
        ("valid", 0),
        ("close", np.nan),
        ("close", 0),
        ("ma20", np.nan),
        ("ma20", 0),
        ("ma20", np.inf),
    ],
)
@pytest.mark.parametrize("day", [25, 27, 30])
def test_any_unknown_identity_window_day_closes_all_four_equally(monkeypatch, field, value, day):
    """场景：六日身份窗口首/中/末任一停牌或均线行情非法；预期四组同样排除。"""
    panel, features, _ = _controlled(monkeypatch)
    features[field][day, 6] = value
    for item in _definitions(panel, features).values():
        assert not item.entry_mask[30, 6]
        assert item.entry_mask[30, 7]


@pytest.mark.parametrize("identifier", [0, np.nan, -1, 0.5])
def test_membership_switch_and_return_cannot_invent_old_identity(monkeypatch, identifier):
    """场景：窗口中换行业后回来或ID非法；预期不能仅比较两端构造连续身份。"""
    panel, features, _ = _controlled(monkeypatch)
    features["sector_index"][27, 6] = identifier
    assert all(not item.entry_mask[30, 6] for item in _definitions(panel, features).values())


@pytest.mark.parametrize("days", [0, 1, 5])
def test_short_histories_have_no_frozen_identity(monkeypatch, days):
    """场景：不足完整六日含空矩阵；预期四组无候选，形状保持原样。"""
    panel, features, _ = _controlled(monkeypatch, days)
    for item in _definitions(panel, features).values():
        assert item.entry_mask.shape == (days, 12)
        assert not item.entry_mask.any()


@pytest.mark.parametrize("name", relay_signals.REQUIRED_FEATURES)
@pytest.mark.parametrize("missing", [True, False])
def test_source_features_missing_or_misaligned_raise(name, missing):
    """场景：任一原必需特征缺失/错维；预期显式拒绝，不把缺失身份当False。"""
    panel, features = _fixture()
    if missing:
        del features[name]
    else:
        features[name] = np.ones(80)
    with pytest.raises(ValueError, match="relay_features_missing_or_misaligned"):
        _definitions(panel, features)


def test_current_day_strength_does_not_rewrite_five_day_old_identity(monkeypatch) -> None:
    """场景：只把今日收盘移到MA20另一侧；预期老参与排序不变，新参与/当日消融改变。"""
    panel, features, _ = _controlled(monkeypatch)
    initial = _definitions(panel, features)
    features["ma20"][30] = features["close"][30] * 1.01
    changed = _definitions(panel, features)
    np.testing.assert_array_equal(
        initial["previous_leaders"].scores[30], changed["previous_leaders"].scores[30]
    )
    assert not np.array_equal(
        initial["new_participants"].scores[30], changed["new_participants"].scores[30]
    )
    assert not np.array_equal(
        initial["current_strength"].scores[30], changed["current_strength"].scores[30]
    )


def test_prefix_and_future_changes_do_not_modify_past_identity() -> None:
    """场景：截断前缀和未来行情/行业任意变化；预期四组历史候选、评分及退出不变。"""
    panel, features = _fixture()
    initial = _definitions(panel, features)
    prefix_features = {name: values[:60].copy() for name, values in features.items()}
    prefix_panel = Panel(
        calendar=panel.calendar[:60],
        codes=panel.codes,
        values=prefix_features,
        signals={},
        market={"breadth": panel.market["breadth"][:60].copy()},
    )
    prefix = _definitions(prefix_panel, prefix_features)
    for values in features.values():
        values[60:] = np.nan
    panel.market["breadth"][60:] = np.nan
    future = _definitions(panel, features)
    for name, original in initial.items():
        for field in ("entry_mask", "scores", "exit_mask", "entry_invalidation_mask"):
            expected = getattr(original, field)[:60]
            np.testing.assert_array_equal(getattr(prefix[name], field), expected)
            np.testing.assert_array_equal(getattr(future[name], field)[:60], expected)
