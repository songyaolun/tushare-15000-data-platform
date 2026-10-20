"""行业初期失败退出的固定入场否决、同业连续性与因果性测试。"""

from dataclasses import fields, replace

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.regime_research.portfolio import PortfolioPolicy
from tushare_entitlement_hub.strategy_discovery import onset_management as management
from tushare_entitlement_hub.strategy_discovery.contracts import StrategyDefinition


def management_fixture(monkeypatch):
    """构造八日两证券，行业0连续恶化但第4/5日指标缺失，行业1一直健康。

    输入pytest monkeypatch，输出定义/个股特征/映射行业指标；原入场全真但行业0多数日被否决。
    """
    shape = (8, 2)
    features = {
        "valid": np.ones(shape),
        "close": np.full(shape, 11.0),
        "ma20": np.full(shape, 10.0),
        "return1": np.full(shape, 0.001),
        "sector_index": np.broadcast_to([0.0, 1.0], shape).copy(),
    }
    features["close"][[2, 3, 5, 6, 7], 0] = 9
    industry = {
        "breadth": np.full(shape, 0.7),
        "momentum20": np.full(shape, 0.1),
        "momentum10": np.full(shape, 0.1),
    }
    industry["breadth"][:, 0] = [0.7, 0.3, 0.3, 0.3, np.nan, 0.3, 0.3, 0.3]
    industry["momentum20"][:, 0] = [0.1, 0.1, 0.1, 0.1, -0.01, np.nan, -0.03, -0.03]
    industry["momentum10"][[2, 3, 6, 7], 0] = -0.01
    exit_mask = (industry["breadth"] < 0.4) | (industry["momentum20"] <= 0)
    definition = StrategyDefinition(
        name=management.BASELINE_NAME,
        family="sector_breadth_onset",
        entry_mask=np.ones(shape, dtype=bool),
        scores=np.ones(shape),
        exit_mask=exit_mask,
        policy=PortfolioPolicy(
            name=management.BASELINE_NAME,
            switching=False,
            stage_allocation=False,
            force_rebalance=False,
            take_profit=0.08,
            stop_loss=0.08,
            maximum_holding=30,
        ),
    )
    monkeypatch.setattr(management, "_sector_features", lambda supplied: industry)
    return definition, features, industry


def indexed_variants(definition, features):
    """按manage后缀索引十二配置；例字典industry_off的退出为None。"""
    return {
        item.name.split("__manage_")[1]: item
        for item in management.build_variants(definition, features)
    }


def test_twelve_declared_rules_change_only_their_exit_component(monkeypatch):
    """场景：完整的十二项预声明对照；预期：唯一名称、固定否决和仅指定SL/H参数变化。"""
    definition, features, _ = management_fixture(monkeypatch)
    definition = replace(
        definition, intraday_exits=True, trailing_stop=0.03, maximum_entry_gap=0.01
    )
    variants = indexed_variants(definition, features)
    assert set(variants) == {
        "breadth_only",
        "momentum20_only",
        "bad_confirm2",
        "bad_confirm3",
        "momentum20_deep",
        "momentum10",
        "bad_below_ma20",
        "stop4",
        "stop6",
        "hold20",
        "hold40",
        "industry_off",
    }
    for label, variant in variants.items():
        assert variant.policy.name == variant.name
        assert variant.entry_invalidation_mask is definition.exit_mask
        np.testing.assert_array_equal(
            variant.entry_mask, definition.entry_mask & ~definition.exit_mask
        )
        assert variant.entry_mask is not definition.entry_mask
        assert variant.scores is definition.scores
        expected_policy = replace(
            definition.policy,
            name=variant.name,
            stop_loss=0.04 if label == "stop4" else 0.06 if label == "stop6" else 0.08,
            maximum_holding=20 if label == "hold20" else 40 if label == "hold40" else 30,
        )
        assert variant.policy == expected_policy
        for field in fields(definition):
            if field.name not in {
                "name",
                "policy",
                "entry_mask",
                "entry_invalidation_mask",
                "exit_mask",
                "scores",
            }:
                assert getattr(variant, field.name) == getattr(definition, field.name)
    assert variants["industry_off"].exit_mask is None
    for label in ("stop4", "stop6", "hold20", "hold40"):
        assert variants[label].exit_mask is definition.exit_mask


def test_single_day_exit_conditions_and_partial_unknown_or_semantics(monkeypatch):
    """场景：广度/动量单项坏、部分指标未知和个股均线差异；预期：只由已知条件触发。"""
    definition, features, industry = management_fixture(monkeypatch)
    variants = indexed_variants(definition, features)
    assert variants["breadth_only"].exit_mask[1, 0]
    assert not variants["breadth_only"].exit_mask[4, 0]
    assert not variants["momentum20_only"].exit_mask[1, 0]
    assert variants["momentum20_only"].exit_mask[4, 0]
    assert not variants["momentum20_only"].exit_mask[5, 0]
    assert not variants["momentum20_deep"].exit_mask[4, 0]
    assert variants["momentum20_deep"].exit_mask[5, 0]
    assert not variants["bad_below_ma20"].exit_mask[1, 0]
    assert variants["bad_below_ma20"].exit_mask[2, 0]
    industry["breadth"][0, 0] = 0.4
    industry["momentum20"][0, 0] = -0.02
    industry["momentum10"][0, 0] = 0
    variants = indexed_variants(definition, features)
    assert not variants["breadth_only"].exit_mask[0, 0]
    assert variants["momentum20_deep"].exit_mask[0, 0]
    assert variants["momentum10"].exit_mask[0, 0]
    assert not variants["momentum10"].exit_mask[4, 0]


def test_consecutive_bad_requires_both_indicators_known_and_does_not_bridge_gaps(monkeypatch):
    """场景：原OR坏信号在缺失期间仍为真；预期：连续坏不能把部分未知日计入2/3日。"""
    definition, features, _ = management_fixture(monkeypatch)
    variants = indexed_variants(definition, features)
    np.testing.assert_array_equal(
        variants["bad_confirm2"].exit_mask[:, 0],
        [False, False, True, True, False, False, False, True],
    )
    np.testing.assert_array_equal(
        variants["bad_confirm3"].exit_mask[:, 0],
        [False, False, False, True, False, False, False, False],
    )
    features["sector_index"][2:, 0] = 1
    changed = indexed_variants(definition, features)
    assert not changed["bad_confirm2"].exit_mask[2, 0]
    assert changed["bad_confirm2"].exit_mask[3, 0]
    assert not changed["bad_confirm3"].exit_mask[3, 0]


def test_explicit_original_entry_veto_takes_precedence_over_exit(monkeypatch):
    """场景：基线入场否决与行业卖出阵不同；预期：全部变体固定真正的原否决，不取exit替代。"""
    definition, features, _ = management_fixture(monkeypatch)
    original_veto = np.zeros(definition.entry_mask.shape, dtype=bool)
    original_veto[0, 1] = True
    definition = replace(definition, entry_invalidation_mask=original_veto)
    for variant in management.build_variants(definition, features):
        assert variant.entry_invalidation_mask is original_veto
        np.testing.assert_array_equal(variant.entry_mask, definition.entry_mask & ~original_veto)
        assert not variant.entry_mask[0, 1]
        assert variant.entry_mask[1, 0]


@pytest.mark.parametrize(("field", "bad"), [("valid", 0), ("close", np.nan), ("close", 0)])
def test_new_industry_conditions_refuse_nontradable_days(monkeypatch, field, bad):
    """场景：股票停牌或收盘未知/非正；预期：新行业退出不触发，原退出保留组保持原矩阵。"""
    definition, features, _ = management_fixture(monkeypatch)
    features[field][:] = bad
    variants = indexed_variants(definition, features)
    for label in (
        "breadth_only",
        "momentum20_only",
        "bad_confirm2",
        "bad_confirm3",
        "momentum20_deep",
        "momentum10",
        "bad_below_ma20",
    ):
        assert not variants[label].exit_mask.any()
    assert variants["stop4"].exit_mask is definition.exit_mask


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0, -1])
def test_individual_ma20_confirmation_requires_known_positive_average(monkeypatch, bad):
    """场景：个股均线缺失或不合法；预期：行业坏且跌破均线版本不将未知当确认。"""
    definition, features, _ = management_fixture(monkeypatch)
    features["ma20"][:] = bad
    assert not indexed_variants(definition, features)["bad_below_ma20"].exit_mask.any()


def test_consecutive_helper_checks_membership_knowledge_resets_and_empty_shape():
    """场景：换行业、未知行业、非整数行业和未知状态；预期：计数逐段重启且空矩阵保持。"""
    bad = np.ones((9, 1), dtype=bool)
    known = np.ones_like(bad)
    membership = np.array([[0.0], [0], [1], [1], [np.nan], [1.5], [1], [1], [1]])
    known[7] = False
    result = management._consecutive_bad(bad, known, membership, 2)
    np.testing.assert_array_equal(
        result[:, 0], [False, True, False, True, False, False, False, False, False]
    )
    bad[1] = False
    assert not management._consecutive_bad(bad, known, membership, 2)[1, 0]
    assert management._consecutive_bad(bad[:0], known[:0], membership[:0], 2).shape == (0, 1)


@pytest.mark.parametrize(
    "mutation", ["zero", "boolean", "one_dimensional", "known_shape", "member_shape"]
)
def test_invalid_consecutive_contract_is_rejected(mutation):
    """场景：非法连续长度或错轴；预期：明确ValueError，不做广播。"""
    bad = np.ones((2, 2), dtype=bool)
    known, membership, sessions = bad.copy(), np.zeros((2, 2)), 2
    if mutation == "zero":
        sessions = 0
    elif mutation == "boolean":
        sessions = True
    elif mutation == "one_dimensional":
        bad = bad[0]
    elif mutation == "known_shape":
        known = known[:1]
    else:
        membership = membership[:1]
    with pytest.raises(ValueError, match="invalid_consecutive_industry_bad_inputs"):
        management._consecutive_bad(bad, known, membership, sessions)


@pytest.mark.parametrize("mutation", ["family", "no_exit", "missing", "shape"])
def test_management_requires_expected_anchor_and_features(monkeypatch, mutation):
    """场景：其他族、缺原退出或缺字段/错维度；预期：拒绝不明确的对照输入。"""
    definition, features, _ = management_fixture(monkeypatch)
    if mutation == "family":
        definition = replace(definition, family="trend")
    elif mutation == "no_exit":
        definition = replace(definition, exit_mask=None)
    elif mutation == "missing":
        del features["return1"]
    else:
        features["close"] = np.ones((2, 8))
    with pytest.raises(ValueError, match="onset_management_"):
        management.build_variants(definition, features)


def real_feature_fixture():
    """生成40日两个行业各三证券的实际聚合输入；例末十日行业0广度下降且动量转弱。"""
    shape = (40, 6)
    daily = np.full(shape, 0.001)
    daily[25:, :3] = -0.005
    close = 10 * np.cumprod(1 + daily, axis=0)
    features = {
        "valid": np.ones(shape),
        "close": close,
        "ma20": close * 0.99,
        "return1": daily,
        "return5": np.full(shape, -0.02),
        "return20": np.full(shape, 0.05),
        "turnover": np.broadcast_to([1, 2, 3, 1, 2, 3], shape).astype(float),
        "sector_index": np.broadcast_to([0, 0, 0, 1, 1, 1], shape).astype(float),
    }
    features["ma20"][25:, :3] = close[25:, :3] * 1.01
    panel = Panel([f"day{day}" for day in range(40)], list("ABCDEF"), features, {}, {})
    return panel, features


def test_group_builder_selects_fixed_anchor_and_returns_twelve():
    """场景：真实行业聚合与原onset生成器；预期：只选择固定锚并生成12项，未改原矩阵。"""
    panel, features = real_feature_fixture()
    source = next(
        item
        for item in management.build_onset_strategies(panel, features)
        if item.name == management.BASELINE_NAME
    )
    variants = management.build_strategies(panel, features)
    assert len(variants) == 12
    for item in variants:
        assert item.name.startswith(management.BASELINE_NAME + "__manage_")
        np.testing.assert_array_equal(item.entry_mask, source.entry_mask & ~source.exit_mask)
        np.testing.assert_array_equal(item.entry_invalidation_mask, source.exit_mask)


def test_real_industry_future_perturbation_and_prefix_preserve_history():
    """场景：真实聚合加入未来行业ID/涨跌并截断历史；预期：此前退出、否决和入场完全一致。"""
    panel, features = real_feature_fixture()
    original = management.build_strategies(panel, features)
    changed_features = {name: values.copy() for name, values in features.items()}
    for values in changed_features.values():
        values[32:] = 999
    changed = management.build_strategies(panel, changed_features)
    prefix_features = {name: values[:32] for name, values in features.items()}
    prefix_panel = Panel(panel.calendar[:32], panel.codes, prefix_features, {}, {})
    prefix = management.build_strategies(prefix_panel, prefix_features)
    for before, after, shorter in zip(original, changed, prefix, strict=True):
        for field in ("entry_mask", "scores", "exit_mask", "entry_invalidation_mask"):
            values = getattr(before, field)
            if values is not None:
                np.testing.assert_array_equal(values[:32], getattr(after, field)[:32])
                np.testing.assert_array_equal(values[:32], getattr(shorter, field))


def test_inputs_and_variant_entries_are_independent(monkeypatch):
    """场景：生成对照后修改其中一组入场；预期：原输入和其他变体不会被别名写入影响。"""
    definition, features, _ = management_fixture(monkeypatch)
    before = {name: values.copy() for name, values in features.items()}
    original_entry = definition.entry_mask.copy()
    variants = management.build_variants(definition, features)
    variants[0].entry_mask[:] = False
    assert variants[1].entry_mask.any()
    np.testing.assert_array_equal(definition.entry_mask, original_entry)
    for name, values in features.items():
        np.testing.assert_array_equal(values, before[name])
