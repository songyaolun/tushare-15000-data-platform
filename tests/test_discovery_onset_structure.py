"""结构单因素策略的控制复现、阈值、缺失窗口与历史因果性验证。"""

from dataclasses import fields, replace

import numpy as np
import pytest
from test_discovery_sector_onset import _fixture

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery import onset_structure


def _complete_fixture():
    """补齐已有源面板的市场和复权价格；例如第30日股票6有合格源信号。"""
    panel, features = _fixture()
    panel.market["breadth"] = np.full(80, 0.6)
    features["open"] = features["close"].copy()
    features["high"] = features["close"] + 0.3
    features["low"] = features["close"] - 0.7
    return panel, features


def _baseline(panel, features):
    """按真实组合链构造基准；例如返回q50__manage_momentum20_only定义。"""
    quality = next(
        item
        for item in onset_structure.build_quality_strategies(panel, features)
        if item.name == onset_structure.QUALITY_BASE_NAME
    )
    return next(
        item
        for item in onset_structure.build_management_variants(quality, features)
        if item.name == onset_structure.BASE_NAME
    )


def _definitions(panel, features):
    """用预声明短名索引9组；例如['gap_2pct']返回上限2%的原候选。"""
    return {
        item.name.removeprefix("onset_structure_"): item
        for item in onset_structure.build_strategies(panel, features)
    }


def _controlled(monkeypatch):
    """固定有效源与行业指标隔离附加规则；例如市场窗口可独立变化而不改源候选。"""
    panel, features = _complete_fixture()
    baseline = _baseline(panel, features)
    baseline = replace(baseline, entry_mask=np.ones((80, 12), bool))
    industry = {"momentum10": np.full((80, 12), 0.01), "momentum20": np.full((80, 12), 0.05)}

    def management(_quality, _features):
        """返回同一固定基准；例如控制组可用is验证所有数组未复制或变更。"""
        return [baseline]

    def sectors(_features):
        """返回可修改的动量矩阵；例如momentum10=.009应拒绝1%门槛。"""
        return industry

    monkeypatch.setattr(onset_structure, "build_management_variants", management)
    monkeypatch.setattr(onset_structure, "_sector_features", sectors)
    return panel, features, baseline, industry


def test_nine_predeclared_definitions_and_exact_control() -> None:
    """场景：实际q50与动量退出组合；预期9组且控制所有非名称字段逐项精确相同。"""
    panel, features = _complete_fixture()
    baseline = _baseline(panel, features)
    definitions = _definitions(panel, features)
    assert set(definitions) == {
        "control",
        "market_diffusion5_positive",
        "market_diffusion5_ge10pct",
        "market_breadth_50_75",
        "sector_m10_ge1pct",
        "stock_relative_strength_positive",
        "stock_relative_strength_negative",
        "close_position_ge60",
        "gap_2pct",
    }
    assert definitions["control"].entry_mask[30, 6]
    for item in definitions.values():
        for field in fields(baseline):
            if field.name == "name":
                continue
            if item.name != "onset_structure_control" and field.name in (
                "entry_mask",
                "maximum_entry_gap",
            ):
                continue
            expected, actual = getattr(baseline, field.name), getattr(item, field.name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
        assert not (item.entry_mask & ~baseline.entry_mask).any()
    assert definitions["gap_2pct"].maximum_entry_gap == 0.02
    np.testing.assert_array_equal(definitions["gap_2pct"].entry_mask, baseline.entry_mask)


def test_variants_keep_array_and_policy_identity(monkeypatch) -> None:
    """场景：固定基准并保留原退出与否决；预期只控制/过滤/追价指定字段可变。"""
    panel, features, baseline, _ = _controlled(monkeypatch)
    definitions = _definitions(panel, features)
    for item in definitions.values():
        for field in fields(baseline):
            if field.name not in ("name", "entry_mask", "maximum_entry_gap"):
                assert getattr(item, field.name) is getattr(baseline, field.name)
    assert definitions["control"].entry_mask is baseline.entry_mask
    assert definitions["gap_2pct"].entry_mask is baseline.entry_mask
    assert baseline.entry_mask.all()


@pytest.mark.parametrize(
    "previous,current,positive,ten_percent",
    [
        (0.5, 0.6, True, True),
        (0.6, 0.7, True, True),
        (0.6, 0.6, False, False),
        (0.6, 0.65, True, False),
        (0.6, 0.59, False, False),
        (0.6, 0.699, True, False),
    ],
)
def test_market_diffusion_threshold_boundaries(
    monkeypatch, previous, current, positive, ten_percent
):
    """场景：广度上升/不变/下降及恰10个百分点；预期严格正增长与含边界10点各自生效。"""
    panel, features, _, _ = _controlled(monkeypatch)
    panel.market["breadth"][25] = previous
    panel.market["breadth"][30] = current
    definitions = _definitions(panel, features)
    assert definitions["market_diffusion5_positive"].entry_mask[30, 6] == positive
    assert definitions["market_diffusion5_ge10pct"].entry_mask[30, 6] == ten_percent
    assert not definitions["market_diffusion5_positive"].entry_mask[:5].any()


@pytest.mark.parametrize("offset", range(6))
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -0.1, 1.1])
def test_market_diffusion_requires_every_day_known(monkeypatch, offset, invalid):
    """场景：含源/确认端点在内六日任一缺失或越界；预期两项扩散过滤都关闭且不跨missing。"""
    panel, features, _, _ = _controlled(monkeypatch)
    panel.market["breadth"][25] = 0.4
    panel.market["breadth"][25 + offset] = invalid
    definitions = _definitions(panel, features)
    assert not definitions["market_diffusion5_positive"].entry_mask[30].any()
    assert not definitions["market_diffusion5_ge10pct"].entry_mask[30].any()
    assert definitions["control"].entry_mask[30].all()


@pytest.mark.parametrize(
    "breadth,expected",
    [
        (0.499, False),
        (0.5, True),
        (0.75, True),
        (0.751, False),
        (np.nan, False),
        (np.inf, False),
        (-0.1, False),
        (1.1, False),
    ],
)
def test_market_breadth_band_boundary(monkeypatch, breadth, expected):
    """场景：市场广度上下边界及未知；预期仅[.5,.75]含边界通过。"""
    panel, features, _, _ = _controlled(monkeypatch)
    panel.market["breadth"][30] = breadth
    assert _definitions(panel, features)["market_breadth_50_75"].entry_mask[30, 6] == expected


def test_sector_momentum_and_relative_strength_boundaries(monkeypatch) -> None:
    """场景：行业10日门槛与相对收益等值/强弱/缺失；预期等值进入正负两组，未知均拒绝。"""
    panel, features, _, industry = _controlled(monkeypatch)
    industry["momentum10"][30, :4] = [0.009, 0.01, np.nan, np.inf]
    features["return20"][30, :6] = [0.049, 0.05, 0.051, np.nan, np.inf, 0.05]
    industry["momentum20"][30, 5] = np.nan
    definitions = _definitions(panel, features)
    assert definitions["sector_m10_ge1pct"].entry_mask[30, :4].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert definitions["stock_relative_strength_positive"].entry_mask[30, :6].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert definitions["stock_relative_strength_negative"].entry_mask[30, :6].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "opening,high,low,close,expected",
    [
        (12, 15, 10, 13, True),
        (12, 15, 10, 12.99, False),
        (12, 15, 10, 15, True),
        (12, 15, 10, 10, False),
        (10, 10, 10, 10, False),
        (12, 9, 10, 13, False),
        (16, 15, 10, 13, False),
        (9, 15, 10, 13, False),
        (12, 15, 10, 16, False),
        (12, 15, 10, 9, False),
        (0, 15, 10, 13, False),
        (12, 15, 0, 13, False),
        (np.nan, 15, 10, 13, False),
        (12, np.inf, 10, 13, False),
        (12, 15, np.nan, 13, False),
        (12, 15, 10, np.nan, False),
    ],
)
def test_close_position_requires_consistent_finite_ohlc(
    monkeypatch, opening, high, low, close, expected
):
    """场景：收盘位置60%边界、一字板、非法区间、缺失/非正价；预期仅合格OHLC计算过滤。"""
    panel, features, _, _ = _controlled(monkeypatch)
    for name, value in (("open", opening), ("high", high), ("low", low), ("close", close)):
        features[name][30, 6] = value
    assert _definitions(panel, features)["close_position_ge60"].entry_mask[30, 6] == expected


@pytest.mark.parametrize("name", ["open", "high", "low"])
def test_missing_optional_price_only_disables_close_position(name) -> None:
    """场景：缺少一项复权OHLC；预期位置过滤关闭但真实控制保持有信号。"""
    panel, features = _complete_fixture()
    del features[name]
    definitions = _definitions(panel, features)
    assert definitions["control"].entry_mask[30, 6]
    assert not definitions["close_position_ge60"].entry_mask.any()


def test_missing_market_breadth_closes_baseline_and_variants() -> None:
    """场景：市场广度整列缺失；预期基准q50无法确认，因此九组均无信号。"""
    panel, features = _complete_fixture()
    del panel.market["breadth"]
    assert all(not item.entry_mask.any() for item in _definitions(panel, features).values())


@pytest.mark.parametrize("name", ["open", "high", "low", "breadth"])
def test_misaligned_optional_inputs_raise(name) -> None:
    """场景：行情一维或广度二维；预期拒绝异常shape而非跨证券广播。"""
    panel, features = _complete_fixture()
    if name == "breadth":
        panel.market[name] = np.ones((80, 1))
    else:
        features[name] = np.ones(80)
    with pytest.raises(ValueError, match="onset_structure_.*misaligned"):
        _definitions(panel, features)


@pytest.mark.parametrize("name", onset_structure.REQUIRED_FEATURES)
@pytest.mark.parametrize("missing", [True, False])
def test_required_source_fields_keep_original_validation(name, missing) -> None:
    """场景：原必需字段缺失或错维；预期原生成器拒绝，不绕过基础技术和连续窗口要求。"""
    panel, features = _complete_fixture()
    if missing:
        del features[name]
    else:
        features[name] = np.ones(80)
    with pytest.raises(ValueError, match="sector_onset_features_missing_or_misaligned"):
        _definitions(panel, features)


def test_original_twenty_one_market_day_requirement_is_preserved() -> None:
    """场景：入场前20日内股票停牌一天；预期所有变体沿用原21连续日门而取消候选。"""
    panel, features = _complete_fixture()
    features["valid"][25, 6] = 0
    assert all(not item.entry_mask[30, 6] for item in _definitions(panel, features).values())


def test_future_mutations_and_prefix_length_do_not_change_history() -> None:
    """场景：全量、60日前缀、未来完全改写；预期历史入场/评分/退出/否决逐元素一致。"""
    panel, features = _complete_fixture()
    full = _definitions(panel, features)
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
    mutated = _definitions(panel, features)
    for name, original in full.items():
        for field in ("entry_mask", "scores", "exit_mask", "entry_invalidation_mask"):
            expected = getattr(original, field)[:60]
            np.testing.assert_array_equal(expected, getattr(prefix[name], field))
            np.testing.assert_array_equal(expected, getattr(mutated[name], field)[:60])
