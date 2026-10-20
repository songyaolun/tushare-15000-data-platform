"""防御轮动的窗口完整性、预声明配置、市场日时钟与因果性测试。"""

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import Panel
from tushare_entitlement_hub.strategy_discovery.defensive import (
    REQUIRED_FEATURES,
    _historical_risk,
    _known_universe,
    build_strategies,
)


def _fixture(days: int = 160) -> tuple[Panel, dict[str, np.ndarray]]:
    """生成十只按波动率递增的股票；输入160输出(160,10)因果行情矩阵。"""
    shape = days, 10
    amplitudes = np.arange(1, 11) * 0.001
    returns = np.where(np.arange(days)[:, None] % 2, amplitudes, -amplitudes)
    features = {name: np.ones(shape) for name in REQUIRED_FEATURES}
    features.update(
        {
            "return1": returns,
            "close": 10 * np.cumprod(1 + returns, axis=0),
            "turnover": np.broadcast_to(np.r_[100, np.arange(1, 10)], shape).astype(float),
            "sector_index": np.broadcast_to(np.arange(10), shape).astype(float),
            "raw_amount": np.full(shape, 20000.0),
            "capitalization": np.full(shape, 100000.0),
            "raw_up_limit": np.full(shape, 11.0),
            "raw_down_limit": np.full(shape, 9.0),
        }
    )
    panel = Panel(
        [f"day{day:03}" for day in range(days)],
        [str(column) for column in range(10)],
        features,
        {},
        {},
    )
    return panel, features


def test_six_frozen_configurations_and_independent_sort_direction() -> None:
    """场景：低波动首股换手过热；预期纯低波动选首股，过滤版拒绝，六配置参数固定。"""
    panel, features = _fixture()
    definitions = build_strategies(panel, features)
    assert len(definitions) == len({item.name for item in definitions}) == 6
    for item in definitions:
        assert item.family == "defensive_rotation"
        assert item.entry_mask.shape == item.scores.shape == (160, 10)
        assert item.entry_mask.dtype == item.exit_mask.dtype == np.bool_
        assert not item.entry_mask[:60].any()
        assert item.entry_mask[60, 1]
        assert item.entry_mask[60, 0] == ("turnover" not in item.name)
        assert not item.entry_mask[:, 8:].any()
        assert not (item.entry_mask & item.exit_mask).any()
        assert np.isfinite(item.scores[item.entry_mask]).all()
        assert not item.policy.switching and not item.policy.stage_allocation
        assert not item.policy.force_rebalance
        assert item.policy.allocations == (0.9,) * 5
        assert (item.policy.position_weight, item.policy.sector_weight) == (0.05, 0.25)
        assert (item.policy.stop_loss, item.policy.take_profit) == (0.12, 0.95)
        assert item.policy.maximum_holding == 60
        assert item.trailing_stop == item.policy.trailing == 0
        assert item.maximum_entry_gap is None and not item.intraday_exits


def test_risk_windows_use_exact_market_sessions_and_resume_after_suspension() -> None:
    """场景：正常61日与第65日停牌；预期首个波动日为60，复牌后完整61日才恢复。"""
    _, features = _fixture()
    volatility, turnover = _historical_risk(features)
    assert np.isnan(volatility[:60]).all()
    np.testing.assert_allclose(volatility[60], np.arange(1, 11) * 0.001)
    assert np.isnan(turnover[:19]).all()
    np.testing.assert_allclose(turnover[19], np.r_[100, np.arange(1, 10)])
    features["valid"][65, 0] = 0
    features["return1"][66, 0] = 0.8
    volatility, turnover = _historical_risk(features)
    assert np.isnan(volatility[65:126, 0]).all()
    assert volatility[126, 0] == pytest.approx(0.001)
    assert np.isnan(turnover[65:85, 0]).all()
    assert turnover[85, 0] == 100


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -np.inf, -1.0, -2.0))
def test_invalid_return_is_not_replaced_with_zero(invalid: float) -> None:
    """场景：60日收益窗口包含缺值、无穷或<=-100%；预期波动保持未知至坏数据滚出。"""
    _, features = _fixture()
    features["return1"][70, 0] = invalid
    volatility, _ = _historical_risk(features)
    assert np.isnan(volatility[70:130, 0]).all()
    assert np.isfinite(volatility[130, 0])


@pytest.mark.parametrize("invalid", (np.nan, np.inf, -1.0))
def test_missing_turnover_only_blocks_turnover_variant(invalid: float) -> None:
    """场景：最稳股票20日换手缺失或非法；预期过滤版拒绝，纯低波动仍能独立入场。"""
    panel, features = _fixture()
    features["turnover"][50, 0] = invalid
    for item in build_strategies(panel, features):
        assert item.entry_mask[60, 0] == ("turnover" not in item.name)
    _, turnover = _historical_risk(features)
    assert np.isnan(turnover[50:70, 0]).all()
    assert turnover[70, 0] == 100


@pytest.mark.parametrize(
    "field,value",
    (
        ("valid", 0),
        ("close", 0),
        ("close", np.nan),
        ("beta", np.nan),
        ("capitalization", 0),
        ("capitalization", np.inf),
        ("sector_index", -1),
        ("sector_index", np.nan),
        ("sector_index", 0.5),
        ("raw_amount", 19999),
        ("raw_amount", np.inf),
        ("raw_up_limit", 10),
        ("raw_down_limit", 0),
        ("raw_up_limit", np.nan),
    ),
)
def test_common_universe_rejects_unknown_or_ineligible_data(field: str, value: float) -> None:
    """场景：共同资格各字段独立失效；预期只排除该证券，不用未来数据补资格。"""
    _, features = _fixture()
    features[field][60, 0] = value
    eligible = _known_universe(features)
    assert not eligible[60, 0]
    assert eligible[60, 1:].all()


def test_rebalance_clock_rank_buffer_and_exit_on_unknown() -> None:
    """场景：正常排名及调仓日收益缺失；预期只在5/10/20日时钟发单，缓冲区不卖。"""
    panel, features = _fixture()
    for item in build_strategies(panel, features):
        interval = int(item.name.rsplit("d", 1)[1])
        outside = np.arange(160) % interval != 0
        assert not item.entry_mask[outside].any()
        assert not item.exit_mask[outside].any()
        assert (~item.entry_mask[60] & ~item.exit_mask[60]).any()
        assert item.exit_mask[60, -1]
    features["return1"][80, 0] = np.nan
    assert all(
        item.exit_mask[80, 0] and not item.entry_mask[80, 0]
        for item in build_strategies(panel, features)
    )


def test_tied_risk_does_not_create_artificial_winners() -> None:
    """场景：十股波动完全相同或仅一只有资格；预期平均秩不凭证券顺序制造前20%。"""
    panel, features = _fixture()
    features["return1"][:] = features["return1"][:, :1]
    features["turnover"][:] = 1
    assert all(not item.entry_mask.any() for item in build_strategies(panel, features))
    features["beta"][:, 1:] = np.nan
    assert all(not item.entry_mask.any() for item in build_strategies(panel, features))


def test_future_changes_prefix_and_inputs_remain_independent() -> None:
    """场景：未来所有特征改变、截短历史、旧指标任意；预期此前信号完全相同且不改输入。"""
    panel, features = _fixture()
    snapshot = {name: values.copy() for name, values in features.items()}
    baseline = build_strategies(panel, features)
    shortened = Panel(panel.calendar[:101], panel.codes, {}, {}, {})
    prefix = build_strategies(shortened, {name: values[:101] for name, values in features.items()})
    changed = {name: values.copy() for name, values in features.items()}
    for values in changed.values():
        values[101:] = np.nan
    changed["return20"] = np.full((160, 10), 999.0)
    panel.signals["volume_reversal__sector_group"] = np.ones((160, 10), dtype=bool)
    altered = build_strategies(panel, changed)
    for original, limited, modified in zip(baseline, prefix, altered, strict=True):
        for attribute in ("entry_mask", "exit_mask", "scores"):
            np.testing.assert_allclose(
                getattr(original, attribute)[:101], getattr(limited, attribute), equal_nan=True
            )
            np.testing.assert_allclose(
                getattr(original, attribute)[:101],
                getattr(modified, attribute)[:101],
                equal_nan=True,
            )
    for name, values in snapshot.items():
        np.testing.assert_array_equal(features[name], values)


@pytest.mark.parametrize("field", REQUIRED_FEATURES)
def test_missing_or_misaligned_feature_is_rejected(field: str) -> None:
    """场景：必需特征缺失或shape错位；预期明确拒绝，不静默降级或广播填零。"""
    panel, features = _fixture()
    value = features.pop(field)
    with pytest.raises(ValueError, match="invalid_defensive_feature_matrix"):
        build_strategies(panel, features)
    features[field] = value[:, 0]
    with pytest.raises(ValueError, match="invalid_defensive_feature_matrix"):
        build_strategies(panel, features)


def test_empty_and_fully_unknown_history_fail_closed() -> None:
    """场景：空日期、空证券或全缺失；预期保留六配置及正确shape，全部无入场。"""
    for shape in ((0, 0), (0, 3), (70, 0), (70, 3)):
        panel = Panel(
            [str(day) for day in range(shape[0])],
            [str(column) for column in range(shape[1])],
            {},
            {},
            {},
        )
        features = {name: np.full(shape, np.nan) for name in REQUIRED_FEATURES}
        definitions = build_strategies(panel, features)
        assert len(definitions) == 6
        assert all(
            item.entry_mask.shape == shape and not item.entry_mask.any() for item in definitions
        )
