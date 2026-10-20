"""全样本策略探索的离线因果性与独立成交复算，不使用真实凭据或网络。"""

import json
from datetime import date, timedelta

import numpy as np
import pytest

from tushare_entitlement_hub.signal_research.simulation import (
    SimulationConfiguration,
    simulate_signal_trades,
)
from tushare_entitlement_hub.strategy_optimization.dataset import (
    daily_percentiles,
    membership_matrix,
    read_envelope,
)
from tushare_entitlement_hub.strategy_optimization.execution import (
    ExitPolicy,
    entry_indices,
    exit_grid,
    independent_trades,
    outcome_arrays,
)
from tushare_entitlement_hub.strategy_optimization.features import (
    compute_features,
    kdj,
    lag,
    ratio,
    rolling,
)
from tushare_entitlement_hub.strategy_optimization.search import maximum_signal_gap
from tushare_entitlement_hub.strategy_optimization.strategies import (
    StrategyDefinition,
    strategy_grid,
    strategy_manifest,
    strategy_masks,
)


def synthetic_rows(count=160, seed=12):
    """构建可重复的波动行情与市场日历；输入160，输出160根有效价格和日期。"""
    random = np.random.default_rng(seed)
    closes = 10 * np.exp(np.cumsum(random.normal(0.0005, 0.027, count)))
    openings = closes * np.exp(random.normal(0, 0.012, count))
    calendar = [
        (date(2023, 7, 1) + timedelta(days=index)).strftime("%Y%m%d") for index in range(count)
    ]
    rows = [
        dict(
            ts_code="000001.SZ",
            trade_date=day,
            open=float(openings[index]),
            high=float(max(openings[index], closes[index]) * 1.01),
            low=float(min(openings[index], closes[index]) * 0.99),
            close=float(closes[index]),
            vol=100.0,
            amount=1000.0,
            adj_factor=1.0,
            up_limit=float(max(openings[index], closes[index]) * 1.1),
            down_limit=float(min(openings[index], closes[index]) * 0.9),
        )
        for index, day in enumerate(calendar)
    ]
    return calendar, rows


def test_rolling_lag_and_safe_ratios():
    """场景：预热、缺值与零分母；预期：所有不完整窗口未知，不循环引用未来。"""
    values = np.array([1.0, 3.0, 5.0])
    np.testing.assert_allclose(lag(values), [np.nan, 1, 3])
    assert np.isnan(lag(values, 4)).all()
    np.testing.assert_allclose(rolling(values, 2, "mean"), [np.nan, 2, 4])
    assert np.isnan(rolling(values, 4, "mean")).all()
    np.testing.assert_allclose(
        ratio(np.array([2.0, 2.0, np.inf]), np.array([1.0, 0.0, 2.0])), [2, np.nan, np.nan]
    )
    for period in (0, True, -1):
        with pytest.raises(ValueError):
            lag(values, period)
    for window, operation in [(0, "mean"), (True, "mean"), (1, "bad")]:
        with pytest.raises(ValueError):
            rolling(values, window, operation)


def test_kdj_hand_calculation_and_zero_range():
    """场景：九日RSV=100与零振幅；预期：K/D递推与手算相等，零带宽不虚构信号。"""
    high = np.full(12, 10.0)
    low = np.zeros(12)
    fast, slow, oscillator = kdj(high, low, high)
    assert np.isnan(fast[:8]).all()
    assert fast[8] == pytest.approx(200 / 3)
    assert slow[8] == pytest.approx(500 / 9)
    assert oscillator[8] == pytest.approx(800 / 9)
    assert np.isnan(kdj(high, high, high)[0]).all()


def test_all_features_are_prefix_causal_and_split_invariant():
    """场景：任意历史前缀、未来价格改动和拆股；预期：已知特征不变且复权锚固定。"""
    calendar, rows = synthetic_rows()
    complete = compute_features(rows, calendar)
    for length in (1, 9, 20, 61, 100):
        prefix = compute_features(rows, calendar[:length])
        for name in complete:
            np.testing.assert_allclose(prefix[name], complete[name][:length], equal_nan=True)
    split = [dict(row) for row in rows]
    for row in split[80:]:
        row["adj_factor"] = 2.0
        for name in ("open", "high", "low", "close", "up_limit", "down_limit"):
            row[name] /= 2
    transformed = compute_features(split, calendar)
    for name in ("close", "ma20", "ma60", "atr", "kdj_k", "boll_position"):
        np.testing.assert_allclose(transformed[name], complete[name], equal_nan=True)
    with pytest.raises(ValueError):
        compute_features(rows + [rows[0]], calendar)
    with pytest.raises(ValueError):
        compute_features(rows, list(reversed(calendar)))
    assert np.isnan(compute_features([], calendar)["close"]).all()


def test_membership_effective_intervals_conflicts_and_ranking(tmp_path):
    """场景：剔除日、行业冲突、无日期和并列分位；预期：冲突失败关闭，不回填未来成员。"""
    calendar = ["20230101", "20230102", "20230103"]
    original = dict(
        ts_code="A.SH", l1_code="X", in_date="20230101", out_date="20230103", is_new="N"
    )
    values, diagnostics = membership_matrix(
        [
            original,
            original,
            {**original, "l1_code": "Y", "in_date": "20230102"},
            {**original, "ts_code": "B.SH", "in_date": None},
            {**original, "ts_code": "C.SH", "out_date": None},
        ],
        calendar,
        ["X", "Y"],
    )
    assert values["A.SH"].tolist() == [0, -1, -1]
    assert diagnostics["conflicting_security_days"] == 1
    assert diagnostics["invalid_or_outside_universe_membership_rows"] == 2
    actual = daily_percentiles(np.array([[1.0, 2.0, np.nan], [2.0, 2.0, 1.0], [3.0, 1.0, np.nan]]))
    np.testing.assert_allclose(actual, [[0, 0.75, np.nan], [0.5, 0.75, 0.5], [1, 0, np.nan]])
    path = tmp_path / "raw.json"
    path.write_text(
        json.dumps({"response": {"code": 0, "data": {"fields": ["x"], "items": [[1]]}}})
    )
    assert read_envelope(path) == [{"x": 1}]


@pytest.mark.parametrize(
    "policy", [ExitPolicy(0.025, 0.06, 20), ExitPolicy(0.08, 0.03, 40), ExitPolicy(0.015, 0.10, 60)]
)
def test_vector_execution_matches_independent_scalar_engine(policy):
    """场景：波动、限板、停牌与订单到期；预期：向量结果逐笔等于原独立状态机成交。"""
    calendar, rows = synthetic_rows()
    for index in (5, 6, 7, 35, 70, 90):
        rows[index]["up_limit"] = rows[index]["open"]
    for index in (22, 23, 24, 55, 58):
        rows[index]["down_limit"] = rows[index]["open"]
    rows[30]["vol"] = 0
    features = compute_features(rows, calendar)
    outcomes = outcome_arrays(features, calendar, policy)
    candidates = np.arange(0, len(calendar), 2, dtype=np.int64)
    selected, skipped = independent_trades(candidates, outcomes, len(calendar))
    signals = [
        dict(ts_code="000001.SZ", confirmation_date=calendar[index], experiment_id="test")
        for index in candidates
    ]
    scalar = simulate_signal_trades(
        rows,
        signals,
        SimulationConfiguration(
            maximum_entry_wait_sessions=3,
            maximum_holding_sessions=policy.holding_sessions,
            price_stop_loss_fraction=policy.stop_loss,
            price_take_profit_fraction=policy.take_profit,
        ),
    )
    assert len(skipped) == sum(trade["status"] == "skipped" for trade in scalar)
    retained = [trade for trade in scalar if trade["status"] != "skipped"]
    for index, trade in zip(selected, retained, strict=True):
        entry = int(outcomes["entry_index"][index])
        exit_position = int(outcomes["exit_index"][index])
        assert trade["entry_date"] == (
            date.fromisoformat(calendar[entry]).isoformat() if entry >= 0 else None
        )
        assert trade["exit_date"] == (
            date.fromisoformat(calendar[exit_position]).isoformat()
            if exit_position < len(calendar)
            else None
        )
        if trade["status"] == "closed":
            assert trade["fee_adjusted_research_return"] == pytest.approx(outcomes["return"][index])
        else:
            assert np.isnan(outcomes["return"][index])


def test_execution_and_frequency_boundaries():
    """场景：无信号、集中信号、缺行情和非法参数；预期：频率按最长空窗且非法输入拒绝。"""
    assert maximum_signal_gap([3, 8], 0, 10) == 5
    assert maximum_signal_gap([], 0, 10) == 11
    assert maximum_signal_gap([5, 5, 6], 0, 30) == 24
    assert len(exit_grid()) == 45
    for arguments in [(0, 0.1, 20), (0.1, 0, 20), (0.1, 0.1, True), (0.1, 0.1, 0)]:
        with pytest.raises(ValueError):
            ExitPolicy(*arguments)
    calendar, rows = synthetic_rows(3)
    features = compute_features(rows, calendar)
    with pytest.raises(ValueError):
        entry_indices(features, 0)
    with pytest.raises(ValueError):
        outcome_arrays(features, [], ExitPolicy(0.1, 0.1, 2))
    with pytest.raises(ValueError):
        outcome_arrays(features, calendar, ExitPolicy(0.1, 0.1, 2), np.array([100, 1, 2]))
    with pytest.raises(ValueError):
        independent_trades(np.array([1, 1]), {}, 3)
    for argument in (0, float("nan"), 1):
        with pytest.raises(ValueError):
            SimulationConfiguration(price_take_profit_fraction=argument)
    with pytest.raises(ValueError):
        SimulationConfiguration(maximum_entry_wait_sessions=True)


def test_strategy_masks_require_independent_group_members():
    """场景：同业信息缺失或只有四只同行；预期：P02条件不通过，纯技术组仍可独立研究。"""
    calendar, rows = synthetic_rows()
    features = compute_features(rows, calendar)
    for name in (
        "sector_peers",
        "sector_coverage",
        "sector_breadth",
        "sector_mean_return",
        "sector_heat",
        "sector_amount_ratio",
        "sector_trend",
        "market_breadth",
    ):
        features[name] = np.full(len(calendar), np.nan)
    masks = strategy_masks(features)
    assert len(masks) == 60 and len(strategy_manifest()) == 60
    assert not any(mask.any() for name, mask in masks.items() if "sector" in name)
    assert len({item.name for item in strategy_grid()}) == 60
    with pytest.raises(ValueError):
        strategy_masks(features, (StrategyDefinition("unknown"),))
    assert strategy_masks(features, ()) == {}
