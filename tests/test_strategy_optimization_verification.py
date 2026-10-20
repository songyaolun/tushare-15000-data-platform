"""最终候选交付前独立复算的正常与拒绝路径。"""

import json
from datetime import date, timedelta

import numpy as np
import pytest
from typer.testing import CliRunner

from tushare_entitlement_hub.strategy_optimization import cli, verification
from tushare_entitlement_hub.strategy_optimization.execution import ExitPolicy
from tushare_entitlement_hub.strategy_optimization.features import compute_features
from tushare_entitlement_hub.strategy_optimization.strategies import StrategyDefinition


def verified_input(root, count=1):
    """生成固定随机种子的完整微型特征；输入目录与证券数，输出可供独立复算的目录。"""
    root.mkdir()
    calendar = [
        (date(2021, 10, 18) + timedelta(days=index)).strftime("%Y%m%d") for index in range(160)
    ]
    calendar[-1] = "20261016"
    random = np.random.default_rng(12)
    closes = 10 * np.exp(np.cumsum(random.normal(0.0005, 0.027, 160)))
    rows = [
        dict(
            trade_date=day,
            open=float(closes[index]),
            high=float(closes[index] * 1.01),
            low=float(closes[index] * 0.99),
            close=float(closes[index]),
            vol=100.0,
            amount=1000.0,
            adj_factor=1.0,
            up_limit=float(closes[index] * 1.1),
            down_limit=float(closes[index] * 0.9),
        )
        for index, day in enumerate(calendar)
    ]
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
        features[name] = np.full(160, np.nan)
    codes = [f"{index:06}.SZ" for index in range(count)]
    for code in codes:
        np.savez_compressed(root / f"{code}.npz", allow_pickle=False, **features)
    (root / "dataset.json").write_text(
        json.dumps({"calendar": calendar, "securities": codes, "source_fingerprint": "synthetic"})
    )
    return root


def test_independent_final_candidate_and_empty_candidate(tmp_path):
    """场景：250个可复算候选与无信号策略；预期：逐笔日期收益一致，空策略不能通过目标。"""
    source = verified_input(tmp_path / "source", 250)
    definition = StrategyDefinition("range_bottom")
    policy = ExitPolicy(0.025, 0.06, 20)
    path = verification.verify_candidate(source, tmp_path / "verified", definition, policy)
    summary = json.loads(path.read_text())
    assert summary["closed_trades"] > 0
    assert summary["independent_execution_verified"] is True
    assert summary["historical_pit_certified"] is False
    assert (path.parent / "trades.csv").exists()
    with pytest.raises(ValueError, match="verified_output_exists"):
        verification.verify_candidate(source, tmp_path / "verified", definition, policy)
    empty = verification.verify_candidate(
        source, tmp_path / "empty", StrategyDefinition("range_bottom", "sector_group"), policy
    )
    assert json.loads(empty.read_text())["win_rate"] is None


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("count", "count"),
        ("overlap", "overlap"),
        ("dates", "dates"),
        ("returns", "return"),
    ],
)
def test_final_candidate_refuses_independent_disagreement(tmp_path, monkeypatch, mutation, reason):
    """场景：独立模拟器计数、重叠、日期或收益被破坏；预期：拒绝发布验证通过清单。"""
    source = verified_input(tmp_path / "source")
    original = verification.simulate_signal_trades

    def altered(*args):
        """先真实模拟再注入单项不一致；输入原参数，输出明确损坏的验证反例。"""
        trades = original(*args)
        if mutation == "count":
            return trades[:-1]
        target = next(trade for trade in trades if trade["status"] == "closed")
        if mutation == "overlap":
            target["status"] = "skipped"
        elif mutation == "dates":
            target["entry_date"] = target["confirmation_date"]
        else:
            target["fee_adjusted_research_return"] = 1.0
        return trades

    monkeypatch.setattr(verification, "simulate_signal_trades", altered)
    with pytest.raises(ValueError, match=reason):
        verification.verify_candidate(
            source,
            tmp_path / "output",
            StrategyDefinition("range_bottom"),
            ExitPolicy(0.025, 0.06, 20),
        )
    assert not (tmp_path / "output/verified.json").exists()


def test_optimization_cli_passes_explicit_inputs(tmp_path, monkeypatch):
    """场景：三个离线CLI入口；预期：路径和策略参数完整传递，不触发网络或扩大证券范围。"""
    observed = []

    def build(*args):
        """记录prepare的三个路径；输入路径参数，输出确定性清单路径。"""
        observed.append(args)
        return tmp_path / "dataset.json"

    def search(*args, **kwargs):
        """记录搜索的显式范围；输入路径与限额，输出确定性搜索路径。"""
        observed.append((args, kwargs))
        return tmp_path / "search.csv"

    def verify(*args):
        """记录选定策略和退出参数；输入配置，输出确定性复核路径。"""
        observed.append(args)
        return tmp_path / "verified.json"

    monkeypatch.setattr(cli, "build_dataset", build)
    monkeypatch.setattr(cli, "run_search", search)
    monkeypatch.setattr(cli, "verify_candidate", verify)
    for arguments in [
        ["prepare", "history", "sectors", "output"],
        ["search", "features", "output", "--maximum-securities", "10"],
        ["verify", "features", "output", "range_bottom", "none", ".025", ".06", "20"],
    ]:
        result = CliRunner().invoke(cli.optimization_app, arguments)
        assert result.exit_code == 0, result.output
    assert observed[1][1]["maximum_securities"] == 10
    assert observed[2][-1] == ExitPolicy(0.025, 0.06, 20)


def test_open_position_marks_stale_quotes_without_claiming_realized_profit():
    """场景：期末停牌无报价，末次报价下跌10%；预期：保留浮亏、陈旧日数，不计已退出。"""
    result = verification.open_position_mark(
        {"raw_close": np.array([10.0, 9.0, np.nan]), "raw_adj_factor": np.ones(3)},
        ["20261014", "20261015", "20261016"],
        {"ts_code": "A.SH", "entry_date": "2026-10-14", "entry_price": 10.0},
    )
    assert result["gross_mark_return"] == pytest.approx(-0.1)
    assert result["stale_sessions"] == 1


def test_final_report_includes_positions_still_open_at_sample_end(tmp_path):
    """场景：最大持有期超过样本且止盈止损未触发；预期：未平仓计数与浮盈亏记录一一对应。"""
    source = verified_input(tmp_path / "source")
    path = verification.verify_candidate(
        source,
        tmp_path / "output",
        StrategyDefinition("range_bottom"),
        ExitPolicy(0.99, 0.99, 1000),
    )
    result = json.loads(path.read_text())
    assert len(result["open_positions"]) == result["counts"]["open"]
    assert result["win_rate_if_all_open_positions_lose"] == 0
