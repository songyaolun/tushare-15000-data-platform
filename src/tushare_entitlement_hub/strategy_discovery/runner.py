"""独立策略的同协议五年比较、可恢复结果和可审计尝试表。"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..regime_research.accounting import load_distributions
from ..regime_research.attribution import evaluate_portfolio, factor_series, matched_trade_controls
from ..regime_research.market import classify_regimes
from ..regime_research.panel import load_panel
from ..regime_research.portfolio import PortfolioPolicy, simulate_portfolio
from ..regime_research.runner import input_fingerprint, write_csv
from .combination import simulate_sleeves
from .contracts import StrategyDefinition
from .dataset import load_features
from .execution import simulate_strategy
from .hierarchy import load_hierarchy
from .refinement import defensive_exit_variants, onset_exit_variants, structural_variants
from .registry import build_candidates, neighborhood, resolve_definition


def discovery_fingerprint(
    panel: Path, raw: Path, features: Path, hierarchy_path: Path | None = None
) -> str:
    """哈希公开面板、分红、特征及明确的执行源码依赖；不读取凭据或任意目录。

    输入三个明确数据目录，输出SHA256；代码变化必须新输出目录，避免混用旧结果。
    hierarchy_path可为目录或清单，例如SIDECAR/hierarchy.json；仅额外读取该清单
    与同目录values-sector_level2_index.npy，未提供时保持原数据范围。
    除两个研究包外，纳入突破策略复用的right_side_structure和right_side_policy源码。
    """
    digest = hashlib.sha256(input_fingerprint(panel, raw).encode())
    for root, patterns in (
        (features, ("*.npy", "*.json")),
        (Path(__file__).parent, ("*.py",)),
        (
            Path(__file__).parent.parent / "equity_research",
            ("right_side_structure.py", "right_side_policy.py"),
        ),
    ):
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                digest.update(path.name.encode())
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
    if hierarchy_path is not None:
        manifest = hierarchy_path / "hierarchy.json" if hierarchy_path.is_dir() else hierarchy_path
        digest.update(b"hierarchy-sidecar")
        for path in (manifest, manifest.parent / "values-sector_level2_index.npy"):
            digest.update(path.name.encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def configuration(definition: StrategyDefinition | None) -> dict[str, Any]:
    """保存执行参数与信号摘要，不把行情矩阵写入配置；输入None返回原策略基线参数。

    数组按C序列哈希，同名策略信号改变也会拒绝恢复，NaN在评分中保留而非当作零。
    profit_protection独立保存完整数据类参数；关闭时为None，参数变化也禁止混用结果。
    """
    if definition is None:
        return {"baseline": True, "policy": asdict(baseline_policy())}
    result = {
        "name": definition.name,
        "family": definition.family,
        "policy": asdict(definition.policy),
        "maximum_entry_gap": definition.maximum_entry_gap,
        "trailing_stop": definition.trailing_stop,
        "intraday_exits": definition.intraday_exits,
        "maximum_pending_sessions": definition.maximum_pending_sessions,
        "profit_protection": asdict(definition.profit_protection)
        if definition.profit_protection is not None
        else None,
    }
    for name in ("entry_mask", "scores", "exit_mask", "entry_invalidation_mask"):
        values = getattr(definition, name)
        result[name] = (
            hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            if values is not None
            else None
        )
    return result


def baseline_policy() -> PortfolioPolicy:
    """返回上一轮固定放量反转资金对照；无输入，输出仅名称更改的原策略配置。"""
    return PortfolioPolicy(name="baseline_volume", switching=False, stage_allocation=False)


def additional_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """补充盈亏结构与持仓占用；例如一单赚10元、一单亏5元，profit_factor为2。

    无亏损的profit_factor为None并保留亏损笔数，不以无穷大参与排名。
    """
    trades = result["trades"]
    returns = np.array([trade["return"] for trade in trades])
    profits = np.array([trade["proceeds"] + trade["dividend"] - trade["cost"] for trade in trades])
    won, lost = returns > 0, returns < 0
    return {
        "mean_winning_return": float(np.mean(returns[won])) if np.any(won) else None,
        "mean_losing_return": float(np.mean(returns[lost])) if np.any(lost) else None,
        "profit_factor": float(np.sum(profits[won]) / -np.sum(profits[lost]))
        if np.any(lost)
        else None,
        "losing_trades": int(np.sum(lost)),
        "mean_holding_sessions": float(
            np.mean([trade["exit"] - trade["entry"] + 1 for trade in trades])
        )
        if trades
        else None,
        "realized_profit": float(np.sum(profits)),
        "entry_count": len(result["entry_days"]),
        "open_cost": result["open_cost"],
        "open_market_value": result["open_market_value"],
    }


def save_result(
    path: Path, result: dict[str, Any], signature: dict[str, Any], metrics: dict[str, Any]
) -> None:
    """先保存逐笔和逐日明细，再原子提交metrics完成标记；例如OUT含三份CSV和JSON。

    未有metrics的半成品重跑，不从CSV推断已完成，现金收益与市值每天同时可审计。
    """
    for name, rows in result.get("execution_evidence", {}).items():
        write_csv(path / f"{name}.csv", rows)
    if "liquidity_orders" in result:
        write_csv(path / "liquidity_orders.csv", result["liquidity_orders"])
    write_csv(path / "trades.csv", result["trades"])
    write_csv(path / "entries.csv", [{"day": day} for day in result["entry_days"]])
    write_csv(
        path / "daily.csv",
        [
            {key: value for key, value in row.items() if key != "weights"}
            | {
                "return": result["daily_returns"][index],
                "timed_market_return": result["timed_market_returns"][index],
            }
            for index, row in enumerate(result["daily"])
        ],
    )
    temporary = path / "metrics.tmp"
    temporary.write_text(json.dumps({"configuration": signature, "metrics": metrics}, indent=2))
    temporary.replace(path / "metrics.json")


def summarize(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """按资金收益和原双目标汇总全部尝试；空输入输出零配置，不制造胜率或最佳策略。"""
    eligible = [row for row in attempts if row["both_targets"]]
    return {
        "attempts": len(attempts),
        "both_target_pass": len(eligible),
        "positive_return_meeting_targets": sum(row["annualized_return"] > 0 for row in eligible),
        "highest_return": max(attempts, key=lambda row: row["annualized_return"], default=None),
        "highest_return_meeting_targets": max(
            eligible, key=lambda row: row["annualized_return"], default=None
        ),
    }


def run_discovery(
    panel_path: Path,
    features_path: Path,
    raw: Path,
    output: Path,
    *,
    group: str = "all",
    selection: Path | None = None,
    limit: int | None = None,
    hierarchy_path: Path | None = None,
) -> Path:
    """同账本运行一个策略组并保存完整证据；例如group='baseline'重现旧固定组合。

    selection可为{'names':['某配置'],'neighbors':true}，记录全样本自适应选优来源。
    onset_exits=true为行业初期策略生成八组止盈与执行方式对照，和其他扩展标志互斥。
    不同组可用独立进程和输出目录并行；limit只用于软件烟测，结果明确标记不完整。
    hierarchy_path例如SIDECAR/hierarchy.json仅补充经过验证的L2特征，不修改原Panel。
    """
    if limit is not None and limit < 1:
        raise ValueError("discovery_limit_must_be_positive")
    panel = load_panel(panel_path)
    features = load_features(features_path, panel)
    if hierarchy_path is not None:
        features |= load_hierarchy(hierarchy_path, panel)
    for day in panel.calendar:
        if day >= "20211001" and not (raw / f"dividend-{day}.json").exists():
            raise ValueError("incomplete_dividend_partitions")
    definitions: list[StrategyDefinition | None] = (
        [None] if group == "baseline" else list(build_candidates(panel, features, group))
    )
    selection_data = json.loads(selection.read_text()) if selection else None
    if selection_data is not None:
        if (
            sum(
                bool(selection_data.get(key))
                for key in ("structural", "defensive_exits", "onset_exits", "neighbors")
            )
            > 1
        ):
            raise ValueError("ambiguous_discovery_refinement")
        requested = selection_data["names"]
        available = [item for item in definitions if item is not None]
        chosen = [
            resolve_definition(available, name, features) for name in dict.fromkeys(requested)
        ]
        definitions = [
            variant
            for item in chosen
            for variant in (
                structural_variants(item, features)
                if selection_data.get("structural", False)
                else defensive_exit_variants(item)
                if selection_data.get("defensive_exits", False)
                else onset_exit_variants(item)
                if selection_data.get("onset_exits", False)
                else neighborhood(item)
                if selection_data.get("neighbors", False)
                else [item]
            )
        ]
    fingerprint = discovery_fingerprint(panel_path, raw, features_path, hierarchy_path)
    output.mkdir(parents=True, exist_ok=True)
    contract = {"fingerprint": fingerprint, "group": group, "selection": selection_data}
    contract_path = output / "experiment.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("discovery_inputs_or_selection_changed_use_new_output")
    elif any(output.iterdir()):
        raise ValueError("unversioned_discovery_output")
    contract_path.write_text(json.dumps(contract, indent=2))
    events, factors = load_distributions(raw, panel.calendar, panel.codes), factor_series(panel)
    market = panel.market
    states = classify_regimes(
        market["index"],
        market["breadth"],
        market["advancing"],
        market["amount"],
        market["limit_down"],
    )
    attempts = []
    for definition in definitions[:limit]:
        name = definition.name if definition is not None else "baseline_volume"
        path = output / name
        path.mkdir(exist_ok=True)
        signature = json.loads(json.dumps(configuration(definition)))
        completed = path / "metrics.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            if saved["configuration"] != signature:
                raise ValueError("discovery_configuration_changed")
            metrics = saved["metrics"]
        else:
            result = (
                simulate_strategy(panel, states, events, definition)
                if definition is not None
                else simulate_portfolio(panel, states, events, baseline_policy())
            )
            metrics = evaluate_portfolio(result, panel, factors) | additional_metrics(result)
            metrics["execution_model"] = result.get("execution_model", "close_next_open")
            peers = matched_trade_controls(result, panel)
            write_csv(path / "matched_trades.csv", peers.pop("records"))
            metrics["matched_peers"] = peers
            save_result(path, result, signature, metrics)
        row = {"name": name, "family": definition.family if definition else "baseline"} | {
            key: metrics[key]
            for key in (
                "annualized_return",
                "maximum_drawdown",
                "win_rate",
                "mean_trade_return",
                "closed_trades",
                "maximum_entry_gap",
                "both_targets",
                "mean_exposure",
                "mean_beta",
                "profit_factor",
                "mean_holding_sessions",
                "stale_value",
            )
        }
        attempts.append(row)
        write_csv(output / "attempts.csv", attempts)
        print(json.dumps(row), flush=True)
    write_csv(output / "attempts.csv", attempts)
    report = summarize(attempts) | {
        "complete_grid": limit is None,
        "all_data_used_for_optimization": True,
        "historical_pit_certified": False,
        "benchmark": "000985.CSI / H00985.CSI",
        "experiment": contract,
        "limitations": [
            "Full-sample optimization; intervals are not adjusted for strategy selection.",
            "Inherited incomplete historical ST, rights issues and corporate actions.",
            "Historical industry reconstruction is not historical concept membership.",
            "Stale holdings are marked separately; index controls are theoretical.",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def run_combination(
    panel_path: Path,
    features_path: Path,
    raw: Path,
    output: Path,
    selection: Path,
    *,
    hierarchy_path: Path | None = None,
) -> Path:
    """按声明资金权重重跑具名策略组合，并复用审计输出；例如两策略weights=[.5,.5]。

    selection包含name/names/weights，names可使用已声明邻域或__intraday后缀。
    使用不同结果目录记录每组实际资金分配；不把大账户净值曲线线性平均当成交结果。
    例如hierarchy_path=SIDECAR允许组合选择L2策略，层级清单和矩阵同时进入恢复指纹。
    """
    panel = load_panel(panel_path)
    features = load_features(features_path, panel)
    if hierarchy_path is not None:
        features |= load_hierarchy(hierarchy_path, panel)
    for day in panel.calendar:
        if day >= "20211001" and not (raw / f"dividend-{day}.json").exists():
            raise ValueError("incomplete_dividend_partitions")
    selected = json.loads(selection.read_text())
    available = build_candidates(panel, features, "all")
    definitions = [resolve_definition(available, name, features) for name in selected["names"]]
    signature = {
        "name": selected["name"],
        "weights": selected["weights"],
        "sleeves": [configuration(definition) for definition in definitions],
    }
    signature = json.loads(json.dumps(signature))
    contract = {
        "fingerprint": discovery_fingerprint(panel_path, raw, features_path, hierarchy_path),
        "configuration": signature,
    }
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "experiment.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("combination_inputs_changed_use_new_output")
    elif any(output.iterdir()):
        raise ValueError("unversioned_combination_output")
    contract_path.write_text(json.dumps(contract, indent=2))
    completed = output / "metrics.json"
    if completed.exists():
        if json.loads(completed.read_text())["configuration"] != signature:
            raise ValueError("combination_configuration_changed")
        return completed
    events, factors = load_distributions(raw, panel.calendar, panel.codes), factor_series(panel)
    market = panel.market
    states = classify_regimes(
        market["index"],
        market["breadth"],
        market["advancing"],
        market["amount"],
        market["limit_down"],
    )
    result = simulate_sleeves(panel, states, events, definitions, selected["weights"])
    metrics = evaluate_portfolio(result, panel, factors) | additional_metrics(result)
    metrics["execution_model"] = [definition.intraday_exits for definition in definitions]
    peers = matched_trade_controls(result, panel)
    write_csv(output / "matched_trades.csv", peers.pop("records"))
    metrics["matched_peers"] = peers
    save_result(output, result, signature, metrics)
    return completed
