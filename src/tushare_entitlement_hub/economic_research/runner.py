"""短期反转四臂实验的离线数据契约、完整账本与不可复用输出。"""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..regime_research.accounting import load_distributions
from ..regime_research.attribution import block_alpha, evaluate_portfolio, factor_series
from ..regime_research.market import classify_regimes
from ..regime_research.panel import Panel, load_panel
from ..regime_research.runner import write_csv
from ..strategy_discovery.runner import additional_metrics, save_result
from .execution import simulate_reversal
from .reversal import ReversalStrategy, build_strategies

RULES = {
    "universe": "eligible_top_1000_known_capitalization_then_top_500_mean_20day_amount",
    "signal": "bottom_20pct_of_universe_negative_5day_return_relative_to_same_industry_pool",
    "ranking": "larger_negative_relative_return_first_stable_security_order_ties",
    "matching": "same_day_pool_industry_non_treated_capitalization_liquidity_nearest_without_reuse",
    "entry_clock": "frozen_dispatch_close_to_immediate_next_open_only",
    "delayed_clock": "frozen_source_signal_shifted_one_market_session",
    "unfilled_order": "cancel_at_dispatch_plus_one_close_without_later_fill",
    "holding": "five_market_sessions_from_actual_entry_then_next_executable_open",
    "mean_holding_sessions": "exit_open_day_minus_actual_entry_open_day",
    "mean_touched_calendar_sessions": "exit_day_minus_entry_day_plus_one_legacy_calendar_count",
    "exit": "time_only_no_fixed_price_trailing_signal_or_allocation_exit",
    "primary_control": "independently_replayed_matched_control_account",
    "source_signal": "original_signal_day_index_before_dispatch_delay",
    "historical_pit_certified": False,
    "sample_semantics": "full_sample_descriptive_not_out_of_sample",
}


def _hash_files(paths: list[tuple[str, Path]]) -> str:
    """按明确标签和文件字节流摘要，不读取环境凭据；例同内容不同标签得到不同SHA256。"""
    digest = hashlib.sha256()
    for label, path in paths:
        digest.update(label.encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> str:
    """摘要整个本地Python包的源码以覆盖研究和执行的传递依赖；无输入输出64位SHA256。

    包含regime_research、strategy_discovery、economic_research、strategy_optimization
    及它们引用的共同解析/结构模块；例如任一费用公式源码变化都会改变结果。
    """
    package = Path(__file__).resolve().parent.parent
    return _hash_files(
        [(path.relative_to(package).as_posix(), path) for path in sorted(package.rglob("*.py"))]
    )


def data_fingerprint(panel_path: Path, raw: Path) -> str:
    """一次摘要面板数组/清单和分红加载器实际读取的文件；例如修改分红支付日会改变摘要。

    原始目录只读取dividend分区及distribution-resolutions.json；probe与无关文件不参与。
    不再次扫描真实数据用于运行后检查，源码漂移由独立的source_fingerprint检测。
    """
    paths = [(f"panel/{path.name}", path) for path in sorted(panel_path.glob("*.npy"))]
    paths.append(("panel/panel.json", panel_path / "panel.json"))
    paths.extend(
        (f"raw/{path.name}", path)
        for path in sorted(raw.glob("dividend-*.json"))
        if path.name != "dividend-probe.json"
    )
    resolution = raw / "distribution-resolutions.json"
    if resolution.exists():
        paths.append(("raw/distribution-resolutions.json", resolution))
    return _hash_files(paths)


def _array_fingerprint(values: NDArray[Any]) -> str:
    """分块摘要C序数组及shape/dtype，不一次复制全市场矩阵；例同值C/F布局输出相同摘要。"""
    digest = hashlib.sha256(
        json.dumps({"shape": values.shape, "dtype": str(values.dtype)}).encode()
    )
    for offset in range(0, len(values), 128):
        digest.update(np.ascontiguousarray(values[offset : offset + 128]).tobytes())
    return digest.hexdigest()


def configuration(strategy: ReversalStrategy) -> dict[str, Any]:
    """保存策略参数、冻结资格/分数/原信号日摘要；例改动source_days会改变configuration_hash。"""
    signature = {
        "name": strategy.name,
        "policy": asdict(strategy.policy),
        "entry_mask_hash": _array_fingerprint(strategy.entry_mask),
        "scores_hash": _array_fingerprint(strategy.scores),
        "source_days_hash": _array_fingerprint(strategy.source_days),
        "matched_pairs_hash": _array_fingerprint(
            np.asarray(strategy.matched_pairs, dtype=np.int64).reshape(-1, 3)
        ),
        "maximum_pending_sessions": strategy.maximum_pending_sessions,
        "signal_count": int(np.count_nonzero(strategy.entry_mask)),
    }
    signature["configuration_hash"] = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()
    return signature


def _validate_inputs(panel: Panel, raw: Path, start: str, output: Path) -> None:
    """验证真实日期、目标期逐日分红与全新输出；例非空OUT或缺少某交易日分区直接拒绝。"""
    try:
        valid_start = datetime.strptime(start, "%Y%m%d").strftime("%Y%m%d") == start
    except ValueError:
        valid_start = False
    if (
        not valid_start
        or not panel.calendar
        or panel.calendar != sorted(set(panel.calendar))
        or start > panel.calendar[-1]
    ):
        raise ValueError("invalid_reversal_start_or_calendar")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("reversal_output_must_be_empty")
    for day in panel.calendar:
        if day >= start and not (raw / f"dividend-{day}.json").is_file():
            raise ValueError("incomplete_reversal_dividend_partitions")


def matched_account_comparison(base: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """按同日期真实账户收益比较基础与匹配账户，不把年化差当作已验证alpha。

    例如base当日收益.01、control=.005输出日差.005；日期不一致立即拒绝。
    连续日期块bootstrap仅描述样本内日收益差均值的不确定性，未校正策略选择。
    """
    if base["dates"] != control["dates"]:
        raise ValueError("reversal_comparison_dates_mismatch")
    spread = np.asarray(base["returns"]) - np.asarray(control["returns"])
    interval = block_alpha(spread, np.empty((len(spread), 0))) | {"selection_adjusted": False}
    return {
        "total_return_difference": base["metrics"]["total_return"]
        - control["metrics"]["total_return"],
        "annualized_return_difference": base["metrics"]["annualized_return"]
        - control["metrics"]["annualized_return"],
        "mean_exposure_difference": base["mean_exposure"] - control["mean_exposure"],
        "daily_return_spread": [
            {
                "date": day,
                "reversal_return": float(left),
                "matched_control_return": float(right),
                "spread": float(difference),
            }
            for day, left, right, difference in zip(
                base["dates"], base["returns"], control["returns"], spread, strict=True
            )
        ],
        "descriptive_spread_interval": interval,
        "interpretation": (
            "Full-sample descriptive account differences and factor estimates; not validated alpha."
        ),
    }


def run_reversal_experiment(
    panel_path: Path, raw: Path, output: Path, start: str = "20211018"
) -> Path:
    """运行四个固定短反账户并原子提交总报告；例如(PANEL,RAW,OUT)返回OUT/report.json。

    仅接受不存在或空输出，不恢复半成品。目标期每天必须有分红分区，所有资金账户独立
    重放；matched_control是实际可交易账户。数据摘要只计算一次，开始及全部回放后
    复核整包源码摘要；源码漂移留下不完整审计文件并拒绝生成report完成标记。
    """
    panel = load_panel(panel_path)
    _validate_inputs(panel, raw, start, output)
    initial_source = source_fingerprint()
    inputs = data_fingerprint(panel_path, raw)
    strategies = build_strategies(panel)
    if len(strategies) != 4 or {strategy.name for strategy in strategies} != {
        "reversal",
        "delayed",
        "cost_stress",
        "matched_control",
    }:
        raise ValueError("reversal_requires_four_unique_strategies")
    signatures = [configuration(strategy) for strategy in strategies]
    events = load_distributions(raw, panel.calendar, panel.codes)
    factors = factor_series(panel)
    states = classify_regimes(
        panel.market["index"],
        panel.market["breadth"],
        panel.market["advancing"],
        panel.market["amount"],
        panel.market["limit_down"],
    )
    contract = {
        "version": "economic-reversal/v1",
        "start": start,
        "calendar_first": panel.calendar[0],
        "calendar_last": panel.calendar[-1],
        "input_hash": inputs,
        "source_hash": initial_source,
        "rules": RULES,
        "configurations": signatures,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment.json").write_text(json.dumps(contract, indent=2))
    comparisons = []
    accounts = {}
    for strategy, signature in zip(strategies, signatures, strict=True):
        path = output / strategy.name
        path.mkdir()
        result = simulate_reversal(panel, states, events, strategy, start=start)
        metrics = evaluate_portfolio(result, panel, factors) | additional_metrics(result)
        metrics["mean_touched_calendar_sessions"] = metrics["mean_holding_sessions"]
        metrics["mean_holding_sessions"] = (
            float(np.mean([trade["exit"] - trade["entry"] for trade in result["trades"]]))
            if result["trades"]
            else None
        )
        metrics["execution_model"] = result["execution_model"]
        save_result(path, result, signature, metrics)
        write_csv(
            path / "matched_pairs.csv",
            [
                {
                    "source_signal": day,
                    "treatment": treatment,
                    "treatment_code": panel.codes[treatment],
                    "control": control,
                    "control_code": panel.codes[control],
                }
                for day, treatment, control in strategy.matched_pairs
            ],
        )
        comparisons.append({"name": strategy.name, **metrics})
        accounts[strategy.name] = {
            "dates": [row["date"] for row in result["daily"]],
            "returns": result["daily_returns"],
            "metrics": metrics,
            "mean_exposure": float(np.mean([row["exposure"] for row in result["daily"]])),
        }
    write_csv(
        output / "comparison.csv",
        [
            {
                key: row[key]
                for key in (
                    "name",
                    "total_return",
                    "annualized_return",
                    "maximum_drawdown",
                    "closed_trades",
                    "win_rate",
                    "both_targets",
                    "maximum_entry_gap",
                    "fees",
                    "reconciliation_error",
                )
            }
            for row in comparisons
        ],
    )
    matched = matched_account_comparison(accounts["reversal"], accounts["matched_control"])
    write_csv(output / "matched_account_spread.csv", matched["daily_return_spread"])
    by_name = {strategy.name: strategy for strategy in strategies}
    first_target = next(index for index, day in enumerate(panel.calendar) if day >= start)
    candidates = int(np.count_nonzero(by_name["reversal"].entry_mask[first_target:]))
    paired = sum(day >= first_target for day, _, _ in by_name["matched_control"].matched_pairs)
    report = {
        "experiment": contract,
        "strategies": comparisons,
        "completed": True,
        "primary_control": "matched_control",
        "matched_account_comparison": matched,
        "matching": {
            "source_period_start": start,
            "base_candidate_count": candidates,
            "matched_count": paired,
            "unmatched_count": candidates - paired,
            "matching_rate": paired / candidates if candidates else None,
            "unmatched_handling": "retain_base_candidates_without_cross_industry_substitution",
        },
        "interpretation": (
            "Full-sample descriptive comparison; annualized return differences and factor "
            "estimates are not validated alpha. Matched accounts can differ in exposure and risk."
        ),
    }
    if source_fingerprint() != initial_source:
        raise ValueError("reversal_source_changed_during_run")
    temporary = output / "report.tmp"
    temporary.write_text(json.dumps(report, indent=2))
    completed = output / "report.json"
    temporary.replace(completed)
    return completed
