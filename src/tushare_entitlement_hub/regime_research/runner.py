"""全样本市场阶段研究：预声明对照、参数邻域、逐日结果和完整尝试记录。"""

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .accounting import load_distributions
from .attribution import evaluate_portfolio, factor_series, matched_trade_controls
from .market import RegimeParameters, classify_regimes
from .panel import load_panel
from .portfolio import PortfolioPolicy, simulate_portfolio


def research_grid() -> list[tuple[RegimeParameters, PortfolioPolicy]]:
    """初轮48项、第二轮54项，再对接近可行的方案作18项仓位/期限邻域检验。

    输入无，输出唯一名称及参数；所有结果保留，使用五年全样本，无留出集。
    """
    regime = RegimeParameters()
    base = PortfolioPolicy()
    candidates = [
        (regime, replace(base, name="fixed_volume", switching=False, stage_allocation=False)),
        (regime, replace(base, name="volume_allocation", switching=False)),
        (regime, replace(base, name="switch_fixed_allocation", stage_allocation=False)),
        (regime, replace(base, name="switch_allocation")),
        (regime, replace(base, name="switch_trailing", trailing=0.06)),
        (regime, replace(base, name="switch_early", early_exit=5)),
        (regime, replace(base, name="switch_beta", beta_tilt=0.3)),
        (regime, replace(base, name="fixed_mixture", mixture=True, stage_allocation=False)),
    ]
    for target in (0.025, 0.04, 0.06):
        for stop in (0.06, 0.10):
            for early in (0, 5):
                for allocation in (0, 1, 2):
                    weights = (
                        (0.3, 0.9, 0.6, 0.1, 0),
                        (0.4, 1, 0.7, 0.1, 0),
                        (0.6, 1, 0.8, 0.2, 0),
                    )[allocation]
                    name = f"grid_tp{target:g}_sl{stop:g}_early{early}_alloc{allocation}"
                    candidates.append(
                        (
                            regime,
                            replace(
                                base,
                                name=name,
                                take_profit=target,
                                stop_loss=stop,
                                early_exit=early,
                                allocations=weights,
                            ),
                        )
                    )
    for advance, decline in ((0.55, 0.4), (0.65, 0.4), (0.6, 0.35), (0.6, 0.45)):
        parameters = replace(regime, advance_breadth=advance, decline_breadth=decline)
        candidates.append((parameters, replace(base, name=f"neighbor_b{advance:g}_{decline:g}")))
    for improvement in (0.05, 0.15):
        candidates.append(
            (
                replace(regime, repair_improvement=improvement),
                replace(base, name=f"neighbor_repair{improvement:g}"),
            )
        )
    for ranking in ("original", "oversold", "liquid"):
        for target in (0.015, 0.025, 0.04):
            for family, horizon in (
                ("volume_reversal__sector_group", 20),
                ("boll_reentry__sector_group", 60),
            ):
                for mode in ("static", "soft", "switch"):
                    name = f"second_{ranking}_tp{target:g}_{family.split('__')[0]}_{mode}"
                    candidates.append(
                        (
                            regime,
                            replace(
                                base,
                                name=name,
                                switching=mode == "switch",
                                stage_allocation=mode != "static",
                                force_rebalance=False,
                                allow_weak_entries=True,
                                ranking=ranking,
                                take_profit=target,
                                maximum_holding=horizon,
                                recovery_family=family,
                                advance_family="volume_reversal__sector_group",
                                allocations=(0.6, 1.0, 0.8, 0.3, 0.3),
                            ),
                        )
                    )
    for ranking in ("original", "oversold"):
        for horizon in (20, 40, 60):
            for floor in (0.3, 0.5, 0.7):
                candidates.append(
                    (
                        regime,
                        replace(
                            base,
                            name=f"third_{ranking}_h{horizon}_floor{floor:g}",
                            force_rebalance=False,
                            allow_weak_entries=True,
                            ranking=ranking,
                            take_profit=0.015,
                            maximum_holding=horizon,
                            recovery_family="boll_reentry__sector_group",
                            advance_family="volume_reversal__sector_group",
                            allocations=(max(0.6, floor), 1.0, 0.8, floor, floor),
                        ),
                    )
                )
    unique = []
    signatures = set()
    for parameters, policy in candidates:
        signature = (parameters, replace(policy, name=""))
        if signature not in signatures:
            signatures.add(signature)
            unique.append((parameters, policy))
    return unique


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存公开研究行，空列表写空文件；例如[{x:1}]写表头x和数值1。"""
    with path.open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def input_fingerprint(panel: Path, raw: Path) -> str:
    """哈希公开缓存、面板及研究代码，拒绝不同输入/代码复用结果；不读取任何凭据。

    输入两个目录输出SHA256字符串；逐块读文件以限制内存，无关隐藏文件不参与。
    """
    digest = hashlib.sha256()
    for root, pattern in (
        (panel, "*.npy"),
        (panel, "panel.json"),
        (raw, "*.json"),
        (Path(__file__).parent, "*.py"),
    ):
        for path in sorted(root.glob(pattern)):
            digest.update(path.name.encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def run_research(panel_path: Path, raw: Path, output: Path, *, limit: int | None = None) -> Path:
    """离线运行声明网格，逐配置原子保存结果，已有结果可恢复且不覆盖不同参数。

    例 run_research(PANEL,RAW,OUT,limit=1)只跑基线并标注smoke；完整运行输出report.json。
    需所有除权日分区存在；不把补采未完成解释为无分红。
    """
    panel = load_panel(panel_path)
    for day in panel.calendar:
        if day >= "20211001" and not (raw / f"dividend-{day}.json").exists():
            raise ValueError("incomplete_dividend_partitions")
    events = load_distributions(raw, panel.calendar, panel.codes)
    factors = factor_series(panel)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = input_fingerprint(panel_path, raw)
    fingerprint_path = output / "input-fingerprint.txt"
    if fingerprint_path.exists() and fingerprint_path.read_text() != fingerprint:
        raise ValueError("research_inputs_or_code_changed_use_new_output")
    if not fingerprint_path.exists() and any(output.iterdir()):
        raise ValueError("unversioned_results_use_new_output")
    fingerprint_path.write_text(fingerprint)
    attempts = []
    for parameters, policy in research_grid()[:limit]:
        path = output / policy.name
        path.mkdir(exist_ok=True)
        signature = {"regime": asdict(parameters), "policy": asdict(policy)}
        signature = json.loads(json.dumps(signature))
        result_path = path / "metrics.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved["configuration"] != signature:
                raise ValueError("research_resume_configuration_mismatch")
            metrics = saved["metrics"]
        else:
            market = panel.market
            states = classify_regimes(
                market["index"],
                market["breadth"],
                market["advancing"],
                market["amount"],
                market["limit_down"],
                parameters,
            )
            result = simulate_portfolio(panel, states, events, policy)
            metrics = evaluate_portfolio(result, panel, factors)
            peers = matched_trade_controls(result, panel)
            write_csv(path / "matched_trades.csv", peers.pop("records"))
            metrics["matched_peers"] = peers
            write_csv(path / "trades.csv", result["trades"])
            rows = [
                {name: value for name, value in row.items() if name != "weights"}
                | {
                    "return": result["daily_returns"][index],
                    "timed_market_return": result["timed_market_returns"][index],
                }
                for index, row in enumerate(result["daily"])
            ]
            write_csv(path / "daily.csv", rows)
            result_path.write_text(
                json.dumps({"configuration": signature, "metrics": metrics}, indent=2)
            )
        attempts.append(
            {
                "name": policy.name,
                **{
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
                    )
                },
            }
        )
        print(json.dumps(attempts[-1]), flush=True)
    write_csv(output / "attempts.csv", attempts)
    complete = limit is None
    eligible = [row for row in attempts if row["both_targets"]]
    ranked = sorted(eligible or attempts, key=lambda row: row["annualized_return"], reverse=True)
    report = {
        "complete_grid": complete,
        "attempts": len(attempts),
        "both_target_pass": len(eligible),
        "highest_return": max(attempts, key=lambda row: row["annualized_return"])
        if attempts
        else None,
        "highest_return_meeting_targets": ranked[0] if eligible else None,
        "positive_return_meeting_targets": sum(row["annualized_return"] > 0 for row in eligible),
        "all_data_used_for_optimization": True,
        "historical_pit_certified": False,
        "input_fingerprint": fingerprint,
        "distribution_events": len(events),
        "benchmark": "000985.CSI / H00985.CSI",
        "limitations": [
            "Full-sample optimization; bootstrap intervals do not adjust strategy selection.",
            "Cash dividend withholding fixed at conservative 20%; no individual tax refund.",
            "No rights-issue subscription or complete historical ST certification.",
            "Missing dividend payment/listing date leaves entitlement outstanding.",
            "Unlisted bonus shares conservatively delay full-position exit.",
            "Stale prices carried at last observation and separately stress-reported.",
            "Index controls are theoretical; no directly investable all-share total-return open.",
            "Historical provider versions and SW2021 industry taxonomy are reconstructed.",
        ],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report_path
