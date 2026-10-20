"""全五年参数搜索与完整尝试记录，不设验证集或测试集。"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .acceleration import optional_compiled_accumulator
from .concepts import load_features
from .execution import entry_indices, exit_grid, independent_trades, outcome_arrays
from .strategies import strategy_grid, strategy_masks


def maximum_signal_gap(days: Iterable[int], start: int, end: int) -> int:
    """计算实际入场日的最长间隔，包含研究起止空窗；输入[3,8]、0、10返回5。

    无入场返回区间长度加一；同日多股只计一次，不能用集中爆发掩盖长期空窗。
    """
    unique = sorted({int(day) for day in days if start <= day <= end})
    if not unique:
        return end - start + 1
    return int(max(np.diff([start, *unique, end]), default=0))


def run_search(
    dataset: Path,
    output: Path,
    *,
    maximum_securities: int | None = None,
    accelerate: bool = False,
    concepts_only: bool = False,
) -> Path:
    """逐证券遍历60个选股条件与45个退出参数，保存全部2700组统计和最优候选。

    输入冻结特征目录和新产物目录；输出search.csv。样本限定五年全部参与优化。
    开仓到期3市场日、同股每策略只保留一笔持仓；计算所有实际入场日期的空窗。
    maximum_securities仅用于确定性开发烟测，带此参数时禁止标记达到正式目标。
    """
    manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    calendar = manifest["calendar"]
    codes = manifest["securities"]
    if maximum_securities is not None:
        if maximum_securities < 1:
            raise ValueError("maximum_securities_must_be_positive")
        codes = codes[:maximum_securities]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "search.csv").exists():
        raise ValueError("search_output_exists")
    definitions, policies = strategy_grid(), exit_grid()
    if "concept_root" in manifest:
        definitions = strategy_grid(include_concepts=True)
    if concepts_only:
        if "concept_root" not in manifest:
            raise ValueError("concept_search_requires_history_overlay")
        definitions = tuple(item for item in definitions if "concept" in item.context)
    accumulator = optional_compiled_accumulator() if accelerate else None
    trials = [(definition, policy) for policy in policies for definition in definitions]
    count = len(trials)
    sums = {
        name: np.zeros(count)
        for name in (
            "signals",
            "closed",
            "wins",
            "open",
            "expired",
            "censored",
            "skipped",
            "return_sum",
            "positive_sum",
            "negative_sum",
            "holding_sum",
        )
    }
    sums_matrix = np.zeros((len(sums), count))
    sums = {name: sums_matrix[index] for index, name in enumerate(sums)}
    entry_days = np.zeros((count, len(calendar)), dtype=bool)
    years = sorted({day[:4] for day in calendar if day >= "20211018"})
    annual_sum = np.zeros((count, len(years)))
    annual_count = np.zeros((count, len(years)))
    year_indices = np.asarray(
        [years.index(day[:4]) if day[:4] in years else -1 for day in calendar]
    )
    active_window = np.asarray(["20211018" <= day <= "20261016" for day in calendar])
    protocol = {
        "strategies": [{**asdict(item), "name": item.name} for item in definitions],
        "exit_policies": [asdict(policy) for policy in policies],
        "start": "20211018",
        "end": "20261016",
        "all_data_used_for_optimization": True,
        "acceleration": accelerate,
        "concept_contexts_only": concepts_only,
        "source_fingerprint": manifest["source_fingerprint"],
        "securities": len(codes),
        "smoke_only": maximum_securities is not None,
        "target_win_rate": 0.65,
        "target_maximum_entry_gap": 10,
        "maximum_entry_wait_sessions": 3,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for processed, code in enumerate(codes, 1):
        features = load_features(dataset, code)
        masks = strategy_masks(features, definitions)
        signal_indices = [
            np.flatnonzero(masks[definition.name] & active_window) for definition in definitions
        ]
        candidates_flat = np.concatenate(signal_indices)
        offsets = np.asarray([0, *np.cumsum([len(values) for values in signal_indices])])
        entries = entry_indices(features)
        for policy_index, policy in enumerate(policies):
            outcomes = outcome_arrays(features, calendar, policy, entries)
            if accumulator is not None:
                accumulator(
                    candidates_flat,
                    offsets,
                    entries,
                    outcomes["exit_index"],
                    outcomes["return"],
                    year_indices,
                    policy_index * len(definitions),
                    sums_matrix,
                    entry_days,
                    annual_sum,
                    annual_count,
                )
                continue
            for strategy_index, candidates in enumerate(signal_indices):
                index = policy_index * len(definitions) + strategy_index
                if not len(candidates):
                    continue
                selected, skipped = independent_trades(candidates, outcomes, len(calendar))
                closed = selected[np.isfinite(outcomes["return"][selected])]
                bought = selected[outcomes["entry_index"][selected] >= 0]
                not_bought = selected[outcomes["entry_index"][selected] < 0]
                returns = outcomes["return"][closed]
                sums["signals"][index] += len(candidates)
                sums["closed"][index] += len(closed)
                sums["wins"][index] += np.sum(returns > 0)
                sums["open"][index] += len(bought) - len(closed)
                sums["expired"][index] += np.sum(not_bought + 3 < len(calendar))
                sums["censored"][index] += np.sum(not_bought + 3 >= len(calendar))
                sums["skipped"][index] += len(skipped)
                sums["return_sum"][index] += np.sum(returns)
                sums["positive_sum"][index] += np.sum(returns[returns > 0])
                sums["negative_sum"][index] += np.sum(returns[returns < 0])
                sums["holding_sum"][index] += np.sum(
                    outcomes["exit_index"][closed] - outcomes["entry_index"][closed]
                )
                entry_days[index, outcomes["entry_index"][bought]] = True
                for year_index in range(len(years)):
                    annual = returns[year_indices[closed] == year_index]
                    annual_sum[index, year_index] += np.sum(annual)
                    annual_count[index, year_index] += len(annual)
        if processed % 100 == 0 or processed == len(codes):
            print(f"[search] securities={processed}/{len(codes)} trials={count}", flush=True)
    results = []
    start, end = calendar.index("20211018"), calendar.index("20261016")
    for index, (definition, policy) in enumerate(trials):
        closed = sums["closed"][index]
        win_rate = sums["wins"][index] / closed if closed else None
        average_return = sums["return_sum"][index] / closed if closed else None
        gap = maximum_signal_gap(np.flatnonzero(entry_days[index]), start, end)
        losses = sums["negative_sum"][index]
        result = {
            "strategy": definition.name,
            "exit_policy": policy.name,
            **asdict(policy),
            **{name: float(values[index]) for name, values in sums.items()},
            "win_rate": win_rate,
            "mean_return": average_return,
            "profit_factor": sums["positive_sum"][index] / -losses if losses < 0 else None,
            "entry_dates": int(np.sum(entry_days[index])),
            "maximum_entry_gap": gap,
            "target_passed": bool(
                maximum_securities is None
                and win_rate is not None
                and win_rate >= 0.65
                and gap <= 10
            ),
            **{
                f"mean_{year}": annual_sum[index, position] / annual_count[index, position]
                if annual_count[index, position]
                else None
                for position, year in enumerate(years)
            },
        }
        results.append(result)
    results.sort(
        key=lambda item: (
            bool(item["target_passed"]),
            (item["mean_return"] or -1) > 0,
            item["mean_return"] if item["mean_return"] is not None else -1,
            item["win_rate"] if item["win_rate"] is not None else -1,
        ),
        reverse=True,
    )
    path = output / "search.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary: dict[str, Any] = {
        "trials": count,
        "target_passed_trials": sum(row["target_passed"] for row in results),
        "best": results[:20],
        "all_data_used_for_optimization": True,
        "historical_pit_certified": False,
        "source_fingerprint": manifest["source_fingerprint"],
    }
    (output / "search.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
