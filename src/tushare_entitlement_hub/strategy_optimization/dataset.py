"""冻结历史行情特征，按供应商成员有效区间重建行业群动。"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from ..raw import assert_public
from ..research_history.collection import response_records
from ..signal_research.history import assert_complete_history, iter_security_history
from .features import FloatArray, compute_features, lag, ratio, rolling


def read_envelope(path: Path) -> list[dict[str, Any]]:
    """读取补采公开响应，保留原始缓存且拒绝凭据字段；空成功响应输出 []。

    例如 fields=['x'],items=[[1]] 返回 [{'x':1}]，失败响应抛安全契约错误。
    """
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert_public(envelope)
    return response_records(envelope["response"])


def membership_matrix(
    records: Sequence[dict[str, Any]],
    calendar: Sequence[str],
    sectors: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """按 [in_date,out_date) 构造每日唯一一级行业；同日跨行业冲突记 -1。

    输入 A 在20230101纳入X、20230103退出，输出前两日X、第三日未知。
    同行业多三级记录先去重；缺纳入日期、历史记录缺退出日期拒绝重建。
    行业体系来自当前可取SW2021定义，结果不声明不可变历史版本或历史PIT认证。
    """
    membership: dict[str, np.ndarray] = {}
    diagnostics: defaultdict[str, int] = defaultdict(int)
    sector_indices = {code: index for index, code in enumerate(sectors)}
    days = np.asarray(calendar)
    for row in records:
        code, sector = str(row.get("ts_code", "")), str(row.get("l1_code", ""))
        start = str(row.get("in_date") or "")
        stop = str(row.get("out_date") or "99999999")
        if (
            not code.endswith((".SH", ".SZ"))
            or sector not in sector_indices
            or len(start) != 8
            or not start.isdigit()
            or len(stop) != 8
            or not stop.isdigit()
            or stop <= start
            or (row.get("is_new") == "N" and stop == "99999999")
        ):
            diagnostics["invalid_or_outside_universe_membership_rows"] += 1
            continue
        values = membership.setdefault(code, np.full(len(days), -1, dtype=np.int16))
        active = (days >= start) & (days < stop)
        conflicts = active & (values >= 0) & (values != sector_indices[sector])
        values[active & (values == -1)] = sector_indices[sector]
        values[conflicts] = -2
    for values in membership.values():
        diagnostics["conflicting_security_days"] += int(np.sum(values == -2))
        values[values == -2] = -1
    return membership, dict(diagnostics)


def daily_percentiles(values: FloatArray) -> FloatArray:
    """按列对当日行业值计算平均秩分位，缺值保持未知；两行业1/2映射0/1。

    输入二维[行业,日期]；单个有效行业为0.5，并列取平均秩，不借未来分布。
    """
    result = np.full(values.shape, np.nan)
    for column in range(values.shape[1]):
        valid = np.flatnonzero(np.isfinite(values[:, column]))
        if len(valid) == 1:
            result[valid, column] = 0.5
        elif len(valid) > 1:
            current = values[valid, column]
            for index in valid:
                value = values[index, column]
                result[index, column] = (
                    np.sum(current < value) + (np.sum(current == value) - 1) / 2
                ) / (len(valid) - 1)
    return result


def build_dataset(history_database: Path, sector_root: Path, output: Path) -> Path:
    """只读原行情与补采行业，输出逐证券特征和剔除本股的行业统计，支持完整复用。

    例如同业六只均有行情，输出某股sector_peers=5；只剩四只同业则群动条件不可用。
    全部源沪深证券含退市股，北交所历史身份缺口单独披露。无未来复权锚点。
    已有完整manifest返回原路径；不完整目录拒绝覆盖，避免混合两轮特征。
    """
    manifest_path = output / "dataset.json"
    if manifest_path.exists():
        return manifest_path
    if output.exists() and any(output.iterdir()):
        raise ValueError("incomplete_dataset_use_new_directory")
    output.mkdir(parents=True, exist_ok=True)
    catalogue = read_envelope(sector_root / "catalog.json")
    sectors = sorted(str(row["index_code"]) for row in catalogue)
    member_rows = [
        row for path in sorted(sector_root.glob("members-*.json")) for row in read_envelope(path)
    ]
    with duckdb.connect(str(history_database), read_only=True) as connection:
        connection.execute("SET threads=1")
        connection.execute("SET memory_limit='512MB'")
        entry = connection.execute("SELECT manifest FROM cache_manifest").fetchone()
        if entry is None:
            raise ValueError("history_manifest_missing")
        source_manifest = json.loads(entry[0])
        assert_complete_history(source_manifest)
        calendar = [day for day in source_manifest["trading_dates"] if day <= "20261016"]
        membership, diagnostics = membership_matrix(member_rows, calendar, sectors)
        shape = (len(sectors), len(calendar))
        aggregate = {
            name: np.zeros(shape)
            for name in (
                "expected",
                "valid",
                "positive",
                "return",
                "above",
                "breakout",
                "limit",
                "limit_known",
            )
        }
        for labels in membership.values():
            valid_positions = np.flatnonzero(labels >= 0)
            np.add.at(aggregate["expected"], (labels[valid_positions], valid_positions), 1)
        codes = []
        market_count, market_up = np.zeros(len(calendar)), np.zeros(len(calendar))
        for code, rows in iter_security_history(connection):
            if not code.endswith((".SH", ".SZ")):
                diagnostics["excluded_bj_securities"] = (
                    diagnostics.get("excluded_bj_securities", 0) + 1
                )
                continue
            features = compute_features(rows, calendar)
            codes.append(code)
            labels = membership.get(code, np.full(len(calendar), -1, dtype=np.int16))
            features["sector_index"] = labels
            valid = np.isfinite(features["return1"])
            market_count += valid
            market_up += valid & (features["return1"] > 0)
            positions = np.flatnonzero((labels >= 0) & valid)
            key = (labels[positions], positions)
            contributions = {
                "valid": np.ones(len(positions)),
                "positive": (features["return1"][positions] > 0).astype(float),
                "return": features["return1"][positions],
                "above": (features["close"][positions] > features["ma20"][positions]).astype(float),
                "breakout": (
                    features["close"][positions] > features["previous_high20"][positions]
                ).astype(float),
                "limit": np.nan_to_num(features["limit_up"][positions]),
                "limit_known": np.isfinite(features["limit_up"][positions]).astype(float),
            }
            for name, values in contributions.items():
                np.add.at(aggregate[name], key, values)
            np.savez_compressed(output / f"{code}.npz", allow_pickle=False, **features)
            if len(codes) % 250 == 0:
                print(f"[features] securities={len(codes)}", flush=True)
    industry_returns = np.full(shape, np.nan)
    industry_amount_ratio = np.full(shape, np.nan)
    industry_trend = np.full(shape, np.nan)
    for index, code in enumerate(sectors):
        observations = {
            row["trade_date"]: row for row in read_envelope(sector_root / f"daily-{code}.json")
        }
        closes = np.asarray([observations.get(day, {}).get("close", np.nan) for day in calendar])
        amounts = np.asarray([observations.get(day, {}).get("amount", np.nan) for day in calendar])
        average = rolling(closes, 20, "mean")
        industry_returns[index] = ratio(closes, lag(closes, 5)) - 1
        industry_amount_ratio[index] = ratio(amounts, lag(rolling(amounts, 20, "median")))
        industry_trend[index] = np.where(
            np.isfinite(average), (closes > average).astype(float), np.nan
        )
    ranks = daily_percentiles(industry_returns)
    for code in codes:
        with np.load(output / f"{code}.npz", allow_pickle=False) as saved:
            features = {name: saved[name] for name in saved.files}
        labels = features["sector_index"].astype(int)
        positions = np.flatnonzero((labels >= 0) & np.isfinite(features["return1"]))
        key = (labels[positions], positions)
        for name in (
            "sector_peers",
            "sector_coverage",
            "sector_breadth",
            "sector_mean_return",
            "sector_above_ma20",
            "sector_breakout_breadth",
            "sector_limit_breadth",
            "sector_heat",
            "sector_amount_ratio",
            "sector_trend",
        ):
            features[name] = np.full(len(calendar), np.nan)
        peers = aggregate["valid"][key] - 1
        features["sector_peers"][positions] = peers
        features["sector_coverage"][positions] = ratio(peers, aggregate["expected"][key] - 1)
        own_contributions = {
            "sector_breadth": ("positive", (features["return1"][positions] > 0).astype(float)),
            "sector_mean_return": ("return", features["return1"][positions]),
            "sector_above_ma20": (
                "above",
                (features["close"][positions] > features["ma20"][positions]).astype(float),
            ),
            "sector_breakout_breadth": (
                "breakout",
                (features["close"][positions] > features["previous_high20"][positions]).astype(
                    float
                ),
            ),
        }
        for name, (total, own) in own_contributions.items():
            features[name][positions] = ratio(aggregate[total][key] - own, peers)
        known_limit = aggregate["limit_known"][key] - np.isfinite(features["limit_up"][positions])
        features["sector_limit_breadth"][positions] = np.where(
            ratio(known_limit, peers) >= 0.9,
            ratio(aggregate["limit"][key] - np.nan_to_num(features["limit_up"][positions]), peers),
            np.nan,
        )
        features["sector_heat"][positions] = ranks[key]
        features["sector_amount_ratio"][positions] = industry_amount_ratio[key]
        features["sector_trend"][positions] = industry_trend[key]
        features["market_breadth"] = ratio(market_up, market_count)
        np.savez_compressed(output / f"{code}.npz", allow_pickle=False, **features)
    manifest = {
        "version": "strategy-features/v1",
        "calendar": calendar,
        "securities": codes,
        "sector_codes": sectors,
        "membership_diagnostics": diagnostics,
        "source_fingerprint": source_manifest["raw_fingerprint"],
        "history_database": str(history_database),
        "sector_root": str(sector_root),
        "historical_pit_certified": False,
        "data_semantics": "retrospective_provider_reconstruction",
        "industry_heat_semantics": (
            "five_session_sector_price_strength_percentile_not_attention_rank"
        ),
        "membership_interval": "in_date_inclusive_out_date_exclusive",
        "limitations": [
            "SW2021 classification may restate historical taxonomy",
            "source announcement and historical risk versions not fully certified",
            "sector prices published after close; decisions assume available before next open",
            "industry membership coverage gates use supplied intervals, not certified full history",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
