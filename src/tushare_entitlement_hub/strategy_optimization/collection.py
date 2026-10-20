"""可恢复的行业与概念历史补采；先保存公开原文，再检验分区契约。"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..client import HttpTransport, ProviderError
from ..raw import assert_public
from ..research_history.collection import response_records


def cached_response(
    transport: HttpTransport,
    root: Path,
    filename: str,
    api: str,
    parameters: dict[str, Any],
    cap: int,
) -> list[dict[str, Any]]:
    """缓存一次公开响应并拒绝疑似截断、错日期与请求混用；例如空成功响应返回[]。

    命中缓存不请求网络；失败最多尝试三次，间隔一秒；凭据仅由transport管理。
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    if path.exists():
        envelope = json.loads(path.read_text(encoding="utf-8"))
    else:
        for attempt in range(3):
            try:
                response = transport.request(api, parameters)
                assert_public(response)
                envelope = {
                    "api_name": api,
                    "parameters": parameters,
                    "observed_at": datetime.now(UTC).isoformat(),
                    "response": response,
                }
                path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
                break
            except ProviderError:
                if attempt == 2:
                    raise
                time.sleep(1)
    assert_public(envelope)
    if envelope["api_name"] != api or envelope["parameters"] != parameters:
        raise ValueError("historical_cache_request_mismatch")
    rows = response_records(envelope["response"])
    if len(rows) >= cap:
        raise ValueError("historical_partition_potential_truncation")
    if "trade_date" in parameters and any(
        row.get("trade_date") != parameters["trade_date"] for row in rows
    ):
        raise ValueError("historical_partition_date_mismatch")
    return rows


def collect_industries(transport: HttpTransport, root: Path) -> None:
    """补采SW2021一级行业目录、当前/退出成员及日线；输入传输与目录，输出公开缓存。

    示例31个行业产生1个目录、62个成员、31个日线包络，已有分区直接复用。
    """
    catalogue = cached_response(
        transport, root, "catalog.json", "index_classify", {"level": "L1", "src": "SW2021"}, 2000
    )
    for row in catalogue:
        code = row["index_code"]
        for mode in ("Y", "N"):
            cached_response(
                transport,
                root,
                f"members-{code}-{mode}.json",
                "index_member_all",
                {"l1_code": code, "is_new": mode},
                2000,
            )
        cached_response(
            transport,
            root,
            f"daily-{code}.json",
            "sw_daily",
            {"ts_code": code, "start_date": "20200101", "end_date": "20261016"},
            4000,
        )


def active_concepts(rows: list[dict[str, Any]]) -> list[str]:
    """按当日涨幅与换手率合计秩选择前五概念，并列按代码排序；空有效值返回[]。

    示例两主题涨幅/换手均更大的A排在B之前；不读取未来排名或现在成员。
    """
    valid = [
        row
        for row in rows
        if all(
            isinstance(row.get(name), (int, float)) and math.isfinite(row[name])
            for name in ("pct_change", "turnover_rate")
        )
    ]
    scores = {
        str(row["ts_code"]): sum(
            other[name] < row[name] for other in valid for name in ("pct_change", "turnover_rate")
        )
        for row in valid
    }
    return sorted(scores, key=lambda code: (-scores[code], code))[:5]


def collect_concepts(transport: HttpTransport, root: Path, calendar: list[str]) -> Path:
    """补采可获取区间每日概念全集与活跃前五的当日成员；输入日历，输出完成清单。

    示例日历只有20241220则写该日目录、最多五个成员文件及collection.json。
    每次成员请求间隔0.4秒，支持重跑恢复，不以当前成员回填过去。
    """
    results = []
    for day in calendar:
        if not "20241220" <= day <= "20261016":
            continue
        rows = cached_response(
            transport,
            root,
            f"index-{day}.json",
            "dc_index",
            {"trade_date": day, "idx_type": "概念板块"},
            5000,
        )
        chosen = active_concepts(rows)
        counts = []
        for code in chosen:
            if not (root / f"members-{day}-{code}.json").exists():
                time.sleep(0.4)
            members = cached_response(
                transport,
                root,
                f"members-{day}-{code}.json",
                "dc_member",
                {"trade_date": day, "ts_code": code},
                5000,
            )
            counts.append(len(members))
        results.append(
            {
                "trade_date": day,
                "catalogue_count": len(rows),
                "chosen": chosen,
                "member_counts": counts,
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    path = root / "collection.json"
    path.write_text(
        json.dumps(
            {
                "selection": "top5_same_day_return_and_turnover_average_rank",
                "historical_pit_certified": False,
                "days": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
