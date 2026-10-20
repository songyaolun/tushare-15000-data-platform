"""显式授权后补采公开基准、市值和分红，原文仅写指定私有缓存。"""

from pathlib import Path
from typing import Any

from ..client import HttpTransport
from ..strategy_optimization.collection import cached_response


def collection_jobs(calendar: list[str]) -> list[tuple[str, str, dict[str, Any], int]]:
    """声明基准、月末市值及逐日除权分区，不请求网络。

    示例 calendar=['20211018'] 输出两指数、一市值、一分红共四个请求。
    市值只在对应观察日期之后使用，不把月底信息回填到月初。
    """
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("calendar_must_be_unique_sorted_nonempty")
    jobs: list[tuple[str, str, dict[str, Any], int]] = []
    for code in ("000985.CSI", "H00985.CSI"):
        jobs.append(
            (
                f"index-{code}.json",
                "index_daily",
                {
                    "ts_code": code,
                    "start_date": calendar[0],
                    "end_date": calendar[-1],
                },
                8000,
            )
        )
    month_ends = {day[:6]: day for day in calendar}
    for day in month_ends.values():
        jobs.append(
            (
                f"capitalization-{day}.json",
                "daily_basic",
                {
                    "trade_date": day,
                    "fields": "ts_code,trade_date,total_mv,circ_mv",
                },
                6500,
            )
        )
    for day in calendar:
        if day >= "20211001":
            jobs.append((f"dividend-{day}.json", "dividend", {"ex_date": day}, 6000))
    return jobs


def collect_partition(
    transport: HttpTransport,
    root: Path,
    job: tuple[str, str, dict[str, Any], int],
) -> int:
    """原文缓存后核对分区代码或除权日，返回行数；缓存空分红返回0。

    输入 ('dividend-20211018.json','dividend',{'ex_date':'20211018'},6000)
    输出匹配当日除权的记录数，错误日期或代码拒绝。
    """
    filename, api, parameters, cap = job
    rows = cached_response(transport, root, filename, api, parameters, cap)
    for row in rows:
        if api == "dividend" and row.get("ex_date") != parameters["ex_date"]:
            raise ValueError("dividend_partition_date_mismatch")
        if api == "index_daily" and row.get("ts_code") != parameters["ts_code"]:
            raise ValueError("benchmark_partition_code_mismatch")
    return len(rows)
