"""可恢复的真实历史响应采集；凭据只由既有 HTTPS transport 管理。"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tushare_entitlement_hub.client import ProviderError, Transport

FIELDS = {
    "daily": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    "adj_factor": "ts_code,trade_date,adj_factor",
    "stk_limit": "ts_code,trade_date,pre_close,up_limit,down_limit",
    "trade_cal": "exchange,cal_date,is_open,pretrade_date",
    "stock_basic": (
        "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
    ),
    "daily_basic": (
        "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
        "free_share,total_mv,circ_mv"
    ),
    "income_vip": (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
        "update_flag,total_revenue,revenue,n_income,n_income_attr_p"
    ),
    "cashflow_vip": (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
        "update_flag,net_profit,n_cashflow_act,c_pay_acq_const_fiolta"
    ),
    "balancesheet_vip": (
        "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
        "update_flag,total_assets,total_liab,money_cap,total_cur_liab"
    ),
    "fina_indicator_vip": (
        "ts_code,ann_date,end_date,update_flag,profit_dedt,"
        "netprofit_yoy,dt_netprofit_yoy,roe,debt_to_assets,ocf_to_profit"
    ),
    "st": "ts_code,name,pub_date,imp_date,st_type,st_reason,st_explain",
    "namechange": "ts_code,name,start_date,end_date,ann_date,change_reason",
}
ROW_CAPS = {
    "daily": 6000,
    "adj_factor": 6000,
    "stk_limit": 5800,
    "trade_cal": 6000,
    "stock_basic": 6000,
    "namechange": 6000,
    "st": 10000,
    "daily_basic": 6000,
    "income_vip": 5000,
    "cashflow_vip": 5000,
    "balancesheet_vip": 5000,
    "fina_indicator_vip": 10000,
}


def response_records(response: dict[str, Any]) -> list[dict[str, Any]]:
    """校验成功表结构并转为源字段记录，不改单位、不吞掉错误。

    输入 {code:0,data:{fields:['x'],items:[[1]]}}，输出 [{'x':1}]。
    提供方失败或畸形行抛固定 ValueError，不回显提供方错误文本。
    """
    if response.get("code") != 0:
        raise ValueError("provider_response_failure")
    data = response.get("data")
    if not isinstance(data, dict):
        raise ValueError("provider_data_missing")
    fields, items = data.get("fields"), data.get("items")
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) for field in fields)
        or len(set(fields)) != len(fields)
        or not isinstance(items, list)
    ):
        raise ValueError("provider_table_invalid")
    if any(not isinstance(row, list) or len(row) != len(fields) for row in items):
        raise ValueError("provider_row_invalid")
    return [dict(zip(fields, row, strict=True)) for row in items]


def partition_quality(
    api_name: str, parameters: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    """核验字段、日期、唯一键、数值及行数上限，达到上限即拒绝证明完整。

    输入 daily 的单交易日响应，输出 rows、status、issues、sha256；
    少字段/重复代码/日期错误/沪深主板负价格均令 status='failed'。
    北交所历史缺值原样保留；该价格契约只认证沪深主板。
    """
    records = response_records(response)
    issues: list[str] = []
    fields = response["data"]["fields"]
    if not set(FIELDS[api_name].split(",")) <= set(fields):
        issues.append("missing_fields")
    if len(records) >= ROW_CAPS[api_name] or (
        response["data"].get("has_more") and "offset" not in parameters
    ):
        issues.append("possible_truncation")
    if "trade_date" in parameters:
        if not records:
            issues.append("empty_trade_day")
        if any(row.get("trade_date") != parameters["trade_date"] for row in records):
            issues.append("wrong_trade_date")
        codes = [row.get("ts_code") for row in records]
        if len(set(codes)) != len(codes):
            issues.append("duplicate_security_date")
    if "period" in parameters and any(
        row.get("end_date") != parameters["period"] for row in records
    ):
        issues.append("wrong_report_period")
    positive_fields = {
        "daily": ("open", "high", "low", "close", "pre_close"),
        "adj_factor": ("adj_factor",),
        "stk_limit": (),
    }.get(api_name, ())
    value_records = records
    if api_name == "daily":
        value_records = [
            row
            for row in records
            if str(row.get("ts_code", "")).startswith(("60", "00"))
            and str(row.get("ts_code", "")).endswith((".SH", ".SZ"))
        ]
    for name in positive_fields:
        if any(
            not isinstance(row.get(name), (int, float))
            or not math.isfinite(row[name])
            or row[name] < (0 if api_name == "daily" else 1e-100)
            for row in value_records
        ):
            issues.append("invalid_" + name)
    encoded = json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
    return {
        "rows": len(records),
        "status": "failed" if issues else "passed",
        "issues": issues,
        "response_sha256": hashlib.sha256(encoded).hexdigest(),
    }


class HistoryCollector:
    """限制全局请求频率的原始响应采集器，可供线程池共享。"""

    def __init__(
        self,
        transport: Transport,
        output: Path,
        *,
        interval_seconds: float = 0.16,
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """保存公共采集配置；不读取凭据。

        输入 transport、缓存目录，输出采集器；interval_seconds=0 用于离线测试。
        """
        if interval_seconds < 0 or not math.isfinite(interval_seconds) or retries < 1:
            raise ValueError("invalid_collection_configuration")
        self.transport, self.output = transport, output
        self.interval_seconds, self.retries, self.sleep = interval_seconds, retries, sleep
        self._lock = threading.Lock()
        self._next_request = 0.0

    def collect(self, api_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """采集一个公开参数分区，先落原始响应再质量检查；失败保留证据。

        输入 collect('daily', {'trade_date':'20210104'})，输出含 raw_path、quality 的清单。
        已通过且摘要相符的缓存直接复用；临时传输失败按指数间隔重试。
        """
        request_parameters = {**parameters, "fields": FIELDS[api_name]}
        signature = hashlib.sha256(
            json.dumps(request_parameters, sort_keys=True).encode()
        ).hexdigest()
        directory = self.output / "raw" / api_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (signature + ".json")
        if path.exists():
            envelope = json.loads(path.read_text())
            quality = partition_quality(api_name, parameters, envelope["response"])
            if quality["status"] == "passed" and quality == envelope["quality"]:
                return {**envelope, "raw_path": str(path), "cache_reused": True}
        response: dict[str, Any] = {}
        for attempt in range(self.retries):
            with self._lock:
                delay = max(0.0, self._next_request - time.monotonic())
                self.sleep(delay)
                self._next_request = time.monotonic() + self.interval_seconds
            try:
                response = self.transport.request(api_name, request_parameters)
                if response.get("code") != 0 and attempt + 1 < self.retries:
                    self.sleep(min(30.0, 2.0**attempt))
                    continue
                break
            except ProviderError as error:
                if error.kind != "transient_failure" or attempt + 1 == self.retries:
                    raise
                self.sleep(min(30.0, 2.0**attempt))
        envelope = {
            "api_name": api_name,
            "parameters": parameters,
            "observed_at": datetime.now(UTC).isoformat(),
            "response": response,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")))
        temporary.replace(path)
        quality = partition_quality(api_name, parameters, response)
        envelope["quality"] = quality
        temporary.write_text(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")))
        temporary.replace(path)
        if quality["status"] != "passed":
            raise ValueError("history_partition_quality_failed")
        return {**envelope, "raw_path": str(path), "cache_reused": False}

    def collect_pages(
        self,
        api_name: str,
        parameters: dict[str, Any],
        *,
        page_size: int = 3000,
        maximum_pages: int = 100,
    ) -> list[dict[str, Any]]:
        """按 offset 完整采集公开分区，保留每页原始证据并拒绝重复分页。

        输入 income_vip 的 period 与 page_size=3000，输出所有原始页清单；
        仅最后一页少于 page_size 可结束，重复页或页数耗尽抛固定错误。
        """
        if page_size < 1 or page_size >= ROW_CAPS[api_name] or maximum_pages < 1:
            raise ValueError("invalid_pagination_configuration")
        pages: list[dict[str, Any]] = []
        seen: set[str] = set()
        for page_number in range(maximum_pages):
            page = self.collect(
                api_name, {**parameters, "limit": page_size, "offset": page_number * page_size}
            )
            digest = page["quality"]["response_sha256"]
            if digest in seen and page["quality"]["rows"]:
                raise ValueError("history_pagination_repeated_page")
            seen.add(digest)
            pages.append(page)
            if page["quality"]["rows"] < page_size and page["response"]["data"].get("has_more"):
                raise ValueError("history_pagination_unexpected_page_size")
            if page["quality"]["rows"] < page_size and not page["response"]["data"].get("has_more"):
                return pages
        raise ValueError("history_pagination_exhausted")
