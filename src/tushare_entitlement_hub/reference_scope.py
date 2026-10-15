"""为每次正式采集生成可复算的参数、分区和日期范围证书。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .store import digest


def _date_value(value: Any) -> str | None:
    """把八位日期或 ISO 日期规范成 YYYY-MM-DD，其他参数返回 None。

    输入 ``20261009`` 返回 ``2026-10-09``，输入股票代码返回 None；本函数只识别
    完整自然日，不把月份、报告期或普通数字猜成交易日。
    """
    text = str(value)
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    return None


def _parameter_dates(parameters: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """提取名称含 date 的请求日期并按字段去重排序。

    输入两个 ``trade_date`` 分区返回两个日期；``ts_code`` 即使含数字也不会进入结果。
    输出只描述真实请求参数，不根据响应内容扩大范围。
    """
    dates: dict[str, set[str]] = {}
    for partition in parameters:
        for name, value in partition.items():
            parsed = _date_value(value) if "date" in name.lower() else None
            if parsed is not None:
                dates.setdefault(name, set()).add(parsed)
    return {name: sorted(values) for name, values in sorted(dates.items())}


def _record_dates(rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """从合格标准行提取实际业务日期，供范围证书核对返回覆盖。

    输入含 ``trade_date`` 和 ``observed_at`` 的行只返回 ``trade_date``；观察时间不代表
    业务覆盖。日期字段的空值和非自然日值会被忽略。
    """
    dates: dict[str, set[str]] = {}
    for row in rows:
        for name, value in row.items():
            parsed = _date_value(value) if name == "date" or name.endswith("_date") else None
            if parsed is not None:
                dates.setdefault(name, set()).add(parsed)
    return {name: sorted(values) for name, values in sorted(dates.items())}


def build_scope_certificate(
    api_name: str,
    parameters: dict[str, Any] | list[dict[str, Any]],
    partitions: list[dict[str, Any]],
    collection_status: str,
    observed_at: str,
    quality: dict[str, Any],
    accepted_rows: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """生成一次采集范围证书，分开记录采集完整性和行质量。

    输入两个成功日期分区和 ``quality.status='passed'``，输出 ``status='certified'``；
    任一分区失败或质量未通过则输出 ``incomplete``，并保留具体失败分区。证书不声称
    参数之外的全历史或全市场覆盖。
    """
    requested = [parameters] if isinstance(parameters, dict) else list(parameters)
    completed = [
        item for item in partitions if str(item.get("status", "")).startswith("complete")
    ]
    failures = [
        {
            "partition_id": digest(item.get("parameters", {})),
            "status": str(item.get("status", "unknown")),
            "parameters": item.get("parameters", {}),
        }
        for item in partitions
        if not str(item.get("status", "")).startswith("complete")
    ]
    collection_complete = (
        collection_status.startswith("complete")
        and len(partitions) == len(requested)
        and len(completed) == len(requested)
    )
    quality_status = str(quality.get("status", "not_checked"))
    quality_passed = quality_status in {"passed", "empty"}
    body = {
        "api_name": api_name,
        "observed_at": observed_at,
        "request_scope_fingerprint": digest(requested),
        "requested_partition_count": len(requested),
        "completed_partition_count": len(completed),
        "collection_status": collection_status,
        "collection_complete": collection_complete,
        "quality_status": quality_status,
        "quality_passed": quality_passed,
        "accepted_rows": int(quality.get("accepted_rows", 0)),
        "quarantined_rows": int(quality.get("quarantined_rows", 0)),
        "parameter_dates": _parameter_dates(requested),
        "data_dates": _record_dates(accepted_rows),
        "failed_partitions": failures,
        "scope_basis": "complete_declared_request_only",
        "status": "certified" if collection_complete and quality_passed else "incomplete",
    }
    return {**body, "certificate_id": digest(body)}


def merge_scope_certificates(
    previous: Iterable[dict[str, Any]], current: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按证书身份合并历史与当前范围证书，返回稳定不可变顺序。

    输入相同证书两次只保留一份；输入两个不同交易日证书会保留两份，供 20 日稳定性
    验收逐日核对。缺少 ``certificate_id`` 的旧记录会按完整内容生成兼容身份。
    """
    indexed: dict[str, dict[str, Any]] = {}
    for certificate in (*tuple(previous), *tuple(current)):
        item = dict(certificate)
        identity = str(item.get("certificate_id") or digest(item))
        item["certificate_id"] = identity
        indexed[identity] = item
    return [indexed[identity] for identity in sorted(indexed)]


def certified_dates(
    certificates: Iterable[dict[str, Any]], api_name: str, date_field: str = "trade_date",
) -> tuple[str, ...]:
    """返回某接口已认证的请求日期，不把失败证书或其他日期字段混入。

    输入 daily 的两个通过证书和一个失败证书，输出两个日期的排序元组；接口从不按日期
    请求时返回空元组，调用方不能把空元组解释成全历史完整。
    """
    return tuple(
        sorted(
            {
                day
                for certificate in certificates
                if certificate.get("api_name") == api_name
                and certificate.get("status") == "certified"
                for day in (
                    certificate.get("data_dates", {}).get(date_field, [])
                    or certificate.get("parameter_dates", {}).get(date_field, [])
                )
            }
        )
    )
