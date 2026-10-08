"""逐行数据验收，不把采集完整性、业务空值和历史回测资格混为一谈。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

from .reference_contract import ReferenceContract, ReferenceRecord, normalize_reference_row
from .store import digest


def row_findings(row: dict[str, Any], contract: ReferenceContract) -> list[str]:
    """检查可直接验证的数值关系；例如高价小于低价返回 invalid_ohlc_range。

    输入标准行与契约，输出问题码列表。没有交易的缺失价格仍作为缺失保存，
    不将结算价、报价收益率或单独的 close 强行拼成完整 OHLC。
    """
    findings = []
    prices = [row.get(field) for field in ("open", "high", "low", "close")]
    if all(field in row for field in ("open", "high", "low", "close")):
        if any(value is None for value in prices) and (row.get("vol") or 0) > 0:
            findings.append("incomplete_traded_ohlc")
    if all(isinstance(value, (float, int)) for value in prices):
        opening, high, low, closing = [float(cast(float, value)) for value in prices]
        if low > high or not low <= opening <= high or not low <= closing <= high:
            findings.append("invalid_ohlc_range")
    for field in ("vol", "volume", "amount", "trade_volume"):
        value = row.get(field)
        if isinstance(value, (float, int)) and value < 0:
            findings.append("negative_volume_or_amount")
    if contract.policy.api_name in {"adj_factor", "fund_adj"}:
        if row.get("adj_factor") is None or row["adj_factor"] <= 0:
            findings.append("invalid_adjustment_factor")
    return findings


def evaluate_records(
    records: list[ReferenceRecord], contract: ReferenceContract, observed_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """标准化并逐行验收，完整保留失败记录及所有重复的来源位置。

    输入 (原始行, 响应摘要, 页内序号, 请求身份) 列表、契约与观察时间；返回合格行、隔离行和统计。
    同键同内容只保留一个规范行并登记全部来源；同键不同内容全部隔离，不猜测覆盖。
    示例：一条有效行、一条错误数值行产生 accepted=1、quarantined=1。
    """
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    null_counts: dict[str, int] = defaultdict(int)
    for source, response_hash, position, request_id in records:
        try:
            row = normalize_reference_row(
                source, contract, observed_at, response_hash, position, request_id
            )
            issues = row_findings(row, contract)
        except (ValueError, TypeError, OverflowError) as error:
            row = {
                "source_fields": source, "response_hash": response_hash,
                "source_request_id": request_id, "source_index": position,
                "source_record_id": digest(
                    [contract.policy.api_name, observed_at, response_hash, request_id, position]
                ),
            }
            issues = [str(error) if isinstance(error, ValueError) else "normalization_failed"]
        if issues:
            quarantined.append({**row, "issues": issues})
        else:
            candidates[row["business_key"]].append(row)
            for field, value in row.items():
                if value is None:
                    null_counts[field] += 1
    duplicates = 0
    business_variant_rows = 0
    for group in candidates.values():
        by_content: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_content[digest(row["source_fields"])].append(row)
        if (len(by_content) > 1
                and contract.rules.get("identity_policy") == "preserve_source_variants"):
            for variant_key, occurrences in by_content.items():
                row = occurrences[0]
                row.update(
                    business_variant_key=variant_key,
                    business_key=digest([row["business_group_key"], variant_key]),
                    identity_resolution="business_group_source_variant",
                    quality_tier="usable_with_annotation",
                    source_occurrences=[item["source_record_id"] for item in occurrences],
                )
                accepted.append(row)
                business_variant_rows += 1
                duplicates += len(occurrences) - 1
        elif len(by_content) > 1:
            quarantined.extend(
                {**row, "issues": ["conflicting_business_key"]} for row in group
            )
        else:
            row = group[0]
            row["source_occurrences"] = [item["source_record_id"] for item in group]
            accepted.append(row)
            duplicates += len(group) - 1
    usable_with_annotation_rows = sum(
        row.get("quality_tier") == "usable_with_annotation" for row in accepted
    )
    quality = {
        "source_rows": len(records), "accepted_rows": len(accepted),
        "quarantined_rows": len(quarantined), "exact_duplicate_rows": duplicates,
        "business_variant_rows": business_variant_rows,
        "usable_with_annotation_rows": usable_with_annotation_rows,
        "null_counts": dict(null_counts),
        "status": "empty" if not records else "failed" if not accepted else
        "partial" if quarantined or usable_with_annotation_rows else "passed",
        "scope": "schema_identity_values_only",
        "calendar_universe_checked": False,
        "historical_pit_certified": False,
    }
    return accepted, quarantined, quality
