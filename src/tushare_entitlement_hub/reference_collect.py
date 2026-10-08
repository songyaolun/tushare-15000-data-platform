"""正式执行声明范围的完整响应采集，基于提供方分页证据恢复并保存全部原始页。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .client import Client, ErrorKind, ProviderError
from .models import CatalogEntry
from .reference_contract import ReferenceContract, ReferenceRecord
from .store import digest, encode


def collect_reference_pages(
    entry: CatalogEntry, contract: ReferenceContract, client: Client,
    parameters: dict[str, Any], observed_at: str, max_pages: int = 100,
) -> tuple[list[dict[str, Any]], list[ReferenceRecord], str]:
    """取完所声明请求范围的页面，不将 probe 单行或 provider has_more 当成完成。

    输入目录、契约、客户端、完整业务参数、观察时间与页上限，返回页证据、行和完整度。
    示例：第一页 has_more=True、第二页False时采集两页，offset累计真实返回条数。
    明确 has_more 优先于旧文档的分页猜测；无此信号时只有契约分页短页证明结束。
    相同非空页重复、页上限或无进展返回 incomplete，已有原始页仍全部保留。
    """
    if max_pages < 1:
        raise ValueError("invalid_max_pages")
    params = {name: value for name, value in parameters.items() if name not in {"limit", "offset"}}
    if contract.requested_fields and entry.api_name != "pro_bar":
        params["fields"] = contract.requested_fields
    page_size = min(entry.row_limit or contract.policy.page_size, 5000)
    pagination = contract.policy.supports_pagination
    offset = 0
    evidence: list[dict[str, Any]] = []
    records: list[ReferenceRecord] = []
    seen_pages = set()
    for _ in range(max_pages):
        request = {**params, **({"limit": page_size, "offset": offset} if pagination else {})}
        try:
            page = client.fetch(entry.api_name, request, observed_at)
        except ProviderError as error:
            if not evidence:
                raise
            return evidence, records, "incomplete_" + error.kind
        data = page.raw.response["data"]
        evidence.append({
            "raw_path": str(page.raw.path), "response_hash": page.raw.response_hash,
            "parameters": request, "rows": len(page.rows), "fields": page.fields,
            "has_more": data.get("has_more"), "processed_rows": 0,
        })
        fingerprint = digest(page.rows)
        if page.rows and fingerprint in seen_pages:
            return evidence, records, "incomplete_repeated_page"
        evidence[-1]["processed_rows"] = len(page.rows)
        seen_pages.add(fingerprint)
        records.extend((row, page.raw.response_hash, position, page.raw.request_id)
                       for position, row in enumerate(page.rows))
        has_more = data.get("has_more")
        if has_more is False:
            return evidence, records, "complete_provider_response"
        if has_more is not None and type(has_more) is not bool:
            raise ProviderError("contract_failure", "分页状态不是布尔值")
        if has_more is True:
            if not page.rows:
                return evidence, records, "incomplete_no_progress"
            pagination = True
        elif pagination and len(page.rows) < page_size:
            return evidence, records, "complete_contract_short_page"
        elif not pagination:
            if entry.api_name == "pro_bar":
                return evidence, records, "complete_composed_range"
            return evidence, records, "unknown_provider_limit"
        offset += len(page.rows)
    return evidence, records, "incomplete_page_budget"


def collect_reference_partitions(
    entry: CatalogEntry, contract: ReferenceContract, client: Client,
    parameters: dict[str, Any] | list[dict[str, Any]], observed_at: str,
    progress: Callable[[str], None],
) -> tuple[list[dict[str, Any]], list[ReferenceRecord], str,
           list[dict[str, Any]]]:
    """逐分区取完声明范围，单个分区失败仍保留证据并继续其他分区。

    输入一个请求或请求列表及进度回调，返回页面、行、整体完整度和逐分区结果。
    例如两只ETF分别请求可绕开全市场offset上限；其中一只失败则整体为incomplete_partitions。
    没有成功页面且提供方拒绝时保留错误分类，不把权限失败标成空数据。
    """
    partitions = [parameters] if isinstance(parameters, dict) else parameters
    if not partitions or not all(isinstance(item, dict) for item in partitions):
        raise ValueError('invalid_collection_partitions')
    pages: list[dict[str, Any]] = []
    records: list[ReferenceRecord] = []
    results: list[dict[str, Any]] = []
    for index, partition in enumerate(partitions):
        try:
            part_pages, part_records, status = collect_reference_pages(
                entry, contract, client, partition, observed_at
            )
            for page in part_pages:
                page['partition_index'] = index
            pages.extend(part_pages)
            records.extend(part_records)
        except ProviderError as error:
            status = error.kind
        results.append({'parameters': partition, 'status': status})
        progress(encode({'api_name': entry.api_name, 'partition': index + 1,
                         'partition_count': len(partitions), 'collection_status': status}))
    if not pages:
        raise ProviderError(cast(ErrorKind, results[0]['status']), '所有分区请求均未取得成功页面')
    complete = all(item['status'].startswith('complete') for item in results)
    return pages, records, 'complete_partitions' if complete else 'incomplete_partitions', results
