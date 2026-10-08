"""用时点可见的数据生成可追溯的信息产品和研究、归档数据集。"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .access import readable_policy
from .coverage import coverage_report
from .models import ENABLED, PRODUCTS
from .pit import as_of
from .store import Store, digest, encode


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """提取可解释的样本统计，不将缺失值补零；两条有 pct_chg 的行输出均值及样本数。

    参数是时点可见行列表；返回行数和明确数值字段的样本统计，缺失值不参与计算。
    示例：summarize([{'close': 2}, {'close': 4}]) 的 close.mean → 3.0。
    """
    statistics: dict[str, Any] = {"rows": len(rows)}
    for field in ("pct_chg", "close", "unit_nav", "net_mf_amount", "amount", "oi", "rank"):
        values = [
            float(row[field])
            for row in rows
            if isinstance(row.get(field), (int, float))
            and not isinstance(row[field], bool)
            and math.isfinite(float(row[field]))
        ]
        if values:
            statistics[field] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    return statistics


def product_content(product: str, datasets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """按产品组织数据视图；security 按证券形成全景档案，events 给出明确事件日期。

    参数为产品标识与 API→记录列表；返回产品视图字典，不生成无来源判断。
    示例：product_content('security', {'daily': [{'ts_code': 'A.SZ'}]})
    返回 profiles 下的 A.SZ 档案；product_content('events', {}) 返回空 events 列表。
    """
    flattened = [dict(row, api_name=name) for name, rows in datasets.items() for row in rows]
    if product == "security":
        profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in flattened:
            profiles[str(row.get("ts_code", "未提供证券代码"))].append(row)
        return {"profiles": dict(profiles)}
    if product == "events":
        events = [
            {
                "api_name": row["api_name"],
                "ts_code": row.get("ts_code"),
                "dates": {
                    key: row[key]
                    for key in ("ann_date", "f_ann_date", "ex_date", "div_proc", "end_date")
                    if row.get(key) is not None
                },
                "response_hash": row["response_hash"],
            }
            for row in flattened
        ]
        return {"events": events, "interpretation": "展示已观察到的事件，不预测未公布事项"}
    if product in {"rotation", "sentiment"}:
        ranking = sorted(flattened, key=lambda row: float(row.get("pct_chg") or 0), reverse=True)
        return {"ranking": ranking, "ranking_basis": "当前样本的 pct_chg 降序，缺值保留为未知"}
    return {
        "statistics": {name: summarize(rows) for name, rows in datasets.items()},
        "records": datasets,
    }


def build_products(
    store: Store, destination: Path, cutoff: str, source: str = "fixture"
) -> list[Path]:
    """生成十类 Markdown/JSON 及明确映射的研究归档文件，记录实际消费证据。

    示例：fixture daily 已采集，build_products(..., '2024-01-03') 写市场总览及来源哈希；
    缺少输入的产品明确显示缺数据，不虚构结论，所有 fixture 产品标注离线示例。
    """
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {key: {} for key in PRODUCTS}
    for entry in store.catalog():
        if entry.policy is None or (source != "live" and entry.lifecycle not in ENABLED):
            continue
        if source == "live":
            try:
                readable_policy(entry)
            except ValueError:
                continue
        records = store.rows(entry.api_name, source)
        if not records:
            continue
        visible = as_of(records, cutoff, entry.policy.primary_key)
        if not visible:
            continue
        for consumer in entry.policy.consumers:
            grouped.setdefault(consumer, {})[entry.api_name] = visible
    paths: list[Path] = []
    for product, datasets in grouped.items():
        if product == "coverage":
            continue
        title = PRODUCTS.get(product, product).replace(":", "：").replace("/", "／")
        slug = product if product in PRODUCTS else digest(product)[:16]
        payload = {
            "product": product,
            "title": title,
            "as_of": cutoff,
            "source": source,
            "fixture": source == "fixture",
            "inputs": sorted(datasets),
            "lineage": sorted({row["response_hash"] for rows in datasets.values() for row in rows}),
            "status": "available" if datasets else "missing_inputs",
            "content": product_content(product, datasets),
        }
        json_path = destination / f"{slug}.json"
        json_path.write_text(encode(payload) + "\n", encoding="utf-8")
        markdown_path = destination / f"{title}.md"
        markdown = f"# {title}\n\n"
        markdown += (
            "确定性离线示例，不代表真实市场结论。\n\n"
            if source == "fixture"
            else "账户采集数据。\n\n"
        )
        markdown += f"查询时点：{cutoff}。状态：{payload['status']}。\n\n"
        if datasets:
            markdown += "| 接口 | 可见记录 | 响应哈希 |\n| --- | ---: | --- |\n"
            for name, rows in datasets.items():
                hashes = ", ".join(sorted({row["response_hash"] for row in rows}))
                markdown += f"| {name} | {len(rows)} | {hashes} |\n"
        else:
            markdown += "尚无符合时点约束的输入数据，暂不生成业务判断。\n"
        markdown += "\n## 数据视图\n\n```json\n" + encode(payload["content"]) + "\n```\n"
        markdown_path.write_text(markdown, encoding="utf-8")
        manifest_hash = digest(payload)
        for name, rows in datasets.items():
            store.consume(product, name, source, cutoff, len(rows), manifest_hash)
        paths.extend([json_path, markdown_path])
    report = coverage_report(store, cutoff, source)
    coverage_path = destination / "coverage.json"
    coverage_path.write_text(encode(report) + "\n", encoding="utf-8")
    markdown_path = destination / "接口利用率与覆盖债务.md"
    markdown_path.write_text(
        "# 接口利用率与覆盖债务\n\n"
        + report["note"]
        + "\n\n```json\n"
        + encode(report)
        + "\n```\n",
        encoding="utf-8",
    )
    return paths + [coverage_path, markdown_path]
