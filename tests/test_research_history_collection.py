"""历史采集契约：全部使用合成响应，禁止网络和真实凭据。"""

from pathlib import Path

import pytest

from tushare_entitlement_hub.client import ProviderError
from tushare_entitlement_hub.research_history.collection import (
    FIELDS,
    HistoryCollector,
    partition_quality,
    response_records,
)


def table(api_name: str, rows: int = 1) -> dict:
    """生成合法合成行；输入 daily，输出一天一只证券的原始响应示例。"""
    fields = FIELDS[api_name].split(",")
    records = []
    for number in range(rows):
        record = {field: 1.0 for field in fields}
        record.update(ts_code=f"{number:06}.SZ", trade_date="20210104", end_date="20201231")
        records.append([record[field] for field in fields])
    return {"code": 0, "data": {"fields": fields, "items": records}}


class SequenceTransport:
    """顺序返回响应，用于重试和分页场景。"""

    def __init__(self, responses: list) -> None:
        """输入响应列表，输出计数为零的假传输实例。"""
        self.responses, self.calls = responses, 0

    def request(self, api_name: str, parameters: dict) -> dict:
        """输入公开参数，输出下一响应；异常实例被原样抛出。"""
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize(
    "response,message",
    [
        ({"code": -1}, "provider_response_failure"),
        ({"code": 0}, "provider_data_missing"),
        ({"code": 0, "data": {}}, "provider_table_invalid"),
        ({"code": 0, "data": {"fields": ["x", "x"], "items": []}}, "provider_table_invalid"),
        ({"code": 0, "data": {"fields": ["x"], "items": [[1, 2]]}}, "provider_row_invalid"),
    ],
)
def test_response_rejects_malformed_tables(response: dict, message: str) -> None:
    """场景：提供方失败或畸形表；预期固定错误且不回显原文。"""
    with pytest.raises(ValueError, match=message):
        response_records(response)


def test_response_records_and_partition_quality() -> None:
    """场景：合法行情和停牌零值；预期保留原始值并通过契约。"""
    response = table("daily")
    assert len(response_records(response)) == 1
    response["data"]["items"][0][2] = 0
    assert partition_quality("daily", {"trade_date": "20210104"}, response)["status"] == "passed"
    assert partition_quality("trade_cal", {}, table("trade_cal", 0))["status"] == "passed"


def test_partition_quality_rejects_truncation_dates_duplicates_and_prices() -> None:
    """场景：同日重复、缺字段、错误日期和负/缺价格；预期列明质量问题。"""
    response = table("daily", 2)
    response["data"]["items"][1] = response["data"]["items"][0].copy()
    response["data"]["has_more"] = True
    response["data"]["items"][0][2] = -1
    response["data"]["items"][1][3] = None
    response["data"]["fields"].pop()
    for row in response["data"]["items"]:
        row.pop()
    quality = partition_quality("daily", {"trade_date": "20210105"}, response)
    assert {
        "missing_fields",
        "possible_truncation",
        "wrong_trade_date",
        "duplicate_security_date",
        "invalid_open",
        "invalid_high",
    } <= set(quality["issues"])
    assert (
        partition_quality("daily", {"trade_date": "20210104"}, table("daily", 0))["status"]
        == "failed"
    )
    response = table("adj_factor")
    response["data"]["items"][0][-1] = 0
    assert partition_quality("adj_factor", {}, response)["status"] == "failed"
    assert partition_quality("daily", {}, table("daily", 6000))["status"] == "failed"


def test_collector_cache_and_corrupt_quality(tmp_path: Path) -> None:
    """场景：同公开参数二次采集及缓存摘要损坏；预期复用成功缓存并重采损坏缓存。"""
    import json

    transport = SequenceTransport([table("daily"), table("daily")])
    collector = HistoryCollector(
        transport, tmp_path, interval_seconds=0, sleep=lambda seconds: None
    )
    result = collector.collect("daily", {"trade_date": "20210104"})
    assert not result["cache_reused"]
    assert collector.collect("daily", {"trade_date": "20210104"})["cache_reused"]
    path = Path(result["raw_path"])
    envelope = json.loads(path.read_text())
    envelope["quality"] = {}
    path.write_text(json.dumps(envelope))
    assert not collector.collect("daily", {"trade_date": "20210104"})["cache_reused"]
    assert transport.calls == 2


@pytest.mark.parametrize("failure", [ProviderError("transient_failure", "temporary"), {"code": -1}])
def test_collector_retries_transient_and_provider_errors(tmp_path: Path, failure: object) -> None:
    """场景：暂时失败后恢复；预期有限重试且成功原始层存在。"""
    transport = SequenceTransport([failure, table("daily")])
    collector = HistoryCollector(
        transport, tmp_path, interval_seconds=0, sleep=lambda seconds: None
    )
    assert collector.collect("daily", {"trade_date": "20210104"})["quality"]["status"] == "passed"
    assert transport.calls == 2


@pytest.mark.parametrize(
    "failure,retries,expected",
    [
        (ProviderError("unauthorized", "denied"), 2, ProviderError),
        (ProviderError("transient_failure", "temporary"), 1, ProviderError),
        ({"code": -1}, 1, ValueError),
        (table("daily", 0), 1, ValueError),
    ],
)
def test_collector_failure_preserves_raw_when_available(
    tmp_path: Path, failure: object, retries: int, expected: type
) -> None:
    """场景：权限失败、重试用尽及质量失败；预期固定异常，收到的原始数据先保存。"""
    collector = HistoryCollector(
        SequenceTransport([failure]),
        tmp_path,
        retries=retries,
        interval_seconds=0,
        sleep=lambda seconds: None,
    )
    with pytest.raises(expected):
        collector.collect("daily", {"trade_date": "20210104"})
    if isinstance(failure, dict):
        assert list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("interval,retries", [(-1, 1), (float("nan"), 1), (0, 0)])
def test_collector_invalid_configuration(tmp_path: Path, interval: float, retries: int) -> None:
    """场景：非有限速率或零重试；预期初始化拒绝。"""
    with pytest.raises(ValueError):
        HistoryCollector(
            SequenceTransport([]), tmp_path, interval_seconds=interval, retries=retries
        )


def test_pagination_complete_and_rejects_repeated_or_exhausted(tmp_path: Path) -> None:
    """场景：多页、重复页、页数用尽及错误配置；预期完整返回或明确拒绝。"""
    transport = SequenceTransport([table("income_vip", 2), table("income_vip", 0)])
    collector = HistoryCollector(
        transport, tmp_path, interval_seconds=0, sleep=lambda seconds: None
    )
    assert len(collector.collect_pages("income_vip", {"period": "20201231"}, page_size=2)) == 2
    with pytest.raises(ValueError, match="configuration"):
        collector.collect_pages("income_vip", {}, page_size=5000)
    with pytest.raises(ValueError, match="configuration"):
        collector.collect_pages("income_vip", {}, maximum_pages=0)
    collector = HistoryCollector(
        SequenceTransport([table("income_vip", 2)] * 2),
        tmp_path / "repeat",
        interval_seconds=0,
        sleep=lambda seconds: None,
    )
    with pytest.raises(ValueError, match="repeated_page"):
        collector.collect_pages("income_vip", {}, page_size=2)
    collector = HistoryCollector(
        SequenceTransport([table("income_vip", 2)]),
        tmp_path / "exhaust",
        interval_seconds=0,
        sleep=lambda seconds: None,
    )
    with pytest.raises(ValueError, match="exhausted"):
        collector.collect_pages("income_vip", {}, page_size=2, maximum_pages=1)


def test_non_mainboard_null_prices_are_preserved() -> None:
    """场景：北交所历史缺昨日收盘；预期保存该行且沪深主板价格契约通过。"""
    response = table("daily")
    response["data"]["items"][0][0] = "920527.BJ"
    response["data"]["items"][0][6] = None
    assert response_records(response)[0]["pre_close"] is None
    assert partition_quality("daily", {"trade_date": "20210104"}, response)["status"] == "passed"


def test_pagination_has_more_requires_tail_page(tmp_path: Path) -> None:
    """场景：满页标明has_more；预期继续抓取且短尾页才能证明完整。"""
    response = table("income_vip", 2)
    response["data"]["has_more"] = True
    collector = HistoryCollector(
        SequenceTransport([response, table("income_vip", 0)]),
        tmp_path,
        interval_seconds=0,
        sleep=lambda seconds: None,
    )
    assert len(collector.collect_pages("income_vip", {}, page_size=2)) == 2


def test_pagination_rejects_provider_smaller_limit(tmp_path: Path) -> None:
    """场景：提供方忽略请求limit且短页has_more；预期拒绝，避免offset跨过未读行。"""
    response = table("income_vip", 1)
    response["data"]["has_more"] = True
    collector = HistoryCollector(
        SequenceTransport([response]),
        tmp_path,
        interval_seconds=0,
        sleep=lambda seconds: None,
    )
    with pytest.raises(ValueError, match="unexpected_page_size"):
        collector.collect_pages("income_vip", {}, page_size=2)


def test_financial_period_mismatch_is_rejected() -> None:
    """场景：财务响应返回别的报告期；预期明确拒绝被忽略的period参数。"""
    response = table("income_vip")
    response["data"]["items"][0][4] = "20201231"
    assert (
        "wrong_report_period"
        in partition_quality("income_vip", {"period": "20210331"}, response)["issues"]
    )
