"""持久化校验的结构计数：逐结果严格解析，不因结果总数反复扫描其他结果。"""

from collections import Counter

import duckdb
import pytest
from research_fixture import AS_OF, write_reference

from tushare_entitlement_hub.equity_research.models import ResearchResult
from tushare_entitlement_hub.equity_research.pipeline import run_research
from tushare_entitlement_hub.equity_research.storage import validate_research


@pytest.mark.parametrize("entities", [2, 8])
def test_validation_parses_each_stored_result_once(tmp_path, monkeypatch, entities):
    """两档真实六产品合成库各结果只完整反序列化一次，校验消耗的字节数等于库中载荷总量。"""
    reference, config = write_reference(tmp_path, entities)
    output = tmp_path / "validation-scale.duckdb"
    summary = run_research(reference, output, AS_OF, config)
    counts = Counter()
    original = ResearchResult.from_json.__func__

    def counted(cls, payload):
        """透传完整公开解析边界并计数；例：21条结果恰好调用21次，不跳过任何嵌套校验。"""
        counts["results"] += 1
        counts["bytes"] += len(payload.encode("utf-8"))
        return original(cls, payload)

    monkeypatch.setattr(ResearchResult, "from_json", classmethod(counted))
    assert validate_research(reference, output)["valid"]
    with duckdb.connect(
        str(output), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
    ) as connection:
        expected_count, expected_bytes = connection.execute(
            "SELECT count(*),sum(octet_length(encode(CAST(payload AS VARCHAR)))) "
            "FROM research.result"
        ).fetchone()
    assert counts["results"] == expected_count == summary["result_count"] == 5 * entities + 1
    assert counts["bytes"] == expected_bytes
