"""结果投影的旧版字节契约与重复序列化调用计数。"""

from collections import Counter
from dataclasses import replace

from test_research_model_reuse import rich_research_results

from tushare_entitlement_hub import raw
from tushare_entitlement_hub.equity_research.storage import record_rows
from tushare_entitlement_hub.product_foundation import model
from tushare_entitlement_hub.product_foundation.contracts import Truth
from tushare_entitlement_hub.store import digest


def test_projection_keeps_legacy_bytes_and_serializes_root_once(monkeypatch):
    """指标、贡献、门禁、共用正反证及父血缘的完整投影与旧头字节相同，每结果只序列化根一次。"""
    _, results = rich_research_results(1)
    first = results[0]
    result = replace(
        first,
        envelope=replace(first.envelope, contrary_evidence=first.envelope.evidence_refs),
        parent_result_ids=("parent-one",),
        predicate_results={"second": Truth.UNKNOWN, "first": Truth.TRUE},
    )
    calls = Counter()
    original = model.Model.to_json

    def counted(value):
        """计数后调用真实公开序列化边界；例：一个ResearchResult只计一次，不替换校验。"""
        calls[type(value).__name__] += 1
        return original(value)

    monkeypatch.setattr(model.Model, "to_json", counted)
    rows = list(record_rows(result))
    assert digest(rows) == "7a4dbb9460aebf072c1db716f6ef60ef582f4a00cfb16df01d407451991cb8e6"
    assert calls == {"ResearchResult": 1}


def test_non_assignment_strings_skip_regex_but_credentials_still_fail(monkeypatch):
    """SHA256和枚举文本不含赋值分隔符，跳过不可能命中的正则；含冒号或等号仍完整检查。"""
    import pytest

    original = raw._ASSIGNMENT
    searched = []

    class CheckedPattern:
        """记录真正执行的敏感赋值检查，不替换正则语义。"""

        def search(self, text):
            """透传原正则；例：password=<REDACTED>仍返回匹配。"""
            searched.append(text)
            return original.search(text)

    monkeypatch.setattr(raw, "_ASSIGNMENT", CheckedPattern())
    raw.assert_public(["a" * 64, "observed", "unknown", "中文", ""] * 200)
    assert searched == []
    raw.assert_public("cn:security:A:listed")
    class MisleadingString(str):
        """模拟覆写包含操作的字符串；公开边界仍须按实际字符串内容检查。"""

        def __contains__(self, value):
            """故意返回False；例：in不能作为安全预筛选的可信依据。"""
            return False

    for text in ("password=<REDACTED>", "TOKEN: <REDACTED>", MisleadingString("token=<REDACTED>")):
        with pytest.raises(ValueError):
            raw.assert_public(text)
    assert len(searched) == 4


def test_projection_snapshot_isolation_and_public_boundary(monkeypatch):
    """修改已产出的字典不影响后续投影，原模型可变字段被注入凭据时仍在任何输出前拒绝。"""
    import pytest

    _, results = rich_research_results(1)
    result = results[0]
    rows = record_rows(result)
    first_table, first = next(rows)
    assert first_table == "result"
    first["payload"]["metrics"][0]["value"] = -1
    metrics = [row for table, row in rows if table == "feature"]
    assert metrics[0]["payload"]["value"] == 0.5
    result.predicate_results["token"] = Truth.UNKNOWN
    with pytest.raises(ValueError):
        next(record_rows(result))
