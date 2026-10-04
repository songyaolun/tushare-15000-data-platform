"""局部消费模型缓存仍保留完整输入安全边界与规范语义。"""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_research_model_reuse import rich_research_results

from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_validation import ConsumptionModels
from tushare_entitlement_hub.equity_research.models import PRODUCT_IDS
from tushare_entitlement_hub.product_foundation.contracts import EvidenceRef, QualityGateResult


def test_repeated_consumption_evidence_parses_once_and_keeps_model_bytes(monkeypatch):
    """真实attach结果中重复引用只解析唯一字典一次，gate约束与结果字节保持完整。"""
    inputs, results = rich_research_results(1)
    result = attach_consumption(inputs, PRODUCT_IDS[4], results)[0]
    payload = result.to_json()
    original = EvidenceRef.from_dict.__func__
    calls = []

    def counted(cls, value):
        """计数但仍执行原公开转换；例：重复gate引用不能反复解析相同证据字典。"""
        calls.append(value)
        return original(cls, value)

    # 先严格重建本体，避免首次不可变模型认证也计入消费记录的字典解析次数。
    result = type(result).from_json(payload)
    monkeypatch.setattr(EvidenceRef, "from_dict", classmethod(counted))
    validate_result_consumption(result)
    records = result.details["consumption"]
    occurrences = sum(
        len(record["evidence_refs"]) + sum(len(gate["evidence_refs"]) for gate in record["gates"])
        for record in records
    )
    assert calls and len(calls) < occurrences
    assert result.to_json() == payload
    print(f"消费证据字典出现={occurrences}, 实际from_dict={len(calls)}")


def test_local_cache_isolated_from_mutation_and_rejects_invalid_nested_models():
    """同一可变字典修改后必须重新验证；未知字段、NaN、凭据、错误证据序列和错误模型类型仍拒绝。"""
    inputs, results = rich_research_results(1)
    result = attach_consumption(inputs, PRODUCT_IDS[4], results)[0]
    reference = result.envelope.evidence_refs[0].to_dict()
    cache = ConsumptionModels()
    first = cache.reference(reference)
    assert cache.reference(deepcopy(reference)) is first
    reference["source_record_id"] = "another_record"
    assert cache.reference(reference) != first
    assert cache.reference(reference).to_json() == EvidenceRef.from_dict(reference).to_json()
    gate = replace(result.envelope.quality_gates[0], evidence_refs=(first,)).to_dict()
    assert cache.gate_json(gate) == QualityGateResult.from_dict(gate).to_json()
    assert cache.gate_json(deepcopy(gate)) == cache.gate_json(gate)
    with pytest.raises(ValueError):
        cache.reference(reference | {"unknown_field": 1})
    with pytest.raises(ValueError):
        cache.reference(reference | {"source_record_id": float("nan")})
    with pytest.raises(ValueError):
        cache.reference(reference | {"token": "<REDACTED>"})
    with pytest.raises(ValueError):
        cache.gate_json(gate | {"evidence_refs": [reference | {"unknown_field": 1}]})
    with pytest.raises(ValueError):
        cache.gate_json(gate | {"evidence_refs": "invalid"})
    with pytest.raises(ValueError):
        cache.gate_json(gate | {"unknown_field": 1})
    with pytest.raises(ValueError):
        cache.gate_json([])
    without_refs = dict(gate)
    del without_refs["evidence_refs"]
    assert cache.gate_json(without_refs) == QualityGateResult.from_dict(without_refs).to_json()
    assert not ConsumptionModels().references
