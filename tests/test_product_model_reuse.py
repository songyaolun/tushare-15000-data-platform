"""嵌套模型复用的严格边界：验证凭据、递归不可变性、容器隔离及序列化一致性。"""

from dataclasses import dataclass, field, replace
from functools import cached_property
from types import SimpleNamespace

import pytest
from test_product_foundation import evidence
from test_product_foundation_edges import sample_envelope

from tushare_entitlement_hub.equity_research.models import ResearchResult
from tushare_entitlement_hub.product_foundation import model
from tushare_entitlement_hub.product_foundation.contracts import (
    ConfidenceComponents,
    EvidenceRef,
    QualityGateResult,
    Truth,
)
from tushare_entitlement_hub.product_foundation.model import Model, Score


@dataclass(frozen=True, kw_only=True)
class MutableLeaf(Model):
    """含嵌套映射及列表输入的合成契约，不能因为外层冻结就共享内部容器。"""

    values: dict[str, tuple[Score, ...]] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MutableTree(Model):
    """用于验证tuple中的含字典子模型也必须复制，而非只检查根对象类型。"""

    children: tuple[MutableLeaf, ...]


@dataclass(frozen=True, kw_only=True)
class CachedLeaf(Model):
    """声明字段不可变但带额外可变实例缓存，不允许共享缓存状态。"""

    value: str

    @cached_property
    def cache(self):
        """生成非契约的可变缓存；例：value='x'得到{'value':'x'}，JSON仍只含声明字段。"""
        return {"value": self.value}


def test_validated_nested_models_reuse_without_reconversion(monkeypatch):
    """证据和门禁首次复核成功后按原实例复用；后续100次复用不调用asdict/from_dict。"""
    reference = evidence()
    gate = QualityGateResult(gate_id="ready", result=Truth.TRUE, evidence_refs=(reference,))
    assert gate.evidence_refs[0] is reference
    assert model._typed(gate, QualityGateResult) is gate
    before = gate.to_json()

    def unexpected(*args, **kwargs):
        """已认证不可变实例复用不应序列化或重建；任意调用均说明退回了旧递归路径。"""
        pytest.fail("不可变模型发生重复转换")

    with monkeypatch.context() as patch:
        patch.setattr(model, "asdict", unexpected)
        patch.setattr(Model, "from_dict", classmethod(unexpected))
        for _ in range(100):
            assert model._typed(gate, QualityGateResult) is gate
    assert gate.to_json() == before


def test_mutable_inputs_and_model_instances_remain_isolated():
    """dict/list输入深度隔离；含dict子模型再嵌入时继续复制，输出字典也不能回改原模型。"""
    payload = {"children": [{"values": {"scores": [10, 20]}}]}
    tree = MutableTree.from_dict(payload)
    payload["children"][0]["values"]["scores"].append(101)
    assert tree.children[0].values == {"scores": (10.0, 20.0)}
    assert not model._current_immutable(tree.children[0].values)
    copied = model._typed(tree, MutableTree)
    assert copied is not tree and copied.children[0] is not tree.children[0]
    assert copied.children[0].values is not tree.children[0].values
    tree.children[0].values["scores"] = (30.0,)
    assert copied.children[0].values == {"scores": (10.0, 20.0)}
    output = copied.to_dict()
    output["children"][0]["values"]["scores"][0] = 200
    assert copied.children[0].values["scores"][0] == 10
    assert not model._current_immutable(copied)


@pytest.mark.parametrize("bad", [True, -1, 101, float("nan"), float("inf"), -float("inf"), "10"])
def test_invalid_nested_dictionary_numbers_still_fail(bad):
    """非法嵌套字典中的布尔、越界、非有限或字符串分数均被完整递归校验拒绝。"""
    with pytest.raises(ValueError):
        MutableTree.from_dict({"children": [{"values": {"scores": [bad]}}]})
    with pytest.raises(ValueError):
        sample_envelope(confidence_components={"quality": bad})


@pytest.mark.parametrize(
    "change", [{"unknown": 1}, {"fields": []}, {"source_api": ""}, {"observed_at": "bad"}]
)
def test_invalid_nested_evidence_and_private_fields_still_fail(change):
    """额外字段、空字段、非法身份和时间均不可通过嵌套字典绕过验证，私有凭据也不能注入。"""
    payload = evidence().to_dict() | change
    with pytest.raises(ValueError):
        QualityGateResult.from_dict(
            {"gate_id": "ready", "result": "true", "evidence_refs": [payload]}
        )
    with pytest.raises(ValueError, match="model_fields_invalid"):
        EvidenceRef.from_dict(evidence().to_dict() | {model._IMMUTABLE_SEAL: []})


@pytest.mark.parametrize("extended", [False, True])
def test_model_subclass_never_matches_base_annotation(extended):
    """同字段和新增字段的Model子类都不能冒充基类，即使自身已验证也必须拒绝。"""

    @dataclass(frozen=True, kw_only=True)
    class SameShape(EvidenceRef):
        """与EvidenceRef结构相同的子类，仍不是精确匹配的契约类型。"""

    @dataclass(frozen=True, kw_only=True)
    class Extended(EvidenceRef):
        """增加字段的子类，不得通过asdict降格成基类。"""

        extra: str = "synthetic"

    child = (Extended if extended else SameShape).from_dict(evidence().to_dict())
    model._typed(child, type(child))
    with pytest.raises(ValueError, match="invalid_model_type"):
        QualityGateResult(gate_id="ready", result=Truth.TRUE, evidence_refs=(child,))
    with pytest.raises(ValueError, match="invalid_model_type"):
        model._typed(ConfidenceComponents(), EvidenceRef)


def test_replace_keeps_parent_validation_and_copies_mutable_envelope(monkeypatch):
    """dataclasses.replace保留父级校验及encode；ResearchResult复制信封字典，同时共享不可变证据。"""
    reference = evidence()
    gate = QualityGateResult(gate_id="ready", result=Truth.TRUE, evidence_refs=(reference,))
    envelope = sample_envelope(
        evidence_refs=(reference,), quality_gates=(gate,), source_as_of={"daily": "2026-09-30"}
    )
    result = ResearchResult(envelope=envelope, predicate_results={"manual": Truth.UNKNOWN})
    assert result.envelope is not envelope
    assert result.envelope.source_as_of is not envelope.source_as_of
    assert result.envelope.quality_gates[0] is gate
    assert result.envelope.evidence_refs[0] is reference
    encoded = []
    original_encode = model.encode

    def counted_encode(value):
        """透传实际JSON校验并计数；例如replace仍至少校验新信封及新结果一次。"""
        encoded.append(type(value))
        return original_encode(value)

    with monkeypatch.context() as patch:
        patch.setattr(model, "encode", counted_encode)
        repeated = replace(result)
        assert len(encoded) >= 2
        with pytest.raises(ValueError, match="historical_trading_not_supported"):
            replace(envelope, historical_tradable=True)
    assert repeated.to_json() == result.to_json()
    envelope.source_as_of["daily"] = "2026-09-29"
    result.predicate_results["manual"] = Truth.TRUE
    assert repeated.envelope.source_as_of["daily"] == "2026-09-30"
    assert repeated.predicate_results["manual"] is Truth.UNKNOWN


@pytest.mark.parametrize("target", ["leaf", "parent"])
def test_bypassing_frozen_invalidates_even_nested_validation_seals(target):
    """低层绕过frozen修改字段后，递归凭据检查失效；非法子字段和门禁约束均重新校验并拒绝。"""
    reference = evidence()
    gate = QualityGateResult(gate_id="ready", result=Truth.TRUE, evidence_refs=(reference,))
    model._typed(gate, QualityGateResult)
    if target == "leaf":
        object.__setattr__(reference, "source_api", "")
    else:
        object.__setattr__(gate, "result", Truth.UNKNOWN)
    assert not model._current_immutable(gate)
    with pytest.raises(ValueError):
        model._typed(gate, QualityGateResult)


def test_unvalidated_domain_state_is_not_certified_before_constructor_finishes():
    """伪造的同类型实例虽字段类型合法，领域约束非法时仍拒绝，不因基类encode通过而获凭据。"""
    gate = QualityGateResult(gate_id="ready", result=Truth.TRUE)
    object.__setattr__(gate, "result", Truth.UNKNOWN)
    with pytest.raises(ValueError, match="gate_reason_required"):
        model._typed(gate, QualityGateResult)
    assert model._IMMUTABLE_SEAL not in gate.__dict__


def test_noncanonical_model_state_is_normalized_without_reusing_original():
    """绕过frozen写入尚未规范化的整数分数后，返回规范float副本，不能按数值相等共享原实例。"""
    value = ConfidenceComponents(quality=10)
    object.__setattr__(value, "quality", 10)
    converted = model._typed(value, ConfidenceComponents)
    assert converted is not value and type(converted.quality) is float
    assert converted.to_json() == ConfidenceComponents(quality=10).to_json()
    assert not model._certify_immutable(0.0, -0.0)
    assert not model._certify_immutable(float("nan"), float("nan"))
    assert not model._certify_immutable((1,), (1, 2))


def test_unregistered_or_mutable_extra_instance_state_is_not_shared(monkeypatch):
    """继承但未声明冻结dataclass的类型以及额外可变缓存不复用；缓存不进入产品JSON。"""

    class Undeclared(EvidenceRef):
        """非dataclass声明的精确子类，作为自身契约时只重建、不认证共享。"""

    item = Undeclared(**evidence().to_dict())
    assert model._typed(item, Undeclared) is not item
    cached = CachedLeaf(value="x")
    assert cached.cache == {"value": "x"}
    copied = model._typed(cached, CachedLeaf)
    assert copied is not cached and "cache" not in copied.__dict__
    assert copied.to_json() == cached.to_json()
    assert not model._certify_immutable(copied, cached)
    monkeypatch.setattr(CachedLeaf, "__dataclass_params__", SimpleNamespace(frozen=False))
    assert not model._certify_immutable(copied, copied)


@pytest.mark.parametrize(
    "seal",
    [
        None,
        (),
        (None, None, ()),
        (model._VALIDATED, Model, ()),
        (model._VALIDATED, EvidenceRef, []),
        (model._VALIDATED, EvidenceRef, ()),
    ],
)
def test_invalid_seal_never_grants_reuse(seal):
    """缺失、错误长度、错误标记或错误类型的凭据均不能授权复用；字段仍按公开契约验证。"""
    reference = evidence()
    object.__setattr__(reference, model._IMMUTABLE_SEAL, seal)
    assert not model._current_immutable(reference)
    assert model._typed(reference, EvidenceRef).to_json() == reference.to_json()


def test_extra_or_missing_attributes_invalidate_current_seal():
    """凭据存在时增删实例字段也不可信；即使缺失字段原值为None且总键数不变也能检出。"""
    reference = evidence()
    model._typed(reference, EvidenceRef)
    object.__setattr__(reference, "extra", {})
    assert not model._current_immutable(reference)
    del reference.__dict__["public_available_at"]
    assert not model._current_immutable(reference)
    assert not model._current_immutable(float("inf"))


def test_validation_metadata_never_overwrites_declared_contract_field():
    """契约字段恰好同名时禁用缓存；私有优化元数据不能覆盖合法字段或改变输出字节。"""

    @dataclass(frozen=True, kw_only=True)
    class ReservedName(Model):
        """显式声明与内部缓存同名的普通字符串字段，仍沿用原序列化契约。"""

        _model_immutable_validation: str

    original = ReservedName(_model_immutable_validation="declared")
    copied = model._typed(original, ReservedName)
    assert copied is not original
    assert copied.to_json() == original.to_json() == '{"_model_immutable_validation":"declared"}'
