"""单结果内复用重复消费证据的严格解析，不缓存跨结果或可变输入对象。"""

from __future__ import annotations

from typing import Any

from ..product_foundation.contracts import EvidenceRef, QualityGateResult
from ..store import encode


class ConsumptionModels:
    """一次消费校验的局部模型索引，以完整规范输入为键；使用后随结果释放。"""

    def __init__(self) -> None:
        """创建空证据和门禁索引；例：两个独立结果不会共享任何解析缓存。"""
        self.references: dict[str, EvidenceRef] = {}
        self.gates: dict[str, str] = {}

    def reference(self, value: dict[str, Any]) -> EvidenceRef:
        """完整验证每种唯一证据，重复字典复用不可变模型；例：相同证据出现30次只from_dict一次。

        每次输入仍通过encode的凭据、有限数值检查；不同字段、类型或时间文本分别验证。
        键为完整不可变文本，不按对象身份信任可能被修改的字典，也不跳过JSON公开边界。
        """
        key = encode(value)
        if key not in self.references:
            self.references[key] = EvidenceRef.from_dict(value)
        return self.references[key]

    def gate_json(self, value: dict[str, Any]) -> str:
        """完整检查唯一门禁，并复用本次已验证的嵌套证据，返回规范模型JSON。

        例：同一证据被十个不同gate引用，证据字典只解析一次；门禁自身约束逐个检查。
        非列表/元组证据不预处理，交给原模型报告类型错误；未知字段仍由from_dict拒绝。
        """
        key = encode(value)
        if key not in self.gates:
            converted = value
            if isinstance(value, dict) and isinstance(value.get("evidence_refs"), (list, tuple)):
                converted = value | {
                    "evidence_refs": tuple(self.reference(ref) for ref in value["evidence_refs"])
                }
            self.gates[key] = QualityGateResult.from_dict(converted).to_json()
        return self.gates[key]
