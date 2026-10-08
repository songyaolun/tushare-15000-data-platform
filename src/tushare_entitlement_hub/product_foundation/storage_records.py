"""共享产品表的质量、范围、贡献和血缘记录契约。"""

from dataclasses import dataclass
from typing import Literal

from .contracts import FeatureContribution, QualityGateResult
from .model import Count, Identifier, Model, Period, ReasonCode, Timestamp


@dataclass(frozen=True, kw_only=True)
class ScoreComponent(Model):
    """一个结果的一个可复算评分贡献。"""

    result_id: Identifier
    component: FeatureContribution


@dataclass(frozen=True, kw_only=True)
class GateRecord(Model):
    """结果与门禁、规则版本绑定，保留每次运行状态。"""

    result_id: Identifier
    rule_version: Identifier
    gate: QualityGateResult


@dataclass(frozen=True, kw_only=True)
class CoverageSlice(Model):
    """请求范围不冒充全市场，缺失接口的计数允许未知。"""

    source_api: Identifier
    scope_id: Identifier
    partition: Identifier
    run_id: Identifier
    request_scope: str
    source_rows: Count | None
    accepted_rows: Count | None
    quarantined_rows: Count | None
    duplicate_rows: Count | None
    observed_at: Timestamp | None
    data_as_of: Period | None
    manifest_hash: Identifier | None
    reason_codes: tuple[ReasonCode, ...]

    def __post_init__(self) -> None:
        """完整账本必须守恒；例如 2 来源、1 合格、1 隔离、0 重复 → 通过。

        任一计数未知时保留未知，不能把缺表补成零条合格记录。
        """
        super().__post_init__()
        counts = (self.source_rows, self.accepted_rows, self.quarantined_rows, self.duplicate_rows)
        if all(value is not None for value in counts):
            if counts[0] != sum(value for value in counts[1:] if value is not None):
                raise ValueError("coverage_accounting_mismatch")


@dataclass(frozen=True, kw_only=True)
class LineageEdge(Model):
    """输出到来源字段的有向边；业务组摘要不充当跨接口外键。"""

    output_id: Identifier
    source_api: Identifier
    source_record_id: Identifier
    fields: tuple[Identifier, ...]
    contract_hash: Identifier
    relationship: Literal["used", "excluded", "parent"] = "used"
    reason_code: ReasonCode | None = None
    parent_result_id: Identifier | None = None
