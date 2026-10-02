"""不可变数据契约及严格、可逆的 JSON 边界，不依赖在线服务。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from math import isfinite
from types import UnionType
from typing import Annotated, Any, Literal, Self, Union, get_args, get_origin, get_type_hints

from ..standard import normalize_period, normalize_timestamp
from ..store import encode

Identifier = Annotated[str, "identifier"]
EntityId = Annotated[str, "entity_id"]
ReasonCode = Annotated[str, "reason_code"]
Timestamp = Annotated[str, "timestamp"]
Period = Annotated[str, "period"]
Score = Annotated[float, "score"]
Fraction = Annotated[float, "fraction"]
Positive = Annotated[float, "positive"]
Count = Annotated[int, "count"]


def _constraint(value: Any, name: str) -> Any:
    """检查语义标记并规范时间和期间，失败仅给固定原因码。

    输入数值和 score 标记，返回 0—100 分数；例如 (80, 'score') → 80。
    """
    patterns = {
        "identifier": r"[A-Za-z0-9][A-Za-z0-9_.@/:-]*",
        "entity_id": r"[A-Za-z0-9_.-]+:[a-z_]+:[^:\s]+:[^:\s]+",
        "reason_code": r"[a-z][a-z0-9_]*(?::[A-Za-z0-9_]+)?",
    }
    if name in patterns and not re.fullmatch(patterns[name], value):
        raise ValueError("invalid_" + name)
    if name == "timestamp":
        return normalize_timestamp(value)
    if name == "period":
        return normalize_period(value)
    if (
        (name == "score" and not 0 <= value <= 100)
        or (name == "fraction" and not 0 <= value <= 1)
        or (name == "positive" and value <= 0)
        or (name == "count" and value < 0)
    ):
        raise ValueError("invalid_" + name)
    return value


def _typed(value: Any, annotation: Any) -> Any:
    """递归校验字段类型，拒绝 bool 冒充数值及任意未声明对象。

    输入 JSON 数组及 tuple[str, ...]，输出不可变元组；例如 ['a'] → ('a',)。
    """
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _constraint(_typed(value, arguments[0]), arguments[1])
    if origin in {UnionType, Union}:
        for candidate in arguments:
            try:
                return _typed(value, candidate)
            except (ValueError, TypeError):
                continue
        raise ValueError("invalid_optional_value")
    if origin is Literal:
        if value not in arguments:
            raise ValueError("invalid_literal")
        return value
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise ValueError("invalid_sequence")
        return tuple(_typed(child, arguments[0]) for child in value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError("invalid_mapping")
        return {
            _typed(key, arguments[0]): _typed(child, arguments[1]) for key, child in value.items()
        }
    if isinstance(annotation, type) and issubclass(annotation, Model):
        return annotation.from_dict(value.to_dict() if isinstance(value, annotation) else value)
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        try:
            return annotation(value)
        except (ValueError, TypeError):
            raise ValueError("invalid_enum") from None
    if annotation is float:
        if type(value) not in {int, float} or not isfinite(value):
            raise ValueError("invalid_finite_number")
        return float(value)
    if type(value) is not annotation:
        raise ValueError("invalid_field_type")
    return value


@dataclass(frozen=True, kw_only=True)
class Model:
    """所有产品数据类共用的严格序列化边界，字段不允许隐式扩展。"""

    def __post_init__(self) -> None:
        """验证并规范每个声明字段；输入带时区时间，输出 UTC 字符串字段。

        例如 EvidenceRef(observed_at='2026-01-01T08:00:00+08:00', ...) 保存 00:00Z。
        """
        hints = get_type_hints(type(self), include_extras=True)
        for field in fields(self):
            object.__setattr__(
                self, field.name, _typed(getattr(self, field.name), hints[field.name])
            )
        encode(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        """返回不共享可变子对象的 JSON 字典；模型 → 包含版本和字段的字典。

        例如 ConfidenceComponents(quality=80).to_dict()['quality'] → 80.0。
        """
        result: dict[str, Any] = json.loads(self.to_json())
        return result

    def to_json(self) -> str:
        """输出排序、有限数值且无凭据字段的 JSON；同一模型反复调用输出一致。

        例如包含 quality=80 的模型 → JSON 中的数值 80.0，而不是 NaN 文本。
        """
        return encode(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Self:
        """严格反序列化嵌套结构并重新校验，额外字段或缺必填项均拒绝。

        例如 ModelSubclass.from_dict(instance.to_dict()) → 与 instance 相等的对象。
        """
        if not isinstance(value, dict):
            raise ValueError("model_mapping_required")
        try:
            return cls(**value)
        except TypeError:
            raise ValueError("model_fields_invalid") from None

    @classmethod
    def from_json(cls, value: str) -> Self:
        """读取 JSON 后走同一契约边界，解析错误不泄露输入。

        例如 ModelSubclass.from_json(instance.to_json()) → 完整恢复的模型。
        """
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            raise ValueError("invalid_model_json") from None
        return cls.from_dict(decoded)
