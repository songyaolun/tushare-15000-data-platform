"""不可变数据契约及严格、可逆的 JSON 边界，不依赖在线服务。"""

from __future__ import annotations

import json
import re
from collections.abc import Hashable
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from functools import cache
from math import isfinite
from types import UnionType
from typing import Annotated, Any, Literal, Self, Union, cast, get_args, get_origin, get_type_hints

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

_IMMUTABLE_SEAL = "_model_immutable_validation"
_VALIDATED = object()


@cache
def _model_fields(model: type[Model]) -> tuple[str, ...]:
    """缓存声明字段名，不包含实例缓存或验证凭据；例：Coverage得到四个契约字段名。"""
    return tuple(field.name for field in fields(model))


def _current_immutable(value: Any) -> bool:
    """检查已验证不可变状态仍与凭据一致，递归检查子模型，拒绝任何可变容器。

    例：已验证EvidenceRef返回True；其字段或嵌套子模型被绕过frozen强改后返回False。
    凭据保存在实例上而非全局对象缓存，且不属于dataclass字段，不进入JSON或持久化。
    """
    if type(value) in {str, int, bool, type(None)} or isinstance(value, StrEnum):
        return True
    if type(value) is float:
        return isfinite(value)
    if type(value) is tuple:
        return all(_current_immutable(child) for child in value)
    if not isinstance(value, Model):
        return False
    names = _model_fields(cast(Hashable, type(value)))
    state = value.__dict__
    seal = state.get(_IMMUTABLE_SEAL)
    if (
        type(seal) is not tuple
        or len(seal) != 3
        or seal[0] is not _VALIDATED
        or seal[1] is not type(value)
        or type(seal[2]) is not tuple
        or len(seal[2]) != len(names)
        or len(state) != len(names) + 1
    ):
        return False
    return all(
        name in state and state[name] is previous and _current_immutable(previous)
        for name, previous in zip(names, seal[2], strict=True)
    )


def _certify_immutable(value: Any, validated: Any) -> bool:
    """仅在完整构造校验已成功后，证明原值与规范值类型和值相同且递归不可变。

    例：冻结EvidenceRef可登记供后续复用；含dict的DecisionEnvelope始终返回False。
    int与float、正负零、非dataclass子类或带额外实例状态的对象不视作等价可复用值。
    验证失败的构造不会走到本函数，因此子类的领域约束不会被基类提前认证。
    """
    if type(value) is not type(validated):
        return False
    if type(value) in {str, int, bool, type(None)} or isinstance(value, StrEnum):
        return bool(value == validated)
    if type(value) is float:
        return isfinite(value) and value.hex() == validated.hex()
    if type(value) is tuple:
        return len(value) == len(validated) and all(
            _certify_immutable(left, right) for left, right in zip(value, validated, strict=True)
        )
    if not isinstance(value, Model):
        return False
    parameters = type(value).__dict__.get("__dataclass_params__")
    names = _model_fields(cast(Hashable, type(value)))
    if (
        parameters is None
        or not parameters.frozen
        or _IMMUTABLE_SEAL in names
        or value.__dict__.keys() - set(names) - {_IMMUTABLE_SEAL}
        or validated.__dict__.keys() - set(names) - {_IMMUTABLE_SEAL}
        or not all(
            _certify_immutable(getattr(value, name), getattr(validated, name)) for name in names
        )
    ):
        return False
    object.__setattr__(
        value,
        _IMMUTABLE_SEAL,
        (_VALIDATED, type(value), tuple(getattr(value, name) for name in names)),
    )
    return True


@cache
def _model_hints(model: type[Model]) -> dict[str, Any]:
    """缓存固定模型类的类型解析，不缓存输入值或跳过实例校验。

    输入 EvidenceRef 类型，输出带 Annotated 的字段定义；重复输入只解析一次。
    """
    return get_type_hints(model, include_extras=True)


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
        if isinstance(value, Model):
            if type(value) is not annotation:
                raise ValueError("invalid_model_type")
            if _current_immutable(value):
                return value
            # 只投影当前字段，容器仍由递归类型校验复制；避免asdict先破坏子模型身份。
            converted = annotation.from_dict(
                {name: getattr(value, name) for name in _model_fields(cast(Hashable, annotation))}
            )
            if _certify_immutable(value, converted):
                return value
            return converted
        converted = annotation.from_dict(value)
        _certify_immutable(converted, converted)
        return converted
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
        hints = _model_hints(cast(Hashable, type(self)))
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
