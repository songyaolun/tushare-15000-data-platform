"""每项条件只读取原始特征，缺失证据保持三态未知。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from .indicators import IndicatorPoint


class TruthValue(StrEnum):
    """条件结果；例：缺失财务特征为UNKNOWN，不等同TRUE。"""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FilterDefinition:
    """具名条件及其原始特征依赖；predicate仅取得声明的数值元组。

    例：FilterDefinition('positive', ('dif',), positive)独立检查DIF，不读取MA结果。
    """

    name: str
    feature_names: tuple[str, ...]
    predicate: Callable[[tuple[float, ...]], bool]


def _descending(values: tuple[float, ...]) -> bool:
    """逐项严格递减；输入(3,2,1)输出True，(3,3,1)输出False。"""
    return all(left > right for left, right in zip(values, values[1:], strict=False))


def _positive(values: tuple[float, ...]) -> bool:
    """检查单个原始特征严格为正；输入(1,)输出True，(0,)输出False。"""
    return values[0] > 0


DEFAULT_FILTERS = (
    FilterDefinition("close_above_ma20", ("close", "ma20"), _descending),
    FilterDefinition("ma20_above_ma60", ("ma20", "ma60"), _descending),
    FilterDefinition("close_above_ma20_above_ma60", ("close", "ma20", "ma60"), _descending),
    FilterDefinition("ma20_rising", ("ma20", "previous_ma20"), _descending),
    FilterDefinition("ma60_rising", ("ma60", "previous_ma60"), _descending),
    FilterDefinition("dif_positive", ("dif",), _positive),
    FilterDefinition("histogram_positive", ("histogram",), _positive),
    FilterDefinition("fundamentals", ("fundamentals",), _positive),
    FilterDefinition("sector", ("sector",), _positive),
)
TECHNICAL_FILTER_NAMES = tuple(
    definition.name for definition in DEFAULT_FILTERS[:7]
    if definition.name != "close_above_ma20_above_ma60"
)


TECHNICAL_FEATURE_NAMES = frozenset({
    "close", "ma20", "ma60", "previous_ma20", "previous_ma60", "dif", "dea",
    "histogram", "normalized_histogram", "atr14",
})


def validate_filter_definitions(definitions: tuple[FilterDefinition, ...]) -> frozenset[str]:
    """一次校验规则独立性并返回禁止注入的规则结果名；默认规则输出技术条件名集合。

    输入重名规则或读取另一规则名的依赖时抛ValueError；允许基本面规则同名原始特征。
    """
    names = {definition.name for definition in definitions}
    if len(names) != len(definitions):
        raise ValueError("条件名称必须唯一")
    if any(set(definition.feature_names).intersection(names - {definition.name})
           for definition in definitions):
        raise ValueError("规则不能读取其他规则布尔结果")
    return frozenset(definition.name for definition in definitions
                     if definition.name not in definition.feature_names)


def _evaluate_validated_filters(
    point: IndicatorPoint, injected_features: Mapping[str, float | None] | None,
    definitions: tuple[FilterDefinition, ...], reserved: frozenset[str],
) -> dict[str, TruthValue]:
    """按预校验规则独立求一个日期的原始特征；dif=1输出dif_positive=TRUE。

    无注入时省略特征字典和集合分配；未知、bool和非有限值保持UNKNOWN。
    """
    if injected_features:
        if TECHNICAL_FEATURE_NAMES.intersection(injected_features):
            raise ValueError("注入特征不能覆盖技术指标")
        if reserved.intersection(injected_features):
            raise ValueError("条件布尔结果不能作为注入特征")
    results = {}
    for definition in definitions:
        values = []
        valid = True
        for name in definition.feature_names:
            value = getattr(point, name) if name in TECHNICAL_FEATURE_NAMES \
                else (injected_features or {}).get(name)
            if value is None or isinstance(value, (bool, TruthValue)) or not isfinite(value):
                valid = False
                break
            values.append(float(value))
        results[definition.name] = (
            TruthValue.TRUE if definition.predicate(tuple(values)) else TruthValue.FALSE
        ) if valid else TruthValue.UNKNOWN
    return results


def evaluate_filter_series(
    points: tuple[IndicatorPoint, ...],
    features_by_date: Mapping[str, Mapping[str, float | None]] | None = None,
    definitions: tuple[FilterDefinition, ...] = DEFAULT_FILTERS,
) -> dict[str, dict[str, TruthValue]]:
    """整段行情只校验规则一次，每日只求值一次，返回可复用原始条件矩阵。

    输入两日指标输出两个日期的独立三态结果；缓存不包含任何实验通过/失败布尔。
    """
    reserved = validate_filter_definitions(definitions)
    return {point.trade_date: _evaluate_validated_filters(
        point, (features_by_date or {}).get(point.trade_date), definitions, reserved,
    ) for point in points}


def evaluate_filters(
    point: IndicatorPoint,
    injected_features: Mapping[str, float | None] | None = None,
    definitions: tuple[FilterDefinition, ...] = DEFAULT_FILTERS,
) -> dict[str, TruthValue]:
    """独立校验并求一个日期的三态条件；dif=1为TRUE，缺失fundamentals为UNKNOWN。

    注入不得覆盖技术原始值或读取其他规则结果，整段重复使用请调用evaluate_filter_series。
    """
    return _evaluate_validated_filters(
        point, injected_features, definitions, validate_filter_definitions(definitions),
    )
