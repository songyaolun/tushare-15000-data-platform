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


def evaluate_filters(
    point: IndicatorPoint,
    injected_features: Mapping[str, float | None] | None = None,
    definitions: tuple[FilterDefinition, ...] = DEFAULT_FILTERS,
) -> dict[str, TruthValue]:
    """对每条规则独立读取特征，输出名字到TRUE/FALSE/UNKNOWN的映射。

    输入point和可选基本面/板块等数值特征；缺失或非有限特征输出UNKNOWN。
    例：dif=1输出dif_positive=TRUE；无fundamentals输出fundamentals=UNKNOWN。
    注入值不能覆盖技术特征，也不能读取其他条件布尔结果；0和正数分别为失败/通过。
    """
    technical = {
        "close": point.close, "ma20": point.ma20, "ma60": point.ma60,
        "previous_ma20": point.previous_ma20, "previous_ma60": point.previous_ma60,
        "dif": point.dif, "dea": point.dea, "histogram": point.histogram,
        "normalized_histogram": point.normalized_histogram, "atr14": point.atr14,
    }
    features = dict(injected_features or {})
    if set(features).intersection(technical):
        raise ValueError("注入特征不能覆盖技术指标")
    if len({definition.name for definition in definitions}) != len(definitions):
        raise ValueError("条件名称必须唯一")
    condition_names = {definition.name for definition in definitions}
    if any(set(definition.feature_names).intersection(condition_names - {definition.name})
           for definition in definitions):
        raise ValueError("规则不能读取其他规则布尔结果")
    if set(features).intersection(definition.name for definition in definitions):
        # Basic external conditions intentionally share their feature and rule name.
        reserved = {definition.name for definition in definitions
                    if definition.name not in definition.feature_names}
        if set(features).intersection(reserved):
            raise ValueError("条件布尔结果不能作为注入特征")
    features.update(technical)
    results = {}
    for definition in definitions:
        values = tuple(features.get(name) for name in definition.feature_names)
        if any(value is None or isinstance(value, (bool, TruthValue))
               or not isfinite(value) for value in values):
            results[definition.name] = TruthValue.UNKNOWN
        else:
            numeric = tuple(float(value) for value in values if value is not None)
            results[definition.name] = (
                TruthValue.TRUE if definition.predicate(numeric) else TruthValue.FALSE
            )
    return results
