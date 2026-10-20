"""行业初期策略的十二个失败退出对照，冻结原有效入场候选。"""

from dataclasses import replace
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from ..regime_research.panel import Panel
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition
from .sector import _sector_features
from .sector_onset import build_strategies as build_onset_strategies

REQUIRED_FEATURES = ("valid", "close", "ma20", "return1", "sector_index")
BASELINE_NAME = "sector_onset_selective_quiet_risk_tp8_sl8_h30"


class _PolicyChanges(TypedDict, total=False):
    """只允许退出实验声明的政策键；例{'stop_loss': .04}只修改止损。"""

    stop_loss: float
    maximum_holding: int


def _consecutive_bad(
    bad: NDArray[np.bool_], known: NDArray[np.bool_], membership: FloatArray, sessions: int
) -> NDArray[np.bool_]:
    """只累计同一股票连续属于同一已知行业的已知坏状态，不跨缺失或换行业拼接。

    例bad=[真,真,真]且行业=[0,1,1]、sessions=2，输出[假,假,真]；
    known中间为假时连续计数归零，该未知日既不判坏也不构造连续好状态。
    sessions须为正整数，三个日×证券矩阵同形，否则抛ValueError。
    """
    if (
        type(sessions) is not int
        or sessions < 1
        or bad.ndim != 2
        or known.shape != bad.shape
        or membership.shape != bad.shape
    ):
        raise ValueError("invalid_consecutive_industry_bad_inputs")
    result = np.zeros(bad.shape, dtype=bool)
    consecutive = np.zeros(bad.shape[1], dtype=int)
    previous_membership = np.full(bad.shape[1], np.nan)
    for day in range(len(bad)):
        identifiers = membership[day]
        comparable = (
            known[day]
            & np.isfinite(identifiers)
            & (identifiers >= 0)
            & (identifiers == np.floor(identifiers))
        )
        same_industry = identifiers == previous_membership
        consecutive = np.where(
            comparable & bad[day], np.where(same_industry, consecutive + 1, 1), 0
        )
        result[day] = consecutive >= sessions
        previous_membership = np.where(comparable, identifiers, np.nan)
    return result


def build_variants(
    definition: StrategyDefinition, features: dict[str, FloatArray]
) -> list[StrategyDefinition]:
    """生成十二个预声明行业初期退出对照，原锚为TP8%/SL8%/最大持有30日。

    例base输出base__manage_breadth_only、momentum20_only、bad_confirm2/3、
    momentum20_deep、momentum10、bad_below_ma20、stop4/6、hold20/40、industry_off。
    行业条件依次为仅广度<.4、仅20日动量<=0、原坏信号连续2/3日、
    广度<.4或20日动量<=-2%、广度<.4或10日动量<=0、原坏信号且个股close<MA20。
    最后五组保留原行业退出但SL改4%/6%、持有改20/40日，或单独取消行业退出。
    所有变体固定原entry_invalidation_mask，原值为空时回退至原exit_mask；入场矩阵
    冻结为entry_mask & ~原入场否决。评分、TP、仓位、追价和执行方式保留，新exit_mask
    只影响持仓退出，候选抑制与待成交撤单继续使用原否决矩阵。退出、止损或持有期
    改变仍会改变资金占用和后续可成交集合，不能解释为固定逐笔样本的纯退出收益。
    连续坏必须两项行业指标已知、股票当日可交易、行业ID连续相同，未知或换行业
    中断计数；单日OR仅由已知坏指标触发。保留原退出的组完全遵循原exit_mask语义。
    仅接受有exit_mask的sector_breadth_onset；缺字段/错维度或其他家族抛ValueError。
    """
    if definition.family != "sector_breadth_onset" or definition.exit_mask is None:
        raise ValueError("onset_management_requires_onset_with_industry_exit")
    if any(
        name not in features or features[name].shape != definition.entry_mask.shape
        for name in REQUIRED_FEATURES
    ):
        raise ValueError("onset_management_features_missing_or_misaligned")
    industry = _sector_features(features)
    breadth = industry["breadth"]
    momentum20, momentum10 = industry["momentum20"], industry["momentum10"]
    close, average = features["close"], features["ma20"]
    tradable = (features["valid"] == 1) & np.isfinite(close) & (close > 0)
    breadth_bad = np.isfinite(breadth) & (breadth < 0.4)
    momentum20_bad = np.isfinite(momentum20) & (momentum20 <= 0)
    known = tradable & np.isfinite(breadth) & np.isfinite(momentum20)
    original_exit = definition.exit_mask
    original_invalidation = (
        definition.entry_invalidation_mask
        if definition.entry_invalidation_mask is not None
        else original_exit
    )
    fixed_entry = definition.entry_mask & ~original_invalidation
    variants: tuple[tuple[str, NDArray[np.bool_] | None, _PolicyChanges], ...] = (
        ("breadth_only", tradable & breadth_bad, {}),
        ("momentum20_only", tradable & momentum20_bad, {}),
        ("bad_confirm2", _consecutive_bad(original_exit, known, features["sector_index"], 2), {}),
        ("bad_confirm3", _consecutive_bad(original_exit, known, features["sector_index"], 3), {}),
        (
            "momentum20_deep",
            tradable & (breadth_bad | (np.isfinite(momentum20) & (momentum20 <= -0.02))),
            {},
        ),
        (
            "momentum10",
            tradable & (breadth_bad | (np.isfinite(momentum10) & (momentum10 <= 0))),
            {},
        ),
        (
            "bad_below_ma20",
            original_exit & tradable & np.isfinite(average) & (average > 0) & (close < average),
            {},
        ),
        ("stop4", original_exit, {"stop_loss": 0.04}),
        ("stop6", original_exit, {"stop_loss": 0.06}),
        ("hold20", original_exit, {"maximum_holding": 20}),
        ("hold40", original_exit, {"maximum_holding": 40}),
        ("industry_off", None, {}),
    )
    definitions = []
    for label, exit_mask, changes in variants:
        name = f"{definition.name}__manage_{label}"
        definitions.append(
            replace(
                definition,
                name=name,
                entry_mask=fixed_entry.copy(),
                exit_mask=exit_mask,
                entry_invalidation_mask=original_invalidation,
                policy=replace(definition.policy, name=name, **changes),
            )
        )
    return definitions


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """从既有行业初期生成器选择固定最优锚，返回十二组失败退出研究配置。

    例输入原行业面板与特征，取sector_onset_selective_quiet_risk_tp8_sl8_h30，
    输出该名称追加__manage_*的十二组；不重新选优，不修改源生成器或原信号。
    """
    baseline = next(
        definition
        for definition in build_onset_strategies(panel, features)
        if definition.name == BASELINE_NAME
    )
    return build_variants(baseline, features)
