"""独立策略登记和有限参数邻域；不预设原放量反转为必要条件。"""

from dataclasses import replace

import numpy as np

from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition

_MANAGEMENT_LABELS = frozenset(
    {
        "breadth_only",
        "momentum20_only",
        "bad_confirm2",
        "bad_confirm3",
        "momentum20_deep",
        "momentum10",
        "bad_below_ma20",
        "stop4",
        "stop6",
        "hold20",
        "hold40",
        "industry_off",
    }
)
_PROTECTION_LABELS = frozenset(
    {
        "floor_arm3_lock0",
        "floor_arm3_lock1",
        "floor_arm5_lock0",
        "floor_arm5_lock2",
        "trail_arm3_dd2",
        "trail_arm3_dd3",
        "trail_arm5_dd2",
        "trail_arm5_dd3",
    }
)


def build_candidates(
    panel: Panel, features: dict[str, FloatArray], group: str
) -> list[StrategyDefinition]:
    """按研究组创建具名策略，拒绝重名或未知组；例如group='trend'返回两类趋势策略。

    延迟导入独立模块，避免准备数据时生成大矩阵。baseline由原执行路径单独运行。
    all缺少sector_level2_index时只跳过hierarchy_signals；显式请求该组仍校验缺字段。
    """
    from . import (
        breakout_retest,
        defensive,
        hierarchy_signals,
        onset_management,
        onset_protection,
        onset_quality,
        onset_structure,
        relay_signals,
        reversion,
        sector,
        sector_onset,
        selection_timing,
        trend,
    )

    builders = {
        "trend": trend.build_strategies,
        "sector": sector.build_strategies,
        "reversion": reversion.build_strategies,
        "defensive": defensive.build_strategies,
        "selection_timing": selection_timing.build_strategies,
        "sector_onset": sector_onset.build_strategies,
        "onset_quality": onset_quality.build_strategies,
        "onset_management": onset_management.build_strategies,
        "onset_structure": onset_structure.build_strategies,
        "onset_protection": onset_protection.build_strategies,
        "hierarchy_signals": hierarchy_signals.build_strategies,
        "relay_signals": relay_signals.build_strategies,
        "breakout_retest": breakout_retest.build_strategies,
        "control": control_candidates,
    }
    if group not in {*builders, "all"}:
        raise ValueError("unknown_discovery_group")
    definitions = [
        definition
        for name, builder in builders.items()
        if group in (name, "all")
        and not (
            group == "all" and name == "hierarchy_signals" and "sector_level2_index" not in features
        )
        for definition in builder(panel, features)
    ]
    if len({definition.name for definition in definitions}) != len(definitions):
        raise ValueError("duplicate_strategy_name")
    return definitions


def control_candidates(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """把原放量反转及原排序映射为可切换执行方式的对照定义，不更改信号。

    例如control_volume默认使用收盘退出，追加盘中模型时只改变执行方式。
    """
    values = panel.values
    scores = (
        np.nan_to_num(values["sector_heat"], nan=0)
        + np.nan_to_num(values["sector_breadth"], nan=0)
        - np.nan_to_num(values["kdj_j"], nan=100) / 200
    )
    return [
        StrategyDefinition(
            name="control_volume",
            family="baseline_volume",
            entry_mask=np.asarray(panel.signals["volume_reversal__sector_group"], dtype=bool),
            scores=scores,
            policy=PortfolioPolicy(
                name="control_volume",
                switching=False,
                stage_allocation=False,
                force_rebalance=False,
            ),
        )
    ]


def resolve_definition(
    definitions: list[StrategyDefinition], name: str, features: dict[str, FloatArray]
) -> StrategyDefinition:
    """由保存的唯一名称恢复初始、邻域、结构、管理交叉或盘中变体，未知名称拒绝。

    例某初始名称追加__intraday，返回其intraday_exits=True的独立定义。
    onset_quality_market_breadth_50__manage_momentum20_only把该质量信号接入
    已声明的动量退出；仅完整匹配管理前缀和合法标签时才构建十二组管理矩阵。
    onset_structure_control__protect_floor_arm3_lock1按需叠加3%激活、1%收盘保护阈值，
    普通名称不构建保护变体，未知保护标签仍拒绝。
    只允许程序声明过的变体，不执行配置中的任意代码。
    """
    from .refinement import defensive_exit_variants, onset_exit_variants, structural_variants

    intraday = name.endswith("__intraday")
    target = name.removesuffix("__intraday") if intraday else name
    for definition in definitions:
        if target != definition.name and not target.startswith(definition.name + "__"):
            continue
        variants = [definition, *neighborhood(definition)]
        if definition.family in {"boll_reentry", "rsi_recovery"}:
            variants.extend(structural_variants(definition, features))
        if definition.family == "defensive_rotation":
            variants.extend(defensive_exit_variants(definition))
        if definition.family == "sector_breadth_onset":
            variants.extend(onset_exit_variants(definition))
            management_prefix = definition.name + "__manage_"
            if (
                target.startswith(management_prefix)
                and target[len(management_prefix) :] in _MANAGEMENT_LABELS
            ):
                from .onset_management import build_variants as build_management_variants

                variants.extend(build_management_variants(definition, features))
            protection_prefix = definition.name + "__protect_"
            if (
                target.startswith(protection_prefix)
                and target[len(protection_prefix) :] in _PROTECTION_LABELS
            ):
                from .onset_protection import build_variants as build_protection_variants

                variants.extend(build_protection_variants(definition))
        for variant in variants:
            if variant.name == target:
                return (
                    replace(
                        variant,
                        name=name,
                        policy=replace(variant.policy, name=name),
                        intraday_exits=True,
                    )
                    if intraday
                    else variant
                )
    raise ValueError("unknown_selected_strategy")


def neighborhood(definition: StrategyDefinition) -> list[StrategyDefinition]:
    """一次只改一个执行或退出参数，含原点共最多11组，避免宽泛组合爆炸。

    例20日持有生成15/25日邻域；止损和止盈各上下浮动25%，追价与跟踪退出单独检验。
    仅对全样本选出的方向继续探索，不把邻域称作样本外验证。
    """
    policy = definition.policy
    variants = [definition]
    for label, changes in (
        ("stop_tighter", {"stop_loss": policy.stop_loss * 0.75}),
        ("stop_wider", {"stop_loss": min(0.95, policy.stop_loss * 1.25)}),
        ("profit_nearer", {"take_profit": policy.take_profit * 0.75}),
        ("profit_farther", {"take_profit": min(0.95, policy.take_profit * 1.25)}),
        ("hold_shorter", {"maximum_holding": max(1, round(policy.maximum_holding * 0.75))}),
        ("hold_longer", {"maximum_holding": round(policy.maximum_holding * 1.25)}),
    ):
        name = f"{definition.name}__{label}"
        variants.append(
            replace(definition, name=name, policy=replace(policy, name=name, **changes))
        )
    for label, definition_changes in (
        ("entry_unrestricted", {"maximum_entry_gap": None}),
        ("entry_tight", {"maximum_entry_gap": 0.01}),
        ("trail_off", {"trailing_stop": 0.0}),
        ("trail_six", {"trailing_stop": 0.06}),
    ):
        name = f"{definition.name}__{label}"
        variants.append(
            replace(definition, name=name, policy=replace(policy, name=name), **definition_changes)
        )
    unique: list[StrategyDefinition] = []
    signatures = set()
    for variant in variants:
        signature = (
            replace(variant.policy, name=""),
            variant.maximum_entry_gap,
            variant.trailing_stop,
        )
        if signature not in signatures:
            signatures.add(signature)
            unique.append(variant)
    return unique
