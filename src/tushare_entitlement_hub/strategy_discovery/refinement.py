"""有限结构退出与防御策略退出网格，保留各实验约定的入场和资金政策。"""

from dataclasses import replace

import numpy as np

from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition


def structural_variants(
    definition: StrategyDefinition, features: dict[str, FloatArray]
) -> list[StrategyDefinition]:
    """为BOLL收回或RSI修复锚生成五种明确命名的结构退出变体。

    例family='boll_reentry'生成__struct_signal_off、signal_medium、signal_high、
    early3、early5；中/高档分别在BOLL位置>=.65/.80退出。RSI修复对应>=65/70，
    不再沿用MA20的OR条件。early3/5保留原退出矩阵，设置原执行器的未回本退出年龄。
    每个入场矩阵固定为原entry_mask & ~原exit_mask；原退出为空时复制原入场。
    更改exit_mask仍会改变待成交订单的撤单时点，资金占用也可能改变实际成交集合，
    因而不能把结果解释成固定逐笔样本上的纯退出收益。replace保留其余执行设置，
    包括新增的盘中执行开关；止盈、止损、持有上限、排名、追价和仓位都不改变。
    其他策略族、缺字段或维度不符抛ValueError；未知/无效当日报价不产生新退出信号。
    """
    if definition.family not in {"boll_reentry", "rsi_recovery"}:
        raise ValueError("structural_variants_unsupported_family")
    indicator_name = "boll_position" if definition.family == "boll_reentry" else "rsi6"
    required = ("valid", "close", indicator_name)
    if any(
        name not in features or features[name].shape != definition.entry_mask.shape
        for name in required
    ):
        raise ValueError("structural_variants_features_missing_or_misaligned")
    indicator = features[indicator_name]
    observed = (
        (features["valid"] == 1)
        & np.isfinite(features["close"])
        & (features["close"] > 0)
        & np.isfinite(indicator)
    )
    medium, high = (0.65, 0.80) if definition.family == "boll_reentry" else (65.0, 70.0)
    entry_mask = definition.entry_mask.copy()
    if definition.exit_mask is not None:
        entry_mask &= ~definition.exit_mask
    variants = []
    for label, exit_mask, early_exit in (
        ("signal_off", None, definition.policy.early_exit),
        ("signal_medium", observed & (indicator >= medium), definition.policy.early_exit),
        ("signal_high", observed & (indicator >= high), definition.policy.early_exit),
        ("early3", definition.exit_mask, 3),
        ("early5", definition.exit_mask, 5),
    ):
        name = f"{definition.name}__struct_{label}"
        variants.append(
            replace(
                definition,
                name=name,
                entry_mask=entry_mask.copy(),
                exit_mask=exit_mask,
                policy=replace(definition.policy, name=name, early_exit=early_exit),
            )
        )
    return variants


def defensive_exit_variants(definition: StrategyDefinition) -> list[StrategyDefinition]:
    """为防御轮动生成明确限定的十二组退出参数，不修改入场与执行方式。

    网格来自止盈2%/4%/6% × 止损8%/12% × 最大持有20/40日，共3×2×2=12组；
    用于检验原95%止盈近乎关闭、60日到期占多数的防御策略，不扩大到其他入场族。
    初轮343笔中63笔曾收盘回撤8%、其中58笔最终亏损，因此以8%为有证据的探索档；
    目标首次触及中位数为10/16/19日，保留40日慢兑现，主规格为4%/8%/40日。
    例名称base生成base__defexit_tp0.02_sl0.08_h20等十二个唯一名称，policy名称同步。
    只改take_profit、stop_loss、maximum_holding及名称；入场、分数、退出矩阵保留
    原对象，early_exit、跟踪退出、盘中执行、追价和仓位等其余字段保持原值。
    family不是defensive_rotation时抛ValueError；返回新定义，不修改输入定义。
    """
    if definition.family != "defensive_rotation":
        raise ValueError("defensive_exit_variants_unsupported_family")
    variants = []
    for take_profit in (0.02, 0.04, 0.06):
        for stop_loss in (0.08, 0.12):
            for maximum_holding in (20, 40):
                name = (
                    f"{definition.name}__defexit_tp{take_profit:g}"
                    f"_sl{stop_loss:g}_h{maximum_holding}"
                )
                variants.append(
                    replace(
                        definition,
                        name=name,
                        policy=replace(
                            definition.policy,
                            name=name,
                            take_profit=take_profit,
                            stop_loss=stop_loss,
                            maximum_holding=maximum_holding,
                        ),
                    )
                )
    return variants
