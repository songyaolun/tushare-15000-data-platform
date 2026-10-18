"""行业初期追价基准的固定止盈上调对照，不改变信号、止损或行业退出。"""

from dataclasses import replace

from .contracts import StrategyDefinition


def build_variants(base: StrategyDefinition) -> list[StrategyDefinition]:
    """输出8%、12%、16%和关闭止盈四组，固定8%止损和30日持有上限。

    输入onset_structure_gap_2pct定义，输出profit_extension_08/12/16/off四个配置；
    off用显式开关关闭止盈，仍保留原止损、行业退出及到期退出，不用巨大阈值代替。
    仅接受当前冻结基准以避免混入不同入场逻辑，输出共享原信号矩阵，不修改输入。
    """
    if (
        base.name != "onset_structure_gap_2pct"
        or base.policy.stop_loss != 0.08
        or base.policy.take_profit != 0.08
        or base.policy.maximum_holding != 30
        or not base.policy.fixed_price_exits
        or not base.policy.take_profit_enabled
        or base.intraday_exits
        or base.trailing_stop
        or base.profit_protection is not None
    ):
        raise ValueError("profit_extension_requires_frozen_anchor")
    return [
        replace(
            base,
            name=f"profit_extension_{label}",
            policy=replace(
                base.policy,
                name=f"profit_extension_{label}",
                take_profit=threshold,
                take_profit_enabled=enabled,
            ),
        )
        for label, threshold, enabled in (
            ("08", 0.08, True),
            ("12", 0.12, True),
            ("16", 0.16, True),
            ("off", 0.08, False),
        )
    ]
