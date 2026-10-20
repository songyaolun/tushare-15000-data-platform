"""行业初期胜者的八个收盘利润保护对照，不改变入场或行业退出规则。"""

from dataclasses import replace

from ..regime_research.panel import Panel
from ..strategy_optimization.features import FloatArray
from .contracts import StrategyDefinition
from .onset_management import build_variants as build_management_variants
from .onset_quality import REQUIRED_FEATURES as REQUIRED_FEATURES
from .onset_quality import build_strategies as build_quality_strategies
from .profit_protection import ProfitProtection

QUALITY_BASELINE_NAME = "onset_quality_market_breadth_50"
BASELINE_NAME = QUALITY_BASELINE_NAME + "__manage_momentum20_only"


def build_variants(definition: StrategyDefinition) -> list[StrategyDefinition]:
    """给尚未启用利润保护的行业初期定义增加八组预声明收盘保护，不生成控制组。

    例base__protect_floor_arm3_lock1在已知复权收盘峰值收益达到3%后，收盘收益
    不高于1%时触发；base__protect_trail_arm5_dd2在峰值涨幅达到5%后，从该峰值
    收盘回撤2%触发。四个固定地板组合为(激活3%,地板0%/1%)及(激活5%,地板0%/2%)；
    四个跟踪组合为激活3%/5% × 回撤2%/3%。触发后等待下一合格开盘，不保证保本，
    不推断盘中高低价。检测和成交由共用账本负责，本函数只声明配置。
    原入场、排名、行业退出与入场否决数组保持原对象；policy仅同步名称，其余字段
    包括TP/SL/持有期、仓位、追价和既有执行设置均保留。默认锚仍为TP8%/SL8%/H30。
    family不是sector_breadth_onset或已有profit_protection时抛ValueError。
    """
    if definition.family != "sector_breadth_onset" or definition.profit_protection is not None:
        raise ValueError("onset_protection_requires_unprotected_onset")
    parameters = (
        ("floor_arm3_lock0", ProfitProtection(0.03, minimum_return=0.0)),
        ("floor_arm3_lock1", ProfitProtection(0.03, minimum_return=0.01)),
        ("floor_arm5_lock0", ProfitProtection(0.05, minimum_return=0.0)),
        ("floor_arm5_lock2", ProfitProtection(0.05, minimum_return=0.02)),
        ("trail_arm3_dd2", ProfitProtection(0.03, trailing_drawdown=0.02)),
        ("trail_arm3_dd3", ProfitProtection(0.03, trailing_drawdown=0.03)),
        ("trail_arm5_dd2", ProfitProtection(0.05, trailing_drawdown=0.02)),
        ("trail_arm5_dd3", ProfitProtection(0.05, trailing_drawdown=0.03)),
    )
    variants = []
    for label, protection in parameters:
        name = f"{definition.name}__protect_{label}"
        variants.append(
            replace(
                definition,
                name=name,
                policy=replace(definition.policy, name=name),
                profit_protection=protection,
            )
        )
    return variants


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """从既有质量和行业退出builder构造固定胜者，再生成八组利润保护对照。

    例先取onset_quality_market_breadth_50，再取其__manage_momentum20_only，输出
    该固定名称追加__protect_*的八项；不按新数据重新选优，也不包含重复控制组。
    所需特征沿用onset_quality；当日市场breadth缺失时原质量过滤不产生候选。
    """
    quality = next(
        definition
        for definition in build_quality_strategies(panel, features)
        if definition.name == QUALITY_BASELINE_NAME
    )
    baseline = next(
        definition
        for definition in build_management_variants(quality, features)
        if definition.name == BASELINE_NAME
    )
    return build_variants(baseline)
