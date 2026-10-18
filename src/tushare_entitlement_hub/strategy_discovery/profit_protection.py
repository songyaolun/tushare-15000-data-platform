"""以已知复权收盘峰值激活利润保护，不假定保护价能够成交。"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True)
class ProfitProtection:
    """声明激活收益及恰好一种保护阈值，全部收益均相对复权入场参考价。

    例ProfitProtection(.04, minimum_return=.01)在峰值收益达到4%后，
    收盘收益不高于1%时触发；ProfitProtection(.04, trailing_drawdown=.02)
    改为从最高复权收盘回撤2%触发。成交仍等待之后合格开盘，不承诺保本。
    """

    activation_return: float
    minimum_return: float | None = None
    trailing_drawdown: float | None = None

    def __post_init__(self) -> None:
        """拒绝布尔、非有限数及冲突阈值；例如激活0或同时指定两种阈值抛ValueError。"""
        parameters = (self.activation_return, self.minimum_return, self.trailing_drawdown)
        if (
            any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value))
                for value in parameters
            )
            or self.activation_return is None
            or self.activation_return <= 0
            or (self.minimum_return is None) == (self.trailing_drawdown is None)
            or (
                self.minimum_return is not None
                and not 0 <= self.minimum_return < self.activation_return
            )
            or (self.trailing_drawdown is not None and not 0 < self.trailing_drawdown < 1)
        ):
            raise ValueError("invalid_profit_protection")


def profit_protection_triggered(
    entry_adjusted: float,
    peak_adjusted: float,
    adjusted_close: float,
    protection: ProfitProtection,
) -> bool:
    """纯函数判断收盘是否触发保护，未知或非正价格一律不激活。

    输入须是同尺度绝对复权价格，峰值包括入场参考价和当前已知收盘。
    例(10,10.5,10.1,ProfitProtection(.04,minimum_return=.01))返回True；
    峰值10.3则返回False。函数不读高低价、不修改持仓，也不指定成交价格。
    """
    if not all(
        isfinite(value) and value > 0 for value in (entry_adjusted, peak_adjusted, adjusted_close)
    ):
        return False
    known_peak = max(entry_adjusted, peak_adjusted, adjusted_close)
    if known_peak / entry_adjusted < 1 + protection.activation_return:
        return False
    if protection.minimum_return is not None:
        return adjusted_close <= entry_adjusted * (1 + protection.minimum_return)
    assert protection.trailing_drawdown is not None
    return adjusted_close <= known_peak * (1 - protection.trailing_drawdown)
