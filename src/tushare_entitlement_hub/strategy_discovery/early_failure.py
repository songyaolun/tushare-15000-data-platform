"""对入场初期快速下跌的独立收盘退出规则，不改变原止损。"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class EarlyFailure:
    """限定早期观察天数和亏损阈值；例(3,.03)只在入场前3个市场日生效。"""

    sessions: int = 3
    loss: float = 0.03

    def __post_init__(self) -> None:
        """拒绝非正天数、布尔和非有限阈值；例loss=0抛ValueError。"""
        if (
            type(self.sessions) is not int
            or self.sessions < 1
            or isinstance(self.loss, bool)
            or not isfinite(self.loss)
            or not 0 < self.loss < 1
        ):
            raise ValueError("invalid_early_failure")

    def triggered(self, held_sessions: int, entry_price: float, close: float) -> bool:
        """当有效价格在前sessions日亏损至少loss时为真；例(2,10,9.6)返回真。

        入场日收盘也允许生成卖单，但实际成交仍必须等待T+1；缺价不作判断。
        """
        return bool(
            1 <= held_sessions <= self.sessions
            and isfinite(entry_price)
            and isfinite(close)
            and entry_price > 0
            and close > 0
            and close <= entry_price * (1 - self.loss)
        )
