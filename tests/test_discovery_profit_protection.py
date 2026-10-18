"""利润保护参数和纯收盘阈值的离线边界测试。"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.profit_protection import (
    ProfitProtection,
    profit_protection_triggered,
)


def test_protection_configuration_is_frozen():
    """两种合法保护方式可构造，激活阈值在构造后不可修改。"""
    protection = ProfitProtection(0.03, minimum_return=0)
    assert protection.trailing_drawdown is None
    assert ProfitProtection(0.05, trailing_drawdown=0.02).minimum_return is None
    with pytest.raises(FrozenInstanceError):
        protection.activation_return = 0.05


@pytest.mark.parametrize(
    "changes",
    [
        {"activation_return": None},
        {"activation_return": True},
        {"activation_return": np.bool_(True)},
        {"activation_return": "0.03"},
        {"activation_return": 0},
        {"activation_return": -0.01},
        {"activation_return": np.nan},
        {"activation_return": np.inf},
        {"minimum_return": None},
        {"minimum_return": True},
        {"minimum_return": -0.01},
        {"minimum_return": 0.03},
        {"minimum_return": 0.04},
        {"minimum_return": np.nan},
        {"trailing_drawdown": 0.02},
        {"minimum_return": None, "trailing_drawdown": True},
        {"minimum_return": None, "trailing_drawdown": 0},
        {"minimum_return": None, "trailing_drawdown": 1},
        {"minimum_return": None, "trailing_drawdown": -0.01},
        {"minimum_return": None, "trailing_drawdown": np.inf},
    ],
)
def test_protection_rejects_invalid_configuration(changes):
    """非数字、布尔、非有限值、冲突方式和越界阈值都应在执行前被拒绝。"""
    parameters = {"activation_return": 0.03, "minimum_return": 0} | changes
    with pytest.raises(ValueError, match="invalid_profit_protection"):
        ProfitProtection(**parameters)


@pytest.mark.parametrize(
    "peak,closing,protection,expected",
    [
        (10.3, 10, ProfitProtection(0.03, minimum_return=0), True),
        (10.299, 10, ProfitProtection(0.03, minimum_return=0), False),
        (10.5, 10.1, ProfitProtection(0.03, minimum_return=0.01), True),
        (10.5, 10.101, ProfitProtection(0.03, minimum_return=0.01), False),
        (10.5, 9, ProfitProtection(0.03, minimum_return=0), True),
        (10.5, 10.5 * 0.98, ProfitProtection(0.03, trailing_drawdown=0.02), True),
        (10.5, 10.4, ProfitProtection(0.03, trailing_drawdown=0.02), False),
        (10.5, 11, ProfitProtection(0.03, trailing_drawdown=0.02), False),
        (9, 10, ProfitProtection(0.03, minimum_return=0), False),
    ],
)
def test_protection_uses_known_close_peak_and_inclusive_thresholds(
    peak, closing, protection, expected
):
    """只有已激活且收盘达到保护线才触发；新高、未激活和尚在线上均不触发。"""
    assert profit_protection_triggered(10, peak, closing, protection) is expected


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf])
def test_protection_invalid_prices_fail_closed(position, value):
    """入场价、历史峰值或当前复权收盘任一无效时均不得激活或触发保护。"""
    prices = [10, 10.5, 10]
    prices[position] = value
    assert not profit_protection_triggered(*prices, ProfitProtection(0.03, minimum_return=0))
