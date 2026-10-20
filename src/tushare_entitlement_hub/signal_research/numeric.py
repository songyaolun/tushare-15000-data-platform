"""统一浮点舍入比较；容差不是额外经济交易阈值。"""

RELATIVE_TOLERANCE = 1e-12
RATIO_ABSOLUTE_TOLERANCE = 1e-12


def strictly_greater(
    left: float, right: float, *, scale_ref: float = 0.0, absolute_tolerance: float = 0.0,
) -> bool:
    """在统一尺度相对舍入容差外严格大于，近相等按相等FALSE处理。

    输入10与10-1e-14输出False；输入11与10输出True。
    DIF/H对0比较时scale_ref=close保持价格缩放一致；不增加经济门槛。
    调用方须先核验有限原始特征，absolute_tolerance只供无量纲比率舍入边界。
    """
    tolerance = RELATIVE_TOLERANCE * max(abs(left), abs(right), abs(scale_ref))
    return left - right > tolerance + absolute_tolerance


def ratio_greater(left: float, right: float) -> bool:
    """无量纲H/C阈值比较，加入1e-12绝对舍入容差；.002与.002+1e-14视为相等。

    输入.003,.002输出True，输入.002+1e-14,.002输出False。
    事件穿越、跌回和模拟退出均应调用同一比较，策略阈值本身不变。
    """
    return strictly_greater(left, right, absolute_tolerance=RATIO_ABSOLUTE_TOLERANCE)
