"""研究实验的显式证券范围，不改变全市场广度及历史特征。"""

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def mainboard_scope_mask(codes: Sequence[str]) -> NDArray[np.bool_]:
    """按产品scope-v1-c0-s0返回证券轴掩码，在行业限额和Top10之前应用。

    输入['600000.SH', '300001.SZ']，输出[True, False]；未知格式、北交所、
    B股、创业板及科创板均不纳入。只限制新入榜股票，不重算全市场指标。
    """
    pattern = r"(?:600|601|603|605)\d{3}\.SH|(?:000|001|002|003)\d{3}\.SZ"
    return np.asarray([re.fullmatch(pattern, code) is not None for code in codes], dtype=bool)


def account_scope_mask(codes: Sequence[str], scope: str) -> NDArray[np.bool_]:
    """Return the extra account filter; original preserves the frozen candidate pool.

    Example: codes=['600000.SH', '300001.SZ'], scope='original' returns [True, True];
    scope='mainboard' returns [True, False]. True does not grant original eligibility:
    the caller must intersect this mask with the unchanged historical entry mask.
    Unknown scope names raise ValueError instead of silently expanding the universe.
    """
    if scope == "original":
        return np.ones(len(codes), dtype=bool)
    if scope == "mainboard":
        return mainboard_scope_mask(codes)
    raise ValueError("unknown_account_scope")
