"""可选编译的纯数值累计内核，与未加速逐笔统计保持相同语义。"""

import importlib
import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np


def accumulate_policy(
    candidates: np.ndarray,
    offsets: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    returns: np.ndarray,
    year_indices: np.ndarray,
    trial_offset: int,
    sums: np.ndarray,
    entry_days: np.ndarray,
    annual_sum: np.ndarray,
    annual_count: np.ndarray,
) -> None:
    """原地累计一个退出参数下全部策略，避免每条小数组重复调用NumPy归约。

    candidates按策略分段，offsets指示边界；每段独立处理重叠持仓、三日挂单期限。
    示例某段只有一笔0日确认、1日买入、2日退出且收益.03，closed加1、wins加1。
    输入均为公开数值，不访问行情缓存或网络；同一输入的Python与JIT输出一致。
    """
    count = len(entries)
    for strategy_index in range(len(offsets) - 1):
        trial_index = trial_offset + strategy_index
        occupied_until = -1
        for position in range(offsets[strategy_index], offsets[strategy_index + 1]):
            signal_index = candidates[position]
            sums[0, trial_index] += 1
            if signal_index < occupied_until:
                sums[6, trial_index] += 1
                continue
            entry = entries[signal_index]
            if entry < 0:
                occupied_until = min(signal_index + 3, count)
                if signal_index + 3 < count:
                    sums[4, trial_index] += 1
                else:
                    sums[5, trial_index] += 1
                continue
            occupied_until = exits[signal_index]
            entry_days[trial_index, entry] = True
            value = returns[signal_index]
            if not math.isfinite(value):
                sums[3, trial_index] += 1
                continue
            sums[1, trial_index] += 1
            sums[7, trial_index] += value
            if value > 0:
                sums[2, trial_index] += 1
                sums[8, trial_index] += value
            elif value < 0:
                sums[9, trial_index] += value
            sums[10, trial_index] += exits[signal_index] - entry
            annual_sum[trial_index, year_indices[signal_index]] += value
            annual_count[trial_index, year_indices[signal_index]] += 1


def optional_compiled_accumulator() -> Callable[..., Any]:
    """显式请求时加载隔离环境中的Numba；输入无，输出编译内核或清晰缺依赖错误。

    用户可用 uv run --with numba 执行搜索；不自动修改项目依赖或安装全局包。
    """
    try:
        module = importlib.import_module("numba")
    except ImportError:
        raise ValueError("acceleration_requires_uv_run_with_numba") from None
    return cast(Callable[..., Any], module.njit(cache=True)(accumulate_policy))
