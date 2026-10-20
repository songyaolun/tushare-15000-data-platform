"""累计加速必须保留逐笔重叠、到期、截尾及年度统计语义。"""

import types

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_optimization import acceleration


def test_accumulator_matches_explicit_trade_states():
    """场景：盈利、亏损、零收益、到期、截尾、未退出和重叠；预期：各状态独立准确累计。"""
    candidates = np.array([0, 1, 2, 4, 7, 8, 9, 10, 11, 12, 13])
    entries = np.array([1, 2, 3, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1])
    exits = np.array([2, 3, 4, 14, 14, 14, 14, 9, 10, 11, 14, 14, 14, 14])
    returns = np.array(
        [
            0.03,
            0.01,
            -0.02,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
            0.02,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    sums = np.zeros((11, 3))
    days = np.zeros((3, 14), dtype=bool)
    annual_sum = np.zeros((3, 1))
    annual_count = np.zeros((3, 1))
    acceleration.accumulate_policy(
        candidates,
        np.array([0, 8, 11]),
        entries,
        exits,
        returns,
        np.zeros(14, dtype=int),
        1,
        sums,
        days,
        annual_sum,
        annual_count,
    )
    np.testing.assert_allclose(sums[:, 1], [8, 3, 1, 1, 1, 0, 3, 0.01, 0.03, -0.02, 3])
    assert sums[5, 2] == 1
    assert annual_count[1, 0] == 3
    assert annual_sum[1, 0] == pytest.approx(0.01)
    assert set(np.flatnonzero(days[1])) == {1, 3, 8, 10}


def test_optional_compiler_is_explicit_and_missing_dependency_is_clear(monkeypatch):
    """场景：环境有/无可选编译器；预期：明确加载缓存编译器或抛可执行安装提示。"""
    observed = []

    def decorator(**kwargs):
        """输入缓存选项，返回记录函数的模拟编译器；示例cache=True被完整保留。"""
        observed.append(kwargs)
        return lambda function: function

    monkeypatch.setattr(
        acceleration.importlib, "import_module", lambda name: types.SimpleNamespace(njit=decorator)
    )
    assert acceleration.optional_compiled_accumulator() is acceleration.accumulate_policy
    assert observed == [{"cache": True}]

    def missing(name):
        """模拟环境缺依赖；输入numba，输出ImportError供上层转安全错误。"""
        raise ImportError(name)

    monkeypatch.setattr(acceleration.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="acceleration_requires"):
        acceleration.optional_compiled_accumulator()
