"""非线性模型输入白名单与缺失值语义测试，不要求安装训练依赖。"""

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.nonlinear_scoring import (
    NONLINEAR_FEATURE_NAMES,
    nonlinear_input_features,
)


def test_nonlinear_inputs_keep_fixed_order_and_unknown_values():
    """场景：原始字典倒序且有缺失值；预期固定列顺序、保留NaN与二值信号。"""
    raw = {name: np.array([index, np.nan]) for index, name in enumerate(NONLINEAR_FEATURE_NAMES)}
    features, names = nonlinear_input_features(
        dict(reversed(list(raw.items()))), np.array([[0.0], [1.0]]), ("breakout",)
    )
    assert features.shape == (2, 17)
    assert names == NONLINEAR_FEATURE_NAMES + ("signal:breakout",)
    np.testing.assert_array_equal(features[0, :16], np.arange(16))
    assert np.isnan(features[1, :16]).all()
    np.testing.assert_array_equal(features[:, -1], [0, 1])


def test_future_rows_cannot_change_existing_nonlinear_inputs():
    """场景：追加未来候选并任意改变其数值；预期历史行输入完全不变。"""
    raw = {name: np.array([1.0, 9.0]) for name in NONLINEAR_FEATURE_NAMES}
    historical, _ = nonlinear_input_features(
        {name: values[:1] for name, values in raw.items()}, np.array([[1.0]]), ("breakout",)
    )
    extended, _ = nonlinear_input_features(raw, np.array([[1.0], [0.0]]), ("breakout",))
    np.testing.assert_array_equal(extended[:1], historical)


@pytest.mark.parametrize(
    "scenario",
    [
        "extra_date",
        "missing",
        "axis",
        "signal_axis",
        "signal_names",
        "duplicate",
        "empty_name",
        "whitespace",
        "nonbinary",
        "infinity",
    ],
)
def test_invalid_nonlinear_contract_rejects_unknown_inputs(scenario):
    """场景：未来/身份列、错轴、非法信号或无穷值；预期拒绝而非静默参与拟合。"""
    raw = {name: np.array([1.0, 2.0]) for name in NONLINEAR_FEATURE_NAMES}
    hits = np.array([[0.0], [1.0]])
    names = ("breakout",)
    if scenario == "extra_date":
        raw["date"] = np.array([1.0, 2.0])
    if scenario == "missing":
        raw.pop(NONLINEAR_FEATURE_NAMES[0])
    if scenario == "axis":
        raw[NONLINEAR_FEATURE_NAMES[0]] = np.ones(1)
    if scenario == "signal_axis":
        hits = np.ones(2)
    if scenario == "signal_names":
        names = ()
    if scenario == "duplicate":
        names = ("breakout", "breakout")
        hits = np.ones((2, 2))
    if scenario == "empty_name":
        names = ("",)
    if scenario == "whitespace":
        names = (" breakout",)
    if scenario == "nonbinary":
        hits[0, 0] = 0.5
    if scenario == "infinity":
        raw[NONLINEAR_FEATURE_NAMES[0]][0] = np.inf
    with pytest.raises(ValueError):
        nonlinear_input_features(raw, hits, names)
