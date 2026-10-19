"""可移植树推理的软件契约测试，不训练模型或声称金融样本外效果。"""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.boosted_tree_scoring import (
    BoostedTreeModel,
    NumericTreeNode,
    load_boosted_tree_model,
)


def model_payload():
    """构造两树、相反缺失方向的JSON对象；例输入[1,2]输出基准.1加.2减.4=-.1。"""
    return {
        "schema_version": 1,
        "feature_names": ["momentum", "volatility"],
        "baseline_prediction": 0.1,
        "trees": [
            [
                dict(
                    feature_idx=0,
                    numeric_threshold=1.0,
                    missing_go_to_left=True,
                    left=1,
                    right=2,
                    is_leaf=False,
                    value=99.0,
                ),
                dict(
                    feature_idx=0,
                    numeric_threshold=0.0,
                    missing_go_to_left=False,
                    left=0,
                    right=0,
                    is_leaf=True,
                    value=0.2,
                ),
                dict(
                    feature_idx=0,
                    numeric_threshold=0.0,
                    missing_go_to_left=False,
                    left=0,
                    right=0,
                    is_leaf=True,
                    value=-0.3,
                ),
            ],
            [
                dict(
                    feature_idx=1,
                    numeric_threshold=2.0,
                    missing_go_to_left=False,
                    left=1,
                    right=2,
                    is_leaf=False,
                    value=-99.0,
                    is_categorical=False,
                ),
                dict(
                    feature_idx=0,
                    numeric_threshold=0.0,
                    missing_go_to_left=False,
                    left=0,
                    right=0,
                    is_leaf=True,
                    value=-0.4,
                ),
                dict(
                    feature_idx=0,
                    numeric_threshold=0.0,
                    missing_go_to_left=False,
                    left=0,
                    right=0,
                    is_leaf=True,
                    value=0.5,
                ),
            ],
        ],
    }


def test_prediction_threshold_missing_leaf_values_and_row_independence():
    """场景：阈值相等、左右分支及NaN；预期只加叶值，缺失按标记且分批结果相同。"""
    model = BoostedTreeModel.from_dict(model_payload())
    features = np.array([[1, 2], [2, 3], [np.nan, 2], [2, np.nan], [np.nan, np.nan]])
    prediction = model.predict(features, feature_names=("momentum", "volatility"))
    np.testing.assert_allclose(prediction, [-0.1, 0.3, -0.1, 0.3, 0.8], atol=1e-15)
    for index in range(len(features)):
        assert (
            model.predict(features[index : index + 1], feature_names=model.feature_names)[0]
            == (prediction[index])
        )
    assert prediction.dtype == np.float64
    assert model.predict(np.empty((0, 2)), feature_names=model.feature_names).shape == (0,)
    np.testing.assert_allclose(
        model.predict(np.array([[1, 2]], dtype=np.int64), feature_names=model.feature_names), [-0.1]
    )


def test_deeper_tree_noncontiguous_input_and_baseline_only():
    """场景：两层树、非连续数组、单叶和无树；预期逐层选叶且空集成只返回基准。"""
    payload = model_payload()
    nodes = payload["trees"][0]
    nodes[1] = dict(nodes[0], feature_idx=1, numeric_threshold=0.0, left=3, right=4)
    nodes.extend([dict(nodes[2], value=0.7), dict(nodes[2], value=0.9)])
    payload["trees"] = [nodes]
    model = BoostedTreeModel.from_dict(payload)
    features = np.array([[0.0, -1.0, 99.0], [0.0, 1.0, 99.0]])[:, :2]
    assert not features.flags.c_contiguous
    np.testing.assert_allclose(
        model.predict(features, feature_names=model.feature_names), [0.8, 1.0]
    )
    single = replace(model, trees=((NumericTreeNode(0, 0, False, 0, 0, True, 0.4),),))
    np.testing.assert_allclose(
        single.predict(features, feature_names=model.feature_names), [0.5, 0.5]
    )
    baseline = replace(model, trees=())
    np.testing.assert_allclose(
        baseline.predict(features, feature_names=model.feature_names), [0.1, 0.1]
    )


@pytest.mark.parametrize(
    "features,names",
    [
        ([[1, 2]], ("momentum", "volatility")),
        (np.ones(2), ("momentum", "volatility")),
        (np.ones((1, 3)), ("momentum", "volatility")),
        (np.array([[np.inf, 0]]), ("momentum", "volatility")),
        (np.array([[-np.inf, 0]]), ("momentum", "volatility")),
        (np.array([["1", "2"]]), ("momentum", "volatility")),
        (np.array([[1, 2]], dtype=object), ("momentum", "volatility")),
        (np.array([[True, False]]), ("momentum", "volatility")),
        (np.array([[1j, 0]]), ("momentum", "volatility")),
        (np.ones((1, 2)), ["momentum", "volatility"]),
        (np.ones((1, 2)), ("volatility", "momentum")),
        (np.ones((1, 2)), ("momentum", "unknown")),
    ],
)
def test_input_contract_rejects_unsafe_conversion_and_wrong_axis(features, names):
    """场景：非数值、Inf、错维度或特征错序；预期在推理前统一拒绝。"""
    model = BoostedTreeModel.from_dict(model_payload())
    with pytest.raises(ValueError, match="invalid_boosted_tree_features"):
        model.predict(features, feature_names=names)


@pytest.mark.parametrize(
    "changes",
    [
        {"feature_names": ["x"]},
        {"feature_names": ()},
        {"feature_names": ("x", "x")},
        {"feature_names": ("",)},
        {"feature_names": ("  ",)},
        {"feature_names": (1,)},
        {"baseline_prediction": True},
        {"baseline_prediction": "0"},
        {"baseline_prediction": np.inf},
        {"baseline_prediction": np.nan},
        {"trees": []},
    ],
)
def test_direct_model_construction_rejects_invalid_contract(changes):
    """场景：直接构造绕开JSON时传入重复轴或非法基准；预期仍由模型验证拒绝。"""
    model = BoostedTreeModel.from_dict(model_payload())
    with pytest.raises(ValueError, match="invalid_boosted_tree_model"):
        replace(model, **changes)


@pytest.mark.parametrize(
    "key,value",
    [
        ("feature_idx", -1),
        ("feature_idx", 2),
        ("feature_idx", True),
        ("left", -1),
        ("right", 99),
        ("right", 1),
        ("left", 1.0),
        ("missing_go_to_left", 1),
        ("is_leaf", 0),
        ("is_categorical", 0),
        ("is_categorical", True),
        ("numeric_threshold", np.nan),
        ("numeric_threshold", np.inf),
        ("numeric_threshold", True),
        ("value", "1"),
        ("value", -np.inf),
    ],
)
def test_node_contract_rejects_categories_indices_flags_and_nonfinite_values(key, value):
    """场景：类别分裂、无效节点索引或非有限参数；预期不尝试退化或修补模型。"""
    payload = model_payload()
    payload["trees"][0][0][key] = value
    with pytest.raises(ValueError, match="invalid_boosted_tree_node"):
        BoostedTreeModel.from_dict(payload)


def test_tree_graph_rejects_cycles_shared_children_unreachable_and_leaf_pointers():
    """场景：环、共享子树、孤立节点、叶子伪指针、空树；预期全部在预测前拒绝。"""
    initial = model_payload()
    cycle = copy.deepcopy(initial)
    cycle["trees"][0][0]["left"] = 0
    shared = copy.deepcopy(initial)
    shared["trees"][0][1] = dict(shared["trees"][0][0], left=2, right=3)
    shared["trees"][0].append(dict(shared["trees"][0][2]))
    disconnected = copy.deepcopy(initial)
    disconnected["trees"][0].append(dict(disconnected["trees"][0][2]))
    for payload in (cycle, shared, disconnected):
        with pytest.raises(ValueError, match="invalid_boosted_tree_graph"):
            BoostedTreeModel.from_dict(payload)
    initial["trees"][0][1]["left"] = 1
    with pytest.raises(ValueError, match="invalid_boosted_tree_node"):
        BoostedTreeModel.from_dict(initial)
    for trees in (((),), ([],), ((None,),)):
        with pytest.raises(ValueError, match="invalid_boosted_tree_node"):
            BoostedTreeModel(("x",), 0.0, trees)


@pytest.mark.parametrize(
    "kind",
    [
        "root_list",
        "unknown_root",
        "missing_root",
        "version",
        "boolean_version",
        "names_tuple",
        "trees_tuple",
        "tree_dict",
        "node_list",
        "unknown_node",
        "missing_node",
    ],
)
def test_json_schema_rejects_unknown_or_incomplete_structures(kind):
    """场景：未知版本/字段、缺字段或非JSON集合形状；预期严格拒绝而非忽略。"""
    payload = model_payload()
    if kind == "root_list":
        payload = []
    elif kind == "unknown_root":
        payload["loss_link"] = "exp"
    elif kind == "missing_root":
        del payload["baseline_prediction"]
    elif kind == "version":
        payload["schema_version"] = 2
    elif kind == "boolean_version":
        payload["schema_version"] = True
    elif kind == "names_tuple":
        payload["feature_names"] = ("x",)
    elif kind == "trees_tuple":
        payload["trees"] = ()
    elif kind == "tree_dict":
        payload["trees"] = [{}]
    elif kind == "node_list":
        payload["trees"] = [[[]]]
    elif kind == "unknown_node":
        payload["trees"][0][0]["bitset_idx"] = 0
    else:
        del payload["trees"][0][0]["numeric_threshold"]
    with pytest.raises(ValueError, match="invalid_boosted_tree_schema"):
        BoostedTreeModel.from_dict(payload)


def test_file_loader_rejects_duplicate_keys_and_reads_utf8(tmp_path):
    """场景：完整JSON文件及重复键攻击；预期合法模型可推理、重复键或坏JSON拒绝。"""
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model_payload()), encoding="utf-8")
    model = load_boosted_tree_model(path)
    assert model.feature_names == ("momentum", "volatility")
    path.write_text('{"value": 1, "value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate_boosted_tree_json_key"):
        load_boosted_tree_model(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_boosted_tree_model(path)


def test_finite_parameters_with_overflowing_sum_fail_closed():
    """场景：有限基准与叶值相加溢出；预期抛错，不返回Inf作为最高评分。"""
    model = BoostedTreeModel(("x",), 1e308, ((NumericTreeNode(0, 0, False, 0, 0, True, 1e308),),))
    with pytest.raises(ValueError, match="invalid_boosted_tree_prediction"):
        model.predict(np.ones((1, 1)), feature_names=("x",))


def test_input_conversion_overflow_cannot_silently_choose_a_branch():
    """场景：扩展精度数值无法表示为float64；预期拒绝转换溢出而不是当Inf走右侧。"""
    model = BoostedTreeModel.from_dict(model_payload())
    with np.errstate(over="ignore"):
        large = np.longdouble(np.finfo(np.float64).max) * 2
    features = np.array([[large, 0]], dtype=np.longdouble)
    with pytest.raises(ValueError, match="invalid_boosted_tree_features"):
        model.predict(features, feature_names=model.feature_names)
