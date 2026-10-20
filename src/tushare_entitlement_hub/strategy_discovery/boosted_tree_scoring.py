"""数值提升树的可移植研究推理：只读JSON和NumPy，不加载pickle或训练依赖。"""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NumericTreeNode:
    """不可变数值节点；例阈值.5、左右子节点1/2将等于.5的输入送往节点1。"""

    feature_idx: int
    numeric_threshold: float
    missing_go_to_left: bool
    left: int
    right: int
    is_leaf: bool
    value: float
    is_categorical: bool = False


def _validate_tree(nodes: tuple[NumericTreeNode, ...], feature_count: int) -> None:
    """验证根0的完整数值树，拒绝循环、共享或不可达节点；例仅一个合法叶节点通过。

    叶子左右索引必须为0，与HGB导出一致；内部节点索引须在树内且不能相同。
    所有节点的特征、阈值、数值和布尔标记都验证，即使该字段在叶子上不参与预测。
    """
    if type(nodes) is not tuple or not nodes:
        raise ValueError("invalid_boosted_tree_nodes")
    for node in nodes:
        if (
            not isinstance(node, NumericTreeNode)
            or any(type(getattr(node, key)) is not int for key in ("feature_idx", "left", "right"))
            or not 0 <= node.feature_idx < feature_count
            or not 0 <= node.left < len(nodes)
            or not 0 <= node.right < len(nodes)
            or any(
                type(getattr(node, key)) is not bool
                for key in ("missing_go_to_left", "is_leaf", "is_categorical")
            )
            or node.is_categorical
            or any(
                type(value) not in (int, float) or not isfinite(value)
                for value in (node.numeric_threshold, node.value)
            )
            or (node.is_leaf and (node.left != 0 or node.right != 0))
            or (not node.is_leaf and node.left == node.right)
        ):
            raise ValueError("invalid_boosted_tree_node")
    visited: set[int] = set()
    pending = [0]
    while pending:
        index = pending.pop()
        if index in visited:
            raise ValueError("invalid_boosted_tree_graph")
        visited.add(index)
        node = nodes[index]
        if not node.is_leaf:
            pending.extend((node.left, node.right))
    if len(visited) != len(nodes):
        raise ValueError("invalid_boosted_tree_graph")


@dataclass(frozen=True)
class BoostedTreeModel:
    """冻结的标量净收益提升树；例baseline=.01加两棵叶值.02/-.01输出.02。

    叶值必须已经包含训练时的学习率，预测时不再缩放。不作概率校准或输出变换，
    调用者可按较高预测净收益优先排序；这不是胜率或历史可部署能力的声明。
    """

    feature_names: tuple[str, ...]
    baseline_prediction: float
    trees: tuple[tuple[NumericTreeNode, ...], ...]

    def __post_init__(self) -> None:
        """验证不可变轴、有限基准与全部树；例重复特征名或baseline=Inf抛ValueError。"""
        if (
            type(self.feature_names) is not tuple
            or not self.feature_names
            or any(type(name) is not str or not name.strip() for name in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
            or type(self.baseline_prediction) not in (int, float)
            or not isfinite(self.baseline_prediction)
            or type(self.trees) is not tuple
        ):
            raise ValueError("invalid_boosted_tree_model")
        for nodes in self.trees:
            _validate_tree(nodes, len(self.feature_names))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoostedTreeModel":
        """读取严格v1对象，拒绝未知字段及类别分裂；例trees=[]返回仅基准值的模型。

        根字段恰为schema_version=1、feature_names列表、baseline_prediction标量、
        trees列表。每树为节点列表；节点含NumericTreeNode前七个同名字段，
        可额外声明is_categorical=false。类别位集、损失链接等未支持语义不会忽略。
        """
        if (
            type(payload) is not dict
            or set(payload) != {"schema_version", "feature_names", "baseline_prediction", "trees"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or type(payload["feature_names"]) is not list
            or type(payload["trees"]) is not list
        ):
            raise ValueError("invalid_boosted_tree_schema")
        required = {
            "feature_idx",
            "numeric_threshold",
            "missing_go_to_left",
            "left",
            "right",
            "is_leaf",
            "value",
        }
        trees = []
        for records in payload["trees"]:
            if type(records) is not list:
                raise ValueError("invalid_boosted_tree_schema")
            nodes = []
            for record in records:
                if (
                    type(record) is not dict
                    or not required <= set(record)
                    or set(record) - required - {"is_categorical"}
                ):
                    raise ValueError("invalid_boosted_tree_schema")
                nodes.append(NumericTreeNode(**record))
            trees.append(tuple(nodes))
        return cls(tuple(payload["feature_names"]), payload["baseline_prediction"], tuple(trees))

    def predict(self, features: FloatArray, *, feature_names: tuple[str, ...]) -> FloatArray:
        """按明确列名返回一维float64净收益，缺失走指定方向；例阈值1将输入1送左。

        输入为二维实数数组，行数可为0，允许NaN但拒绝正负Inf；列名必须逐项同序。
        每行独立沿根到叶路径计算baseline加全部叶值，追加未来行不改变历史预测。
        不允许字符串、布尔或复数隐式转成价格特征，累加溢出也拒绝交付。
        """
        if (
            not isinstance(features, np.ndarray)
            or features.ndim != 2
            or features.dtype.kind not in "fiu"
            or features.shape[1] != len(self.feature_names)
            or type(feature_names) is not tuple
            or feature_names != self.feature_names
            or np.any(np.isinf(features))
        ):
            raise ValueError("invalid_boosted_tree_features")
        with np.errstate(over="ignore"):
            values = np.asarray(features, dtype=np.float64)
        if np.any(np.isinf(values)):
            raise ValueError("invalid_boosted_tree_features")
        predictions = np.full(len(values), self.baseline_prediction, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for nodes in self.trees:
                pending = [(0, np.arange(len(values)))]
                while pending:
                    index, rows = pending.pop()
                    node = nodes[index]
                    if node.is_leaf:
                        predictions[rows] += node.value
                        continue
                    observed = values[rows, node.feature_idx]
                    left = np.where(
                        np.isnan(observed),
                        node.missing_go_to_left,
                        observed <= node.numeric_threshold,
                    )
                    if np.any(left):
                        pending.append((node.left, rows[left]))
                    if np.any(~left):
                        pending.append((node.right, rows[~left]))
        if not np.all(np.isfinite(predictions)):
            raise ValueError("invalid_boosted_tree_prediction")
        return predictions


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝JSON重复键，避免后值悄悄覆盖模型语义；例两个value键抛ValueError。"""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_boosted_tree_json_key")
        result[key] = value
    return result


def load_boosted_tree_model(path: Path) -> BoostedTreeModel:
    """从UTF-8 JSON加载并验证模型，不执行序列化代码；例合法model.json得到可预测模型。"""
    return BoostedTreeModel.from_dict(
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    )
