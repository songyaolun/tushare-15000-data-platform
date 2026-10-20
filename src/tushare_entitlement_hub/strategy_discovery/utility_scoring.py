"""可解释的分箱加性评分；净收益、大亏损及期限分别估计，权重可逐项审计。"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..strategy_optimization.features import FloatArray


def bucket_features(values: FloatArray, boundaries: tuple[float, ...]) -> FloatArray:
    """按预先固定阈值编码非基准箱与缺失箱，未知不填成正常值。

    输入一维特征及递增有限阈值，输出行×(阈值数+1)矩阵，最小箱为全零。
    例values=[-1,0,2,nan]、boundaries=(0,1)输出[[0,0,0],[1,0,0],
    [0,1,0],[0,0,1]]。阈值与当天其他候选无关，追加未来行不改变历史编码。
    """
    cuts = np.asarray(boundaries, dtype=float)
    if (
        values.ndim != 1
        or cuts.ndim != 1
        or not len(cuts)
        or not np.all(np.isfinite(cuts))
        or np.any(np.diff(cuts) <= 0)
    ):
        raise ValueError("invalid_bucket_features")
    known = np.isfinite(values)
    buckets = np.searchsorted(cuts, values, side="right")
    return np.column_stack(
        [known & (buckets == index) for index in range(1, len(cuts) + 1)] + [~known]
    ).astype(float)


@dataclass(frozen=True)
class AdditiveModel:
    """冻结的多目标加性模型；例系数[[.01]]表示该特征对目标增加1个百分点。"""

    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    intercept: FloatArray
    coefficients: FloatArray

    def __post_init__(self) -> None:
        """拒绝名称重复、错轴及非有限权重；例两个同名特征抛ValueError。"""
        if (
            not self.feature_names
            or not self.target_names
            or len(set(self.feature_names)) != len(self.feature_names)
            or len(set(self.target_names)) != len(self.target_names)
            or self.coefficients.shape != (len(self.feature_names), len(self.target_names))
            or self.intercept.shape != (len(self.target_names),)
            or not np.all(np.isfinite(self.coefficients))
            or not np.all(np.isfinite(self.intercept))
        ):
            raise ValueError("invalid_additive_model")

    def predict(self, features: FloatArray) -> FloatArray:
        """输入有限行×特征矩阵，返回行×目标预测；例截距.01加特征贡献.02为.03。"""
        if (
            features.ndim != 2
            or features.shape[1] != len(self.feature_names)
            or not np.all(np.isfinite(features))
        ):
            raise ValueError("invalid_additive_features")
        return features @ self.coefficients + self.intercept

    def contributions(self, features: FloatArray, target: int) -> FloatArray:
        """逐项分解指定目标，首列截距；例输出[.01,.02]相加等于预测.03。"""
        if not 0 <= target < len(self.target_names):
            raise ValueError("invalid_additive_target")
        self.predict(features)
        return np.column_stack(
            (
                np.full(len(features), self.intercept[target]),
                features * self.coefficients[:, target],
            )
        )


def fit_additive_model(
    features: FloatArray,
    targets: FloatArray,
    days: NDArray[np.int64],
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    *,
    regularization: float = 0.01,
) -> AdditiveModel:
    """按日期等权拟合多目标岭回归，截距不惩罚，显式减少同日大量候选的权重。

    输入特征、目标及对应市场日，输出冻结模型；例同日复制所有记录不改变拟合。
    targets可分别为净收益、超过5%亏损的金额比例、持有日数和盈利指示。
    返回值未自动校准概率或选择参数，也不创建训练/验证分割。调用者若使用完整
    五年标签，回放必为样本内拟合，不可把无未来特征声称为历史可部署权重。
    """
    if (
        features.ndim != 2
        or targets.ndim != 2
        or days.ndim != 1
        or not len(features)
        or len(targets) != len(features)
        or len(days) != len(features)
        or features.shape[1] != len(feature_names)
        or targets.shape[1] != len(target_names)
        or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(targets))
        or not np.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError("invalid_additive_fit")
    _, inverse, counts = np.unique(days, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse].astype(float)
    weights /= weights.sum()
    feature_mean = weights @ features
    target_mean = weights @ targets
    centered_features = features - feature_mean
    centered_targets = targets - target_mean
    covariance = centered_features.T @ (centered_features * weights[:, None])
    covariance.flat[:: len(feature_names) + 1] += regularization
    response = centered_features.T @ (centered_targets * weights[:, None])
    coefficients = np.asarray(np.linalg.solve(covariance, response), dtype=np.float64)
    return AdditiveModel(
        feature_names, target_names, target_mean - feature_mean @ coefficients, coefficients
    )


def utility_priority(
    expected_return: FloatArray,
    expected_tail_loss: FloatArray,
    expected_holding: FloatArray,
    *,
    tail_penalty: float = 1,
    daily_cost: float = 0.0002,
    scale: float = 0.02,
) -> tuple[FloatArray, FloatArray]:
    """计算经济效用与固定尺度分数，不依赖当天候选排名，也不保证最高分接近100。

    效用=预测净收益-尾部惩罚×预测超过5%的亏损-每日资金成本×预测持有日数。
    分数=50+50*tanh(效用/scale)。例净收益0、尾损0、持有10、每日成本.0002，
    效用=-.002且分数低于50；全池亏损也不会强造100分。负尾损预测截为0，
    持有期最低1日。分数不是胜率；scale=.02只是固定展示尺度，不参与拟合。
    """
    if (
        expected_return.shape != expected_tail_loss.shape
        or expected_return.shape != expected_holding.shape
        or any(
            not np.all(np.isfinite(value))
            for value in (expected_return, expected_tail_loss, expected_holding)
        )
        or not np.all(np.isfinite([tail_penalty, daily_cost, scale]))
        or tail_penalty < 0
        or daily_cost < 0
        or scale <= 0
    ):
        raise ValueError("invalid_utility_priority")
    utility = (
        expected_return
        - tail_penalty * np.maximum(expected_tail_loss, 0)
        - daily_cost * np.maximum(expected_holding, 1)
    )
    return utility, 50 + 50 * np.tanh(utility / scale)
