"""将退出标签接入已有经济评分，显式保留删失及双期限目标语义。"""

import numpy as np
from numpy.typing import NDArray

from ..strategy_optimization.features import FloatArray
from .utility_scoring import utility_priority

EXIT_TARGET_NAMES = ("net_return", "tail_loss", "holding_sessions", "win_indicator")


def within_date_center(values: FloatArray, days: NDArray[np.int64]) -> FloatArray:
    """按信号日期对每列去均值，保留候选轴顺序，不混用其他日期的信息。

    输入有限行×列矩阵和同轴整数日期编号，输出每个日期内列均值为零的矩阵。
    例values=[[1,4],[8,3],[3,6]]、days=[2,1,2]，返回[[-1,-1],
    [0,0],[1,1]]。调用者可分别处理成熟训练特征和未来标签，从而拟合同日
    相对排序；推理时日期内共同常数不影响Top10。输出不代表绝对预期收益，
    去均值本身也不证明消除了市场因子。单候选日返回零，输入不会被修改。
    """
    if (
        values.ndim != 2
        or days.ndim != 1
        or not len(values)
        or len(days) != len(values)
        or not np.issubdtype(days.dtype, np.integer)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("invalid_within_date_center")
    _, inverse, counts = np.unique(days, return_inverse=True, return_counts=True)
    totals = np.zeros((len(counts), values.shape[1]), dtype=float)
    np.add.at(totals, inverse, values)
    means = totals / counts[:, None]
    return values - means[inverse]


def exit_training_targets(labels: FloatArray) -> tuple[NDArray[np.bool_], FloatArray]:
    """从独立成交标签提取已实现目标，不另划金融验证集或把删失收益填零。

    输入候选行×4矩阵，列为净收益、持有日数、退出日、状态；状态0/1/2/3
    依次代表不合格、未成交、右删失、已平仓。输出已平仓布尔轴及对应四目标。
    例输入[[-.08,5,9,3],[nan,nan,nan,2]]，输出[True,False]和
    [[-.08,.03,5,0]]。调用方须先限制研究日期；所有返回目标均含未来信息，
    只能用于显式披露的全样本拟合，不能作为信号日特征。
    """
    if labels.ndim != 2 or labels.shape[1] != 4:
        raise ValueError("invalid_exit_label_shape")
    if not np.all(np.isin(labels[:, 3], [0, 1, 2, 3])):
        raise ValueError("invalid_exit_label_status")
    closed = labels[:, 3] == 3
    realized = labels[closed, :3]
    if (
        not np.all(np.isfinite(realized))
        or np.any(realized[:, 1] < 2)
        or np.any(realized[:, 1:] != np.floor(realized[:, 1:]))
        or np.any(realized[:, 2] < realized[:, 1])
        or np.any(np.isfinite(labels[~closed, :3]))
    ):
        raise ValueError("invalid_exit_label_values")
    returns = realized[:, 0]
    targets = np.column_stack(
        (returns, np.maximum(-returns - 0.05, 0), realized[:, 1], returns > 0)
    )
    return closed, targets


def dual_horizon_priority(
    short_expected_return: FloatArray,
    long_expected_return: FloatArray,
    *,
    short_weight: float = 0.5,
) -> tuple[FloatArray, FloatArray]:
    """按固定权重混合短持有和长持有预期净收益，再映射为已有固定尺度分数。

    输入两个同轴一维收益预测及[0,1]短目标权重，返回混合收益与0至100分数。
    例短目标.01、长目标.03、权重.5得到效用.02及约88.08分。两模型可分别
    使用各自已平仓标签拟合；这里不按未来实际收益挑目标、不改变逐股退出规则，
    也不将分数解释为胜率。默认等权须在账户比较前固定。
    """
    if (
        short_expected_return.ndim != 1
        or long_expected_return.shape != short_expected_return.shape
        or not np.isfinite(short_weight)
        or not 0 <= short_weight <= 1
    ):
        raise ValueError("invalid_dual_horizon_priority")
    expected_return = (
        short_weight * short_expected_return + (1 - short_weight) * long_expected_return
    )
    return utility_priority(
        expected_return,
        np.zeros_like(expected_return),
        np.ones_like(expected_return),
        tail_penalty=0,
        daily_cost=0,
    )
