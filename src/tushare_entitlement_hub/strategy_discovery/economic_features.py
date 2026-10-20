"""经济含义明确的固定分箱特征；不使用旧总分、未来标签或证券身份。"""

import numpy as np

from ..strategy_optimization.features import FloatArray
from .utility_scoring import AdditiveModel, bucket_features, utility_priority

# 阈值在拟合前固定；每项包含独立缺失箱，最小箱为解释基准。
FEATURE_BUCKETS: dict[str, tuple[str, tuple[float, ...]]] = {
    "signal_group_score": ("legacy", (20, 40, 60, 80)),
    "liquidity_score": ("legacy", (20, 40, 60, 80)),
    "risk_efficiency_score": ("legacy", (25, 50, 75)),
    "return1": ("entry", (-0.03, 0, 0.03, 0.07)),
    "return5": ("entry", (-0.08, -0.02, 0.02, 0.08)),
    "return20": ("entry", (-0.15, -0.03, 0.05, 0.20)),
    "extension_atr": ("entry", (-2, 0, 2, 4)),
    "close_location": ("entry", (0.25, 0.5, 0.75)),
    "boll_position": ("entry", (0, 0.3, 0.7, 1)),
    "kdj_j": ("entry", (0, 20, 80, 100)),
    "atr_fraction": ("risk", (0.015, 0.025, 0.04, 0.06)),
    "above_ma20_age": ("entry", (1, 3, 10, 30)),
    "sector_breadth": ("industry", (0.3, 0.5, 0.7)),
    "sector_diffusion": ("industry", (-0.1, 0, 0.1)),
    "sector_momentum20": ("industry", (-0.1, 0, 0.1, 0.2)),
    "relative_return20": ("industry", (-0.1, -0.02, 0.05, 0.15)),
    "market_breadth": ("market", (0.3, 0.4, 0.6, 0.8)),
    "market_diffusion": ("market", (-0.1, 0, 0.1)),
}


def economic_features(
    raw: dict[str, FloatArray],
    hits: FloatArray,
    signal_names: tuple[str, ...],
) -> tuple[FloatArray, tuple[str, ...], tuple[str, ...]]:
    """将当日事实组成分箱、信号和显式交互矩阵，返回矩阵/名称/分组。

    raw包含FEATURE_BUCKETS的全部一维字段，hits为行×信号的0/1矩阵。
    例突破信号分别与市场强、行业扩散、涨幅过大相乘，形成可单独解释的贡献。
    每个信号都用相同的三个交互，不按回测结果筛选特定组合；不输入日期编号、
    股票代码、旧总分或未来收益。缺失状态不会被当作“强市”或“行业扩散”。
    """
    if (
        set(raw) != set(FEATURE_BUCKETS)
        or hits.ndim != 2
        or hits.shape[1] != len(signal_names)
        or not signal_names
        or len(set(signal_names)) != len(signal_names)
        or not np.all(np.isin(hits, [0, 1]))
        or any(value.shape != (len(hits),) for value in raw.values())
    ):
        raise ValueError("invalid_economic_features")
    columns: list[FloatArray] = [hits]
    names = ["signal:" + name for name in signal_names]
    groups = ["signals"] * len(signal_names)
    for name, (group, cuts) in FEATURE_BUCKETS.items():
        encoded = bucket_features(raw[name], cuts)
        columns.append(encoded)
        names.extend(
            [f"{name}:bucket{index}" for index in range(1, len(cuts) + 1)] + [name + ":missing"]
        )
        groups.extend([group] * encoded.shape[1])
    for name, state in (
        ("strong_market", raw["market_breadth"] >= 0.4),
        ("sector_diffusing", raw["sector_diffusion"] >= 0.1),
        ("extended_price", raw["extension_atr"] >= 2),
    ):
        columns.append(hits * state[:, None])
        names.extend([f"interaction:{signal}:{name}" for signal in signal_names])
        groups.extend(["interactions"] * len(signal_names))
    return np.column_stack(columns), tuple(names), tuple(groups)


def score_economic_candidates(
    raw: dict[str, FloatArray],
    hits: FloatArray,
    signal_names: tuple[str, ...],
    model: AdditiveModel,
    *,
    tail_penalty: float = 1,
    daily_cost: float = 0.0002,
) -> dict[str, FloatArray]:
    """只用当前事实和冻结模型输出高分优先结果，名称逐项对齐而不依赖列位置。

    输入与economic_features相同的事实及模型，返回score、utility、预测净收益、
    尾损、持有期和未校准盈利指示回归值；例模型净收益.01且无惩罚时score>50。
    支持消融模型特征子集；未知特征或目标顺序不符立即拒绝，避免默默错配。
    model中的权重必须独立归档拟合日期和来源；本接口不加载标签或自动再训练。
    """
    features, names, _ = economic_features(raw, hits, signal_names)
    if model.target_names != (
        "net_return",
        "tail_loss",
        "holding_sessions",
        "win_indicator",
    ) or not set(model.feature_names) <= set(names):
        raise ValueError("economic_model_schema_mismatch")
    selected = [names.index(name) for name in model.feature_names]
    prediction = model.predict(features[:, selected])
    utility, score = utility_priority(
        prediction[:, 0],
        prediction[:, 1],
        prediction[:, 2],
        tail_penalty=tail_penalty,
        daily_cost=daily_cost,
    )
    return {
        "score": score,
        "utility": utility,
        "expected_net_return": prediction[:, 0],
        "expected_tail_loss": np.maximum(prediction[:, 1], 0),
        "expected_holding_sessions": np.maximum(prediction[:, 2], 1),
        "uncalibrated_win_indicator": prediction[:, 3],
    }
