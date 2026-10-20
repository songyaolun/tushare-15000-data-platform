"""既有P03价格结构的三臂事件实验，不包含财务或资金流产品门禁。"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..equity_research.right_side_policy import RightSidePolicy
from ..equity_research.right_side_structure import PriceBar, structure_at
from ..regime_research.panel import Panel
from ..regime_research.portfolio import PortfolioPolicy
from ..strategy_optimization.features import FloatArray, lag, rolling
from .contracts import StrategyDefinition
from .sector import _sector_features
from .windows import complete_history

REQUIRED_FEATURES = (
    "valid",
    "close",
    "high",
    "low",
    "raw_amount",
    "ma20",
    "return1",
    "sector_index",
)
STRATEGY_NAMES = ("breakout_direct", "first_retest", "repeat_retest")


def _valid_quotes(panel: Panel, features: dict[str, FloatArray]) -> NDArray[np.bool_]:
    """验证输入轴并标记有限正OHLC与成交额；例low>close或停牌对应False。

    市场日不能重复或逆序，缺字段/错维度抛ValueError；未知行情不压缩为观察日。
    """
    shape = len(panel.calendar), len(panel.codes)
    if panel.calendar != sorted(set(panel.calendar)) or any(
        name not in features or features[name].shape != shape for name in REQUIRED_FEATURES
    ):
        raise ValueError("breakout_retest_features_or_calendar_invalid")
    valid = features["valid"] == 1
    for name in ("close", "high", "low", "raw_amount"):
        valid &= np.isfinite(features[name]) & (features[name] > 0)
    return np.asarray(
        valid & (features["low"] <= features["close"]) & (features["close"] <= features["high"]),
        dtype=np.bool_,
    )


def _source_candidates(
    features: dict[str, FloatArray], valid: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """用必要条件预筛源日，避免为全部日×证券构造61个PriceBar对象。

    例连续61有效日、收盘突破此前10日最高价且金额>=前20日中位1.2倍才进入精确结构检查。
    这些是默认structure_at的必要条件，筛选通过不代表回调/空间等其余条件已通过。
    """
    candidates = complete_history(valid, 60)
    for column in range(valid.shape[1]):
        prior_high = lag(rolling(features["high"][:, column], 10, "max"))
        prior_amount = lag(rolling(features["raw_amount"][:, column], 20, "median"))
        candidates[:, column] &= (features["close"][:, column] > prior_high) & (
            features["raw_amount"][:, column] >= 1.2 * prior_amount
        )
    return candidates


def _bar(panel: Panel, features: dict[str, FloatArray], day: int, column: int) -> PriceBar:
    """将已验证的一日复权价格与原成交额转为结构输入；例(day=60,column=0)返回该日PriceBar。

    金额只比较同单位比例；close/high/low已由共用特征层保持相同的因果复权锚。
    """
    return PriceBar(
        panel.calendar[day],
        float(features["close"][day, column]),
        float(features["high"][day, column]),
        float(features["low"][day, column]),
        float(features["raw_amount"][day, column]),
    )


def _evaluate(
    panel: Panel, features: dict[str, FloatArray]
) -> tuple[
    dict[str, NDArray[np.bool_]],
    FloatArray,
    NDArray[np.bool_],
    NDArray[np.bool_],
    list[dict[str, Any]],
]:
    """一次纯计算生成三臂、共享分数/行业规则及完整源事件诊断。

    例源在60日、61日首触有效、62日再次有效，direct仅60、first仅61、repeat为61/62。
    所有臂共用同一状态机与五市场日窗口；首触失败、缺价、停牌或失守后事件注销。
    首触成功后，重复臂可接受以后仍守位的有效触碰；不允许重建已失败的同一事件。
    诊断结尾状态只描述截至输入末日的信息；right_censored不是没有机会或零收益。
    """
    valid = _valid_quotes(panel, features)
    candidates = _source_candidates(features, valid)
    policy = RightSidePolicy()
    masks = {name: np.zeros(valid.shape, dtype=bool) for name in STRATEGY_NAMES}
    scores = np.full(valid.shape, np.nan)
    price_failed = np.zeros(valid.shape, dtype=bool)
    industry = _sector_features(features)
    tradable = (features["valid"] == 1) & np.isfinite(features["close"]) & (features["close"] > 0)
    industry_exit = tradable & (industry["momentum20"] <= 0)
    industry_blocked = tradable & ((industry["breadth"] < 0.4) | (industry["momentum20"] <= 0))
    events: list[dict[str, Any]] = []
    active: dict[int, tuple[dict[str, Any], list[PriceBar]]] = {}
    next_source_day = np.zeros(valid.shape[1], dtype=int)
    for day in range(len(panel.calendar)):
        occupied_today = set(active)
        for column, (event, bars) in list(active.items()):
            scores[day, column] = event["source_score"]
            within_window = day <= event["source_day"] + policy.confirmation_sessions
            if not valid[day, column]:
                if within_window:
                    event.update(status="invalid_observation", end_day=day)
                else:
                    event["tail_failure"] = {"day": day, "reason": "invalid_observation"}
                price_failed[day, column] = True
                del active[column]
                continue
            current = _bar(panel, features, day, column)
            bars.append(current)
            pivot = event["pivot"]
            touched = current.low <= pivot * (1 + policy.retest_tolerance)
            first_touch = within_window and touched and event["first_touch_day"] is None
            if first_touch:
                event["first_touch_day"] = day
            if current.low < pivot * (1 - policy.retest_tolerance) or current.close < pivot:
                if within_window:
                    event.update(status="structure_failed", end_day=day)
                else:
                    event["tail_failure"] = {"day": day, "reason": "structure_failed"}
                price_failed[day, column] = True
                del active[column]
                continue
            if not within_window:
                if day == event["source_day"] + policy.confirmation_sessions + 3:
                    del active[column]
                continue
            if touched:
                observation = structure_at(tuple(bars), 60, policy)
                confirmed = observation.reason is None
                event["touches"].append(
                    {"day": day, "confirmed": confirmed, "reason": observation.reason}
                )
                if first_touch and not confirmed:
                    event.update(status="first_touch_failed", end_day=day)
                    price_failed[day, column] = True
                    del active[column]
                    continue
                if confirmed:
                    masks["repeat_retest"][day, column] = True
                    event["repeat_confirmation_days"].append(day)
                    if first_touch:
                        masks["first_retest"][day, column] = True
                        event["first_confirmation_day"] = day
            if day == event["source_day"] + policy.confirmation_sessions:
                event.update(
                    status="window_expired_without_touch"
                    if event["first_touch_day"] is None
                    else "window_expired_after_retest",
                    end_day=day,
                )
        for position in np.flatnonzero(candidates[day]):
            column = int(position)
            if column in occupied_today or day < next_source_day[column]:
                continue
            bars = [_bar(panel, features, prior, column) for prior in range(day - 60, day + 1)]
            observation = structure_at(tuple(bars), 60, policy)
            if observation.reason is not None:
                continue
            values = observation.values
            event = {
                "event_id": f"{panel.codes[column]}:{panel.calendar[day]}",
                "security": column,
                "ts_code": panel.codes[column],
                "source_day": day,
                "source_date": panel.calendar[day],
                "pivot": values["confirmation_price"],
                "support": min(bar.low for bar in bars[50:60]),
                "invalidation_price": values["invalidation_price"],
                "historical_resistance": values["historical_resistance"],
                "source_close": bars[-1].close,
                "source_amount": bars[-1].amount,
                "source_score": values["structural_reward_risk"],
                "source_industry_blocked": bool(industry_blocked[day, column]),
                "first_touch_day": None,
                "first_confirmation_day": None,
                "repeat_confirmation_days": [],
                "touches": [],
                "status": "right_censored",
                "end_day": len(panel.calendar) - 1,
                "tail_failure": None,
                "rearm_not_before_day": day + policy.confirmation_sessions + 4,
            }
            masks["breakout_direct"][day, column] = True
            scores[day, column] = event["source_score"]
            events.append(event)
            active[column] = (event, bars)
            next_source_day[column] = event["rearm_not_before_day"]
    return masks, scores, industry_exit, industry_blocked | price_failed, events


def build_strategies(panel: Panel, features: dict[str, FloatArray]) -> list[StrategyDefinition]:
    """生成breakout_direct、first_retest、repeat_retest三臂，固定源事件RR评分。

    源结构使用61连续市场日及RightSidePolicy原阈值：10日整理、回调至少5%、抬低.5%、
    整理宽<=12%、均线转强、金额1.2倍、追离<=3%、结构风险<=8%、空间风险比>=2。
    只迁移价格结构，不称完整财务P03。突破源上下沿/阻力冻结，后1至5日首次触碰
    pivot上方1%以内须低价守住下方1%、收盘守位且上涨、金额不超过突破日，并通过
    原价格风险门；首触失败注销，三臂不提前重启源事件。回踩时结构_at始终以源日为锚。
    各臂TP8%/SL8%/H30、2%追价、90%/5%/25%资金规则相同，无额外市场广度过滤。
    共同持仓退出为行业momentum20<=0；共同入场否决/撤单为原行业广度<.4或动量<=0，
    加价格事件失败当日。确认日收盘发单后沿用三日订单期限，五日事件到期不取消已确认
    且未失败的挂单。三臂共同锁定至源后第8日收盘，最早第9日重武装，提前失败也不缩短；
    第6至8日仅监控原价位失守/缺价以撤原待单，不再确认入场，避免新事件失败误撤旧单。
    持仓不因价格事件结束而新增卖出。成交仍为之后合格开盘。
    """
    masks, scores, exit_mask, invalidation, _ = _evaluate(panel, features)
    return [
        StrategyDefinition(
            name=name,
            family="breakout_retest",
            entry_mask=masks[name],
            scores=scores,
            exit_mask=exit_mask,
            entry_invalidation_mask=invalidation,
            policy=PortfolioPolicy(
                name=name,
                switching=False,
                stage_allocation=False,
                force_rebalance=False,
                allocations=(0.9,) * 5,
                position_weight=0.05,
                sector_weight=0.25,
                take_profit=0.08,
                stop_loss=0.08,
                maximum_holding=30,
            ),
            maximum_entry_gap=0.02,
        )
        for name in STRATEGY_NAMES
    ]


def diagnose_events(panel: Panel, features: dict[str, FloatArray]) -> list[dict[str, Any]]:
    """输出所有合格价格源事件、冻结锚、首触、重复确认和结束状态，不计算收益。

    例从未回踩但完整观察五日的事件为window_expired_without_touch；末日仍在等待
    为right_censored。包括源日被行业否决的事件，未成交或拒绝事件没有收益字段，
    不会被误写为零回报；实际成交/费用和无回踩机会成本需由独立账本及后续分析评估。
    """
    return _evaluate(panel, features)[-1]
