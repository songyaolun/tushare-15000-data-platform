"""观察、公开、生效和报告期分别建模；未认证历史用途始终关闭。"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..pit import time_boundary
from ..standard import normalize_date, normalize_timestamp
from .contracts import QualityGateResult, Truth
from .model import Identifier, Model, Period, Timestamp


def public_time_boundary(value: str, market_timezone: str) -> str:
    """日期级公告按来源市场下一日零点可用，时间戳必须自带时区。

    输入 ('2026-09-30','Asia/Shanghai')，输出 '2026-09-30T16:00:00Z'。
    没有市场时区或非法日期不套用统一 UTC 日期边界。
    """
    try:
        zone = ZoneInfo(market_timezone)
        if "T" in value or " " in value:
            return normalize_timestamp(value)
        day = datetime.fromisoformat(normalize_date(value)).date() + timedelta(days=1)
        return normalize_timestamp(datetime.combine(day, time.min, tzinfo=zone))
    except (ValueError, TypeError, OverflowError, ZoneInfoNotFoundError):
        raise ValueError("public_time_unverified") from None


@dataclass(frozen=True, kw_only=True)
class AvailabilityFact(Model):
    """来源记录的时间证据；单纯回填不会赋予历史 PIT 认证。"""

    source_api: Identifier
    source_record_id: Identifier
    certification_version: Identifier
    observed_at: Timestamp
    data_as_of: Period | None
    public_available_at: Timestamp | None = None
    effective_from: Timestamp | None = None
    effective_to: Timestamp | None = None
    report_period: Period | None = None
    historical_pit_certified: bool = False
    time_precision: Literal["timestamp", "date", "observed_only"] = "observed_only"
    market_timezone: str | None = None

    def __post_init__(self) -> None:
        """拒绝颠倒区间和本阶段无法认证的历史声明；输入 true 认证 → 拒绝。

        输入公开时间未知的合法观测，输出仅允许观察后研究的时间事实。
        """
        super().__post_init__()
        if self.effective_to is not None and (
            self.effective_from is None
            or time_boundary(self.effective_to) <= time_boundary(self.effective_from)
        ):
            raise ValueError("invalid_effective_interval")
        if self.historical_pit_certified:
            raise ValueError("historical_certification_not_implemented")

    @property
    def available_at(self) -> str:
        """返回观测与公开保守边界的较晚时点；未知公开时间返回 observed_at。

        例如公开 9 月 30 日、观测 10 月 2 日，返回 10 月 2 日的观测时刻。
        """
        values = [self.observed_at]
        if self.public_available_at is not None:
            values.append(self.public_available_at)
        return max(values, key=time_boundary)


def availability_gate(
    fact: AvailabilityFact,
    analysis_as_of: str,
    *,
    require_effective: bool = False,
    mode: str = "observed_research",
) -> QualityGateResult:
    """按用途检查可见性，已知未来事件可入日历但不能当已生效状态。

    输入 observed_research 下已观察事实，返回 TRUE；mode='historical' 返回 FALSE。
    没有生效时间且 require_effective=True 时返回 UNKNOWN，不默认已生效。
    """
    cutoff = time_boundary(normalize_timestamp(analysis_as_of))
    codes = []
    if mode == "historical":
        codes.append("historical_pit_uncertified")
    elif mode != "observed_research":
        raise ValueError("unknown_availability_mode")
    if time_boundary(fact.available_at) > cutoff:
        codes.append("not_yet_available")
    if require_effective:
        if fact.effective_from is None:
            codes.append("effective_time_unknown")
        elif time_boundary(fact.effective_from) > cutoff:
            codes.append("not_yet_effective")
        if fact.effective_to is not None and time_boundary(fact.effective_to) <= cutoff:
            codes.append("no_longer_effective")
    if codes:
        return QualityGateResult(
            gate_id="availability",
            result=Truth.UNKNOWN if codes == ["effective_time_unknown"] else Truth.FALSE,
            reason_codes=tuple(codes),
        )
    return QualityGateResult(
        gate_id="availability",
        result=Truth.TRUE,
        reason_codes=("observed_only",) if fact.public_available_at is None else (),
    )
