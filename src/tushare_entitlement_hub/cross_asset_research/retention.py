"""以增量日期桶保留完整业务期间，避免每行重扫同实体的历史成员。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from ..equity_research.models import ResearchRow


@dataclass
class _PeriodBucket:
    """同一业务期的全部行及累计载荷字节；例如两条 20 字节记录累计为 40。"""

    rows: list[ResearchRow] = field(default_factory=list)
    byte_count: int = 0


@dataclass
class RetentionGroup:
    """维护同实体来源的最近两个完整日期桶，版本来源则保留单个全版本桶。

    同期追加只做列表 append 和计数累加；旧期淘汰只释放被淘汰桶，每行至多释放一次。
    例如普通来源依次追加 01 日、03 日、02 日后保留 02 日与 03 日的全部成员。
    行数和字节容量由调用方按 row_count、byte_count 对整个实体或来源拒绝。
    """

    keep_versions: bool = False
    buckets: dict[str, _PeriodBucket] = field(default_factory=dict)
    row_count: int = 0
    byte_count: int = 0

    def append(self, candidate: ResearchRow, byte_count: int) -> tuple[int, int]:
        """追加一条已适配行，返回保留行数与字节数的增量，不遍历已有成员。

        byte_count 是本行已编码载荷的非负长度，由调用方只计算一次。
        例如已有两个各 20 字节日期桶，新日期的 30 字节行返回 (0, 10)；
        早于两个保留日期的行返回 (0, 0)，keep_versions=True 则始终追加。
        """
        period = "" if self.keep_versions else candidate.availability.data_as_of or ""
        bucket = self.buckets.get(period)
        count_delta = 1
        byte_delta = byte_count
        if bucket is None:
            if len(self.buckets) == 2:
                oldest = min(self.buckets)
                if period < oldest:
                    return 0, 0
                removed = self.buckets.pop(oldest)
                count_delta -= len(removed.rows)
                byte_delta -= removed.byte_count
            bucket = _PeriodBucket()
            self.buckets[period] = bucket
        bucket.rows.append(candidate)
        bucket.byte_count += byte_count
        self.row_count += count_delta
        self.byte_count += byte_delta
        return count_delta, byte_delta

    def iter_rows(self) -> Iterator[ResearchRow]:
        """一次性遍历保留桶内全部成员，供流扫描结束后按来源记录编号排序。

        例如两个日期桶分别有 3 行和 2 行时产生 5 行，不在追加热路径中调用。
        """
        for bucket in self.buckets.values():
            yield from bucket.rows
