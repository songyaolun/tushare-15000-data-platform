"""私有磁盘输入分区：完整来源保留于 B-tree，业务计算仅物化有界完整分组。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from ..equity_research.key_index import DiskKeyIndex
from ..equity_research.models import ResearchInputs, ResearchRow
from ..product_foundation.numeric import TypedFact
from ..product_foundation.temporal import AvailabilityFact
from ..store import encode

MAX_PARTITION_BYTES = 16 * 1024 * 1024
MAX_SPOOL_BYTES = 32 * 1024 * 1024 * 1024


def encode_row(row: ResearchRow) -> str:
    """序列化已适配行及原有事实证书，不回填、重认证或丢弃选定字段。

    输入一条含 close 证书的 ResearchRow，输出可无损恢复实体、可见性和数值的 JSON。
    原始响应仍保存在 reference；暂存只包含消费投影，不复制原始响应宽字段。
    """
    return encode(
        {
            "entity_id": row.entity_id,
            "source_api": row.source_api,
            "raw": row.raw,
            "availability": row.availability.to_dict(),
            "numeric": {name: fact.to_dict() for name, fact in row.numeric.items()},
        }
    )


def decode_row(payload: str) -> ResearchRow:
    """恢复私有暂存的行并通过公共模型验证数值与可见性结构。

    输入 encode_row(row) 得到与 row 值相同的 ResearchRow，不使用不安全对象反序列化。
    """
    value = json.loads(payload)
    return ResearchRow(
        entity_id=value["entity_id"],
        source_api=value["source_api"],
        raw=value["raw"],
        availability=AvailabilityFact.from_dict(value["availability"]),
        numeric={name: TypedFact.from_dict(item) for name, item in value["numeric"].items()},
    )


class PeriodIndex(Mapping[tuple[str, str], str]):
    """按来源和记录键读取业务期，避免每个分区重建全来源时间字典。"""

    def __init__(
        self, spool: InputSpool, fallback: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        """保留暂存及旧来源日期索引引用；例如关系旁证来自未暂存旧接口时仍可查原日期。"""
        self.spool = spool
        self.fallback = {} if fallback is None else fallback

    def __getitem__(self, key: tuple[str, str]) -> str:
        """查询已有业务期；例如 ('daily','r1') 得日期，缺键或无业务日抛 KeyError。"""
        found = self.spool.connection.execute(
            "SELECT period FROM observations WHERE api=? AND record=?", key
        ).fetchone()
        if found is None:
            return self.fallback[key]
        if not found[0]:
            raise KeyError(key)
        return str(found[0])

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """按确定顺序流式遍历有业务日期的键；例如无业务日目录不进入该索引。"""
        yield from self.spool.connection.execute(
            "SELECT api,record FROM observations WHERE period<>'' ORDER BY api,record"
        )
        for key in self.fallback:
            if self.spool.connection.execute(
                "SELECT 1 FROM observations WHERE api=? AND record=?", key,
            ).fetchone() is None:
                yield key

    def __len__(self) -> int:
        """在磁盘计算键数；例如三行中两行有业务日时返回 2，不物化键集合。"""
        return sum(1 for _ in self)


@dataclass(frozen=True, kw_only=True)
class PartitionInputs(ResearchInputs):
    """单个完整计算分区及覆盖所有暂存来源的只读证据日期索引。"""

    periods: Mapping[tuple[str, str], str]
    scope: ResearchInputs

    @cached_property
    def entity_ids(self) -> tuple[str, ...]:
        """复用全观察范围的证券缓存，不因分区重复遍历和排序完整配置。

        例如三万个合约分区引用同一证券元组，只构建一次；原范围中已排除的
        转债及指数不会因当前分区没有相应行而重新进入覆盖数量。
        """
        return self.scope.entity_ids

    @cached_property
    def entity_kinds(self) -> dict[str, str]:
        """引用全观察范围已建立的实体类型索引，消费时只读，不复制字典。

        例如两个基金组合分区查询股票 A 时复用同一类型索引，不各自扫描全实体。
        """
        return self.scope.entity_kinds

    @cached_property
    def record_periods(self) -> Mapping[tuple[str, str], str]:
        """复用磁盘证据日期索引；例如成员结果引用外部行情也能取得原业务日。"""
        return self.periods


class InputSpool(DiskKeyIndex):
    """采用 8MiB SQLite 页缓存暂存完整来源，关闭时清理，不扩大 DuckDB 内存。"""

    def __init__(self) -> None:
        """创建来源、实体和代码索引；例如十万合约扩大磁盘文件，不占满保留行内存。"""
        super().__init__()
        self.payload_bytes = 0
        try:
            self.connection.execute(
                "CREATE TABLE observations(api TEXT,entity TEXT,record TEXT,period TEXT,"
                "code TEXT,payload TEXT,bytes INTEGER,PRIMARY KEY(api,entity,record)) WITHOUT ROWID"
            )
            self.connection.execute("CREATE UNIQUE INDEX source_record ON observations(api,record)")
            self.connection.execute("CREATE INDEX source_code ON observations(api,code,entity)")
            self.connection.execute("CREATE INDEX source_period ON observations(api,entity,period)")
        except BaseException:
            self.close()
            raise

    def append(self, row: ResearchRow) -> None:
        """追加一条通过适配及单位处理的完整行，重复来源键拒绝而不覆盖。

        输入 fund_company 使用 name 建立法律主体名称查找；index_weight 使用 index_code，
        其余使用 ts_code。累计暂存载荷超 32GiB 拒绝本次发布，不丢掉已输入记录。
        """
        payload = encode_row(row)
        size = len(payload.encode("utf-8"))
        if self.payload_bytes + size > MAX_SPOOL_BYTES:
            raise ValueError("research_input_spool_capacity_exceeded")
        field = (
            "name"
            if row.source_api == "fund_company"
            else ("index_code" if row.source_api == "index_weight" else "ts_code")
        )
        self.connection.execute(
            "INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
            (
                row.source_api,
                row.entity_id or "",
                str(row.raw["source_record_id"]),
                row.availability.data_as_of or "",
                row.text(field) or "",
                payload,
                size,
            ),
        )
        self.payload_bytes += size

    def retain(self, api: str, *, keep_versions: bool, maximum_rows: int) -> int:
        """在磁盘保留每个实体最新两个完整期间，版本来源保留全部可见版本。

        输入三个日期各千名成员，返回最近两日的 2000 行；同期间不按前 N 行裁剪。
        计数超显式来源或剩余总预算时拒绝整次运行，保证不发布看似完整的部分来源。
        """
        if not keep_versions:
            self.connection.execute(
                "DELETE FROM observations WHERE api=? AND (entity,period) IN ("
                "SELECT entity,period FROM (SELECT entity,period,dense_rank() OVER ("
                "PARTITION BY entity ORDER BY period DESC) AS position FROM observations "
                "WHERE api=? GROUP BY entity,period) WHERE position>2)",
                (api, api),
            )
        count = self.count(api)
        if count > maximum_rows:
            raise ValueError("research_input_source_capacity_exceeded")
        return count

    def count(self, api: str) -> int:
        """返回来源完整保留数；例如两实体各三行得到 6，空来源得到 0。"""
        return int(
            self.connection.execute(
                "SELECT count(*) FROM observations WHERE api=?", (api,)
            ).fetchone()[0]
        )

    def records(
        self,
        api: str,
        *,
        entity: str | None = None,
        code: str | None = None,
    ) -> Iterator[tuple[str, int]]:
        """按来源、可选实体或显式代码流式查找规范行载荷，最多读取 128 行一批。

        例如 records('fund_manager',entity='fund:A') 只扫描 A 的完整管理人记录；
        代码作为 SQL 参数，不会被解释为 SQL 语句。
        """
        clause, arguments = "api=?", [api]
        if entity is not None:
            clause += " AND entity=?"
            arguments.append(entity)
        if code is not None:
            clause += " AND code=?"
            arguments.append(code)
        cursor = self.connection.execute(
            "SELECT payload,bytes FROM observations WHERE " + clause + " ORDER BY entity,record",
            arguments,
        )
        while batch := cursor.fetchmany(128):
            yield from batch

    def rows(self, api: str) -> Iterator[ResearchRow]:
        """逐行恢复来源，用于完整消费认证；例如百万行审计不创建百万行 Python 列表。"""
        for payload, _ in self.records(api):
            yield decode_row(payload)

    def entities(self, apis: Iterable[str]) -> Iterator[str]:
        """在磁盘合并并排序所需来源实体，不创建全局 Python 唯一键集合。

        输入两个接口共享 A、B 时只产生 A、B 两键；空接口集合不产生任何分区。
        """
        selected = tuple(sorted(set(apis)))
        if not selected:
            return
        placeholders = ",".join("?" for _ in selected)
        cursor = self.connection.execute(
            "SELECT entity FROM observations WHERE api IN (" + placeholders + ") "
            "GROUP BY entity ORDER BY entity",
            selected,
        )
        for (entity,) in cursor:
            yield entity

    def materialize(
        self,
        template: ResearchInputs,
        apis: Iterable[str],
        *,
        entity: str | None = None,
        codes: Mapping[str, str] | None = None,
        maximum_bytes: int = MAX_PARTITION_BYTES,
    ) -> PartitionInputs:
        """只物化一个完整计算分区，先核算字节再恢复模型，超限直接拒绝发布。

        例如 materialize(inputs,('dc_member',),entity='theme:A') 取得 A 的全部保留成员；
        单分区超过 16MiB 时抛 research_input_partition_capacity_exceeded，绝不返回前半组。
        """
        return self.materialize_selection(
            template,
            (
                (
                    api,
                    None if codes is not None and api in codes else entity,
                    None if codes is None else codes.get(api),
                )
                for api in sorted(set(apis))
            ),
            maximum_bytes=maximum_bytes,
        )

    def materialize_selection(
        self,
        template: ResearchInputs,
        selectors: Iterable[tuple[str, str | None, str | None]],
        *,
        maximum_bytes: int = MAX_PARTITION_BYTES,
    ) -> PartitionInputs:
        """在同一字节预算中装入显式关联的完整来源组，不假定代码相同就是实体关系。

        输入基金 A 的本体选择器及其明确 index_code 的成分选择器，输出完整联合分区。
        多个选择器共同计入 16MiB；任何一个组超限均拒绝，先前装入行不会被发布。
        """
        rows: dict[str, list[ResearchRow]] = {}
        used_bytes = 0
        for api, entity, code in selectors:
            for payload, size in self.records(
                api,
                entity=entity,
                code=code,
            ):
                used_bytes += size
                if used_bytes > maximum_bytes:
                    raise ValueError("research_input_partition_capacity_exceeded")
                rows.setdefault(api, []).append(decode_row(payload))
        return PartitionInputs(
            analysis_as_of=template.analysis_as_of,
            run_id=template.run_id,
            config=template.config,
            rows_by_api={api: tuple(values) for api, values in rows.items()},
            source_reasons=template.source_reasons,
            entity_source_reasons=template.entity_source_reasons,
            periods=PeriodIndex(self, template.record_periods),
            scope=template,
        )

    def statistics(self) -> dict[str, Any]:
        """返回实际暂存规模及有界缓存政策，便于运行摘要核验而非把暂存视为 observed。

        输入累计千行暂存返回完整行数与 JSON 字节数，包含全量处理后的历史释放差额。
        """
        count, retained_bytes = self.connection.execute(
            "SELECT count(*),coalesce(sum(bytes),0) FROM observations"
        ).fetchone()
        return {
            "storage": "sqlite_partitioned/v1",
            "rows": count,
            "retained_payload_bytes": retained_bytes,
            "written_payload_bytes": self.payload_bytes,
            "maximum_partition_bytes": MAX_PARTITION_BYTES,
            "maximum_spool_bytes": MAX_SPOOL_BYTES,
            "sqlite_cache_bytes": 8 * 1024 * 1024,
        }
