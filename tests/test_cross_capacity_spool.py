"""私有输入暂存的无损恢复、完整期间分区、显式容量拒绝和资源清理。"""

import sqlite3
from contextlib import nullcontext
from pathlib import Path

import pytest
from test_research_governance import A, B, inputs, row

from tushare_entitlement_hub.cross_asset_research import partitioned_inputs
from tushare_entitlement_hub.cross_asset_research.partitioned_inputs import (
    InputSpool,
    PeriodIndex,
    decode_row,
    encode_row,
)
from tushare_entitlement_hub.equity_research import key_index


@pytest.mark.parametrize("numeric", [False, True], ids=["text_only", "certified_numeric"])
def test_spool_row_encoding_round_trip_keeps_all_evidence(numeric: bool) -> None:
    """场景：纯文本行或携带数值证书的行；预期：原字段、实体、时间及证据无损往返。

    例如 close=3 元恢复后仍带原始事实证书，无实体的文本行仍为 None。
    """
    original = row(
        "fx_daily",
        "record",
        entity=A if numeric else None,
        raw={"ts_code": "FX", "name": "完整来源", "optional": None},
        values={"close": (3, "元")} if numeric else {},
    )
    restored = decode_row(encode_row(original))
    assert restored == original
    assert encode_row(restored) == encode_row(original)
    assert restored.availability == original.availability
    assert restored.numeric == original.numeric


def test_spool_records_use_exact_entity_and_code_filters() -> None:
    """场景：基金法律名称、指数代码和普通证券代码混合；预期：仅返回完整匹配的记录。

    例如名称含单引号时作为 SQL 参数处理，不匹配其他主体；空实体保持独立键。
    """
    records = [
        row("fund_company", "company", entity=B, raw={"name": "基金'公司"}),
        row("index_weight", "index", entity=B, raw={"index_code": "IDX"}),
        row("fx_daily", "first", entity=A, raw={"ts_code": "FX-A"}),
        row("fx_daily", "second", entity=B, raw={"ts_code": "FX-B"}),
        row("fx_daily", "empty", entity=None, raw={}),
    ]
    with InputSpool() as spool:
        for record in reversed(records):
            spool.append(record)
        assert [
            decode_row(value).raw["source_record_id"]
            for value, _ in spool.records("fx_daily", entity=A)
        ] == ["first"]
        assert [
            decode_row(value).raw["source_record_id"]
            for value, _ in spool.records("fund_company", code="基金'公司")
        ] == ["company"]
        assert [
            decode_row(value).raw["source_record_id"]
            for value, _ in spool.records("index_weight", entity=B, code="IDX")
        ] == ["index"]
        assert list(spool.records("fx_daily", entity=A, code="FX-B")) == []
        assert list(spool.records("absent")) == []
        assert list(spool.entities(())) == []
        assert list(spool.entities(("fx_daily", "index_weight", "fx_daily"))) == ["", A, B]
        assert spool.count("absent") == 0
        assert [record.raw["source_record_id"] for record in spool.rows("fx_daily")] == [
            "empty",
            "first",
            "second",
        ]


def test_spool_streams_multiple_batches_and_preserves_source_record_uniqueness() -> None:
    """场景：记录数超过 128 行，随后重复同来源记录键；预期：分批返回全部行，重复拒绝。

    例如 257 行跨三批返回；同 API 同记录键即使换实体也不得覆盖原行。
    """
    with InputSpool() as spool:
        for index in range(257):
            spool.append(row("fx_daily", f"record-{index:04d}", raw={"ts_code": "FX"}))
        expected_bytes = spool.payload_bytes
        assert len(list(spool.records("fx_daily"))) == 257
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            spool.append(row("fx_daily", "record-0000", entity=B))
        assert spool.count("fx_daily") == 257
        assert spool.payload_bytes == expected_bytes
        spool.append(row("fund_company", "record-0000", raw={"name": "管理人"}))
        assert spool.count("fund_company") == 1


@pytest.mark.parametrize("keep_versions", [False, True], ids=["latest_two_periods", "all_versions"])
@pytest.mark.parametrize("exceed_rows", [False, True], ids=["exact_row_limit", "one_row_over"])
def test_spool_retains_complete_period_members_and_enforces_source_budget(
    keep_versions: bool, exceed_rows: bool
) -> None:
    """场景：同一实体三期各 600 个成员；预期：期间完整保留，不按旧 512 行上限裁剪。

    例如普通来源保留最近两期的 1200 行，版本来源保留 1800 行；预算少一行则拒绝。
    """
    with InputSpool() as spool:
        for day in ("2026-10-01", "2026-10-02", "2026-10-03"):
            for index in range(600):
                spool.append(
                    row(
                        "dc_member",
                        f"{day}-{index}",
                        day=day,
                        raw={"ts_code": "THEME", "con_code": f"MEMBER{index}"},
                    )
                )
        written = spool.payload_bytes
        expected = 1800 if keep_versions else 1200
        if exceed_rows:
            with pytest.raises(ValueError, match="research_input_source_capacity_exceeded"):
                spool.retain("dc_member", keep_versions=keep_versions, maximum_rows=expected - 1)
        else:
            assert (
                spool.retain("dc_member", keep_versions=keep_versions, maximum_rows=expected)
                == expected
            )
        assert spool.count("dc_member") == expected
        periods = {record.availability.data_as_of for record in spool.rows("dc_member")}
        assert periods == (
            {"2026-10-01", "2026-10-02", "2026-10-03"}
            if keep_versions
            else {"2026-10-02", "2026-10-03"}
        )
        stats = spool.statistics()
        assert stats["rows"] == expected
        assert stats["written_payload_bytes"] == written
        if keep_versions:
            assert stats["retained_payload_bytes"] == written
        else:
            assert stats["retained_payload_bytes"] < written


def test_materialized_partition_preserves_global_period_index_and_scope() -> None:
    """场景：基金本体按实体读取，管理人和指数按显式代码关联；预期：联合分区完整且不串组。

    例如当前分区没有 FX-B，仍能通过全局日期索引查询它的原业务日期。
    """
    template = inputs(reasons={"fund_company": ("source_local_entity_only",)})
    records = [
        row("fx_daily", "first", entity=A, raw={"ts_code": "FX-A"}),
        row("fx_daily", "other", entity=B, raw={"ts_code": "FX-B"}, day="2026-10-02"),
        row("fund_company", "company", entity=B, day=None, raw={"name": "管理人"}),
        row("index_weight", "weight", entity=B, raw={"index_code": "IDX"}),
    ]
    with InputSpool() as spool:
        for record in records:
            spool.append(record)
        partition = spool.materialize(
            template,
            ("fx_daily", "fund_company", "index_weight"),
            entity=A,
            codes={"fund_company": "管理人", "index_weight": "IDX"},
        )
        assert {
            record.raw["source_record_id"]
            for values in partition.rows_by_api.values()
            for record in values
        } == {"first", "company", "weight"}
        assert partition.config is template.config
        assert partition.source_reasons == template.source_reasons
        assert partition.entity_source_reasons == template.entity_source_reasons
        assert partition.record_periods is partition.record_periods
        assert partition.record_periods[("fx_daily", "other")] == "2026-10-02"
        assert len(partition.record_periods) == 3
        assert list(partition.record_periods) == [
            ("fx_daily", "first"),
            ("fx_daily", "other"),
            ("index_weight", "weight"),
        ]
        for key in (("absent", "missing"), ("fund_company", "company")):
            with pytest.raises(KeyError):
                _ = partition.record_periods[key]
        assert isinstance(partition.periods, PeriodIndex)
        assert spool.materialize(template, ()).rows_by_api == {}
        assert len(spool.materialize(template, ("fx_daily",)).rows("fx_daily")) == 2


def test_partition_byte_limit_accepts_exact_size_and_rejects_partial_selection() -> None:
    """场景：两个显式关联来源共同使用分区预算；预期：恰好等于预算通过，少一字节拒绝。

    例如第一来源能放入但第二来源超限时，不返回只有第一来源的部分分区。
    """
    records = [row("fx_daily", "first"), row("index_weight", "second")]
    size = sum(len(encode_row(record).encode("utf-8")) for record in records)
    with InputSpool() as spool:
        for record in records:
            spool.append(record)
        selectors = (("fx_daily", A, None), ("index_weight", A, None))
        part = spool.materialize_selection(inputs(), selectors, maximum_bytes=size)
        assert sum(len(values) for values in part.rows_by_api.values()) == 2
        with pytest.raises(ValueError, match="research_input_partition_capacity_exceeded"):
            spool.materialize_selection(inputs(), selectors, maximum_bytes=size - 1)
        assert spool.count("fx_daily") == spool.count("index_weight") == 1


def test_period_index_preserves_old_support_without_inventing_snapshot_dates() -> None:
    """场景：旧来源日期作为旁证，暂存含更新日期和无日期快照；预期：新证据优先且不补假日期。

    例如旧 daily 日期可从 fallback 读取，fund_company 的无日期快照仍抛 KeyError。
    """
    with InputSpool() as spool:
        spool.append(row("fx_daily", "live"))
        spool.append(row("fund_company", "snapshot", day=None))
        spool.append(row("fina_mainbz", "support"))
        fallback = {
            ("fx_daily", "live"): "2020-01-01",
            ("daily", "history"): "2026-10-02",
            ("fund_company", "snapshot"): "2020-01-01",
        }
        index = PeriodIndex(spool, fallback)
        assert index[("fx_daily", "live")] == "2026-10-03"
        assert index[("daily", "history")] == "2026-10-02"
        assert len(index) == 3
        assert list(index) == [
            ("fina_mainbz", "support"),
            ("fx_daily", "live"),
            ("daily", "history"),
        ]
        with pytest.raises(KeyError):
            _ = index[("fund_company", "snapshot")]
        assert len(PeriodIndex(spool)) == 2
        assert spool.statistics()["rows"] == 3
        assert spool.count("fx_daily") + spool.count("fund_company") == 2
        assert spool.count("fina_mainbz") == 1


def test_spool_capacity_rejection_does_not_overwrite_previous_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """场景：显式磁盘载荷预算只容纳一条行；预期：第二条拒绝，第一条及累计字节不变。

    例如预算设为第一行实际 UTF-8 字节数，恰好等于预算的追加成功。
    """
    first = row("fx_daily", "first", raw={"name": "多字节"})
    size = len(encode_row(first).encode("utf-8"))
    monkeypatch.setattr(partitioned_inputs, "MAX_SPOOL_BYTES", size)
    with InputSpool() as spool:
        spool.append(first)
        with pytest.raises(ValueError, match="research_input_spool_capacity_exceeded"):
            spool.append(row("fx_daily", "second"))
        assert list(spool.rows("fx_daily")) == [first]
        assert spool.payload_bytes == size
        assert spool.connection.execute("PRAGMA cache_size").fetchone()[0] == -8192
        assert spool.connection.execute("PRAGMA mmap_size").fetchone()[0] == 0
        assert spool.statistics()["sqlite_cache_bytes"] == 8 * 1024**2


@pytest.mark.parametrize("fail", [False, True], ids=["normal_close", "exception_close"])
def test_spool_context_removes_private_database_on_every_exit(fail: bool) -> None:
    """场景：正常退出或业务处理抛错；预期：私有目录被清理，原业务异常继续传播。

    例如暂存一行后抛 RuntimeError，退出 with 后 SQLite 文件与目录均不存在。
    """
    guard = (
        pytest.raises(RuntimeError, match="synthetic-processing-failure") if fail else nullcontext()
    )
    with guard:
        with InputSpool() as spool:
            directory = Path(spool.temporary.name)
            spool.append(row("fx_daily"))
            assert directory.is_dir()
            if fail:
                raise RuntimeError("synthetic-processing-failure")
    assert not directory.exists()
    spool.close()


def test_spool_initialization_failure_closes_connection_and_removes_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """场景：SQLite 创建输入表时发生 I/O 错误；预期：构造失败也关闭连接并清理目录。

    例如 observations 建表失败后，不遗留父类已经创建的 keys 数据库。
    """
    original_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []
    directories: list[Path] = []

    class FailingConnection(sqlite3.Connection):
        """仅在输入表建表时注入数据库故障，其他操作使用真实 SQLite。"""

        def execute(self, sql, parameters=(), /):
            """执行普通 SQL，仅对输入建表报错；例如父类的 PRAGMA 仍正常执行。"""
            if sql.startswith("CREATE TABLE observations"):
                raise sqlite3.OperationalError("synthetic-schema-failure")
            return super().execute(sql, parameters)

    def connect(database, *args, **kwargs):
        """创建可注入故障的真实连接并记录私有目录；例如失败后可检查该目录已删除。"""
        connection = original_connect(database, *args, factory=FailingConnection, **kwargs)
        opened.append(connection)
        directories.append(Path(database).parent)
        return connection

    monkeypatch.setattr(key_index.sqlite3, "connect", connect)
    with pytest.raises(sqlite3.OperationalError, match="synthetic-schema-failure"):
        InputSpool()
    assert len(opened) == len(directories) == 1
    assert not directories[0].exists()
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")
