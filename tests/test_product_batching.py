"""产品初始化的离线批量性能与完整性回归，不使用真实数据或网络。"""

import json
from dataclasses import replace
from math import ceil
from pathlib import Path

import duckdb
import pytest
from test_product_foundation import NOW
from test_product_storage import bindings, reference_fixture
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.reference import initialize_library
from tushare_entitlement_hub.product_foundation.storage import ProductStore, validate_library
from tushare_entitlement_hub.reference_storage import file_checksum


def bulk_reference(root: Path, rows: int) -> Path:
    """用一条合成 accepted 行集合式生成多行，保持来源记录身份不同并同步账本。

    输入 rows=1000，输出仅包含 daily 的合成参考库路径；不下载或读取真实载荷。
    """
    source = reference_fixture(root)
    with duckdb.connect(str(source), config={"threads": 1, "memory_limit": "512MB"}) as conn:
        conn.execute(
            "CREATE OR REPLACE TABLE accepted.daily AS "
            "SELECT * EXCLUDE (i) REPLACE ('synthetic-' || i AS source_record_id) "
            "FROM accepted.daily CROSS JOIN range(?) generated(i)",
            [rows],
        )
        manifest = json.loads(conn.execute("SELECT manifest FROM catalog").fetchone()[0])
        manifest["quality"].update(source_rows=rows, accepted_rows=rows)
        conn.execute("UPDATE catalog SET manifest=?", [json.dumps(manifest)])
    return source


class QueryRecorder:
    """透传真实 DuckDB 操作，只统计调用次数和 SQL，不替换数据库行为。"""

    def __init__(self, connection, statements):
        """输入连接和共享 SQL 列表，输出透传包装器；例如 execute 后列表增加一项。"""
        self.connection = connection
        self.statements = statements

    def execute(self, query, *args, **kwargs):
        """记录后真实执行；输入 SELECT 1，输出自身，fetchone 仍取真实结果。"""
        self.statements.append(query)
        self.connection.execute(query, *args, **kwargs)
        return self

    def sql(self, query, *args, **kwargs):
        """记录关系查询并返回真实惰性结果；输入 SELECT，输出可 fetchmany 的 relation。"""
        self.statements.append(query)
        return self.connection.sql(query, *args, **kwargs)

    def __getattr__(self, name):
        """将 fetchmany、close 等交还连接；输入 fetchone，输出真实绑定方法。"""
        return getattr(self.connection, name)

    def __enter__(self):
        """透传上下文进入；输入 with，输出包装器以继续记录 SQL。"""
        self.connection.__enter__()
        return self

    def __exit__(self, *args):
        """透传退出及提交/回滚；正常退出返回底层结果，保留原有事务语义。"""
        return self.connection.__exit__(*args)


def record_queries(monkeypatch):
    """包装本轮新建连接并返回共享计数列表；输入 monkeypatch，输出空 SQL 列表。"""
    statements = []
    connect = duckdb.connect

    def recording_connect(*args, **kwargs):
        """用原始参数建立真实连接；输入数据库路径，输出可计数连接。"""
        return QueryRecorder(connect(*args, **kwargs), statements)

    monkeypatch.setattr(duckdb, "connect", recording_connect)
    return statements


def test_unbound_initialization_audits_without_decoding_rows(tmp_path, monkeypatch):
    """场景：无绑定且含大量来源行；预期：零逐行解码、206覆盖及三项明确缺失原因。

    输入一万行合成 daily，输出可校验的 incomplete 产品库，事实表为空且源文件不变。
    """
    source = bulk_reference(tmp_path, 10000)
    before = file_checksum(source)

    def reject_row_mapping(*args, **kwargs):
        """阻止无绑定时进入逐行路径；任意调用都抛断言，预期从不调用。"""
        raise AssertionError("无绑定不应逐行解析时间事实")

    monkeypatch.setattr(
        "tushare_entitlement_hub.product_foundation.reference_scan.map_availability",
        reject_row_mapping,
    )
    output = tmp_path / "products.duckdb"
    result = initialize_library(source, output, NOW, FoundationPolicy(version="batch-v1"))
    assert result["counts"]["availability_fact"] == 0
    assert result["counts"]["typed_fact"] == 0
    assert result["counts"]["coverage_slice"] == 206
    assert result["counts"]["quality_gate_result"] == 206
    assert result["processing"]["scanned_rows"] == 0
    assert result["processing"]["source_batches"] == 0
    assert result["processing"]["audit_only_apis"] == 206
    with ProductStore(output, read_only=True) as store:
        coverage = store.records("coverage_slice")
        assert all(
            {"entity_mapping_missing", "time_mapping_missing", "unit_certification_missing"}
            <= set(item.reason_codes)
            for item in coverage
        )
        assert next(item for item in coverage if item.source_api == "daily").accepted_rows == 10000
    assert validate_library(source, output)["valid"]
    assert file_checksum(source) == before


def test_bound_initialization_uses_batch_sql_calls(tmp_path, monkeypatch):
    """场景：显式绑定一批来源；预期：事实、血缘与覆盖写入不按每行发起 SQL。

    输入64行及完整绑定，输出64条时间事实，SQL总调用少于200且库可完整校验。
    """
    source = bulk_reference(tmp_path, 64)
    statements = record_queries(monkeypatch)
    output = tmp_path / "products.duckdb"
    summary = initialize_library(
        source, output, NOW, FoundationPolicy(version="batch-v1"), bindings()
    )
    assert summary["counts"]["availability_fact"] == 64
    assert len(statements) < 200, f"逐行 SQL 调用共 {len(statements)} 次"
    assert validate_library(source, output)["valid"]


def test_cli_reports_progress_separately_from_final_json(tmp_path):
    """场景：CLI初始化无绑定来源；预期：stderr含阶段进度，stdout仍为单一最终JSON。

    输入离线命令，输出开始、接口审计和完成事件，结束时 processed_apis=206。
    """
    source = reference_fixture(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "product-foundation",
            "initialize",
            "--reference",
            str(source),
            "--output",
            str(tmp_path / "products.duckdb"),
            "--as-of",
            NOW,
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["registered_apis"] == 206
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert events and events[0]["stage"] == "start"
    assert events[-1]["stage"] == "complete"
    assert events[-1]["processed_apis"] == 206


@pytest.mark.parametrize("rows", [1025, 4097])
def test_large_initialization_and_validation_scale_with_batches(tmp_path, monkeypatch, rows):
    """场景：跨越多个512行批次的配置来源；预期：写入和校验SQL随批次增长。

    输入1025或4097行，输出全部时间事实与血缘；SQL上限按ceil(rows/512)计算。
    断言真实数据库调用量和最终行数，不使用运行时间阈值，也不模拟被测写入逻辑。
    """
    source = bulk_reference(tmp_path, rows)
    statements = record_queries(monkeypatch)
    output = tmp_path / "products.duckdb"
    events = []
    policy = FoundationPolicy(version="batch-v1", fact_batch_size=512)
    result = initialize_library(source, output, NOW, policy, bindings(), progress=events.append)
    initialization_calls = len(statements)
    assert initialization_calls < 100 + 70 * ceil(rows / 512)
    assert result["counts"]["availability_fact"] == rows
    assert result["processing"]["scanned_rows"] == rows
    assert result["processing"]["source_batches"] == ceil(rows / 512)
    batch_events = [event for event in events if event["stage"] == "source_batch"]
    assert len(batch_events) == ceil(rows / 512)
    assert batch_events[-1]["scanned_rows"] == rows
    statements.clear()
    assert validate_library(source, output)["valid"]
    validation_calls = len(statements)
    assert validation_calls < 100 + 40 * ceil(rows / 512)
    assert not any("WHERE record_id=?" in query for query in statements)
    statements.clear()
    assert initialize_library(source, output, NOW, policy, bindings()) == result
    assert len(statements) < 120 + 40 * ceil(rows / 512)
    print(
        f"rows={rows} batches={ceil(rows / 512)} "
        f"initialize_sql={initialization_calls} validate_sql={validation_calls}"
    )


@pytest.mark.parametrize("batch_size", [1, 2, 512])
def test_bulk_store_preserves_conflicts_order_and_transactions(tmp_path, batch_size):
    """场景：同批重复、跨批重入及冲突；预期：身份顺序稳定、不可变且整次事务回滚。

    输入A、B、A三个实体，输出三身份但仅两行；后续批次异值冲突回滚此前新行。
    """
    path = tmp_path / "products.duckdb"
    first = bindings().entities[0]
    second = replace(first, identity=replace(first.identity, local_id="B.SZ"))
    with ProductStore(path) as store:
        identities = store.put_many("entity", (first, second, first), batch_size=batch_size)
        assert identities[0] == identities[2] and identities[0] != identities[1]
        assert store.counts()["entity"] == 2
        assert store.put_many("entity", (first, second, first), batch_size=batch_size) == identities
        assert store.put_many("entity", (), batch_size=batch_size) == []
    third = replace(first, identity=replace(first.identity, local_id="C.SZ"))
    with pytest.raises(ValueError, match="immutable_record_conflict"), ProductStore(path) as store:
        store.put_many("entity", [third, replace(first, name="冲突")], batch_size=batch_size)
    with ProductStore(path, read_only=True) as store:
        assert store.counts()["entity"] == 2
        store.verify_contents(batch_size=batch_size)


def test_bulk_store_rejects_in_batch_conflicts_and_invalid_input(tmp_path):
    """场景：批内同键异值、错模型或无效批大小；预期：校验拒绝且不写入事实。

    输入A与改名A，输出immutable_record_conflict，事务回滚后entity仍为空。
    """
    first = bindings().entities[0]
    with ProductStore(tmp_path / "products.duckdb") as store:
        with pytest.raises(ValueError, match="immutable_record_conflict"):
            store.put_many("entity", [first, replace(first, name="另一名称")])
        for size in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="invalid_batch_size"):
                store.put_many("entity", [first], batch_size=size)
            with pytest.raises(ValueError, match="invalid_batch_size"):
                store.verify_contents(batch_size=size)
        with pytest.raises(ValueError, match="product_table_model_mismatch"):
            store.put_many("typed_fact", [first])
        assert store.counts()["entity"] == 0


def test_bulk_lineage_conflicts_prevent_partial_fact_insertion(tmp_path):
    """场景：事实尚未写入但对应血缘同键异值；预期：先检测冲突，不留下无血缘事实。

    输入预存错误contract_hash的证据边，再批量写入评分分量，输出冲突且分量表为空。
    """
    from test_product_foundation import evidence

    from tushare_entitlement_hub.product_foundation.contracts import FeatureContribution
    from tushare_entitlement_hub.product_foundation.storage_records import (
        LineageEdge,
        ScoreComponent,
    )
    from tushare_entitlement_hub.store import digest

    component = FeatureContribution(
        feature_id="sample",
        raw_value=1,
        unit="元",
        transformed_value=1,
        weight=1,
        contribution=100,
        evidence_refs=(evidence(),),
    )
    record = ScoreComponent(result_id="result", component=component)
    identity = digest(["result", "sample"])
    edge = LineageEdge(
        output_id=identity,
        source_api="daily",
        source_record_id="row1",
        fields=evidence().fields,
        contract_hash="different",
    )
    with ProductStore(tmp_path / "products.duckdb") as store:
        store.put_many("lineage_edge", [edge])
        with pytest.raises(ValueError, match="immutable_record_conflict"):
            store.put_many("score_component", [record])
        assert store.counts()["score_component"] == 0


def test_interrupted_schema_only_output_can_be_reinitialized(tmp_path):
    """场景：中断后仅有schema_version和空表；预期：重跑快速审计即可生成可校验快照。

    输入空产品库及未绑定来源，输出206覆盖、0时间事实，并在提交后才发送complete。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    with ProductStore(output):
        pass
    events = []

    def on_progress(event):
        """采集阶段，并在完成时重开只读库；输入complete，预期摘要已提交。"""
        events.append(event)
        if event["stage"] == "complete":
            with ProductStore(output, read_only=True) as store:
                assert store.metadata("summary")["counts"]["coverage_slice"] == 206

    result = initialize_library(
        source, output, NOW, FoundationPolicy(version="v1"), progress=on_progress
    )
    assert result["processing"]["scanned_rows"] == 0
    assert events[-1]["stage"] == "complete"
    assert validate_library(source, output)["valid"]


def test_model_type_metadata_is_cached_without_caching_values(monkeypatch):
    """场景：批量创建同类严格模型；预期：类型定义最多解析一次，非法值仍逐条拒绝。

    输入100个合法Coverage和负计数，输出最多一次类型解析及invalid_count异常。
    """
    from tushare_entitlement_hub.product_foundation import model
    from tushare_entitlement_hub.product_foundation.contracts import Coverage

    calls = []
    original = model.get_type_hints

    def count_hints(*args, **kwargs):
        """计数并真实解析类型；输入模型类型，输出原始字段定义。"""
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "get_type_hints", count_hints)
    for index in range(100):
        assert Coverage(observed_count=index).observed_count == index
    with pytest.raises(ValueError, match="invalid_count"):
        Coverage(observed_count=-1)
    assert calls.count(Coverage) <= 1


def test_unbound_million_row_reference_with_full_catalog(tmp_path, monkeypatch):
    """场景：251目录项、206 accepted表和1452191行；预期：只生成206覆盖审计。

    输入按正式库规模生成的合成库，输出零逐行事实、所有接口可审计且最终库校验通过。
    一百多万行集中在合成daily，其余表显式声明空；不代表真实数据质量或分布验证。
    """
    from tushare_entitlement_hub.catalog import load_policies
    from tushare_entitlement_hub.product_foundation.registry import load_registry
    from tushare_entitlement_hub.reference_contract import load_reference_contracts

    source = bulk_reference(tmp_path, 1452191)
    entries = load_registry()
    contracts = load_reference_contracts(load_policies())
    metadata = {
        "source_api",
        "source_record_id",
        "business_group_key",
        "contract_hash",
        "observed_at",
        "quality_tier",
        "standard_units",
        "historical_pit_certified",
    }
    with duckdb.connect(str(source), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        for entry in entries:
            if entry.api == "daily":
                continue
            columns = ",".join(
                f'"{name}" VARCHAR'
                for name in sorted(metadata | set(entry.business_key) | set(entry.selected_fields))
            )
            connection.execute(f'CREATE TABLE accepted."{entry.api}" ({columns})')
            manifest = {
                "contract_hash": contracts[entry.api].fingerprint,
                "observed_at": NOW,
                "collection_status": "complete_declared_request",
                "parameters": {},
                "quality": {
                    "source_rows": 0,
                    "accepted_rows": 0,
                    "quarantined_rows": 0,
                    "exact_duplicate_rows": 0,
                },
            }
            connection.execute(
                "INSERT INTO catalog VALUES (?, 'synthetic', ?)",
                [
                    entry.api,
                    json.dumps(manifest),
                ],
            )
        connection.execute(
            "INSERT INTO catalog SELECT 'extra_' || i, 'archived', '{}'::JSON FROM range(45) t(i)"
        )
        assert connection.execute("SELECT count(*) FROM catalog").fetchone()[0] == 251
        assert (
            connection.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema='accepted'"
            ).fetchone()[0]
            == 206
        )
    queries = record_queries(monkeypatch)
    output = tmp_path / "products.duckdb"
    result = initialize_library(source, output, NOW, FoundationPolicy(version="scale-v1"))
    assert result["registered_apis"] == result["readable_apis"] == 206
    assert result["processing"]["scanned_rows"] == 0
    assert result["counts"]["availability_fact"] == result["counts"]["typed_fact"] == 0
    assert result["counts"]["coverage_slice"] == result["counts"]["quality_gate_result"] == 206
    assert not any("to_json(source_row)" in query for query in queries)
    assert len(queries) < 800
    print(f"synthetic_rows=1452191 catalog=251 accepted=206 audit_sql={len(queries)}")
    assert validate_library(source, output)["valid"]


def test_progress_interruption_rolls_back_all_processed_batches(tmp_path):
    """场景：已写入部分批次后通过回调模拟中止；预期：不留下部分事实或完成事件。

    输入10行、每批2行且第二批后抛错，输出异常，重开产品库仍只有空表和schema版本。
    """
    source = bulk_reference(tmp_path, 10)
    output = tmp_path / "products.duckdb"
    events = []

    def interrupt(event):
        """记录后在第二批中止；输入source_batches=2，输出模拟异常。"""
        events.append(event)
        if event["stage"] == "source_batch" and event["source_batches"] == 2:
            raise RuntimeError("synthetic_interruption")

    with pytest.raises(RuntimeError, match="synthetic_interruption"):
        initialize_library(
            source,
            output,
            NOW,
            FoundationPolicy(version="v1", fact_batch_size=2),
            bindings(),
            progress=interrupt,
        )
    assert not any(event["stage"] == "complete" for event in events)
    with ProductStore(output, read_only=True) as store:
        assert sum(store.counts().values()) == 0
        assert store.metadata("run_id") is None


def test_bulk_writer_revalidates_nested_model_values(tmp_path):
    """场景：调用方绕过冻结模型篡改嵌套值；预期：批量写入仍重新校验而非信任缓存。

    输入confidence被改为NaN的别名，输出模型错误，别名表无新增记录。
    """
    alias = bindings().aliases[0]
    object.__setattr__(alias, "confidence", float("nan"))
    with ProductStore(tmp_path / "products.duckdb") as store:
        with pytest.raises(ValueError):
            store.put_many("entity_alias", [alias])
        assert store.counts()["entity_alias"] == 0


def test_validation_reaches_records_after_first_engine_vector(tmp_path):
    """场景：篡改第2048条之后的分量投影；预期：流式校验不能因辅助查询截断遍历。

    输入2050条带血缘分量，修改最后一条的SQL贡献，输出product_projection_mismatch。
    """
    from test_product_foundation import evidence

    from tushare_entitlement_hub.product_foundation.contracts import FeatureContribution
    from tushare_entitlement_hub.product_foundation.storage_records import ScoreComponent

    component = FeatureContribution(
        feature_id="sample",
        raw_value=1,
        unit="元",
        transformed_value=1,
        weight=1,
        contribution=100,
        evidence_refs=(evidence(),),
    )
    path = tmp_path / "products.duckdb"
    with ProductStore(path) as store:
        store.put_many(
            "score_component",
            (
                ScoreComponent(result_id=f"result-{index}", component=component)
                for index in range(2050)
            ),
        )
        store.connection.execute(
            "UPDATE product.score_component SET contribution=99 WHERE result_id='result-2049'"
        )
    with ProductStore(path, read_only=True) as store:
        with pytest.raises(ValueError, match="product_projection_mismatch"):
            store.verify_contents()


def test_validation_rejects_premature_end_of_stream(tmp_path):
    """场景：读取器提前结束但物理表仍有行；预期：计数防线明确拒绝不完整校验。

    输入一个实体及返回空结果的读取器，输出product_scan_incomplete，不冒充valid。
    """

    class EmptyResult:
        """模拟读取结果被提前关闭，数据库内容保持真实不变。"""

        def fetchmany(self, size):
            """输入批大小，输出空列表以模拟意外中断。"""
            return []

    class InterruptedReader(QueryRecorder):
        """保留所有真实查询，仅让流式读取提前结束。"""

        def sql(self, query, *args, **kwargs):
            """输入表的流式查询，输出提前结束的结果。"""
            return EmptyResult()

    with ProductStore(tmp_path / "products.duckdb") as store:
        store.put_many("entity", bindings().entities)
        store.connection = InterruptedReader(store.connection, [])
        with pytest.raises(ValueError, match="product_scan_incomplete"):
            store.verify_contents()


def test_scan_without_mapping_and_invalid_batch_size_fail_closed(tmp_path):
    """场景：直接调用扫描器但缺实体映射，或给读取器非法批大小；预期：不读取行。

    输入无映射返回entity_mapping_missing；输入batch_size=0抛invalid_batch_size。
    """
    from tushare_entitlement_hub.catalog import load_policies
    from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings
    from tushare_entitlement_hub.product_foundation.reference_scan import (
        consume_rows,
        reference_rows,
    )
    from tushare_entitlement_hub.product_foundation.registry import load_registry
    from tushare_entitlement_hub.reference_contract import load_reference_contracts

    with duckdb.connect(config={"threads": 1, "memory_limit": "512MB"}) as source:
        entry = next(item for item in load_registry() if item.api == "daily")
        contract = load_reference_contracts(load_policies())["daily"]
        with ProductStore(tmp_path / "products.duckdb") as store:
            assert consume_rows(
                source, store, entry, contract, FoundationBindings(), NOW, "run"
            ) == (
                {"entity_mapping_missing"},
                None,
            )
        with pytest.raises(ValueError, match="invalid_batch_size"):
            list(reference_rows(source, "daily", batch_size=0))
    with pytest.raises(ValueError, match="invalid_foundation_policy"):
        FoundationPolicy(version="v1", fact_batch_size=0)
