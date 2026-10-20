"""研究存储与依赖流水线的真实离线数据库测试。"""

import json
from dataclasses import replace

import duckdb
import pytest
from research_fixture import AS_OF, DATABASE_CONFIG, write_reference

from tushare_entitlement_hub.equity_research import pipeline
from tushare_entitlement_hub.equity_research.common import gate, make_result, metric
from tushare_entitlement_hub.equity_research.models import (
    MARKET_ENTITY,
    PRODUCT_IDS,
    ResearchConfig,
    ResearchInputs,
    ResearchResult,
)
from tushare_entitlement_hub.equity_research.pipeline import run_research, select_products
from tushare_entitlement_hub.equity_research.storage import (
    fingerprints,
    initialize_tables,
    insert_rows,
    record_rows,
    run_identity,
    save_results,
    stored_rows,
    validate_research,
    verify_run,
)
from tushare_entitlement_hub.product_foundation.contracts import FeatureContribution, Truth
from tushare_entitlement_hub.reference_storage import file_checksum
from tushare_entitlement_hub.store import encode


@pytest.fixture
def libraries(tmp_path):
    """提供真实合成参考库与独立输出路径；例：32只股票、11类来源，全部可离线读取。"""
    reference, config = write_reference(tmp_path, 2)
    return reference, tmp_path / "products.duckdb", config


def read_metadata(connection, run_id):
    """读取单个运行元数据用于定点篡改；例：合法run_id得到配置、输入指纹和摘要。"""
    return json.loads(
        connection.execute("SELECT payload FROM research.run WHERE run_id=?", [run_id]).fetchone()[
            0
        ]
    )


def test_real_synthetic_pipeline_metrics_idempotency_and_dependencies(tmp_path):
    """场景：32只合成股运行六产品；预期：公式有值、引用完整、重复运行完全幂等。"""
    reference, config = write_reference(tmp_path, 32)
    output = tmp_path / "products.duckdb"
    before = file_checksum(reference)
    progress = []
    first = run_research(reference, output, AS_OF, config, progress=progress.append)
    assert first["result_count"] >= 32 * 4 + 1
    assert first["products"] == list(PRODUCT_IDS)
    assert first["processing"]["scanned_rows"] == 32 * 15
    assert [event["stage"] for event in progress][-1] == "complete"
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        values = [
            ResearchResult.from_json(row[0])
            for row in connection.execute("SELECT payload FROM research.result").fetchall()
        ]
        market = next(item for item in values if item.envelope.product_id == PRODUCT_IDS[0])
        assert market.metric("breadth") == 0.5
        assert market.details["denominator_count"] == 32
        finance = next(
            item
            for item in values
            if item.envelope.product_id == PRODUCT_IDS[1]
            and item.envelope.entity_id == "cn:security:S0000:fixture"
        )
        assert finance.metric("cash_margin") == pytest.approx(0.2)
        assert finance.metric("leverage") == pytest.approx(0.4)
        assert finance.metric("cash_cover") == 2
        assert finance.metric("Q0") is not None
        candidates = [item for item in values if item.envelope.product_id == PRODUCT_IDS[4]]
        assert len(candidates) == 32
        assert all(item.envelope.rank is None for item in candidates)
        assert all(item.envelope.feature_profile.startswith("right_side/") for item in candidates)
        assert verify_run(connection, read_metadata(connection, first["run_id"])) == first
    replay = []
    assert run_research(reference, output, AS_OF, config, progress=replay.append) == first
    assert [event["stage"] for event in replay if not event["stage"].endswith("-progress")] == [
        "start",
        "verify-start",
        "verify-complete",
        "reused",
    ]
    assert {event["operation"] for event in replay if event["stage"] == "verify-progress"} == {
        "primary_keys",
        "rebuild",
        "physical_projections",
        "summary",
    }
    assert run_research(reference, output, AS_OF, config) == first
    assert validate_research(reference, output)["valid"]
    assert before == file_checksum(reference)
    selected = run_research(reference, output, AS_OF, config, ("P03",))
    assert selected["products"] == list(PRODUCT_IDS[1:5])
    assert selected["run_id"] != first["run_id"]
    assert validate_research(reference, output)["valid"]
    for retired in ("P03S", "P03M", "P03L"):
        with pytest.raises(ValueError, match="unknown_research_product"):
            run_research(reference, output, AS_OF, config, (retired,))


def test_empty_reference_and_missing_bindings_remain_incomplete(tmp_path):
    """场景：空参考表和默认缺绑定配置；预期：每个产品保留诚实降级记录而非伪成功。"""
    reference, _ = write_reference(tmp_path, 0)
    output = tmp_path / "empty-products.duckdb"
    summary = run_research(reference, output, AS_OF)
    assert summary["status"] == "incomplete" and summary["result_count"] == 6
    assert summary["processing"]["scanned_rows"] == 0
    assert validate_research(reference, output)["valid"]
    assert not summary["historical_pit_certified"]


@pytest.mark.parametrize(
    "selection,expected",
    [
        ((), PRODUCT_IDS),
        (("P01",), PRODUCT_IDS[:1]),
        (("P03",), PRODUCT_IDS[1:5]),
        (("P13",), PRODUCT_IDS[1:4] + PRODUCT_IDS[5:]),
        (("P04", "P04"), PRODUCT_IDS[1:2]),
        ((PRODUCT_IDS[0],), PRODUCT_IDS[:1]),
    ],
)
def test_selection_is_topological_and_deduplicated(selection, expected):
    """场景：别名、正式名称和重复产品选择；预期：依赖闭包与稳定拓扑顺序一致。"""
    requested, expanded = select_products(selection)
    assert expanded == expected
    assert requested == tuple(sorted(set(requested)))


@pytest.mark.parametrize("scenario", ["missing", "collision", "hardlink", "time", "product"])
def test_run_rejects_invalid_inputs_without_overwriting_reference(libraries, scenario):
    """场景：缺库、同文件/硬链接、非法时点或产品；预期：失败且参考库指纹不变。"""
    reference, output, config = libraries
    before = file_checksum(reference)
    if scenario == "missing":
        reference = reference.with_name("absent.duckdb")
    elif scenario == "collision":
        output = reference
    elif scenario == "hardlink":
        output.hardlink_to(reference)
    with pytest.raises(ValueError):
        run_research(
            reference,
            output,
            "bad-time" if scenario == "time" else AS_OF,
            config,
            ("P99",) if scenario == "product" else (),
        )
    if reference.exists():
        assert file_checksum(reference) == before


@pytest.mark.parametrize("scenario", ["catalog", "field", "table"])
def test_reference_contract_damage_is_visible(libraries, scenario):
    """场景：目录/来源表/登记字段缺失；预期：目录失败或产品明确降级，绝不补造广度。"""
    reference, output, config = libraries
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        if scenario == "catalog":
            connection.execute("DROP TABLE catalog")
        elif scenario == "field":
            connection.execute("ALTER TABLE accepted.daily DROP COLUMN close")
        else:
            connection.execute("DROP TABLE accepted.daily")
    if scenario == "catalog":
        with pytest.raises(ValueError, match="reference_catalog_missing"):
            run_research(reference, output, AS_OF, config)
    else:
        summary = run_research(reference, output, AS_OF, config, ("P01",))
        assert summary["status"] == "incomplete"
        with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
            result = ResearchResult.from_json(
                connection.execute("SELECT payload FROM research.result").fetchone()[0]
            )
            assert result.metric("breadth") is None


def test_read_change_and_transaction_failure_leave_no_partial_results(libraries, monkeypatch):
    """场景：扫描时输入改变或写事务后校验失败；预期：拒绝快照并回滚所有本次写入。"""
    reference, output, config = libraries
    original_checksum = pipeline.file_checksum
    checks = iter(["first", "changed"])
    monkeypatch.setattr(pipeline, "file_checksum", lambda _: next(checks))
    with pytest.raises(ValueError, match="reference_changed_during_read"):
        run_research(reference, output, AS_OF, config, ("P01",))
    monkeypatch.setattr(pipeline, "file_checksum", original_checksum)

    def reject_saved_run(*args, **kwargs):
        """模拟事务内校验失败；例：写入后抛出ValueError，验证ROLLBACK的实际数据状态。"""
        raise ValueError("synthetic_validation_failure")

    monkeypatch.setattr(pipeline, "verify_run", reject_saved_run)
    with pytest.raises(ValueError, match="research_verify_validation_failed"):
        run_research(reference, output, AS_OF, config, ("P01",))
    assert not output.exists()
    assert not list(output.parent.glob(".*.stage-*"))


@pytest.mark.parametrize("count", [0, 1, 511, 512, 513, 1024])
def test_batched_persistence_keeps_every_result_and_tail(tmp_path, count):
    """场景：批大小512的前后边界；预期：每个结果唯一保存，尾批与多批均无漏行。"""
    context = ResearchInputs(analysis_as_of=AS_OF, run_id="synthetic-run", config=ResearchConfig())
    results = [
        make_result(
            context,
            PRODUCT_IDS[0],
            f"cn:security:X{index}:fixture",
            gates=(gate("known", Truth.UNKNOWN, "input_missing"),),
            predicate_results={"predicate": Truth.UNKNOWN},
        )
        for index in range(count)
    ]
    with duckdb.connect(str(tmp_path / "batch.duckdb"), config=DATABASE_CONFIG) as connection:
        initialize_tables(connection)
        initialize_tables(connection)
        counts = save_results(connection, results, 512)
        assert counts["result"] == counts["gate"] == counts["signal"] == count
        assert fingerprints(stored_rows(connection, "synthetic-run", 512)) == fingerprints(
            entry for result in results for entry in record_rows(result)
        )
        with pytest.raises(ValueError, match="unknown_research_table"):
            insert_rows(connection, "result; DROP SCHEMA research", [])
        if count:
            with pytest.raises(duckdb.ConstraintException):
                save_results(connection, results[:1], 512)


def test_record_projection_preserves_shared_evidence_and_parent(libraries):
    """场景：正反证共享一引用并有上游结果；预期：证据去重、两种关系和parent关系均保留。"""
    _, _, config = libraries
    reference = config.foundation.certifications[0].evidence_refs[0]
    context = ResearchInputs(analysis_as_of=AS_OF, run_id="synthetic-run", config=config)
    contribution = FeatureContribution(
        feature_id="value",
        raw_value=1,
        unit="ratio",
        transformed_value=0.5,
        weight=1,
        contribution=50,
        transform="identity",
        evidence_refs=(reference,),
    )
    result = make_result(
        context,
        PRODUCT_IDS[0],
        MARKET_ENTITY,
        metrics=(metric("value", 1, "ratio", (reference,)),),
        contributions=(contribution,),
        references=(reference,),
        contrary=(reference,),
        parents=("parent-id",),
        predicate_results={"truth": Truth.UNKNOWN},
    )
    rows = list(record_rows(result))
    assert sum(table == "evidence" for table, _ in rows) == 1
    assert {row["payload"]["relationship"] for table, row in rows if table == "lineage"} == {
        "used",
        "contrary",
        "parent",
    }
    assert fingerprints(rows) == fingerprints(reversed(rows))
    assert run_identity("hash", AS_OF, config, ("P01", "P04")) == run_identity(
        "hash", AS_OF, config, ("P04", "P01")
    )
    assert run_identity("hash", AS_OF, config, ()) != run_identity(
        "hash", AS_OF, replace(config, maximum_source_rows=333), ()
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "identity",
        "current",
        "reference",
        "run_key",
        "missing_runs",
        "orphan_run",
        "orphan_result",
        "parent",
        "unknown",
        "number",
        "json",
        "context",
        "projection",
        "counts",
        "primary_key",
    ],
)
def test_validation_rejects_tampering(libraries, scenario):
    """场景：运行键、来源、JSON、数值、三值、主键或引用被篡改；预期：统一校验失败。"""
    reference, output, config = libraries
    summary = run_research(reference, output, AS_OF, config, ("P03",))
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        metadata = read_metadata(connection, summary["run_id"])
        if scenario in {"identity", "counts"}:
            if scenario == "identity":
                metadata["rule_version"] = "invalid"
            else:
                metadata["summary"]["counts"]["result"] += 1
            connection.execute("UPDATE research.run SET payload=?", [encode(metadata)])
        elif scenario == "current":
            connection.execute("DELETE FROM research.metadata")
        elif scenario == "reference":
            metadata["reference_sha256"] = "other"
            connection.execute("UPDATE research.run SET payload=?", [encode(metadata)])
        elif scenario == "run_key":
            connection.execute("UPDATE research.run SET run_id='wrong'")
        elif scenario == "missing_runs":
            connection.execute("DELETE FROM research.run")
        elif scenario == "orphan_run":
            connection.execute(
                "INSERT INTO research.feature VALUES ('foreign','absent','absent','{}')"
            )
        elif scenario == "orphan_result":
            connection.execute("UPDATE research.feature SET result_id='absent'")
        elif scenario == "parent":
            connection.execute(
                "UPDATE research.lineage SET payload=? WHERE payload->>'relationship'='parent'",
                [encode({"relationship": "parent", "parent_result_id": "absent"})],
            )
        elif scenario == "unknown":
            connection.execute(
                "UPDATE research.gate SET payload=json_merge_patch(payload, ?)",
                [encode({"result": "false"})],
            )
        elif scenario == "projection":
            connection.execute(
                "DELETE FROM research.evidence WHERE record_id="
                "(SELECT record_id FROM research.evidence LIMIT 1)"
            )
        elif scenario == "primary_key":
            connection.execute("CREATE TABLE research.duplicate AS SELECT * FROM research.feature")
            connection.execute("DROP TABLE research.feature")
            connection.execute("ALTER TABLE research.duplicate RENAME TO feature")
            connection.execute(
                "INSERT INTO research.feature SELECT * FROM research.feature LIMIT 1"
            )
        else:
            row_id, text = connection.execute(
                "SELECT record_id,payload FROM research.result LIMIT 1"
            ).fetchone()
            payload = json.loads(text)
            if scenario == "json":
                payload = {"unexpected": "structure"}
            elif scenario == "context":
                payload["envelope"]["model_version"] = "other-version"
            else:
                payload["envelope"]["score"] = 101
            connection.execute(
                "UPDATE research.result SET payload=? WHERE record_id=?", [encode(payload), row_id]
            )
    checked = validate_research(reference, output)
    assert checked == {
        "valid": False,
        "reason_codes": ["research_validation_failed"],
        "historical_pit_certified": False,
    }


def test_verify_batch_size_and_validate_missing_paths(libraries):
    """场景：非法校验批大小、同库校验和缺库；预期：明确拒绝，不创建缺失产品库。"""
    reference, output, config = libraries
    summary = run_research(reference, output, AS_OF, config, ("P01",))
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        metadata = read_metadata(connection, summary["run_id"])
        for invalid in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="invalid_batch_size"):
                verify_run(connection, metadata, batch_size=invalid)
    assert not validate_research(reference, reference)["valid"]
    missing = output.with_name("missing.duckdb")
    assert not validate_research(reference, missing)["valid"]
    assert not missing.exists()


@pytest.mark.parametrize(
    "scenario",
    [
        "empty_selection",
        "duplicate_selection",
        "unknown_selection",
        "wrong_products",
        "summary_status",
        "summary_count",
        "processing_negative",
        "processing_balance",
        "result_run",
        "result_product",
        "result_time",
    ],
)
def test_verify_metadata_and_result_context_fail_closed(libraries, scenario):
    """场景：派生摘要、产品范围、行处理账本或结果时点损坏；预期：校验拒绝不一致批次。"""
    reference, output, config = libraries
    summary = run_research(reference, output, AS_OF, config, ("P01",))
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        metadata = read_metadata(connection, summary["run_id"])
        if scenario == "empty_selection":
            metadata["selection"] = []
        elif scenario == "duplicate_selection":
            metadata["selection"] *= 2
        elif scenario == "unknown_selection":
            metadata["selection"] = ["P99"]
        elif scenario == "wrong_products":
            metadata["products"] = []
        elif scenario == "summary_status":
            metadata["summary"]["status"] = "incomplete"
        elif scenario == "summary_count":
            metadata["summary"]["result_count"] = 0
        elif scenario == "processing_negative":
            metadata["summary"]["processing"]["batches"] = -1
        elif scenario == "processing_balance":
            metadata["summary"]["processing"]["scanned_rows"] += 1
        else:
            payload = json.loads(
                connection.execute("SELECT payload FROM research.result").fetchone()[0]
            )
            field, value = {
                "result_run": ("run_id", "different-run"),
                "result_product": ("product_id", PRODUCT_IDS[1]),
                "result_time": ("analysis_as_of", "2026-10-04T12:00:00Z"),
            }[scenario]
            payload["envelope"][field] = value
            if scenario == "result_time":
                payload["envelope"]["generated_at"] = value
            connection.execute("UPDATE research.result SET payload=?", [encode(payload)])
        with pytest.raises(ValueError, match="research_"):
            verify_run(connection, metadata)


@pytest.mark.parametrize(
    "query_fragment,expected",
    [
        ("count(DISTINCT record_id)", "research_primary_key_invalid"),
        ("LEFT JOIN research.result parent ON child", "research_result_reference_missing"),
        ("FROM research.lineage edge", "research_parent_reference_missing"),
    ],
)
def test_verify_database_count_result_missing_is_rejected(libraries, query_fragment, expected):
    """场景：数据库驱动未返回完整性计数；预期：不把None当0，按对应校验错误拒绝。"""
    reference, output, config = libraries
    summary = run_research(reference, output, AS_OF, config, ("P01",))

    class MissingCount:
        """只模拟特定查询的驱动缺失返回，其他SQL仍由真实DuckDB执行。"""

        def execute(self, query, parameters):
            """拦截目标SQL；例：匹配count查询返回缺失游标，其余返回真实结果。"""
            if query_fragment in query:
                return self
            return connection.execute(query, parameters)

        def fetchone(self):
            """模拟缺失计数结果；例：驱动未返回行时得到None。"""
            return None

    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        metadata = read_metadata(connection, summary["run_id"])
        if query_fragment == "count(DISTINCT record_id)":
            # 有主键时由数据库约束保证唯一；移除约束后仍必须执行完整重复检查。
            connection.execute(
                "CREATE TABLE research.unconstrained AS SELECT * FROM research.feature"
            )
            connection.execute("DROP TABLE research.feature")
            connection.execute("ALTER TABLE research.unconstrained RENAME TO feature")
        with pytest.raises(ValueError, match=expected):
            verify_run(MissingCount(), metadata)


def test_unknown_thesis_signal_cannot_be_rewritten_as_false(tmp_path):
    """场景：空库的市场bundle产生unknown论点及证券确认；预期：只存一组信号且拒绝篡改。"""
    reference, _ = write_reference(tmp_path, 0)
    output = tmp_path / "signals.duckdb"
    run_research(reference, output, AS_OF, selection=("P13",))
    assert validate_research(reference, output)["valid"]
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        signals = [
            json.loads(row[0])
            for row in connection.execute("SELECT payload FROM research.signal").fetchall()
        ]
        thesis = [item for item in signals if item["predicate_id"] != "entity_confirmation"]
        assert {item["predicate_id"] for item in thesis} == {
            "cash_support",
            "limited_unlock",
            "cash_support_and_supply",
        }
        assert all(item["result"] == "unknown" for item in thesis)
        assert len(signals) == 4
        assert all(
            item["result"] == "unknown"
            for item in signals
            if item["predicate_id"] == "entity_confirmation"
        )
        connection.execute(
            "UPDATE research.signal SET payload=json_merge_patch(payload, ?)",
            [encode({"result": "false"})],
        )
    assert not validate_research(reference, output)["valid"]
