"""全产品运行身份、投影完整性和参考重放的篡改检测。"""

import json
from collections import Counter
from copy import deepcopy

import duckdb
import pytest
from all_products_fixture import AS_OF, DATABASE_CONFIG, write_reference
from test_all_products_pipeline import saved_results

from tushare_entitlement_hub.cross_asset_research import storage
from tushare_entitlement_hub.cross_asset_research.bindings import build_config
from tushare_entitlement_hub.cross_asset_research.execution import ExecutionState, RunLimits
from tushare_entitlement_hub.cross_asset_research.pipeline import run
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.equity_research.partitions import partition_catalog
from tushare_entitlement_hub.equity_research.storage import initialize_tables
from tushare_entitlement_hub.store import encode


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """创建可复用的只读合成批次；例：P05预期结果含非空利润指标及真实来源投影。"""
    directory = tmp_path_factory.mktemp("all-products-storage")
    reference = write_reference(directory)
    config = build_config(reference, AS_OF).config
    output = directory / "products.duckdb"
    run(reference, output, AS_OF, config, ("P05",))
    return reference, output, saved_results(output)[1]


@pytest.mark.parametrize("field", ["hash", "cutoff", "config", "products", "holdings", "limits"])
def test_run_identity_changes_for_each_execution_boundary(field):
    """场景：逐项改变来源、时间、配置、产品、持仓或容量；预期：每次产生不同运行身份。"""
    arguments = ["reference", AS_OF, ResearchConfig(), ("P05",), (), RunLimits()]
    original = storage.identity(*arguments)
    assert original == storage.identity(*arguments)
    index, value = {
        "hash": (0, "changed"),
        "cutoff": (1, "2026-10-04T00:00:00Z"),
        "config": (2, ResearchConfig(maximum_source_rows=1)),
        "products": (3, ("P02",)),
        "holdings": (4, ({"entity_id": "cn:security:A:listed", "weight": 0.5},)),
        "limits": (5, RunLimits(maximum_results=1)),
    }[field]
    arguments[index] = value
    assert storage.identity(*arguments) != original


@pytest.mark.parametrize(
    "status", ["complete", "annotated", "incomplete", "blocked", "unsupported", "unresolved"]
)
def test_summary_preserves_state_counts_and_marks_degraded_results(status):
    """场景：完整或各类降级结果；预期：只有无缺口的完整/注释结果可标记complete。"""
    state = ExecutionState(
        processing={"retained_rows": 2},
        states={"P05": Counter({status if status != "unresolved" else "complete": 1})},
        consumption={"unresolved": int(status == "unresolved")},
        result_count=1,
    )
    summary = storage.execution_summary(state)
    assert summary["status"] == (
        "complete" if status in {"complete", "annotated"} else "incomplete"
    )
    assert summary["result_count"] == 1 and summary["historical_pit_certified"] is False
    assert encode(summary) == encode(storage.execution_summary(state))


@pytest.mark.parametrize("field", ["version", "registry_fingerprint", "run_id"])
def test_saved_identity_rejects_each_inconsistent_metadata_field(stored, field):
    """场景：改变版本、注册指纹或运行标识；预期：读投影前即拒绝身份不一致。"""
    _, output, original = stored
    metadata = deepcopy(original)
    metadata[field] = "modified"
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        with pytest.raises(ValueError, match="all_products_identity_mismatch"):
            storage.verify_saved(connection, metadata)


def test_saved_product_selection_must_match_dependency_order(stored):
    """场景：元数据声称只跑P03而缺依赖；预期：返回selection_invalid，不接受不完整闭包。"""
    _, output, original = stored
    metadata = deepcopy(original)
    metadata["products"] = ["P03"]
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        with pytest.raises(ValueError, match="all_products_selection_invalid"):
            storage.verify_saved(connection, metadata)


@pytest.mark.parametrize(
    "field", ["projections", "counts", "result_count", "product_states", "physical_json_bytes"]
)
def test_saved_summary_and_projection_tampering_are_rejected(stored, field):
    """场景：篡改投影、记录数、产品状态或输出字节；预期：分别检出投影或摘要不一致。"""
    _, output, original = stored
    metadata = deepcopy(original)
    if field == "projections":
        metadata["projections"]["result"][0] += 1
    elif field == "counts":
        metadata["summary"]["counts"]["result"] += 1
    elif field == "product_states":
        metadata["summary"][field] = {}
    else:
        metadata["summary"][field] += 1
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        with pytest.raises(ValueError, match="all_products_(projection|summary)_mismatch"):
            storage.verify_saved(connection, metadata)


@pytest.mark.parametrize("field", ["projections", "consumption"])
def test_replay_rejects_forged_projection_or_consumption_summary(stored, field):
    """场景：已存摘要把真实消费或投影改写；预期：从reference重算后拒绝伪造。"""
    reference, _, original = stored
    metadata = deepcopy(original)
    if field == "projections":
        metadata["projections"]["result"][1] += 1
    else:
        metadata["summary"]["consumption"]["observed"] += 1
    with pytest.raises(ValueError, match="all_products_replay_mismatch"):
        storage.replay(reference, metadata)


@pytest.mark.parametrize("field", ["product_id", "run_id", "model_version", "analysis_as_of"])
def test_saved_result_context_must_match_run_metadata(tmp_path, stored, field):
    """场景：只改结果产品、运行ID、模型或分析时点；预期：不能混入其他运行的结果。"""
    _, original_output, metadata = stored
    output = tmp_path / "changed.duckdb"
    output.write_bytes(original_output.read_bytes())
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        part = partition_catalog(connection).get("result", ("result",))[0]
        row = connection.execute(f'SELECT record_id,payload FROM research."{part}"').fetchone()
        payload = json.loads(row[1])
        payload["envelope"][field] = (
            "2026-10-04T12:00:00Z" if field == "analysis_as_of" else "changed/v1"
        )
        if field == "analysis_as_of":
            payload["envelope"]["generated_at"] = payload["envelope"][field]
        connection.execute(
            f'UPDATE research."{part}" SET payload=? WHERE record_id=?', [encode(payload), row[0]]
        )
        with pytest.raises(ValueError, match="all_products_context_mismatch"):
            storage.verify_saved(connection, metadata)


@pytest.mark.parametrize(
    "scenario", ["missing_current", "missing_run", "run_identity", "reference", "orphan"]
)
def test_validate_rejects_database_or_reference_corruption(tmp_path, stored, scenario):
    """场景：当前批次丢失、身份伪造、参考变化或孤儿行；预期：安全固定错误返回valid=false。"""
    reference, original_output, original = stored
    output = tmp_path / "changed.duckdb"
    output.write_bytes(original_output.read_bytes())
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        if scenario == "missing_current":
            connection.execute("DELETE FROM research.metadata WHERE key='current_run'")
        elif scenario == "missing_run":
            connection.execute(
                "UPDATE research.metadata SET payload='\"absent\"' WHERE key='current_run'"
            )
        elif scenario == "run_identity":
            metadata = deepcopy(original)
            metadata["run_id"] = "changed"
            connection.execute("UPDATE research.run SET payload=?", [encode(metadata)])
        elif scenario == "orphan":
            part = partition_catalog(connection).get("result", ("result",))[0]
            connection.execute(
                f'INSERT INTO research."{part}" VALUES (?,?,?,?)',
                ["orphan", "missing", "orphan", "{}"],
            )
            if part != "result":
                connection.execute(
                    "INSERT INTO research._keys_result VALUES (?,?,?)",
                    ["orphan", "missing", "orphan"],
                )
        else:
            reference_copy = tmp_path / "reference.duckdb"
            reference_copy.write_bytes(reference.read_bytes())
            with duckdb.connect(str(reference_copy), config=DATABASE_CONFIG) as changed:
                changed.execute("CREATE TABLE changed(value INTEGER)")
            reference = reference_copy
    assert storage.validate(reference, output) == {
        "valid": False,
        "reason_codes": ["all_products_validation_failed"],
        "historical_pit_certified": False,
    }


def test_validate_missing_files_collision_and_empty_runs_fail_closed(tmp_path):
    """场景：缺库、输入输出同库或有current标记但无run；预期：固定完整性失败。"""
    reference = write_reference(tmp_path)
    assert not storage.validate(reference, reference)["valid"]
    assert not storage.validate(reference, tmp_path / "missing")["valid"]
    output = tmp_path / "empty.duckdb"
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        initialize_tables(connection)
        connection.execute("INSERT INTO research.metadata VALUES ('current_run', '\"absent\"')")
    assert not storage.validate(reference, output)["valid"]


def test_validate_rechecks_reference_after_replay(stored, monkeypatch):
    """场景：重放之后来源文件指纹改变；预期：不将读到两个快照的验证记为成功。"""
    reference, output, metadata = stored
    checks = iter((metadata["reference_sha256"], "changed"))
    monkeypatch.setattr(storage, "file_checksum", lambda path: next(checks))
    assert not storage.validate(reference, output)["valid"]
