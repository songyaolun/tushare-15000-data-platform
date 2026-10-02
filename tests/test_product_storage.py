"""独立产品库和命令行的离线集成验收，先于存储实现编写。"""

import json
from dataclasses import replace

import duckdb
import pytest
from test_product_foundation import ENTITY, NOW, evidence
from typer.testing import CliRunner

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.product_foundation.bindings import EntityMapping, TimeMapping
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityAlias, EntityIdentity
from tushare_entitlement_hub.product_foundation.numeric import UnitCertification
from tushare_entitlement_hub.product_foundation.reference import (
    FoundationBindings,
    initialize_library,
)
from tushare_entitlement_hub.product_foundation.storage import ProductStore, validate_library
from tushare_entitlement_hub.reference_contract import load_reference_contracts
from tushare_entitlement_hub.reference_quality import evaluate_records
from tushare_entitlement_hub.reference_storage import (
    file_checksum,
    publish_library,
    write_dataset_files,
)


def reference_fixture(tmp_path, *, quarantined=False):
    """发布最小真实参考库格式；输入隔离标志，输出只包含合成 daily 的 reference.duckdb。"""
    contract = load_reference_contracts(load_policies())["daily"]
    source = {
        "ts_code": "A.SZ",
        "trade_date": "20260930",
        "close": 10,
        "open": 10,
        "high": 11,
        "low": 9,
        "vol": 1,
        "amount": 2,
    }
    records = [(source, "hash1", 0, "request1")]
    if quarantined:
        records.append((source | {"ts_code": "BAD.SZ", "high": 1}, "hash2", 1, "request1"))
    accepted, quarantine, quality = evaluate_records(records, contract, NOW)
    root = tmp_path / "reference"
    published = write_dataset_files(root / "daily", accepted, quarantine, contract)
    manifest = {
        "api_name": "daily",
        "status": "partial" if quarantined else "accepted",
        "observed_at": NOW,
        "contract_hash": contract.fingerprint,
        "collection_status": "complete_declared_request",
        "parameters": {"trade_date": "20260930"},
        "quality": quality,
        **published,
    }
    return publish_library(root, [manifest])


def bindings():
    """提供人工认证的合成实体和单位；默认输入，输出可适配 daily.amount 的绑定。"""
    entity = Entity(
        identity=EntityIdentity(
            namespace="cn", kind="security", local_id="A.SZ", lifecycle_id="listing1"
        ),
        name="合成证券",
    )
    alias = EntityAlias(
        namespace="tushare",
        alias="A.SZ",
        entity_id=ENTITY,
        effective_from="2020-01-01T00:00:00Z",
        effective_to=None,
        observed_at=NOW,
        source="fixture",
        version="v1",
        confidence=100,
        evidence_refs=(evidence(),),
    )
    unit = UnitCertification(
        version="v1",
        source_api="daily",
        field="amount",
        source_unit="千元",
        target_unit="元",
        dimension="money",
        currency="CNY",
        multiplier=1000,
        evidence_refs=(evidence(),),
    )
    return FoundationBindings(
        entities=(entity,),
        aliases=(alias,),
        certifications=(unit,),
        entity_mappings=(
            EntityMapping(source_api="daily", namespace="tushare", key_field="ts_code"),
        ),
        time_mappings=(TimeMapping(source_api="daily", data_field="trade_date"),),
    )


def test_product_store_tables_roundtrip_idempotency_and_conflicts(tmp_path):
    """场景：重复写相同实体及错表写入；预期：幂等、类型校验且不覆盖既有不同事实。

    输入同一实体两次，输出一行；同键不同名称抛 ValueError。
    """
    path = tmp_path / "products.duckdb"
    with ProductStore(path) as store:
        assert set(store.tables()) == {
            "entity",
            "entity_alias",
            "entity_relation",
            "availability_fact",
            "typed_fact",
            "feature_value",
            "signal_fact",
            "score_component",
            "quality_gate_result",
            "coverage_slice",
            "lineage_edge",
        }
        entity = bindings().entities[0]
        store.put("entity", entity)
        store.put("entity", entity)
        assert store.counts()["entity"] == 1
        assert store.records("entity")[0] == entity
        with pytest.raises(ValueError):
            store.put("entity", replace(entity, name="不同事实"))
        with pytest.raises(ValueError):
            store.put("arbitrary", entity)
        with pytest.raises(ValueError):
            store.put("feature_value", entity)


def test_initialization_reads_reference_only_and_reports_partial_coverage(tmp_path):
    """场景：206范围仅有daily且含隔离行；预期：源库不变、分项降级、隔离不入特征。

    输入仅有 daily 的参考库，输出 206 项覆盖及 incomplete，源文件摘要不变。
    """
    reference = reference_fixture(tmp_path, quarantined=True)
    before = file_checksum(reference)
    output = tmp_path / "products.duckdb"
    policy = FoundationPolicy(version="test")
    summary = initialize_library(reference, output, NOW, policy, bindings())
    assert summary["registered_apis"] == 206
    assert summary["status"] == "incomplete" and not summary["historical_pit_certified"]
    assert file_checksum(reference) == before
    with ProductStore(output) as store:
        assert store.counts()["coverage_slice"] == 206
        assert store.counts()["typed_fact"] >= 1
        assert store.counts()["feature_value"] == 0
        facts = store.records("typed_fact")
        assert any(fact.field == "amount" and fact.value == 2000 for fact in facts)
        assert all(fact.quality_tier != "quarantine" for fact in facts)
        assert store.counts()["lineage_edge"] >= 1
    assert initialize_library(reference, output, NOW, policy, bindings()) == summary
    assert validate_library(reference, output)["valid"]
    with duckdb.connect(str(reference)) as connection:
        connection.execute("UPDATE accepted.daily SET amount=123")
    assert not validate_library(reference, output)["valid"]


def test_missing_source_and_same_path_never_overwrite_input(tmp_path):
    """场景：缺参考库或输出与输入相同；预期：明确失败，不创建伪参考库或破坏原库。"""
    policy = FoundationPolicy(version="test")
    with pytest.raises(ValueError):
        initialize_library(tmp_path / "missing.duckdb", tmp_path / "products.duckdb", NOW, policy)
    source = reference_fixture(tmp_path)
    before = file_checksum(source)
    with pytest.raises(ValueError):
        initialize_library(source, source, NOW, policy)
    assert file_checksum(source) == before


def test_offline_cli_initializes_and_validates_without_credentials(tmp_path, monkeypatch):
    """场景：CLI离线初始化、校验及错误输入；预期：机器可读摘要，不触碰凭据入口。"""

    def forbidden(*args, **kwargs):
        """拦截凭据访问；任意调用输入，输出 AssertionError，证明命令无需token。"""
        raise AssertionError("基础层不得访问凭据")

    monkeypatch.setattr("tushare_entitlement_hub.cli.configured_credential", forbidden)
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "product-foundation",
            "initialize",
            "--reference",
            str(source),
            "--output",
            str(output),
            "--as-of",
            NOW,
        ],
    )
    assert result.exit_code == 2
    summary = json.loads(result.stdout)
    assert summary["status"] == "incomplete" and summary["registered_apis"] == 206
    checked = runner.invoke(
        app, ["product-foundation", "validate", "--reference", str(source), "--output", str(output)]
    )
    assert checked.exit_code == 0 and json.loads(checked.stdout)["valid"]
    failed = runner.invoke(
        app,
        [
            "product-foundation",
            "initialize",
            "--reference",
            "missing.duckdb",
            "--output",
            str(output),
            "--as-of",
            NOW,
        ],
    )
    assert failed.exit_code == 1 and json.loads(failed.stdout)["status"] == "failed"


def test_new_time_mapping_replaces_current_snapshot_and_preserves_run_summary(tmp_path):
    """场景：补充公开时间映射后重跑；预期：替换当前事实，保留旧批次摘要。

    输入：同一参考行补充公开时间；输出：当前仅一版事实且认证版本改变。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    policy = FoundationPolicy(version="v1")
    previous_summary = initialize_library(source, output, NOW, policy, bindings())
    with ProductStore(output, read_only=True) as store:
        previous_version = store.records("availability_fact")[0].certification_version
    configured = replace(
        bindings(),
        time_mappings=(
            TimeMapping(
                source_api="daily",
                data_field="trade_date",
                public_fields=("trade_date",),
                market_timezone="Asia/Shanghai",
            ),
        ),
    )
    current_summary = initialize_library(source, output, NOW, policy, configured)
    with ProductStore(output, read_only=True) as store:
        assert store.counts()["availability_fact"] == 1
        assert store.records("availability_fact")[0].certification_version != previous_version
        assert current_summary["run_id"] != previous_summary["run_id"]
        assert store.metadata(f"run:{previous_summary['run_id']}")["summary"] == previous_summary
        assert any(fact.value == 2000 for fact in store.records("typed_fact"))
