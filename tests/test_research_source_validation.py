"""真实DuckDB上的来源证据回查测试，覆盖身份漂移、配置断言和只读批次边界。"""

from dataclasses import replace

import duckdb
import pytest
from test_research_adapter import (
    CONTRACTS,
    certification,
    configured,
    reference_database,
    source,
)
from test_research_governance import relation

from tushare_entitlement_hub.equity_research.source_validation import verify_source_evidence
from tushare_entitlement_hub.product_foundation.numeric import source_evidence
from tushare_entitlement_hub.product_foundation.reference_scan import map_availability
from tushare_entitlement_hub.reference_storage import file_checksum


@pytest.fixture
def products():
    """提供独立内存产品库，只建立真实证据表；例：插入current批次引用后可参数化查询。"""
    with duckdb.connect(config={"threads": 1, "memory_limit": "512MB"}) as connection:
        connection.execute("CREATE SCHEMA research")
        connection.execute("CREATE TABLE research.evidence(run_id VARCHAR, payload VARCHAR)")
        yield connection


def proof(record, fields=("close",), config=None):
    """按正式映射构建字段级证据；例：同一行的close和amount合并为双字段引用。"""
    api = record["source_api"]
    config = config or configured(api)
    mapping = next(item for item in config.foundation.time_mappings if item.source_api == api)
    availability = map_availability(record, CONTRACTS[api], mapping)
    return replace(source_evidence(record, fields[0], availability), fields=fields)


def put(products, *references, run_id="current"):
    """向真实产品表写入待查证据；例：两份相同引用仍按两行存储，回查自行去重。"""
    products.executemany(
        "INSERT INTO research.evidence VALUES (?, ?)",
        [(run_id, reference.to_json()) for reference in references],
    )


def test_source_lookup_preserves_reference_file_and_ignores_other_runs(tmp_path, products):
    """场景：当前批次双字段及重复引用、历史批次坏引用；预期：仅查当前批次，参考库哈希不变。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record]})
    reference = proof(record, ("close", "amount"))
    put(products, reference, reference)
    put(products, replace(reference, source_record_id="missing"), run_id="historical")
    checksum = file_checksum(path)
    verify_source_evidence(path, products, "current", configured("daily"))
    assert file_checksum(path) == checksum
    assert products.execute("SELECT count(*) FROM research.evidence").fetchone()[0] == 3


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
def test_invalid_batch_size_fails_before_opening_reference(tmp_path, products, batch_size):
    """场景：批容量非正整数；预期：固定原因拒绝，不创建不存在的参考库。"""
    path = tmp_path / "must-not-create.duckdb"
    with pytest.raises(ValueError, match="invalid_batch_size"):
        verify_source_evidence(path, products, "current", configured(), batch_size=batch_size)
    assert not path.exists()


def test_empty_current_batch_needs_no_source_rows(tmp_path, products):
    """场景：产品表没有当前批次证据；预期：合法空回查，参考库保持原样。"""
    path = reference_database(tmp_path, {"daily": [source()]})
    checksum = file_checksum(path)
    verify_source_evidence(path, products, "empty-run", configured())
    assert file_checksum(path) == checksum


@pytest.mark.parametrize(
    "change,config_api", [({"source_api": "unknown_api"}, "unknown_api"), ({}, None)]
)
def test_unknown_source_or_missing_time_mapping_is_rejected(tmp_path, products, change, config_api):
    """场景：来源未登记或缺时间映射；预期：不拼接未知来源查询、不猜时间语义。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record]})
    put(products, replace(proof(record), **change))
    config = configured(config_api) if config_api else configured()
    with pytest.raises(ValueError, match="research_evidence_source_unknown"):
        verify_source_evidence(path, products, "current", config)


def test_missing_registered_source_table_is_rejected(tmp_path, products):
    """场景：已登记来源的accepted表不存在；预期：回查失败，不能把缺表当零证据。"""
    path = reference_database(tmp_path, {"daily": [source()]})
    put(products, proof(source("share_float")))
    with pytest.raises((duckdb.Error, ValueError)):
        verify_source_evidence(path, products, "current", configured("share_float"))


def test_missing_source_record_is_rejected(tmp_path, products):
    """场景：引用的来源行已不存在；预期：明确source_missing，不能被同组其他记录代替。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record]})
    put(products, replace(proof(record), source_record_id="missing"))
    with pytest.raises(ValueError, match="research_evidence_source_missing"):
        verify_source_evidence(path, products, "current", configured("daily"))


def test_duplicate_source_id_is_not_arbitrarily_selected(tmp_path, products):
    """场景：accepted同一source_record_id有两行；预期：拒绝而不是任选一个值。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record, record | {"close": 20}]})
    put(products, proof(record))
    with pytest.raises(ValueError, match="research_evidence_source_duplicate"):
        verify_source_evidence(path, products, "current", configured("daily"))


@pytest.mark.parametrize(
    "patch", [{"quality_tier": "quarantine"}, {"quality_tier": "unknown"}, {"is_quarantined": True}]
)
def test_quarantined_record_cannot_impersonate_accepted_evidence(tmp_path, products, patch):
    """场景：accepted表内记录变成隔离、未知或显式隔离状态；预期：均拒绝作为可用来源。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record | patch]})
    put(products, proof(record))
    with pytest.raises(ValueError, match="research_evidence_source_missing"):
        verify_source_evidence(path, products, "current", configured("daily"))


def test_annotated_source_remains_valid_evidence(tmp_path, products):
    """场景：记录质量为usable_with_annotation；预期：可回查原始证据，不因此授予数值资格。"""
    record = source(quality_tier="usable_with_annotation")
    path = reference_database(tmp_path, {"daily": [record]})
    put(products, proof(record))
    verify_source_evidence(path, products, "current", configured("daily"))


def test_evidence_field_must_exist_in_source_row(tmp_path, products):
    """场景：引用包含来源行没有的字段；预期：完整拒绝双字段引用，不能仅检查第一个字段。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record]})
    put(products, replace(proof(record), fields=("close", "invented_field")))
    with pytest.raises(ValueError, match="research_evidence_field_missing"):
        verify_source_evidence(path, products, "current", configured("daily"))


@pytest.mark.parametrize(
    "patch",
    [
        {"business_group_key": "other-group"},
        {"business_variant_key": "other-variant"},
        {"contract_hash": "other-contract"},
        {"observed_at": "2026-10-02T12:00:00Z"},
        {"public_available_at": "2026-10-02T12:00:00Z"},
        {"locator": "reference:accepted/daily/other"},
        {"source_occurrences": ("other-occurrence",)},
    ],
)
def test_reference_metadata_and_locator_drift_is_rejected(tmp_path, products, patch):
    """场景：证据身份、合同、时间、重复来源或定位符漂移；预期：必须与原始行完整一致。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record]})
    put(products, replace(proof(record), **patch))
    with pytest.raises(ValueError, match="research_evidence_identity_mismatch"):
        verify_source_evidence(path, products, "current", configured("daily"))


@pytest.mark.parametrize(
    "patch",
    [
        {"business_group_key": "other-group"},
        {"business_variant_key": "other-variant"},
        {"observed_at": "2026-10-02T12:00:00Z"},
    ],
)
def test_source_metadata_changes_invalidate_existing_reference(tmp_path, products, patch):
    """场景：产品证据未变但accepted元数据改变；预期：双向一致性检查同样拒绝。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record | patch]})
    put(products, proof(record))
    with pytest.raises(ValueError, match="research_evidence_identity_mismatch"):
        verify_source_evidence(path, products, "current", configured("daily"))


def test_occurrence_json_metadata_is_decoded_and_compared(tmp_path, products):
    """场景：重复来源列表在DuckDB中为JSON字符串；预期：恢复顺序明确的来源列表后比较。"""
    record = source(source_occurrences=["request-one", "request-two"])
    path = reference_database(tmp_path, {"daily": [record]})
    put(products, proof(record))
    verify_source_evidence(path, products, "current", configured("daily"))


@pytest.mark.parametrize("kind", ["alias", "certification", "relation"])
def test_explicit_configuration_evidence_requires_exact_match(tmp_path, products, kind):
    """场景：独立别名/单位/关系认证以外部依据作证；预期：精确匹配才信任，任一字段改变即拒绝。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record]})
    reference = replace(proof(record), source_api="external_assertion", locator="assertion:manual")
    config = configured("daily")
    if kind == "alias":
        alias = replace(config.foundation.aliases[0], evidence_refs=(reference,))
        config = replace(config, foundation=replace(config.foundation, aliases=(alias,)))
    elif kind == "certification":
        certificate = replace(certification(), evidence_refs=(reference,))
        config = replace(
            config, foundation=replace(config.foundation, certifications=(certificate,))
        )
    else:
        config = replace(
            config, confirmed_relations=(replace(relation(), evidence_refs=(reference,)),)
        )
    put(products, reference)
    verify_source_evidence(path, products, "current", config)
    put(products, replace(reference, locator="assertion:changed"))
    with pytest.raises(ValueError, match="research_evidence_source_unknown"):
        verify_source_evidence(path, products, "current", config)


def test_more_than_512_references_crosses_batch_boundary_without_mutation(tmp_path, products):
    """场景：513条不同来源引用及第二个API；预期：默认512分批全部回查且两个数据库数据不变。"""
    records = [source("daily", f"record-{index}") for index in range(513)]
    other = source("share_float", "event")
    path = reference_database(tmp_path, {"daily": records, "share_float": [other]})
    put(products, *(proof(record) for record in records), proof(other))
    checksum = file_checksum(path)
    config = configured("daily", "share_float")
    verify_source_evidence(path, products, "current", config)
    verify_source_evidence(path, products, "current", config, batch_size=1)
    assert file_checksum(path) == checksum
    assert products.execute("SELECT count(*) FROM research.evidence").fetchone()[0] == 514


def test_synchronized_source_and_reference_contract_tampering_is_rejected(tmp_path, products):
    """场景：来源行和产品证据同步改为非当前合同；预期：不能以彼此一致绕过当前合同认证。"""
    record = source(contract_hash="non-current-contract")
    path = reference_database(tmp_path, {"daily": [record]})
    put(products, proof(record))
    with pytest.raises(ValueError, match="research_evidence_contract_mismatch"):
        verify_source_evidence(path, products, "current", configured("daily"))


def test_source_api_identity_cannot_move_between_accepted_tables(tmp_path, products):
    """场景：daily表的来源行伪称其他API；预期：即使记录ID和字段相同，仍拒绝跨来源身份。"""
    record = source()
    path = reference_database(tmp_path, {"daily": [record | {"source_api": "income"}]})
    put(products, proof(record))
    with pytest.raises(ValueError, match="research_evidence_contract_mismatch"):
        verify_source_evidence(path, products, "current", configured("daily"))
