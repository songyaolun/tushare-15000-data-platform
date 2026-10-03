"""绑定构建的容量边界、只读投影和安全发布。"""

import json
from dataclasses import FrozenInstanceError

import duckdb
import pytest
from test_research_binding_fixture import DATABASE_CONFIG

from tushare_entitlement_hub.equity_research.binding_io import (
    BuildLimits,
    check_paths,
    projected_rows,
)


@pytest.mark.parametrize(
    "field", ["maximum_source_rows", "maximum_total_rows", "maximum_entities", "maximum_row_bytes"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_build_limits_rejects_nonpositive_and_noninteger_limits(field, value):
    """容量仅接受严格正整数；例如maximum_entities=True不当作1并返回固定错误码。"""
    with pytest.raises(ValueError, match="binding_limits_invalid"):
        BuildLimits(**{field: value})


@pytest.mark.parametrize("value", [0, 513, True, 1.5])
def test_build_limits_rejects_invalid_batch(value):
    """批量大小必须为1到512整数；输入0、513或布尔值均拒绝。"""
    with pytest.raises(ValueError, match="invalid_batch_size"):
        BuildLimits(batch_size=value)


def test_build_limits_are_frozen():
    """有效边界配置不可修改；例如batch_size=512接受，后续赋值抛FrozenInstanceError。"""
    limits = BuildLimits(batch_size=512)
    assert limits.batch_size == 512
    with pytest.raises(FrozenInstanceError):
        limits.batch_size = 1


@pytest.mark.parametrize("collision", ["same", "symlink", "hardlink", "outputs"])
def test_output_paths_protect_input_and_each_other(tmp_path, collision):
    """同路径、软链接、硬链接以及两个输出互撞均拒绝，输入字节保持不变。

    例如output硬链到reference时抛input_output_path_collision，不覆盖原始库。
    """
    reference = tmp_path / "reference.duckdb"
    reference.write_bytes(b"synthetic")
    output = tmp_path / "config.json"
    if collision == "same":
        output = reference
    elif collision == "symlink":
        output.symlink_to(reference)
    elif collision == "hardlink":
        output.hardlink_to(reference)
    outputs = [output, output] if collision == "outputs" else [output]
    with pytest.raises(ValueError, match="input_output_path_collision"):
        check_paths(reference, *outputs)
    assert reference.read_bytes() == b"synthetic"


def test_output_paths_accept_distinct_outputs_and_reject_missing_input(tmp_path):
    """独立输出允许尚不存在；输入不存在时抛reference_library_missing且不创建目录。"""
    reference = tmp_path / "reference.duckdb"
    with pytest.raises(ValueError, match="reference_library_missing"):
        check_paths(reference, tmp_path / "nested/config.json")
    assert not (tmp_path / "nested").exists()
    reference.write_bytes(b"synthetic")
    assert check_paths(reference, tmp_path / "config.json", tmp_path / "audit.json") is None


def test_projected_rows_decodes_metadata_and_bounds_batches():
    """投影只读取声明列，并解码字符串JSON和原生JSON；260行完整跨过128行批次。

    输入source_units='{"amount":"千元"}'输出字典，未选择的source_fields不返回。
    """
    with duckdb.connect(config=DATABASE_CONFIG) as connection:
        connection.execute("CREATE SCHEMA accepted")
        connection.execute(
            "CREATE TABLE accepted.daily AS SELECT range AS id, "
            '\'{"amount":"千元"}\' AS source_units, '
            '\'{"amount":"元"}\'::JSON AS standard_units, '
            "'[\"synthetic\"]' AS source_occurrences, "
            "'excluded' AS source_fields FROM range(260)"
        )
        fields = {"id", "source_units", "standard_units", "source_occurrences"}
        rows = list(projected_rows(connection, "daily", fields, BuildLimits(batch_size=512)))
    assert len(rows) == 260
    assert rows[0] == {
        "id": 0,
        "source_units": {"amount": "千元"},
        "standard_units": {"amount": "元"},
        "source_occurrences": ["synthetic"],
    }
    assert rows[-1]["id"] == 259


def test_projected_rows_rejects_oversize_payload_at_exact_boundary():
    """JSON投影字节数等于上限时通过，大一个字节即拒绝；输入{'value':'合成'}按UTF8计。

    不以字符数代替字节数，容量失败抛binding_row_capacity_exceeded。
    """
    with duckdb.connect(config=DATABASE_CONFIG) as connection:
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE TABLE accepted.daily AS SELECT '合成' AS value")
        size = len(
            json.dumps({"value": "合成"}, ensure_ascii=False, separators=(",", ":")).encode()
        )
        assert list(
            projected_rows(connection, "daily", {"value"}, BuildLimits(maximum_row_bytes=size))
        ) == [{"value": "合成"}]
        with pytest.raises(ValueError, match="binding_row_capacity_exceeded"):
            list(
                projected_rows(
                    connection, "daily", {"value"}, BuildLimits(maximum_row_bytes=size - 1)
                )
            )


def test_builder_creates_deterministic_auditable_config(tmp_path):
    """同一参考库与时点生成字节一致配置，单位不重复缩放，审计不冒称已消费。

    输入默认两股票四来源，输出99规则、2实体，scanned_rows=8且observed_apis=0。
    """
    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import (
        build_research_config,
        configuration_hash,
        verify_binding_audit,
        write_research_config,
    )
    from tushare_entitlement_hub.equity_research.models import ResearchConfig
    from tushare_entitlement_hub.reference_storage import file_checksum

    reference = write_reference(tmp_path)
    first = build_research_config(reference, AS_OF)
    second = build_research_config(reference, AS_OF)
    assert first.config.to_json() == second.config.to_json()
    assert first.audit == first.config.binding_audit == second.audit
    assert first.audit["summary"]["configured_apis"] == 99
    assert first.audit["summary"]["scanned_rows"] == 8
    assert first.audit["summary"]["entities"] == 2
    assert first.audit["summary"]["observed_apis"] == 0
    assert first.audit["reference_sha256"] == file_checksum(reference)
    assert first.audit["configuration_sha256"] == configuration_hash(first.config)
    certificates = {
        (item.source_api, item.field): item for item in first.config.foundation.certifications
    }
    assert certificates["daily", "amount"].multiplier == 1
    assert certificates["daily", "amount"].currency == "CNY"
    assert certificates["daily", "vol"].multiplier == 1
    assert certificates["daily", "vol"].target_unit == "股"
    assert verify_binding_audit(first.config, file_checksum(reference), AS_OF) is None
    assert verify_binding_audit(ResearchConfig(), "unbound", AS_OF) is None
    output = tmp_path / "output/config.json"
    audit = tmp_path / "audit/result.json"
    result = write_research_config(reference, output, AS_OF, audit_path=audit)
    assert result["status"] == "built"
    assert json.loads(audit.read_text()) == first.audit
    before = output.read_bytes(), audit.read_bytes()
    write_research_config(reference, output, AS_OF, audit_path=audit)
    assert (output.read_bytes(), audit.read_bytes()) == before
    with pytest.raises(FrozenInstanceError):
        first.config = ResearchConfig()


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        ("catalog", "reference_catalog_missing"),
        ("stock_basic", "binding_stock_basic_missing"),
        ("duplicate_catalog", "reference_catalog_duplicate"),
        ("source_capacity", "binding_scan_capacity_exceeded"),
        ("total_capacity", "binding_scan_capacity_exceeded"),
        ("entity_capacity", "binding_entity_capacity_exceeded"),
        ("row_capacity", "binding_row_capacity_exceeded"),
    ],
)
def test_builder_global_failures_leave_outputs_absent(tmp_path, variant, code):
    """缺少必要表、目录重复和任一容量超限都在写出前失败。

    例如2条主数据配maximum_entities=1，抛binding_entity_capacity_exceeded且无半成品。
    """
    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import write_research_config

    reference = write_reference(tmp_path)
    if variant in {"catalog", "stock_basic", "duplicate_catalog"}:
        with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
            if variant == "duplicate_catalog":
                connection.execute("INSERT INTO catalog SELECT * FROM catalog LIMIT 1")
            else:
                table = "catalog" if variant == "catalog" else "accepted.stock_basic"
                connection.execute(f"DROP TABLE {table}")
    limits = BuildLimits(
        **{
            "source_capacity": {"maximum_source_rows": 1},
            "total_capacity": {"maximum_total_rows": 7},
            "entity_capacity": {"maximum_entities": 1},
            "row_capacity": {"maximum_row_bytes": 1},
        }.get(variant, {})
    )
    output = tmp_path / "unwritten/config.json"
    with pytest.raises(ValueError, match=code):
        write_research_config(reference, output, AS_OF, limits=limits)
    assert not output.parent.exists()


def test_builder_exact_capacity_and_empty_master_degrade_explicitly(tmp_path):
    """容量等于实际行数允许；stock_basic空表保留审计而不制造实体。

    输入2条每源、共8行可成功；主数据空表时entities=0且记录no_eligible_rows。
    """
    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import build_research_config

    reference = write_reference(tmp_path / "full")
    result = build_research_config(
        reference,
        AS_OF,
        limits=BuildLimits(maximum_source_rows=2, maximum_total_rows=8, maximum_entities=2),
    )
    assert result.audit["summary"]["entities"] == 2
    empty = write_reference(tmp_path / "empty", {"stock_basic": []})
    result = build_research_config(empty, AS_OF)
    assert result.audit["summary"]["entities"] == 0
    assert result.audit["sources"]["stock_basic"]["reason_counts"]["no_eligible_rows"] == 1


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"pub_date": "2026-10-01"}, "public_field_mapping_missing"),
        ({"source_api": "weekly"}, "source_contract_mismatch"),
        ({"contract_hash": "changed"}, "source_contract_mismatch"),
        ({"business_variant_key": "version"}, "business_variant_unresolved"),
        ({"identity_resolution": "business_group_source_variant"}, "business_variant_unresolved"),
        ({"quality_tier": "unknown"}, "binding_quality_unverified"),
        ({"is_quarantined": True}, "binding_quality_unverified"),
        ({"observed_at": "2026-10-04T00:00:00Z"}, "binding_future_observation"),
        ({"trade_date": "2026-10-04"}, "future_business_date"),
    ],
)
def test_binding_availability_rejects_unusable_source_rows(changes, expected):
    """来源错配、变体、低质量和未来观察/业务日期均返回固定拒绝原因。

    例如daily含未注册pub_date时抛public_field_mapping_missing，不能忽略该公开字段。
    """
    from test_research_binding_fixture import AS_OF, default_rows

    from tushare_entitlement_hub.catalog import load_policies
    from tushare_entitlement_hub.equity_research.binding_builder import binding_availability
    from tushare_entitlement_hub.equity_research.binding_rules import time_registry
    from tushare_entitlement_hub.reference_contract import (
        load_reference_contracts,
        normalize_reference_row,
    )

    contracts = load_reference_contracts(load_policies())
    mapping = next(item for item in time_registry(contracts) if item.source_api == "daily")
    row = normalize_reference_row(
        default_rows()["daily"][0], contracts["daily"], AS_OF, "synthetic", 0
    )
    row.update(changes)
    with pytest.raises(ValueError, match=expected):
        binding_availability(row, contracts["daily"], mapping, AS_OF)


def test_binding_availability_future_public_time_and_event_date():
    """未来公告拒绝，已经观察的未来事件日期可保留；报告期不会替代公告时点。

    输入share_float未来解禁日且当前公告允许，公告自身晚于AS_OF则拒绝。
    """
    from test_research_binding_fixture import AS_OF

    from tushare_entitlement_hub.catalog import load_policies
    from tushare_entitlement_hub.equity_research.binding_builder import binding_availability
    from tushare_entitlement_hub.equity_research.binding_rules import time_registry
    from tushare_entitlement_hub.reference_contract import load_reference_contracts

    contracts = load_reference_contracts(load_policies())
    mapping = next(item for item in time_registry(contracts) if item.source_api == "share_float")
    row = {
        "source_api": "share_float",
        "contract_hash": contracts["share_float"].fingerprint,
        "source_record_id": "synthetic",
        "quality_tier": "verified",
        "historical_pit_certified": False,
        "observed_at": AS_OF,
        "float_date": "2026-12-01",
        "ann_date": "2026-10-01",
    }
    fact = binding_availability(row, contracts["share_float"], mapping, AS_OF)
    assert fact.data_as_of == "2026-12-01"
    row["ann_date"] = "2026-10-04"
    with pytest.raises(ValueError, match="binding_future_public_time"):
        binding_availability(row, contracts["share_float"], mapping, AS_OF)


def test_safe_reason_and_projection_exclude_untrusted_content():
    """异常载荷只映射固定码，新闻正文和原始对象不进入绑定投影。

    输入带任意文本ValueError返回binding_row_invalid；major_news投影保留pub_time。
    """
    from tushare_entitlement_hub.catalog import load_policies
    from tushare_entitlement_hub.equity_research.binding_builder import (
        projection_fields,
        safe_row_reason,
    )
    from tushare_entitlement_hub.equity_research.binding_rules import time_registry
    from tushare_entitlement_hub.reference_contract import load_reference_contracts

    contracts = load_reference_contracts(load_policies())
    mapping = next(item for item in time_registry(contracts) if item.source_api == "major_news")
    fields = {"pub_time", "content", "title", "source_fields", "source_record_id", "pub_date"}
    assert projection_fields(fields, contracts["major_news"], mapping) == {
        "pub_time",
        "pub_date",
        "source_record_id",
    }
    assert safe_row_reason(ValueError("arbitrary source payload")) == "binding_row_invalid"
    assert safe_row_reason(ValueError("binding_future_public_time")) == "binding_future_public_time"


@pytest.mark.parametrize("field", ["version", "rules_fingerprint", "configuration_sha256"])
def test_verify_binding_audit_rejects_tampered_certification(tmp_path, field):
    """审计版本、规则指纹和配置指纹任一变化必须拒绝，避免迁移旧认证。

    例如rules_fingerprint='changed'时抛binding_configuration_mismatch。
    """
    from dataclasses import replace

    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import (
        build_research_config,
        verify_binding_audit,
    )

    result = build_research_config(write_reference(tmp_path), AS_OF)
    from tushare_entitlement_hub.store import digest

    audit = {**result.audit, field: "changed"}
    audit.pop("audit_sha256")
    audit["audit_sha256"] = digest(audit)
    changed = replace(result.config, binding_audit_json=json.dumps(audit))
    with pytest.raises(ValueError, match="binding_configuration_mismatch"):
        verify_binding_audit(changed, result.audit["reference_sha256"], AS_OF)


def test_builder_detects_reference_change_before_any_output(tmp_path, monkeypatch):
    """扫描开始和结束指纹不同就放弃输出；模拟文件系统变化不修改真实参考库。

    输入前后摘要first/second，抛reference_changed_during_read且输出目录不存在。
    """
    from unittest.mock import Mock

    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research import binding_builder

    reference = write_reference(tmp_path)
    checksum = Mock(side_effect=["first", "second"])
    monkeypatch.setattr(binding_builder, "file_checksum", checksum)
    output = tmp_path / "unwritten/config.json"
    with pytest.raises(ValueError, match="reference_changed_during_read"):
        binding_builder.write_research_config(reference, output, AS_OF)
    assert checksum.call_count == 2
    assert not output.parent.exists()


@pytest.mark.parametrize("variant", ["public_omission", "invalid_master", "missing_date"])
def test_builder_reports_source_gaps_and_withholds_bad_bindings(tmp_path, variant):
    """遗漏公开字段时整体构建失败，坏主数据拒绝身份，缺业务日期不声明新鲜度。

    例如daily.pub_date非空却不在规则中，抛public_field_mapping_missing。
    """
    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import build_research_config

    reference = write_reference(tmp_path)
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        if variant == "public_omission":
            connection.execute("ALTER TABLE accepted.daily ADD COLUMN pub_date VARCHAR")
            connection.execute("UPDATE accepted.daily SET pub_date='2026-10-01'")
        elif variant == "invalid_master":
            connection.execute("UPDATE accepted.stock_basic SET observed_at='2026-12-01'")
        else:
            connection.execute("UPDATE accepted.daily SET trade_date=NULL")
    if variant == "public_omission":
        with pytest.raises(ValueError, match="public_field_mapping_missing"):
            build_research_config(reference, AS_OF)
        return
    result = build_research_config(reference, AS_OF)
    if variant == "invalid_master":
        assert not result.config.foundation.entities
        assert (
            "binding_future_observation" in result.audit["sources"]["stock_basic"]["reason_counts"]
        )
    else:
        assert "daily" not in result.config.foundation.expected_data_as_of


def test_audit_self_hash_reference_and_cutoff_are_independent_guards(tmp_path):
    """审计自校验、参考库指纹与分析时点分别把关；篡改摘要计数不能伪装可信审计。

    输入summary.entities=999而未重算audit_sha256，抛binding_audit_mismatch；
    原配置对不同reference或as_of分别抛reference/cutoff固定错误码。
    """
    from dataclasses import replace

    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import (
        build_research_config,
        verify_binding_audit,
    )

    result = build_research_config(write_reference(tmp_path), AS_OF)
    changed_audit = result.config.binding_audit
    changed_audit["summary"]["entities"] = 999
    changed = replace(result.config, binding_audit_json=json.dumps(changed_audit))
    with pytest.raises(ValueError, match="binding_audit_mismatch"):
        verify_binding_audit(changed, result.audit["reference_sha256"], AS_OF)
    with pytest.raises(ValueError, match="binding_reference_mismatch"):
        verify_binding_audit(result.config, "changed-reference", AS_OF)
    with pytest.raises(ValueError, match="binding_cutoff_mismatch"):
        verify_binding_audit(
            result.config, result.audit["reference_sha256"], "2026-10-04T12:00:00Z"
        )


def test_binding_summary_counts_actual_observed_and_preserves_unit_gaps():
    """摘要只计算真实observed来源，已观察但单位缺失仍属未解决接口。

    输入3个可读源，其中daily已观察无缺口、daily_basic已观察有单位缺口、stk_limit未观察，
    输出readable=3、observed=2、unresolved=2及各自原因。
    """
    from tushare_entitlement_hub.equity_research.binding_builder import binding_summary
    from tushare_entitlement_hub.equity_research.models import ResearchConfig

    audit = {
        "reference_sha256": "reference",
        "generated_as_of": "2026-10-03T12:00:00Z",
        "rules_fingerprint": "rules",
        "audit_sha256": "audit",
        "sources": {
            "daily": {"readable": True, "reason_counts": {}, "unit_gaps": {}},
            "daily_basic": {
                "readable": True,
                "reason_counts": {},
                "unit_gaps": {"pe_ttm": {"unit_unknown": 1}},
            },
            "stk_limit": {
                "readable": True,
                "reason_counts": {"no_eligible_rows": 1},
                "unit_gaps": {},
            },
        },
    }
    result = binding_summary(
        ResearchConfig(binding_audit_json=json.dumps(audit)),
        {
            "by_api": {
                "daily": {"observed": 1},
                "daily_basic": {"observed": 2},
                "stk_limit": {"observed": 0},
            }
        },
    )
    assert result["configured_apis"] == 0
    assert result["readable_apis"] == 3
    assert result["observed_apis"] == 2
    assert result["unresolved_apis"] == 2
    assert result["unresolved_reasons"] == {
        "daily_basic": ["unit_certification_gaps"],
        "stk_limit": ["no_eligible_rows", "no_observed_consumption"],
    }


def test_wal_sidecar_prevents_uncertified_reference_read(tmp_path):
    """参考库存在WAL侧文件时拒绝，文件摘要无法代表尚未checkpoint的完整数据库。

    输入reference.duckdb与同名.wal，check_paths抛binding_reference_wal_present。
    """
    reference = tmp_path / "reference.duckdb"
    reference.write_bytes(b"synthetic")
    (tmp_path / "reference.duckdb.wal").write_bytes(b"pending")
    with pytest.raises(ValueError, match="binding_reference_wal_present"):
        check_paths(reference, tmp_path / "config.json")


def test_write_checks_reference_again_after_finished_build(tmp_path, monkeypatch):
    """扫描前后摘要一致但发布前发生变化时仍拒绝，避免将认证绑定到旧文件。

    模拟三次指纹first/first/changed，输出reference_changed_during_read且目录未创建。
    """
    from unittest.mock import Mock

    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research import binding_builder

    reference = write_reference(tmp_path)
    checksum = Mock(side_effect=["first", "first", "changed"])
    monkeypatch.setattr(binding_builder, "file_checksum", checksum)
    output = tmp_path / "unwritten/config.json"
    with pytest.raises(ValueError, match="reference_changed_during_read"):
        binding_builder.write_research_config(reference, output, AS_OF)
    assert checksum.call_count == 3
    assert not output.parent.exists()


def test_scan_future_event_keeps_observation_without_freshness_claim():
    """未来解禁事件可作为已公告事件保留，却不能把未来日期当作当前数据新鲜度。

    输入AS_OF已观察、12月解禁的share_float，scanned=eligible=1且expected=None。
    """
    from test_research_binding_fixture import AS_OF

    from tushare_entitlement_hub.catalog import load_policies
    from tushare_entitlement_hub.equity_research.binding_builder import scan_source
    from tushare_entitlement_hub.equity_research.binding_entities import EntityBuild
    from tushare_entitlement_hub.equity_research.binding_rules import time_registry
    from tushare_entitlement_hub.reference_contract import load_reference_contracts

    contracts = load_reference_contracts(load_policies())
    contract = contracts["share_float"]
    mapping = next(item for item in time_registry(contracts) if item.source_api == "share_float")
    row = {
        "source_api": "share_float",
        "source_record_id": "synthetic",
        "contract_hash": contract.fingerprint,
        "quality_tier": "verified",
        "historical_pit_certified": False,
        "observed_at": AS_OF,
        "float_date": "2026-12-01",
        "ann_date": "2026-10-01",
    }
    with duckdb.connect(config=DATABASE_CONFIG) as connection:
        connection.execute("CREATE SCHEMA accepted")
        columns = ",".join(
            f'"{name}" ' + ("BOOLEAN" if name == "historical_pit_certified" else "VARCHAR")
            for name in row
        )
        connection.execute(f"CREATE TABLE accepted.share_float ({columns})")
        placeholders = ",".join("?" for _ in row)
        connection.execute(
            f"INSERT INTO accepted.share_float VALUES ({placeholders})", list(row.values())
        )
        report, units, expected = scan_source(
            connection, contract, mapping, set(row), AS_OF, BuildLimits(), EntityBuild(5)
        )
    assert report["scanned_rows"] == report["eligible_rows"] == 1
    assert report["reason_counts"] == {"business_freshness_uncertified": 1}
    assert not units.accepted()
    assert expected is None


def test_catalog_capacity_rejects_excess_entries_before_loading_manifests(tmp_path):
    """目录记录数量大于正式契约数时立即拒绝，避免把无界清单载入内存。

    输入len(contracts)+1条目录行，抛binding_catalog_capacity_exceeded且不写配置。
    """
    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.catalog import load_policies
    from tushare_entitlement_hub.equity_research.binding_builder import write_research_config
    from tushare_entitlement_hub.reference_contract import load_reference_contracts

    reference = write_reference(tmp_path)
    count = len(load_reference_contracts(load_policies())) + 1
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("DELETE FROM catalog")
        connection.execute(
            "INSERT INTO catalog SELECT 'synthetic-' || range, '{}' FROM range(?)", [count]
        )
    output = tmp_path / "unwritten/config.json"
    with pytest.raises(ValueError, match="binding_catalog_capacity_exceeded"):
        write_research_config(reference, output, AS_OF)
    assert not output.parent.exists()


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_atomic_publish_failures_preserve_old_config_and_cleanup(tmp_path, monkeypatch, failure):
    """临时写入或原子替换失败时旧配置完整保留，所有临时文件都会清理。

    例如第二个临时文件write失败，两个目标仍是旧字节；replace失败同样传播OSError。
    """
    from pathlib import Path
    from unittest.mock import Mock

    from tushare_entitlement_hub.equity_research import binding_io

    report, config = tmp_path / "audit.json", tmp_path / "config.json"
    report.write_text("old-report")
    config.write_text("old-config")
    if failure == "replace":
        replacement = Mock(side_effect=OSError("synthetic replace failure"))
        monkeypatch.setattr(Path, "replace", replacement)
    else:
        from contextlib import contextmanager

        original = binding_io.NamedTemporaryFile
        calls = []

        @contextmanager
        def temporary_file(*args, **kwargs):
            """第一次返回真实句柄，第二次模拟文件系统写入失败；退出仍关闭真实文件。"""
            with original(*args, **kwargs) as handle:
                calls.append(handle.name)
                if len(calls) == 2:
                    proxy = Mock(wraps=handle)
                    proxy.name = handle.name
                    proxy.write.side_effect = OSError("synthetic write failure")
                    yield proxy
                else:
                    yield handle

        monkeypatch.setattr(binding_io, "NamedTemporaryFile", temporary_file)
    with pytest.raises(OSError, match=f"synthetic {failure} failure"):
        binding_io.publish_files(((report, "new-report"), (config, "new-config")))
    assert report.read_text() == "old-report"
    assert config.read_text() == "old-config"
    assert not list(tmp_path.glob(".binding-*.tmp"))
    if failure == "replace":
        assert replacement.call_count == 1
    else:
        assert len(calls) == 2


def test_build_failure_codes_hide_unexpected_exception_payloads():
    """CLI详情只透出白名单固定码；输入任意文件错误文本输出binding_io_failed。

    已知binding_limits_invalid保留，未知异常不返回文件路径或来源内容。
    """
    from tushare_entitlement_hub.equity_research.binding_builder import build_failure_code

    assert build_failure_code(ValueError("binding_limits_invalid")) == "binding_limits_invalid"
    assert build_failure_code(OSError("synthetic private payload")) == "binding_io_failed"


@pytest.mark.parametrize(
    ("period", "cutoff", "expected"),
    [
        ("2026-Q2", "2026-10-03T12:00:00Z", True),
        ("2026-Q4", "2026-10-03T12:00:00Z", False),
        ("2026-Q3", "2026-09-30T15:59:59.999998Z", False),
        ("2026-Q3", "2026-09-30T15:59:59.999999Z", True),
        ("2026-09", "2026-09-30T15:59:59.999998Z", False),
        ("2026-09", "2026-09-30T15:59:59.999999Z", True),
        ("2026-10", "2026-10-03T12:00:00Z", False),
        ("2025", "2025-12-31T15:59:59.999998Z", False),
        ("2025", "2025-12-31T15:59:59.999999Z", True),
        ("2026", "2026-10-03T12:00:00Z", False),
        ("2026-10-04", "2026-10-03T20:00:00Z", True),
        ("2026-10-04", "2026-10-03T15:59:59Z", False),
    ],
)
def test_period_visibility_uses_market_calendar_bounds(period, cutoff, expected):
    """日按上海本地日期判定，月季年必须覆盖真实期间终点，不按文本字典序比较。

    例如2026-Q2在10月已完成；9月30日15:59:59.999999Z恰好是上海第三季度闭区间终点。
    """
    from tushare_entitlement_hub.equity_research.binding_builder import period_visible

    assert period_visible(period, cutoff, "Asia/Shanghai") is expected


@pytest.mark.parametrize(
    ("api", "period_field", "completed", "future"),
    [
        ("cn_gdp", "quarter", "2026Q2", "2026Q4"),
        ("cn_cpi", "month", "202609", "202610"),
    ],
)
def test_builder_preserves_completed_period_and_rejects_incomplete_period(
    tmp_path,
    api,
    period_field,
    completed,
    future,
):
    """完整构建保留已结束月/季度的新鲜度基准，并明确拒绝尚未结束的期间。

    输入cn_gdp.quarter=2026Q2输出eligible=1、expected=2026-Q2；Q4输出未来期间原因。
    """
    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import build_research_config
    from tushare_entitlement_hub.standard import normalize_period

    for label, value in [("completed", completed), ("future", future)]:
        reference = write_reference(tmp_path / label, {api: [{period_field: value}]})
        result = build_research_config(reference, AS_OF)
        source = result.audit["sources"][api]
        if label == "completed":
            assert source["eligible_rows"] == 1
            assert result.config.foundation.expected_data_as_of[api] == normalize_period(value)
        else:
            assert source["eligible_rows"] == 0
            assert source["reason_counts"]["future_business_date"] == 1
            assert api not in result.config.foundation.expected_data_as_of


def test_daily_business_date_uses_market_date_across_utc_midnight(tmp_path):
    """上海已跨日但UTC仍前一天时，当日行情不能误报未来数据。

    输入UTC10月3日20时分析、trade_date=10月4日，输出daily两行均可用且新鲜度10月4日。
    """
    from test_research_binding_fixture import default_rows, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import build_research_config

    daily = default_rows()["daily"]
    for row in daily:
        row["trade_date"] = "20261004"
    result = build_research_config(
        write_reference(tmp_path, {"daily": daily}), "2026-10-03T20:00:00Z"
    )
    assert result.audit["sources"]["daily"]["eligible_rows"] == 2
    assert result.config.foundation.expected_data_as_of["daily"] == "2026-10-04"


def test_builder_only_projects_consumer_numeric_fields(tmp_path):
    """只有消费注册选中的数值字段参与证书和缺口统计，其他数值列不扩大配置范围。

    输入daily含open/high/low/change等非消费数值，投影排除这些列且审计不记录其单位缺口。
    """
    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.catalog import load_policies
    from tushare_entitlement_hub.equity_research.binding_builder import (
        build_research_config,
        projection_fields,
    )
    from tushare_entitlement_hub.equity_research.binding_rules import time_registry
    from tushare_entitlement_hub.reference_contract import load_reference_contracts

    contracts = load_reference_contracts(load_policies())
    mapping = next(item for item in time_registry(contracts) if item.source_api == "daily")
    fields = {"ts_code", "trade_date", "amount", "vol", "open", "high", "low", "change"}
    assert projection_fields(fields, contracts["daily"], mapping) == {
        "ts_code",
        "trade_date",
        "amount",
        "vol",
    }
    result = build_research_config(write_reference(tmp_path), AS_OF)
    excluded = {"open", "high", "low", "change"}
    certificates = {
        item.field for item in result.config.foundation.certifications if item.source_api == "daily"
    }
    assert not certificates & excluded
    assert not set(result.audit["sources"]["daily"]["unit_gaps"]) & excluded


def test_normalized_amount_missing_standard_unit_never_gets_scaled_again(tmp_path):
    """真实normalize_reference_row已将100千元转100000元后，删标准单位必须拒绝认证。

    输入两条已规范化daily，移除standard_units.amount但保留source_units千元；
    构建后amount无证书，unit_standard_missing=2，数据库金额仍为100000和101000。
    """
    from test_research_binding_fixture import AS_OF, write_reference

    from tushare_entitlement_hub.equity_research.binding_builder import build_research_config

    reference = write_reference(tmp_path)
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        rows = connection.execute("SELECT ts_code, standard_units FROM accepted.daily").fetchall()
        for code, encoded in rows:
            units = json.loads(encoded)
            units.pop("amount")
            connection.execute(
                "UPDATE accepted.daily SET standard_units=? WHERE ts_code=?",
                [json.dumps(units), code],
            )
        amounts = connection.execute(
            "SELECT amount FROM accepted.daily ORDER BY ts_code"
        ).fetchall()
        assert amounts == [(100000.0,), (101000.0,)]
    result = build_research_config(reference, AS_OF)
    assert ("daily", "amount") not in {
        (item.source_api, item.field) for item in result.config.foundation.certifications
    }
    assert result.audit["sources"]["daily"]["unit_gaps"]["amount"] == {"unit_standard_missing": 2}
