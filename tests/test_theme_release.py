"""P02 参考库读取、范围门禁和不可变本地发布测试。"""

import json
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.cross_asset_research import theme_release as release_module
from tushare_entitlement_hub.cross_asset_research.theme_release import (
    _read_records,
    build_cross_source_consensus,
    build_theme_release,
    load_scope_certificates,
    load_source_theme_rankings,
    normalize_theme_name,
    publish_theme_release,
    resolve_theme_trade_date,
    validate_theme_release,
)
from tushare_entitlement_hub.product_readiness import P02_SCOPE_REQUIREMENTS
from tushare_entitlement_hub.reference_storage import atomic_json, file_checksum
from tushare_entitlement_hub.store import encode

TARGET = "2026-10-09"


def write_reference(path, *, complete=True):
    """创建含一个完整五成员主题的轻量参考库；complete=False 缺少热度范围证书。"""
    days = [f"2026-09-{day:02d}" for day in range(10, 30)] + [TARGET]
    members = [f"S{index}.SZ" for index in range(5)]
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE SCHEMA accepted")
        connection.execute(
            "CREATE TABLE accepted.ths_index(ts_code VARCHAR,name VARCHAR,"
            "observed_at TIMESTAMPTZ,source_record_id VARCHAR)"
        )
        connection.execute(
            "INSERT INTO accepted.ths_index VALUES ('T1.TI','合成主题','2026-10-10','index')"
        )
        connection.execute(
            "CREATE TABLE accepted.ths_member(ts_code VARCHAR,con_code VARCHAR,"
            "observed_at TIMESTAMPTZ,source_record_id VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO accepted.ths_member VALUES ('T1.TI',?,'2026-10-10',?)",
            [(member, f"member-{member}") for member in members],
        )
        connection.execute(
            "CREATE TABLE accepted.daily(ts_code VARCHAR,trade_date DATE,pct_chg DOUBLE,"
            "close DOUBLE,source_record_id VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO accepted.daily VALUES (?,?::DATE,0.01,10,?)",
            [(member, TARGET, f"daily-{member}") for member in members],
        )
        connection.execute(
            "CREATE TABLE accepted.stk_limit(ts_code VARCHAR,trade_date DATE,up_limit DOUBLE,"
            "source_record_id VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO accepted.stk_limit VALUES (?,?::DATE,?,?)",
            [
                (member, TARGET, 10 if index < 2 else 11, f"limit-{member}")
                for index, member in enumerate(members)
            ],
        )
        connection.execute(
            "CREATE TABLE accepted.ths_hot(ts_code VARCHAR,trade_date DATE,rank DOUBLE,"
            "source_record_id VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO accepted.ths_hot VALUES (?,?::DATE,?,?)",
            [(member, TARGET, index + 1, f"hot-{member}") for index, member in enumerate(members)],
        )
        connection.execute(
            "CREATE TABLE accepted.moneyflow_cnt_ths(ts_code VARCHAR,trade_date DATE,"
            "net_amount DOUBLE,company_num DOUBLE,source_record_id VARCHAR)"
        )
        connection.execute(
            "INSERT INTO accepted.moneyflow_cnt_ths VALUES ('T1.TI',?::DATE,100,5,'flow')",
            [TARGET],
        )
        connection.execute(
            "CREATE TABLE accepted.kpl_list(ts_code VARCHAR,trade_date DATE,status VARCHAR,"
            "theme VARCHAR,lu_desc VARCHAR,source_record_id VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO accepted.kpl_list VALUES (?,?::DATE,?,'AI','原因',?)",
            [
                (members[0], TARGET, "3连板", "event-0"),
                (members[1], TARGET, "首板", "event-1"),
            ],
        )
        connection.execute(
            "CREATE TABLE scope_certificate AS SELECT * FROM (VALUES "
            "('seed','seed','certified',TIMESTAMPTZ '2026-10-10','{}')) "
            "t(certificate_id,api_name,status,observed_at,payload)"
        )
        connection.execute("DELETE FROM scope_certificate")
        for requirement in P02_SCOPE_REQUIREMENTS:
            if not complete and requirement.api_name == "ths_hot":
                continue
            date_values = days if requirement.minimum_certified_days else []
            payload = {
                "certificate_id": "cert-" + requirement.api_name,
                "api_name": requirement.api_name,
                "status": "certified",
                "observed_at": "2026-10-10T00:00:00Z",
                "data_dates": {requirement.date_field: date_values} if date_values else {},
                "parameter_dates": {},
                "completed_partition_count": (1666 if requirement.api_name == "ths_member" else 1),
            }
            connection.execute(
                "INSERT INTO scope_certificate VALUES (?,?,?,?,?)",
                [
                    payload["certificate_id"],
                    requirement.api_name,
                    "certified",
                    payload["observed_at"],
                    encode(payload),
                ],
            )
        connection.execute("CHECKPOINT")
    return path


def test_theme_release_builds_ready_main_ranking_and_reusable_publication(tmp_path):
    """场景：九项范围和五成员主题完整；预期：ready 主榜、Parquet、视图和幂等发布。"""
    reference = write_reference(tmp_path / "reference.duckdb")
    release = build_theme_release(reference, "2026-10-10")
    assert release["target_date"] == TARGET
    assert release["readiness"]["status"] == "ready"
    assert len(release["main_ranking"]) == len(release["price_diffusion"]) == 1
    assert release["main_ranking"][0]["effective_member_count"] == 5
    output = tmp_path / "p02"
    manifest = publish_theme_release(output, release)
    assert publish_theme_release(output, release) == manifest
    assert json.loads((output / "current.json").read_text())["release_id"] == release["release_id"]
    with duckdb.connect(manifest["database"], read_only=True) as connection:
        assert (
            connection.execute("SELECT theme_name FROM p02.main_ranking").fetchone()[0]
            == "合成主题"
        )
        assert connection.execute("SELECT count(*) FROM p02.evidence_link").fetchone()[0] > 5
    validation = validate_theme_release(reference, output)
    assert validation["valid"] and validation["counts"]["main_ranking"] == 1
    cli_validation = CliRunner().invoke(
        app,
        [
            "research-all",
            "p02-validate",
            "--reference",
            str(reference),
            "--output",
            str(output),
        ],
    )
    assert cli_validation.exit_code == 0 and json.loads(cli_validation.output)["valid"]
    cli_build = CliRunner().invoke(
        app,
        [
            "research-all",
            "p02-build",
            "--reference",
            str(reference),
            "--output",
            str(tmp_path / "cli-ready"),
        ],
    )
    assert cli_build.exit_code == 0


def test_incomplete_scope_keeps_price_diffusion_but_suppresses_main(tmp_path):
    """场景：热度范围证书缺失但实际行情存在；预期：不发综合主榜，价格扩散仍可用。"""
    reference = write_reference(tmp_path / "reference.duckdb", complete=False)
    release = build_theme_release(reference)
    assert release["readiness"]["status"] == "incomplete"
    assert "scope_certificate_missing:ths_hot" in release["readiness"]["reason_codes"]
    assert release["main_ranking"] == [] and len(release["price_diffusion"]) == 1


def test_p02_cli_returns_two_for_honest_degraded_release(tmp_path):
    """场景：CLI 构建范围未齐的 P02；预期：产物落地、输出摘要并以退出码2提示降级。"""
    reference = write_reference(tmp_path / "reference.duckdb", complete=False)
    result = CliRunner().invoke(
        app,
        [
            "research-all",
            "p02-build",
            "--reference",
            str(reference),
            "--output",
            str(tmp_path / "p02"),
        ],
    )
    assert result.exit_code == 2
    assert json.loads(result.output)["readiness"]["status"] == "incomplete"
    failed = CliRunner().invoke(
        app,
        [
            "research-all",
            "p02-build",
            "--reference",
            str(tmp_path / "missing.duckdb"),
            "--output",
            str(tmp_path / "failed"),
        ],
    )
    assert failed.exit_code == 1


def test_p02_validation_rejects_changed_product_file(tmp_path):
    """场景：发布后篡改主榜 Parquet；预期：validate 返回固定失败，不读取坏载荷。"""
    reference = write_reference(tmp_path / "reference.duckdb")
    output = tmp_path / "p02"
    manifest = publish_theme_release(output, build_theme_release(reference))
    with open(manifest["files"]["main_ranking"]["path"], "ab") as handle:
        handle.write(b"changed")
    assert validate_theme_release(reference, output) == {
        "valid": False,
        "reason_codes": ["p02_validation_failed"],
    }
    cli_result = CliRunner().invoke(
        app,
        [
            "research-all",
            "p02-validate",
            "--reference",
            str(reference),
            "--output",
            str(output),
        ],
    )
    assert cli_result.exit_code == 1


def rewrite_parquet(path, query):
    """用 DuckDB 查询原文件并原子替换，供验证故障注入测试使用。"""
    temporary = path.with_suffix(".changed.parquet")
    escaped = str(path).replace("'", "''")
    with duckdb.connect() as connection:
        connection.execute(
            f"COPY ({query.format(path=escaped)}) TO ? (FORMAT PARQUET)", [str(temporary)]
        )
    temporary.replace(path)


@pytest.mark.parametrize(
    "scenario",
    ["reference", "database", "rows", "identity", "main_gate", "evidence", "ready_empty"],
)
def test_p02_validation_rejects_each_release_integrity_failure(tmp_path, scenario):
    """场景：逐项破坏来源、数据库、计数、身份、门禁或证据；预期：统一失败关闭。"""
    reference = write_reference(tmp_path / "reference.duckdb")
    output = tmp_path / "p02"
    manifest = publish_theme_release(output, build_theme_release(reference))
    current = json.loads((output / "current.json").read_text())
    validation_reference = reference
    if scenario == "reference":
        validation_reference = write_reference(tmp_path / "other.duckdb")
        with duckdb.connect(str(validation_reference)) as connection:
            connection.execute(
                "INSERT INTO accepted.daily VALUES ('OTHER',?::DATE,0,1,'other')", [TARGET]
            )
    elif scenario == "database":
        current["database_sha256"] = "different"
    elif scenario == "rows":
        current["files"]["main_ranking"]["rows"] += 1
    elif scenario == "identity":
        with duckdb.connect(manifest["database"]) as connection:
            connection.execute(
                'UPDATE release SET payload=json_merge_patch(payload, \'{"release_id":"bad"}\')'
            )
        current["database_sha256"] = file_checksum(Path(manifest["database"]))
    elif scenario == "main_gate":
        item = current["files"]["main_ranking"]
        path = Path(item["path"])
        rewrite_parquet(
            path, "SELECT * REPLACE (NULL::DOUBLE AS score) FROM read_parquet('{path}')"
        )
        item.update(sha256=file_checksum(path), bytes=path.stat().st_size)
    elif scenario == "evidence":
        item = current["files"]["evidence_link"]
        path = Path(item["path"])
        rewrite_parquet(
            path,
            "SELECT * FROM read_parquet('{path}') UNION ALL SELECT 'ORPHAN','missing'",
        )
        item.update(sha256=file_checksum(path), bytes=path.stat().st_size, rows=item["rows"] + 1)
    else:
        item = current["files"]["main_ranking"]
        path = Path(item["path"])
        rewrite_parquet(path, "SELECT * FROM read_parquet('{path}') LIMIT 0")
        item.update(sha256=file_checksum(path), bytes=path.stat().st_size, rows=0)
    atomic_json(output / "current.json", current)
    assert validate_theme_release(validation_reference, output) == {
        "valid": False,
        "reason_codes": ["p02_validation_failed"],
    }


def test_trade_date_resolution_fails_closed_without_daily(tmp_path):
    """场景：参考库没有日行情表；预期：目标日解析明确失败，不使用系统日期。"""
    path = tmp_path / "empty.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE SCHEMA accepted")
        with pytest.raises(ValueError, match="p02_daily_missing"):
            resolve_theme_trade_date(connection, None)


def test_release_read_helpers_fail_closed_and_support_fallback_source_loading(tmp_path):
    """场景：无证书、空日表、缺表/缺字段及未传 THS 观察；预期：关闭日期并走来源回退。"""
    empty = tmp_path / "empty-fields.duckdb"
    with duckdb.connect(str(empty)) as connection:
        connection.execute("CREATE SCHEMA accepted")
        connection.execute("CREATE TABLE accepted.daily(trade_date DATE,value VARCHAR)")
        assert load_scope_certificates(connection) == ()
        assert _read_records(connection, "missing", ("ts_code",)) == []
        assert _read_records(connection, "daily", ("ts_code",)) == []
        with pytest.raises(ValueError, match="p02_trade_date_unavailable"):
            resolve_theme_trade_date(connection, None)
    reference = write_reference(tmp_path / "fallback.duckdb")
    with duckdb.connect(str(reference)) as connection:
        connection.execute(
            "INSERT INTO accepted.ths_member VALUES "
            "('SMALL.TI','S0.SZ','2026-10-10','small-member')"
        )
        connection.execute(
            "CREATE TABLE accepted.kpl_concept_cons("
            'ts_code VARCHAR,con_code VARCHAR,trade_date DATE,"desc" VARCHAR,'
            "source_record_id VARCHAR)"
        )
        connection.execute(
            "INSERT INTO accepted.kpl_concept_cons VALUES "
            "('K1','S0.SZ',?::DATE,'合成题材','kpl-member')",
            [TARGET],
        )
    with duckdb.connect(str(reference), read_only=True) as connection:
        rankings, members = load_source_theme_rankings(connection, TARGET)
    assert len(rankings) == 1 and ("ths", "T1.TI") in members


def test_theme_build_detects_reference_change(tmp_path, monkeypatch):
    """场景：读取前后参考指纹变化；预期：拒绝生成混合快照。"""
    reference = write_reference(tmp_path / "reference.duckdb")
    values = iter(("before", "after"))
    monkeypatch.setattr(release_module, "reference_fingerprint", lambda _: next(values))
    with pytest.raises(ValueError, match="reference_changed_during_p02_read"):
        build_theme_release(reference)


def test_publish_rejects_release_collision_and_row_count_mismatch(tmp_path, monkeypatch):
    """场景：同目录身份冲突或写出行数声明错误；预期：原子发布前明确拒绝。"""
    reference = write_reference(tmp_path / "reference.duckdb")
    release = build_theme_release(reference)
    output = tmp_path / "p02"
    manifest = publish_theme_release(output, release)
    release_path = output / "releases" / release["release_id"][:20] / "release.json"
    atomic_json(release_path, json.loads(release_path.read_text()) | {"release_id": "different"})
    with pytest.raises(ValueError, match="p02_release_identity_collision"):
        publish_theme_release(output, release)
    original = release_module._write_parquet

    def wrong_count(path, columns, rows):
        """保留真实文件但把主榜声明行数加一，模拟写后计数不一致。"""
        result = original(path, columns, rows)
        if path.name == "main_ranking.parquet":
            result["rows"] += 1
        return result

    monkeypatch.setattr(release_module, "_write_parquet", wrong_count)
    with pytest.raises(ValueError, match="p02_release_row_count_mismatch"):
        publish_theme_release(tmp_path / "mismatch", release)
    assert manifest["release_id"] == release["release_id"]


def test_cross_source_consensus_requires_same_normalized_name_and_member_overlap():
    """场景：同名高重合、同名低重合和异名高重合；预期：只合并同名且重合达标的来源。"""
    rankings = [
        {"source": "ths", "theme_code": "T", "theme_name": "AI 传媒", "price_diffusion_score": 80},
        {"source": "dc", "theme_code": "D", "theme_name": "AI传媒", "price_diffusion_score": 70},
        {"source": "tdx", "theme_code": "X", "theme_name": "AI传媒", "price_diffusion_score": 90},
        {"source": "kpl", "theme_code": "K", "theme_name": "算力", "price_diffusion_score": 95},
    ]
    members = {
        ("ths", "T"): frozenset({"1", "2", "3", "4"}),
        ("dc", "D"): frozenset({"1", "2", "3", "5"}),
        ("tdx", "X"): frozenset({"8", "9"}),
        ("kpl", "K"): frozenset({"1", "2", "3", "4"}),
    }
    result = build_cross_source_consensus(rankings, members)
    assert normalize_theme_name("AI 传媒") == "ai传媒"
    assert len(result) == 1 and result[0]["sources"] == ["dc", "ths"]
    assert result[0]["member_jaccard"]["dc"] == pytest.approx(0.6)
