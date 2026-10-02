"""通过 Typer 入口离线验证全命令与错误退出状态。"""

import json

import httpx
import pytest
from typer.testing import CliRunner

from tushare_entitlement_hub import cli
from tushare_entitlement_hub.client import HttpTransport
from tushare_entitlement_hub.models import CatalogEntry
from tushare_entitlement_hub.store import Store

OBS = "2024-01-03T00:00:00Z"


def invoke(tmp_path, arguments):
    """调用隔离数据库CLI；输入参数列表，输出CliRunner结果，所有文件限tmp_path。"""
    return CliRunner().invoke(cli.app, ["--db", str(tmp_path / "hub.duckdb"), *arguments])


def test_cli_survey_live_opt_in_and_credential_file(tmp_path, entry, monkeypatch):
    """场景：批量探测显式联网且凭据来自指定文件；预期：拒绝隐式联网并生成样本。"""
    assert invoke(tmp_path, ["capability", "survey"]).exit_code == 1
    assert invoke(tmp_path, ["capability", "survey", "--live"]).exit_code == 1
    store = Store(tmp_path / "hub.duckdb")
    store.save_catalog([entry])
    store.close()
    assert invoke(tmp_path, ["capability", "survey", "--live", "--apis", "unknown"]).exit_code == 1
    credentials = tmp_path / "credentials.env"
    credentials.write_text("TUSHARE_TOKEN=fixture-value\n")

    def handler(request):
        """离线确认发送时取得凭据；输入HTTP请求，返回固定公开数据。"""
        assert json.loads(request.content)["token"] == "fixture-value"
        return httpx.Response(200, json={"code": 0, "data": {
            "fields": ["ts_code", "trade_date", "close"], "items": [["A.SH", "20240102", 2]],
        }})

    def transport(provider, live):
        """构造模拟HTTPS传输，确保测试不访问真实网络。"""
        return HttpTransport(provider, live=live, http_transport=httpx.MockTransport(handler))

    monkeypatch.setattr(cli, "HttpTransport", transport)
    result = invoke(tmp_path, ["capability", "survey", "--live", "--credential-file",
                              str(credentials), "--output", str(tmp_path / "survey")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "survey/report.json").exists()
    assert "fixture-value" not in result.output


def test_cli_full_offline_workflow(tmp_path, entry):
    """场景：探活、计划、执行、审计、产品与债务；预期：CLI闭环成功。

    示例：fixture两日 → 2个committed。"""
    store = Store(tmp_path / "hub.duckdb")
    store.save_catalog([entry])
    store.close()
    response = {
        "code": 0,
        "data": {"fields": ["ts_code", "trade_date", "close"], "items": [["A.SH", "20240102", 2]]},
    }
    fixture = tmp_path / "responses.json"
    fixture.write_text(json.dumps({entry.api_name: [response]}))
    raw = str(tmp_path / "raw")
    result = invoke(
        tmp_path,
        [
            "capability",
            "probe",
            entry.api_name,
            "--fixture",
            str(fixture),
            "--raw-directory",
            raw,
            "--observed-at",
            OBS,
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "available"
    plan = tmp_path / "plan.json"
    parameters = tmp_path / "parameters.json"
    parameters.write_text('{"trade_dates":["2024-01-02"]}')
    result = invoke(
        tmp_path,
        [
            "ingest",
            "plan",
            entry.api_name,
            "2024-01-02",
            "2024-01-03",
            "--output",
            str(plan),
            "--parameters",
            str(parameters),
            "--observed-at",
            OBS,
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["tasks"] == 1
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                entry.api_name: {
                    "expected_dates": ["2024-01-02"],
                    "previous_fields": ["ts_code", "trade_date", "close"],
                }
            }
        )
    )
    arguments = [
        "ingest",
        "run",
        str(plan),
        "--fixture",
        str(fixture),
        "--raw-directory",
        raw,
        "--references",
        str(reference),
        "--observed-at",
        OBS,
    ]
    result = invoke(tmp_path, arguments)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["state"] == "committed"
    assert json.loads(invoke(tmp_path, arguments).output)[0]["state"] == "already_committed"
    result = invoke(tmp_path, ["quality", "audit", entry.api_name, "--references", str(reference)])
    assert result.exit_code == 0
    assert json.loads(result.output) == []
    result = invoke(
        tmp_path,
        [
            "products",
            "build",
            "--output",
            str(tmp_path / "products"),
            "--parquet",
            str(tmp_path / "standard.parquet"),
        ],
    )
    assert result.exit_code == 0 and len(json.loads(result.output)["files"]) == 20
    output = tmp_path / "coverage.json"
    result = invoke(tmp_path, ["coverage", "report", "--output", str(output), "--gate"])
    assert result.exit_code == 0 and json.loads(output.read_text())["gate_passed"]
    result = invoke(
        tmp_path,
        [
            "backfill",
            "plan",
            entry.api_name,
            "2024-01-01",
            "2024-01-01",
            "--output",
            str(tmp_path / "backfill.json"),
            "--observed-at",
            OBS,
        ],
    )
    assert result.exit_code == 0
    assert json.loads((tmp_path / "backfill.json").read_text())[0]["mode"] == "backfill"
    assert len(json.loads(invoke(tmp_path, ["catalog", "list"]).output)) == 1
    assert (
        json.loads(invoke(tmp_path, ["catalog", "list", "--lifecycle", "unclassified"]).output)
        == []
    )


def test_cli_error_boundaries_and_source_mismatch(tmp_path, entry):
    """场景：缺目录、冲突来源、质量失败；预期：固定安全错误非零退出。

    示例：未知API → operation_failed。"""
    assert invoke(tmp_path, ["--help"]).exit_code == 0
    assert not (tmp_path / "hub.duckdb").exists()
    assert invoke(tmp_path, ["capability", "probe", "unknown"]).exit_code == 1
    assert invoke(tmp_path, ["coverage", "report", "--gate"]).exit_code == 1
    database = Store(tmp_path / "hub.duckdb")
    database.save_catalog([entry, CatalogEntry("unknown")])
    database.close()
    assert invoke(tmp_path, ["quality", "audit", "unknown"]).exit_code == 1
    assert invoke(tmp_path, ["quality", "audit", entry.api_name]).exit_code == 1
    assert invoke(tmp_path, ["capability", "probe", entry.api_name]).exit_code == 1
    from dataclasses import asdict

    from tushare_entitlement_hub.store import digest

    database = Store(tmp_path / "hub.duckdb")
    database.record_probe(entry.api_name, OBS, "live", "available", {
        "account": "local", "request_policy": digest(asdict(entry.policy)),
    })
    database.close()
    plan = tmp_path / "live.json"
    assert (
        invoke(
            tmp_path,
            [
                "ingest",
                "plan",
                entry.api_name,
                "2024-01-01",
                "2024-01-01",
                "--live",
                "--output",
                str(plan),
                "--observed-at",
                OBS,
            ],
        ).exit_code
        == 0
    )
    assert invoke(tmp_path, ["ingest", "run", str(plan)]).exit_code == 1
    assert invoke(tmp_path, ["catalog", "sync"]).exit_code == 1


def test_catalog_sync_file_directory_and_mock_online(tmp_path, monkeypatch):
    """场景：文件、目录及注入的公开文档下载；预期：同接口幂等登记。

    示例：daily文档 → registered=1。"""
    directory = tmp_path / "docs"
    directory.mkdir()
    document = directory / "daily.md"
    document.write_text(
        "# 日线行情\n\n接口：daily\n\n积分：2000\n\n每次最多6000行\n", encoding="utf-8"
    )
    for path in [document, directory]:
        result = invoke(tmp_path, ["catalog", "sync", "--fixture", str(path), "--observed-at", OBS])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["registered"] == 1
    calls = []

    def fetch(destination):
        """模拟公开文档下载依赖；输入目标目录，输出已有本地文档路径并记录调用。"""
        calls.append(destination)
        return [document]

    monkeypatch.setattr(cli, "fetch_documentation", fetch)
    result = invoke(
        tmp_path,
        ["catalog", "sync", "--online", "--destination", str(directory), "--observed-at", OBS],
    )
    assert result.exit_code == 0 and calls == [directory]


def test_open_client_credential_prompt_mock_http_and_json(tmp_path, entry, monkeypatch):
    """场景：显式live但HTTP离线注入；预期：最后发送读取占位凭据。

    示例：MockTransport → available。"""
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "<REDACTED>")
    assert cli.credential_prompt() == "<REDACTED>"
    assert cli.timestamp(OBS) == OBS
    assert cli.timestamp(None).endswith("Z")
    payload = {"code": 0, "data": {"fields": ["ts_code"], "items": [["A.SH"]]}}
    requests = []

    def handler(request):
        """离线HTTP边界；输入请求对象，输出固定成功响应，保存仅公开请求路径。"""
        requests.append(str(request.url))
        return httpx.Response(200, json=payload)

    def transport(provider, live):
        """注入MockTransport；输入凭据回调及live标记，输出复用HTTP客户端。"""
        return HttpTransport(provider, live=live, http_transport=httpx.MockTransport(handler))

    monkeypatch.setattr(cli, "HttpTransport", transport)
    with cli.open_client(entry, tmp_path / "raw", None, True) as client:
        assert client.fetch(entry.api_name, {}, OBS).rows == [{"ts_code": "A.SH"}]
    assert requests == ["https://api.tushare.pro"]
    fixture = tmp_path / "response.json"
    fixture.write_text(json.dumps([payload]))
    with cli.open_client(entry, tmp_path / "raw2", fixture, False) as client:
        assert client.fetch(entry.api_name, {}, OBS).fields == ["ts_code"]
    with pytest.raises(ValueError, match="choose_exactly"):
        with cli.open_client(entry, tmp_path / "raw", fixture, True):
            pass
    fixture.write_text('{"token":"<REDACTED>"}')
    with pytest.raises(ValueError, match="凭据"):
        cli.read_json(fixture)


def test_quality_audit_keeps_historical_revisions_separate(tmp_path, entry, policy):
    """场景：同主键多个历史观察版本；预期：审计不把合法修订当批内重复。

    示例：两次观察close分别2和3 → quality audit无duplicate_primary_key。
    """
    from tushare_entitlement_hub.ingest import plan_tasks
    from tushare_entitlement_hub.standard import normalize_rows

    database = Store(tmp_path / "hub.duckdb")
    database.save_catalog([entry])
    for observed, close in [(OBS, 2), ("2024-01-04T00:00:00Z", 3)]:
        task = plan_tasks(
            entry, "2024-01-02", "2024-01-02", "lookback" if close == 2 else "event", observed
        )[0]
        database.begin_task(task, observed, "fixture")
        rows = normalize_rows(
            [{"ts_code": "A.SH", "trade_date": "20240102", "close": close}],
            policy,
            observed,
            "public-hash",
        )
        database.commit_task(
            task, rows, policy.primary_key, observed, "fixture", list(rows[0]["source_fields"])
        )
    database.close()
    result = invoke(tmp_path, ["quality", "audit", entry.api_name])
    assert result.exit_code == 0, result.output
    assert "duplicate_primary_key" not in {item["code"] for item in json.loads(result.output)}


@pytest.mark.parametrize(
    "value,error",
    [
        ("", "invalid_observed_at"),
        ("not-a-date", "invalid_observed_at"),
        ("2024-01-03", "invalid_observed_at"),
        ("2024-01-03T00:00:00", "invalid_observed_at"),
        ("2024-13-03T00:00:00Z", "invalid_observed_at"),
        ("0001-01-01T00:00:00+14:00", "invalid_observed_at"),
        ("9999-12-31T23:59:59-14:00", "invalid_observed_at"),
        ("9999-12-30T00:00:00Z", "future_observed_at"),
    ],
)
def test_timestamp_rejects_invalid_or_future_observations(value, error):
    """场景：非法、无时区和未来观察时间；预期：固定错误阻止进入版本数据。

    示例：2024-01-03 或空字符串 → invalid_observed_at。
    """
    with pytest.raises(ValueError, match=error):
        cli.timestamp(value)


def test_timestamp_normalizes_offsets_and_accepts_current_instant(monkeypatch):
    """场景：偏移时区、当前边界与自动时间；预期：统一 UTC 且相等不误拒。

    示例：08:00+08:00 → 00:00Z；未传时间 → 固定当前时刻。
    """
    from datetime import UTC, datetime

    class FixedDatetime(datetime):
        """固定测试时钟，同时沿用标准 ISO 解析。"""

        @classmethod
        def now(cls, tz=None):
            """返回已知时刻；例如 now(UTC) → 2024-01-03T00:00:00+00:00。"""
            return datetime(2024, 1, 3, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(cli, "datetime", FixedDatetime)
    for value in [None, OBS, "2024-01-03T08:00:00+08:00", "2024-01-02T19:00:00-05:00"]:
        assert cli.timestamp(value) == OBS
    with pytest.raises(ValueError, match="future_observed_at"):
        cli.timestamp("2024-01-03T00:00:00.000001Z")


@pytest.mark.parametrize("value", ["not-a-date", "", "9999-12-30T00:00:00Z"])
def test_catalog_sync_invalid_observation_leaves_catalog_empty(tmp_path, value):
    """场景：CLI 明确传入非法观察时间；预期：非零退出且目录不被污染。

    示例：--observed-at not-a-date → operation_failed，catalog 仍为空。
    """
    document = tmp_path / "daily.md"
    document.write_text("# 日线\n\n接口：daily\n\n积分：2000\n")
    result = invoke(
        tmp_path, ["catalog", "sync", "--fixture", str(document), "--observed-at", value]
    )
    assert result.exit_code == 1
    assert "operation_failed" in result.output
    database = Store(tmp_path / "hub.duckdb")
    try:
        assert database.catalog() == []
    finally:
        database.close()


def test_quality_audit_later_batch_uses_actual_previous_schema(tmp_path, entry, policy):
    """场景：显式基线不含后来被删除的字段；预期：第二批仍按真实历史 schema 报删除。

    示例：首批有 vendor_status、第二批无 → schema_field_removed，退出 1。
    """
    from tushare_entitlement_hub.ingest import plan_tasks
    from tushare_entitlement_hub.standard import normalize_rows

    database = Store(tmp_path / "hub.duckdb")
    database.save_catalog([entry])
    for mode, observed, extra in [
        ("initialize", OBS, {"vendor_status": "ok"}),
        ("lookback", "2024-01-04T00:00:00Z", {}),
    ]:
        task = plan_tasks(entry, "2024-01-02", "2024-01-02", mode, observed)[0]
        database.begin_task(task, observed, "fixture")
        rows = normalize_rows(
            [{"ts_code": "A.SH", "trade_date": "20240102", "close": 2, **extra}],
            policy,
            observed,
            "response",
        )
        database.commit_task(
            task, rows, policy.primary_key, observed, "fixture", list(rows[0]["source_fields"])
        )
    database.close()
    reference = tmp_path / "references.json"
    reference.write_text(
        json.dumps(
            {
                entry.api_name: {
                    "previous_fields": ["ts_code", "trade_date", "close"],
                    "expected_dates": ["20240102"],
                }
            }
        )
    )
    result = invoke(tmp_path, ["quality", "audit", entry.api_name, "--references", str(reference)])
    assert result.exit_code == 1, result.output
    findings = json.loads(result.output)
    assert any(
        item["code"] == "schema_field_removed" and item["field"] == "vendor_status"
        for item in findings
    )
    assert "schema_drift_not_checked" not in {item["code"] for item in findings}
