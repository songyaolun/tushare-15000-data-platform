"""独立合成参考库逐项证明99接口实际读取、消费、留证及CLI幂等。"""

import json
from collections import defaultdict

import duckdb
import pytest
from consumption_fixture import consumption_payloads, write_consumption_reference
from research_fixture import AS_OF, DATABASE_CONFIG
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.models import ResearchConfig
from tushare_entitlement_hub.equity_research.pipeline import run_research
from tushare_entitlement_hub.product_foundation.registry import load_registry
from tushare_entitlement_hub.reference_storage import file_checksum

ASSIGNMENTS = tuple(item for item in load_registry() if item.product in PRODUCT_ALIASES)


@pytest.fixture(scope="module")
def complete_consumption_run(tmp_path_factory):
    """构造真实99接口合成库并两次运行CLI；例：99个API均可在保存的消费详情定位。

    使用模块级夹具共享同一次真实运行，每个API断言自己的行、字段、角色和引用。
    """
    directory = tmp_path_factory.mktemp("complete-consumption")
    reference, config = write_consumption_reference(directory)
    configuration = directory / "config.json"
    configuration.write_text(config.to_json())
    output = directory / "products.duckdb"
    fingerprint = file_checksum(reference)
    arguments = [
        "research",
        "run",
        "--reference",
        str(reference),
        "--output",
        str(output),
        "--config",
        str(configuration),
        "--as-of",
        AS_OF,
    ]
    runner = CliRunner()
    first = runner.invoke(app, arguments)
    assert first.exit_code == 2, first.output
    second = runner.invoke(app, arguments)
    assert second.exit_code == 2, second.output
    summary = json.loads(first.stdout)
    assert json.loads(second.stdout) == summary
    assert json.loads(second.stderr.splitlines()[-1])["stage"] == "reused"
    validated = runner.invoke(
        app, ["research", "validate", "--reference", str(reference), "--output", str(output)]
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout)["valid"] is True
    assert file_checksum(reference) == fingerprint
    records = defaultdict(list)
    references = set()
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        cursor = connection.execute("SELECT payload FROM research.result")
        while batch := cursor.fetchmany(128):
            for (payload,) in batch:
                result = json.loads(payload)
                details = json.loads(result["details_json"])
                for record in details["consumption"]:
                    records[record["api"]].append(record)
        cursor = connection.execute("SELECT payload FROM research.evidence")
        while batch := cursor.fetchmany(128):
            for (payload,) in batch:
                reference_data = json.loads(payload)
                references.add((reference_data["source_api"], reference_data["source_record_id"]))
    return reference, summary, records, references


@pytest.mark.parametrize("assignment", ASSIGNMENTS, ids=[item.api for item in ASSIGNMENTS])
def test_each_of_99_apis_has_read_rows_and_persisted_product_evidence(
    complete_consumption_run, assignment
):
    """场景：矩阵每个接口有合成合格记录；预期：真实消费本行且证据进入独立产品库。

    本断言不以白名单包含关系代替消费，逐一核对实际行数、字段质量、角色和持久化引用。
    """
    _, summary, records, references = complete_consumption_run
    observations = [item for item in records[assignment.api] if item["retained_row_count"] > 0]
    assert observations, assignment.api
    for item in observations:
        assert item["product"] == assignment.product
        assert item["role"] == assignment.role
        assert item["purpose"] == assignment.purpose
        assert item["executed"] is True
        assert any(value["present_rows"] > 0 for value in item["field_quality"].values())
        assert item["facts"], assignment.api
        assert item["gates"], assignment.api
    own = [
        evidence
        for item in observations
        for evidence in item["evidence_refs"]
        if evidence["source_api"] == assignment.api
        and evidence["source_record_id"] == f"synthetic/{assignment.api}/0"
        and evidence["fields"]
    ]
    assert own, assignment.api
    assert (assignment.api, f"synthetic/{assignment.api}/0") in references
    assert sum(summary["consumption"]["by_api"][assignment.api].values()) > 0


def test_full_cli_summary_and_quarantine_are_honest(complete_consumption_run):
    """场景：99张accepted表另含指数因子隔离毒值；预期：只扫描99合格行并保留历史降级。"""
    _, summary, records, references = complete_consumption_run
    assert len(consumption_payloads()) == len(records) == 99
    assert summary["processing"]["scanned_rows"] == 99
    assert summary["processing"]["retained_rows"] == 99
    assert summary["processing"]["excluded_rows"] == 0
    consumption = summary["consumption"]
    assert consumption["assigned_apis"] == consumption["registered_apis"] == 99
    assert consumption["executed_apis"] == 99
    assert consumption["status_counts"]["not_enabled_due_to_history"] > 0
    assert summary["historical_pit_certified"] is False
    assert all(identity != "synthetic/quarantine/poison" for _, identity in references)
    assert "999999" not in json.dumps(records["idx_factor_pro"])


def test_all_99_missing_bindings_still_report_unknown(complete_consumption_run, tmp_path):
    """场景：同一99接口参考库没有任何绑定；预期：不扫描行，逐来源执行降级且不假称观测。"""
    reference = complete_consumption_run[0]
    summary = run_research(reference, tmp_path / "unbound.duckdb", AS_OF, ResearchConfig())
    assert summary["processing"]["scanned_rows"] == 0
    assert summary["consumption"]["executed_apis"] == 99
    assert summary["consumption"]["observed_apis"] == 0
    # 一个接口可被多个事件结果消费，逐结果计数；每一次都必须保持unknown。
    assert all(
        set(states) == {"unknown"} and states["unknown"] > 0
        for states in summary["consumption"]["by_api"].values()
    )
