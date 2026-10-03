"""目录来源过滤的真实DuckDB回归，只读取研究注册中的99项清单。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import duckdb
import pytest
from test_research_binding_acceptance import build_arguments
from test_research_binding_fixture import AS_OF, DATABASE_CONFIG, write_reference
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.equity_research import binding_builder
from tushare_entitlement_hub.equity_research.consumption_registry import load_consumption_registry
from tushare_entitlement_hub.reference_storage import file_checksum

REQUIRED_APIS = tuple(sorted(item.spec.api for item in load_consumption_registry()))


@pytest.fixture(scope="module")
def catalog_reference_template(tmp_path_factory):
    """一次创建合成99清单模板，每个用例复制到自己的临时目录后独立修改。

    输入pytest临时目录工厂，输出4张accepted表、8条数据及99项必需清单的只读模板路径。
    """
    return write_reference(tmp_path_factory.mktemp("catalog-template"))


def copy_reference(template: Path, directory: Path) -> Path:
    """把已关闭的合成数据库复制到独立测试目录，避免用例间共享可变状态。

    输入模板与tmp_path，返回tmp_path/reference.duckdb；模板字节保持不变。
    """
    reference = directory / "reference.duckdb"
    shutil.copyfile(template, reference)
    return reference


def change_required_manifest(reference: Path, api: str, mutation: str) -> None:
    """按场景仅修改指定必需清单，保留其他清单及数据表。

    输入daily/missing删除清单；duplicate复制清单；hash把contract_hash改为合成错误值。
    """
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        if mutation == "missing":
            connection.execute("DELETE FROM catalog WHERE api_name=?", [api])
        elif mutation == "duplicate":
            connection.execute("INSERT INTO catalog SELECT * FROM catalog WHERE api_name=?", [api])
        else:
            manifest = json.loads(
                connection.execute(
                    "SELECT manifest FROM catalog WHERE api_name=?", [api]
                ).fetchone()[0]
            )
            manifest["contract_hash"] = "synthetic-wrong-contract"
            connection.execute(
                "UPDATE catalog SET manifest=? WHERE api_name=?", [json.dumps(manifest), api]
            )


@pytest.mark.parametrize("api", REQUIRED_APIS)
@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("missing", "reference_manifest_missing"),
        ("duplicate", "reference_catalog_duplicate"),
        ("hash", "source_contract_mismatch"),
    ],
)
def test_each_required_manifest_is_present_unique_and_contract_bound(
    tmp_path,
    catalog_reference_template,
    api,
    mutation,
    error_code,
):
    """逐项验证全部99必需清单的存在、唯一和合同绑定，不因来源没有accepted表而跳过。

    例如移除cn_gdp清单仍抛reference_manifest_missing；重复或错误hash分别返回固定码。
    任一失败均不创建配置/审计目录，也不修改输入库。
    """
    assert len(REQUIRED_APIS) == 99
    reference = copy_reference(catalog_reference_template, tmp_path)
    change_required_manifest(reference, api, mutation)
    original = file_checksum(reference)
    output = tmp_path / "unwritten/config.json"
    with pytest.raises(ValueError, match=f"^{error_code}$"):
        binding_builder.write_research_config(reference, output, AS_OF)
    assert not output.parent.exists()
    assert file_checksum(reference) == original


@pytest.mark.parametrize(
    "extra_api", ["derived.cn_equity_trade_calendar", "fund_daily", "unknown'); SELECT 1 --"]
)
def test_extra_duplicate_and_malformed_manifests_are_not_parsed(
    tmp_path,
    catalog_reference_template,
    extra_api,
):
    """非研究注册清单即使重复且JSON损坏也完全隔离，不触发解析或唯一性失败。

    输入额外两条同名坏JSON，构建仍与移除它们后的实体、单位、来源审计一致；
    reference指纹随文件内容变化属于预期，不拿不同文件的指纹作等值断言。
    """
    assert extra_api not in REQUIRED_APIS
    reference = copy_reference(catalog_reference_template, tmp_path)
    baseline = binding_builder.build_research_config(reference, AS_OF)
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.executemany(
            "INSERT INTO catalog VALUES (?, ?)",
            [(extra_api, "{not-json"), (extra_api, "also-invalid")],
        )
    result = binding_builder.build_research_config(reference, AS_OF)
    assert result.config.foundation == baseline.config.foundation
    assert result.audit["sources"] == baseline.audit["sources"]
    assert result.audit["summary"] == baseline.audit["summary"]
    assert extra_api not in result.audit["sources"]


def test_catalog_count_and_projection_queries_filter_exact_registered_parameters(
    tmp_path,
    catalog_reference_template,
    monkeypatch,
):
    """记录真实数据库查询，确认计数和JSON投影都先使用99个注册API参数过滤。

    输入含额外坏JSON的库，输出两条catalog查询，均有参数占位符且参数恰为99个排序API。
    数据库仍真实执行查询，证明额外JSON没有参与解析。
    """
    reference = copy_reference(catalog_reference_template, tmp_path)
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute("INSERT INTO catalog VALUES (?, ?)", ["extra", "{bad-json"])
    connect = duckdb.connect
    catalog_queries: list[tuple[str, Any]] = []

    class ConnectionRecorder:
        """保留真实DuckDB执行，仅记录catalog查询与参数；不替换数据或构建逻辑。"""

        def __init__(self, connection):
            """输入真实只读连接，输出可记录SQL的上下文包装器。"""
            self.connection = connection

        def __enter__(self):
            """进入构建上下文并返回记录器；例如with包装器得到可执行查询对象。"""
            return self

        def __exit__(self, *args):
            """无论成功或失败均关闭真实连接；输入异常上下文，输出不吞异常。"""
            self.connection.close()

        def execute(self, query, parameters=None):
            """透传SQL并收集catalog查询；例如COUNT查询保留原始99项参数列表。"""
            if "FROM catalog" in query:
                catalog_queries.append((query, parameters))
            return (
                self.connection.execute(query, parameters)
                if parameters is not None
                else (self.connection.execute(query))
            )

    def recorded_connect(*args, **kwargs):
        """打开真实连接再包装；传入read_only=True时仍保持数据库只读。"""
        return ConnectionRecorder(connect(*args, **kwargs))

    monkeypatch.setattr(binding_builder.duckdb, "connect", recorded_connect)
    result = binding_builder.build_research_config(reference, AS_OF)
    assert result.audit["summary"]["configured_apis"] == 99
    assert len(catalog_queries) == 2
    for query, parameters in catalog_queries:
        assert "WHERE api_name IN (SELECT unnest(?))" in query
        assert parameters == [list(REQUIRED_APIS)]
    assert "count(*)" in catalog_queries[0][0]
    assert "manifest::JSON" in catalog_queries[1][0]


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("missing", "reference_manifest_missing"),
        ("duplicate", "reference_catalog_duplicate"),
        ("hash", "source_contract_mismatch"),
    ],
)
def test_catalog_failure_cli_returns_fixed_detail_and_no_partial_output(
    tmp_path,
    catalog_reference_template,
    mutation,
    error_code,
):
    """目录存在性、重复和合同失败沿CLI返回精确固定码，不写配置或审计半成品。

    输入被修改的daily清单，输出exit=1、research_build_config_failed及对应detail_code。
    """
    reference = copy_reference(catalog_reference_template, tmp_path)
    change_required_manifest(reference, "daily", mutation)
    output = tmp_path / "unwritten/config.json"
    result = CliRunner().invoke(app, build_arguments(reference, output))
    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert summary == {
        "status": "failed",
        "reason_codes": ["research_build_config_failed"],
        "detail_code": error_code,
    }
    assert not output.parent.exists()


def test_duplicate_required_manifest_precedes_missing_required_manifest(
    tmp_path,
    catalog_reference_template,
):
    """目录同时缺必需项且另一个必需项重复时，优先报告重复而非缺失。

    输入缺daily且daily_basic有两条清单，输出reference_catalog_duplicate，不创建配置。
    """
    reference = copy_reference(catalog_reference_template, tmp_path)
    change_required_manifest(reference, "daily", "missing")
    change_required_manifest(reference, "daily_basic", "duplicate")
    output = tmp_path / "unwritten/config.json"
    with pytest.raises(ValueError, match="^reference_catalog_duplicate$"):
        binding_builder.write_research_config(reference, output, AS_OF)
    assert not output.parent.exists()
