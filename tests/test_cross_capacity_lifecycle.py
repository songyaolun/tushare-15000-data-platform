"""输入容量失败、提前结束和回调异常均清理私有数据，并保留原子发布边界。"""

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from test_cross_asset_bindings import write_all_reference
from test_cross_capacity_fixture import AS_OF, write_capacity_reference
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.cross_asset_research import execution, partitioned_inputs, pipeline
from tushare_entitlement_hub.cross_asset_research.bindings import build_config
from tushare_entitlement_hub.cross_asset_research.execution import ExecutionState, RunLimits
from tushare_entitlement_hub.cross_asset_research.partitioned_inputs import InputSpool
from tushare_entitlement_hub.equity_research import key_index
from tushare_entitlement_hub.equity_research.persistence_runtime import PersistenceError
from tushare_entitlement_hub.reference_storage import file_checksum


def track_private_directories(monkeypatch, directory: Path) -> list[Path]:
    """只改变测试临时目录的父路径并登记创建事实，保留真实 SQLite 生命周期。

    例如执行器创建输入暂存后，列表中的目录应先存在，关闭后应全部删除。
    """
    created: list[Path] = []

    def create(*args, **kwargs):
        """把真实临时目录放到测试沙箱，例如 SQLite 索引仍实际创建及关闭。"""
        kwargs["dir"] = directory
        temporary = TemporaryDirectory(*args, **kwargs)
        created.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(key_index, "TemporaryDirectory", create)
    return created


def small_capacity_reference(directory: Path) -> Path:
    """生成含六个新来源的小型正式参考库，例如每源四条可见行及两条排除行。"""
    reference, _ = write_capacity_reference(
        directory,
        rows_per_source=4,
        fund_members=2,
        theme_members=2,
        legacy_entities=0,
        factors_per_entity=0,
        stock_count=4,
    )
    return reference


@pytest.mark.parametrize("finish", ["exhaust", "early_close", "callback_error"])
def test_real_execution_cleans_spool_on_every_generator_exit(tmp_path, monkeypatch, finish):
    """场景：真实执行完成、首结果后关闭或扫描回调抛错；预期：全部私有索引立即清理。

    例如提前 close 前目录确实存在，之后不存在，原参考库的字节指纹不变。
    """
    reference = small_capacity_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    before = file_checksum(reference)
    created = track_private_directories(monkeypatch, tmp_path)

    def progress(event):
        """在真实扫描回调注入失败，例如首个跨资产扫描事件触发异常。"""
        if finish == "callback_error" and event["stage"] == "scan_cross_asset_reference":
            raise RuntimeError("synthetic_callback_failure")

    state = ExecutionState()
    results = execution.execute(
        reference,
        AS_OF,
        "capacity-lifecycle",
        config,
        ("P02", "P08", "P11"),
        (),
        RunLimits(),
        state,
        progress,
    )
    if finish == "callback_error":
        with pytest.raises(RuntimeError, match="^synthetic_callback_failure$"):
            next(results)
    elif finish == "early_close":
        first = next(results)
        assert first.envelope.result_id
        assert created and all(path.exists() for path in created)
        results.close()
    else:
        assert sum(1 for _ in results) > 0
        assert state.consumption["executed"] > 0
        assert state.input_storage["new_source_rows"] == state.processing["spooled_rows"]
        assert state.input_storage["supporting_rows"] == 4
        assert state.input_storage["rows"] == (
            state.input_storage["new_source_rows"] + state.input_storage["supporting_rows"]
        )
    assert created and all(not path.exists() for path in created)
    assert file_checksum(reference) == before


@pytest.mark.parametrize("boundary", ["spool", "source", "partition"])
def test_capacity_errors_keep_original_code_category_and_published_database(
    tmp_path,
    monkeypatch,
    boundary,
):
    """场景：显式小预算触发三种真实容量检查；预期：pipeline 和 CLI 保留原码及 capacity。

    这些是受控预算边界，不宣称写入过 32GiB；来源采用正式支持的无审计手工配置。
    每次失败均要求旧库与参考库字节不变、输入及发布暂存清理、不回显原始载荷。
    """
    reference = small_capacity_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "published.duckdb"
    pipeline.run(reference, output, AS_OF, config, ("P05",))
    before = {path: file_checksum(path) for path in (reference, output)}
    created = track_private_directories(monkeypatch, tmp_path)
    if boundary == "spool":
        monkeypatch.setattr(partitioned_inputs, "MAX_SPOOL_BYTES", 1)
    elif boundary == "source":
        config = replace(config, maximum_source_rows=1, binding_audit_json="{}")
        assert config.binding_audit == {}
    else:
        original = InputSpool.materialize_selection

        def materialize_one_byte(self, template, selectors, *, maximum_bytes=16 * 1024**2):
            """向真实分区函数传一字节预算，例如首条完整行超限而不截断输入。"""
            assert maximum_bytes == 16 * 1024**2
            return original(self, template, selectors, maximum_bytes=1)

        monkeypatch.setattr(InputSpool, "materialize_selection", materialize_one_byte)
    expected = f"research_input_{boundary}_capacity_exceeded"
    events = []
    with pytest.raises(PersistenceError, match=f"^{expected}$") as captured:
        pipeline.run(reference, output, AS_OF, config, ("P02",), progress=events.append)
    assert captured.value.category == "capacity"
    failures = [event for event in events if event["stage"] == "persistence-failed"]
    assert failures and failures[-1]["reason_code"] == expected
    assert failures[-1]["category"] == "capacity"
    assert {path: file_checksum(path) for path in before} == before
    assert created and all(not path.exists() for path in created)
    assert not list(tmp_path.glob(".published.duckdb.stage-*"))

    config_path = tmp_path / "capacity-config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "research-all",
            "run",
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--config",
            str(config_path),
            "--as-of",
            AS_OF,
            "--product",
            "P02",
        ],
    )
    assert result.exit_code == 1
    failure = json.loads(result.stdout)
    assert failure["status"] == "failed"
    assert failure["reason_codes"] == [expected]
    assert failure["category"] == "capacity"
    assert "合成股票" not in result.output and "合成成员" not in result.output
    assert {path: file_checksum(path) for path in before} == before
    assert all(not path.exists() for path in created)
    assert not list(tmp_path.glob(".published.duckdb.stage-*"))


def test_complete_basket_over_sixteen_mib_rejects_publication(tmp_path, monkeypatch):
    """场景：同 ETF 的 2500 条正式标准化申赎成分共同超过 16MiB；预期：整组拒绝发布。

    数量、溢价及折价均经正式单位规则判定，不补人工证书。测试记录真实字节，不扩预算。
    """
    reference = write_all_reference(
        tmp_path,
        {
            "etf_sh_cons": [
                {
                    "ts_code": "510001.SH",
                    "con_code": f"{index + 1:06d}.SZ",
                    "trade_date": "20261002",
                    "qty": 10,
                    "sub_flag": "允许",
                    "cpr": 1,
                    "rdr": 1,
                    "sca": 1,
                    "exchange": "SZ",
                }
                for index in range(2500)
            ]
        },
    )
    config = build_config(reference, AS_OF).config
    output = tmp_path / "published.duckdb"
    pipeline.run(reference, output, AS_OF, config, ("P05",))
    before = {path: file_checksum(path) for path in (reference, output)}
    created = track_private_directories(monkeypatch, tmp_path)
    actual = []
    original = InputSpool.statistics

    def record_statistics(self):
        """读取暂存实际字节，例如申赎完整来源超过 16MiB，保持原方法返回值。"""
        actual.append(
            self.connection.execute(
                "SELECT count(*),sum(bytes) FROM observations WHERE api='etf_sh_cons'"
            ).fetchone()
        )
        return original(self)

    monkeypatch.setattr(InputSpool, "statistics", record_statistics)
    with pytest.raises(PersistenceError, match="^research_input_partition_capacity_exceeded$"):
        pipeline.run(reference, output, AS_OF, config, ("P08",))
    assert actual and actual[0][0] == 2500 and actual[0][1] > 16 * 1024**2
    assert {path: file_checksum(path) for path in before} == before
    assert created and all(not path.exists() for path in created)
    assert not list(tmp_path.glob(".published.duckdb.stage-*"))
