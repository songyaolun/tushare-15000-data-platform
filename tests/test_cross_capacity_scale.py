"""真实混合负载跨越旧行数及字节门槛，核验完整指标、持久化、重算和资源记录。"""

import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import duckdb
import pytest
from test_cross_capacity_fixture import (
    AS_OF,
    CAPACITY_APIS,
    DATABASE_CONFIG,
    write_capacity_reference,
)

from tushare_entitlement_hub.cross_asset_research import execution, pipeline, storage
from tushare_entitlement_hub.cross_asset_research.bindings import build_config
from tushare_entitlement_hub.cross_asset_research.partitioned_inputs import InputSpool
from tushare_entitlement_hub.cross_asset_research.registry import PRODUCTS
from tushare_entitlement_hub.equity_research.models import ResearchResult
from tushare_entitlement_hub.equity_research.storage import _stored_records
from tushare_entitlement_hub.reference_storage import file_checksum
from tushare_entitlement_hub.store import encode


def run_capacity_case(tmp_path, monkeypatch, **dimensions):
    """以正式构建、发布和重算链执行给定规模，并保存即时阶段及真实资源证据。

    例如每源四行可用于检查测试基础设施；正式压力用例始终传入每源三万行。
    观察性包装只记录原函数结果，不替换绑定、适配、消费或持久化计算。
    """
    started = time.monotonic()
    evidence = Path(os.environ.get("CAPACITY_EVIDENCE_PATH", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    progress_path = Path(
        os.environ.get("CAPACITY_PROGRESS_PATH", str(evidence / "live-progress.jsonl"))
    )
    report = {"dimensions": dimensions, "phase": "fixture", "stages": []}

    def progress(event):
        """追加后立即关闭阶段日志，例如扫描尚未结束时也能看到真实行数及 RSS。"""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record = {
            "elapsed_seconds": time.monotonic() - started,
            "cpu_user_seconds": usage.ru_utime,
            "cpu_system_seconds": usage.ru_stime,
            "process_max_rss_kib": usage.ru_maxrss,
            **event,
        }
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(encode(record) + "\n")

    def checkpoint(phase):
        """落盘当前已完成阶段，不提前标记后续成功，例如发布完成尚不代表重算通过。"""
        report["phase"] = phase
        report["stages"].append(phase)
        report["elapsed_seconds"] = time.monotonic() - started
        report["process_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (evidence / "mixed-capacity.json").write_text(encode(report) + "\n", encoding="utf-8")
        progress({"stage": phase})

    progress({"stage": "fixture_start"})
    reference, source_counts = write_capacity_reference(tmp_path, progress=progress, **dimensions)
    report["fixture_counts"] = source_counts
    report["reference_sha256"] = file_checksum(reference)
    checkpoint("fixture_complete")
    checkpoint("config_start")
    built = build_config(reference, AS_OF)
    report["runtime_budgets"] = {
        "maximum_source_rows": built.config.maximum_source_rows,
        "maximum_total_rows": built.config.maximum_total_rows,
    }
    assert report["runtime_budgets"] == {
        "maximum_source_rows": 2_000_000,
        "maximum_total_rows": 5_000_000,
    }
    checkpoint("config_complete")
    retained_snapshots = []
    legacy_counts = []
    original_statistics = InputSpool.statistics
    original_legacy = execution.load_equity_inputs
    original_materialize = InputSpool.materialize_selection
    partition_peak = {"bytes": 0, "rows": 0}
    original_verify = storage.verify_saved
    original_replay = storage.replay
    original_replay_execute = storage.execute
    verification_pass = 0

    def record_verification(connection, metadata):
        """记录真实保存后校验的开始和完成，例如首次发布、幂等复用及 validate 分别可见。"""
        nonlocal verification_pass
        verification_pass += 1
        progress({"stage": "verify_saved_start", "verification_pass": verification_pass})
        result = original_verify(connection, metadata)
        progress({"stage": "verify_saved_complete", "verification_pass": verification_pass})
        return result

    def record_replay(reference, metadata):
        """记录完整来源重算的实际边界，例如输入重载之前即落盘 replay_start。"""
        progress({"stage": "replay_start"})
        result = original_replay(reference, metadata)
        progress({"stage": "replay_complete"})
        return result

    def replay_progress(event):
        """为重算的真实执行事件加前缀，例如 replay_product_complete 表示该产品已重算。"""
        progress({**event, "stage": "replay_" + event["stage"]})

    def record_replay_execution(*args, **kwargs):
        """向原重算执行器传入只记录进度的回调，返回同一真实生成器，不预先物化结果。"""
        return original_replay_execute(*args, **kwargs, progress=replay_progress)

    def record_partition(self, template, selectors, *, maximum_bytes=16 * 1024**2):
        """只读核算本次完整选择器的实际字节峰值，例如 ETF 三类来源共同计入一个分区。"""
        selected = tuple(selectors)
        byte_count = row_count = 0
        for api, entity, code in selected:
            clause, values = "api=?", [api]
            if entity is not None:
                clause += " AND entity=?"
                values.append(entity)
            if code is not None:
                clause += " AND code=?"
                values.append(code)
            count, size = self.connection.execute(
                "SELECT count(*),coalesce(sum(bytes),0) FROM observations WHERE " + clause,
                values,
            ).fetchone()
            row_count += count
            byte_count += size
        if byte_count > partition_peak["bytes"]:
            partition_peak.update(bytes=byte_count, rows=row_count)
        return original_materialize(self, template, selected, maximum_bytes=maximum_bytes)

    def record_statistics(self):
        """只读汇总真实保留载荷的原 raw 字节和适配字节，避免用清单计数冒充运行消费。"""
        first = self.connection.execute("SELECT payload FROM observations LIMIT 1").fetchone()
        assert first is not None
        payload = json.loads(first[0])
        sql_size = self.connection.execute(
            "SELECT length(CAST(json_extract(?,'$.raw') AS BLOB))",
            (first[0],),
        ).fetchone()[0]
        assert sql_size == len(encode(payload["raw"]).encode("utf-8"))
        snapshot = {
            api: {"rows": count, "raw_bytes": raw_bytes, "adapted_bytes": adapted_bytes}
            for api, count, raw_bytes, adapted_bytes in self.connection.execute(
                "SELECT api,count(*),sum(length(CAST(json_extract(payload,'$.raw') AS BLOB))),"
                "sum(bytes) FROM observations GROUP BY api"
            )
        }
        retained_snapshots.append(snapshot)
        report["retained_sources"] = snapshot
        checkpoint("input_spool_complete")
        return original_statistics(self)

    def record_legacy(*args, **kwargs):
        """保留旧 loader 的真实行和计数，例如筹码行必须实际进入返回的 ResearchInputs。"""
        inputs, counts = original_legacy(*args, **kwargs)
        counted = {api: len(rows) for api, rows in inputs.rows_by_api.items() if rows}
        assert counts["retained_rows"] == sum(counted.values())
        assert (
            counted["cyq_chips"] == dimensions["legacy_entities"] * dimensions["factors_per_entity"]
        )
        legacy_counts.append(counts)
        report["legacy_retained_by_api"] = counted
        progress({"stage": "legacy_input_complete", **counts})
        return inputs, counts

    monkeypatch.setattr(InputSpool, "statistics", record_statistics)
    monkeypatch.setattr(execution, "load_equity_inputs", record_legacy)
    monkeypatch.setattr(InputSpool, "materialize_selection", record_partition)
    monkeypatch.setattr(pipeline, "verify_saved", record_verification)
    monkeypatch.setattr(storage, "verify_saved", record_verification)
    monkeypatch.setattr(storage, "replay", record_replay)
    monkeypatch.setattr(storage, "execute", record_replay_execution)
    output = tmp_path / "capacity-products.duckdb"
    summary = pipeline.run(
        reference, output, AS_OF, built.config, ("P02", "P08", "P11"), progress=progress
    )
    report["summary"] = summary
    report["largest_complete_partition"] = partition_peak
    checkpoint("published")
    rows_per_source = dimensions["rows_per_source"]
    assert summary["processing"]["excluded_rows"] == 12
    assert summary["processing"]["discarded_rows"] == 0
    assert (
        summary["processing"]["retained_rows"] - summary["processing"]["spooled_rows"]
        == legacy_counts[0]["retained_rows"]
    )
    assert summary["input_storage"]["supporting_rows"] == dimensions["stock_count"]
    assert summary["input_storage"]["rows"] == (
        summary["input_storage"]["new_source_rows"] + summary["input_storage"]["supporting_rows"]
    )
    for api in CAPACITY_APIS:
        consumed = summary["consumption"]["by_api"][api]
        assert consumed["retained_rows"] == rows_per_source
        assert consumed["observed"] and consumed["observed_rows"] > 0
        assert consumed["handler"] == "partitioned_execution.partitioned_results"
        assert consumed["reason_counts"]["rows_excluded_by_adapter"] > 0
        assert consumed["reason_counts"]["row_unavailable"] > 0
        assert consumed["reason_counts"]["quarantined_or_unknown_quality"] > 0
        assert retained_snapshots[0][api]["rows"] == rows_per_source
        assert not any("capacity_exceeded" in reason for reason in consumed["reason_counts"])
    assert summary["consumption"]["by_api"]["opt_basic"]["observed_rows"] == rows_per_source
    assert summary["consumption"]["by_api"]["opt_daily"]["retained_rows"] == 2

    products = Counter()
    available_refs = {api: set() for api in CAPACITY_APIS}
    metrics = []
    option_results = 0
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        for record in _stored_records(connection, summary["run_id"], ("result",)):
            result = ResearchResult.from_json(record.payload)
            products[result.envelope.product_id] += 1
            assert len(record.payload.encode()) <= 8 * 1024**2
            for metric in result.metrics:
                if metric.status == "available":
                    for ref in metric.evidence_refs:
                        if ref.source_api in available_refs:
                            available_refs[ref.source_api].add(ref.source_record_id)
            if result.envelope.product_id == PRODUCTS["P02"]:
                api = (
                    "dc_member"
                    if result.metric("dc_member_observed_members") is not None
                    else "tdx_member"
                )
                assert result.metric(api + "_observed_members") == dimensions["theme_members"]
                assert result.metric(api + "_member_price_coverage") == 1
                assert result.metric(api + "_equal_weight_return") == pytest.approx(0.1)
            elif result.envelope.product_id == PRODUCTS["P08"]:
                assert result.metric("active_fund_managers") == dimensions["fund_members"]
                assert (
                    result.metric("same_manager_observed_etfs")
                    == rows_per_source // dimensions["fund_members"]
                )
                for api in ("etf_sh_cons", "etf_sz_cons"):
                    assert result.metric(api + "_member_count") == dimensions["fund_members"]
                    assert result.metric(api + "_positive_quantity_fraction") == 1
            else:
                option_results += result.details.get("source_api") == "opt_basic"
                continue
            metrics.append(
                {
                    "product": result.envelope.product_id,
                    "entity": result.envelope.entity_id,
                    "values": {
                        metric.name: metric.value
                        for metric in result.metrics
                        if metric.value is not None
                    },
                }
            )
    assert option_results == rows_per_source
    assert products[PRODUCTS["P02"]] == 2 * rows_per_source // dimensions["theme_members"]
    assert products[PRODUCTS["P08"]] == rows_per_source // dimensions["fund_members"]
    report["product_counts"] = dict(products)
    report["business_metrics"] = metrics
    report["persisted_available_reference_rows"] = {
        api: len(values) for api, values in available_refs.items()
    }
    for api, values in available_refs.items():
        assert len(values) == summary["consumption"]["by_api"][api]["observed_rows"]
    checkpoint("stored_business_assertions_complete")
    before = file_checksum(output)
    repeated = pipeline.run(
        reference, output, AS_OF, built.config, ("P02", "P08", "P11"), progress=progress
    )
    assert encode(repeated) == encode(summary)
    assert file_checksum(output) == before
    report["idempotent_summary_and_database"] = True
    checkpoint("idempotent_reuse_complete")
    validation = storage.validate(reference, output)
    report["validation"] = validation
    assert validation["valid"] is True
    assert len(retained_snapshots) == len(legacy_counts) == 2
    assert retained_snapshots[0] == retained_snapshots[1]
    assert legacy_counts[0] == legacy_counts[1]
    assert file_checksum(reference) == report["reference_sha256"]
    assert file_checksum(output) == before
    report["output_sha256"] = before
    report["resource_measurement"] = (
        "Linux RUSAGE_SELF.ru_maxrss in KiB; wall and CPU times recorded separately"
    )
    checkpoint("complete")
    return report


@pytest.mark.resource_isolated
def test_large_mixed_input_preserves_complete_business_partitions(tmp_path, monkeypatch):
    """场景：旧来源实际保留 145000 行，六个高基数新来源各 30000 行；预期：完整发布及重算。

    每源真实 raw 投影超过旧 16MiB、累计超过 64MiB。主题每组 5000 成员、
    基金每组 1000 成分及经理均完整计数，同管理人 ETF 数为 30，三万期权目录均实际观察。
    """
    report = run_capacity_case(
        tmp_path,
        monkeypatch,
        rows_per_source=30_000,
        fund_members=1_000,
        theme_members=5_000,
        legacy_entities=300,
        factors_per_entity=450,
        stock_count=5_000,
    )
    assert sum(report["legacy_retained_by_api"].values()) == 145_000 > 131_072
    assert report["legacy_retained_by_api"]["cyq_chips"] == 135_000
    assert all(report["retained_sources"][api]["raw_bytes"] > 16 * 1024**2 for api in CAPACITY_APIS)
    assert sum(report["retained_sources"][api]["raw_bytes"] for api in CAPACITY_APIS) > 64 * 1024**2
    assert report["summary"]["input_storage"]["sqlite_cache_bytes"] == 8 * 1024**2
    assert report["summary"]["input_storage"]["maximum_partition_bytes"] == 16 * 1024**2
