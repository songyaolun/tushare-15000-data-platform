"""全产品配置、执行、重放、原子发布和容量预算的合成集成验证。"""

import json
from hashlib import sha256

import duckdb
import pytest
from all_products_fixture import AS_OF, DATABASE_CONFIG, write_reference
from typer.testing import CliRunner

from tushare_entitlement_hub.cli import app
from tushare_entitlement_hub.cross_asset_research import pipeline, storage
from tushare_entitlement_hub.cross_asset_research.bindings import build_config
from tushare_entitlement_hub.cross_asset_research.execution import RunLimits
from tushare_entitlement_hub.cross_asset_research.registry import DEFAULT_PRODUCTS, PRODUCTS
from tushare_entitlement_hub.equity_research.models import ResearchConfig, ResearchResult
from tushare_entitlement_hub.equity_research.storage import _stored_records
from tushare_entitlement_hub.reference_storage import file_checksum
from tushare_entitlement_hub.store import encode


def saved_results(output):
    """从当前批次逐行读取结果；例：返回模型元组和保存的运行元数据。"""
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        payload = connection.execute(
            "SELECT run.payload FROM research.run run JOIN research.metadata m "
            "ON run.run_id=(m.payload->>'$') WHERE m.key='current_run'"
        ).fetchone()[0]
        metadata = json.loads(payload)
        results = tuple(
            ResearchResult.from_json(row.payload)
            for row in _stored_records(connection, metadata["run_id"], ("result",))
        )
    return results, metadata


def assert_persisted_portfolio_parent_digest(results, portfolio):
    """从落库结果按父ID独立重算完整多重集摘要，消费审计字段也在覆盖范围内。

    输入同时含旧消费父与P05新父的结果列表和对应P12；摘要相等时返回
    matches_persisted_parents=True，例如两条父结果均保留时返回两条父ID；不等则断言失败。
    """
    by_identity = {
        (item.envelope.run_id, item.envelope.result_id): item for item in results
    }
    parents = [
        by_identity[(portfolio.envelope.run_id, result_id)]
        for result_id in portfolio.parent_result_ids
    ]
    assert parents
    assert len(parents) == len(set(portfolio.parent_result_ids))
    assert any("consumption" in parent.details for parent in parents)
    assert any(parent.envelope.product_id == PRODUCTS["P05"] for parent in parents)
    recomputed = sum(
        int(sha256(encode(parent.to_dict()).encode("utf-8")).hexdigest(), 16)
        for parent in parents
    ) % 2**256
    declared = portfolio.details["parent_multiset_sha256"]
    assert declared == f"{recomputed:064x}"
    return {
        "weight_basis": portfolio.details["weight_basis"],
        "parent_ids": list(portfolio.parent_result_ids),
        "parent_products": sorted({parent.envelope.product_id for parent in parents}),
        "parents_with_consumption": sum("consumption" in parent.details for parent in parents),
        "declared_parent_multiset_sha256": declared,
        "persisted_parent_multiset_sha256": f"{recomputed:064x}",
        "matches_persisted_parents": True,
    }


def test_build_run_validate_cli_has_nonempty_metrics_and_exact_replay(tmp_path):
    """场景：正式build-config→run→validate链；预期：七产品实执行、指标可用、摘要重跑一致。"""
    reference = write_reference(tmp_path)
    before = file_checksum(reference)
    output, config_path = tmp_path / "products.duckdb", tmp_path / "config.json"
    runner = CliRunner()
    built = runner.invoke(
        app,
        [
            "research-all",
            "build-config",
            "--reference",
            str(reference),
            "--output",
            str(config_path),
            "--as-of",
            AS_OF,
        ],
    )
    assert built.exit_code == 0, built.output
    assert json.loads(built.stdout)["summary"]["configured_apis"] == 206
    args = [
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
    ]
    first = runner.invoke(app, args)
    assert first.exit_code == 2, first.output
    summary = json.loads(first.stdout)
    assert summary["consumption"]["registered"] == summary["consumption"]["executed"] == 206
    assert set(summary["product_states"]) == set(DEFAULT_PRODUCTS)
    results, metadata = saved_results(output)
    metrics = {
        product: {
            metric.name: metric.value
            for result in results
            if result.envelope.product_id == product
            for metric in result.metrics
            if metric.value is not None
        }
        for product in (PRODUCTS[name] for name in DEFAULT_PRODUCTS)
    }
    for name in ("P02", "P05", "P08", "P09", "P10", "P11"):
        assert any(value is not None for value in metrics[PRODUCTS[name]].values()), (
            name,
            metrics[PRODUCTS[name]],
        )
    assert metrics[PRODUCTS["P02"]]["price_flow_confirmation"] == 1
    assert metrics[PRODUCTS["P05"]]["forecast_profit_midpoint"] == 150000
    assert metrics[PRODUCTS["P08"]]["share_change"] == 10000
    assert metrics[PRODUCTS["P08"]]["share_change_ratio"] == 1
    assert metrics[PRODUCTS["P09"]]["open_interest_volume_ratio"] == 2
    assert metrics[PRODUCTS["P09"]]["session_open"] == 1
    assert metrics[PRODUCTS["P09"]]["commodity_index_return"] == pytest.approx(0.02)
    assert metrics[PRODUCTS["P10"]]["nominal_ten_two_year_spread"] == pytest.approx(0.01)
    assert metrics[PRODUCTS["P11"]]["net_financing_balance_ratio"] == pytest.approx(0.1)
    portfolio = [result for result in results if result.envelope.product_id == PRODUCTS["P12"]]
    assert portfolio and portfolio[0].details["weight_basis"] == "disclosed_stock_sleeve"
    assert portfolio[0].metric("disclosed_weight") == pytest.approx(0.5)
    assert portfolio[0].metric("holding_hhi") == pytest.approx(0.25)
    (tmp_path / "parent-digest-verification.json").write_text(
        encode(assert_persisted_portfolio_parent_digest(results, portfolio[0]))
    )
    assert all(not result.envelope.decision_allowed for result in results)
    first_database_checksum = file_checksum(output)
    second = runner.invoke(app, args)
    assert file_checksum(output) == first_database_checksum
    assert second.exit_code == first.exit_code and second.stdout == first.stdout
    results_again, metadata_again = saved_results(output)
    assert tuple(result.to_json() for result in results_again) == tuple(
        result.to_json() for result in results
    )
    assert encode(metadata_again) == encode(metadata)
    validated = runner.invoke(
        app, ["research-all", "validate", "--reference", str(reference), "--output", str(output)]
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout)["valid"] is True
    assert file_checksum(reference) == before
    (tmp_path / "build-audit.json").write_text(built.stdout)
    (tmp_path / "run-summary.json").write_text(first.stdout)
    (tmp_path / "validate-summary.json").write_text(validated.stdout)
    (tmp_path / "derived-metrics.json").write_text(
        encode(
            {
                "P02": {
                    "price_flow_confirmation": metrics[PRODUCTS["P02"]]["price_flow_confirmation"]
                },
                "P05": {
                    "forecast_profit_midpoint": metrics[PRODUCTS["P05"]]["forecast_profit_midpoint"]
                },
                "P08": {
                    "share_change": metrics[PRODUCTS["P08"]]["share_change"],
                    "share_change_ratio": metrics[PRODUCTS["P08"]]["share_change_ratio"],
                },
                "P09": {
                    "open_interest_volume_ratio": metrics[PRODUCTS["P09"]][
                        "open_interest_volume_ratio"
                    ]
                },
                "P10": {
                    "nominal_ten_two_year_spread": metrics[PRODUCTS["P10"]][
                        "nominal_ten_two_year_spread"
                    ]
                },
                "P11": {
                    "net_financing_balance_ratio": metrics[PRODUCTS["P11"]][
                        "net_financing_balance_ratio"
                    ]
                },
                "P12": {
                    "disclosed_weight": portfolio[0].metric("disclosed_weight"),
                    "holding_hhi": portfolio[0].metric("holding_hhi"),
                },
            }
        )
    )
    (tmp_path / "replay-checksums.json").write_text(
        encode(
            {
                "reference_sha256": before,
                "output_sha256": first_database_checksum,
                "summary_bytes_identical": first.stdout == second.stdout,
                "database_bytes_identical": first_database_checksum == file_checksum(output),
            }
        )
    )


def test_explicit_candidate_horizons_are_independent_and_replayable(tmp_path):
    """场景：显式运行三套 P03 排名；预期：产品身份独立、依赖闭包准确且可完整重放。"""
    reference = write_reference(tmp_path)
    output, config_path = tmp_path / "rankings.duckdb", tmp_path / "config.json"
    runner = CliRunner()
    built = runner.invoke(
        app,
        [
            "research-all",
            "build-config",
            "--reference",
            str(reference),
            "--output",
            str(config_path),
            "--as-of",
            AS_OF,
        ],
    )
    assert built.exit_code == 0, built.output
    run = runner.invoke(
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
            "P03S",
            "--product",
            "P03M",
            "--product",
            "P03L",
        ],
    )
    assert run.exit_code == 2, run.output
    summary = json.loads(run.stdout)
    assert set(summary["product_states"]) == {
        "P04",
        "P06",
        "P07",
        "P03S",
        "P03M",
        "P03L",
    }
    results, _ = saved_results(output)
    ranking_products = {PRODUCTS[name] for name in ("P03S", "P03M", "P03L")}
    ranking_results = [
        result for result in results if result.envelope.product_id in ranking_products
    ]
    assert {result.envelope.product_id for result in ranking_results} == ranking_products
    assert {result.details["ranking_horizon"] for result in ranking_results} == {
        "short_term",
        "medium_term",
        "long_term",
    }
    for alias in ("P03S", "P03M", "P03L"):
        selected = runner.invoke(
            app,
            [
                "research-all",
                "results",
                "--output",
                str(output),
                "--product",
                alias,
                "--limit",
                "5",
            ],
        )
        assert selected.exit_code == 0, selected.output
        payloads = [json.loads(line) for line in selected.stdout.splitlines()]
        assert payloads
        assert {payload["envelope"]["product_id"] for payload in payloads} == {
            PRODUCTS[alias]
        }
    validated = runner.invoke(
        app,
        ["research-all", "validate", "--reference", str(reference), "--output", str(output)],
    )
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.stdout)["valid"] is True


def test_missing_bindings_produces_explicit_unknown_for_all_products(tmp_path):
    """场景：206清单存在但配置为空；预期：13处理器仍执行且所有观察数为0，不伪造值。"""
    reference = write_reference(tmp_path)
    output = tmp_path / "empty.duckdb"
    summary = pipeline.run(reference, output, AS_OF, ResearchConfig())
    assert summary["status"] == "incomplete"
    assert summary["consumption"]["executed"] == 206
    assert summary["consumption"]["observed"] == 0
    assert summary["consumption"]["unresolved"] == 206
    assert storage.validate(reference, output)["valid"]
    assert (
        next(
            result
            for result in saved_results(output)[0]
            if result.envelope.product_id == PRODUCTS["P12"]
        ).details["status"]
        == "input_required"
    )


@pytest.mark.parametrize("variant", ["result", "output"])
def test_capacity_failure_preserves_previously_published_bytes(tmp_path, variant):
    """场景：第二次运行突破结果数或输出字节预算；预期：旧库逐字不变且私有副本清理。"""
    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"
    pipeline.run(reference, output, AS_OF, config, ("P05",))
    before = output.read_bytes()
    limits = (
        RunLimits(maximum_results=1) if variant == "result" else RunLimits(maximum_output_bytes=1)
    )
    with pytest.raises(ValueError, match="all_products_(result|output)_capacity_exceeded"):
        pipeline.run(reference, output, AS_OF, config, ("P02", "P05"), limits=limits)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".research-*"))
    assert storage.validate(reference, output)["valid"]


@pytest.mark.parametrize("variant", ["missing", "same", "hardlink", "invalid_time"])
def test_pipeline_rejects_invalid_paths_and_cutoff(tmp_path, variant):
    """场景：参考不存在、输入输出同文件或日期非法；预期：立即失败并保留参考字节。"""
    reference = write_reference(tmp_path)
    output = tmp_path / "products.duckdb"
    if variant == "same":
        output = reference
    elif variant == "hardlink":
        output.hardlink_to(reference)
    before = file_checksum(reference)
    with pytest.raises(ValueError):
        pipeline.run(
            reference.with_name("missing") if variant == "missing" else reference,
            output,
            "invalid" if variant == "invalid_time" else AS_OF,
            ResearchConfig(),
        )
    assert file_checksum(reference) == before


def test_failed_publish_retains_old_result_and_cleans_private_snapshot(tmp_path, monkeypatch):
    """场景：真实保存后在发布前校验失败；预期：旧版本保留且新版本不可见。"""
    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"
    original = pipeline.run(reference, output, AS_OF, config, ("P05",))
    before = output.read_bytes()

    def rejected(*args, **kwargs):
        """模拟发布前完整性门禁拒绝；例：任何元数据都返回固定测试错误。"""
        raise ValueError("synthetic_verification_failure")

    monkeypatch.setattr(pipeline, "verify_saved", rejected)
    with pytest.raises(ValueError, match="research_stage_validation_failed") as captured:
        pipeline.run(reference, output, AS_OF, config, ("P02",))
    assert str(captured.value.__cause__) == "synthetic_verification_failure"
    assert output.read_bytes() == before
    assert saved_results(output)[1]["summary"] == original


def test_reference_mutation_during_run_prevents_publication(tmp_path, monkeypatch):
    """场景：扫描之后参考指纹改变；预期：拒绝发布且没有最终输出文件。"""
    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    actual = file_checksum(reference)
    calls = iter((actual, "changed"))
    monkeypatch.setattr(pipeline, "file_checksum", lambda path: next(calls))
    output = tmp_path / "products.duckdb"
    with pytest.raises(ValueError, match="reference_changed_during_read"):
        pipeline.run(reference, output, AS_OF, config, ("P05",))
    assert not output.exists()


def test_future_and_quarantined_rows_are_excluded_with_precise_reasons(tmp_path):
    """场景：同主题混入未来行情和隔离行情；预期：只计算3%可见换手率，并保留两类排除原因。"""
    reference = write_reference(
        tmp_path,
        {
            "ths_daily": [
                {
                    "ts_code": "885001.TI",
                    "trade_date": "20261002",
                    "pct_change": 2,
                    "vol": 100,
                    "turnover_rate": 3,
                },
                {
                    "ts_code": "885001.TI",
                    "trade_date": "20261202",
                    "pct_change": 99,
                    "vol": 100,
                    "turnover_rate": 3,
                },
                {
                    "ts_code": "885001.TI",
                    "trade_date": "20260902",
                    "pct_change": 88,
                    "vol": 100,
                    "turnover_rate": 3,
                },
            ]
        },
    )
    with duckdb.connect(str(reference), config=DATABASE_CONFIG) as connection:
        connection.execute(
            "UPDATE accepted.ths_daily SET quality_tier='quarantine' WHERE trade_date='2026-09-02'"
        )
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"
    summary = pipeline.run(reference, output, AS_OF, config, ("P02",))
    source = summary["consumption"]["by_api"]["ths_daily"]
    assert source["retained_rows"] == source["observed_rows"] == 1
    reasons = source["reason_counts"]
    assert "future_business_date" in reasons and "quarantined_or_unknown_quality" in reasons
    results, _ = saved_results(output)
    theme = next(
        result
        for result in results
        if any(ref.source_api == "ths_daily" for ref in result.envelope.evidence_refs)
    )
    assert theme.metric("ths_daily_turnover_rate") == pytest.approx(0.03)
    assert storage.validate(reference, output)["valid"]


def test_concurrent_writer_is_rejected_while_private_run_owns_output(tmp_path):
    """场景：第一写者仍在构造时第二写者进入；预期：第二写者busy，第一写者可完整发布。"""
    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"
    attempted = []

    def competing_writer(event):
        """在首个产品完成事件模拟竞争写入；例：同目标立即报research_output_busy。"""
        if event.get("stage") == "product_complete" and not attempted:
            attempted.append(True)
            assert not output.exists()
            with pytest.raises(ValueError, match="research_output_busy"):
                pipeline.run(reference, output, AS_OF, config, ("P02",))

    pipeline.run(reference, output, AS_OF, config, ("P05",), progress=competing_writer)
    assert attempted == [True]
    assert storage.validate(reference, output)["valid"]


def test_single_result_over_eight_mib_never_publishes(tmp_path, monkeypatch):
    """场景：处理器返回超过8MiB的合法模型；预期：保存前拒绝且不存在半份最终数据库。"""
    from tushare_entitlement_hub.cross_asset_research import execution
    from tushare_entitlement_hub.equity_research.common import make_result

    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"

    def oversized(current):
        """制造超8MiB的合成解释文本；例：保存必须因UTF8字节门禁拒绝整条结果。"""
        return (
            make_result(
                current,
                PRODUCTS["P05"],
                "cn:security:synthetic:listed",
                details={"capacity_fixture": "x" * (8 * 1024 * 1024)},
            ),
        )

    monkeypatch.setattr(execution, "resolve_handler", lambda name: oversized)
    with pytest.raises(ValueError, match="research_result_payload_capacity_exceeded"):
        pipeline.run(reference, output, AS_OF, config, ("P05",))
    assert not output.exists()


def test_multi_partition_results_preserve_payload_and_replay(tmp_path, monkeypatch):
    """场景：20条约1MiB结果跨16MiB分区；预期：所有结果可重放、每段有界且摘要幂等。"""
    from tushare_entitlement_hub.cross_asset_research import execution
    from tushare_entitlement_hub.equity_research.common import make_result
    from tushare_entitlement_hub.equity_research.partitions import (
        MAXIMUM_PARTITION_BYTES,
        partition_catalog,
    )

    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"

    def wide(current):
        """生成20条不同实体的确定性宽结果；例：全部写入且不通过截断满足分区上限。"""
        return tuple(
            make_result(
                current,
                PRODUCTS["P05"],
                f"cn:security:S{index}:listed",
                details={"capacity_fixture": str(index % 10) * (1024 * 1024)},
            )
            for index in range(20)
        )

    monkeypatch.setattr(execution, "resolve_handler", lambda name: wide)
    first = pipeline.run(reference, output, AS_OF, config, ("P05",))
    assert first["result_count"] == 20
    assert first["physical_json_bytes"] > 20 * 1024 * 1024
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        parts = partition_catalog(connection)["result"]
        assert len(parts) > 1
        for name in parts:
            size = connection.execute(
                f'SELECT sum(octet_length(encode(payload::VARCHAR))) FROM research."{name}"'
            ).fetchone()[0]
            assert size <= MAXIMUM_PARTITION_BYTES
    assert pipeline.run(reference, output, AS_OF, config, ("P05",)) == first
    assert storage.validate(reference, output)["valid"]


def test_switching_to_retained_run_updates_current_without_duplicate_results(tmp_path):
    """场景：依次执行P05、P02、P05；预期：切回旧批次且不重复结果，两个批次均可验证。"""
    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"
    original = pipeline.run(reference, output, AS_OF, config, ("P05",))
    other = pipeline.run(reference, output, AS_OF, config, ("P02",))
    assert other["run_id"] != original["run_id"]
    assert saved_results(output)[1]["run_id"] == other["run_id"]
    switched = pipeline.run(reference, output, AS_OF, config, ("P05",))
    assert switched == original
    assert saved_results(output)[1]["run_id"] == original["run_id"]
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        assert connection.execute("SELECT count(*) FROM research.run").fetchone() == (2,)
        assert connection.execute(
            "SELECT count(*) FROM research.result WHERE run_id=?", [original["run_id"]]
        ).fetchone() == (original["result_count"],)
    assert storage.validate(reference, output)["valid"]


@pytest.mark.parametrize("scenario", ["version", "guard"])
def test_idempotent_reuse_rechecks_output_version_and_read_guard(tmp_path, monkeypatch, scenario):
    """场景：复用当前批次时原库版本变化或读锁守护退出；预期：拒绝复用并保持旧文件。"""
    from contextlib import contextmanager
    from types import SimpleNamespace

    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"
    pipeline.run(reference, output, AS_OF, config, ("P05",))
    original = output.read_bytes()
    if scenario == "version":
        monkeypatch.setattr(pipeline, "_version", lambda path: None)
    else:
        original_snapshot = pipeline.research_snapshot

        @contextmanager
        def lost_guard(*args, **kwargs):
            """保留真实私有快照但模拟守护退出；例：poll返回0代表锁守护不再存活。"""
            with original_snapshot(*args, **kwargs) as work:
                work.guard = SimpleNamespace(poll=lambda: 0)
                yield work

        monkeypatch.setattr(pipeline, "research_snapshot", lost_guard)
    with pytest.raises(ValueError, match="research_output_changed"):
        pipeline.run(reference, output, AS_OF, config, ("P05",))
    assert output.read_bytes() == original


def test_existing_legacy_run_coexists_with_all_product_run(tmp_path):
    """场景：已有旧研究P01批次后写入P05；预期：旧批次保留，新旧验证均通过。"""
    from tushare_entitlement_hub.cross_asset_research.bindings import equity_config
    from tushare_entitlement_hub.equity_research.pipeline import run_research

    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    output = tmp_path / "products.duckdb"
    original = run_research(reference, output, AS_OF, equity_config(config), ("P01",))
    current = pipeline.run(reference, output, AS_OF, config, ("P05",))
    assert current["run_id"] != original["run_id"]
    verified = storage.validate(reference, output)
    assert verified["valid"] and len(verified["runs"]) == 2


def test_explicit_portfolio_cli_preserves_weights_and_replays_all_dependencies(tmp_path):
    """场景：显式60%/40%持仓走完整命令链；预期：HHI=0.52、权重不重算且父结果可重放。"""
    reference = write_reference(tmp_path)
    config = build_config(reference, AS_OF).config
    stocks = sorted(
        entity.identity.entity_id
        for entity in config.foundation.entities
        if entity.identity.namespace == "tushare_stock" and entity.identity.kind.value == "security"
    )
    assert len(stocks) == 2
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json())
    holdings_path = tmp_path / "holdings.json"
    holdings_path.write_text(
        encode(
            [
                {"entity_id": entity, "entity_type": "security", "as_of": AS_OF, "weight": weight}
                for entity, weight in zip(stocks, (0.6, 0.4), strict=True)
            ]
        )
    )
    output = tmp_path / "products.duckdb"
    result = CliRunner().invoke(
        app,
        [
            "research-all",
            "run",
            "--reference",
            str(reference),
            "--output",
            str(output),
            "--as-of",
            AS_OF,
            "--config",
            str(config_path),
            "--product",
            "P12",
            "--holdings",
            str(holdings_path),
        ],
    )
    assert result.exit_code == 2, result.output
    results, _ = saved_results(output)
    portfolio = next(item for item in results if item.envelope.product_id == PRODUCTS["P12"])
    assert portfolio.metric("holding_hhi") == pytest.approx(0.52)
    assert portfolio.metric("largest_holding_weight") == pytest.approx(0.6)
    assert portfolio.metric("disclosed_weight") == 1
    assert portfolio.details["weight_basis"] == "explicit_portfolio"
    (tmp_path / "parent-digest-verification.json").write_text(
        encode(assert_persisted_portfolio_parent_digest(results, portfolio))
    )
    assert portfolio.parent_result_ids
    assert storage.validate(reference, output)["valid"]


def test_holdings_count_budget_rejects_before_execution(tmp_path):
    """场景：显式持仓条数超过父结果预算；预期：预检立即拒绝，不创建输出数据库。"""
    reference = write_reference(tmp_path)
    output = tmp_path / "products.duckdb"
    with pytest.raises(ValueError, match="portfolio_input_capacity_exceeded"):
        pipeline.run(
            reference,
            output,
            AS_OF,
            ResearchConfig(),
            ("P05",),
            holdings=({}, {}),
            limits=RunLimits(maximum_parent_results=1),
        )
    assert not output.exists()
