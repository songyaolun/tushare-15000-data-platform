"""产品快照替换与持久化完整性缺陷的离线回归测试。"""

import json
from dataclasses import replace

import duckdb
import pytest
from test_product_foundation import NOW
from test_product_storage import bindings, reference_fixture

from tushare_entitlement_hub.product_foundation import reference as reference_module
from tushare_entitlement_hub.product_foundation.config import FoundationPolicy
from tushare_entitlement_hub.product_foundation.reference import initialize_library
from tushare_entitlement_hub.product_foundation.storage import ProductStore, validate_library


def replace_reference_snapshot(reference, *, empty=False):
    """更新合成参考快照的观察时间或清空行，并同步采集账本。

    输入参考库和 empty=True，输出 None；daily 表与账本的接受行数均变为 0。
    """
    observed_at = "2026-10-01T00:00:00Z"
    with duckdb.connect(str(reference), config={"threads": 1, "memory_limit": "512MB"}) as source:
        manifest = json.loads(source.execute("SELECT manifest FROM catalog").fetchone()[0])
        if empty:
            source.execute("DELETE FROM accepted.daily")
            for field in (
                "source_rows",
                "accepted_rows",
                "quarantined_rows",
                "exact_duplicate_rows",
            ):
                manifest["quality"][field] = 0
        else:
            source.execute("UPDATE accepted.daily SET observed_at=?", [observed_at])
        manifest["observed_at"] = observed_at
        source.execute("UPDATE catalog SET manifest=?", [json.dumps(manifest)])


def test_replaced_reference_refreshes_conflicting_facts_and_current_counts(tmp_path):
    """场景：同一来源行在新快照更改观察时间；预期：新事实生效且旧覆盖不会累积。

    输入两次初始化同一路径，输出不同 run_id、206 项覆盖和更新后的观察时间。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    policy = FoundationPolicy(version="v1")
    previous = initialize_library(source, output, NOW, policy, bindings())
    replace_reference_snapshot(source)

    current = initialize_library(source, output, NOW, policy, bindings())

    assert current["run_id"] != previous["run_id"]
    assert current["counts"] == previous["counts"]
    assert "row_adaptation_failed" not in current["reason_counts"]
    with ProductStore(output, read_only=True) as store:
        assert {item.run_id for item in store.records("coverage_slice")} == {current["run_id"]}
        assert [item.observed_at for item in store.records("availability_fact")] == [
            "2026-10-01T00:00:00Z"
        ]
        assert store.metadata("run:" + previous["run_id"])["summary"] == previous
    assert initialize_library(source, output, NOW, policy, bindings()) == current
    assert validate_library(source, output)["valid"]


def test_replaced_reference_removes_absent_facts_and_their_lineage(tmp_path):
    """场景：新快照已无旧行；预期：当前事实及其血缘一起消失，不计入新摘要。

    输入先有一行再清空的 daily 参考库，输出 availability_fact 与 typed_fact 均为 0。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    policy = FoundationPolicy(version="v1")
    previous = initialize_library(source, output, NOW, policy, bindings())
    with ProductStore(output, read_only=True) as store:
        previous_fact_ids = {
            row[0]
            for row in store.connection.execute(
                "SELECT record_id FROM product.typed_fact"
            ).fetchall()
        }
    replace_reference_snapshot(source, empty=True)

    current = initialize_library(source, output, NOW, policy, bindings())

    assert current["run_id"] != previous["run_id"]
    assert current["counts"]["coverage_slice"] == 206
    assert current["counts"]["availability_fact"] == 0
    assert current["counts"]["typed_fact"] == 0
    with ProductStore(output, read_only=True) as store:
        assert all(
            item.output_id not in previous_fact_ids for item in store.records("lineage_edge")
        )
        assert store.metadata("summary") == current
        store.verify_contents()


def test_new_policy_rebuilds_current_snapshot_and_preserves_history_summary(tmp_path):
    """场景：参考文件不变但策略版本变化；预期：当前表仅有新批次，历史摘要仍可查询。

    输入 v1 与 v2 两个策略，输出相同物理计数及两个 run 元数据摘要。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    previous = initialize_library(source, output, NOW, FoundationPolicy(version="v1"), bindings())

    current = initialize_library(source, output, NOW, FoundationPolicy(version="v2"), bindings())

    assert current["run_id"] != previous["run_id"]
    assert current["counts"] == previous["counts"]
    with ProductStore(output, read_only=True) as store:
        assert {item.run_id for item in store.records("coverage_slice")} == {current["run_id"]}
        assert store.metadata("run:" + previous["run_id"])["summary"] == previous
        assert store.metadata("run:" + current["run_id"])["summary"] == current


def test_failed_snapshot_rebuild_rolls_back_facts_and_metadata(tmp_path, monkeypatch):
    """场景：新快照替换期间消费失败；预期：旧事实、覆盖和元数据在事务回滚后完整保留。

    输入故意抛错的逐行消费函数，输出 RuntimeError；再次打开产品库仍是旧 run_id。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    policy = FoundationPolicy(version="v1")
    previous = initialize_library(source, output, NOW, policy, bindings())
    with ProductStore(output, read_only=True) as store:
        previous_facts = store.records("availability_fact")
    replace_reference_snapshot(source)

    def fail_consumption(*args, **kwargs):
        """模拟新快照消费中止；输入任意消费参数，输出 RuntimeError 以触发事务回滚。"""
        raise RuntimeError("synthetic_snapshot_failure")

    monkeypatch.setattr(reference_module, "consume_rows", fail_consumption)
    with pytest.raises(RuntimeError, match="synthetic_snapshot_failure"):
        initialize_library(source, output, NOW, policy, bindings())

    with ProductStore(output, read_only=True) as store:
        assert store.metadata("run_id") == previous["run_id"]
        assert store.metadata("summary") == previous
        assert store.counts() == previous["counts"]
        assert store.records("availability_fact") == previous_facts
        assert {item.run_id for item in store.records("coverage_slice")} == {previous["run_id"]}
        store.verify_contents()


@pytest.mark.parametrize(
    "corruption",
    (
        "coverage_and_gate_removed",
        "unexpected_api",
        "stale_run",
        "duplicate_api_scope",
        "readiness_gate_removed",
        "readiness_gate_renamed",
        "readiness_gate_reassigned",
    ),
)
def test_validation_rejects_incomplete_snapshot_even_with_matching_counts(tmp_path, corruption):
    """场景：覆盖或就绪门禁被篡改，但摘要计数已同步；预期：仍拒绝不完整快照。

    输入删掉 daily 覆盖及门禁的库，输出 valid=False，而普通模型验证仍可通过。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    initialize_library(source, output, NOW, FoundationPolicy(version="v1"), bindings())
    with ProductStore(output) as store:
        coverage = next(
            item for item in store.records("coverage_slice") if item.source_api == "daily"
        )
        coverage_id = store.connection.execute(
            "SELECT record_id FROM product.coverage_slice WHERE source_api='daily'"
        ).fetchone()[0]
        readiness = next(
            item
            for item in store.records("quality_gate_result")
            if item.result_id == coverage_id and item.gate.gate_id == "source_readiness"
        )
        if corruption == "coverage_and_gate_removed":
            store.connection.execute(
                "DELETE FROM product.coverage_slice WHERE record_id=?", [coverage_id]
            )
            store.connection.execute(
                "DELETE FROM product.quality_gate_result WHERE result_id=?", [coverage_id]
            )
        elif corruption in {"unexpected_api", "stale_run", "duplicate_api_scope"}:
            changes = {
                "unexpected_api": {"source_api": "unknown_api"},
                "stale_run": {"run_id": "previous-run"},
                "duplicate_api_scope": {"scope_id": "second_scope"},
            }[corruption]
            if corruption != "duplicate_api_scope":
                store.connection.execute(
                    "DELETE FROM product.coverage_slice WHERE record_id=?", [coverage_id]
                )
                store.connection.execute(
                    "DELETE FROM product.quality_gate_result WHERE result_id=?", [coverage_id]
                )
            replacement_id = store.put("coverage_slice", replace(coverage, **changes))
            store.put("quality_gate_result", replace(readiness, result_id=replacement_id))
        else:
            store.connection.execute(
                "DELETE FROM product.quality_gate_result WHERE result_id=?", [coverage_id]
            )
            if corruption == "readiness_gate_renamed":
                store.put(
                    "quality_gate_result",
                    replace(readiness, gate=replace(readiness.gate, gate_id="unrelated_gate")),
                )
            elif corruption == "readiness_gate_reassigned":
                store.put("quality_gate_result", replace(readiness, result_id="unrelated_result"))
        summary = store.metadata("summary")
        summary["counts"] = store.counts()
        store.save_metadata("summary", summary)
        store.verify_contents()

    assert not validate_library(source, output)["valid"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_id", "unrelated-run"),
        ("registered_apis", 205),
        ("schema_version", "product-foundation/unknown"),
        ("counts", {}),
    ),
)
def test_validation_reconciles_each_summary_identity_field(tmp_path, field, value):
    """场景：物理产品数据正常，但摘要身份、版本或数量不一致；预期：校验失败。

    输入 registered_apis=205 且实际覆盖206个API的摘要，输出 valid=False。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    initialize_library(source, output, NOW, FoundationPolicy(version="v1"), bindings())
    with ProductStore(output) as store:
        summary = store.metadata("summary")
        summary[field] = value
        store.save_metadata("summary", summary)

    assert not validate_library(source, output)["valid"]


@pytest.mark.parametrize("run_id", (None, "", "unrelated-run"))
def test_validation_requires_current_run_metadata(tmp_path, run_id):
    """场景：当前run元信息缺失或与摘要不符；预期：拒绝仅凭摘要宣称完整。

    输入 run_id=None 或无关运行号，输出 valid=False。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    initialize_library(source, output, NOW, FoundationPolicy(version="v1"), bindings())
    with ProductStore(output) as store:
        store.save_metadata("run_id", run_id)

    assert not validate_library(source, output)["valid"]


def test_summary_counts_require_integer_values(tmp_path):
    """场景：摘要用布尔值冒充物理计数1；预期：类型对账拒绝该摘要。

    输入 entity=True 而物理实体计数为1，输出 valid=False。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    initialize_library(source, output, NOW, FoundationPolicy(version="v1"), bindings())
    with ProductStore(output) as store:
        summary = store.metadata("summary")
        summary["counts"]["entity"] = True
        store.save_metadata("summary", summary)

    assert not validate_library(source, output)["valid"]


def test_same_run_revalidates_summary_after_manual_fact_addition(tmp_path):
    """场景：初始化后人工追加研究实体；预期：同run重跑拒绝返回过时计数，不删除事实。

    输入初始化库和额外合法实体，输出 ValueError；物理表仍保留两条实体供显式处理。
    """
    source = reference_fixture(tmp_path)
    output = tmp_path / "products.duckdb"
    policy = FoundationPolicy(version="v1")
    summary = initialize_library(source, output, NOW, policy, bindings())
    with ProductStore(output) as store:
        entity = bindings().entities[0]
        store.put("entity", replace(entity, identity=replace(entity.identity, local_id="B.SZ")))

    with pytest.raises(ValueError, match="product_summary"):
        initialize_library(source, output, NOW, policy, bindings())

    with ProductStore(output, read_only=True) as store:
        assert store.counts()["entity"] == 2
        assert store.metadata("summary") == summary
        store.verify_contents()


def test_manually_managed_store_does_not_require_initialized_snapshot(tmp_path):
    """场景：直接用ProductStore管理少量事实；预期：基本写读及模型校验不依赖完整初始化。

    输入一个手工实体而无run元数据，输出一条实体，verify_contents返回None。
    """
    with ProductStore(tmp_path / "manual.duckdb") as store:
        entity = bindings().entities[0]
        store.put("entity", entity)
        assert store.records("entity") == [entity]
        assert store.metadata("run_id") is None
        assert store.verify_contents() is None
