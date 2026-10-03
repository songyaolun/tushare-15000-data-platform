"""P13 父身份、完整内容指纹、版本歧义和实体边界的确定性回归。"""

import hashlib
import json
from dataclasses import replace

import pytest
from test_research_events import event_inputs, event_relation, research_inputs, thesis_parents

from tushare_entitlement_hub.equity_research import event_parents
from tushare_entitlement_hub.equity_research.common import gate
from tushare_entitlement_hub.equity_research.events import build_events
from tushare_entitlement_hub.equity_research.models import (
    MARKET_ENTITY,
    ResearchInputs,
    ResearchResult,
)
from tushare_entitlement_hub.product_foundation.contracts import ProductState, Truth
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityIdentity
from tushare_entitlement_hub.store import digest, encode


def test_streamed_fingerprint_matches_full_json_without_model_serialization(monkeypatch):
    """完整字段、中文、转义和复用凭据均参与规范摘要；流式路径不调用to_json/asdict。"""
    parent = replace(thesis_parents(research_inputs(1))[0], details_json=encode({"备注": "甲\n乙"}))
    expected = hashlib.sha256(parent.to_json().encode()).hexdigest()

    def unexpected(*args, **kwargs):
        """摘要只能投影声明字段；如调用深拷贝或完整模型序列化则立即失败。"""
        pytest.fail("父结果发生完整序列化或深拷贝")

    with monkeypatch.context() as patch:
        patch.setattr(ResearchResult, "to_json", unexpected)
        patch.setattr("dataclasses.asdict", unexpected)
        assert event_parents.parent_fingerprint(parent) == expected
    with pytest.raises(TypeError, match="^research_event_parent_fingerprint_type_invalid$"):
        event_parents._model_fields(object())


def test_equal_duplicate_objects_are_idempotent_and_only_duplicates_use_json(monkeypatch):
    """新对象但相同ID和内容可幂等去重，只有这两个重复候选调用完整JSON比较。"""
    inputs = research_inputs(1)
    parents = thesis_parents(inputs)
    duplicate = replace(parents[0])
    expected = build_events(inputs, parents)
    original = ResearchResult.to_json
    calls = []

    def counted(value):
        """记录序列化对象；预期仅出现同ID的原父和重复父，不出现其他唯一父。"""
        calls.append(id(value))
        return original(value)

    with monkeypatch.context() as patch:
        patch.setattr(ResearchResult, "to_json", counted)
        actual = build_events(inputs, parents + (duplicate,))
    assert calls == [id(parents[0]), id(duplicate)]
    assert actual == expected


@pytest.mark.parametrize("reverse", [False, True])
def test_same_identity_content_conflict_fails_closed_in_either_order(reverse):
    """同result_id只改未被论点直接使用的明细仍是内容冲突；调换输入顺序也报同一固定码。"""
    inputs = research_inputs(1)
    parent = thesis_parents(inputs)[0]
    changed = replace(parent, details_json='{"changed":true}')
    candidates = (changed, parent) if reverse else (parent, changed)
    with pytest.raises(ValueError, match="^research_event_parent_conflict$"):
        build_events(inputs, candidates)


@pytest.mark.parametrize("parent_index", [0, 1, 2])
def test_any_parent_detail_change_updates_bundle_digest_without_losing_lineage(parent_index):
    """三个父产品任意一个未展示明细变化都改变全量摘要；ID集合未变时事件ID和全部血缘保留。"""
    inputs = research_inputs(1)
    parents = thesis_parents(inputs)
    before = build_events(inputs, parents)[0]
    changed = list(parents)
    changed[parent_index] = replace(parents[parent_index], details_json='{"unselected":1}')
    after = build_events(inputs, tuple(changed))[0]
    assert after.details["upstream_fingerprint"] != before.details["upstream_fingerprint"]
    assert after.envelope.scope_id == before.envelope.scope_id
    assert after.parent_result_ids == before.parent_result_ids
    assert after.to_json() != before.to_json()
    changed[parent_index] = replace(
        changed[parent_index], envelope=replace(changed[parent_index].envelope, scope_id="revision")
    )
    revised = build_events(inputs, tuple(changed))[0]
    assert revised.envelope.scope_id != before.envelope.scope_id


def test_all_versions_keep_references_hard_checks_and_contrary_evidence():
    """同产品不同ID的版本全保留；现金覆盖未知，严格硬门禁与全部反证进入唯一bundle。"""
    inputs = research_inputs(1)
    fundamental, governance, flow = thesis_parents(inputs)
    references = inputs.rows("daily")[0].evidence("close")
    other = replace(
        fundamental,
        envelope=replace(
            fundamental.envelope,
            scope_id="revision",
            contrary_evidence=references,
            quality_gates=(
                gate("revision_check", Truth.FALSE, "revision_rejected", references=references),
            ),
        ),
    )
    result = build_events(inputs, (fundamental, governance, flow, other))[0]
    assert result.details["upstream_parent_count"] == 4
    assert result.details["upstream_ambiguous_products"] == [fundamental.envelope.product_id]
    assert result.metric("cash_cover") is None
    assert len(result.parent_result_ids) == 4
    assert result.envelope.state is ProductState.BLOCKED
    assert set(references) <= set(result.envelope.contrary_evidence)
    assert any(
        check.reason_codes == ("revision_rejected",) for check in result.envelope.quality_gates
    )
    assert "upstream_product_versions_ambiguous" in result.envelope.gate_codes


def test_wrong_cutoff_is_excluded_from_bundles_and_direct_event_parents():
    """未来截止父结果不参与bundle或同实体真实事件；没有有效父时保持空来源降级。"""
    inputs = research_inputs(1)
    parents = thesis_parents(inputs)
    future = replace(
        parents[0],
        envelope=replace(
            parents[0].envelope,
            analysis_as_of="2026-10-04T00:00:00Z",
            generated_at="2026-10-04T00:00:00Z",
        ),
    )
    inputs = event_inputs(
        inputs, "namechange", {"start_date": "20260930"}, identity=inputs.entity_ids[0]
    )
    results = build_events(inputs, parents[1:] + (future,))
    assert len(results) == 2
    for result in results:
        assert result.parent_result_ids == tuple(
            sorted(parent.envelope.result_id for parent in parents[1:])
        )
        assert result.metric("cash_cover") is None
    empty = build_events(research_inputs(0), (future,))[0]
    assert empty.parent_result_ids == () and empty.details["event_count"] == 0


@pytest.mark.parametrize("identity", [None, ""])
def test_missing_parent_entity_cannot_enter_an_index(identity):
    """绕过公开模型验证构造缺实体父结果时，索引仍以固定错误拒绝，不归入市场或任意证券。"""
    inputs = research_inputs(1)
    parent = thesis_parents(inputs)[0]
    object.__setattr__(parent.envelope, "entity_id", identity)
    with pytest.raises(ValueError, match="^research_event_parent_entity_missing$"):
        build_events(inputs, (parent,))


def test_market_bundle_is_separate_and_unresolved_event_inherits_no_parents():
    """市场父单独形成待确认bundle，不传播到证券；无实体新闻也不能继承市场或证券父风险。"""
    inputs = research_inputs(1)
    parents = thesis_parents(inputs)
    market = replace(parents[0], envelope=replace(parents[0].envelope, entity_id=MARKET_ENTITY))
    inputs = event_inputs(inputs, "major_news", {"title": "无实体的新闻"})
    results = build_events(inputs, parents + (market,))
    assert len(results) == 3
    by_scope = {
        (result.details["event_type"], result.details["confirmed_entity"]): result
        for result in results
    }
    stock = by_scope[("upstream_bundle", inputs.entity_ids[0])]
    market_result = by_scope[("upstream_bundle", MARKET_ENTITY)]
    unresolved = by_scope[("major_news", None)]
    assert len(stock.parent_result_ids) == 3
    assert market_result.parent_result_ids == (market.envelope.result_id,)
    assert market_result.predicate_results["entity_confirmation"] is Truth.UNKNOWN
    assert market_result.details["conflict_reason"] == "upstream_market_scope_not_security"
    assert unresolved.parent_result_ids == ()
    assert unresolved.details["upstream_fingerprint"] is None


def test_parent_capacity_counts_unique_current_versions_without_truncation():
    """64个当前唯一父版本全部进入血缘，重复不占容量；第65个版本明确失败，不截断父证据。"""
    inputs = research_inputs(1)
    parent = thesis_parents(inputs)[0]
    parents = tuple(
        replace(parent, envelope=replace(parent.envelope, scope_id=f"version-{index}"))
        for index in range(65)
    )
    result = build_events(inputs, parents[:64] * 2)[0]
    assert len(result.parent_result_ids) == event_parents.MAXIMUM_ENTITY_PARENTS == 64
    assert result.details["upstream_parent_count"] == 64
    assert result.metric("cash_cover") is None
    with pytest.raises(ValueError, match="^research_event_parent_capacity_exceeded$"):
        build_events(inputs, parents)


def test_bundle_identity_binds_the_sorted_complete_parent_set():
    """bundle事件身份绑定实体及排序后的全部父ID；逆序输入不会选择代表父或改变输出。"""
    inputs = research_inputs(1)
    parents = thesis_parents(inputs)
    result = build_events(inputs, parents)[0]
    expected = (
        "research:event:"
        + digest(
            [
                "upstream_bundle",
                inputs.entity_ids[0],
                sorted(parent.envelope.result_id for parent in parents),
            ]
        )
        + ":v1"
    )
    assert result.envelope.scope_id == expected
    assert result.to_json() == build_events(inputs, tuple(reversed(parents)))[0].to_json()


@pytest.mark.parametrize(
    "reason", sorted(event_parents.PARENT_FAILURE_CODES) + ["unrecognized_error_detail"]
)
def test_parent_failures_reach_cli_as_fixed_codes_only(monkeypatch, tmp_path, reason):
    """父冲突、缺实体和容量错误从CLI输出固定原因；其他异常内容仍被通用错误替代。"""
    from typer.testing import CliRunner

    from tushare_entitlement_hub.cli import app
    from tushare_entitlement_hub.equity_research import cli

    def failed(*args, **kwargs):
        """模拟父索引边界失败，不打开任何参考库；CLI应退出1并只披露白名单原因。"""
        raise ValueError(reason)

    monkeypatch.setattr(cli, "run_research", failed)
    result = CliRunner().invoke(
        app,
        [
            "research",
            "run",
            "--reference",
            str(tmp_path / "ref"),
            "--output",
            str(tmp_path / "out"),
            "--as-of",
            "2026-10-03T00:00:00Z",
            "--product",
            "P13",
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "reason_codes": [
            reason if reason in event_parents.PARENT_FAILURE_CODES else "research_run_failed"
        ],
    }


@pytest.mark.parametrize("api", ["cb_daily", "index_daily"])
def test_entity_universe_is_cached_per_snapshot_and_replaced_inputs_recompute(monkeypatch, api):
    """反复读取实体集只遍历一次；replace更换行、关系或实体配置后重算，债券/指数/series过滤不变。"""
    inputs = research_inputs(3)
    identities = tuple(entity.identity.entity_id for entity in inputs.config.foundation.entities)
    original = EntityIdentity.entity_id.fget
    reads = []

    def counted(identity):
        """透传身份构造并计数；100次旧计算读取600次，新缓存只读取6次。"""
        reads.append(identity.local_id)
        return original(identity)

    with monkeypatch.context() as patch:
        patch.setattr(EntityIdentity, "entity_id", property(counted))
        for _ in range(100):
            assert ResearchInputs.entity_ids.func(inputs) == identities
        assert len(reads) == 600
        reads.clear()
        for _ in range(100):
            assert inputs.entity_ids == identities
        assert len(reads) == 6
    cached = inputs.entity_ids
    changed = replace(inputs, rows_by_api={**inputs.rows_by_api, api: (inputs.rows("daily")[0],)})
    assert changed.entity_ids == identities[1:]
    relation = event_relation(inputs, identities[2], identities[1], relation_type="cb_underlying")
    series = Entity(
        identity=EntityIdentity(
            namespace="cn", kind="series", local_id="macro", lifecycle_id="synthetic"
        ),
        name="合成序列",
    )
    final = replace(
        changed,
        config=replace(
            changed.config,
            confirmed_relations=(relation,),
            foundation=replace(
                changed.config.foundation, entities=changed.config.foundation.entities + (series,)
            ),
        ),
    )
    assert final.entity_ids == (identities[1],)
    assert inputs.entity_ids is cached and changed.entity_ids == identities[1:]
