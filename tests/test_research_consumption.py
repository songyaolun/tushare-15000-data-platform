"""六产品消费注册、有限来源保留与真实持久化账本的离线契约。"""

from collections import Counter, defaultdict
from dataclasses import replace

import duckdb
import pytest
from research_fixture import AS_OF, DATABASE_CONFIG, write_reference
from test_research_adapter import CONTRACTS, configured, reference_database, source
from test_research_governance import NOW, A, inputs, relation, row

from tushare_entitlement_hub.equity_research import consumption_registry, pipeline
from tushare_entitlement_hub.equity_research.adapter import (
    SOURCE_APIS,
    adapt_row,
    load_inputs,
    retain_current,
    retention_identity,
    retention_period,
)
from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    consumption_rows,
    count_consumption,
    execute_consumption,
    field_quality,
    summarize_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import (
    CONSUMPTION_VERSION,
    PRODUCT_ALIASES,
    load_consumption_registry,
    registry_fingerprint,
    resolve_consumer,
    source_product,
    validate_consumption_registry,
)
from tushare_entitlement_hub.equity_research.models import (
    CANDIDATE_RANKING_PRODUCT_IDS,
    MARKET_ENTITY,
    PRODUCT_IDS,
    ResearchConfig,
)
from tushare_entitlement_hub.equity_research.pipeline import run_research
from tushare_entitlement_hub.equity_research.storage import (
    record_rows,
    run_identity,
    validate_research,
)
from tushare_entitlement_hub.product_foundation.bindings import TimeMapping
from tushare_entitlement_hub.product_foundation.registry import load_registry
from tushare_entitlement_hub.store import digest, encode


def registration(api):
    """取本地矩阵的消费绑定；例：hm_list返回flow处理器和dictionary范围。"""
    return next(item for item in load_consumption_registry() if item.spec.api == api)


def context_rows(datasets, relations=()):
    """把领域夹具标为已接受来源并交给既有输入工厂；例：daily一行生成一个来源分组。"""
    return inputs(
        *(
            replace(item, raw=item.raw | {"quality_tier": "verified"})
            for values in datasets.values()
            for item in values
        ),
        relations=relations,
    )


def test_every_assigned_api_has_real_handler_and_exact_csv_metadata():
    """场景：六产品矩阵99接口；预期：扫描集合、执行器和原CSV字段用途逐项一致。"""
    validate_consumption_registry()
    entries = load_consumption_registry()
    expected = {item.api: item for item in load_registry() if item.product in PRODUCT_ALIASES}
    assert len(entries) == len(SOURCE_APIS) == len(expected) == 99
    assert {item.spec.api for item in entries} == SOURCE_APIS == set(expected)
    assert Counter(item.handler for item in entries) == {
        "core": 48,
        "context": 26,
        "disclosure": 15,
        "flow": 10,
    }
    for item in entries:
        assert item.spec == expected[item.spec.api]
        assert callable(resolve_consumer(item.handler))
    assert len(registry_fingerprint()) == 64
    with pytest.raises(ValueError, match="handler_missing"):
        resolve_consumer("missing")


def test_missing_or_duplicate_registration_is_rejected(monkeypatch):
    """场景：新增接口名但没有处理器或两组重复声明；预期：不能靠凑99名称通过注册。"""
    original = consumption_registry.CORE_APIS
    for invalid in (original - {"daily"}, original | {"factor_list"}):
        monkeypatch.setattr(consumption_registry, "CORE_APIS", invalid)
        load_consumption_registry.cache_clear()
        with pytest.raises(ValueError, match="registry_mismatch"):
            load_consumption_registry()
    monkeypatch.setattr(consumption_registry, "CORE_APIS", original)
    load_consumption_registry.cache_clear()


@pytest.mark.parametrize(
    "previous_version",
    [
        "equity-consumption/v2",
        "equity-consumption/v3",
        "equity-consumption/v4",
        "equity-consumption/v5",
        "equity-consumption/v6",
        "equity-consumption/v7",
        "equity-consumption/v8",
        "equity-consumption/v9",
        "equity-consumption/v10",
        "equity-consumption/v11",
        "equity-consumption/v12",
        "equity-consumption/v13",
    ],
)
def test_registry_rule_version_prevents_reusing_original_v1_run_key(previous_version):
    """场景：同参考指纹和v1产品配置；预期：有界聚合不复用旧消费运行键，产品ID仍为v1。"""
    config = ResearchConfig()
    selection = (PRODUCT_ALIASES["P03"],)
    old_key = digest(["equity-research/v1", "synthetic", NOW, config.to_dict(), sorted(selection)])
    assert run_identity("synthetic", NOW, config, selection) != old_key
    previous_gate_key = digest(
        [
            "equity-research/v1",
            previous_version,
            registry_fingerprint(),
            "synthetic",
            NOW,
            config.to_dict(),
            sorted(selection),
        ]
    )
    assert CONSUMPTION_VERSION == "equity-consumption/v14"
    assert run_identity("synthetic", NOW, config, selection) != previous_gate_key
    assert all(name.endswith("/v1") for name in PRODUCT_ALIASES.values())


def test_empty_results_execute_all_99_without_claiming_observations():
    """场景：六产品没有任何数据；预期：99接口确实调用处理器，全部保持unknown。"""
    context = inputs(entities=())
    counts = {}
    for product in PRODUCT_IDS:
        result = make_result(context, product, MARKET_ENTITY)
        attached = attach_consumption(context, product, (result,))
        assert len(attached) == 1
        validate_result_consumption(attached[0])
        count_consumption(counts, attached[0])
        assert all(
            item["executed"] and item["retained_row_count"] == 0
            for item in attached[0].details["consumption"]
        )
    summary = summarize_consumption(counts, list(PRODUCT_ALIASES.values()))
    assert summary["version"] == CONSUMPTION_VERSION
    assert summary["assigned_apis"] == summary["registered_apis"] == summary["executed_apis"] == 99
    assert summary["observed_apis"] == 0
    assert summary["status_counts"]["unknown"] == 99


@pytest.mark.parametrize("product", CANDIDATE_RANKING_PRODUCT_IDS)
def test_candidate_rankings_reuse_p03_source_contract(product):
    """场景：三个新榜附加消费记录；预期：均复用 P03 的 16 项来源并可独立校验。"""
    context = inputs(entities=())
    result = make_result(context, product, MARKET_ENTITY)
    attached = attach_consumption(context, product, (result,))[0]
    validate_result_consumption(attached)
    assert source_product(product) == "P03"
    assert len(attached.details["consumption"]) == 16
    assert {item["product"] for item in attached.details["consumption"]} == {"P03"}


@pytest.mark.parametrize("api", ["factor_list", "hm_list", "index_basic", "cb_daily"])
def test_nonstock_sources_scan_without_equity_alias_and_preserve_original_fields(tmp_path, api):
    """场景：只有字典、指数或未关联转债并缺股票别名；预期：保留来源行，股票范围仍空。"""
    record = source(api, factor_name="alpha", name="席位说明", asset_type="stock")
    config = configured(api)
    config = replace(
        config,
        foundation=replace(
            config.foundation,
            entities=(),
            aliases=(),
            entity_mappings=(),
            time_mappings=(
                TimeMapping(
                    source_api=api, data_field="trade_date", market_timezone="Asia/Shanghai"
                ),
            ),
        ),
    )
    path = reference_database(tmp_path, {api: [record]})
    loaded, processing = load_inputs(path, NOW, "synthetic", config)
    assert processing["scanned_rows"] == processing["retained_rows"] == 1
    assert loaded.rows(api)[0].entity_id is None
    assert loaded.rows(api)[0].raw["source_record_id"] == record["source_record_id"]
    assert loaded.entity_ids == ()


def test_missing_business_field_remains_visible_quality_gap(tmp_path):
    """场景：来源缺一个非主键消费字段；预期：其余真实字段仍可观察，缺字段不补零。"""
    record = source()
    record.pop("close")
    path = reference_database(tmp_path, {"daily": [record]})
    loaded, processing = load_inputs(path, NOW, "synthetic", configured("daily"))
    assert processing["retained_rows"] == 1
    assert "consumption_fields_missing" in loaded.source_reasons["daily"]
    observation, references, checks = execute_consumption(loaded, registration("daily"), A)
    assert observation["field_quality"]["close"]["missing_rows"] == 1
    assert all("close" not in reference.fields for reference in references)
    assert all(not check.hard for check in checks)


def test_bond_relation_and_dictionary_matching_do_not_attribute_other_entities():
    """场景：同批两只债券与多条字典；预期：股票只消费已认证关系和实际用到的标签。"""
    bond = "cn:security:CB:fixture"
    related = replace(relation(), from_entity_id=bond, to_entity_id=A)
    context = context_rows(
        {
            "cb_daily": (row("cb_daily", entity=bond), row("cb_daily", "other", entity=None)),
            "factor_value": (row("factor_value", raw={"factor_name": "alpha"}),),
            "factor_list": (
                row("factor_list", entity=None, raw={"factor_name": "alpha"}),
                row("factor_list", "unused", entity=None, raw={"factor_name": "beta"}),
            ),
            "hm_detail": (row("hm_detail", raw={"hm_name": "seat"}),),
            "hm_list": (row("hm_list", entity=None, raw={"name": "seat"}),),
        },
        relations=(related,),
    )
    assert len(consumption_rows(context, registration("cb_daily"), A)) == 1
    assert len(consumption_rows(context, registration("cb_daily"), MARKET_ENTITY)) == 2
    assert len(consumption_rows(context, registration("factor_list"), A)) == 1
    assert len(consumption_rows(context, registration("hm_list"), A)) == 1
    context = replace(context, config=replace(context.config, confirmed_relations=()))
    assert consumption_rows(context, registration("cb_daily"), A) == ()


def test_field_quality_distinguishes_missing_nonfinite_and_uncertified():
    """场景：缺字段、NaN与未认证有限数共存；预期：质量计数分开且不生成认证数字。"""
    values = (
        row("daily", raw={"close": float("nan")}),
        row("daily", "empty"),
        row("daily", "finite", raw={"close": 10}),
    )
    assert field_quality(values, ("close",))["close"] == {
        "present_rows": 2,
        "missing_rows": 1,
        "certified_rows": 0,
        "nonfinite_rows": 1,
    }


def test_new_retention_preserves_frequencies_rating_agencies_and_two_periods():
    """场景：周月混合标签、两家评级机构及三期户数；预期：按真实统计终点与来源维度保留。"""
    config = ResearchConfig()
    groups = defaultdict(list)
    for index, (frequency, period) in enumerate(
        (
            ("week", "2026-09-04"),
            ("week", "2026-09-11"),
            ("week", "2026-09-18"),
            ("month", "2026-08-31"),
        )
    ):
        retain_current(
            groups,
            row("stk_weekly_monthly", str(index), raw={"freq": frequency, "end_date": period}),
            config,
        )
    assert len(groups[(A, "week", "")]) == 2
    assert retention_period(groups[(A, "month", "")][0]) == "2026-08-31"
    assert retention_period(row("stk_weekly_monthly", raw={"end_date": "bad"})) == ""
    ratings = defaultdict(list)
    for agency in ("agency-a", "agency-b"):
        for day in ("2026-09-01", "2026-09-02", "2026-09-03"):
            retain_current(
                ratings,
                row("cb_rating", agency + day, raw={"rating_com_name": agency, "ann_date": day}),
                config,
            )
    assert {key[1] for key in ratings} == {"agency-a", "agency-b"}
    assert all(len(values) == 2 for values in ratings.values())
    assert (
        retention_identity(row("cb_daily", entity=None, raw={"ts_code": "CB"})) == "unresolved/CB"
    )


@pytest.mark.parametrize("field", ["publish_date", "pub_time"])
def test_unmapped_public_time_is_rejected_before_consumption(field):
    """场景：新增来源含未映射公开时间；预期：拒绝行，不能靠observed_at绕过未来公告。"""
    record = source("cb_daily", **{field: "2026-10-04"})
    with pytest.raises(ValueError, match="public_field_mapping_missing"):
        adapt_row(record, CONTRACTS["cb_daily"], configured("cb_daily"), NOW)


@pytest.mark.parametrize("damage", ["missing", "status", "role", "quality", "reference"])
def test_consumption_tampering_is_rejected_without_trusting_embedded_claims(damage):
    """场景：消费记录被删除或改写状态、角色、质量、引用；预期：完整校验逐项拒绝。"""
    context = context_rows({"daily": (row("daily", raw={"close": 10}),)})
    product = PRODUCT_ALIASES["P03"]
    attached = attach_consumption(context, product, (make_result(context, product, A),))[0]
    details = attached.details
    record = next(item for item in details["consumption"] if item["api"] == "daily")
    if damage == "missing":
        details["consumption"].pop()
    elif damage == "status":
        record["status"] = "success"
    elif damage == "role":
        record["role"] = "invented"
    elif damage == "quality":
        record["field_quality"]["close"]["certified_rows"] = 2
    else:
        record["evidence_refs"][0]["source_record_id"] = "invented"
    with pytest.raises(ValueError, match="research_consumption"):
        validate_result_consumption(replace(attached, details_json=encode(details)))


def test_real_pipeline_stores_consumption_refs_and_reconstructs_summary(tmp_path):
    """场景：离线参考库执行完整六产品并重复运行；预期：99执行、真实血缘可回查且幂等。"""
    reference, config = write_reference(tmp_path, 1)
    output = tmp_path / "consumption.duckdb"
    summary = run_research(reference, output, AS_OF, config)
    assert summary["consumption"]["executed_apis"] == 99
    assert summary["consumption"]["observed_apis"] > 0
    assert validate_research(reference, output)["valid"]
    assert run_research(reference, output, AS_OF, config) == summary
    with duckdb.connect(str(output), config=DATABASE_CONFIG) as connection:
        assert connection.execute("SELECT count(*) FROM research.evidence").fetchone()[0] > 0


def test_consumption_evidence_enters_projection_without_changing_core_score():
    """场景：原产品补上当前字段消费；预期：新增证据/血缘可持久化，分数贡献不被旁证修改。"""
    context = context_rows({"daily": (row("daily", raw={"close": 10}),)})
    product = PRODUCT_ALIASES["P03"]
    original = make_result(context, product, A)
    attached = attach_consumption(context, product, (original,))[0]
    assert attached.envelope.score == original.envelope.score
    assert attached.envelope.contributions == original.envelope.contributions
    assert any(table == "evidence" for table, _ in record_rows(attached))
    assert any(table == "lineage" for table, _ in record_rows(attached))


def test_missing_handler_fails_before_creating_product_database(tmp_path, monkeypatch):
    """场景：注册只剩名称而真实实现不可用；预期：运行在打开产品库前失败。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "must-not-exist.duckdb"

    def missing_handler():
        """模拟注册校验失败；例：被调用时抛出handler_missing且不执行任何数据库写入。"""
        raise ValueError("research_consumption_handler_missing")

    monkeypatch.setattr(pipeline, "validate_consumption_registry", missing_handler)
    with pytest.raises(ValueError, match="handler_missing"):
        run_research(reference, output, AS_OF, config)
    assert not output.exists()


@pytest.mark.parametrize("status", ["invented", None, 1])
def test_execute_rejects_handler_status_outside_registered_enum(monkeypatch, status):
    """场景：处理器返回未注册状态或非字符串；预期：包装器拒绝错误契约，不持久化状态。"""
    from tushare_entitlement_hub.equity_research import consumption

    def handler(*args):
        """返回参数化无效状态；例：invented应在读取其余字段前被包装器拒绝。"""
        return {"status": status}

    monkeypatch.setattr(consumption, "resolve_consumer", lambda name: handler)
    with pytest.raises(ValueError, match="research_consumption_status_invalid"):
        execute_consumption(inputs(), registration("daily"), A)


def test_execute_retains_quality_rejection_reason_and_no_false_observation():
    """场景：消费候选为隔离daily行；预期：保留拒绝原因、合格行数0且状态unknown。"""
    context = context_rows({"daily": (row("daily", raw={"close": 10, "is_quarantined": True}),)})
    outcome, refs, _ = execute_consumption(context, registration("daily"), A)
    assert outcome["status"] == "unknown"
    assert outcome["retained_row_count"] == 0
    assert "consumption_rows_rejected" in outcome["reason_codes"]
    assert refs == ()


@pytest.mark.parametrize("mutation", ["version", "not_list", "negative_count", "wrong_fields"])
def test_persisted_consumption_rejects_missing_schema_and_invalid_row_count(mutation):
    """场景：账本版本、列表类型、行数或字段集合被篡改；预期：验证明确拒绝不完整契约。"""
    context = context_rows({"daily": (row("daily", raw={"close": 10}),)})
    product = PRODUCT_ALIASES["P03"]
    original = attach_consumption(context, product, (make_result(context, product, A),))[0]
    details = original.details
    if mutation == "version":
        details["consumption_version"] = "obsolete"
    elif mutation == "not_list":
        details["consumption"] = {}
    elif mutation == "negative_count":
        details["consumption"][0]["retained_row_count"] = -1
    else:
        details["consumption"][0]["field_quality"] = {}
    with pytest.raises(ValueError, match="research_consumption_(missing|field_quality_invalid)"):
        validate_result_consumption(replace(original, details_json=encode(details)))


def test_source_observation_retains_unlinked_dictionary_without_inventing_period():
    """场景：游资字典无股票归属且没有来源日期索引；预期：增设来源结果，引用保留而日期不编造。"""
    dictionary = row(
        "hm_list", entity=None, day=None, raw={"name": "甲", "desc": "合成", "orgs": "营业部"}
    )
    context = context_rows({"hm_list": (dictionary,)})
    product = PRODUCT_ALIASES["P07"]
    original = make_result(context, product, A)
    results = attach_consumption(context, product, (original,))
    assert len(results) == 2
    standalone = next(result for result in results if result.envelope.entity_id == MARKET_ENTITY)
    assert standalone.details["stock_candidate"] is False
    assert standalone.envelope.source_as_of == {}
    assert any(ref.source_api == "hm_list" for ref in standalone.envelope.evidence_refs)
    validate_result_consumption(standalone)


def test_metadata_with_entity_mapping_but_unknown_business_time_stays_unlinked():
    """场景：游资字典有可匹配股票别名但无业务生效日期；预期：保留字典原值，不强行归因股票。"""
    from tushare_entitlement_hub.product_foundation.bindings import EntityMapping

    record = source("hm_list", name="合成游资", desc="供应商标签", orgs="合成营业部")
    config = configured("hm_list")
    config = replace(
        config,
        foundation=replace(
            config.foundation,
            entity_mappings=(
                EntityMapping(source_api="hm_list", namespace="synthetic", key_field="ts_code"),
            ),
            time_mappings=(TimeMapping(source_api="hm_list", market_timezone="Asia/Shanghai"),),
        ),
    )
    adapted = adapt_row(record, CONTRACTS["hm_list"], config, NOW)
    assert adapted.entity_id is None
    assert adapted.availability.data_as_of is None
    assert adapted.availability.effective_from is None
    assert adapted.raw["name"] == "合成游资"
    assert adapted.raw["ts_code"] == "A"
    assert adapted.evidence("name")[0].source_record_id == record["source_record_id"]
