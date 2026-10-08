"""合成证券主数据的身份、别名、命名空间和正股关系。"""

from dataclasses import replace
from hashlib import sha256

import pytest
from test_research_binding_fixture import AS_OF

from tushare_entitlement_hub.equity_research.binding_entities import EntityBuild
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact


def entity_row(api="stock_basic", code="000001.SZ", **changes):
    """生成最小可追溯主数据与时间事实；默认输入股票代码，输出2000年起有效的合成记录。"""
    row = {
        "ts_code": code,
        "source_api": api,
        "source_record_id": f"{api}/{sha256(str(code).encode()).hexdigest()}",
        "business_group_key": f"group/{api}/{code}",
        "contract_hash": "synthetic-contract",
        "observed_at": AS_OF,
        "list_date": "2000-01-01",
        "list_status": "L",
        "industry": "合成行业",
        "name": "合成证券",
        "curr_type": "CNY",
    }
    row.update(changes)
    fact = AvailabilityFact(
        source_api=api,
        source_record_id=row["source_record_id"],
        certification_version="synthetic-v1",
        observed_at=AS_OF,
        data_as_of=None,
        effective_from="2000-01-01T00:00:00Z",
    )
    return row, fact


def test_entities_keep_namespaces_distinct_and_observed_peers():
    """同一代码分别出现在股票、指数、转债时保持三个身份，同行只给股票。

    输入三个000001.SZ主数据，输出3实体、3别名及1条转债正股关系。
    """
    build = EntityBuild(10)
    for api in ("stock_basic", "index_basic", "cb_basic"):
        build.add(*entity_row(api, stk_code="000001.SZ", cb_type="CB"))
    build.materialize()
    assert len(build.entities) == len(build.aliases) == 3
    assert len({item.identity.entity_id for item in build.entities}) == 3
    assert {item.identity.namespace for item in build.entities} == {
        "tushare_stock",
        "tushare_index",
        "tushare_cb",
    }
    assert len(build.peers) == 1
    assert next(iter(build.peers.values()))[0].startswith("observed-industry/")
    assert len(build.relations) == 1
    assert build.relations[0].relation_type == "cb_underlying"
    assert build.relations[0].confirmed
    assert set(build.currency_sources) == {"000001.SZ"}


@pytest.mark.parametrize("code", [None, "", "A.SZ", "000001.SH", "000001.SZ;DROP TABLE"])
def test_entity_codes_fail_closed(code):
    """代码缺失、错交易所或非法文本不构成实体；例如A.SZ记entity_code_unrecognized。"""
    build = EntityBuild(5)
    build.add(*entity_row(code=code))
    build.materialize()
    assert not build.entities
    assert build.reasons["stock_basic"]["entity_code_unrecognized"] == 1


def test_entity_duplicates_and_capacity_are_explicit():
    """重复代码整组降级，第二个独立代码超过容量即硬失败。

    输入两条000001.SZ输出0实体与重复原因；最大1实体时再加000002.SZ抛容量错误。
    """
    build = EntityBuild(1)
    build.add(*entity_row())
    build.add(*entity_row())
    build.materialize()
    assert not build.entities
    assert build.reasons["stock_basic"]["entity_duplicate_conflict"] == 1
    with pytest.raises(ValueError, match="binding_entity_capacity_exceeded"):
        build.add(*entity_row(code="000002.SZ"))


@pytest.mark.parametrize(
    ("changes", "time_changes", "reason"),
    [
        ({}, {"effective_from": None}, "entity_lifecycle_missing"),
        ({"list_status": "unknown"}, {}, "entity_lifecycle_conflict"),
        ({"list_status": "D"}, {}, "entity_lifecycle_conflict"),
        ({"market": "CDR"}, {}, "entity_lifecycle_conflict"),
        ({}, {"effective_to": "2025-01-01T00:00:00Z"}, "entity_lifecycle_conflict"),
    ],
)
def test_stock_lifecycle_gaps_prevent_identity(changes, time_changes, reason):
    """上市起点缺失、退市状态不一致或CDR均不生成可交易股票身份。

    例如list_status=D但无delist_date输出entity_lifecycle_conflict及0实体。
    """
    row, fact = entity_row(**changes)
    build = EntityBuild(5)
    build.add(row, replace(fact, **time_changes))
    build.materialize()
    assert not build.entities
    assert build.reasons["stock_basic"][reason] == 1


@pytest.mark.parametrize("industry", [None, "", "  ", 123])
def test_missing_industry_preserves_entity_without_peer_classification(industry):
    """缺少可用行业仍保留合法股票；例如industry空字符串输出1实体、0同行分类和缺口。"""
    build = EntityBuild(5)
    build.add(*entity_row(industry=industry))
    build.materialize()
    assert len(build.entities) == 1 and not build.peers
    assert build.reasons["stock_basic"]["industry_unclassified"] == 1


@pytest.mark.parametrize("known", ["430001.BJ", "920001.BJ"])
def test_bse_mapping_alias_uses_observation_boundary(known):
    """旧或新代码任一端已知、名称上市日相同才补另一别名，不回填历史。

    输入430001.BJ→920001.BJ，输出同实体两别名；补充别名从AS_OF起有效。
    """
    build = EntityBuild(5)
    build.add(*entity_row(code=known))
    build.add(*entity_row("bse_mapping", o_code="430001.BJ", n_code="920001.BJ"))
    build.materialize()
    assert len(build.entities) == 1 and len(build.aliases) == 2
    alias = next(item for item in build.aliases if item.source == "bse_mapping")
    assert alias.alias == ("920001.BJ" if known == "430001.BJ" else "430001.BJ")
    assert alias.effective_from == AS_OF
    assert len(alias.evidence_refs) == 2
    assert build.currency_sources["430001.BJ"][0]["curr_type"] == "CNY"


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        ("new_missing", "bse_code_unrecognized"),
        ("new_invalid", "bse_code_unrecognized"),
        ("non_bse", "bse_code_unrecognized"),
        ("old_non_bse", "bse_code_unrecognized"),
        ("both_known", "bse_mapping_conflict"),
        ("none_known", "bse_mapping_conflict"),
        ("same", "bse_mapping_conflict"),
        ("duplicate", "bse_mapping_conflict"),
        ("one_to_many", "bse_mapping_conflict"),
        ("name_missing", "bse_evidence_insufficient"),
        ("name_changed", "bse_evidence_insufficient"),
        ("date_changed", "bse_evidence_insufficient"),
        ("expired", "bse_lifecycle_expired"),
    ],
)
def test_bse_mapping_requires_unique_matching_live_evidence(variant, reason):
    """BSE别名仅接受唯一、同名同上市日期且生命周期有效的映射。

    例如两端均已存在不同实体，或新代码非法，均保留对应原因且不新增别名。
    """
    build = EntityBuild(10)
    stock, fact = entity_row(code="430001.BJ")
    if variant == "expired":
        stock["list_status"] = "D"
        fact = replace(fact, effective_to="2020-01-01T00:00:00Z")
    if variant != "none_known":
        build.add(stock, fact)
    if variant == "both_known":
        build.add(*entity_row(code="920001.BJ"))
    changes = {"o_code": "430001.BJ", "n_code": "920001.BJ"}
    changes.update(
        {
            "new_missing": {"n_code": None},
            "new_invalid": {"n_code": "invalid"},
            "non_bse": {"n_code": "000001.SZ"},
            "old_non_bse": {"o_code": "000001.SZ"},
            "same": {"n_code": "430001.BJ"},
            "name_missing": {"name": None},
            "name_changed": {"name": "其他名称"},
            "date_changed": {"list_date": "2001-01-01"},
        }.get(variant, {})
    )
    build.add(*entity_row("bse_mapping", **changes))
    if variant == "duplicate":
        build.add(*entity_row("bse_mapping", **changes))
    if variant == "one_to_many":
        build.add(*entity_row("bse_mapping", o_code="430002.BJ", n_code="920001.BJ"))
    build.materialize()
    assert not any(item.source == "bse_mapping" for item in build.aliases)
    assert build.reasons["bse_mapping"][reason] >= 1


@pytest.mark.parametrize("variant", ["missing_stock", "exchange_bond", "expired", "bounded"])
def test_convertible_bond_requires_underlying_and_overlapping_lifecycle(variant):
    """转债正股必须明确且生命周期相交；EB不推断，有限区间使用较早终点。

    输入2025年发行债与2020年已退市股票产生生命周期缺口；有效交集生成关系。
    """
    build = EntityBuild(5)
    stock, stock_fact = entity_row(list_status="D" if variant in {"expired", "bounded"} else "L")
    if variant in {"expired", "bounded"}:
        stock_fact = replace(stock_fact, effective_to="2020-01-01T00:00:00Z")
    if variant != "missing_stock":
        build.add(stock, stock_fact)
    bond, bond_fact = entity_row(
        "cb_basic",
        code="123001.SZ",
        stk_code="000001.SZ",
        cb_type="EB" if variant == "exchange_bond" else "CB",
    )
    if variant == "expired":
        bond_fact = replace(bond_fact, effective_from="2025-01-01T00:00:00Z")
    elif variant == "bounded":
        bond_fact = replace(bond_fact, effective_to="2021-01-01T00:00:00Z")
    build.add(bond, bond_fact)
    build.materialize()
    if variant == "bounded":
        assert build.relations[0].effective_to == "2020-01-01T00:00:00Z"
    else:
        assert not build.relations
        reason = (
            "cb_underlying_lifecycle_conflict"
            if variant == "expired"
            else "cb_underlying_unresolved"
        )
        assert build.reasons["cb_basic"][reason] == 1


def test_research_alias_index_keeps_all_candidates_and_namespaces():
    """别名索引保留同码多个候选和独立命名空间，缓存不改变原配置序列化。

    输入股票同码两条候选及指数同码一条，输出2个索引键；重复读取返回同一索引对象。
    """
    from tushare_entitlement_hub.equity_research.models import ResearchConfig
    from tushare_entitlement_hub.product_foundation.bindings import FoundationBindings

    build = EntityBuild(5)
    build.add(*entity_row())
    build.add(*entity_row("index_basic"))
    build.materialize()
    stock = next(item for item in build.aliases if item.namespace == "tushare_stock")
    config = ResearchConfig(foundation=FoundationBindings(aliases=(*build.aliases, stock)))
    before = config.to_json()
    index = config.aliases_by_key
    assert len(index["tushare_stock", "000001.SZ"]) == 2
    assert len(index["tushare_index", "000001.SZ"]) == 1
    assert index is config.aliases_by_key
    assert config.to_json() == before


def test_bse_alias_uses_true_fractional_observation_order():
    """整秒与同秒小数观察时间按真实时点比较，别名不得回退到更早证据。

    输入主数据12:00:00Z、换码12:00:00.5Z，新增别名起点与观察时间均为后者。
    """
    build = EntityBuild(5)
    build.add(*entity_row(code="430001.BJ"))
    row, fact = entity_row("bse_mapping", o_code="430001.BJ", n_code="920001.BJ")
    fact = replace(fact, observed_at="2026-10-03T12:00:00.500000Z")
    build.add(row, fact)
    build.materialize()
    alias = next(item for item in build.aliases if item.source == "bse_mapping")
    assert alias.effective_from == "2026-10-03T12:00:00.500000Z"
    assert alias.observed_at == "2026-10-03T12:00:00.500000Z"


def test_bond_relation_intersects_fractional_lifecycle_bounds():
    """同秒小数生命周期取真实较晚起点、较早终点，观察时点也取真实较晚值。

    输入股票生效00秒、债券生效00.5秒且股票终止01.5秒/债券终止01秒，
    输出关系[00.5秒,01秒)，不发生文本排序导致的区间倒置。
    """
    build = EntityBuild(5)
    stock, stock_fact = entity_row(list_status="D")
    stock_fact = replace(
        stock_fact,
        effective_from="2000-01-01T00:00:00Z",
        effective_to="2000-01-01T00:00:01.500000Z",
    )
    bond, bond_fact = entity_row("cb_basic", code="123001.SZ", stk_code="000001.SZ")
    bond_fact = replace(
        bond_fact,
        effective_from="2000-01-01T00:00:00.500000Z",
        effective_to="2000-01-01T00:00:01Z",
        observed_at="2026-10-03T12:00:00.500000Z",
    )
    build.add(stock, stock_fact)
    build.add(bond, bond_fact)
    build.materialize()
    relation = build.relations[0]
    assert relation.effective_from == "2000-01-01T00:00:00.500000Z"
    assert relation.effective_to == "2000-01-01T00:00:01Z"
    assert relation.observed_at == "2026-10-03T12:00:00.500000Z"


@pytest.mark.parametrize("invalid_new_master", ["duplicate", "lifecycle"])
def test_bse_alias_cannot_restore_conflicting_new_master(invalid_new_master):
    """新代码已有冲突主数据时，BSE映射不得绕过身份拒绝并借用旧代码实体。

    输入旧码合法、新码重复或退市生命周期缺失，即使名称上市日一致也不新增新码别名。
    """
    build = EntityBuild(10)
    build.add(*entity_row(code="430001.BJ"))
    build.add(
        *entity_row(code="920001.BJ", list_status="D" if invalid_new_master == "lifecycle" else "L")
    )
    if invalid_new_master == "duplicate":
        build.add(*entity_row(code="920001.BJ"))
    build.add(*entity_row("bse_mapping", o_code="430001.BJ", n_code="920001.BJ"))
    build.materialize()
    assert [item.alias for item in build.aliases] == ["430001.BJ"]
    assert build.reasons["bse_mapping"]["bse_mapping_conflict"] == 1
    assert "920001.BJ" not in build.currency_sources
