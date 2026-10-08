"""披露旁证索引的语义等价与访问次数回归，全部使用确定性合成来源。"""

from collections import Counter
from dataclasses import replace

import pytest
from test_research_disclosure_consumers import bond_context, consume, quote_sources, source
from test_research_governance import BOND, A, B, inputs, relation

from tushare_entitlement_hub.equity_research import disclosure_consumers as disclosure
from tushare_entitlement_hub.equity_research.models import ResearchRow


def legacy_other_rows(context, api, original):
    """保留旧版全表过滤作为语义基准；例：有实体时忽略代码，无实体时仅按非空代码比较。"""
    return tuple(
        candidate
        for candidate in context.rows(api)
        if disclosure._visible(context, candidate)
        and (
            candidate.entity_id == original.entity_id
            if original.entity_id is not None
            else original.text("ts_code") is not None
            and candidate.text("ts_code") == original.text("ts_code")
        )
    )


@pytest.fixture(scope="module")
def unrelated_daily_basics():
    """创建4096个不同证券的无关底表行；各行身份独立且可见，不能靠预先隔离减少旧查询成本。"""
    return tuple(
        source("daily_basic", f"unrelated-{index}", entity=f"cn:security:U{index}:life")
        for index in range(4096)
    )


@pytest.mark.parametrize("mapped", [True, False])
def test_only_matching_candidates_reach_visibility_gate(
    unrelated_daily_basics, monkeypatch, mapped
):
    """4096条无关行并存时，仅检查同实体/同代码的6条候选；质量、隔离和未来时点过滤与旧版相同。"""
    identity = A if mapped else None
    original = source("bak_basic", entity=identity, raw={"ts_code": "MATCH"})
    valid = source("daily_basic", "current", entity=identity, raw={"ts_code": "MATCH"})
    old = source("daily_basic", "old", entity=identity, raw={"ts_code": "MATCH"}, day="2026-10-02")
    rejected = (
        replace(valid, raw=valid.raw | {"is_quarantined": True}),
        replace(valid, raw=valid.raw | {"quality_tier": "unknown"}),
        replace(
            valid,
            availability=replace(valid.availability, public_available_at="2026-10-04T00:00:00Z"),
        ),
        replace(valid, raw=valid.raw | {"ann_date": "20261004"}),
    )
    candidates = (old, *rejected, valid)
    context = inputs(*unrelated_daily_basics[:2048], *candidates, *unrelated_daily_basics[2048:])
    checked = []
    visible = disclosure._visible

    def counted_visible(selected_inputs, candidate):
        """透传真实门禁并记录检查对象；例如未来同实体行也必须经过门禁后才被拒绝。"""
        checked.append(candidate)
        return visible(selected_inputs, candidate)

    monkeypatch.setattr(disclosure, "_visible", counted_visible)
    expected = legacy_other_rows(context, "daily_basic", original)
    before = len(checked)
    checked.clear()
    actual = disclosure._other_rows(context, "daily_basic", original)
    assert actual == expected == (old, valid)
    assert checked == list(candidates)
    assert before == 4102 and len(checked) == 6
    assert ("code_rows" in context.__dict__) == (not mapped)
    checked.clear()
    assert disclosure._other_rows(context, "daily_basic", original) == actual
    assert checked == list(candidates)
    print(f"{'实体' if mapped else '代码'}旁证门禁检查：{before} -> {len(checked)}")


@pytest.mark.parametrize("scenario", ["unique", "missing", "ambiguous", "duplicate"])
def test_backup_result_is_identical_to_full_scan(unrelated_daily_basics, monkeypatch, scenario):
    """真实P04备份核对在唯一、缺失、歧义、重复场景下与旧版结果完全一致，重排行序也保持原行为。"""
    values = {"pb": (2, "ratio"), "float_share": (1000, "股")}
    backup = source("bak_basic", "backup", values=values)
    current = source("daily_basic", "current", values=values)
    candidates = (
        ()
        if scenario == "missing"
        else (current, source("daily_basic", "conflict", values=values))
        if scenario == "ambiguous"
        else (current, current)
        if scenario == "duplicate"
        else (current,)
    )
    indexed = disclosure._other_rows
    for ordered in (candidates, tuple(reversed(candidates))):
        context = inputs(backup, *unrelated_daily_basics, *ordered)
        with monkeypatch.context() as patch:
            patch.setattr(disclosure, "_other_rows", legacy_other_rows)
            expected = consume("bak_basic", backup, context=context)
        assert disclosure._other_rows is indexed
        actual = consume("bak_basic", backup, context=context)
        assert actual == expected
        assert actual["status"] == (
            "unknown" if scenario in {"missing", "ambiguous"} else "observed"
        )


def test_entity_lookup_never_falls_back_to_equal_code(monkeypatch):
    """实体已知但仅有其他实体的同代码旁证时，不按代码回退，也不检查无关行的可见性。"""
    original = source("bak_basic", entity=A, raw={"ts_code": "SAME"})
    other = source("daily_basic", entity=B, raw={"ts_code": "SAME"})

    def unexpected(*args):
        """无同实体候选时不应调用门禁；任何调用均代表发生了全表扫描或代码回退。"""
        pytest.fail("不应检查不同实体的同代码行")

    monkeypatch.setattr(disclosure, "_visible", unexpected)
    context = inputs(other)
    assert disclosure._other_rows(context, "daily_basic", original) == ()
    assert "code_rows" not in context.__dict__


@pytest.mark.parametrize("code", [None, "", "  ", {}, True])
def test_missing_code_without_entity_does_not_scan_or_build_index(code, monkeypatch):
    """缺实体且代码不可读时直接返回空，不把None/空白/结构体/布尔代码连接到其他缺字段行。"""
    original = source("cb_rate", entity=None, raw={"ts_code": code})
    context = inputs(source("cb_basic", entity=None))

    def unexpected(*args):
        """没有合法连接键时不应检查任何旁证；任何调用都使本测试失败。"""
        pytest.fail("缺代码不能触发全表门禁")

    monkeypatch.setattr(disclosure, "_visible", unexpected)
    assert disclosure._other_rows(context, "cb_basic", original) == ()
    assert "code_rows" not in context.__dict__


def test_code_index_is_cached_preserves_order_and_text_semantics(monkeypatch):
    """代码索引只构造一次，保留重复和已绑定行，数字沿用text口径，空白不被擅自规范化。"""
    first = source("cb_basic", "first", entity=None, raw={"ts_code": "CB1"})
    second = source("cb_basic", "second", entity=BOND, raw={"ts_code": "CB1"})
    numeric = source("cb_basic", "zero", entity=None, raw={"ts_code": 0})
    padded = source("cb_basic", "padded", entity=None, raw={"ts_code": " CB1 "})
    missing = source("cb_basic", "missing", entity=None, raw={"ts_code": None})
    other_api = source("cb_daily", "quote", entity=None, raw={"ts_code": "CB1"})
    context = inputs(first, missing, numeric, second, first, padded, other_api)
    calls = Counter()
    original_text = ResearchRow.text

    def counted_text(item, name):
        """透传原text读取并记录调用；例如一次构建只为每条输入读取一次ts_code。"""
        calls[name] += 1
        return original_text(item, name)

    monkeypatch.setattr(ResearchRow, "text", counted_text)
    index = context.code_rows
    assert calls == {"ts_code": 7}
    assert context.code_rows is index and calls == {"ts_code": 7}
    assert index["cb_basic", "CB1"] == (first, second, first)
    assert index["cb_basic", "CB1"][0] is first
    assert index["cb_basic", "0"] == (numeric,)
    assert index["cb_basic", " CB1 "] == (padded,)
    assert index["cb_daily", "CB1"] == (other_api,)
    assert len(index) == 4
    original = source("cb_rate", entity=None, raw={"ts_code": "CB1"})
    assert disclosure._other_rows(context, "cb_basic", original) == (first, second, first)
    assert disclosure._other_rows(context, "absent", original) == ()
    changed = replace(context, rows_by_api={"cb_basic": (second,)})
    assert changed.code_rows is not index
    assert changed.code_rows["cb_basic", "CB1"] == (second,)


def test_relations_index_preserves_gates_order_duplicates_and_config_identity(monkeypatch):
    """2048条无关关系只遍历一次，重复查询只重验本债确认/有效期；冲突、重复、配置指纹均保留。"""
    good = relation()
    conflict = replace(good, to_entity_id=B)
    unconfirmed = replace(good, confirmed=False)
    expired = replace(good, effective_to="2026-01-01T00:00:00Z")
    future = replace(good, observed_at="2026-10-04T00:00:00Z")
    other_type = replace(good, relation_type="other_relation")
    candidates = (good, conflict, unconfirmed, expired, future, good)
    unrelated = tuple(
        replace(good, from_entity_id=f"cn:security:CB{index}:life") for index in range(2048)
    )
    context = inputs(relations=(*unrelated, *candidates, other_type))
    before = context.config.to_json()
    visited = Counter()

    class CountedRelations(tuple):
        """仅给同一关系元组加迭代计数，不改变顺序或任何关系字段。"""

        def __iter__(self):
            """透传每个原关系并计数；例如遍历一次N条关系得到iterations=1、rows=N。"""
            visited["iterations"] += 1
            for item in tuple.__iter__(self):
                visited["rows"] += 1
                yield item

    object.__setattr__(
        context.config, "confirmed_relations", CountedRelations(context.config.confirmed_relations)
    )
    checked = []
    usable_at = type(good).usable_at

    def counted_usable(item, cutoff):
        """透传真实时点门禁并记录候选；未确认及无关关系不应进入本次时点校验。"""
        checked.append(item)
        return usable_at(item, cutoff)

    monkeypatch.setattr(type(good), "usable_at", counted_usable)
    original = source("cb_daily", entity=BOND)
    for _ in range(10):
        assert disclosure._underlying_relations(context, original) == (good, conflict, good)
    assert visited == {"iterations": 1, "rows": 2055}
    assert checked == [good, conflict, expired, future, good] * 10
    assert context.config.relations_by_source[BOND, "cb_underlying"] == candidates
    assert context.config.to_json() == before
    assert disclosure._conversion_value(context, original)[2] == "cb_relation_unverified"
    no_entity = source("cb_daily", entity=None)
    assert disclosure._underlying_relations(context, no_entity) == ()
    later = replace(context, analysis_as_of="2026-10-04T12:00:00Z")
    assert disclosure._underlying_relations(later, original) == (good, conflict, future, good)
    assert later.config.relations_by_source is context.config.relations_by_source
    changed = replace(context.config, confirmed_relations=(good,))
    assert changed.relations_by_source[BOND, "cb_underlying"] == (good,)
    print("10次债券关系查询的全表遍历：20550 -> 2055")


def test_bond_quote_paths_reuse_relation_and_basic_indexes(monkeypatch):
    """真实债券消费中的转股核对与最终关系附加共用索引，cb_basic不再全表遍历，输出保持observed。"""
    records = quote_sources()
    context = bond_context(*records)
    basic = context.rows("cb_basic", BOND)
    original_rows = type(context).rows
    calls = Counter()

    def indexed_rows(item, api, entity_id=None):
        """记录真实数据访问；例如cb_basic查询必须携带债券实体，daily必须携带正股实体。"""
        calls[api, entity_id] += 1
        if api in {"cb_basic", "daily"}:
            assert entity_id is not None
        return original_rows(item, api, entity_id)

    monkeypatch.setattr(type(context), "rows", indexed_rows)
    result = consume("cb_daily", *records, context=context)
    assert result["status"] == "observed"
    assert calls["cb_basic", BOND] == 2
    assert calls["daily", A] == 1
    assert context.rows("cb_basic", BOND) == basic
    assert len(context.config.relations_by_source) == 1
