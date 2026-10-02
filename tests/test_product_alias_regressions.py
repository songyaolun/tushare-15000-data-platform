"""历史来源行与别名生命周期的离线回归，测试先于修复实现。"""

from dataclasses import replace

import duckdb
import pytest
from test_product_foundation import ENTITY, NOW, evidence
from test_product_storage import bindings, reference_fixture

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.product_foundation.bindings import TimeMapping
from tushare_entitlement_hub.product_foundation.entities import resolve_alias
from tushare_entitlement_hub.product_foundation.reference_scan import consume_rows
from tushare_entitlement_hub.product_foundation.registry import load_registry
from tushare_entitlement_hub.product_foundation.storage import ProductStore
from tushare_entitlement_hub.reference_contract import load_reference_contracts

NEW_ENTITY = "cn:security:A.SZ:listing2"
REUSED_AT = "2025-01-01T00:00:00Z"


def reused_aliases():
    """构造同代码的两次上市记录，证据在当前分析截止时点可见。

    输入无参数，输出截至2025年初的旧实体别名和从该时点开始的新实体别名。
    """
    alias = bindings().aliases[0]
    return (
        replace(alias, effective_from="2019-01-01T00:00:00Z", effective_to=REUSED_AT),
        replace(alias, entity_id=NEW_ENTITY, effective_from=REUSED_AT),
    )


def scan_synthetic_row(tmp_path, *, day="2020-06-01", configured=None, cutoff=NOW):
    """用合成参考库逐行扫描，返回持久化事实、时间证据和失败原因。

    输入2020年的daily行及新旧别名，输出只属于旧实体的amount事实；不访问网络。
    """
    source = reference_fixture(tmp_path)
    configured = configured or replace(bindings(), aliases=reused_aliases())
    entry = next(item for item in load_registry() if item.api == "daily")
    entry = replace(entry, selected_fields=("amount",))
    contract = load_reference_contracts(load_policies())["daily"]
    with duckdb.connect(str(source), config={"threads": 1}) as connection:
        connection.execute("UPDATE accepted.daily SET trade_date=?", [day])
        with ProductStore(tmp_path / "products.duckdb") as store:
            entity = bindings().entities[0]
            store.put("entity", entity)
            store.put(
                "entity",
                replace(entity, identity=replace(entity.identity, lifecycle_id="listing2")),
            )
            reasons, latest = consume_rows(
                connection, store, entry, contract, configured, cutoff, "alias-regression"
            )
            return (
                store.records("typed_fact"),
                store.records("availability_fact"),
                reasons,
                latest,
            )


def test_historical_source_row_keeps_its_original_lifecycle(tmp_path):
    """场景：2020年旧行在2026年分析且代码2025年复用；预期：只归属旧实体。

    输入两次上市的同代码，输出旧实体amount=2000，观察时间和历史未认证状态不变。
    """
    facts, availability, reasons, latest = scan_synthetic_row(tmp_path)
    assert len(facts) == 1
    assert facts[0].entity_id == ENTITY
    assert facts[0].value == 2000
    assert latest == "2020-06-01"
    assert availability[0].observed_at == NOW
    assert availability[0].public_available_at is None
    assert availability[0].historical_pit_certified is False
    assert "entity_unresolved" not in reasons


def test_resolver_separates_business_time_and_preserves_legacy_calls():
    """场景：同一截止下查询历史归属和当前归属；预期：新参数选旧实体，旧调用兼容。

    输入effective_at=2020年的时间戳，输出旧实体；省略新参数输出2026年的新实体。
    """
    aliases = reused_aliases()
    assert aliases[0].visible_at(NOW) and not aliases[0].usable_at(NOW)
    assert aliases[1].usable_at(NOW)
    assert resolve_alias(aliases, "tushare", "A.SZ", NOW) == NEW_ENTITY
    assert (
        resolve_alias(aliases, "tushare", "A.SZ", NOW, effective_at="2020-06-01T00:00:00Z")
        == ENTITY
    )
    assert resolve_alias(aliases, "tushare", "A.SZ", NOW, effective_at=REUSED_AT) == NEW_ENTITY
    assert resolve_alias(aliases, "other", "A.SZ", NOW, effective_at=REUSED_AT) is None
    assert resolve_alias(aliases, "tushare", "MISSING", NOW, effective_at=REUSED_AT) is None


@pytest.mark.parametrize(
    ("observed_at", "public_available_at", "expected"),
    [
        (NOW, NOW, ENTITY),
        ("2026-10-02T00:00:00.000001Z", None, None),
        (NOW, "2026-10-02T00:00:00.000001Z", None),
        ("2026-10-01T23:59:59Z", None, ENTITY),
    ],
)
def test_historical_alias_evidence_respects_analysis_visibility(
    observed_at, public_available_at, expected
):
    """场景：业务时间相同而观察或公开时刻跨过截止；预期：截止相等可用，晚一微秒拒绝。

    输入2020年业务时间及2026年证据，输出由分析截止控制的旧实体或None。
    """
    alias = replace(
        reused_aliases()[0],
        observed_at=observed_at,
        evidence_refs=(evidence(observed_at=observed_at, public_available_at=public_available_at),),
    )
    assert (
        resolve_alias((alias,), "tushare", "A.SZ", NOW, effective_at="2020-06-01T00:00:00Z")
        == expected
    )


@pytest.mark.parametrize(
    ("business_time", "expected"),
    [
        ("2018-12-31T23:59:59.999999Z", None),
        ("2019-01-01T00:00:00Z", ENTITY),
        ("2024-12-31T23:59:59.999999Z", ENTITY),
        (REUSED_AT, None),
        ("2020", ENTITY),
        ("2020-Q2", ENTITY),
        ("2020-06", ENTITY),
        ("2020-12", ENTITY),
        ("2024-Q4", None),
        ("2024-12-31", None),
    ],
)
def test_business_time_uses_half_open_lifecycles_and_whole_periods(business_time, expected):
    """场景：精确时间边界和无市场时区的年季月日输入；预期：整个可能期间均须受覆盖。

    输入旧实体终点的精确时间返回None；2020年完整期间返回旧实体；跨界日期不猜归属。
    """
    assert (
        resolve_alias(reused_aliases()[:1], "tushare", "A.SZ", NOW, effective_at=business_time)
        == expected
    )


def test_market_day_must_be_covered_completely_and_conflicts_are_rejected():
    """场景：别名在上海市场日开始或日中切换；预期：完整日可归属，日中冲突直接拒绝。

    输入上海2025-01-01及日初启用的新别名输出新实体；附加半天冲突别名抛歧义错误。
    """
    old, new = reused_aliases()
    new = replace(new, effective_from="2024-12-31T16:00:00Z")
    old = replace(old, effective_to=new.effective_from)
    assert (
        resolve_alias(
            (old, new),
            "tushare",
            "A.SZ",
            NOW,
            effective_at="2025-01-01",
            market_timezone="Asia/Shanghai",
        )
        == NEW_ENTITY
    )
    conflicting = replace(old, effective_from="2025-01-01T00:00:00Z", effective_to=None)
    with pytest.raises(ValueError, match="ambiguous_entity_alias"):
        resolve_alias(
            (new, conflicting),
            "tushare",
            "A.SZ",
            NOW,
            effective_at="2025-01-01",
            market_timezone="Asia/Shanghai",
        )


@pytest.mark.parametrize("case", ["unknown", "unmatched", "ambiguous", "unavailable"])
def test_source_scan_fails_closed_without_a_unique_visible_business_alias(tmp_path, case):
    """场景：业务日期未知、无匹配、冲突或来源尚不可见；预期：不写实体数值事实。

    输入任一无法唯一归属的合成行，输出零typed_fact并保留明确的逐行失败原因。
    """
    configured = replace(bindings(), aliases=reused_aliases())
    cutoff = NOW
    expected_reason = "entity_unresolved"
    if case == "unknown":
        configured = replace(
            configured, time_mappings=(TimeMapping(source_api="daily", data_field=None),)
        )
    elif case == "unmatched":
        configured = replace(configured, aliases=configured.aliases[1:])
    elif case == "ambiguous":
        old = configured.aliases[0]
        configured = replace(configured, aliases=(old, replace(old, entity_id=NEW_ENTITY)))
        expected_reason = "row_adaptation_failed"
    else:
        cutoff = "2026-10-01T23:59:59Z"
        expected_reason = "row_unavailable"
    facts, availability, reasons, _ = scan_synthetic_row(
        tmp_path, configured=configured, cutoff=cutoff
    )
    assert facts == []
    assert len(availability) == 1
    assert expected_reason in reasons


def test_explicit_effective_time_resolves_a_row_without_data_date(tmp_path):
    """场景：缺业务日期但显式登记生效字段和市场时区；预期：按生效时间解析旧实体。

    输入trade_date作为生效字段且data_field为空，输出旧实体事实，data_as_of仍为空。
    """
    configured = replace(
        bindings(),
        aliases=reused_aliases(),
        time_mappings=(
            TimeMapping(
                source_api="daily",
                effective_from_field="trade_date",
                market_timezone="Asia/Shanghai",
            ),
        ),
    )
    facts, availability, _, latest = scan_synthetic_row(tmp_path, configured=configured)
    assert len(facts) == 1 and facts[0].entity_id == ENTITY
    assert availability[0].data_as_of is None and latest is None
    assert availability[0].effective_from == "2020-05-31T16:00:00Z"
