"""单位认证依赖可核验元数据与币种，不重复缩放标准层值。"""

from dataclasses import replace

import pytest
from test_research_binding_entities import entity_row

from tushare_entitlement_hub.catalog import load_policies
from tushare_entitlement_hub.equity_research.binding_entities import EntityBuild
from tushare_entitlement_hub.equity_research.binding_units import (
    UnitBuild,
    currency_evidence,
    unit_definition,
)
from tushare_entitlement_hub.reference_contract import load_reference_contracts


def unit_row(api="daily", **changes):
    """构造已标准化金额行与来源事实；默认amount=100000元，原始单位为千元。

    输入currency='CNY'可以直接证明币种；不传时需要股票主数据证据。
    """
    row, fact = entity_row(api)
    row.update(
        amount=100000,
        vol=1000,
        source_units={"amount": "千元", "vol": "手"},
        standard_units={"amount": "元", "vol": "股"},
    )
    row.update(changes)
    return row, fact


@pytest.mark.parametrize(
    ("name", "source", "standard", "expected"),
    [
        ("amount", "千元", "元", ("元", "元", "money", 1.0)),
        ("amount", "千元", None, ("千元", "元", "money", 1000.0)),
        ("vol", "手", "股", ("股", "股", "quantity", 1.0)),
        ("vol", "手", None, ("手", "股", "quantity", 100.0)),
        ("close", "元", "元", ("元", "元", "price", 1.0)),
        ("close", "元/股", None, ("元/股", "元/股", "price", 1.0)),
        ("turnover_rate", "%", "比例", ("比例", "fraction", "ratio", 1.0)),
        ("pct_chg", "%", None, ("%", "ratio", "ratio", 0.01)),
        ("volume", "张", None, ("张", "张", "quantity", 1.0)),
    ],
)
def test_unit_definition_distinguishes_raw_from_already_normalized(
    name, source, standard, expected
):
    """相同证据中标准单位存在就乘1，原始值按声明换算；比例区分fraction和ratio。

    例如daily.amount标准元不再次乘1000；政策未启用转换时，明确原始千元可认证倍数1000。
    """
    contract = load_reference_contracts(load_policies())["daily"]
    contract = replace(
        contract,
        policy=replace(
            contract.policy,
            units={name: source} if standard is not None or expected[3] == 1 else {},
        ),
    )
    row = {"source_units": {name: source}, "standard_units": {name: standard} if standard else {}}
    assert unit_definition(row, name, contract) == expected


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("source_list", "unit_metadata_invalid"),
        ("standard_list", "unit_metadata_invalid"),
        ("unknown_documentation", "unit_documentation_unknown"),
        ("policy_conflict", "unit_policy_conflict"),
        ("unknown", "unit_unknown"),
        ("missing", "unit_unknown"),
        ("unsupported_lot", "unit_unknown"),
        ("standard_conflict", "unit_standard_conflict"),
    ],
)
def test_unit_definition_rejects_unverifiable_or_conflicting_evidence(variant, expected):
    """非法元数据、文档未说明、标准层单位冲突和不支持的手数语义均拒绝。

    例如amount原始千元但standard_units写万元，返回unit_standard_conflict。
    """
    contract = load_reference_contracts(load_policies())["daily"]
    row, _ = unit_row()
    if variant == "source_list":
        row["source_units"] = []
    elif variant == "standard_list":
        row["standard_units"] = []
    elif variant == "unknown_documentation":
        contract = replace(contract, policy=replace(contract.policy, units={"amount": "未说明"}))
    elif variant == "policy_conflict":
        row["source_units"]["amount"] = "万元"
    elif variant in {"unknown", "missing", "unsupported_lot"}:
        contract = replace(contract, policy=replace(contract.policy, api_name="cb_daily", units={}))
        row["source_units"] = (
            {}
            if variant == "missing"
            else {"amount": "手" if variant == "unsupported_lot" else "未知单位"}
        )
        if variant == "missing":
            row["standard_units"] = {}
    elif variant == "standard_conflict":
        row["standard_units"]["amount"] = "万元"
    with pytest.raises(ValueError, match=expected):
        unit_definition(row, "amount", contract)


@pytest.mark.parametrize(
    ("currency", "curr_type", "expected"),
    [
        ("CNY", None, "CNY"),
        (None, "人民币", "CNY"),
        ("CNY", "人民币", "CNY"),
        ("USD", "CNY", None),
        (None, None, None),
    ],
)
def test_currency_requires_explicit_consistent_cny_evidence(currency, curr_type, expected):
    """币种只能来自明确CNY或人民币，两字段冲突拒绝；金额单位元本身不作为币种证明。

    例如currency=USD且curr_type=CNY返回(None,())，不发布人民币证书。
    """
    row, fact = unit_row(currency=currency, curr_type=curr_type)
    result, evidence = currency_evidence(row, fact, EntityBuild(5))
    assert result == expected
    assert bool(evidence) == (expected is not None)


@pytest.mark.parametrize("api", ["daily", "cb_daily", "index_daily"])
def test_currency_master_evidence_does_not_cross_namespace(api):
    """只有股票命名空间消费主数据币种，指数与转债同码不得误用股票币种。

    输入stock_basic.curr_type=CNY及同码daily得到CNY，cb_daily/index_daily无本行声明则未知。
    """
    entities = EntityBuild(5)
    entities.add(*entity_row())
    entities.materialize()
    row, fact = unit_row(api, curr_type=None)
    currency, evidence = currency_evidence(row, fact, entities)
    assert currency == ("CNY" if api == "daily" else None)
    assert len(evidence) == (1 if api == "daily" else 0)


def test_unit_build_keeps_smallest_evidence_and_missing_value_does_not_revoke():
    """一致单位采用确定性最小来源证据，空值仍记缺口但不撤销其他有效行认证。

    输入来源z、a、z和amount=None，输出单个来自a的amount证书且乘数1。
    """
    build = UnitBuild()
    entities = EntityBuild(5)
    contract = load_reference_contracts(load_policies())["daily"]
    for identity in ["z", "a", "z"]:
        row, fact = unit_row(currency="CNY", curr_type=None)
        row["source_record_id"] = identity
        fact = replace(fact, source_record_id=identity)
        build.add(row, fact, contract, {"amount"}, entities)
    row, fact = unit_row(amount=None)
    build.add(row, fact, contract, {"amount"}, entities)
    certificates = build.accepted()
    assert len(certificates) == 1
    assert certificates[0].evidence_refs[0].source_record_id == "a"
    assert certificates[0].multiplier == 1
    assert build.gaps["amount"]["unit_value_missing"] == 1


@pytest.mark.parametrize(
    ("variant", "reason"),
    [
        ("currency_missing", "currency_unverified"),
        ("cross_row", "unit_cross_row_conflict"),
        ("identity_mismatch", "unit_evidence_invalid"),
        ("metadata", "unit_metadata_invalid"),
    ],
)
def test_unit_build_any_provenance_failure_revokes_whole_field(variant, reason):
    """任何有值行的单位、币种或来源身份失败会撤销整列认证，不只跳过坏行。

    例如首行已标准化为元、次行仍是千元，输出unit_cross_row_conflict且不返回amount证书。
    """
    build = UnitBuild()
    entities = EntityBuild(5)
    contract = load_reference_contracts(load_policies())["daily"]
    if variant == "cross_row":
        contract = replace(contract, policy=replace(contract.policy, units={}))
    row, fact = unit_row(currency="CNY", curr_type=None)
    build.add(row, fact, contract, {"amount"}, entities)
    row, fact = unit_row(currency="CNY", curr_type=None)
    if variant == "currency_missing":
        row["currency"] = None
    elif variant == "cross_row":
        row["standard_units"] = {}
    elif variant == "identity_mismatch":
        row["source_record_id"] = "different"
    else:
        row["source_units"] = []
    build.add(row, fact, contract, {"amount"}, entities)
    assert not build.accepted()
    assert build.gaps["amount"][reason] == 1


def test_quantity_certificate_does_not_require_currency():
    """数量维度不需要币种；输入已规范化vol=1000股，输出currency=None、multiplier=1。"""
    row, fact = unit_row(curr_type=None)
    build = UnitBuild()
    build.add(
        row, fact, load_reference_contracts(load_policies())["daily"], {"vol"}, EntityBuild(5)
    )
    certificate = build.accepted()[0]
    assert certificate.currency is None
    assert certificate.dimension == "quantity"
    assert certificate.multiplier == 1


def test_explicit_standard_unit_can_certify_without_source_policy_unit():
    """当政策未声明单位而标准层明确给出单位时，接受该证据且不二次缩放。

    输入无source_units的amount及standard_units.amount=元，输出元到元、money、乘数1。
    """
    contract = load_reference_contracts(load_policies())["daily"]
    contract = replace(contract, policy=replace(contract.policy, units={}))
    assert unit_definition({"standard_units": {"amount": "元"}}, "amount", contract) == (
        "元",
        "元",
        "money",
        1.0,
    )


@pytest.mark.parametrize(
    "name", ["convert_price", "cb_value", "cost_5pct", "cost_50pct", "cost_95pct"]
)
def test_documented_quote_fields_are_price_not_money(name):
    """转股价值和筹码价属于报价维度，不因元单位而误作金额。

    输入cb_value或cost_50pct的元证据，输出price维度、元单位及乘数1。
    """
    contract = load_reference_contracts(load_policies())["daily"]
    contract = replace(contract, policy=replace(contract.policy, units={name: "元"}))
    assert unit_definition({"standard_units": {name: "元"}}, name, contract) == (
        "元",
        "元",
        "price",
        1.0,
    )


@pytest.mark.parametrize("policy_kind", ["scaled", "identity", "absent", "unknown"])
@pytest.mark.parametrize("has_source", [False, True])
@pytest.mark.parametrize("has_standard", [False, True])
def test_unit_metadata_matrix_preserves_normalization_proof(policy_kind, has_source, has_standard):
    """遍历来源与标准单位有无以及政策转换类型，缺少标准化证据时防止二次缩放。

    政策千元转换已启用但标准单位缺失时拒绝；恒等政策元可接受；无政策且明确原始千元
    可按1000转换；仅标准元可接受；完全无单位或文档未知始终拒绝。
    """
    contract = load_reference_contracts(load_policies())["daily"]
    policy_unit = {"scaled": "千元", "identity": "元", "unknown": "文档未说明"}.get(policy_kind)
    contract = replace(
        contract,
        policy=replace(contract.policy, units={"amount": policy_unit} if policy_unit else {}),
    )
    source_unit = "元" if policy_kind == "identity" else "千元"
    row = {
        "source_units": {"amount": source_unit} if has_source else {},
        "standard_units": {"amount": "元"} if has_standard else {},
    }
    error = None
    if policy_kind == "unknown":
        error = "unit_documentation_unknown"
    elif policy_kind == "scaled" and not has_standard:
        error = "unit_standard_missing"
    elif policy_kind == "absent" and not has_source and not has_standard:
        error = "unit_unknown"
    if error:
        with pytest.raises(ValueError, match=error):
            unit_definition(row, "amount", contract)
    else:
        expected_source = "元" if has_standard or policy_kind == "identity" else "千元"
        multiplier = 1.0 if expected_source == "元" else 1000.0
        assert unit_definition(row, "amount", contract) == (
            expected_source,
            "元",
            "money",
            multiplier,
        )
