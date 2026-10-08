"""价格门禁的离线回归；合成行情验证目标字段，不使用网络或真实凭证。"""

from typing import Any

import pytest

from tushare_entitlement_hub.product_foundation.contracts import EvidenceRef
from tushare_entitlement_hub.product_foundation.numeric import (
    UnitCertification,
    adapt_numeric,
    price_gate,
)
from tushare_entitlement_hub.product_foundation.temporal import AvailabilityFact

OBSERVED_AT = "2026-10-02T00:00:00Z"
ENTITY = "cn:future:TEST.SHF:contract1"
PRICE_PURPOSES = ("return", "volatility", "price_rank", "feasibility", "premium")
NON_TRADED_FIELDS = (
    "settle",
    "pre_settle",
    "settlement",
    "settlement_price",
    "reference",
    "reference_close",
    "reference_price",
    "indicative_price",
    "unknown_price",
)


def price_inputs(
    field: str, value: Any
) -> tuple[dict[str, Any], AvailabilityFact, UnitCertification]:
    """构造字段、单位与来源身份一致的合成期货行情和认证输入。

    输入 field='settle'、value=11，输出 close=10、settle=11 且量额为正的行，
    以及绑定 settle 的时间事实和 CNY 价格证书；行内不添加 price_type。
    """
    evidence = EvidenceRef(
        source_api="fut_daily",
        source_record_id="price-row",
        business_group_key="price-group",
        business_variant_key=None,
        contract_hash="price-contract",
        fields=(field,),
        observed_at=OBSERVED_AT,
        locator="fixture:price-row",
    )
    availability = AvailabilityFact(
        source_api="fut_daily",
        source_record_id="price-row",
        certification_version="v1",
        observed_at=OBSERVED_AT,
        data_as_of="2026-09-30",
    )
    certification = UnitCertification(
        version="v1",
        source_api="fut_daily",
        field=field,
        source_unit="元",
        target_unit="元",
        dimension="price",
        currency="CNY",
        multiplier=1,
        evidence_refs=(evidence,),
    )
    row = {
        "source_api": "fut_daily",
        "source_record_id": "price-row",
        "business_group_key": "price-group",
        "contract_hash": "price-contract",
        "quality_tier": "verified",
        "standard_units": {field: "元"},
        "close": 10,
        "vol": 2,
        "amount": 20,
        field: value,
    }
    return row, availability, certification


@pytest.mark.parametrize("field", NON_TRADED_FIELDS)
def test_adaptation_rejects_settlement_and_reference_fields(field: str) -> None:
    """场景：正收盘价与量额掩盖结算、参考或未知字段；预期：目标价格返回空值。

    输入 close=10、settle=11 且无 price_type，输出 value=None 和 no_traded_price。
    """
    # 每个非成交字段都保留正 close 和量额，排除无成交等其他拒绝条件。
    row, availability, certification = price_inputs(field, 11)
    fact = adapt_numeric(row, field, ENTITY, availability, certification)

    assert fact.value is None
    assert fact.reason_codes == ("no_traded_price",)


@pytest.mark.parametrize(
    "value", [0, -1, None, float("nan"), float("inf"), -float("inf"), True, "11"]
)
def test_adaptation_validates_the_target_price(value: Any) -> None:
    """场景：close 和量额正常，目标 open 无效；预期：价格事实不可用。

    输入 open=0、close=10，输出 value=None；非有限、缺失或非数值 open 同样被门禁拒绝。
    """
    # 旧实现只检查 close，零值和负值 open 会被错误认证为可用价格。
    row, availability, certification = price_inputs("open", value)
    fact = adapt_numeric(row, "open", ENTITY, availability, certification)

    assert fact.value is None
    assert "no_traded_price" in fact.reason_codes


@pytest.mark.parametrize("field", ["open", "high", "low", "close"])
def test_adaptation_accepts_a_positive_target_price(field: str) -> None:
    """场景：目标成交价格及量额有效；预期：保留目标字段的认证数值。

    输入 open=11、close=10，输出 open 的 11 元价格事实，原因码为空。
    """
    # 分别覆盖常用成交字段，确保新增类型门禁保留正常行情。
    row, availability, certification = price_inputs(field, 11)
    fact = adapt_numeric(row, field, ENTITY, availability, certification)

    assert fact.field == field
    assert fact.value == 11
    assert fact.reason_codes == ()


@pytest.mark.parametrize("purpose", PRICE_PURPOSES)
def test_price_gate_preserves_the_default_close_contract(purpose: str) -> None:
    """场景：调用者不传目标字段；预期：仍按 close 和量额决定五类价格用途。

    输入 close=10、vol=2、amount=20，输出通过；将 close 改为 0 后输出不通过。
    """
    # 沿用原有两参数调用，证明可选参数不会破坏默认 close 行为。
    row = {"close": 10, "vol": 2, "amount": 20}

    assert price_gate(row, purpose).passed
    assert not price_gate(row | {"close": 0}, purpose).passed


@pytest.mark.parametrize(
    "price_type", ["settlement", "reference", "reference_close", "reference_price"]
)
def test_price_gate_rejects_non_traded_price_markers(price_type: str) -> None:
    """场景：close 数值有效但被标为结算或参考价格；预期：拒绝成交价格资格。

    输入 price_type='reference'、close=10 及正量额，输出 not_traded_price_type。
    """
    # 标记独立于字段名生效，覆盖提供方将参考数值写入 close 的情况。
    result = price_gate(
        {"close": 10, "vol": 2, "amount": 20, "price_type": price_type}, "return"
    )

    assert not result.passed
    assert result.reason_codes == ("not_traded_price_type",)


@pytest.mark.parametrize("field", NON_TRADED_FIELDS)
@pytest.mark.parametrize("purpose", PRICE_PURPOSES)
def test_price_gate_uses_field_semantics_for_every_purpose(field: str, purpose: str) -> None:
    """场景：指定结算、参考或未知字段用于任一价格用途；预期：统一拒绝。

    输入 field='settle'、purpose='premium' 且价格量额有效，输出 not_traded_price_type。
    """
    # 全用途复用同一字段语义，避免只在收益适配路径修复而留出其他入口。
    row, _, _ = price_inputs(field, 11)
    result = price_gate(row, purpose, field=field)

    assert not result.passed
    assert result.reason_codes == ("not_traded_price_type",)


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (0, "no_traded_price"),
        (-1, "no_traded_price"),
        (float("nan"), "price_activity_unknown"),
        (float("inf"), "price_activity_unknown"),
        (None, "price_activity_unknown"),
        (True, "price_activity_unknown"),
        ("11", "price_activity_unknown"),
    ],
)
def test_price_gate_checks_target_value_directly(value: Any, reason: str) -> None:
    """场景：直接调用门禁时目标 open 无效；预期：按目标值给出稳定原因。

    输入 open=0、close=10，输出 no_traded_price；输入 open=NaN，输出 price_activity_unknown。
    """
    # close 保持合法，确保拒绝来自显式目标的有限性、数值类型或正值检查。
    result = price_gate({"close": 10, "open": value, "vol": 2, "amount": 20}, "return", "open")

    assert not result.passed
    assert result.reason_codes == (reason,)


def test_price_gate_accepts_the_target_without_an_unrelated_close() -> None:
    """场景：显式目标 open 与量额有效，但行内缺少 close；预期：按目标值通过。

    输入 open=11、vol=2、amount=20 及 field='open'，输出通过且原因码为空。
    """
    # 目标字段替代默认 close，避免门禁仍隐式依赖无关价格字段。
    result = price_gate({"open": 11, "vol": 2, "amount": 20}, "return", field="open")

    assert result.passed
    assert result.reason_codes == ()
