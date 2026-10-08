"""消费状态必须有真实来源依据，布尔值与缺口原因不能依赖宽松JSON比较。"""

from dataclasses import replace

import pytest
from test_research_consumption import context_rows, registration
from test_research_governance import A, row

from tushare_entitlement_hub.equity_research import consumption
from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    consumption_reasons,
    execute_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.store import encode


@pytest.mark.parametrize(
    "value,status",
    [
        ("missing", "unknown"),
        ([True], "unknown"),
        ([0], "conflict"),
        ([" "], "unknown"),
        ([], "unknown"),
        ([], "conflict"),
        ([], "not_enabled_due_to_history"),
    ],
)
def test_reason_codes_reject_scalar_wrong_elements_and_unexplained_states(value, status):
    """场景：原因使用标量、非文本或未知状态没有原因；预期：固定原因码拒绝，不拆成字符。"""
    with pytest.raises(ValueError, match="reason_codes_invalid"):
        consumption_reasons(value, status)
    assert consumption_reasons(("history_missing",), "not_enabled_due_to_history") == (
        "history_missing",
    )
    assert consumption_reasons([], "observed") == ()


@pytest.mark.parametrize("scenario", ["empty", "foreign", "absent_field", "valid"])
def test_history_state_requires_retained_own_source_business_field(monkeypatch, scenario):
    """场景：历史受限状态只给跨源引用、空行或不存在字段；预期：仅真实本来源字段可通过。"""
    context = context_rows({"daily": (row("daily", raw={"close": 10}),)})
    current = context.rows("daily")[0]
    references = current.evidence("close")
    if scenario == "empty":
        context = replace(context, rows_by_api={})
    elif scenario == "foreign":
        references = row("weekly", raw={"close": 10}).evidence("close")
    elif scenario == "absent_field":
        references = (replace(references[0], fields=("pre_close",)),)

    def handler(inputs, product, api, rows, entity_id):
        """模拟仅有历史窗口不足的处理器；例：输入daily返回history状态及参数化引用。"""
        return {
            "status": "not_enabled_due_to_history",
            "reason_codes": ("history_missing",),
            "facts": {"historical_window_complete": False},
            "evidence_refs": references,
            "gates": (),
        }

    def resolver(name):
        """替换执行器解析器以测试包装契约；例：输入core返回上面的确定性处理器。"""
        return handler

    monkeypatch.setattr(consumption, "resolve_consumer", resolver)
    if scenario == "valid":
        result, _, _ = execute_consumption(context, registration("daily"), A)
        assert result["status"] == "not_enabled_due_to_history"
        assert result["retained_row_count"] == 1
    else:
        with pytest.raises(ValueError, match="observation_without_evidence"):
            execute_consumption(context, registration("daily"), A)


@pytest.mark.parametrize(
    "field,value",
    [
        ("executed", 1),
        ("historical_pit_certified", 0),
        ("reason_codes", "missing"),
        ("reason_codes", [True]),
        ("reason_codes", [" "]),
    ],
)
def test_persisted_boolean_and_reason_types_are_strict(field, value):
    """场景：JSON把true/false改成1/0或损坏原因类型；预期：不能利用Python相等比较绕过。"""
    context = context_rows({"daily": (row("daily", raw={"close": 10}),)})
    product = PRODUCT_ALIASES["P03"]
    attached = attach_consumption(context, product, (make_result(context, product, A),))[0]
    details = attached.details
    record = next(item for item in details["consumption"] if item["api"] == "daily")
    record[field] = value
    with pytest.raises(ValueError, match="research_consumption"):
        validate_result_consumption(replace(attached, details_json=encode(details)))


@pytest.mark.parametrize("scenario", ["foreign_reference", "missing_rows", "missing_reason"])
def test_persisted_history_requires_own_evidence_rows_and_explanation(scenario):
    """场景：同步改写历史受限记录却缺本来源引用、行数或原因；预期：重建校验逐项拒绝。"""
    context = context_rows(
        {
            "daily": (row("daily", raw={"close": 10}),),
            "daily_basic": (row("daily_basic", raw={"pe_ttm": 10}),),
        }
    )
    product = PRODUCT_ALIASES["P03"]
    attached = attach_consumption(context, product, (make_result(context, product, A),))[0]
    details = attached.details
    record = next(item for item in details["consumption"] if item["api"] == "daily")
    record.update(status="not_enabled_due_to_history", reason_codes=["history_missing"])
    if scenario == "foreign_reference":
        other = next(item for item in details["consumption"] if item["api"] == "daily_basic")
        record["evidence_refs"] = other["evidence_refs"]
    elif scenario == "missing_rows":
        record["retained_row_count"] = 0
        for values in record["field_quality"].values():
            values.update({key: 0 for key in values})
    else:
        record["reason_codes"] = []
    with pytest.raises(ValueError, match="research_consumption"):
        validate_result_consumption(replace(attached, details_json=encode(details)))
