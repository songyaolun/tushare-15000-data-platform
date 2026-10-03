"""小规模多组市场上下文也必须能保存，门禁身份不能因同一API而重复。"""

from dataclasses import replace

import duckdb
import pytest
from test_research_context_consumers import INDEX, context_inputs, context_row

from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import PRODUCT_ALIASES
from tushare_entitlement_hub.equity_research.models import MARKET_ENTITY
from tushare_entitlement_hub.equity_research.storage import (
    fingerprints,
    initialize_tables,
    record_rows,
    save_results,
    stored_rows,
)
from tushare_entitlement_hub.store import encode


@pytest.mark.parametrize("api", ["index_basic", "trade_cal"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_small_market_groups_have_unique_stable_gates_and_can_be_saved(api, ambiguous):
    """两个指数或交易所组低于聚合阈值，仍可完整持久化；歧义组保留unknown且门禁不冲突。

    输入逆序或循环移位后完整result JSON字节一致；保存后每张派生表均与结果反向重建一致。
    """
    rows = tuple(
        context_row(
            api,
            {
                "ts_code": f"I{i}",
                "market": "SH",
                "publisher": "P",
                "index_type": "综合",
                "weight_rule": "市值加权",
            }
            if api == "index_basic"
            else {
                "exchange": exchange,
                "cal_date": "20260930",
                "is_open": 1,
                "pretrade_date": "20260929",
            },
            entity=INDEX if api == "index_basic" else None,
            record=f"{api}-{i}",
        )
        for i, exchange in enumerate(("SSE", "SZSE"))
    )
    if ambiguous:
        rows += (
            context_row(
                api,
                {
                    key: value
                    for key, value in rows[0].raw.items()
                    if key not in {"source_record_id"}
                },
                entity=rows[0].entity_id,
                record=f"{api}-variant",
            ),
        )
    inputs = context_inputs(*rows)
    product = PRODUCT_ALIASES["P01"]
    result = attach_consumption(inputs, product, (make_result(inputs, product, MARKET_ENTITY),))[0]
    validate_result_consumption(result)
    gates = result.envelope.quality_gates
    assert len(gates) == len({gate.gate_id for gate in gates})
    for reordered in (tuple(reversed(rows)), rows[1:] + rows[:1]):
        reordered_inputs = replace(inputs, rows_by_api={api: reordered})
        repeated = attach_consumption(
            reordered_inputs, product, (make_result(reordered_inputs, product, MARKET_ENTITY),)
        )[0]
        assert {gate.gate_id for gate in gates} == {
            gate.gate_id for gate in repeated.envelope.quality_gates
        }
        assert encode(result.to_dict()) == encode(repeated.to_dict())
    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "512MB"}) as connection:
        initialize_tables(connection)
        save_results(connection, (result,), 128)
        assert fingerprints(record_rows(result)) == fingerprints(
            stored_rows(connection, inputs.run_id, 128)
        )
