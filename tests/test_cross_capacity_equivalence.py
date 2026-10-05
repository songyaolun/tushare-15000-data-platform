"""磁盘分区与完整小样本计算的字节等价性及跨实体业务关系回归。"""

from dataclasses import replace

import pytest
from test_cross_expectations import disclosure
from test_cross_funds import full_fund_rows
from test_cross_market_products import (
    A as MARKET_A,
)
from test_cross_market_products import (
    B as MARKET_B,
)
from test_cross_market_products import (
    C as MARKET_C,
)
from test_cross_market_products import (
    commodity_sources,
    financing_sources,
    international_sources,
    relation,
    source,
    value,
)
from test_cross_market_products import inputs as market_inputs
from test_cross_portfolio import explicit_holdings, fund_holding, portfolio_inputs
from test_cross_themes import theme_rows
from test_research_governance import A, B, inputs, row

from tushare_entitlement_hub.cross_asset_research.partitioned_execution import (
    ManagerCounts,
    partitioned_results,
)
from tushare_entitlement_hub.cross_asset_research.partitioned_inputs import InputSpool
from tushare_entitlement_hub.cross_asset_research.registry import HANDLERS, resolve_handler
from tushare_entitlement_hub.equity_research.adapter import SOURCE_APIS
from tushare_entitlement_hub.equity_research.models import ResearchInputs, ResearchResult


def partitioned_sample(
    product: str,
    sample: ResearchInputs,
    *,
    parents: tuple[ResearchResult, ...] = (),
    holdings: tuple[dict, ...] = (),
    reverse_append: bool = False,
) -> tuple[ResearchResult, ...]:
    """只在磁盘保留新来源，完整价格旁证仍在旧输入中，运行正式分区分派。

    例如 P09 的旧主营来源显式放入支持暂存，daily/index_daily 等全部旧输入仍保留全局范围。
    """
    template = replace(
        sample,
        rows_by_api={
            api: records for api, records in sample.rows_by_api.items() if api in SOURCE_APIS
        },
    )
    handler = resolve_handler(HANDLERS[product])
    with InputSpool() as spool:
        for api, records in sample.rows_by_api.items():
            if api == "daily":
                continue
            for record in reversed(records) if reverse_append else records:
                spool.append(record)
        return tuple(
            partitioned_results(
                product, spool, template, handler, parents=parents, holdings=holdings
            )
        )


def assert_same_results(actual, expected) -> None:
    """比较完整结果集合的规范 JSON，包含所有指标、原因、证据和来源摘要。

    例如输出顺序变化不影响比较，缺失、重复或任一字段变化都会使断言失败。
    """
    assert sorted(result.to_json() for result in actual) == sorted(
        result.to_json() for result in expected
    )


def canonical_inputs(sample: ResearchInputs) -> ResearchInputs:
    """按原适配器的实体和来源记录键顺序建立完整计算基准，不修改任何行内容。

    例如余额表字段的直接夹具顺序不同，先还原正常适配排序再比较完整来源摘要。
    """
    return replace(
        sample,
        rows_by_api={
            api: tuple(
                sorted(
                    records,
                    key=lambda record: (
                        str(record.entity_id or ""),
                        str(record.raw["source_record_id"]),
                    ),
                )
            )
            for api, records in sample.rows_by_api.items()
        },
    )


@pytest.mark.parametrize("reverse_append", [False, True], ids=["forward", "reverse"])
@pytest.mark.parametrize(
    "product,metric_name,expected_value",
    [
        ("P02", "price_flow_confirmation", 1),
        ("P05", "forecast_profit_midpoint", 110),
        ("P08", "share_change", 10),
        ("P09", "segment_gross_margin", 0.4),
        ("P10", "ten_year_breakeven_proxy", 0),
        ("P11", "net_financing_balance_ratio", 0.1),
    ],
)
def test_partitioned_business_results_equal_complete_in_memory_results(
    product: str, metric_name: str, expected_value: float, reverse_append: bool
) -> None:
    """场景：使用各产品已有的完整合成样本；预期：分区结果与未分区结果逐字节一致。

    例如 P09 分部毛利率保持 0.4，P10 同日利差保持 0，且所有来源证据不变。
    """
    if product == "P02":
        sample = inputs(*theme_rows())
    elif product == "P05":
        sample = inputs(disclosure(), disclosure("express", "actual", day="2026-09-30"))
    elif product == "P08":
        sample = inputs(*full_fund_rows(), entities=(A, B))
    else:
        factory = {
            "P09": commodity_sources,
            "P10": international_sources,
            "P11": financing_sources,
        }[product]
        sample = market_inputs(*factory())
    sample = canonical_inputs(sample)
    handler = resolve_handler(HANDLERS[product])
    expected = tuple(handler(sample))
    actual = partitioned_sample(product, sample, reverse_append=reverse_append)
    assert expected
    assert_same_results(actual, expected)
    assert any(result.metric(metric_name) == pytest.approx(expected_value) for result in actual)


def test_two_funds_keep_global_manager_count_and_explicit_index_sources() -> None:
    """场景：两只 ETF 指向同一法律管理人和同一基准；预期：每只基金的全观察计数仍为 2。

    例如拆成两个实体分区后，same_manager_observed_etfs 不能错误变成 1。
    """
    extra = row(
        "etf_basic",
        "second-etf",
        entity=B,
        raw={"ts_code": "ETF2", "mgr_name": "管理人", "index_code": "I"},
    )
    sample = inputs(*full_fund_rows(), extra, entities=(A, B))
    actual = partitioned_sample("P08", sample)
    expected = tuple(resolve_handler(HANDLERS["P08"])(sample))
    assert_same_results(actual, expected)
    assert len(actual) == 2
    assert all(result.metric("same_manager_observed_etfs") == 2 for result in actual)
    assert all(
        any(ref.source_api == "fund_company" for ref in result.envelope.evidence_refs)
        for result in actual
    )


def test_manager_index_counts_only_visible_named_fund_snapshots() -> None:
    """场景：两个同管理人档案、缺名档案和未来档案；预期：只计入两条可见且有名的档案。

    例如 mapping['管理人'] 返回 2，缺失键抛 KeyError，迭代和长度均只包含该管理人。
    """
    records = [
        row("etf_basic", "first", entity=A, raw={"mgr_name": "管理人"}),
        row("etf_basic", "second", entity=B, raw={"mgr_name": "管理人"}),
        row("etf_basic", "unnamed", entity="cn:security:C:life"),
        row("etf_basic", "future", entity="cn:security:D:life", raw={"mgr_name": "未来管理人"}),
    ]
    records[-1] = replace(
        records[-1],
        availability=replace(records[-1].availability, observed_at="2026-10-04T12:00:00Z"),
    )
    with InputSpool() as spool:
        for record in records:
            spool.append(record)
        counts = ManagerCounts(spool, inputs())
        assert counts["管理人"] == 2
        assert len(counts) == 1
        assert list(counts) == ["管理人"]
        with pytest.raises(KeyError):
            _ = counts["missing"]


@pytest.mark.parametrize("mode", ["explicit", "disclosed", "empty"])
def test_portfolio_partitioning_preserves_holdings_and_empty_input_contract(mode: str) -> None:
    """场景：显式组合、基金披露组合或无持仓；预期：完整组合与未知原因均和原处理器一致。

    例如显式 60%/40% 持仓不受披露分区影响，无持仓仍输出 input_required。
    """
    sample = portfolio_inputs(
        *(
            tuple([fund_holding(), fund_holding("second", symbol="S2", weight=0.4)])
            if mode == "disclosed"
            else ()
        )
    )
    holdings = explicit_holdings() if mode == "explicit" else ()
    handler = resolve_handler(HANDLERS["P12"])
    expected = tuple(handler(sample, holdings=holdings))
    actual = partitioned_sample("P12", sample, holdings=holdings)
    assert_same_results(actual, expected)
    assert actual


def test_option_contract_group_keeps_full_call_and_put_quantities() -> None:
    """场景：同标的同到期的看涨和看跌合约及额外无行情目录；预期：PCR 完整且目录均输出。

    例如看涨成交 10、看跌成交 20 得 PCR=2，额外 40 个无行情合约仍有各自业务结果。
    """
    common = {
        "maturity_date": "2026-12-31",
        "exercise_type": "E",
        "opt_type": "ETF",
        "opt_multiplier": "100",
        "opt_code": "OPT",
    }
    records = [
        source("opt_basic", "call-spec", entity=MARKET_A, raw={**common, "call_put": "C"}),
        source("opt_basic", "put-spec", entity=MARKET_B, raw={**common, "call_put": "P"}),
        source(
            "opt_daily",
            "call-quote",
            entity=MARKET_A,
            values={"vol": value(10, "quantity", "手"), "oi": value(30, "quantity", "手")},
        ),
        source(
            "opt_daily",
            "put-quote",
            entity=MARKET_B,
            values={"vol": value(20, "quantity", "手"), "oi": value(60, "quantity", "手")},
        ),
    ]
    records.extend(
        source(
            "opt_basic",
            f"unused-{index}",
            entity=f"test:security:OPT{index}:life",
            raw={**common, "call_put": "C"},
        )
        for index in range(40)
    )
    sample = market_inputs(
        *records,
        relations=(
            relation("option_underlying", MARKET_A, MARKET_C),
            relation("option_underlying", MARKET_B, MARKET_C),
        ),
    )
    actual = partitioned_sample("P11", sample)
    expected = tuple(resolve_handler(HANDLERS["P11"])(sample))
    assert_same_results(actual, expected)
    assert sum(result.details["source_api"] == "opt_basic" for result in actual) == 42
    ratios = [result for result in actual if result.metric("put_call_vol_ratio") is not None]
    assert len(ratios) == 1
    assert ratios[0].metric("put_call_vol_ratio") == ratios[0].metric("put_call_oi_ratio") == 2


def test_same_day_nominal_and_real_yields_keep_complete_breakeven_evidence() -> None:
    """场景：同日名义与实际国债曲线分别落在独立实体；预期：同日利差和两源证据完整。

    例如十年名义 3%、实际 1% 得 0.02，分区结果与未分区完整结果一致。
    """
    sample = market_inputs(
        source(
            "us_tycr",
            "nominal",
            entity=MARKET_A,
            values={name: value(0.03) for name in ("y5", "y7", "y10", "y20", "y30")},
        ),
        source(
            "us_trycr",
            "real",
            entity=MARKET_B,
            values={name: value(0.01) for name in ("y5", "y7", "y10", "y20", "y30")},
        ),
    )
    actual = partitioned_sample("P10", sample)
    expected = tuple(resolve_handler(HANDLERS["P10"])(sample))
    assert_same_results(actual, expected)
    result = next(
        result for result in actual if result.metric("ten_year_breakeven_proxy") is not None
    )
    assert result.metric("ten_year_breakeven_proxy") == pytest.approx(0.02)
    assert {ref.source_api for ref in result.envelope.evidence_refs} == {"us_tycr", "us_trycr"}
