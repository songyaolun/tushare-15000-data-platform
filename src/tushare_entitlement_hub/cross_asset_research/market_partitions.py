"""市场状态逐实体计算，联合指标按原有明确关系构造完整磁盘分区。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

from ..equity_research.models import Metric, ResearchInputs, ResearchResult, ResearchRow
from ..store import encode
from . import commodities, financing, international
from .market_support import Frame, finish, frames
from .partitioned_inputs import InputSpool, PartitionInputs

Processor = Callable[[Frame], tuple[list[Metric], dict[str, Any]]]
Membership = Callable[[ResearchInputs, ResearchRow, Frame], tuple[str, tuple[str, ...]] | None]


def source_results(
    spool: InputSpool,
    template: ResearchInputs,
    product: str,
    fields: dict[str, tuple[str, ...]],
    dimensions: dict[str, tuple[str, ...]],
    processor: Processor,
) -> Iterator[ResearchResult]:
    """每次读取一个来源实体，保留原细分维度、Frame 门禁与指标处理器。

    输入十万合约目录逐合约产生结果；未处理来源不会因其他来源字节数增加而被清空。
    """
    for api, names in fields.items():
        for entity in spool.entities((api,)):
            part = spool.materialize(template, (api,), entity=entity)
            for frame in frames(part, api, names, dimensions.get(api, ())):
                metrics, details = processor(frame)
                yield finish(part, product, frame, metrics, details)


def future_membership(
    inputs: ResearchInputs,
    specification: ResearchRow,
    quote: Frame,
) -> tuple[str, tuple[str, ...]] | None:
    """适配共享期货分组规则，参数中保留统一上下文但不新增关系推断。

    输入明确交割月和完整规格时返回 commodities.curve_member 的原样结果。
    """
    return commodities.curve_member(specification, quote)


def contract_groups(
    spool: InputSpool,
    template: ResearchInputs,
    specification_api: str,
    quote_api: str,
    fields: dict[str, tuple[str, ...]],
    membership: Membership,
) -> Iterator[PartitionInputs]:
    """只为有行情且规格唯一合格的合约建立磁盘联合键，再读完整组。

    输入十万目录、百份行情时只建立百份候选关系；PCR 或期限曲线的完整分组键
    直接复用业务函数，因此紧凑日期、冲突、乘数或规格不会被不同分区规则改写。
    """
    spool.connection.execute(
        "CREATE TABLE IF NOT EXISTS contract_members(kind TEXT,scope TEXT,entity TEXT,"
        "PRIMARY KEY(kind,scope,entity)) WITHOUT ROWID"
    )
    spool.connection.execute("DELETE FROM contract_members WHERE kind=?", (specification_api,))
    for entity in spool.entities((quote_api,)):
        part = spool.materialize(template, (specification_api, quote_api), entity=entity)
        specifications = {
            frame.entity_id: frame.rows[0]
            for frame in frames(part, specification_api, fields[specification_api])
            if len(frame.rows) == 1 and not frame.reasons
        }
        for quote in frames(part, quote_api, fields[quote_api]):
            if quote.reasons or len(quote.rows) != 1 or quote.entity_id not in specifications:
                continue
            member = membership(part, specifications[quote.entity_id], quote)
            if member is not None:
                spool.connection.execute(
                    "INSERT OR IGNORE INTO contract_members VALUES (?,?,?)",
                    (specification_api, encode(member[1]), entity),
                )
    cursor = spool.connection.execute(
        "SELECT scope FROM contract_members WHERE kind=? GROUP BY scope ORDER BY scope",
        (specification_api,),
    )
    for (scope,) in cursor:
        selectors = (
            (api, entity, None)
            for (entity,) in spool.connection.execute(
                "SELECT entity FROM contract_members WHERE kind=? AND scope=? ORDER BY entity",
                (specification_api, scope),
            )
            for api in (specification_api, quote_api)
        )
        yield spool.materialize_selection(template, selectors)


def financing_results(spool: InputSpool, template: ResearchInputs) -> Iterator[ResearchResult]:
    """融资状态逐来源实体产出，PCR 完整分组从磁盘读取并复用原计算。

    输入同标的、同到期的看涨和看跌合约时合并为同一完整组；无标的关系只保留本体研究。
    """
    yield from source_results(
        spool,
        template,
        financing.PRODUCT_ID,
        financing.API_FIELDS,
        financing.DIMENSIONS,
        financing._process,
    )
    for part in contract_groups(
        spool,
        template,
        "opt_basic",
        "opt_daily",
        financing.API_FIELDS,
        financing.option_member,
    ):
        yield from financing._option_groups(part)


def commodity_checks(spool: InputSpool, template: ResearchInputs) -> Iterator[ResearchResult]:
    """按各比较的真实关联读完整候选，保留歧义而不任意选择一条映射。

    输入连续代码 A 显式映射实际代码 B 时读取所有 B 的候选实体；有两候选则仍为 unknown。
    普通与 VIP 分部使用同一企业完整组，黄金现货仅按已确认关系读取期货。
    """
    for left, right in (("fut_settle", "fut_daily"), ("fina_mainbz_vip", "fina_mainbz")):
        for entity in spool.entities((left,)):
            part = spool.materialize(template, (left, right), entity=entity)
            yield from commodities._cross_checks(part)
    for entity in spool.entities(("fut_daily_adj",)):
        mapping = spool.materialize(template, ("fut_mapping",), entity=entity)
        codes = sorted(
            {
                code
                for row in mapping.rows("fut_mapping")
                if (code := row.text("mapping_ts_code")) is not None
            }
        )
        selectors = [
            ("fut_daily_adj", entity, None),
            ("fut_mapping", entity, None),
            *(("fut_daily", None, code) for code in codes),
        ]
        part = spool.materialize_selection(template, selectors)
        yield from commodities._cross_checks(part)
    for entity in spool.entities(("sge_daily",)):
        relations = [
            relation
            for relation in template.config.relations_by_source.get(
                (entity, "spot_futures_comparable"), ()
            )
            if relation.confirmed and relation.usable_at(template.analysis_as_of)
        ]
        if len(relations) != 1:
            continue
        part = spool.materialize_selection(
            template,
            (
                ("sge_daily", entity, None),
                ("fut_daily", relations[0].to_entity_id, None),
            ),
        )
        yield from commodities._cross_checks(part)


def commodity_results(spool: InputSpool, template: ResearchInputs) -> Iterator[ResearchResult]:
    """商品目录、行情、期限结构和企业证据按原业务关系分区，避免整产品内存汇总。

    输入多个品种，每条期限曲线仍包含该交易所和规格的全部合格合约，而非分区样本。
    """
    yield from source_results(
        spool,
        template,
        commodities.PRODUCT_ID,
        commodities.API_FIELDS,
        commodities.DIMENSIONS,
        commodities._process,
    )
    for part in contract_groups(
        spool,
        template,
        "fut_basic",
        "fut_daily",
        commodities.API_FIELDS,
        future_membership,
    ):
        yield from commodities._curves(part)
    yield from commodity_checks(spool, template)
    for entity in spool.entities(("fina_mainbz_vip",)):
        part = spool.materialize(template, ("fina_mainbz_vip",), entity=entity)
        yield from commodities._industry_exposures(part)


def international_results(spool: InputSpool, template: ResearchInputs) -> Iterator[ResearchResult]:
    """各市场本体独立流式输出，联合利差只读取名义与实际国债两条来源。

    输入大港股目录不会挤占美债比较预算；同日利差仍由原期限和单位门禁判断。
    """
    yield from source_results(
        spool,
        template,
        international.PRODUCT_ID,
        international.API_FIELDS,
        international.DIMENSIONS,
        partial(international._process, inputs=template),
    )
    curves = spool.materialize(template, ("us_trycr", "us_tycr"))
    yield from international._breakeven(curves)
