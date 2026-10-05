"""按完整业务实体消费磁盘来源，复用现有指标而不把分区样本冒充全量。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from importlib import import_module
from typing import Any

from ..equity_research.common import latest
from ..equity_research.models import ResearchInputs, ResearchResult, ResearchRow
from . import funds, themes
from .market_partitions import commodity_results, financing_results, international_results
from .partitioned_inputs import InputSpool
from .registry import HANDLERS
from .thematic_common import accepted_groups


class ManagerCounts(Mapping[str, int]):
    """跨分区基金管理人计数使用精确磁盘索引，不能只计算当前 ETF 分区。"""

    def __init__(self, spool: InputSpool, template: ResearchInputs) -> None:
        """逐 ETF 复用可见性与 latest 门禁，写入同管理人全观察计数。

        输入两只不同 ETF 的唯一档案均指向管理人 A，得到 A=2；同序冲突不增加计数。
        """
        self.spool = spool
        spool.connection.execute("CREATE TABLE managers(name TEXT PRIMARY KEY,total INTEGER)")
        for entity in spool.entities(("etf_basic",)):
            part = spool.materialize(template, ("etf_basic",), entity=entity)
            rows = accepted_groups(part, ("etf_basic",)).get(entity, {}).get("etf_basic", ())
            row = latest(rows)
            name = None if row is None else row.text("mgr_name")
            if name is not None:
                spool.connection.execute(
                    "INSERT INTO managers VALUES (?,1) ON CONFLICT(name) "
                    "DO UPDATE SET total=total+1",
                    (name,),
                )

    def __getitem__(self, key: str) -> int:
        """读取准确管理人计数；例如 A 返回 2，缺失管理人抛 KeyError。"""
        row = self.spool.connection.execute("SELECT total FROM managers WHERE name=?", (key,))
        found = row.fetchone()
        if found is None:
            raise KeyError(key)
        return int(found[0])

    def __iter__(self) -> Iterator[str]:
        """按管理人名称顺序流式取键；例如两名管理人产生两个键，不物化完整集合。"""
        for (name,) in self.spool.connection.execute("SELECT name FROM managers ORDER BY name"):
            yield name

    def __len__(self) -> int:
        """查询管理人数；例如重复出现 A 三次仍返回 1。"""
        return int(self.spool.connection.execute("SELECT count(*) FROM managers").fetchone()[0])


def theme_results(spool: InputSpool, template: ResearchInputs) -> Iterator[ResearchResult]:
    """每个主题读取完整成员和行情组，已有股票价格索引只建立一次。

    输入百个主题和万行成员产生百个完整结果，主题成员数不受其他主题字节数影响。
    每证券的唯一最新行情只判定一次；同序冲突保留为空，不随主题成员重复扫描历史。
    """
    price_groups = accepted_groups(template, ("daily",))
    prices: dict[str, tuple[ResearchRow, ...]] = {}
    for entity, values in price_groups.items():
        selected = latest(values.get("daily", ()))
        prices[entity] = () if selected is None else (selected,)
    for entity in spool.entities(themes.API_FIELDS):
        part = spool.materialize(template, themes.API_FIELDS, entity=entity)
        yield from themes.build_theme_opportunities(part, prices=prices)


def fund_results(spool: InputSpool, template: ResearchInputs) -> Iterator[ResearchResult]:
    """以单基金完整本体及明确基准关系为分区，跨基金计数来自全量磁盘统计。

    输入 ETF A 的档案、全部篮子及 index_code=I，只联接 I 的完整成分与基准行情。
    不把 A 与 B 的篮子合并，也不将分区内管理人数当作全市场观察计数。
    """
    managers = ManagerCounts(spool, template)
    apis = funds.API_FIELDS.keys() - funds.REFERENCE_APIS
    for entity in spool.entities(apis):
        basic_part = spool.materialize(template, ("etf_basic",), entity=entity)
        rows = accepted_groups(basic_part, ("etf_basic",)).get(entity, {}).get("etf_basic", ())
        basic = latest(rows)
        codes = {}
        if basic is not None:
            index = basic.text("index_code")
            manager = basic.text("mgr_name")
            if index is not None:
                codes.update(
                    {
                        api: index
                        for api in (
                            "etf_index",
                            "mkt_idx_bmk",
                            "index_weight",
                            "index_daily",
                        )
                    }
                )
            if manager is not None:
                codes["fund_company"] = manager
        part = spool.materialize(template, apis | codes.keys(), entity=entity, codes=codes)
        yield from funds.build_etf_exposure(part, manager_counts=managers)


def partitioned_results(
    product: str,
    spool: InputSpool,
    template: ResearchInputs,
    handler: Callable[..., Any],
    *,
    parents: tuple[ResearchResult, ...] = (),
    holdings: tuple[dict[str, Any], ...] = (),
) -> Iterator[ResearchResult]:
    """选择业务等价的完整分区策略，只有父组合使用本次明确父结果和持仓。

    输入 P02/P05/P08 采用实体分区；P09/P10/P11 分别保留明确的完整联合关系；
    P12 按基金分区或直接使用显式持仓，所有分区遵守 16MiB 完整输入预算。
    """
    strategies = {
        "P02": theme_results,
        "P08": fund_results,
        "P09": commodity_results,
        "P10": international_results,
        "P11": financing_results,
    }
    if product in strategies:
        yield from strategies[product](spool, template)
        return
    module = import_module(f"{__package__}.{HANDLERS[product].split('.')[0]}")
    if product == "P05":
        for entity in spool.entities(module.API_FIELDS):
            yield from handler(spool.materialize(template, module.API_FIELDS, entity=entity))
    else:
        if holdings:
            yield from handler(spool.materialize(template, ()), parents=parents, holdings=holdings)
            return
        found = False
        for entity in spool.entities(module.API_FIELDS):
            found = True
            part = spool.materialize(template, module.API_FIELDS, entity=entity)
            yield from handler(part, parents=parents, holdings=holdings)
        if not found:
            yield from handler(spool.materialize(template, ()), parents=parents, holdings=holdings)
