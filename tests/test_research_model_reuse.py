"""真实P03消费附加的模型性能回归；对照旧转换算法，不使用绝对墙钟阈值。"""

from collections import Counter
from dataclasses import replace
from time import perf_counter

import duckdb
from test_research_context_consumers import context_inputs, context_row

from tushare_entitlement_hub.equity_research.common import gate, make_result, metric
from tushare_entitlement_hub.equity_research.consumption import (
    attach_consumption,
    validate_result_consumption,
)
from tushare_entitlement_hub.equity_research.consumption_registry import (
    CONSUMPTION_VERSION,
    PRODUCT_ALIASES,
    registry_fingerprint,
)
from tushare_entitlement_hub.equity_research.models import ResearchResult
from tushare_entitlement_hub.equity_research.storage import (
    fingerprints,
    initialize_tables,
    record_rows,
    run_identity,
    save_results,
    stored_rows,
)
from tushare_entitlement_hub.product_foundation import model
from tushare_entitlement_hub.product_foundation.contracts import FeatureContribution, Truth
from tushare_entitlement_hub.product_foundation.entities import Entity, EntityIdentity
from tushare_entitlement_hub.store import digest, encode


def rich_research_results(count=200):
    """生成200个证券结果，每条有8个真实合成引用、8个门禁、4个指标及4项贡献。

    daily行实际交给P03消费；每条附加16个接口记录，其中daily为observed，未供数来源保留unknown。
    不运行其他产品引擎、不读正式reference；所有身份和时间均由确定性夹具生成。
    """
    entities = tuple(
        Entity(
            identity=EntityIdentity(
                namespace="cn", kind="security", local_id=f"S{i:04}", lifecycle_id="listed"
            ),
            name=f"合成证券{i}",
        )
        for i in range(count)
    )
    rows = tuple(
        context_row(
            "daily",
            {
                "ts_code": entity.identity.local_id,
                "trade_date": "20260930",
                "close": 10.0,
                "pre_close": 10.0,
                "pct_chg": 0.0,
                "vol": 100.0,
                "amount": 1000.0,
                "marker": "合成",
            },
            entity=entity.identity.entity_id,
            record=digest(["synthetic", i]),
        )
        for i, entity in enumerate(entities)
    )
    initial = context_inputs()
    inputs = replace(
        initial,
        rows_by_api={"daily": rows},
        config=replace(
            initial.config, foundation=replace(initial.config.foundation, entities=entities)
        ),
    )
    results = []
    for row in rows:
        references = row.evidence(
            "ts_code", "trade_date", "close", "pre_close", "pct_chg", "vol", "amount", "marker"
        )
        components = tuple(
            FeatureContribution(
                feature_id=f"component{i}",
                raw_value=0.5,
                unit="fraction",
                transformed_value=0.5,
                weight=0.25,
                contribution=12.5,
                evidence_refs=references,
            )
            for i in range(4)
        )
        results.append(
            make_result(
                inputs,
                PRODUCT_ALIASES["P03"],
                row.entity_id,
                metrics=tuple(metric(f"feature{i}", 0.5, "fraction", references) for i in range(4)),
                gates=tuple(
                    gate(f"base{i}", Truth.TRUE, "synthetic", references=references)
                    for i in range(8)
                ),
                references=references,
                contributions=components,
                predicate_results={"manual": Truth.UNKNOWN},
            )
        )
    return inputs, tuple(results)


def measured_attachment(inputs, results, monkeypatch, *, legacy):
    """运行真实attach_consumption并计数，legacy只恢复37b8a39的Model转换分支。

    例：基准路径把已有Model经asdict再from_dict；优化路径使用当前实现。
    两边都完整执行消费处理器、所有__post_init__和JSON校验，计时不作为成败阈值。
    """
    counts = Counter()
    typed = model._typed
    original_asdict = model.asdict
    original_from_dict = model.Model.from_dict.__func__

    def counted_asdict(value):
        """透传原asdict并计数；例如一次父模型JSON校验计为一次顶层asdict调用。"""
        counts["asdict"] += 1
        return original_asdict(value)

    def counted_from_dict(cls, value):
        """透传严格公开解析边界并计数，不替换构造或过滤任何验证错误。"""
        counts["from_dict"] += 1
        return original_from_dict(cls, value)

    def legacy_typed(value, annotation):
        """逐字保留旧Model分支，其余类型继续使用未变更的原逻辑，递归调用同样走此分支。"""
        if isinstance(annotation, type) and issubclass(annotation, model.Model):
            return annotation.from_dict(
                model.asdict(value) if isinstance(value, annotation) else value
            )
        return typed(value, annotation)

    with monkeypatch.context() as patch:
        patch.setattr(model, "asdict", counted_asdict)
        patch.setattr(model.Model, "from_dict", classmethod(counted_from_dict))
        if legacy:
            patch.setattr(model, "_typed", legacy_typed)
        started = perf_counter()
        attached = attach_consumption(inputs, PRODUCT_ALIASES["P03"], results)
        seconds = perf_counter() - started
    return attached, dict(counts), seconds


def test_hundreds_of_rich_p03_results_keep_bytes_with_fewer_conversions(monkeypatch):
    """200条多门禁/证据结果逐字节对照旧算法，嵌套转换次数至少减半，且公开解析和512MB重建成功。

    断言只依赖结构与调用数，不依赖机器速度；公开from_json始终重验所有嵌套字典。
    run_id、消费规则版本和注册指纹在优化前后保持一致，不使已有运行失效。
    """
    inputs, results = rich_research_results()
    identity = run_identity(
        "synthetic", inputs.analysis_as_of, inputs.config, (PRODUCT_ALIASES["P03"],)
    )
    rules = (CONSUMPTION_VERSION, registry_fingerprint())
    previous, before, previous_seconds = measured_attachment(
        inputs, results, monkeypatch, legacy=True
    )
    current, after, current_seconds = measured_attachment(
        inputs, results, monkeypatch, legacy=False
    )
    assert len(current) == len(previous) == 200
    assert [item.to_json() for item in current] == [item.to_json() for item in previous]
    assert after["from_dict"] * 2 < before["from_dict"]
    assert after["asdict"] * 2 < before["asdict"]
    assert after["asdict"] >= len(current)
    assert rules == (CONSUMPTION_VERSION, registry_fingerprint())
    assert identity == run_identity(
        "synthetic", inputs.analysis_as_of, inputs.config, (PRODUCT_ALIASES["P03"],)
    )
    for item in current:
        validate_result_consumption(item)
        records = item.details["consumption"]
        assert len(records) == 16
        assert (
            next(record for record in records if record["api"] == "daily")["status"] == "observed"
        )
        assert len(item.envelope.evidence_refs) >= 8
        assert len(item.envelope.quality_gates) >= 24
        assert ResearchResult.from_json(item.to_json()).to_json() == item.to_json()
    print(
        "200条真实P03附加消费 "
        + encode(
            {
                "before": before,
                "after": after,
                "before_seconds": round(previous_seconds, 3),
                "after_seconds": round(current_seconds, 3),
                "product_json_byte_identical": True,
            }
        )
    )
    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "512MB"}) as connection:
        initialize_tables(connection)
        assert save_results(connection, current, 128)["result"] == 200
        assert fingerprints(
            entry for item in current for entry in record_rows(item)
        ) == fingerprints(stored_rows(connection, inputs.run_id, 128))
