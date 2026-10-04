"""正式实体数量级的私有保存、关闭重开、完整重建与发布，保持512MB连接预算。"""

from hashlib import sha256
from time import perf_counter

import duckdb
from research_fixture import AS_OF, DATABASE_CONFIG, write_reference

from tushare_entitlement_hub.equity_research import pipeline, storage
from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.models import PRODUCT_IDS, ResearchResult


def test_5911_full_consumption_results_validate_after_writer_close(tmp_path, monkeypatch):
    """5911条真实attach结果的总JSON超过512MB，逐结果小于8MiB；关闭写连接后完整重建再发布。

    无墙钟阈值；验证只读新连接、每条结果恰好重建一次、七表全部对账、两次运行幂等。
    旧实现在验证入口仍有活动事务，BEGIN立即失败，避免用等待OOM或数十分钟超时判定回归。
    """
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "scale.duckdb"
    count = 5911
    verify, parse = pipeline.verify_run, ResearchResult.from_json.__func__
    parsed = total_bytes = maximum_bytes = 0
    timings = {}
    commits = 0
    start = perf_counter()

    def synthetic_results(inputs):
        """只替代业务输入规模，后续真实attach/save/verify均保留；例：5911证券各有可审计合成说明。"""
        return tuple(
            make_result(
                inputs,
                PRODUCT_IDS[2],
                f"cn:stock:synthetic:{index}",
                details={"synthetic_scale": sha256(str(index).encode()).hexdigest() * 800},
            )
            for index in range(count)
        )

    def parsed_once(cls, payload):
        """完整调用原from_json并统计字节；例：不以模板复用替代实际持久化重建。"""
        nonlocal parsed, total_bytes, maximum_bytes
        parsed += 1
        size = len(payload.encode("utf-8"))
        total_bytes += size
        maximum_bytes = max(maximum_bytes, size)
        return parse(cls, payload)

    def verified(connection, metadata, **kwargs):
        """开始独立事务证明写事务已结束，仍执行真实verify全部模型和物理投影。"""
        connection.execute("BEGIN TRANSACTION")
        connection.execute("ROLLBACK")
        limit = connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        assert limit == "488.2 MiB"
        assert metadata["summary"]["result_count"] == count
        return verify(connection, metadata, **kwargs)

    def progress(event):
        """记录真实阶段耗时；例：save-complete到verify-start必须经过提交、关闭及新连接。"""
        nonlocal commits
        if event.get("operation") == "commit_batch":
            commits += 1
        if event["stage"] in {
            "product_complete",
            "start-save",
            "save-complete",
            "verify-start",
            "verify-complete",
            "complete",
        }:
            timings[event["stage"]] = perf_counter() - start
            print(f"规模阶段 {event['stage']}: {timings[event['stage']]:.3f}s", flush=True)
        if event["stage"] != "complete":
            assert not output.exists()

    monkeypatch.setattr(pipeline, "build_governance", synthetic_results)
    monkeypatch.setattr(pipeline, "verify_run", verified)
    monkeypatch.setattr(ResearchResult, "from_json", classmethod(parsed_once))
    summary = pipeline.run_research(reference, output, AS_OF, config, ("P06",), progress=progress)
    assert commits > 0
    assert parsed == count
    assert total_bytes > 512 * 1024 * 1024
    assert maximum_bytes < storage.MAXIMUM_RESULT_JSON_BYTES
    assert summary["counts"]["result"] == count
    first_bytes = total_bytes
    assert pipeline.run_research(reference, output, AS_OF, config, ("P06",)) == summary
    assert parsed == 2 * count and total_bytes == 2 * first_bytes
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        assert connection.execute("SELECT count(*) FROM research.run").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM research.result").fetchone() == (count,)
    assert not list(tmp_path.glob(".*.stage-*"))
    print(
        f"正式规模合成: results={count}, JSON总字节={first_bytes}, 单结果最大={maximum_bytes}, "
        f"投影行数={sum(summary['counts'].values())}, 阶段秒={timings}"
    )
