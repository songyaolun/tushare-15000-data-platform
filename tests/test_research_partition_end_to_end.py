"""大于DuckDB预算的宽ResearchResult在真实保存、重建与复用链路中保持有界。"""

from hashlib import sha256
from time import perf_counter

import duckdb
from research_fixture import AS_OF, DATABASE_CONFIG, write_reference

from tushare_entitlement_hub.equity_research import partitions, pipeline, storage
from tushare_entitlement_hub.equity_research.common import make_result
from tushare_entitlement_hub.equity_research.models import PRODUCT_IDS, ResearchResult


def test_wide_results_are_saved_rebuilt_and_reused_in_512mb(tmp_path, monkeypatch):
    """640条约1MiB真实消费结果超过512MB，保持逐条完整重建、原子发布及幂等；每段不超过16MiB。"""
    reference, config = write_reference(tmp_path, 0)
    output = tmp_path / "wide-products.duckdb"
    count = 640
    original_parse = ResearchResult.from_json.__func__
    parsed = total_bytes = largest = 0
    checksum = 0
    started = perf_counter()

    def build(inputs):
        """只扩大合成解释字段；后续执行真实attach_consumption/save_results/verify_run，不替换持久化。"""
        return tuple(
            make_result(
                inputs,
                PRODUCT_IDS[2],
                f"cn:stock:synthetic:{index}",
                details={"synthetic_capacity": sha256(str(index).encode()).hexdigest() * 16384},
            )
            for index in range(count)
        )

    def parse(cls, payload):
        """完整公开解析并统计原始字节；例：每条持久化结果在一次verify中恰好解析一次。"""
        nonlocal parsed, total_bytes, largest, checksum
        parsed += 1
        encoded = payload.encode("utf-8")
        total_bytes += len(encoded)
        largest = max(largest, len(encoded))
        checksum = (checksum + int(sha256(encoded).hexdigest(), 16)) % (2**256)
        return original_parse(cls, payload)

    def progress(event):
        """验证前正式文件不存在；例：verify-complete仍不代表已经发布。"""
        if event["stage"] != "complete":
            assert not output.exists()

    monkeypatch.setattr(pipeline, "build_governance", build)
    monkeypatch.setattr(ResearchResult, "from_json", classmethod(parse))
    first = pipeline.run_research(reference, output, AS_OF, config, ("P06",), progress=progress)
    assert parsed == count
    assert total_bytes > 512 * 1024 * 1024
    assert largest < storage.MAXIMUM_RESULT_JSON_BYTES
    first_bytes, first_checksum = total_bytes, checksum
    assert pipeline.run_research(reference, output, AS_OF, config, ("P06",)) == first
    assert parsed == count * 2 and total_bytes == first_bytes * 2
    assert checksum == first_checksum * 2 % (2**256)
    with duckdb.connect(str(output), read_only=True, config=DATABASE_CONFIG) as connection:
        plan = partitions.validate_partition_layout(connection)
        assert len(plan["result"]) > 1
        for name in plan["result"]:
            size = connection.execute(
                f'SELECT sum(octet_length(encode(payload::VARCHAR))) FROM research."{name}"'
            ).fetchone()[0]
            assert size <= partitions.MAXIMUM_PARTITION_BYTES
        assert connection.execute("SELECT count(*) FROM research.result").fetchone() == (count,)
    assert storage.validate_research(reference, output)["valid"]
    assert parsed == count * 3 and checksum == first_checksum * 3 % (2**256)
    assert not list(tmp_path.glob(".*.stage-*"))
    print(
        f"宽结果端到端：{count}结果，{first_bytes}字节，最大{largest}，"
        f"耗时{perf_counter() - started:.3f}s"
    )
