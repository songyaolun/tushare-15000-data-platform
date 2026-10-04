"""生产分段写入器的2.5百万行/宽JSON合成压力验证，逐行全量指纹对账。"""

import json
from hashlib import md5, sha256
from time import perf_counter

import duckdb

from tushare_entitlement_hub.equity_research import partitions, storage
from tushare_entitlement_hub.equity_research.persistence_runtime import PersistenceProgress
from tushare_entitlement_hub.equity_research.projection import EncodedRecord, accumulate_records

layout = {
    "result": (100, 32768),
    "feature": (10000, 4),
    "contribution": (2500, 4),
    "evidence": (10000, 8),
    "gate": (10000, 16),
    "signal": (2500, 2),
    "lineage": (15000, 2),
}


def test_millions_of_wide_projections_keep_full_fingerprints_in_512mb(
    tmp_path, prepared_path=None
):
    """250.5万投影及约6GB JSON使用真实分段写入器，完整核对七表SHA256、多重集计数和跨段键。

    不用墙钟阈值；数据同时包含1MiB宽结果及大量窄引用，连接始终512MB，关闭重开后逐段重建。
    prepared_path仅供显式复核本测试已写完的合成库；仍重新生成全部期望行/字节/摘要，不能跳过校验。
    """
    start = perf_counter()
    path = prepared_path if prepared_path is not None else tmp_path / "stage.duckdb"
    connection = duckdb.connect(
        str(path), read_only=prepared_path is not None,
        config={"threads": 1, "memory_limit": "512MB"},
    )
    trace = PersistenceProgress("save", None, None)
    buffers = {}
    if prepared_path is None:
        storage.initialize_tables(connection)
        connection.execute("BEGIN TRANSACTION")
        controller = storage._StageTransactions(connection, trace)
        writer = partitions.PartitionWriter(connection)
        buffers = {
            t: storage._InsertBuffer(connection, t, 128, trace, controller.advance, writer)
            for t in layout
        }
    offsets = {t: 0 for t in layout}
    expected = {table: [0, 0] for table in layout}
    total_bytes = 0
    try:
        for cycle in range(50):
            for table, (count, width) in layout.items():
                for index in range(offsets[table], offsets[table] + count):
                    identity = sha256((table + str(index)).encode()).hexdigest()
                    parent = (
                        identity
                        if table == "result"
                        else sha256(("result" + str(index % 5000)).encode()).hexdigest()
                    )
                    payload = json.dumps(
                        {"synthetic": md5(str(index).encode()).hexdigest() * width, "value": index},
                        separators=(",", ":"),
                    )
                    row = EncodedRecord(table, identity, "run", parent, payload)
                    accumulate_records((row,), expected)
                    total_bytes += len(payload)
                    if prepared_path is None:
                        buffers[table].add(row)
                offsets[table] += count
            if cycle % 10 == 0:
                print(
                    "write" if prepared_path is None else "regenerate_expected",
                    cycle,
                    sum(offsets.values()),
                    total_bytes,
                    round(perf_counter() - start, 3),
                    flush=True,
                )
        for buffer in buffers.values():
            buffer.flush()
        if prepared_path is None:
            connection.execute("COMMIT")
            connection.execute("CHECKPOINT")
        connection.close()
        connection = duckdb.connect(
            str(path), read_only=True, config={"threads": 1, "memory_limit": "512MB"}
        )
        storage._verify_keys(connection, "run", PersistenceProgress("verify", None, None))
        print("keys_complete", round(perf_counter() - start, 3), flush=True)
        actual = accumulate_records(storage._stored_records(connection, "run", tuple(layout)))
        assert expected == actual
        assert sum(offsets.values()) == 2505000
        assert total_bytes > 4800000000
        print(
            "COMPLETE",
            sum(offsets.values()),
            total_bytes,
            len([p for ps in partitions.partition_catalog(connection).values() for p in ps]),
            path.stat().st_size,
            round(perf_counter() - start, 3),
            flush=True,
        )
    finally:
        connection.close()
