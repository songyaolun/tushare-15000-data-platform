"""Offline contracts for precise chips gap collection and failure evidence."""

import json

import pytest

from tushare_entitlement_hub.client import FakeTransport, Limiter, ProviderError
from tushare_entitlement_hub.return80_research.chips_backfill import (
    build_requests,
    collect_partition,
    run_collection,
    save_json,
)


def response(rows, code=0):
    """Build a provider fixture; e.g. response([]) is a successful empty page."""
    return {"code": code, "data": {"fields": ["ts_code", "trade_date", "price", "percent"],
                                  "items": rows}}


def request():
    """Use a two-date exact denominator; e.g. one returned date leaves one missing."""
    return {"parameters": {"ts_code": "000001.SZ", "start_date": "20261015",
                           "end_date": "20261016"},
            "needed_dates": ["20261015", "20261016"]}


def collect(tmp_path, transport, **kwargs):
    """Inject zero-wait test limits; e.g. collect(path, fake) makes no network call."""
    limiter = Limiter(1000000, sleep=lambda _: None)
    return collect_partition(request(), tmp_path, transport, limiter, limiter, **kwargs)


def test_plan_preserves_exact_dates_and_span():
    """Duplicate pairs are deduplicated, while distant gaps get separate ranges."""
    calendar = ["20261014", "20261015", "20261016"]
    result = build_requests([(day, "000001.SZ") for day in calendar] +
                            [(calendar[0], "000001.SZ")], calendar, 2)
    assert [item["needed_dates"] for item in result] == [calendar[:2], calendar[2:]]
    with pytest.raises(ValueError, match="invalid_chip_pair"):
        build_requests([("20200101", "000001.SZ")], calendar)
    with pytest.raises(ValueError, match="calendar"):
        build_requests([], calendar[::-1])


def test_success_missing_and_hash_checked_resume(tmp_path):
    """A short complete page may still lack an exact date; tampered raw prevents reuse."""
    fake = FakeTransport([response([["000001.SZ", "20261015", 10, 100]])])
    result = collect(tmp_path, fake)
    assert result["status"] == "source_dates_missing"
    assert result["missing_dates"] == ["20261016"]
    assert result["reused"] is False
    resumed = collect(tmp_path, FakeTransport([]))
    assert resumed == {**result, "reused": True}
    partition = next((tmp_path / "partitions").iterdir())
    (partition / result["files"][0]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="hash_mismatch"):
        collect(tmp_path, FakeTransport([]))


def test_provider_unavailable_is_persisted_once(tmp_path):
    """Explicit 50101 is durable non-success evidence and is never retried as empty data."""
    fake = FakeTransport([response([], 50101)])
    result = collect(tmp_path, fake)
    assert result["status"] == "provider_unavailable"
    assert result["missing_dates"] == request()["needed_dates"]
    assert len(fake.calls) == 1
    partition = next((tmp_path / "partitions").iterdir())
    evidence = json.loads((partition / result["files"][0]["path"]).read_text())
    assert evidence["response"]["code"] == 50101
    assert not (partition / "raw").exists()


@pytest.mark.parametrize("second", [
    [["000001.SZ", "20261015", 10, 100]],
    [["000002.SZ", "20261015", 10, 100]],
])
def test_duplicate_page_and_wrong_security_fail_closed(tmp_path, second):
    """Repeated page keys or a response for a different security cannot get receipts."""
    fake = FakeTransport([response([["000001.SZ", "20261015", 10, 100]]),
                          response(second), response([])])
    with pytest.raises((ProviderError, ValueError)):
        collect(tmp_path, fake, page_size=1)
    assert not list(tmp_path.rglob("receipt.json"))


def test_page_limit_is_not_complete(tmp_path):
    """A full final permitted page is truncation, even if its data appears plausible."""
    with pytest.raises(ProviderError):
        collect(tmp_path, FakeTransport([response([["000001.SZ", "20261015", 10, 100]])]),
                page_size=1, maximum_pages=1)
    assert not list(tmp_path.rglob("receipt.json"))


def test_late_unavailable_does_not_erase_earlier_pages(tmp_path):
    """50101 at a later offset is ambiguous pagination, not proof the entire range is absent."""
    with pytest.raises(ProviderError):
        collect(tmp_path, FakeTransport([
            response([["000001.SZ", "20261015", 10, 100]]), response([], 50101),
        ]), page_size=1)
    assert not list(tmp_path.rglob("receipt.json"))
    assert len(list(tmp_path.rglob("failure.json"))) == 1


def test_archive_retains_each_raw_hash_and_resumes(tmp_path):
    """Optional per-partition gzip archive reduces files and validates every original page."""
    import tarfile

    (tmp_path / "archive-raw-pages").touch()
    result = collect(tmp_path, FakeTransport([response([["000001.SZ", "20261015", 10, 100]])]))
    partition = next((tmp_path / "partitions").iterdir())
    assert not (partition / "raw").exists()
    with tarfile.open(partition / "raw.tar.gz", "r:gz") as archive:
        assert archive.extractfile(result["files"][0]["path"]) is not None
    assert collect(tmp_path, FakeTransport([]))["reused"] is True
    (partition / "raw.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive_hash"):
        collect(tmp_path, FakeTransport([]))


def test_runner_counts_required_dates_and_empty_plan(tmp_path):
    """Completion counts source dates; an empty plan makes no live call."""
    result = run_collection([request()], tmp_path,
                            FakeTransport([response([["000001.SZ", "20261015", 10, 100]])]),
                            workers=1, api_rate=1000000, account_rate=1000000)
    assert result["covered_required_dates"] == 1
    assert result["unavailable_dates"] == 1
    assert result["status"] == "finished_with_gaps"
    assert result["new_partitions"] == 1
    assert result["estimated_remaining_seconds"] is None
    resumed = run_collection([request()], tmp_path, FakeTransport([]), workers=1)
    assert resumed["reused_partitions"] == 1
    assert resumed["new_partitions"] == 0
    assert resumed["estimated_remaining_seconds"] is None
    assert run_collection([], tmp_path / "empty", FakeTransport([]))["status"] == "complete"


def test_runner_failure_budget_and_public_json(tmp_path):
    """Fatal failures stop scheduling new requests and never serialize arbitrary exceptions."""
    result = run_collection([request()] * 3, tmp_path, FakeTransport([response([], 400)]),
                            workers=1, maximum_failures=1)
    assert result["failed_partitions"] == 1
    assert result["completed_partitions"] == 0
    with pytest.raises(ValueError):
        save_json(tmp_path / "bad.json", {"token": "fixture"})


def test_offline_cli_freezes_scope_but_allows_execution_tuning(tmp_path, monkeypatch):
    """Offline CLI freezes source scope, and permits a rate-only resumable change."""
    import sys

    import duckdb
    import numpy as np

    from tushare_entitlement_hub.return80_research.chips_backfill import main

    pairs = tmp_path / "pairs.parquet"
    with duckdb.connect() as connection:
        connection.execute("COPY (SELECT '20261016' AS trade_date, '000001.SZ' AS ts_code) "
                           "TO ? (FORMAT PARQUET)", [str(pairs)])
    calendar = tmp_path / "calendar.npy"
    np.save(calendar, np.array(["20261015", "20261016"]))
    args = ["chips", "--pairs", str(pairs), "--calendar", str(calendar),
            "--output", str(tmp_path / "raw")]
    monkeypatch.setattr(sys, "argv", args)
    main()
    original = (tmp_path / "raw/plan.json").read_bytes()
    monkeypatch.setattr(sys, "argv", args + ["--api-rate", "400"])
    main()
    assert (tmp_path / "raw/plan.json").read_bytes() == original
    assert len(list((tmp_path / "raw/executions").glob("*.json"))) == 2
    monkeypatch.setattr(sys, "argv", args + ["--span", "50"])
    with pytest.raises(ValueError, match="plan_changed"):
        main()


def test_worker_has_own_progress_and_shared_raw_identity(tmp_path, monkeypatch):
    """A process worker saves raw under the shared root and status under its own directory."""
    from contextlib import nullcontext

    from tushare_entitlement_hub.return80_research import chips_backfill as module

    monkeypatch.setattr(module, "HttpTransport", lambda *a, **k: nullcontext(
        FakeTransport([response([["000001.SZ", "20261015", 10, 100]])])))
    result = module.collect_worker([request()], tmp_path, 0, 1, 100000, 100000, None)
    assert result["new_covered_dates"] == 1
    assert (tmp_path / "workers/0/process.json").exists()
    assert (tmp_path / "workers/0/progress.json").exists()
    assert len(list((tmp_path / "partitions").rglob("receipt.json"))) == 1


def test_multi_process_budget_and_aggregate(tmp_path, monkeypatch):
    """Disjoint shards split the total thread/rate budget and sum actual date coverage."""
    from concurrent.futures import ThreadPoolExecutor

    from tushare_entitlement_hub.return80_research import chips_backfill as module

    calls = []

    def worker(requests, root, index, workers, api_rate, account_rate, credential_file):
        """Record each shard and return bounded metrics; no provider access is used."""
        calls.append((index, workers, api_rate, account_rate, len(requests)))
        return {"completed_partitions": len(requests), "covered_required_dates": 2,
                "new_partitions": len(requests), "new_pages": 1, "new_covered_dates": 2,
                "new_seconds": 3.0}

    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(module, "collect_worker", worker)
    result = module.run_multiprocess([request()] * 4, tmp_path, None, workers=4,
                                    processes=2, api_rate=600, account_rate=600)
    assert sorted(calls) == [(0, 2, 300, 300, 2), (1, 2, 300, 300, 2)]
    assert result["completed_partitions"] == 4
    assert result["covered_required_dates"] == 4
    assert result["estimated_remaining_seconds"] is None
    with pytest.raises(ValueError, match="divisible"):
        module.run_multiprocess([], tmp_path, None, workers=3,
                                processes=2, api_rate=600, account_rate=600)


def test_monitor_only_finalizes_complete_worker_manifests(tmp_path, monkeypatch):
    """A stale parent can be monitored without rescheduling; missing workers delay completion."""
    from tushare_entitlement_hub.return80_research import chips_backfill as module

    save_json(tmp_path / "plan.json", {"requests": [request()]})
    status = {"completed_partitions": 1, "covered_required_dates": 1,
              "unavailable_dates": 1, "failed_partitions": 0, "new_partitions": 1,
              "new_pages": 1, "new_seconds": 70, "seconds": 71}
    sleeps = []

    def finish(interval):
        """Finalize the worker on the first heartbeat, permitting the monitor to exit."""
        sleeps.append(interval)
        save_json(tmp_path / "workers/0/collection.json", status)

    monkeypatch.setattr(module.time, "sleep", finish)
    result = module.monitor_workers(tmp_path, 1, interval=0)
    assert sleeps == [0]
    assert result["status"] == "finished_with_gaps"
    assert result["covered_required_dates"] == 1
    assert result["finalized_workers"] == 1
