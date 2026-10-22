"""Bounded, resumable chips collection for explicit missing security/date pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from ..client import Client, HttpTransport, Limiter, ProviderError, Transport
from ..credentials import configured_credential
from ..raw import RawStore, canonical_json
from ..strategy_discovery.replay_inputs import file_digest

FIELDS = "ts_code,trade_date,price,percent"


def save_json(path: Path, value: Any) -> None:
    """Atomically write public evidence; e.g. save_json(path, {'rows': 2})."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical_json(value) + "\n")
    temporary.replace(path)


def build_requests(pairs: list[tuple[str, str]], calendar: list[str],
                   span: int = 20) -> list[dict[str, Any]]:
    """Group missing dates by security within a bounded calendar span.

    Example: two missing days within 20 sessions produce one range, preserving
    the exact needed dates separately. No provider absence removes a denominator.
    """
    if span < 1 or len(set(calendar)) != len(calendar) or calendar != sorted(calendar):
        raise ValueError("invalid_chip_calendar_or_span")
    positions = {day: index for index, day in enumerate(calendar)}
    grouped: dict[str, set[str]] = {}
    for day, code in pairs:
        if day not in positions or not re.fullmatch(r"\d{6}\.(SZ|SH|BJ)", code):
            raise ValueError("invalid_chip_pair")
        grouped.setdefault(code, set()).add(day)
    requests = []
    for code, dates in sorted(grouped.items()):
        remaining = sorted(dates)
        while remaining:
            count = next((index for index, day in enumerate(remaining)
                          if positions[day] - positions[remaining[0]] >= span), len(remaining))
            selected, remaining = remaining[:count], remaining[count:]
            requests.append({"parameters": {"ts_code": code, "start_date": selected[0],
                                            "end_date": selected[-1]},
                             "needed_dates": selected})
    return requests


class EvidenceTransport:
    """Record nonzero provider responses without admitting them to successful raw data."""

    def __init__(self, transport: Transport, root: Path, limiter: Limiter) -> None:
        """Bind one partition's evidence and shared account budget; e.g. root/errors."""
        self.transport, self.root, self.limiter = transport, root, limiter
        self.errors: list[dict[str, Any]] = []

    def request(self, api_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Persist provider errors before Client classifies them; 50101 stays unavailable.

        Example: code 50101 creates hashed evidence and returns that original response.
        No error body or credential is printed, and no unbounded retry is performed.
        """
        self.limiter.acquire("chips-backfill", "account")
        response = self.transport.request(api_name, params)
        if response.get("code") != 0:
            envelope = {"api_name": api_name, "params": params, "response": response,
                        "observed_at": datetime.now(UTC).isoformat(),
                        "evidence_kind": "provider_error_not_successful_raw"}
            identity = hashlib.sha256(canonical_json(envelope).encode()).hexdigest()
            path = self.root / "errors" / f"{identity}.json"
            save_json(path, envelope)
            self.errors.append({"path": str(path.relative_to(self.root)),
                                "sha256": file_digest(path), "provider_code": response.get("code"),
                                "offset": params.get("offset", 0)})
        return response


def collect_partition(request: dict[str, Any], root: Path, transport: Transport,
                      api_limiter: Limiter, account_limiter: Limiter, *,
                      page_size: int = 6000, maximum_pages: int = 100) -> dict[str, Any]:
    """Collect and validate a complete bounded partition, retaining exact missing dates.

    Example: a 6000-row page requires a following short page; duplicate keys,
    changed security/date ranges, or incomplete pagination never get success receipts.
    A prior receipt is reusable only when its request and every raw SHA still match.
    """
    identity = hashlib.sha256(canonical_json(request).encode()).hexdigest()[:24]
    partition = root / "partitions" / identity
    receipt_path = partition / "receipt.json"
    if receipt_path.exists():
        receipt: dict[str, Any] = json.loads(receipt_path.read_text())
        if receipt["request"] != request:
            raise ValueError("chip_receipt_request_mismatch")
        archive_reference = receipt.get("raw_archive")
        if archive_reference:
            archive_path = partition / archive_reference["path"]
            if (not archive_path.resolve().is_relative_to(partition.resolve())
                    or file_digest(archive_path) != archive_reference["sha256"]):
                raise ValueError("chip_raw_archive_hash_mismatch")
            with tarfile.open(archive_path, "r:gz") as archive:
                for item in receipt["files"]:
                    member = archive.extractfile(item["path"])
                    if (member is None
                            or hashlib.sha256(member.read()).hexdigest() != item["sha256"]):
                        raise ValueError("chip_raw_hash_mismatch")
        else:
            for item in receipt["files"]:
                path = partition / item["path"]
                if not path.resolve().is_relative_to(partition.resolve()):
                    raise ValueError("chip_receipt_path_escape")
                if file_digest(path) != item["sha256"]:
                    raise ValueError("chip_raw_hash_mismatch")
        return {**receipt, "reused": True}
    evidence = EvidenceTransport(transport, partition, account_limiter)
    client = Client(evidence, RawStore(partition / "raw"), account="chips-backfill",
                    limiter=api_limiter, retries=1, retry_base=2)
    try:
        pages = client.pages("cyq_chips", {**request["parameters"], "fields": FIELDS},
                             datetime.now(UTC).isoformat(),
                             ["ts_code", "trade_date", "price"], page_size, maximum_pages)
    except ProviderError as error:
        if (evidence.errors and evidence.errors[-1]["provider_code"] == 50101
                and evidence.errors[-1]["offset"] == 0):
            receipt = {"request": request, "status": "provider_unavailable",
                       "provider_code": 50101, "rows": 0, "covered_dates": [],
                       "missing_dates": request["needed_dates"], "files": evidence.errors}
            save_json(receipt_path, receipt)
            return {**receipt, "reused": False}
        save_json(partition / "failure.json", {"request": request, "kind": error.kind,
                                               "files": evidence.errors})
        raise
    params = request["parameters"]
    covered: set[str] = set()
    files: list[dict[str, Any]] = []
    for page in pages:
        if not set(FIELDS.split(",")) <= set(page.fields):
            raise ValueError("chip_missing_fields")
        for row in page.rows:
            if row["ts_code"] != params["ts_code"] or not (
                params["start_date"] <= row["trade_date"] <= params["end_date"]
            ):
                raise ValueError("chip_partition_scope_mismatch")
            covered.add(row["trade_date"])
        files.append({"path": str(page.raw.path.relative_to(partition)),
                      "sha256": file_digest(page.raw.path), "rows": len(page.rows)})
    missing = sorted(set(request["needed_dates"]) - covered)
    receipt = {"request": request, "status": "complete" if not missing else "source_dates_missing",
               "rows": sum(len(page.rows) for page in pages), "covered_dates": sorted(covered),
               "missing_dates": missing, "files": files}
    if (root / "archive-raw-pages").exists():
        archive_path = partition / "raw.tar.gz"
        temporary = archive_path.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
            for item in files:
                archive.add(partition / item["path"], arcname=item["path"], recursive=False)
        temporary.replace(archive_path)
        receipt["raw_archive"] = {"path": archive_path.name,
                                  "sha256": file_digest(archive_path)}
    save_json(receipt_path, receipt)
    if receipt.get("raw_archive"):
        shutil.rmtree(partition / "raw")
    return {**receipt, "reused": False}


def run_collection(requests: list[dict[str, Any]], root: Path, transport: Transport, *,
                   workers: int = 12, api_rate: float = 200, account_rate: float = 400,
                   maximum_failures: int = 8, progress_root: Path | None = None) -> dict[str, Any]:
    """Run a bounded work queue with progress/heartbeat and fail-closed receipts.

    Example: run_collection(two_requests, root, fake, workers=1) returns counts
    of partitions and required source dates, not a misleading raw-file completion count.
    """
    if workers < 1 or maximum_failures < 1:
        raise ValueError("invalid_chip_concurrency")
    api_limiter, account_limiter = Limiter(api_rate), Limiter(account_rate)
    progress_root = progress_root or root
    started = time.monotonic()
    completed = covered = unavailable = failed = pages = consecutive_failures = 0
    reused = new_completed = new_covered = new_pages = 0
    new_started: float | None = None
    error_counts: dict[str, int] = {}
    failures = []
    total_pairs = sum(len(request["needed_dates"]) for request in requests)
    status: dict[str, Any] = {"stage": "chips_backfill", "completed_partitions": 0,
              "total_partitions": len(requests), "covered_required_dates": 0,
              "total_required_dates": total_pairs, "unavailable_dates": 0,
              "failed_partitions": 0, "seconds": 0.0}
    remaining = iter(requests)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Any, dict[str, Any]] = {}
        while True:
            while len(pending) < workers and consecutive_failures < maximum_failures:
                request = next(remaining, None)
                if request is None:
                    break
                pending[pool.submit(collect_partition, request, root, transport,
                                    api_limiter, account_limiter)] = request
            if not pending:
                break
            done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                request = pending.pop(future)
                try:
                    receipt = future.result()
                    completed += 1
                    consecutive_failures = 0
                    pages += len(receipt["files"])
                    if receipt["reused"]:
                        reused += 1
                    else:
                        if new_started is None:
                            new_started = time.monotonic()
                        new_completed += 1
                        new_pages += len(receipt["files"])
                        new_covered += (len(request["needed_dates"])
                                        - len(receipt["missing_dates"]))
                    covered += len(request["needed_dates"]) - len(receipt["missing_dates"])
                    unavailable += len(receipt["missing_dates"])
                except Exception as error:
                    failed += 1
                    consecutive_failures += 1
                    kind = error.kind if isinstance(error, ProviderError) else type(error).__name__
                    error_counts[kind] = error_counts.get(kind, 0) + 1
                    failures.append({"request": request, "error_type": type(error).__name__})
            status = {"stage": "chips_backfill", "completed_partitions": completed,
                      "total_partitions": len(requests), "covered_required_dates": covered,
                      "total_required_dates": total_pairs, "unavailable_dates": unavailable,
                      "failed_partitions": failed, "seconds": round(time.monotonic() - started, 2)}
            new_seconds = time.monotonic() - new_started if new_started else 0.0
            status.update({"completed_pages": pages, "error_counts": error_counts,
                           "reused_partitions": reused, "new_partitions": new_completed,
                           "new_pages": new_pages, "new_covered_dates": new_covered,
                           "new_seconds": round(new_seconds, 2),
                           "consecutive_failures": consecutive_failures,
                           "estimated_remaining_seconds": round(
                               (len(requests) - completed - failed) * new_seconds / new_completed
                           ) if new_completed >= 30 and new_seconds >= 60 else None})
            save_json(progress_root / "status.json", status)
            save_json(progress_root / "progress.json", status)
            print(json.dumps(status), flush=True)
    result = {**status, "failures": failures, "historical_pit_certified": False,
              "status": "finished_with_gaps" if failed or unavailable else "complete"}
    save_json(progress_root / "collection.json", result)
    return result


def collect_worker(requests: list[dict[str, Any]], root: Path, index: int, workers: int,
                   api_rate: float, account_rate: float,
                   credential_file: Path | None) -> dict[str, Any]:
    """Own one disjoint request shard and HTTPS pool; e.g. shard 0 handles requests[0::3]."""
    progress_root = root / "workers" / str(index)
    save_json(progress_root / "process.json", {
        "pid": os.getpid(), "process_starttime_ticks": (
            int(Path("/proc/self/stat").read_text().split()[21])
            if Path("/proc/self/stat").exists() else None),
        "workers": workers, "api_rate": api_rate, "account_rate": account_rate,
    })
    with HttpTransport(lambda: configured_credential(credential_file) or "",
                       live=True, timeout=45) as transport:
        return run_collection(requests, root, transport, workers=workers, api_rate=api_rate,
                              account_rate=account_rate, progress_root=progress_root)


def run_multiprocess(requests: list[dict[str, Any]], root: Path,
                     credential_file: Path | None, *, workers: int, processes: int,
                     api_rate: float, account_rate: float) -> dict[str, Any]:
    """Partition work across processes while keeping total workers and request budget bounded.

    Example: processes=3, workers=12, api_rate=750 gives each process four threads
    and 250 requests/minute. Raw request identities remain unchanged across resumption.
    """
    if processes < 1 or workers < processes or workers % processes:
        raise ValueError("chip_workers_must_be_divisible_by_processes")
    started = time.monotonic()
    counters = ("completed_partitions", "covered_required_dates", "unavailable_dates",
                "failed_partitions", "completed_pages", "reused_partitions", "new_partitions",
                "new_pages", "new_covered_dates")
    statuses: dict[int, dict[str, Any]] = {}
    for index in range(processes):
        for name in ("progress.json", "collection.json"):
            (root / "workers" / str(index) / name).unlink(missing_ok=True)
    with ProcessPoolExecutor(max_workers=processes) as pool:
        pending = {pool.submit(collect_worker, requests[index::processes], root, index,
                               workers // processes, api_rate / processes,
                               account_rate / processes, credential_file): index
                   for index in range(processes)}
        while pending:
            done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                statuses[index] = future.result()
            for index in pending.values():
                path = root / "workers" / str(index) / "progress.json"
                if path.exists():
                    try:
                        statuses[index] = json.loads(path.read_text())
                    except (OSError, json.JSONDecodeError):
                        # An eventually consistent mount may briefly expose a rename gap.
                        # Keep the prior snapshot, then retry at the next heartbeat.
                        continue
            result: dict[str, Any] = {key: sum(value.get(key, 0) for value in statuses.values())
                                      for key in counters}
            new_seconds = max((value.get("new_seconds", 0) for value in statuses.values()),
                              default=0)
            result.update({"stage": "chips_backfill", "processes": processes, "workers": workers,
                           "seconds": round(time.monotonic() - started, 2),
                           "new_seconds": new_seconds, "total_partitions": len(requests),
                           "total_required_dates": sum(len(r["needed_dates"]) for r in requests),
                           "estimated_remaining_seconds": round(
                               (len(requests) - result["completed_partitions"]
                                - result["failed_partitions"]) * new_seconds
                               / result["new_partitions"]
                           ) if new_seconds >= 60 and result["new_partitions"] >= 30 else None,
                           "error_counts": {
                               key: sum(value.get("error_counts", {}).get(key, 0)
                                        for value in statuses.values())
                               for key in set().union(*(value.get("error_counts", {})
                                                        for value in statuses.values()))},
                           "historical_pit_certified": False})
            save_json(root / "progress.json", result)
            print(json.dumps(result), flush=True)
    result["status"] = ("finished_with_gaps" if result["failed_partitions"]
                        or result["unavailable_dates"] else "complete")
    save_json(root / "collection.json", result)
    return result


def monitor_workers(root: Path, processes: int, interval: float = 15) -> dict[str, Any]:
    """Reconstruct progress from healthy workers without stopping or rescheduling them.

    Example: monitor_workers(root, 3) repairs stale parent progress while the three
    existing workers continue. Only complete worker manifests permit finalization.
    """
    plan = json.loads((root / "plan.json").read_text())
    statuses: dict[int, dict[str, Any]] = {}
    counters = ("completed_partitions", "covered_required_dates", "unavailable_dates",
                "failed_partitions", "completed_pages", "reused_partitions", "new_partitions",
                "new_pages", "new_covered_dates")
    while True:
        finalized = 0
        for index in range(processes):
            worker = root / "workers" / str(index)
            final = worker / "collection.json"
            try:
                statuses[index] = json.loads((final if final.exists()
                                             else worker / "progress.json").read_text())
                finalized += int(final.exists())
            except (OSError, json.JSONDecodeError):
                continue
        result: dict[str, Any] = {key: sum(value.get(key, 0) for value in statuses.values())
                                  for key in counters}
        new_seconds = max((value.get("new_seconds", 0) for value in statuses.values()), default=0)
        result.update({"stage": "chips_backfill", "processes": processes,
                       "total_partitions": len(plan["requests"]),
                       "total_required_dates": sum(len(r["needed_dates"])
                                                   for r in plan["requests"]),
                       "seconds": max((s["seconds"] for s in statuses.values()), default=0),
                       "new_seconds": new_seconds, "finalized_workers": finalized,
                       "monitor": "worker_receipt_progress_recovery",
                       "estimated_remaining_seconds": round(
                           (len(plan["requests"]) - result["completed_partitions"]
                            - result["failed_partitions"]) * new_seconds
                           / result["new_partitions"]
                       ) if new_seconds >= 60 and result["new_partitions"] >= 30 else None,
                       "error_counts": {
                           key: sum(value.get("error_counts", {}).get(key, 0)
                                    for value in statuses.values())
                           for key in set().union(*(value.get("error_counts", {})
                                                    for value in statuses.values()))},
                       "historical_pit_certified": False})
        save_json(root / "progress.json", result)
        print(json.dumps(result), flush=True)
        if finalized == processes:
            result["status"] = ("finished_with_gaps" if result["failed_partitions"]
                                or result["unavailable_dates"]
                                or result["completed_partitions"] != len(plan["requests"])
                                else "complete")
            save_json(root / "collection.json", result)
            return result
        time.sleep(interval)


def main() -> None:
    """Read explicit parquet gaps and calendar, plan offline, and collect only with --live.

    Example: python -m ...chips_backfill --pairs missing.parquet --calendar calendar.npy
    --output raw/chips creates a frozen plan without making network requests.
    """
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--span", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--api-rate", type=float, default=200)
    parser.add_argument("--account-rate", type=float, default=400)
    parser.add_argument("--maximum-requests", type=int)
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    with duckdb.connect(config={"threads": "2", "memory_limit": "1GB"}) as connection:
        pairs = connection.execute("SELECT trade_date, ts_code FROM read_parquet(?)",
                                   [str(args.pairs)]).fetchall()
    requests = build_requests(pairs, np.load(args.calendar, allow_pickle=False).tolist(), args.span)
    requests.sort(key=lambda request: request["parameters"]["end_date"], reverse=True)
    if args.maximum_requests is not None:
        requests = requests[:args.maximum_requests]
    plan = {"schema": "chips-backfill/v1", "pairs_sha256": file_digest(args.pairs),
            "calendar_sha256": file_digest(args.calendar), "span": args.span,
            "requests": requests, "workers": args.workers, "api_rate": args.api_rate,
            "account_rate": args.account_rate}
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        execution_fields = {"workers", "api_rate", "account_rate"}
        if ({k: v for k, v in previous.items() if k not in execution_fields}
                != {k: v for k, v in plan.items() if k not in execution_fields}):
            raise ValueError("chip_plan_changed_use_new_directory")
    else:
        save_json(plan_path, plan)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    save_json(args.output / "executions" / f"{stamp}.json", {
        "plan_sha256": file_digest(plan_path), "source_sha256": file_digest(Path(__file__)),
        "workers": args.workers, "api_rate": args.api_rate, "account_rate": args.account_rate,
        "processes": args.processes,
        "pid": os.getpid(), "process_starttime_ticks": (
            int(Path("/proc/self/stat").read_text().split()[21])
            if Path("/proc/self/stat").exists() else None),
    })
    (args.output / "executions" / f"{stamp}.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"stage": "plan", "requests": len(requests),
                      "required_dates": sum(len(r["needed_dates"]) for r in requests)}), flush=True)
    if args.live:
        if args.processes > 1:
            result = run_multiprocess(requests, args.output, args.credential_file,
                                      workers=args.workers, processes=args.processes,
                                      api_rate=args.api_rate, account_rate=args.account_rate)
        else:
            with HttpTransport(lambda: configured_credential(args.credential_file) or "",
                               live=True, timeout=45) as transport:
                result = run_collection(requests, args.output, transport, workers=args.workers,
                                        api_rate=args.api_rate, account_rate=args.account_rate)
        if result["failed_partitions"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
