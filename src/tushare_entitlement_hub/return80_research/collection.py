"""Explicit research collection plans reuse the existing audited HTTPS and paging client."""

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..client import Client
from ..raw import assert_public
from ..strategy_discovery.replay_inputs import file_digest


def load_collection_plan(path: Path) -> dict[str, Any]:
    """Validate public collection requests before any network action.

    Example: one daily request declares fields, date, primary key and pagination limits.
    No interface is assumed entitled merely because it occurs in a plan.
    """
    plan: dict[str, Any] = json.loads(path.read_text())
    assert_public(plan)
    if plan.get("schema") != "return80-collection-plan/v1" or not plan.get("requests"):
        raise ValueError("invalid_collection_plan")
    identities = set()
    for request in plan["requests"]:
        identity = json.dumps(request, sort_keys=True)
        if (
            identity in identities
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", request["api"])
            or not request.get("fields")
            or not request.get("primary_key")
            or not set(request["primary_key"]) <= set(request["fields"])
            or not isinstance(request.get("parameters"), dict)
            or type(request.get("page_size")) is not int
            or request["page_size"] < 1
            or type(request.get("maximum_pages")) is not int
            or request["maximum_pages"] < 1
            or any(key in request["parameters"] for key in ("limit", "offset", "fields"))
        ):
            raise ValueError("invalid_collection_request")
        identities.add(identity)
    return plan


def collect_plan(client: Client, plan_path: Path, output: Path) -> Path:
    """Collect declared partitions with resumable per-request receipts and raw SHA evidence.

    Example: a request completed yesterday is reused only after its saved raw files verify.
    The Client saves original responses before parsing and rejects truncation/duplicate keys.
    Only a final collection.json marks the whole plan complete; failures keep raw evidence.
    """
    plan = load_collection_plan(plan_path)
    output.mkdir(parents=True, exist_ok=True)
    plan_hash = file_digest(plan_path)
    identity_path = output / "plan-sha256"
    if identity_path.exists() and identity_path.read_text().strip() != plan_hash:
        raise ValueError("collection_plan_changed_use_new_directory")
    identity_path.write_text(plan_hash + "\n")
    receipts = []
    started = time.monotonic()
    for index, request in enumerate(plan["requests"]):
        receipt_path = output / f"partition-{index:06d}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["request"] != request:
                raise ValueError("collection_receipt_mismatch")
            for item in receipt["files"]:
                source = output / item["path"]
                if (
                    not source.resolve().is_relative_to(output.resolve())
                    or file_digest(source) != item["sha256"]
                ):
                    raise ValueError("collection_raw_changed")
        else:
            pages = client.pages(
                request["api"],
                {**request["parameters"], "fields": ",".join(request["fields"])},
                datetime.now(UTC).isoformat(),
                request["primary_key"],
                request["page_size"],
                request["maximum_pages"],
                paginate=request.get("paginate", True),
            )
            for page in pages:
                if not set(request["fields"]) <= set(page.fields):
                    raise ValueError("collection_missing_requested_fields")
                for row in page.rows:
                    for parameter, field in (("trade_date", "trade_date"), ("period", "end_date")):
                        if (
                            parameter in request["parameters"]
                            and row.get(field) != request["parameters"][parameter]
                        ):
                            raise ValueError("collection_partition_date_mismatch")
            receipt = {
                "request": request,
                "rows": sum(len(page.rows) for page in pages),
                "files": [
                    {
                        "path": page.raw.path.relative_to(output).as_posix(),
                        "sha256": file_digest(page.raw.path),
                    }
                    for page in pages
                ],
            }
            temporary = receipt_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(receipt_path)
        receipts.append(receipt)
        print(
            json.dumps(
                {
                    "stage": "collection",
                    "completed": index + 1,
                    "total": len(plan["requests"]),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    manifest = output / "collection.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "return80-collection/v1",
                "plan_sha256": plan_hash,
                "historical_pit_certified": False,
                "partitions": receipts,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def read_collection(root: Path, api: str) -> list[dict[str, Any]]:
    """Read only a completed, hashed collection for one API; preserve original field units.

    Example: a complete daily collection returns stock/date records for factor calculation.
    Missing API or a changed raw page fails instead of becoming an empty successful dataset.
    """
    from ..research_history.collection import response_records

    manifest = json.loads((root / "collection.json").read_text())
    if manifest.get("schema") != "return80-collection/v1":
        raise ValueError("invalid_collection_manifest")
    rows = []
    matched = False
    for partition in manifest["partitions"]:
        if partition["request"]["api"] != api:
            continue
        matched = True
        for item in partition["files"]:
            path = root / item["path"]
            if (
                not path.resolve().is_relative_to(root.resolve())
                or file_digest(path) != item["sha256"]
            ):
                raise ValueError("collection_raw_changed")
            envelope = json.loads(path.read_text())
            assert_public(envelope)
            if envelope["api_name"] != api:
                raise ValueError("collection_api_mismatch")
            rows.extend(response_records(envelope["response"]))
    if not matched:
        raise ValueError("missing_collection_api")
    return rows
