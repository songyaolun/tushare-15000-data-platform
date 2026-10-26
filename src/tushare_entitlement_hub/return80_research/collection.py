"""Explicit research collection plans reuse the existing audited HTTPS and paging client."""

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..client import Client, ProviderDataUnavailable, chips_data_unavailable
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


def verify_chips_unavailable(root: Path, partition: dict[str, Any]) -> None:
    """Verify a missing partition against its first-page provider error, without inventing rows.

    Example: an unchanged 50101 chips evidence file passes; a forged successful
    response, mismatched request, later page or changed hash raises ValueError.
    """
    request = partition["request"]
    if (
        partition.get("status") != "provider_data_unavailable"
        or partition.get("provider_code") != 50101
        or partition.get("rows") != 0
        or len(partition["files"]) != 1
    ):
        raise ValueError("invalid_chips_unavailable_receipt")
    item = partition["files"][0]
    path = root / item["path"]
    if not path.resolve().is_relative_to(root.resolve()) or file_digest(path) != item["sha256"]:
        raise ValueError("collection_raw_changed")
    envelope = json.loads(path.read_text())
    assert_public(envelope)
    expected = {**request["parameters"], "fields": ",".join(request["fields"])}
    if request.get("paginate", True):
        expected.update(limit=request["page_size"], offset=0)
    if (
        envelope.get("evidence_kind") != "provider_error_not_successful_raw"
        or envelope.get("api_name") != request["api"]
        or envelope.get("params") != expected
        or not chips_data_unavailable(request["api"], expected, envelope["response"])
    ):
        raise ValueError("invalid_chips_unavailable_evidence")


def collect_plan(
    client: Client, plan_path: Path, output: Path, *, allow_chips_unavailable: bool = False
) -> Path:
    """Collect declared partitions with resumable per-request receipts and raw SHA evidence.

    Example: a request completed yesterday is reused only after its saved raw files verify.
    The Client saves original responses before parsing and rejects truncation/duplicate keys.
    Only a final collection.json marks the whole plan complete; failures keep raw evidence.
    allow_chips_unavailable explicitly retains first-page 50101 as a source gap;
    the default and all other provider/pagination failures remain strict.
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
            unavailable = None
            try:
                pages = client.pages(
                    request["api"],
                    {**request["parameters"], "fields": ",".join(request["fields"])},
                    datetime.now(UTC).isoformat(),
                    request["primary_key"],
                    request["page_size"],
                    request["maximum_pages"],
                    paginate=request.get("paginate", True),
                )
            except ProviderDataUnavailable as error:
                if not allow_chips_unavailable:
                    raise
                unavailable = error.raw
                pages = []
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
            if unavailable is not None:
                receipt.update(
                    status="provider_data_unavailable", provider_code=50101,
                    files=[{
                        "path": unavailable.path.relative_to(output).as_posix(),
                        "sha256": file_digest(unavailable.path),
                    }],
                )
                verify_chips_unavailable(output, receipt)
            temporary = receipt_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(receipt, indent=2) + "\n")
            temporary.replace(receipt_path)
        if receipt.get("status") == "provider_data_unavailable":
            if not allow_chips_unavailable:
                raise ValueError("chips_unavailable_policy_required")
            verify_chips_unavailable(output, receipt)
        receipts.append(receipt)
        print(
            json.dumps(
                {
                    "stage": "collection",
                    "completed": index + 1,
                    "total": len(plan["requests"]),
                    "seconds": time.monotonic() - started,
                    "partition_status": receipt.get("status", "complete"),
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
                "status": "completed_with_source_gaps" if any(
                    p.get("status") == "provider_data_unavailable" for p in receipts
                ) else "complete",
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def read_collection(
    root: Path, api: str, *, allow_chips_unavailable: bool = False
) -> list[dict[str, Any]]:
    """Read only a completed, hashed collection for one API; preserve original field units.

    Example: a complete daily collection returns stock/date records for factor calculation.
    Missing API or a changed raw page fails instead of becoming an empty successful dataset.
    Opt-in chips consumers may omit verified unavailable partitions while keeping
    their error evidence in the manifest; no missing rows are synthesized.
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
        if partition.get("status") == "provider_data_unavailable":
            if not allow_chips_unavailable:
                raise ValueError("chips_unavailable_policy_required")
            verify_chips_unavailable(root, partition)
            continue
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
