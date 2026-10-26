"""Independent daily basic eligibility and weekly origins, without signal selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ..raw import assert_public
from ..research_history.collection import response_records


def digest(path: Path) -> str:
    """Hash exact input bytes; for example a saved NPY gets a reproducible SHA256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_origin(source: Path) -> dict[str, Any]:
    """Record installed VCS revision without assuming site-packages belongs to a Git tree.

    Example: a frozen Git wheel's direct_url.json supplies its exact commit_id; an editable
    checkout alone may fall back to local Git. Unversioned wheels retain a null revision.
    """
    direct = metadata.distribution("tushare-entitlement-hub").read_text("direct_url.json")
    origin = json.loads(direct) if direct else {}
    revision = origin.get("vcs_info", {}).get("commit_id")
    kind = "installed_direct_url" if revision else "unversioned_installation"
    if not revision and (not origin or origin.get("dir_info", {}).get("editable")):
        # Only actual source checkouts have this relative layout; wheels are not guessed.
        repository = source.parents[3]
        if (repository / ".git").exists() and (repository / "pyproject.toml").is_file():
            revision = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
            kind = "development_checkout"
    return {
        "module_path": str(source),
        "module_sha256": digest(source),
        "repository_revision": revision,
        "revision_source": kind,
        "published_revision_verified": False,
    }


def source_page(root: Path, relative: str) -> tuple[str, bytes]:
    """Read one confined raw page in an I/O worker; e.g. ('raw/a.json') returns bytes."""
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("invalid_source_path")
    return str(path), path.read_bytes()


def verified_partitions(manifest: Path, api: str, hashes: dict[str, str]) -> list[Any]:
    """Read complete all-market partitions and verify every page before trusting negatives.

    Example: stock_st day with a successful empty final page proves no listed ST records;
    a missing page, restricted ts_code query, or nonterminal page raises ValueError.
    """
    raw = manifest.read_bytes()
    document = json.loads(raw)
    assert_public(document)
    hashes[str(manifest.resolve())] = hashlib.sha256(raw).hexdigest()
    if document.get("schema") != "return80-collection/v1":
        raise ValueError("invalid_collection_schema")
    root = manifest.parent.resolve()
    paths = [
        item["path"]
        for partition in document["partitions"]
        if partition["request"]["api"] == api
        for item in partition["files"]
    ]
    payloads = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, (relative, payload) in enumerate(
            zip(paths, executor.map(source_page, [root] * len(paths), paths), strict=True)
        ):
            payloads[relative] = payload
            if (index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "source_bytes",
                            "api": api,
                            "completed": index + 1,
                            "total": len(paths),
                        }
                    ),
                    flush=True,
                )
    result = []
    for partition in document["partitions"]:
        request = partition["request"]
        if request["api"] != api:
            continue
        parameters = request["parameters"]
        allowed = {
            "stock_basic": {"list_status"},
            "stock_st": {"trade_date"},
            "trade_cal": {"exchange", "start_date", "end_date"},
        }[api]
        if set(parameters) - allowed or not partition["files"]:
            raise ValueError("not_complete_market_query")
        rows = []
        offset = 0
        for item in partition["files"]:
            path, raw = payloads[item["path"]]
            envelope = json.loads(raw)
            assert_public(envelope)
            sha = hashlib.sha256(raw).hexdigest()
            if sha != item["sha256"]:
                raise ValueError("source_hash_mismatch")
            hashes[path] = sha
            params = envelope.get("params", envelope.get("parameters", {}))
            if (
                envelope.get("api_name") != api
                or any(params.get(k) != v for k, v in parameters.items())
                or params.get("offset", 0) != offset
            ):
                raise ValueError("source_partition_mismatch")
            page = response_records(envelope["response"])
            rows.extend(page)
            offset += len(page)
        if envelope["response"]["data"].get("has_more"):
            raise ValueError("incomplete_final_page")
        if len(rows) != partition["rows"]:
            raise ValueError("partition_row_count_mismatch")
        keys = [tuple(row.get(k) for k in request["primary_key"]) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate_partition_keys")
        if api == "stock_st" and any(
            row.get("trade_date") != parameters["trade_date"] for row in rows
        ):
            raise ValueError("stock_st_wrong_day")
        result.append((parameters, rows))
        if len(result) % 100 == 0:
            print(
                json.dumps({"stage": "verified_sources", "api": api, "completed": len(result)}),
                flush=True,
            )
    if not result:
        raise ValueError("missing_source_api:" + api)
    return result


def board(code: str) -> str:
    """Return listing-board audit bucket; for example 300001.SZ becomes ChiNext."""
    if code.endswith(".BJ"):
        return "Beijing"
    if code.startswith(("688", "689")):
        return "STAR"
    if code.startswith("30"):
        return "ChiNext"
    if code.endswith(".SH"):
        return "Shanghai_main"
    if code.endswith(".SZ"):
        return "Shenzhen_main"
    return "other"


def natural_amount20(close: Any, amount: Any) -> Any:
    """Average the latest 20 natural quote bars in yuan without padded suspension slots.

    Example: a missing close between 20 genuine quotes does not consume the window;
    a missing amount on a genuine quote makes every affected 20-bar window unknown.
    Raw Tushare amount is thousands of yuan; current bar is included, never backfilled.
    """
    output = np.full(close.shape, np.nan, dtype=np.float64)
    for column in range(close.shape[1]):
        present = np.flatnonzero(np.isfinite(close[:, column]) & (close[:, column] > 0))
        values = amount[present, column]
        if len(values) < 20:
            continue
        known = np.isfinite(values) & (values >= 0)
        sums = np.r_[0.0, np.cumsum(np.where(known, values, 0.0), dtype=np.float64)]
        counts = np.r_[0, np.cumsum(known)]
        means = (sums[20:] - sums[:-20]) * 50.0
        means[(counts[20:] - counts[:-20]) != 20] = np.nan
        output[present[19:], column] = means
    return output


def weekly_rows(calendar: list[str], days: Any) -> Any:
    """Select candidate rows on each ISO week's last observed trading day.

    Example: a holiday week ending Thursday selects Thursday. A trailing partial week
    is rejected unless its final observed date is Friday, avoiding false live origins.
    """
    weeks = [datetime.strptime(day, "%Y%m%d").isocalendar()[:2] for day in calendar]
    last = {week: index for index, week in enumerate(weeks)}
    if datetime.strptime(calendar[-1], "%Y%m%d").weekday() != 4:
        last.pop(weeks[-1])
    return np.flatnonzero(np.isin(days, list(last.values()))).astype(np.int64)


def verified_aliases(
    evidence: Path | None, panel: Path, codes: list[str], hashes: dict[str, str]
) -> dict[str, str]:
    """Validate official alias evidence and identical overlapping OHLC/amount/adjustment bars.

    Example: an old security code and its renamed code share a legal listing identity;
    old bars must all exist and match on the canonical axis before excluding duplication.
    """
    if evidence is None:
        return {}
    raw = evidence.read_bytes()
    document = json.loads(raw)
    assert_public(document)
    hashes[str(evidence.resolve())] = hashlib.sha256(raw).hexdigest()
    if document.get("schema") != "return80-security-alias-evidence/v1":
        raise ValueError("invalid_alias_evidence")
    aliases = {}
    for alias in document["aliases"]:
        old, canonical = alias["old_code"], alias["canonical_code"]
        if old == canonical or old in aliases or old not in codes or canonical not in codes:
            raise ValueError("invalid_alias_codes")
        proof = evidence.parent / alias["official_file"]
        if digest(proof) != alias["official_sha256"] or not alias["same_listing_entity"]:
            raise ValueError("invalid_official_alias_proof")
        hashes[str(proof.resolve())] = alias["official_sha256"]
        old_index, new_index = codes.index(old), codes.index(canonical)
        for name in (
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "raw_amount",
            "raw_adj_factor",
        ):
            path = panel / ("values-" + name + ".npy")
            hashes[str(path.resolve())] = digest(path)
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            present = np.isfinite(values[:, old_index])
            if not present.any() or not np.array_equal(
                values[present, old_index], values[present, new_index]
            ):
                raise ValueError("alias_market_history_conflict")
        aliases[old] = canonical
    return aliases


def build_basic_universe(
    panel: Path,
    raw: Path,
    output: Path,
    start: str = "20211018",
    end: str = "20261016",
    identity_evidence: Path | None = None,
) -> dict[str, Any]:
    """Build a reproducible basic eligibility axis plus explicit denominator/gap evidence.

    Example: an old delisted security with valid pre-delisting bars remains eligible on
    those dates; technical-risk, industry, positive-signal and ATR values are never read.
    Only already-collected public inputs are read. Existing output is never overwritten.
    """
    if output.exists():
        raise ValueError("output_exists")
    metadata = json.loads((panel / "panel.json").read_text())
    calendar, codes = metadata["calendar"], metadata["codes"]
    if calendar != sorted(set(calendar)) or len(codes) != len(set(codes)):
        raise ValueError("invalid_panel_axes")
    hashes = {str((panel / "panel.json").resolve()): digest(panel / "panel.json")}
    aliases = verified_aliases(identity_evidence, panel, codes, hashes)
    bootstrap = raw / "bootstrap/collection.json"
    master_parts = verified_partitions(bootstrap, "stock_basic", hashes)
    if {params.get("list_status") for params, _ in master_parts} != {"L", "D", "P"}:
        raise ValueError("missing_security_master_status")
    master: dict[str, Any] = {}
    for _, rows in master_parts:
        for row in rows:
            if row["ts_code"] in master:
                raise ValueError("duplicate_master_security")
            master[row["ts_code"]] = row
    official = verified_partitions(bootstrap, "trade_cal", hashes)
    official_days = sorted(
        {
            row["cal_date"]
            for _, rows in official
            for row in rows
            if row["is_open"] == 1 and calendar[0] <= row["cal_date"] <= calendar[-1]
        }
    )
    if official_days != calendar:
        raise ValueError("panel_calendar_not_official")
    outside = []
    for code, row in master.items():
        if code in codes:
            continue
        reason = (
            "excluded_board"
            if board(code) in {"Beijing", "STAR"}
            else "delisted_before_start"
            if row.get("delist_date") and row["delist_date"] <= start
            else "potentially_eligible_missing_panel"
        )
        outside.append(
            {
                "ts_code": code,
                "reason": reason,
                "list_date": row.get("list_date"),
                "delist_date": row.get("delist_date"),
            }
        )
    if any(row["reason"] == "potentially_eligible_missing_panel" for row in outside):
        raise ValueError("potentially_eligible_security_missing_panel")
    st_parts = verified_partitions(raw / "stock_st-collection.json", "stock_st", hashes)
    st_by_day: dict[str, set[str]] = {}
    for params, rows in st_parts:
        day = params["trade_date"]
        if day in st_by_day:
            raise ValueError("duplicate_stock_st_day")
        st_by_day[day] = {aliases.get(row["ts_code"], row["ts_code"]) for row in rows}
    values = {}
    for name in ("raw_close", "raw_amount"):
        path = panel / ("values-" + name + ".npy")
        hashes[str(path.resolve())] = digest(path)
        values[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        if values[name].shape != (len(calendar), len(codes)):
            raise ValueError("panel_shape_mismatch")
    means = natural_amount20(values["raw_close"], values["raw_amount"])
    print(json.dumps({"stage": "natural_amount20", "complete": True}), flush=True)
    output.mkdir(parents=True)
    np.save(output / "calendar.npy", np.asarray(calendar))
    np.save(output / "security_codes.npy", np.asarray(codes))
    np.save(output / "universe_amount.npy", means)
    boards = np.asarray([board(code) for code in codes])
    unknown_master = np.array(
        [
            code not in master
            or not master[code].get("list_date")
            or (
                master[code].get("list_status") not in {"L", "P"}
                and not master[code].get("delist_date")
            )
            for code in codes
        ]
    )
    alias_duplicate = np.array([code in aliases for code in codes])
    unknown_master &= ~alias_duplicate
    listed = np.array(
        [
            np.datetime64(datetime.strptime(master[code]["list_date"], "%Y%m%d"), "D")
            if code in master and master[code].get("list_date")
            else np.datetime64("NaT")
            for code in codes
        ]
    )
    excluded = np.isin(boards, ["Beijing", "STAR"])
    selected_days, selected_codes = [], []
    gaps: dict[str, list[tuple[int, int]]] = {
        "master_unknown": [],
        "st_unknown": [],
        "quote_unknown": [],
        "amount20_unknown": [],
    }
    master_boards = [board(code) for code in master]
    daily = []
    for day, text in enumerate(calendar):
        if not start <= text <= end:
            continue
        date = np.datetime64(datetime.strptime(text, "%Y%m%d"), "D")
        active = np.array(
            [
                not master.get(code, {}).get("delist_date") or master[code]["delist_date"] > text
                for code in codes
            ]
        )
        age = (date - listed).astype("timedelta64[D]").astype(float) > 250
        before_market = ~unknown_master & ~alias_duplicate & ~excluded & active & age
        st_known = text in st_by_day
        st = np.array([code in st_by_day.get(text, set()) for code in codes])
        close = values["raw_close"][day]
        quote_known = np.isfinite(close)
        amount_known = np.isfinite(means[day])
        eligible = (
            before_market
            & st_known
            & ~st
            & quote_known
            & (close > 1)
            & amount_known
            & (means[day] > 30_000_000)
        )
        securities = np.flatnonzero(eligible)
        selected_days.extend([day] * len(securities))
        selected_codes.extend(securities.tolist())
        reasons = {
            "master_unknown": unknown_master & ~excluded,
            "st_unknown": before_market & (not st_known),
            "quote_unknown": before_market & ~quote_known,
            "amount20_unknown": before_market & quote_known & ~amount_known,
        }
        for name, mask in reasons.items():
            gaps[name].extend((day, int(column)) for column in np.flatnonzero(mask))
        for group in sorted(set(boards) | set(master_boards)):
            mask = boards == group
            daily.append(
                {
                    "trade_date": text,
                    "board": group,
                    "full_master_securities": master_boards.count(group),
                    "alias_duplicate": int((mask & alias_duplicate).sum()),
                    "panel_securities": int(mask.sum()),
                    "listed_age_active": int((mask & before_market).sum()),
                    "st_excluded": int((mask & before_market & st).sum()),
                    "eligible": int((mask & eligible).sum()),
                    **{name: int((mask & value).sum()) for name, value in reasons.items()},
                }
            )
        if day % 100 == 0:
            print(
                json.dumps(
                    {"stage": "basic_axis", "day": text, "candidate_rows": len(selected_days)}
                ),
                flush=True,
            )
    days = np.asarray(selected_days, dtype=np.int32)
    securities = np.asarray(selected_codes, dtype=np.int32)
    origins = weekly_rows(calendar, days)
    arrays = {
        "candidate_day_indices": days,
        "candidate_security_indices": securities,
        "weekly_origin_indices": origins,
        "weekly_candidate_day_indices": days[origins],
        "weekly_candidate_security_indices": securities[origins],
    }
    for name, array in arrays.items():
        np.save(output / (name + ".npy"), array)
    gap_arrays: dict[str, Any] = {
        name: np.asarray(rows, dtype=np.int32).reshape(-1, 2) for name, rows in gaps.items()
    }
    np.savez_compressed(output / "gap_indices.npz", **gap_arrays)
    with (output / "daily-board-coverage.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(daily[0]))
        writer.writeheader()
        writer.writerows(daily)
    (output / "master-axis-exclusions.json").write_text(json.dumps(outside, indent=2) + "\n")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
    source_path = Path(__file__).resolve()
    coverage = {
        "schema": "return80-basic-universe/v1",
        "complete": True,
        "eligibility_sources_complete": not gaps["master_unknown"] and not gaps["st_unknown"],
        "historical_pit_certified": False,
        "start": start,
        "end": end,
        "calendar_days": len(calendar),
        "panel_securities": len(codes),
        "security_master_count": len(master),
        "master_status_counts": {params["list_status"]: len(rows) for params, rows in master_parts},
        "outside_panel_count": len(outside),
        "potentially_eligible_missing_panel": 0,
        "canonical_security_aliases": aliases,
        "candidate_rows": len(days),
        "weekly_rows": len(origins),
        "weekly_dates": len(set(days[origins].tolist())),
        "last_origin": calendar[int(days[origins[-1]])] if len(origins) else None,
        "last_origin_rows": int((days[origins] == days[origins[-1]]).sum()) if len(origins) else 0,
        "missing_st_dates": [
            day for day in calendar if start <= day <= end and day not in st_by_day
        ],
        "gap_counts": {name: len(rows) for name, rows in gaps.items()},
        "gap_semantics": "Unknown eligibility inputs retained in full panel/board denominators; "
        "quote_unknown does not claim suspension or confirmed missing data.",
        "rules": {
            "listing_age_calendar_days_gt": 250,
            "delist_date_exclusive": True,
            "non_st": True,
            "excluded_boards": ["688", "689", ".BJ"],
            "raw_close_gt": 1,
            "natural_quote_amount20_yuan_gt": 30000000,
            "positive_signal_filter": False,
            "technical_risk_filter": False,
            "industry_filter": False,
            "atr_filter": False,
        },
        "code": code_origin(source_path),
        "files": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
    }
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps(coverage), flush=True)
    return coverage


def main() -> None:
    """Run the offline builder; for example --panel P --raw R --output O writes O once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity-evidence", type=Path)
    args = parser.parse_args()
    build_basic_universe(
        args.panel, args.raw, args.output, identity_evidence=args.identity_evidence
    )


if __name__ == "__main__":
    main()
