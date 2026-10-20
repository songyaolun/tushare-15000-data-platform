"""Reproducible raw-cache to full-candidate quality/valuation artifacts; no training."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .financial_quality_features import (
    QUALITY_FEATURE_NAMES,
    REASONS,
    VALUATION_FEATURE_NAMES,
    FloatArray,
    candidate_report_values,
    earnings_yield,
    finite_number,
    quality_features,
)

AXIS_SHA256 = "6977287e2f5db8eca533793fcc8e044e845fce0a01f29bb7c63810083ffa74d8"


def file_sha256(path: Path) -> str:
    """Hash a local evidence file; e.g. empty bytes yield the standard SHA256 empty digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_records(
    roots: Sequence[Path], api_name: str, required: tuple[str, ...],
    provenance: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Stream complete declared raw partitions and append safe provenance metadata.

    roots contain api_name/*.json envelopes with response.data.fields/items. Failed,
    truncated, malformed, wrong-period or discontinuous pages raise instead of silently
    skipping source rows. Only explicit public parameters/observation timestamps are
    retained, never a token or opaque transport configuration.
    Example: one daily_basic page yields its field dictionaries without unit conversion.
    """
    paths = sorted({path for root in roots for path in (root / api_name).glob("*.json")})
    if not paths:
        raise ValueError(f"missing raw source: {api_name}")
    partitions: dict[tuple[str, str], dict[int, tuple[int, int]]] = defaultdict(dict)
    for path in paths:
        envelope = json.loads(path.read_text())
        response = envelope.get("response", {})
        table = response.get("data", {}) or {}
        fields, items = table.get("fields", []), table.get("items", [])
        parameters = envelope.get("parameters", {})
        if (
            envelope.get("api_name") != api_name or response.get("code") != 0
            or envelope.get("quality", {}).get("status") != "passed"
            or not isinstance(fields, list) or len(set(fields)) != len(fields)
            or not set(required).issubset(fields) or not isinstance(items, list)
            or any(not isinstance(row, list) or len(row) != len(fields) for row in items)
        ):
            raise ValueError(f"invalid raw table: {api_name}")
        partition_field = "period" if "period" in parameters else "trade_date"
        partition_value = str(parameters.get(partition_field, ""))
        if not partition_value:
            raise ValueError("missing declared source partition")
        offset, limit = int(parameters.get("offset", 0)), int(parameters.get("limit", 6000))
        if offset < 0 or limit <= 0 or len(items) > limit:
            raise ValueError("invalid raw pagination")
        page = (len(items), limit)
        partition = partitions[(partition_field, partition_value)]
        if offset in partition:
            raise ValueError("duplicate raw partition offset")
        partition[offset] = page
        if table.get("has_more") and len(items) < limit:
            raise ValueError("short raw page still declares more")
        provenance.append({
            "path": str(path), "sha256": file_sha256(path), "api_name": api_name,
            "partition": {partition_field: partition_value}, "offset": offset,
            "limit": limit, "rows": len(items), "observed_at": envelope.get("observed_at"),
        })
        row_date_field = "end_date" if partition_field == "period" else "trade_date"
        for values in items:
            record = dict(zip(fields, values, strict=True))
            if record.get(row_date_field) != partition_value:
                raise ValueError("source row differs from declared partition")
            yield record
    for partition in partitions.values():
        expected_offset = 0
        for offset, (count, limit) in sorted(partition.items()):
            if offset != expected_offset:
                raise ValueError("raw pagination gap or overlap")
            if offset != max(partition) and count < limit:
                raise ValueError("short interior raw page")
            expected_offset += limit
        if count >= limit:
            raise ValueError("last raw page may be truncated")


def candidate_capitalization(
    records: Iterable[Mapping[str, Any]],
    candidate_dates: Sequence[str], candidate_codes: Sequence[str],
) -> tuple[FloatArray, dict[str, Any]]:
    """Use same-day unadjusted close × total_share × 10000; audit against total_mv × 10000.

    Tushare daily_basic prices are yuan and total_share/total_mv are ten-thousand
    shares/yuan. Require all positive finite and market value parity within max(10
    yuan, 1ppm), accommodating source rounding. No prior date carry is allowed.
    Conflicting duplicate security-dates stay unknown. Example: close=10,
    total_share=100, total_mv=1000 yields market cap=10000000 yuan.
    """
    if len(candidate_dates) != len(candidate_codes):
        raise ValueError("capitalization candidate axes differ")
    positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(candidate_dates, candidate_codes, strict=True)):
        positions[key].append(index)
    result = np.full(len(candidate_dates), np.nan)
    seen: dict[tuple[str, str], tuple[float | None, ...]] = {}
    conflicts: set[tuple[str, str]] = set()
    audit: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        audit["raw_rows"] += 1
        key = (str(record.get("trade_date", "")), str(record.get("ts_code", "")))
        if key not in positions:
            continue
        numbers = tuple(finite_number(record.get(field)) for field in
                        ("close", "total_share", "total_mv"))
        identity = tuple(value if np.isfinite(value) else None for value in numbers)
        if key in seen and identity != seen[key]:
            conflicts.add(key)
        seen[key] = identity
        if key in conflicts:
            result[positions[key]] = np.nan
            failures.append({"trade_date": key[0], "ts_code": key[1], "reason": "conflict"})
            continue
        close, shares, reported_value = numbers
        if not all(np.isfinite(value) and value > 0 for value in numbers):
            audit["invalid_positive_fields"] += 1
            failures.append({"trade_date": key[0], "ts_code": key[1], "reason": "invalid_fields"})
            continue
        capitalization, reference = close * shares * 10000, reported_value * 10000
        if not np.isfinite(capitalization) or not np.isfinite(reference):
            audit["nonfinite_market_cap"] += 1
            failures.append({"trade_date": key[0], "ts_code": key[1], "reason": "overflow"})
            continue
        if abs(capitalization - reference) > max(10., reference * 1e-6):
            audit["market_value_parity_failures"] += 1
            failures.append({"trade_date": key[0], "ts_code": key[1], "reason": "parity",
                             "close": close, "total_share": shares, "total_mv": reported_value,
                             "computed_yuan": capitalization, "reported_yuan": reference})
            continue
        result[positions[key]] = capitalization
        audit["matched_source_rows"] += 1
        if len(samples) < 8:
            samples.append({"trade_date": key[0], "ts_code": key[1], "close_yuan": close,
                            "total_share_ten_thousand": shares,
                            "total_mv_ten_thousand_yuan": reported_value,
                            "computed_cap_yuan": capitalization})
    return result, {**dict(audit), "conflicting_security_dates": len(conflicts),
                    "candidate_rows": len(result), "known": int(np.isfinite(result).sum()),
                    "unit_parity_examples": samples, "failures": failures}


def coverage_by_group(features: FloatArray, labels: Sequence[str]) -> dict[str, Any]:
    """Count original rows and finite values in every group; ['2021','2021'] has denominator 2."""
    if len(features) != len(labels):
        raise ValueError("coverage labels differ from candidate rows")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    return {label: {"candidate_rows": len(rows),
                    "known": np.isfinite(features[rows]).sum(axis=0).tolist()}
            for label, rows in sorted(groups.items())}


def build_financial_artifacts(
    raw_roots: Sequence[Path], panel_root: Path, axis_path: Path, output_root: Path,
    *, source_revision: str, mature_mask_path: Path, expected_rows: int = 520690,
    expected_mature_rows: int = 490655, expected_axis_sha256: str = AXIS_SHA256,
) -> dict[str, Any]:
    """Create quality/valuation NPY, manifests and audits from declared cached history.

    Example: roots=[cache/raw,supplement/raw], panel_root=panel, axis_path=candidates.npz
    writes output_root/{quality,valuation}/{candidate-features.npy,manifest.json,audit.json}.
    No training, replay, candidate filtering, credential access or network occurs here.
    Each family is independent; empty data or axis mismatch raises before delivery.
    """
    if file_sha256(axis_path) != expected_axis_sha256:
        raise ValueError("candidate axis hash mismatch")
    metadata = json.loads((panel_root / "panel.json").read_text())
    calendar, codes = metadata["calendar"], metadata["codes"]
    with np.load(axis_path, allow_pickle=False) as axis:
        days, securities = axis["days"], axis["securities"]
    mature = np.load(mature_mask_path, allow_pickle=False)
    if (len(days) != expected_rows or mature.shape != days.shape or mature.dtype != np.bool_
            or int(mature.sum()) != expected_mature_rows):
        raise ValueError("candidate count or mature axis mismatch")
    provenance: list[dict[str, Any]] = []
    inputs: dict[str, FloatArray] = {}
    input_reasons: dict[str, NDArray[np.int8]] = {}
    source_audits: dict[str, dict[str, int]] = {}
    specifications = (
        ("fina_indicator_vip", ("dt_netprofit_yoy", "netprofit_yoy"), False, False),
        ("cashflow_vip", ("n_cashflow_act", "net_profit"), True, True),
        ("income_vip", ("n_income_attr_p",), True, False),
    )
    for api_name, fields, cumulative, industrial_only in specifications:
        required = ("ts_code", "ann_date", "end_date", *fields)
        if cumulative:
            required += ("f_ann_date", "report_type", "comp_type")
        report_result = candidate_report_values(
            raw_records(raw_roots, api_name, required, provenance), calendar, codes,
            days, securities, fields, cumulative=cumulative, industrial_only=industrial_only,
        )
        inputs[api_name], input_reasons[api_name], source_audits[api_name] = report_result
    candidate_dates = [calendar[day] for day in days]
    candidate_codes = [codes[security] for security in securities]
    capitalization, market_audit = candidate_capitalization(
        raw_records(raw_roots, "daily_basic", ("ts_code", "trade_date", "close", "total_share",
                                               "total_mv"), provenance),
        candidate_dates, candidate_codes,
    )
    quality = quality_features(inputs["fina_indicator_vip"], inputs["cashflow_vip"])
    valuation = earnings_yield(inputs["income_vip"][:, 0], capitalization)
    quality_reasons = np.column_stack([
        input_reasons["fina_indicator_vip"][:, 0],
        np.maximum(input_reasons["fina_indicator_vip"][:, 0],
                   input_reasons["fina_indicator_vip"][:, 1]),
        np.maximum(input_reasons["cashflow_vip"][:, 0], input_reasons["cashflow_vip"][:, 1]),
    ])
    quality_reasons[(quality_reasons[:, 2] == 0) & (inputs["cashflow_vip"][:, 1] <= 0), 2] = 8
    valuation_reasons = input_reasons["income_vip"].copy()
    valuation_reasons[(valuation_reasons[:, 0] == 0) & ~np.isfinite(capitalization), 0] = 9
    sectors = np.load(panel_root / "values-sector_index.npy", mmap_mode="r")[days, securities]
    year_labels = [day[:4] for day in candidate_dates]
    board_labels = ["mainboard" if code.startswith(("60", "00")) else
                    "star" if code.startswith("688") else
                    "chinext" if code.startswith(("300", "301")) else "other"
                    for code in candidate_codes]
    sector_labels = [str(int(value)) if np.isfinite(value) else "unknown" for value in sectors]
    result: dict[str, Any] = {}
    for family, names, features, reasons in (
        ("quality", QUALITY_FEATURE_NAMES, quality, quality_reasons),
        ("valuation", VALUATION_FEATURE_NAMES, valuation, valuation_reasons),
    ):
        reasons[(reasons == 0) & ~np.isfinite(features)] = 7
        directory = output_root / family
        directory.mkdir(parents=True, exist_ok=True)
        feature_path = directory / "candidate-features.npy"
        np.save(feature_path, features, allow_pickle=False)
        np.save(directory / "candidate-reasons.npy", reasons, allow_pickle=False)
        provenance_path = directory / "sources.json"
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        manifest = {
            "family": family, "feature_names": names, "shape": list(features.shape),
            "dtype": str(features.dtype), "source_revision": source_revision,
            "candidate_axis_path": str(axis_path), "candidate_axis_sha256": file_sha256(axis_path),
            "candidate_features_sha256": file_sha256(feature_path),
            "mature_rows": int(mature.sum()),
            "mature_mask_path": str(mature_mask_path),
            "mature_mask_sha256": file_sha256(mature_mask_path),
            "panel_sha256": file_sha256(panel_root / "panel.json"),
            "source_manifest_sha256": file_sha256(provenance_path),
            "implementation_sha256": file_sha256(Path(__file__).with_name(
                "financial_quality_features.py")),
            "artifact_builder_sha256": file_sha256(Path(__file__)),
            "historical_pit_certified": False, "history_mode": "supplier_reconstructed",
            "visibility": "first trading-session 19:30 strictly after max(ann_date,f_ann_date)",
            "missing": "NaN; no candidate removal or older-report fallback",
            "valuation_units": "parent profit yuan / (raw close yuan * total_share * 10000)",
            "quality_units": "growth percentage /100; same-table consolidated cash and profit yuan",
            "training_or_replay_performed": False,
        }
        audit = {
            "feature_names": names, "candidate_rows": len(features),
            "known": np.isfinite(features).sum(axis=0).tolist(),
            "mature_known": np.isfinite(features[mature]).sum(axis=0).tolist(),
            "by_year": coverage_by_group(features, year_labels),
            "by_board": coverage_by_group(features, board_labels),
            "by_historical_sector": coverage_by_group(features, sector_labels),
            "candidate_reason_counts": {
                name: {reason: int((reasons[:, index] == code).sum())
                       for code, reason in enumerate(REASONS)}
                for index, name in enumerate(names)
            },
            "unknown_security_codes_by_reason": {
                name: {reason: sorted({candidate_codes[row] for row in
                                      np.flatnonzero(reasons[:, index] == code)})
                       for code, reason in enumerate(REASONS) if code != 0}
                for index, name in enumerate(names)
            },
            "source_audits": source_audits, "market_capitalization": market_audit,
            "source_partitions": {api: sorted({str(item["partition"].get("period", ""))
                                                 for item in provenance if item["api_name"] == api})
                                  for api, *_ in specifications},
            "limitations": [
                "Supplier current historical versions are not contemporaneous PIT archives.",
                "Conflicts and missing TTM components remain NaN, never select update_flag.",
                "Raw page completeness certifies requested partitions, not every company's filing.",
                "Only comp_type=1 contributes cash; losses remain valid in earnings yield.",
            ],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        result[family] = {"manifest": manifest, "known": audit["known"]}
    return result
