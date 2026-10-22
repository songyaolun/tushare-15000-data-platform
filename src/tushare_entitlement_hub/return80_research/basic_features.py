"""Rebuild the original 47 inputs on an independently selected weekly universe."""

import argparse
import gc
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ..progress import progress_phase
from ..regime_research.panel import load_panel
from ..strategy_discovery.replay_inputs import file_digest
from .rebuild import (
    baseline_features,
    legacy_liquidity_efficiency,
    read_envelope,
    read_source_envelope,
)

REQUIRED_RAW_FIELDS = {
    "cyq_perf": {"ts_code", "trade_date", "cost_5pct", "cost_50pct", "cost_95pct", "weight_avg"},
    "daily_basic": {"ts_code", "trade_date", "close", "total_share", "total_mv"},
    "forecast_vip": {"ts_code", "ann_date", "end_date", "p_change_min", "p_change_max"},
    "fina_indicator_vip": {"ts_code", "ann_date", "end_date", "dt_netprofit_yoy", "netprofit_yoy"},
    "income_vip": {
        "ts_code",
        "ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "n_income_attr_p",
    },
}

# Forecast uses the documented conservative cap; the other limits were observed
# with deliberately oversized requests and compared with complete smaller pages.
FINANCIAL_ROW_CAPS = {"forecast_vip": 3500, "income_vip": 9000, "fina_indicator_vip": 12000}


def financial_page_keys(api: str, rows: list[dict[str, Any]], limit: int) -> set[str]:
    """Identify exact economic versions and reject possible silent response truncation.

    Example: equal profits with different update_flag values retain distinct source
    versions, as do changed profits or dates. A 9000-row income response
    to limit=20000 fails, while a full limit=2000 page can continue with an offset.
    """
    cap = FINANCIAL_ROW_CAPS.get(api)
    if cap is None:
        return set()
    if limit > cap and len(rows) >= cap:
        raise ValueError("financial_silent_row_cap")
    return {json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows}


def validated_partition(
    manifest: Path,
    partition: dict[str, Any],
    api: str,
) -> list[tuple[Path, str, list[dict[str, Any]]]]:
    """Validate declared bytes, identity, rows and continuous closed pagination before yielding.

    Example: a two-page period with offsets 0/2000 and counts 2000/3 passes;
    a wrong API, mismatched period, declared row count or open final page fails.
    Different financial values remain intact for downstream conflict handling;
    an exact economic version repeated across pages fails instead of being dropped.
    """
    request = partition["request"]
    required = set(request.get("fields", [])) | REQUIRED_RAW_FIELDS.get(api, set())
    parameters = request.get("parameters", {})
    result = []
    seen_economic_versions: set[str] = set()
    expected_offset = 0
    files = partition["files"]
    if (
        request.get("api") != api
        or not files
        or not any(key in parameters for key in ("period", "trade_date"))
    ):
        raise ValueError("basic_source_partition_identity")
    for index, item in enumerate(files):
        path = (manifest.parent / item["path"]).absolute()
        digest = file_digest(path)
        if digest != item["sha256"]:
            raise ValueError("basic_feature_raw_hash_mismatch")
        envelope = read_source_envelope(path)
        response = envelope.get("response", {})
        table = response.get("data") or {}
        fields = table.get("fields", [])
        page_parameters = envelope.get("parameters", envelope.get("params", {}))
        if (
            envelope.get("api_name") != api
            or response.get("code") != 0
            or not required <= set(fields)
            or any(page_parameters.get(key) != value for key, value in parameters.items())
        ):
            raise ValueError("basic_source_api_fields_or_partition")
        rows = read_envelope(path)
        for parameter, field in (("period", "end_date"), ("trade_date", "trade_date")):
            if parameter in parameters and any(
                row.get(field) != parameters[parameter] for row in rows
            ):
                raise ValueError("basic_source_row_partition")
        limit = page_parameters.get("limit", request.get("page_size"))
        offset = page_parameters.get("offset", 0)
        if (
            type(limit) is not int
            or limit < 1
            or type(offset) is not int
            or offset != expected_offset
            or len(rows) > limit
            or ("page_size" in request and limit != request["page_size"])
        ):
            raise ValueError("basic_source_page_offsets")
        economic_versions = financial_page_keys(api, rows, limit)
        if seen_economic_versions & economic_versions:
            raise ValueError("financial_cross_page_economic_duplicate")
        seen_economic_versions.update(economic_versions)
        # Tushare sends count=0 on nonempty successful pages; the receipt rows
        # and offset/limit/terminal page, not this placeholder, define coverage.
        last = index == len(files) - 1
        if (last and (bool(table.get("has_more")) or len(rows) >= limit)) or (
            not last and len(rows) != limit
        ):
            raise ValueError("basic_source_page_not_closed")
        expected_offset += limit
        result.append((path, digest, rows))
    if "rows" in partition and partition["rows"] != sum(len(row[2]) for row in result):
        raise ValueError("basic_source_partition_count")
    return result


def dated_records(
    sources: list[str], api: str, audit: list[dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Read explicit sources only after validating API, fields, partition and page closure.

    Example: a collection claiming income_vip over a daily page is rejected even when
    its SHA matches. Direct raw files require correct API and success and are not proof
    of complete partitions. Identical file references are deduplicated without row deletion.
    """
    seen: set[Path] = set()
    completed = 0
    for source in dict.fromkeys(sources):
        path = Path(source)
        document = read_source_envelope(path)
        batches: Iterator[list[tuple[Path, str, list[dict[str, Any]]]]]
        if "partitions" in document:
            if not any(
                partition["request"].get("api") == api for partition in document["partitions"]
            ):
                raise ValueError("basic_source_missing_api_partition")
            batches = (
                validated_partition(path, partition, api)
                for partition in document["partitions"]
                if partition["request"]["api"] == api
            )
        else:
            if (
                document.get("api_name") != api
                or document.get("response", {}).get("code") != 0
                or not REQUIRED_RAW_FIELDS.get(api, set())
                <= set((document.get("response", {}).get("data") or {}).get("fields", []))
            ):
                raise ValueError("basic_source_raw_identity")
            batches = iter([[(path.absolute(), file_digest(path), read_envelope(path))]])
        for batch in batches:
            for raw_path, digest, rows in batch:
                if raw_path in seen:
                    continue
                seen.add(raw_path)
                audit.append(dict(path=str(raw_path), api=api, sha256=digest, rows=len(rows)))
                yield from rows
                completed += 1
                if completed % 100 == 0:
                    print(
                        json.dumps(dict(stage="baseline_raw", api=api, completed=completed)),
                        flush=True,
                    )


def risk_features(values: np.ndarray) -> np.ndarray:
    """Expose the existing OR-combined Boolean risk and its missingness without bit decoding.

    Example: [0,1,NaN] maps to [[0,0],[1,0],[NaN,1]]. The stored value is not a
    bitmask. Unexpected finite integers fail rather than inventing individual risk types.
    """
    if (
        values.ndim != 1
        or np.any(np.isinf(values))
        or np.any(np.isfinite(values) & ~np.isin(values, [0, 1]))
    ):
        raise ValueError("invalid_technical_risk_indicator")
    return cast(np.ndarray, np.column_stack((values, np.isnan(values))).astype(np.float32))


def weekly_patterns(
    panel: Any, matrices: dict[str, Any], days: np.ndarray, members: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """Recalculate stock and dated L2 signal patterns without original candidate gates.

    Example: a stock with no positive signals is retained with pattern zero. Only
    requested origin days are populated; source matrices keep the complete market axis.
    Rounded total-score inversion remains identical to the original 36-column formula.
    """
    from stockops.screen.return80.ranking import signal_group_score
    from stockops.screen.return80.signals import NAMES
    from stockops.screen.return80.snapshot import sector_signal_hits

    shape = panel.values["raw_close"].shape
    patterns = np.zeros(shape, dtype=np.uint32)
    scores = np.full(shape, np.nan, dtype=np.float32)
    dates, codes = pd.to_datetime(panel.calendar), np.asarray(panel.codes)
    members_frame = pd.DataFrame(members).drop_duplicates()
    unique_days = np.unique(days)
    for count, day in enumerate(unique_days):
        hits = np.array([matrices[name][day] for name in NAMES])
        sector = sector_signal_hits(
            members_frame,
            dates[: day + 1],
            panel.codes,
            {
                "close": panel.values["raw_close"][day],
                **{name: matrices[name][day] for name in ("return5", "return1", "limit_close")},
            },
        )
        for name, selected in sector.items():
            hits[NAMES.index(name)] = np.isin(codes, list(selected))
        patterns[day] = np.sum(
            hits.astype(np.uint32) * (1 << np.arange(20, dtype=np.uint32))[:, None], axis=0
        )
        unique, inverse = np.unique(patterns[day], return_inverse=True)
        group = np.asarray([signal_group_score(int(value)) for value in unique])[inverse]
        liquidity, efficiency = legacy_liquidity_efficiency(matrices, day)
        total = np.clip(0.7 * group + 0.2 * liquidity + 0.1 * efficiency, 0, 100)
        scores[day] = np.floor(np.nan_to_num(total, nan=0) * 100 + 0.5) / 100
        if (count + 1) % 20 == 0:
            print(
                json.dumps(
                    dict(stage="weekly_patterns", completed=count + 1, total=len(unique_days))
                ),
                flush=True,
            )
    return patterns, scores


def additional_columns(
    panel: Any,
    days: np.ndarray,
    securities: np.ndarray,
    sources: dict[str, Any],
    output: Path,
    provenance: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    """Run the frozen eleven formulas with compact reason arrays and counts.

    Example: absent forecast sources leave all five forecast columns NaN. Published
    report dates activate strictly later; latest visible revisions never fill earlier days.
    These are supplier reconstructed histories, not immutable historical PIT certification.
    """
    from stockops.screen.return80.additional_features import _chip_features_from_rows
    from stockops.screen.return80.earnings_features import (
        candidate_forecast_features,
        forecast_event_features,
    )
    from stockops.screen.return80.financial_quality_artifacts import candidate_capitalization
    from stockops.screen.return80.financial_quality_features import (
        REASONS,
        candidate_report_values,
        earnings_yield,
        quality_features,
    )
    from stockops.screen.return80.forecast_event_reaction import (
        candidate_forecast_event_features,
        industry_relative_log_returns,
    )

    def records(api: str) -> Iterator[dict[str, Any]]:
        """Read one explicitly declared source; e.g. absent income yields no visible reports."""
        return dated_records(sources.get(api, []), api, provenance)

    result = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        output / "additional-features.npy", mode="w+", dtype=np.float32, shape=(len(days), 11)
    )
    audit: dict[str, Any] = {}
    with progress_phase("weekly_chips"):
        chips, audit["chips"] = _chip_features_from_rows(
            records("cyq_perf"), panel.calendar, panel.codes, days, securities
        )
        result[:, :3] = chips
        del chips
    with progress_phase("weekly_forecast"):
        forecasts = list(records("forecast_vip"))
        events = forecast_event_features(forecasts)
        result[:, 3:6] = candidate_forecast_features(
            events, panel.calendar, panel.codes, days, securities
        )
        relative = industry_relative_log_returns(
            panel.values["close"], panel.values["sector_index"], panel.values["valid"]
        )
        reaction = candidate_forecast_event_features(
            forecasts, panel.calendar, panel.codes, days, securities, relative
        )
        result[:, 6:8] = reaction.features
        np.save(output / "forecast_event_ids.npy", reaction.event_ids)
        (output / "forecast_event_keys.json").write_text(json.dumps(reaction.event_keys))
        audit["forecasts"] = dict(
            source_rows=len(forecasts),
            events=len(events),
            conflicts=sum(bool(item["conflict"]) for item in events),
        )
        del relative, reaction, forecasts, events
    with progress_phase("weekly_financial"):
        indicators, reasons, audit["indicators"] = candidate_report_values(
            records("fina_indicator_vip"),
            panel.calendar,
            panel.codes,
            days,
            securities,
            ("dt_netprofit_yoy", "netprofit_yoy"),
            cumulative=False,
        )
        result[:, 8:10] = quality_features(indicators, np.full((len(days), 2), np.nan))[:, :2]
        np.save(output / "indicator_reasons.npy", reasons)
        del indicators, reasons
        profits, reasons, audit["income"] = candidate_report_values(
            records("income_vip"),
            panel.calendar,
            panel.codes,
            days,
            securities,
            ("n_income_attr_p",),
            cumulative=True,
        )
        np.save(output / "income_reasons.npy", reasons)
        capitalization, cap_audit = candidate_capitalization(
            records("daily_basic"),
            np.asarray(panel.calendar)[days],
            np.asarray(panel.codes)[securities],
        )
        result[:, 10] = earnings_yield(profits[:, 0], capitalization)[:, 0]
        audit["capitalization"] = {
            key: value for key, value in cap_audit.items() if key != "failures"
        }
        audit["capitalization"]["failure_count"] = len(cap_audit["failures"])
        audit["reason_legend"] = list(REASONS)
    result.flush()
    return result, audit


def validated_signals(
    proof: Path, signals: Path, calendar: list[str], names: tuple[str, ...]
) -> dict[str, str]:
    """Bind signal arrays to a completed execution and its actual CSI300 raw input.

    Example: complete.json beside raw-source-hashes.json, array-comparisons.json
    and code-snapshot proves the consumed bytes; a relabeled CSI985 source fails.
    Comparison against an older incorrect run need not be equal to pass this gate.
    """
    execution = read_source_envelope(proof)
    source_path = proof.parent / "raw-source-hashes.json"
    arrays_path = proof.parent / "array-comparisons.json"
    sources = read_source_envelope(source_path)
    arrays = read_source_envelope(arrays_path)
    if (
        execution.get("complete") is not True
        or execution.get("signal_arrays") != len(arrays)
        or not set(names) <= set(arrays)
        or not execution.get("code_hashes")
    ):
        raise ValueError("basic_signals_execution_incomplete")
    hashes = {str(path): file_digest(path) for path in (proof, source_path, arrays_path)}
    benchmark = Path(execution["benchmark"])
    digest = file_digest(benchmark)
    if sources.get(str(benchmark)) != digest:
        raise ValueError("basic_signals_benchmark_hash")
    envelope = read_source_envelope(benchmark)
    rows = read_envelope(benchmark)
    parameters = envelope.get("parameters", envelope.get("params", {}))
    dates = [row.get("trade_date") for row in rows]
    if (
        envelope.get("api_name") != "index_daily"
        or envelope.get("response", {}).get("code") != 0
        or parameters.get("ts_code", "000300.SH") != "000300.SH"
        or not rows
        or any(row.get("ts_code") != "000300.SH" for row in rows)
        or len(dates) != len(set(dates))
        or not set(calendar) <= set(dates)
    ):
        raise ValueError("basic_signals_benchmark_identity_or_axis")
    hashes[str(benchmark)] = digest
    for source, expected in execution["code_hashes"].items():
        snapshot = proof.parent / "code-snapshot" / (expected + "-" + Path(source).name)
        if file_digest(snapshot) != expected:
            raise ValueError("basic_signals_code_snapshot_hash")
        hashes[str(snapshot)] = expected
    for name, record in arrays.items():
        path = signals / (name + ".npy")
        if file_digest(path) != record["new_sha256"]:
            raise ValueError("basic_signals_array_hash")
        values = np.load(path, mmap_mode="r")
        if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
            raise ValueError("basic_signals_array_contract")
        hashes[str(path)] = record["new_sha256"]
    return hashes


def build_basic_features(specification: Path, output: Path) -> Path:
    """Build weekly 47+2 inputs directly from all-security history and raw disclosures.

    Example: a specification names panel, feature_archives, signals, signals_proof,
    universe, members, original_dataset and raw_sources; writes X.npy and a manifest.
    No historical candidate matrix, trained model scores or label is read as a feature.
    A fresh output is required and incomplete evidence is retained on failure.
    """
    from stockops.screen.return80.additional_features import ADDITIONAL_FEATURE_NAMES
    from stockops.screen.return80.signals import NAMES

    spec = json.loads(specification.read_text())
    if output.exists():
        raise ValueError("fresh_basic_features_output_required")
    output.mkdir(parents=True)
    panel = load_panel(Path(spec["panel"]))
    universe = Path(spec["universe"])
    days = np.load(universe / "weekly_candidate_day_indices.npy", mmap_mode="r")
    securities = np.load(universe / "weekly_candidate_security_indices.npy", mmap_mode="r")
    if (
        days.ndim != 1
        or len(days) == 0
        or days.shape != securities.shape
        or days.dtype.kind not in "iu"
        or securities.dtype.kind not in "iu"
        or np.any(days < 0)
        or np.any(days >= len(panel.calendar))
        or np.any(securities < 0)
        or np.any(securities >= len(panel.codes))
        or not np.array_equal(np.load(universe / "calendar.npy"), panel.calendar)
        or not np.array_equal(np.load(universe / "security_codes.npy"), panel.codes)
    ):
        raise ValueError("basic_feature_axis_mismatch")
    signal_names = (
        *NAMES,
        "average_amount",
        "atr",
        "reference",
        "stop",
        "return5",
        "return1",
        "limit_close",
        "technical_risk",
    )
    signal_proof_hashes = validated_signals(
        Path(spec["signals_proof"]), Path(spec["signals"]), panel.calendar, signal_names
    )
    matrices = {
        name: np.load(Path(spec["signals"]) / (name + ".npy"), mmap_mode="r")
        for name in signal_names
    }
    provenance: list[dict[str, Any]] = []
    with progress_phase("weekly_patterns"):
        members = [row for path in spec["members"] for row in read_envelope(Path(path))]
        patterns, scores = weekly_patterns(panel, matrices, days, members)
    with progress_phase("weekly_original36"):
        values, names = baseline_features(
            panel,
            Path(spec["feature_archives"]),
            matrices,
            dict(days=days, securities=securities),
            patterns,
            scores,
        )
        np.save(output / "base-features.npy", values)
        del patterns, scores
        gc.collect()
    with progress_phase("weekly_additional11"):
        additional, audit = additional_columns(
            panel, days, securities, spec["raw_sources"], output, provenance
        )
    names = [*names, *ADDITIONAL_FEATURE_NAMES]
    original = json.loads(Path(spec["original_dataset"]).read_text())
    if names != original["feature_names"]:
        raise ValueError("original47_feature_order_mismatch")
    risk = risk_features(matrices["technical_risk"][days, securities])
    full_names = [*names, "technical_risk", "technical_risk_unknown"]
    result = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        output / "X.npy", mode="w+", dtype=np.float32, shape=(len(days), len(full_names))
    )
    result[:, :36], result[:, 36:47], result[:, 47:] = values, additional, risk
    if np.isinf(result).any():
        raise ValueError("basic_features_infinite")
    result.flush()
    np.save(output / "feature_names.npy", np.asarray(full_names))
    (output / "feature_names.json").write_text(json.dumps(full_names, indent=2) + "\n")
    known = np.isfinite(result).sum(axis=0)
    metadata = dict(
        schema="return80-basic-weekly-features/v1",
        candidate_rows=len(days),
        feature_names=full_names,
        original_features=47,
        historical_pit_certified=False,
        axis="universe/weekly_candidate_day_indices + weekly_candidate_security_indices",
        availability=dict(
            technical="origin close; backwards windows",
            reports="strictly first market session after max(ann_date,f_ann_date)",
            forecast="after max(ann_date,first_ann_date); 180 calendar day expiry",
            chips="same trade date; six complete sessions for change5",
        ),
        missing_policy="NaN retained; unknown risk is separate; no row or denominator deletion",
        risk_semantics="eight conditions OR-combined Boolean, not a bitmask",
        source_gaps=[
            api
            for api in (
                "cyq_perf",
                "forecast_vip",
                "fina_indicator_vip",
                "income_vip",
                "daily_basic",
            )
            if not spec["raw_sources"].get(api)
        ],
        known_by_feature=dict(zip(full_names, known.tolist(), strict=True)),
        missing_rate_by_feature=dict(
            zip(full_names, (1 - known / len(days)).tolist(), strict=True)
        ),
        additional_audit=audit,
        raw_sources=provenance,
        execution_specification=spec,
        signal_execution_proof=signal_proof_hashes,
        source_sha256={
            str(path): file_digest(path)
            for path in [
                specification,
                Path(__file__),
                Path(spec["panel"]) / "panel.json",
                *sorted(Path(spec["panel"]).glob("values-*.npy")),
                *[Path(spec["feature_archives"]) / (code + ".npz") for code in panel.codes],
                Path(spec["original_dataset"]),
                *[
                    universe / name
                    for name in (
                        "weekly_candidate_day_indices.npy",
                        "weekly_candidate_security_indices.npy",
                        "weekly_origin_indices.npy",
                    )
                ],
                *[Path(spec["signals"]) / (name + ".npy") for name in signal_names],
                *map(Path, spec["members"]),
            ]
        },
        files={path.name: file_digest(path) for path in output.iterdir() if path.is_file()},
    )
    destination = output / "manifest.json"
    destination.write_text(json.dumps(metadata, indent=2) + "\n")
    coverage = {
        key: metadata[key]
        for key in (
            "candidate_rows",
            "feature_names",
            "known_by_feature",
            "missing_rate_by_feature",
            "source_gaps",
            "missing_policy",
        )
    }
    (output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    complete = dict(
        status="complete",
        rows=len(days),
        columns=len(full_names),
        producer_sha256=file_digest(Path(__file__)),
        historical_pit_certified=False,
        availability=metadata["availability"],
        source_manifest="manifest.json",
        files={path.name: file_digest(path) for path in output.iterdir() if path.is_file()},
    )
    (output / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    return destination


def main() -> None:
    """Run the offline preparation entry; example --specification spec.json --output baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(build_basic_features(arguments.specification, arguments.output), flush=True)


if __name__ == "__main__":
    main()


def collect_financial_period(
    client: Any, api: str, period: str, fields: list[str], output: Path, *, page_size: int = 2000
) -> Path:
    """Collect one complete disclosure partition while preserving ambiguous versions.

    Example: two rows sharing report identity but different profits survive raw ingestion;
    the financial feature builder closes the conflicting field. Exact duplicate full pages,
    nonmatching periods, missing fields, repeated economic versions across pages and
    the 100-page ceiling fail. A short final page below the real cap establishes bounded
    pagination closure; it does not establish provider-wide completeness or historical PIT.
    """
    from datetime import UTC, datetime

    if api not in ("forecast_vip", "income_vip") or page_size < 1:
        raise ValueError("invalid_financial_collection_request")
    output.mkdir(parents=True, exist_ok=True)
    request = dict(api=api, parameters=dict(period=period), fields=fields, page_size=page_size)
    seen_pages: set[str] = set()
    seen_economic_versions: set[str] = set()
    pages = []
    for page_index in range(100):
        page = client.fetch(
            api,
            dict(
                period=period,
                fields=",".join(fields),
                offset=page_index * page_size,
                limit=page_size,
            ),
            datetime.now(UTC).isoformat(),
        )
        if (
            not set(fields) <= set(page.fields)
            or len(page.rows) > page_size
            or any(row.get("end_date") != period for row in page.rows)
        ):
            raise ValueError("financial_page_contract_failure")
        import hashlib

        digest = hashlib.sha256(json.dumps(page.rows, sort_keys=True).encode()).hexdigest()
        if len(page.rows) == page_size and digest in seen_pages:
            raise ValueError("financial_repeated_full_page")
        seen_pages.add(digest)
        economic_versions = financial_page_keys(api, page.rows, page_size)
        if seen_economic_versions & economic_versions:
            raise ValueError("financial_cross_page_economic_duplicate")
        seen_economic_versions.update(economic_versions)
        pages.append(dict(path=str(page.raw.path.resolve()), sha256=file_digest(page.raw.path)))
        print(
            json.dumps(
                dict(
                    stage="financial_collection",
                    api=api,
                    period=period,
                    page=page_index + 1,
                    rows=len(page.rows),
                )
            ),
            flush=True,
        )
        if len(page.rows) < page_size:
            receipt = output / "collection.json"
            receipt.write_text(
                json.dumps(
                    dict(
                        schema="return80-collection/v1",
                        historical_pit_certified=False,
                        partitions=[dict(request=request, files=pages)],
                    ),
                    indent=2,
                )
                + "\n"
            )
            return receipt
    raise ValueError("financial_page_ceiling")
