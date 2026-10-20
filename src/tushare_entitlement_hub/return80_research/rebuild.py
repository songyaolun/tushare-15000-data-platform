"""Rebuild the original research candidate axis and features from explicit local sources."""

import hashlib
import json
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ..progress import progress_phase
from ..raw import assert_public
from ..regime_research.accounting import load_distributions
from ..regime_research.panel import load_panel
from ..research_history.collection import response_records
from ..strategy_discovery.contracts import DEFAULT_POLICY
from ..strategy_discovery.nonlinear_scoring import nonlinear_input_features
from ..strategy_discovery.replay_inputs import file_digest, write_replay_inputs
from ..strategy_discovery.sector import _sector_features
from ..strategy_discovery.trade_labels import independent_trade_label
from ..strategy_discovery.training_weights import daily_equal_training_weights
from .dataset import save_dataset

SIGNAL_FIELDS = (
    "average_amount",
    "atr",
    "reference",
    "stop",
    "universe_amount",
    "return5",
    "return1",
    "limit_close",
    "technical_risk",
)
SUPPLEMENT_FIELDS = {
    "daily": ("pre_close",),
    "daily_basic": ("volume_ratio",),
    "moneyflow": ("buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"),
    "stock_st": ("is_st",),
}


_SOURCE_HASHES: ContextVar[dict[str, str] | None] = ContextVar(
    "rebuild_source_hashes", default=None
)


def read_source_envelope(path: Path) -> dict[str, Any]:
    """Parse public bytes once and audit exactly those bytes, rejecting source revision drift.

    Example: a repeated source changing during reconstruction fails before dataset publication.
    Credentials are rejected before hashing; the context belongs to one reconstruction call.
    """
    raw = path.read_bytes()
    envelope: dict[str, Any] = json.loads(raw)
    assert_public(envelope)
    hashes = _SOURCE_HASHES.get()
    if hashes is not None:
        key, digest = str(path.resolve()), hashlib.sha256(raw).hexdigest()
        if key in hashes and hashes[key] != digest:
            raise ValueError("rebuild_source_changed_during_read")
        hashes[key] = digest
    return envelope


def read_envelope(path: Path) -> list[dict[str, Any]]:
    """Read an audited public response; example fields=['x'],items=[[1]] returns [{'x':1}]."""
    return response_records(read_source_envelope(path)["response"])


def source_records(roots: list[Path], api: str) -> Iterator[dict[str, Any]]:
    """Stream declared API directories without executing any cached Python.

    Example: roots=[source/raw] reads source/raw/daily/*.json. Missing API fails;
    identical duplicate rows are handled by the consumer, conflicting versions are explicit.
    """
    paths = sorted({path for root in roots for path in (root / api).glob("*.json")})
    if not paths:
        raise ValueError("missing_source_api:" + api)
    for path in paths:
        yield from read_envelope(path)


def supplemental_arrays(panel: Any, roots: list[Path]) -> dict[str, Any]:
    """Align original units by date and security; conflicting same-key rows fail closed.

    Example: an explicitly collected empty stock_st day produces zero flags; a missing
    day stays NaN. Moneyflow source units stay 10,000 yuan until signal construction.
    """
    dates = {day: index for index, day in enumerate(panel.calendar)}
    codes = {code: index for index, code in enumerate(panel.codes)}
    shape = len(dates), len(codes)
    arrays = {}
    for api, names in SUPPLEMENT_FIELDS.items():
        current = {name: np.full(shape, np.nan) for name in names}
        paths = sorted({path for root in roots for path in (root / api).glob("*.json")})
        if not paths:
            raise ValueError("missing_source_api:" + api)
        seen: dict[Any, Any] = {}
        for file_index, path in enumerate(paths):
            envelope = read_source_envelope(path)
            rows = response_records(envelope["response"])
            day = envelope.get("parameters", envelope.get("params", {})).get("trade_date")
            if api == "stock_st" and day in dates:
                # Only a completed, successful source partition proves the negative list.
                if day not in seen:
                    current["is_st"][dates[day]] = 0
                    seen[day] = True
            for row in rows:
                source_date = row.get("trade_date", day)
                if source_date not in dates or row["ts_code"] not in codes:
                    continue
                if day and source_date != day:
                    raise ValueError("source_partition_date_mismatch")
                index, column = dates[source_date], codes[row["ts_code"]]
                values = tuple(1 if name == "is_st" else row.get(name) for name in names)
                key = source_date, row["ts_code"]
                if key in seen and seen[key] != values:
                    raise ValueError("conflicting_supplemental_rows")
                seen[key] = values
                for name, value in zip(names, values, strict=True):
                    current[name][index, column] = np.nan if value is None else value
            if (file_index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "supplemental_inputs",
                            "api": api,
                            "completed": file_index + 1,
                            "total": len(paths),
                        }
                    ),
                    flush=True,
                )
        arrays.update(current)
    return arrays


def build_signals(
    panel: Any, source: Path, roots: list[Path], benchmark: Path, output: Path
) -> tuple[dict[str, Any], Any]:
    """Reuse the pinned production stock_signals on original natural-history bars.

    Example: raw vol=100 hands becomes 10,000 shares; raw amount=1000 becomes
    1,000,000 yuan. Output matrices preserve missing market slots and legacy NaNs.
    """
    from stockops.screen.return80.signals import NAMES, stock_signals

    arrays = supplemental_arrays(panel, roots)
    dates = pd.to_datetime(panel.calendar)
    benchmark_frame = pd.DataFrame(read_envelope(benchmark))
    benchmark_frame["date"] = pd.to_datetime(benchmark_frame.trade_date)
    benchmark_frame["volume"] = benchmark_frame.vol
    benchmark_frame = benchmark_frame.sort_values("date")
    if benchmark_frame["date"].duplicated().any() or not set(dates) <= set(benchmark_frame.date):
        raise ValueError("incomplete_or_duplicate_benchmark")
    shape = len(dates), len(panel.codes)
    output.mkdir(parents=True)
    matrices = {
        name: np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            output / (name + ".npy"),
            mode="w+",
            dtype="bool" if name in NAMES else "float64",
            shape=shape,
        )
        for name in (*NAMES, *SIGNAL_FIELDS)
    }
    for name, matrix in matrices.items():
        matrix[:] = False if name in NAMES else np.nan
    for column, code in enumerate(panel.codes):
        with np.load(source / (code + ".npz"), allow_pickle=False) as archived:
            volume = archived["raw_vol"]
        values = panel.values
        present = np.flatnonzero(
            np.isfinite(values["raw_open"][:, column])
            & np.isfinite(values["raw_close"][:, column])
            & (values["raw_adj_factor"][:, column] > 0)
        )
        if len(present) < 26:
            continue
        frame = pd.DataFrame(
            {
                name: values["raw_" + name][present, column]
                for name in ("open", "high", "low", "close", "adj_factor", "up_limit")
            },
            index=dates[present],
        )
        frame["volume"] = volume[present] * 100
        frame["amount"] = values["raw_amount"][present, column] * 1000
        frame["pre_close"] = arrays["pre_close"][present, column]
        frame["volume_ratio"] = arrays["volume_ratio"][present, column]
        frame["main_amount"] = (
            arrays["buy_lg_amount"][present, column]
            - arrays["sell_lg_amount"][present, column]
            + arrays["buy_elg_amount"][present, column]
            - arrays["sell_elg_amount"][present, column]
        ) * 10000
        result = stock_signals(  # type: ignore[no-untyped-call]
            frame, benchmark_frame, dates
        )
        for name in matrices:
            matrices[name][present, column] = result[name].to_numpy()
        if (column + 1) % 100 == 0:
            print(
                json.dumps(
                    {"stage": "stock_signals", "completed": column + 1, "total": len(panel.codes)}
                ),
                flush=True,
            )
    for matrix in matrices.values():
        matrix.flush()
    return matrices, arrays["is_st"]


def candidate_axis(
    panel: Any, matrices: dict[str, Any], st_flags: Any, master_rows: Any, member_rows: Any
) -> tuple[dict[str, Any], Any, Any, Any]:
    """Rebuild the original universe, L2 signals and positive/watch gates, before common gates.

    Example: STAR/BJ are excluded, ChiNext remains in the original research scope;
    listing age is >250 calendar days. Common failures retain rows for label status zero.
    """
    from stockops.screen.return80.ranking import signal_group_score
    from stockops.screen.return80.signals import NAMES
    from stockops.screen.return80.snapshot import sector_signal_hits

    policy = json.loads(files("stockops.screen.return80").joinpath("policy.json").read_text())
    codes, dates = np.asarray(panel.codes), pd.to_datetime(panel.calendar)
    master: dict[str, Any] = {}
    for row in master_rows:
        code = row["ts_code"]
        if code in master and master[code] != row:
            raise ValueError("conflicting_security_master")
        master[code] = row
    ipo = pd.to_datetime([master.get(code, {}).get("list_date") for code in codes])
    end = pd.to_datetime([master.get(code, {}).get("delist_date") or None for code in codes])
    members = pd.DataFrame(member_rows).drop_duplicates()
    shape = panel.values["raw_close"].shape
    eligible, patterns = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.uint32)
    old_scores = np.full(shape, np.nan, dtype=np.float32)
    positive = [
        index
        for index, signal in enumerate(policy["signals"])
        if signal["role"] == "positive_evidence"
    ]
    groups = policy["factor_groups"].values()
    watch_ids = policy["watch_signals"]
    for day, date in enumerate(dates):
        if panel.calendar[day] < "20211018":
            continue
        universe = (date - ipo).days.to_numpy() > 250
        universe &= end.isna() | (end > date)
        universe &= np.array(
            [not code.startswith(("688", "689")) and not code.endswith(".BJ") for code in codes]
        )
        universe &= np.array(
            [
                master.get(code, {}).get("list_status") == "L"
                or bool(master.get(code, {}).get("delist_date"))
                for code in codes
            ]
        )
        universe &= (
            (panel.values["raw_close"][day] > 1)
            & (matrices["universe_amount"][day] > 30000000)
            & (st_flags[day] == 0)
        )
        hits = np.array([matrices[name][day] for name in NAMES])
        sector = sector_signal_hits(  # type: ignore[no-untyped-call]
            members,
            dates[: day + 1],
            panel.codes,
            {
                "close": panel.values["raw_close"][day],
                **{name: matrices[name][day] for name in ("return5", "return1", "limit_close")},
            },
        )
        for name, selected in sector.items():
            hits[NAMES.index(name)] = np.isin(codes, list(selected))
        independent = sum(
            np.any([hits[NAMES.index(name)] for name in group["signal_ids"]], axis=0)
            for group in groups
        )
        watch = np.any([hits[NAMES.index(name)] for name in watch_ids], axis=0) & (independent < 2)
        qualified = universe & np.any(hits[positive], axis=0) & ~watch
        for name in ("average_amount", "atr", "reference", "stop"):
            qualified &= np.isfinite(matrices[name][day]) & (matrices[name][day] > 0)
        patterns[day] = np.sum(
            hits.astype(np.uint32) * (1 << np.arange(20, dtype=np.uint32))[:, None], axis=0
        )
        unique, inverse = np.unique(patterns[day], return_inverse=True)
        group_score = np.array([signal_group_score(int(pattern)) for pattern in unique])[inverse]
        liquidity, efficiency = legacy_liquidity_efficiency(matrices, day)
        total = np.clip(0.7 * group_score + 0.2 * liquidity + 0.1 * efficiency, 0, 100)
        old_scores[day] = np.floor(np.nan_to_num(total, nan=0) * 100 + 0.5) / 100
        eligible[day] = qualified
        if day % 100 == 0:
            print(
                json.dumps({"stage": "candidate_axis", "completed": day + 1, "total": len(dates)}),
                flush=True,
            )
    days, securities = np.where(eligible & (matrices["technical_risk"] == 0))
    values = panel.values
    width = (values["raw_up_limit"] - values["raw_down_limit"]) / (
        values["raw_up_limit"] + values["raw_down_limit"]
    )
    common = (
        (values["valid"] == 1)
        & np.isfinite(values["beta"])
        & np.isfinite(values["capitalization"])
        & (values["sector_index"] >= 0)
        & (values["raw_amount"] >= 20000)
        & (width > 0.07)
    )[days, securities]
    return (
        {"days": days, "securities": securities, "common": common},
        eligible,
        patterns,
        old_scores,
    )


def legacy_liquidity_efficiency(matrices: dict[str, Any], positions: Any) -> tuple[Any, Any]:
    """Preserve the original vectorized liquidity/stop-distance calculation and rounding inputs.

    Example: ADV20=20m gives liquidity=0; invalid stop geometry gives efficiency=0.
    """
    average, atr, reference, stop = (
        matrices[name][positions] for name in ("average_amount", "atr", "reference", "stop")
    )
    liquidity = 100 * np.clip(np.log(np.maximum(average, 1) / 20000000) / np.log(25), 0, 1)
    efficiency = 100 * np.minimum(
        1,
        np.divide(
            atr,
            reference - stop,
            out=np.zeros_like(atr),
            where=(atr > 0) & (stop > 0) & (reference > stop),
        ),
    )
    return liquidity, efficiency


def baseline_features(
    panel: Any,
    source: Path,
    matrices: dict[str, Any],
    axis: dict[str, Any],
    patterns: Any,
    old_scores: Any,
) -> tuple[Any, Any]:
    """Assemble the original 36 float32 columns, keeping rounded old-score recovery.

    Example: the 20 binary signal columns follow 16 numeric features; raw labels,
    dates, identifiers and present-day membership are never used as feature columns.
    """
    from stockops.screen.return80.signals import NAMES

    needed = (
        "return1",
        "return5",
        "return20",
        "extension_atr",
        "close_location",
        "boll_position",
        "kdj_j",
        "above_ma20_age",
    )
    shape = panel.values["raw_close"].shape
    features = dict(panel.values)
    missing = [name for name in needed if name not in features]
    for name in missing:
        features[name] = np.full(shape, np.nan)
    for column, code in enumerate(panel.codes):
        with np.load(source / (code + ".npz"), allow_pickle=False) as archive:
            for name in missing:
                features[name][:, column] = archive[name]
    industry = _sector_features(features)
    positions = axis["days"], axis["securities"]
    liquidity, efficiency = legacy_liquidity_efficiency(matrices, positions)
    group = (old_scores[positions].astype(float) - 0.2 * liquidity - 0.1 * efficiency) / 0.7
    raw = {
        "signal_group_score": group,
        "liquidity_score": liquidity,
        "risk_efficiency_score": efficiency,
        **{name: features[name][positions] for name in needed},
        "atr_fraction": matrices["atr"][positions] / matrices["reference"][positions],
        "sector_breadth": industry["breadth"][positions],
        "sector_diffusion": industry["diffusion"][positions],
        "sector_momentum20": industry["momentum20"][positions],
        "relative_return20": features["return20"][positions] - industry["momentum20"][positions],
    }
    hits = ((patterns[positions][:, None] & (1 << np.arange(20))) != 0).astype(float)
    values, names = nonlinear_input_features(raw, hits, NAMES)
    return values.astype(np.float32), names


def build_labels(panel: Any, axis: dict[str, Any], events: list[Any]) -> Any:
    """Run the unchanged independent 50,000 yuan 4/8/30 labels, never as feature inputs.

    Example: a failed common gate receives status0 and NaN outcomes; a completed
    trade receives status3, net return, holding sessions and actual exit index.
    """
    policy = replace(DEFAULT_POLICY, take_profit=0.04, stop_loss=0.08, maximum_holding=30)
    by_security: dict[int, list[Any]] = {}
    for event in events:
        by_security.setdefault(event.security, []).append(event)
    labels = np.full((len(axis["days"]), 4), np.nan)
    for index, (day, security) in enumerate(zip(axis["days"], axis["securities"], strict=True)):
        if not axis["common"][index]:
            labels[index, 3] = 0
        else:
            relevant = [
                event for event in by_security.get(int(security), []) if event.record >= day + 1
            ]
            outcome = independent_trade_label(
                panel, int(day), int(security), relevant, policy, budget=50000, maximum_gap=0.05
            )
            labels[index, 3] = {"unfilled": 1, "censored": 2, "closed": 3}[outcome["status"]]
            if outcome["status"] == "closed":
                labels[index, :3] = [
                    outcome["return"],
                    outcome["holding_sessions"],
                    outcome["exit"],
                ]
        if (index + 1) % 10000 == 0:
            print(
                json.dumps({"stage": "labels", "completed": index + 1, "total": len(labels)}),
                flush=True,
            )
    return labels


def rebuild_matrices(specification: Path, output: Path) -> Path:
    """Run one isolated source audit; example a changed raw file rejects the whole rebuild."""
    token = _SOURCE_HASHES.set({})
    try:
        return _run_rebuild(specification, output)
    finally:
        _SOURCE_HASHES.reset(token)


def _run_rebuild(specification: Path, output: Path) -> Path:
    """Rebuild 47 columns, labels and portable replay from explicitly named source roots.

    Example: spec declares panel, source_dataset, raw_roots, benchmark, members and
    distributions; relative paths resolve next to spec, not to the original developer HOME.
    A fresh output is mandatory. Completed intermediates remain for failure diagnosis.
    """
    from stockops.screen.return80.additional_features import (
        ADDITIONAL_FEATURE_NAMES,
        build_additional_features,
    )

    spec = json.loads(specification.read_text())
    if spec.get("schema") != "return80-rebuild/v1" or output.exists():
        raise ValueError("invalid_rebuild_spec_or_existing_output")

    def resolve(value: str) -> Path:
        """Resolve one explicit input; example './raw' follows the specification directory."""
        return (specification.parent / value).resolve()

    panel_root, source = resolve(spec["panel"]), resolve(spec["source_dataset"])
    roots = [resolve(value) for value in spec["raw_roots"]]
    panel = load_panel(panel_root)
    manifest = json.loads((source / "dataset.json").read_text())
    if panel.calendar != manifest["calendar"] or panel.codes != manifest["securities"]:
        raise ValueError("rebuild_source_axis_mismatch")
    source_hashes = _SOURCE_HASHES.get()
    if source_hashes is not None:
        panel_manifest = read_source_envelope(panel_root / "panel.json")
        read_source_envelope(source / "dataset.json")
        binary_paths = [source / (code + ".npz") for code in panel.codes]
        binary_paths += [
            panel_root / f"{group}-{name}.npy"
            for group in ("values", "signals", "market")
            for name in panel_manifest[group]
        ]
        for path in binary_paths:
            source_hashes[str(path.resolve())] = file_digest(path)
        for path in resolve(spec["distributions"]).glob("*.json"):
            if path.name.startswith("dividend-") or path.name == "distribution-resolutions.json":
                read_source_envelope(path)
    output.mkdir(parents=True)
    with progress_phase("rebuild_signals"):
        matrices, st_flags = build_signals(
            panel, source, roots, resolve(spec["benchmark"]), output / "signals"
        )
    with progress_phase("rebuild_candidates"):
        axis, eligible, patterns, old_scores = candidate_axis(
            panel,
            matrices,
            st_flags,
            source_records(roots, "stock_basic"),
            [row for value in spec["members"] for row in read_envelope(resolve(value))],
        )
        np.savez(output / "candidate-axis.npz", **axis)
        np.save(output / "eligible.npy", eligible)
        np.save(output / "patterns.npy", patterns)
        np.save(output / "legacy_scores.npy", old_scores)
    with progress_phase("rebuild_base_features"):
        base, names = baseline_features(panel, source, matrices, axis, patterns, old_scores)
        np.save(output / "base-features.npy", base)
    with progress_phase("rebuild_additional_features"):
        additional, audit = build_additional_features(
            calendar=panel.calendar,
            codes=panel.codes,
            candidate_days=axis["days"],
            candidate_securities=axis["securities"],
            chip_rows=source_records(roots, "cyq_perf"),
            forecast_rows=source_records(roots, "forecast_vip"),
            indicator_rows=source_records(roots, "fina_indicator_vip"),
            income_rows=source_records(roots, "income_vip"),
            daily_basic_rows=source_records(roots, "daily_basic"),
            adjusted_close=panel.values["close"],
            sector_index=panel.values["sector_index"],
            valid=panel.values["valid"],
        )
        np.save(output / "additional-features.npy", additional)
        (output / "additional-audit.json").write_text(json.dumps(audit) + "\n")
    events = load_distributions(resolve(spec["distributions"]), panel.calendar, panel.codes)
    with progress_phase("rebuild_labels"):
        labels = build_labels(panel, axis, events)
    days, securities = axis["days"], axis["securities"]
    fit = np.flatnonzero(
        (labels[:, 3] == 3)
        & (days >= panel.calendar.index("20211018"))
        & (days < len(panel.calendar) - 31)
    )
    weights, _ = daily_equal_training_weights(
        days[fit], labels[fit, 2], panel.calendar, cutoff=panel.calendar[-1], half_life_sessions=126
    )
    arrays = dict(
        X=np.column_stack((base, additional)),
        fit_indices=fit,
        y=labels[fit, 0],
        sample_weight=weights,
        feature_names=np.array([*names, *ADDITIONAL_FEATURE_NAMES]),
        candidate_dates=np.asarray(panel.calendar)[days],
        candidate_codes=np.asarray(panel.codes)[securities],
        candidate_day_indices=days,
        candidate_security_indices=securities,
        common_eligible=axis["common"],
        labels=labels,
        calendar=np.asarray(panel.calendar),
        security_codes=np.asarray(panel.codes),
    )
    provenance = {
        "spec_sha256": file_digest(specification),
        "source_fingerprint": manifest["source_fingerprint"],
        "domain_revision": "6d6313634734fa7d06cd3414a169fa61e0adb4d9",
    }
    source_hashes = _SOURCE_HASHES.get() or {}
    for filename, expected in source_hashes.items():
        if file_digest(Path(filename)) != expected:
            raise ValueError("rebuild_source_changed_before_publication")
    (output / "raw-source-hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    provenance["raw_source_manifest_sha256"] = file_digest(output / "raw-source-hashes.json")
    result = save_dataset(output / "training", arrays, provenance)
    (output / "distributions.json").write_text(json.dumps([asdict(event) for event in events]))
    write_replay_inputs(
        panel_root,
        {
            "panel/panel.json": panel_root / "panel.json",
            "eligible.npy": output / "eligible.npy",
            "technical_risk.npy": output / "signals/technical_risk.npy",
            "legacy_scores.npy": output / "legacy_scores.npy",
        },
        events,
        output / "replay",
    )
    return result
