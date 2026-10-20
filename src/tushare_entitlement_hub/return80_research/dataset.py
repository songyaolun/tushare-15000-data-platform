"""Versioned numeric research datasets with explicit date/security alignment."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..strategy_discovery.joint_exit_scoring import exit_training_targets
from ..strategy_discovery.replay_inputs import file_digest
from ..strategy_discovery.training_weights import daily_equal_training_weights

SCHEMA = "return80-training-inputs/v1"
REQUIRED = (
    "X",
    "fit_indices",
    "y",
    "sample_weight",
    "feature_names",
    "candidate_dates",
    "candidate_codes",
    "candidate_day_indices",
    "candidate_security_indices",
    "common_eligible",
    "labels",
    "calendar",
    "security_codes",
)


def valid_date(value: str) -> bool:
    """Accept a real YYYYMMDD date; 20240229 passes and 20240230 fails."""
    try:
        return len(value) == 8 and datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except (ValueError, TypeError):
        return False


def validate_arrays(data: dict[str, NDArray[Any]], *, half_life: int = 126) -> None:
    """Check identities, mature labels and weights before training or extension.

    Example: a permuted candidate code or a zero weight rejects the entire dataset.
    The existing full-sample/no-holdout choice is preserved; labels never become features.
    """
    if not set(REQUIRED) <= data.keys():
        raise ValueError("missing_training_arrays")
    features, labels = data["X"], data["labels"]
    rows = len(features)
    names = data["feature_names"].tolist()
    calendar, codes = data["calendar"].tolist(), data["security_codes"].tolist()
    if (
        features.ndim != 2
        or features.dtype != np.float32
        or np.isinf(features).any()
        or len(names) != features.shape[1]
        or not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or not calendar
        or calendar != sorted(set(calendar))
        or any(not valid_date(day) for day in calendar)
        or not codes
        or len(set(codes)) != len(codes)
        or labels.shape != (rows, 4)
    ):
        raise ValueError("invalid_training_contract")
    days, securities = data["candidate_day_indices"], data["candidate_security_indices"]
    for values, limit in ((days, len(calendar)), (securities, len(codes))):
        if (
            values.shape != (rows,)
            or values.dtype.kind not in "iu"
            or (values < 0).any()
            or (values >= limit).any()
        ):
            raise ValueError("invalid_candidate_axis")
    if (
        data["common_eligible"].dtype != bool
        or data["common_eligible"].shape != (rows,)
        or not np.array_equal(data["candidate_dates"], np.asarray(calendar)[days])
        or not np.array_equal(data["candidate_codes"], np.asarray(codes)[securities])
        or len(np.unique(days * len(codes) + securities)) != rows
    ):
        raise ValueError("misaligned_candidate_identity")
    closed, _ = exit_training_targets(labels)
    if "20211018" not in calendar:
        raise ValueError("missing_training_start")
    mature = (days >= calendar.index("20211018")) & (days < len(calendar) - 31)
    expected = np.flatnonzero(closed & mature)
    fit = data["fit_indices"]
    if (
        fit.dtype.kind not in "iu"
        or not np.array_equal(fit, expected)
        or not np.array_equal(data["y"], labels[expected, 0])
    ):
        raise ValueError("changed_training_targets")
    weights, _ = daily_equal_training_weights(
        days[fit], labels[fit, 2], calendar, cutoff=calendar[-1], half_life_sessions=half_life
    )
    if not np.array_equal(data["sample_weight"], weights):
        raise ValueError("changed_training_weights")


def save_dataset(
    output: Path,
    arrays: dict[str, NDArray[Any]],
    provenance: dict[str, Any],
    *,
    half_life: int = 126,
) -> Path:
    """Write a fresh portable dataset with hashes, only after validating all arrays.

    Example: output/X.npy is memory-mappable and dataset.json has relative names.
    An existing destination is never overwritten or mixed with a different experiment.
    """
    if output.exists():
        raise ValueError("fresh_training_output_required")
    validate_arrays(arrays, half_life=half_life)
    output.mkdir(parents=True)
    files = {}
    for name in REQUIRED:
        path = output / (name + ".npy")
        np.save(path, arrays[name], allow_pickle=False)
        files[path.name] = file_digest(path)
    manifest = {
        "schema": SCHEMA,
        "files": files,
        "half_life_sessions": half_life,
        "candidate_rows": len(arrays["X"]),
        "fit_rows": len(arrays["y"]),
        "feature_names": arrays["feature_names"].tolist(),
        "provenance": provenance,
        "historical_pit_certified": False,
        "financial_holdout": False,
    }
    path = output / "dataset.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


def load_dataset(root: Path) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Verify and memory-map a portable dataset; reject tampered bytes before NumPy loading.

    Example: relocating the whole directory preserves all hashes and returned row identities.
    """
    manifest = json.loads((root / "dataset.json").read_text())
    expected_files = {name + ".npy" for name in REQUIRED}
    if manifest.get("schema") != SCHEMA or set(manifest.get("files", {})) != expected_files:
        raise ValueError("invalid_dataset_manifest")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or file_digest(path) != expected:
            raise ValueError("changed_dataset_file")
    arrays = {
        name: np.load(root / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in REQUIRED
    }
    validate_arrays(arrays, half_life=manifest["half_life_sessions"])
    if (
        manifest["feature_names"] != arrays["feature_names"].tolist()
        or manifest["candidate_rows"] != len(arrays["X"])
        or manifest["fit_rows"] != len(arrays["y"])
    ):
        raise ValueError("dataset_manifest_axis_mismatch")
    return arrays, manifest


def import_training_bundle(bundle: Path, output: Path) -> Path:
    """Import the delivered half126 ZIP's extracted directory without using old absolute paths.

    Example: bundle/training.npz and its SHA256SUMS create output/dataset.json.
    Verify every delivered file and the original training protocol before accepting the arrays.
    """
    hashes = {}
    for line in (bundle / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = bundle / relative
        if (
            relative in hashes
            or Path(relative).is_absolute()
            or not path.resolve().is_relative_to(bundle.resolve())
            or file_digest(path) != digest
        ):
            raise ValueError("changed_bundle_file")
        hashes[relative] = digest
    if not {"training.npz", "original-protocol.json"} <= hashes.keys():
        raise ValueError("incomplete_bundle_checksums")
    protocol = json.loads((bundle / "original-protocol.json").read_text())
    with np.load(bundle / "training.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in REQUIRED}
    if (
        protocol["feature_names"] != arrays["feature_names"].tolist()
        or protocol["candidate_rows"] != len(arrays["X"])
        or protocol["fit_rows"] != len(arrays["y"])
    ):
        raise ValueError("bundle_protocol_mismatch")
    return save_dataset(
        output,
        arrays,
        {"bundle_sha256": hashes["training.npz"], "original_protocol": protocol},
        half_life=protocol["training_weights"]["half_life_sessions"],
    )


def append_features(dataset: Path, block: Path, output: Path) -> Path:
    """Join a declared CSV feature block by exact signal date and stock, retaining unknowns.

    block.json declares CSV SHA, unique feature names, source, units and formula version.
    CSV columns: trade_date,ts_code,available_date,<features>. available_date is the first
    trading date on which the values are usable at the signal close, not download time.
    Example: reordered rows join correctly; missing pairs remain NaN; a future-available
    value, duplicate pair/name or a pair outside the frozen candidate axis is rejected.
    """
    arrays, parent = load_dataset(dataset)
    spec = json.loads(block.read_text())
    names = spec.get("feature_names", [])
    filename = spec.get("file", "")
    path = block.parent / filename
    if (
        spec.get("schema") != "return80-feature-block/v1"
        or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
        or set(names) & set(parent["feature_names"])
        or not all(spec.get(key) for key in ("source", "units", "formula_version"))
        or not filename
        or Path(filename).is_absolute()
        or not path.resolve().is_relative_to(block.parent.resolve())
        or file_digest(path) != spec.get("sha256")
    ):
        raise ValueError("invalid_feature_block")
    dates, codes = arrays["candidate_dates"], arrays["candidate_codes"]
    positions = {
        (str(day), str(code)): index
        for index, (day, code) in enumerate(zip(dates, codes, strict=True))
    }
    values = np.full((len(dates), len(names)), np.nan, dtype=np.float32)
    seen = set()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = ["trade_date", "ts_code", "available_date", *names]
        if reader.fieldnames != fields:
            raise ValueError("feature_block_csv_columns")
        for row in reader:
            key = row["trade_date"], row["ts_code"]
            if (
                key not in positions
                or key in seen
                or not valid_date(row["available_date"])
                or row["available_date"] > row["trade_date"]
                or None in row
            ):
                raise ValueError("feature_block_axis_or_visibility")
            seen.add(key)
            for column, name in enumerate(names):
                text = row[name]
                value = float(text) if text else np.nan
                if np.isinf(value) or (text and not np.isfinite(value)):
                    raise ValueError("feature_block_invalid_number")
                with np.errstate(over="ignore"):
                    values[positions[key], column] = value
    if np.isinf(values).any():
        raise ValueError("feature_block_float32_overflow")
    arrays = dict(arrays)
    arrays["X"] = np.column_stack((arrays["X"], values))
    arrays["feature_names"] = np.asarray(parent["feature_names"] + names)
    return save_dataset(
        output,
        arrays,
        {
            "parent_manifest_sha256": file_digest(dataset / "dataset.json"),
            "block_manifest": spec,
            "matched_rows": len(seen),
            "missing_rows": len(dates) - len(seen),
            "finite_counts": np.isfinite(values).sum(axis=0).tolist(),
        },
        half_life=parent["half_life_sessions"],
    )


def map_predictions(dataset: Path, predictions: Path, output: Path) -> Path:
    """Map one score per candidate to the original calendar/security grid for replay.

    Example: 520690 predictions produce 1643×5451 scores with NaN outside candidates.
    Predictions must be finite and retain X row order; labels/fit rows are never substituted.
    """
    arrays, _ = load_dataset(dataset)
    scores = np.load(predictions, allow_pickle=False)
    if scores.shape != (len(arrays["X"]),) or not np.isfinite(scores).all():
        raise ValueError("invalid_candidate_predictions")
    if output.exists():
        raise ValueError("fresh_predictions_output_required")
    matrix = np.full((len(arrays["calendar"]), len(arrays["security_codes"])), np.nan)
    matrix[arrays["candidate_day_indices"], arrays["candidate_security_indices"]] = scores
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.save(stream, matrix, allow_pickle=False)
    return output
