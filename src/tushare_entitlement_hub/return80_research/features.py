"""Build dated feature blocks from declared Tushare collections without changing candidates."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..strategy_discovery.replay_inputs import file_digest
from .collection import read_collection
from .dataset import load_dataset, valid_date


def daily_feature_values(
    records: list[dict[str, Any]],
    calendar: list[str],
    codes: list[str],
    *,
    field: str,
    scale: float = 1.0,
    window: int = 1,
    lag: int = 0,
    operation: str = "value",
    available_field: str | None = None,
) -> NDArray[np.float32]:
    """Align raw daily numeric values, then calculate strictly backward market-session windows.

    Examples: mean/window=5 needs all five days; change/window=5 is x[t]-x[t-5].
    A return uses x[t]/x[t-window]-1 and requires a positive denominator. Missing data
    stays NaN. Explicit availability later than the observation day closes that daily input.
    Duplicate security dates are rejected even when identical, avoiding ambiguous sources.
    """
    if (
        calendar != sorted(set(calendar))
        or not calendar
        or not codes
        or len(set(codes)) != len(codes)
        or any(not valid_date(day) for day in calendar)
        or type(window) is not int
        or window < 1
        or type(lag) is not int
        or lag < 0
        or operation not in ("value", "mean", "std", "change", "return")
        or not np.isfinite(scale)
        or not field
        or (operation == "value" and window != 1)
    ):
        raise ValueError("invalid_daily_feature_recipe")
    days = {day: index for index, day in enumerate(calendar)}
    securities = {code: index for index, code in enumerate(codes)}
    values = np.full((len(calendar), len(codes)), np.nan)
    seen = set()
    for row in records:
        day, code = row.get("trade_date"), row.get("ts_code")
        if not isinstance(day, str) or not valid_date(day) or not isinstance(code, str):
            raise ValueError("invalid_daily_feature_identity")
        if day not in days or code not in securities:
            continue
        key = day, code
        if key in seen:
            raise ValueError("duplicate_daily_feature_identity")
        seen.add(key)
        if field not in row:
            raise ValueError("daily_feature_field_missing")
        if available_field:
            available = row.get(available_field)
            if not isinstance(available, str) or not valid_date(available) or available > day:
                continue
        value = row[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError("invalid_daily_feature_value")
        values[days[day], securities[code]] = value * scale
    result = np.full(values.shape, np.nan)
    if operation == "value":
        result[:] = values
    elif operation in ("mean", "std"):
        for day in range(window - 1, len(calendar)):
            sample = values[day - window + 1 : day + 1]
            result[day] = np.mean(sample, axis=0) if operation == "mean" else np.std(sample, axis=0)
    else:
        if window < len(calendar):
            current, previous = values[window:], values[:-window]
            if operation == "change":
                result[window:] = current - previous
            else:
                result[window:] = (
                    np.divide(
                        current, previous, out=np.full(current.shape, np.nan), where=previous > 0
                    )
                    - 1
                )
    if lag:
        shifted = np.full(result.shape, np.nan)
        if lag < len(calendar):
            shifted[lag:] = result[:-lag]
        result = shifted
    with np.errstate(over="ignore"):
        converted = result.astype(np.float32)
    if np.isinf(converted).any():
        raise ValueError("daily_feature_overflow")
    return converted


def build_daily_block(dataset: Path, collection: Path, recipe: Path, output: Path) -> Path:
    """Create an appendable block from downloaded raw fields and explicit formulas.

    Example: daily_basic.turnover_rate with mean/window=5 emits a stock/date CSV,
    its SHA and formula/version/unit provenance; unknown candidate values remain blank.
    """
    if output.exists():
        raise ValueError("fresh_feature_output_required")
    arrays, _ = load_dataset(dataset)
    spec = json.loads(recipe.read_text())
    if spec.get("schema") != "return80-daily-features/v1" or not spec.get("features"):
        raise ValueError("invalid_daily_feature_spec")
    names = [item["name"] for item in spec["features"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate_daily_feature_name")
    columns = []
    for item in spec["features"]:
        if not item.get("unit") or not item.get("formula_version"):
            raise ValueError("missing_feature_formula_or_unit")
        matrix = daily_feature_values(
            read_collection(collection, item["api"]),
            arrays["calendar"].tolist(),
            arrays["security_codes"].tolist(),
            field=item["field"],
            scale=item.get("scale", 1.0),
            window=item.get("window", 1),
            lag=item.get("lag", 0),
            operation=item.get("operation", "value"),
            available_field=item.get("available_field"),
        )
        columns.append(
            matrix[arrays["candidate_day_indices"], arrays["candidate_security_indices"]]
        )
    values = np.column_stack(columns)
    output.mkdir(parents=True)
    csv_path = output / "features.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trade_date", "ts_code", "available_date", *names])
        for index, (day, code) in enumerate(
            zip(arrays["candidate_dates"], arrays["candidate_codes"], strict=True)
        ):
            writer.writerow(
                [
                    day,
                    code,
                    day,
                    *[float(value) if np.isfinite(value) else "" for value in values[index]],
                ]
            )
    manifest = {
        "schema": "return80-feature-block/v1",
        "feature_names": names,
        "file": csv_path.name,
        "sha256": file_digest(csv_path),
        "source": sorted({item["api"] for item in spec["features"]}),
        "units": {item["name"]: item["unit"] for item in spec["features"]},
        "formula_version": {item["name"]: item["formula_version"] for item in spec["features"]},
        "recipe": spec,
        "historical_pit_certified": False,
        "collection_sha256": file_digest(collection / "collection.json"),
        "finite_counts": np.isfinite(values).sum(axis=0).tolist(),
        "candidate_rows": len(values),
    }
    path = output / "block.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
