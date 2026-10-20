"""Combine pinned candidate feature blocks without changing their row order or unknowns."""

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

CHIPS_EARNINGS_FEATURE_NAMES = (
    "chip_median_weighted_offset",
    "chip_width90",
    "chip_width90_change5",
    "forecast_growth_midpoint",
    "forecast_growth_interval_width",
    "forecast_growth_revision",
)


def combine_frozen_feature_blocks(
    blocks: tuple[tuple[Path, str, tuple[str, ...]], ...],
    *,
    expected_rows: int,
) -> tuple[NDArray[np.float32], tuple[str, ...]]:
    """Load SHA-pinned NPY blocks and append their columns on one frozen candidate axis.

    Each block supplies its path, expected SHA256 and ordered feature names. The
    caller separately pins the shared candidate axis. Missing values remain NaN;
    infinity, wrong row counts, duplicate names and changed bytes fail closed.
    Only candidate matrices are loaded, never the full date/security feature cube.
    Example: two pinned 2x1 files [[1], [NaN]] and [[2], [3]], named a and b,
    yield float32 [[1, 2], [NaN, 3]] and names ('a', 'b'). Files are read-only.
    """
    if not blocks or type(expected_rows) is not int or expected_rows < 1:
        raise ValueError("invalid frozen combination specification")
    names = tuple(name for _, _, columns in blocks for name in columns)
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("invalid frozen feature names")
    arrays = []
    for path, expected_sha256, columns in blocks:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("frozen feature hash mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            not columns
            or values.shape != (expected_rows, len(columns))
            or values.dtype.kind != "f"
            or np.isinf(values).any()
        ):
            raise ValueError("invalid frozen feature block")
        arrays.append(values)
    with np.errstate(over="ignore"):
        combined = np.column_stack(arrays).astype(np.float32)
    if np.isinf(combined).any():
        raise ValueError("frozen feature float32 overflow")
    return combined, names
