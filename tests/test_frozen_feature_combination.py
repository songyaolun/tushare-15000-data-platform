"""Offline checks for one-axis feature concatenation with immutable input hashes."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.frozen_feature_combination import (
    CHIPS_EARNINGS_FEATURE_NAMES,
    combine_frozen_feature_blocks,
)


def test_chips_earnings_six_column_order_without_moneyflow(tmp_path: Path) -> None:
    """Two three-column blocks retain sparse values and produce exactly chips then earnings."""
    chip_path, earnings_path = tmp_path / "chips.npy", tmp_path / "earnings.npy"
    chips = np.array([[0.1, 0.2, np.nan], [0.3, 0.4, 0.5]], dtype=np.float64)
    earnings = np.array([[np.nan, np.nan, np.nan], [0.6, 0.7, np.nan]], dtype=np.float32)
    np.save(chip_path, chips)
    np.save(earnings_path, earnings)
    blocks = tuple(
        (path, hashlib.sha256(path.read_bytes()).hexdigest(), names)
        for path, names in (
            (chip_path, CHIPS_EARNINGS_FEATURE_NAMES[:3]),
            (earnings_path, CHIPS_EARNINGS_FEATURE_NAMES[3:]),
        )
    )
    values, names = combine_frozen_feature_blocks(blocks, expected_rows=2)
    assert values.shape == (2, 6)
    assert names == (
        "chip_median_weighted_offset",
        "chip_width90",
        "chip_width90_change5",
        "forecast_growth_midpoint",
        "forecast_growth_interval_width",
        "forecast_growth_revision",
    )
    np.testing.assert_array_equal(values[:, :3], chips.astype(np.float32))
    np.testing.assert_array_equal(values[:, 3:], earnings)


def test_pinned_combination_preserves_values_order_and_missingness(tmp_path: Path) -> None:
    """Two blocks keep exact column order and NaN, with original source bytes unchanged."""
    first, second = tmp_path / "first.npy", tmp_path / "second.npy"
    np.save(first, np.array([[1.0], [np.nan]]))
    np.save(second, np.array([[2.0], [3.0]]))
    first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
    result, names = combine_frozen_feature_blocks(
        (
            (first, hashlib.sha256(first_bytes).hexdigest(), ("a",)),
            (second, hashlib.sha256(second_bytes).hexdigest(), ("b",)),
        ),
        expected_rows=2,
    )
    np.testing.assert_equal(result, [[1, 2], [np.nan, 3]])
    assert result.dtype == np.float32 and names == ("a", "b")
    assert first.read_bytes() == first_bytes and second.read_bytes() == second_bytes


@pytest.mark.parametrize(
    "case",
    [
        "hash",
        "shape",
        "infinity",
        "integer",
        "overflow",
        "empty_columns",
        "duplicate_names",
        "blank_name",
    ],
)
def test_invalid_frozen_block_fails_closed(tmp_path: Path, case: str) -> None:
    """Changed bytes, ambiguous names or unusable values fail before a model sees features."""
    path = tmp_path / "block.npy"
    values = np.array([[1.0], [2.0]])
    names = ("a",)
    if case == "shape":
        values = np.ones((1, 1))
    elif case == "infinity":
        values[0, 0] = np.inf
    elif case == "integer":
        values = np.ones((2, 1), dtype=int)
    elif case == "overflow":
        values[0, 0] = 1e300
    elif case == "empty_columns":
        names = ()
    elif case == "duplicate_names":
        names = ("a", "a")
    elif case == "blank_name":
        names = (" ",)
    np.save(path, values)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if case != "hash" else "wrong"
    with pytest.raises(ValueError, match="frozen"):
        combine_frozen_feature_blocks(((path, digest, names),), expected_rows=2)


@pytest.mark.parametrize("rows", [0, -1, 1.5])
def test_invalid_combination_specification(rows: int) -> None:
    """Invalid row counts fail before reading any supplied file, even when blocks exist."""
    with pytest.raises(ValueError, match="specification"):
        combine_frozen_feature_blocks(((Path("unread.npy"), "unused", ("a",)),), expected_rows=rows)


def test_empty_combination_specification() -> None:
    """An empty block collection fails even when the expected candidate count is valid."""
    with pytest.raises(ValueError, match="specification"):
        combine_frozen_feature_blocks((), expected_rows=2)
