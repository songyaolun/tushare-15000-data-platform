"""Raw partition completeness, market-value units and full-axis artifact round-trip tests."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tushare_entitlement_hub.strategy_discovery.financial_quality_artifacts import (
    build_financial_artifacts,
    candidate_capitalization,
    coverage_by_group,
    file_sha256,
    raw_records,
)


def raw_page(root: Path, api: str, rows: list[dict[str, Any]], *,
             name: str = "page", **changes: Any) -> Path:
    """Write a deterministic raw envelope; a daily row gets a trade_date partition."""
    fields = list(rows[0]) if rows else ["ts_code", "trade_date"]
    parameters = ({"period": rows[0]["end_date"], "limit": 3000, "offset": 0}
                  if rows and "end_date" in rows[0]
                  else {"trade_date": rows[0]["trade_date"] if rows else "20240426"})
    envelope = {"api_name": api, "parameters": parameters, "quality": {"status": "passed"},
                "observed_at": "2026-10-19T12:00:00Z",
                "response": {"code": 0, "data": {"fields": fields,
                             "items": [[row[field] for field in fields] for row in rows]}}}
    envelope.update(changes)
    target = root / api / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(envelope))
    return target


def market(code: str = "A", day: str = "20240426", **changes: Any) -> dict[str, Any]:
    """Build raw units; price=10, shares=100 ten-thousands gives ten-million-yuan cap."""
    return {"ts_code": code, "trade_date": day, "close": 10., "total_share": 100.,
            "total_mv": 1000., **changes}


def test_stream_complete_raw_records_and_safe_provenance(tmp_path: Path) -> None:
    """A valid raw partition streams original units and records only public provenance fields."""
    path = raw_page(tmp_path, "daily_basic", [market()])
    provenance: list[dict[str, Any]] = []
    assert list(raw_records([tmp_path], "daily_basic", ("total_share",), provenance)) == [market()]
    assert provenance[0]["sha256"] == file_sha256(path)
    assert provenance[0]["partition"] == {"trade_date": "20240426"}


@pytest.mark.parametrize("case", ["absent", "failed", "quality", "field", "duplicate_field",
                                  "width", "partition", "offset", "cap", "more", "wrong_day",
                                  "duplicate_page", "gap", "truncated", "short_interior"])
def test_raw_quality_and_pagination_fail_closed(tmp_path: Path, case: str) -> None:
    """Invalid identity/table/partition and incomplete pagination raise instead of losing rows."""
    if case != "absent":
        path = raw_page(tmp_path, "daily_basic", [market()])
        envelope = json.loads(path.read_text())
        table = envelope["response"]["data"]
        if case == "failed":
            envelope["response"]["code"] = 1
        elif case == "quality":
            envelope["quality"]["status"] = "failed"
        elif case == "field":
            table["fields"][0] = "unexpected"
        elif case == "duplicate_field":
            table["fields"][0] = table["fields"][1]
        elif case == "width":
            table["items"][0].pop()
        elif case == "partition":
            envelope["parameters"] = {}
        elif case == "offset":
            envelope["parameters"]["offset"] = -1
        elif case == "cap":
            envelope["parameters"]["limit"] = 0
        elif case == "more":
            table["has_more"] = True
        elif case == "wrong_day":
            envelope["parameters"]["trade_date"] = "20240425"
        elif case == "gap":
            envelope["parameters"]["offset"] = 1
        elif case == "truncated":
            envelope["parameters"]["limit"] = 1
        elif case == "short_interior":
            following = {**envelope, "parameters": {"trade_date": "20240426", "offset": 6000}}
            (path.parent / "later.json").write_text(json.dumps(following))
        elif case == "duplicate_page":
            (path.parent / "same.json").write_text(json.dumps(envelope))
        path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError):
        list(raw_records([tmp_path], "daily_basic", ("ts_code", "total_share"), []))


def test_real_empty_partition_and_complete_multiple_pages(tmp_path: Path) -> None:
    """An explicit empty page is valid; a full first page needs a shorter terminal page."""
    raw_page(tmp_path, "empty", [])
    assert list(raw_records([tmp_path], "empty", ("ts_code",), [])) == []
    raw_page(tmp_path, "daily_basic", [market()], name="first",
             parameters={"trade_date": "20240426", "offset": 0, "limit": 1})
    raw_page(tmp_path, "daily_basic", [], name="last",
             parameters={"trade_date": "20240426", "offset": 1, "limit": 1})
    assert list(raw_records([tmp_path], "daily_basic", ("ts_code",), [])) == [market()]


def test_market_cap_units_duplicates_parity_and_no_date_carry() -> None:
    """Raw ten-thousand units convert once; conflicting duplicates and value mismatch stay NaN."""
    rows = [market("OUT"), market(), market(), market("B", close=11),
            market("C"), market("C", close=20, total_mv=2000),
            market("D", close=-1), market("E", close=1e308, total_share=1e308)]
    values, audit = candidate_capitalization(rows, ["20240426"] * 6 + ["20240429"],
                                            ["A", "B", "C", "D", "E", "A", "A"])
    np.testing.assert_equal(values, [1e7, np.nan, np.nan, np.nan, np.nan, 1e7, np.nan])
    assert audit["conflicting_security_dates"] == 1
    assert audit["market_value_parity_failures"] == 1
    assert audit["nonfinite_market_cap"] == 1
    assert audit["known"] == 2
    assert len(audit["failures"]) == 4
    with pytest.raises(ValueError, match="axes"):
        candidate_capitalization([], ["20240426"], [])


def test_coverage_preserves_unknown_group_denominators() -> None:
    """Missing features remain inside group denominators and malformed labels fail explicitly."""
    result = coverage_by_group(np.array([[1., np.nan], [np.nan, 2.]]), ["unknown", "unknown"])
    assert result == {"unknown": {"candidate_rows": 2, "known": [1, 1]}}
    with pytest.raises(ValueError, match="coverage"):
        coverage_by_group(np.ones((1, 2)), [])


def financial_fixture(root: Path) -> tuple[Path, Path, Path]:
    """Create five securities and annual reports; return raw, panel and candidate paths."""
    raw, panel = root / "raw", root / "panel"
    panel.mkdir()
    codes = ["600001.SH", "688001.SH", "300001.SZ", "830001.BJ", "000001.SZ"]
    (panel / "panel.json").write_text(json.dumps({"calendar": ["20240425", "20240426"],
                                                "codes": codes}))
    np.save(panel / "values-sector_index.npy", [[1, 2, 3, np.nan, 1], [1, 2, 3, np.nan, 1]])
    axis = root / "axis.npz"
    np.savez(axis, days=np.array([0, 1, 1, 1, 1, 1]), securities=np.array([0, 0, 1, 2, 3, 4]),
             common=np.array([False, True, True, True, True, True]))
    np.save(root / "mature-mask.npy", [False, True, True, True, True, True])
    base = {"end_date": "20231231", "ann_date": "20240425", "f_ann_date": "20240425",
            "report_type": "1", "comp_type": "1"}
    raw_page(raw, "fina_indicator_vip", [
        {**base, "ts_code": code, "dt_netprofit_yoy": 30, "netprofit_yoy": 10} for code in codes])
    raw_page(raw, "cashflow_vip", [
        {**base, "ts_code": code, "n_cashflow_act": 150, "net_profit": profit}
        for code, profit in zip(codes, [100, 0, -10, None, 100], strict=True)])
    raw_page(raw, "income_vip", [
        {**base, "ts_code": code, "n_income_attr_p": profit}
        for code, profit in zip(codes, [100, 0, -10, None, 100], strict=True)])
    raw_page(raw, "daily_basic", [market(code) for code in codes[:4]])
    return raw, panel, axis


def test_full_candidate_artifacts_round_trip_without_filtering(tmp_path: Path) -> None:
    """B and E retain all six rows, exact shapes and source hashes; missing market cap stays NaN."""
    raw, panel, axis = financial_fixture(tmp_path)
    result = build_financial_artifacts([raw], panel, axis, tmp_path / "out",
                                       source_revision="fixture", expected_rows=6,
                                       mature_mask_path=tmp_path / "mature-mask.npy",
                                       expected_mature_rows=5,
                                       expected_axis_sha256=file_sha256(axis))
    assert result["quality"]["known"] == [5, 5, 2]
    assert result["valuation"]["known"] == [3]
    for family, columns in [("quality", 3), ("valuation", 1)]:
        directory = tmp_path / "out" / family
        features = np.load(directory / "candidate-features.npy")
        manifest = json.loads((directory / "manifest.json").read_text())
        audit = json.loads((directory / "audit.json").read_text())
        assert features.shape == (6, columns)
        assert manifest["candidate_features_sha256"] == file_sha256(
            directory / "candidate-features.npy")
        assert audit["by_historical_sector"]["unknown"]["candidate_rows"] == 1
        assert audit["by_board"]["mainboard"]["candidate_rows"] == 3
        assert manifest["historical_pit_certified"] is False
        assert len(json.loads((directory / "sources.json").read_text())) == 4
    np.testing.assert_allclose(np.load(tmp_path / "out/valuation/candidate-features.npy")[:, 0],
                               [np.nan, 1e-5, 0, -1e-6, np.nan, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match="hash"):
        build_financial_artifacts([raw], panel, axis, tmp_path / "bad", source_revision="fixture",
                                  mature_mask_path=tmp_path / "mature-mask.npy")
    with pytest.raises(ValueError, match="count"):
        build_financial_artifacts([raw], panel, axis, tmp_path / "bad", source_revision="fixture",
                                  mature_mask_path=tmp_path / "mature-mask.npy",
                                  expected_axis_sha256=file_sha256(axis))


def test_common_eligibility_is_not_the_mature_label_mask(tmp_path: Path) -> None:
    """Wrong fit-mask cardinality fails even when the NPZ common flag has the expected shape."""
    raw, panel, axis = financial_fixture(tmp_path)
    np.save(tmp_path / "mature-mask.npy", np.ones(6, dtype=bool))
    with pytest.raises(ValueError, match="mature axis"):
        build_financial_artifacts([raw], panel, axis, tmp_path / "out",
                                  source_revision="fixture", expected_rows=6,
                                  mature_mask_path=tmp_path / "mature-mask.npy",
                                  expected_mature_rows=5, expected_axis_sha256=file_sha256(axis))
