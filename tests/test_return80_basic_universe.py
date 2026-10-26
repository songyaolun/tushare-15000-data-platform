"""Offline checks for independent basic qualification and trustworthy negative ST evidence."""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from tushare_entitlement_hub.return80_research.basic_universe import (
    board,
    build_basic_universe,
    digest,
    main,
    natural_amount20,
    verified_partitions,
    weekly_rows,
)


def collection(root, name, partitions):
    """Write tiny public collections; example empty ST rows still retain the date receipt."""
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for index, (api, params, fields, rows) in enumerate(partitions):
        path = root / f"{name}-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "api_name": api,
                    "params": params,
                    "response": {
                        "code": 0,
                        "data": {"fields": fields, "items": rows, "has_more": False},
                    },
                }
            )
        )
        receipts.append(
            {
                "request": {
                    "api": api,
                    "parameters": params,
                    "primary_key": ["ts_code"] if api != "trade_cal" else ["cal_date"],
                },
                "rows": len(rows),
                "files": [{"path": path.name, "sha256": digest(path)}],
            }
        )
    manifest = root / name
    manifest.write_text(json.dumps({"schema": "return80-collection/v1", "partitions": receipts}))
    return manifest


@pytest.fixture
def sources(tmp_path):
    """Create full-status master, official calendar, and complete per-day ST fixture."""
    raw, panel = tmp_path / "raw", tmp_path / "panel"
    panel.mkdir()
    dates = [date(2021, 9, 1) + timedelta(days=i) for i in range(52)]
    calendar = [day.strftime("%Y%m%d") for day in dates if day.weekday() < 5]
    codes = ["000001.SZ", "000002.SZ", "300001.SZ", "688001.SH"]
    fields = ["ts_code", "list_date", "delist_date", "list_status"]
    master = [
        (
            "stock_basic",
            {"list_status": "L"},
            fields,
            [[code, "20000101", None, "L"] for code in codes if code != "000002.SZ"],
        ),
        (
            "stock_basic",
            {"list_status": "D"},
            fields,
            [
                ["000002.SZ", "20000101", "20211021", "D"],
                ["600000.SH", "20000101", "20100101", "D"],
            ],
        ),
        ("stock_basic", {"list_status": "P"}, fields, []),
        ("trade_cal", {"exchange": "SSE"}, ["cal_date", "is_open"], [[day, 1] for day in calendar]),
    ]
    collection(raw / "bootstrap", "collection.json", master)
    collection(
        raw,
        "stock_st-collection.json",
        [
            (
                "stock_st",
                {"trade_date": day},
                ["ts_code", "trade_date"],
                [["300001.SZ", day]] if day == "20211022" else [],
            )
            for day in calendar
        ],
    )
    (panel / "panel.json").write_text(json.dumps({"calendar": calendar, "codes": codes}))
    np.save(panel / "values-raw_close.npy", np.full((len(calendar), len(codes)), 10.0))
    np.save(panel / "values-raw_amount.npy", np.full((len(calendar), len(codes)), 40000.0))
    return panel, raw, calendar


def test_natural_quote_window_and_units():
    """Suspension slots do not consume bars; a genuine quote's missing amount stays unknown."""
    close = np.full((23, 2), 10.0)
    amount = np.full((23, 2), 40000.0)
    close[5, 0] = np.nan
    amount[5, 1] = np.nan
    result = natural_amount20(close, amount)
    assert np.isnan(result[19, 0])
    assert result[20, 0] == 40_000_000
    assert np.isnan(result[20, 1])
    assert np.isnan(natural_amount20(close[:3], amount[:3])).all()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("920001.BJ", "Beijing"),
        ("688001.SH", "STAR"),
        ("301001.SZ", "ChiNext"),
        ("600000.SH", "Shanghai_main"),
        ("000001.SZ", "Shenzhen_main"),
        ("unknown", "other"),
    ],
)
def test_board_audit_bucket(code, expected):
    """Every board, including excluded boards, retains its own denominator bucket."""
    assert board(code) == expected


def test_weekly_origins_holiday_and_partial_tail():
    """Holiday Thursday is a week end when next week exists; trailing Monday is not."""
    calendar = ["20210929", "20210930", "20211008", "20211011"]
    np.testing.assert_array_equal(weekly_rows(calendar, np.array([0, 1, 1, 2, 3])), [1, 2, 3])


def test_complete_basic_axis_without_signal_filters(sources, tmp_path):
    """All base-qualified names survive, including historical delisted names before exit."""
    panel, raw, calendar = sources
    output = tmp_path / "result"
    report = build_basic_universe(panel, raw, output, end="20211022")
    assert report["complete"]
    assert report["candidate_rows"] == 12
    assert report["weekly_rows"] == 1
    assert report["master_status_counts"] == {"L": 3, "D": 2, "P": 0}
    assert report["outside_panel_count"] == 1
    np.testing.assert_array_equal(np.load(output / "weekly_candidate_security_indices.npy"), [0])
    assert calendar[np.load(output / "weekly_candidate_day_indices.npy")[0]] == "20211022"
    assert not any(report["gap_counts"].values())
    with pytest.raises(ValueError, match="output_exists"):
        build_basic_universe(panel, raw, output)


def test_unknown_master_and_st_are_reported(sources, tmp_path):
    """Unknown master and an absent full-market ST day fail closed without shrinking codes."""
    panel, raw, _ = sources
    metadata = json.loads((panel / "panel.json").read_text())
    metadata["codes"][0] = "000001.SZ"
    metadata["codes"][3] = "000099.SZ"
    (panel / "panel.json").write_text(json.dumps(metadata))
    st_path = raw / "stock_st-collection.json"
    st = json.loads(st_path.read_text())
    st["partitions"] = [
        p for p in st["partitions"] if p["request"]["parameters"]["trade_date"] != "20211022"
    ]
    st_path.write_text(json.dumps(st))
    report = build_basic_universe(panel, raw, tmp_path / "unknown", end="20211022")
    assert report["gap_counts"]["master_unknown"] == 5
    assert report["missing_st_dates"] == ["20211022"]
    assert report["gap_counts"]["st_unknown"] == 2


def test_changed_source_is_rejected(sources):
    """A changed raw page cannot inherit a complete negative-list receipt."""
    _, raw, _ = sources
    path = raw / "stock_st-collection.json-0.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        verified_partitions(raw / "stock_st-collection.json", "stock_st", {})


def test_restricted_and_nonterminal_partitions_fail(tmp_path):
    """Single-stock probes and an unconsumed final page never prove all-market coverage."""
    path = collection(
        tmp_path,
        "collection.json",
        [
            (
                "stock_st",
                {"trade_date": "20211022", "ts_code": "000001.SZ"},
                ["ts_code", "trade_date"],
                [],
            )
        ],
    )
    with pytest.raises(ValueError, match="not_complete_market_query"):
        verified_partitions(path, "stock_st", {})
    path = collection(
        tmp_path,
        "collection.json",
        [("stock_st", {"trade_date": "20211022"}, ["ts_code", "trade_date"], [])],
    )
    doc = json.loads(path.read_text())
    raw = tmp_path / doc["partitions"][0]["files"][0]["path"]
    envelope = json.loads(raw.read_text())
    envelope["response"]["data"]["has_more"] = True
    raw.write_text(json.dumps(envelope))
    doc["partitions"][0]["files"][0]["sha256"] = digest(raw)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="incomplete_final_page"):
        verified_partitions(path, "stock_st", {})


def test_main_dispatch(monkeypatch):
    """CLI paths are passed directly into the same verified offline builder."""
    calls = []
    monkeypatch.setattr(
        "sys.argv", ["basic-universe", "--panel", "p", "--raw", "r", "--output", "o"]
    )
    monkeypatch.setattr(
        "tushare_entitlement_hub.return80_research.basic_universe.build_basic_universe",
        lambda *args, **kwargs: calls.append(args),
    )
    main()
    assert calls == [(Path("p"), Path("r"), Path("o"))]


def test_verified_identity_alias_deduplicates_history(sources, tmp_path):
    """Official same-listing proof plus equal raw histories removes unknown duplicate identity."""
    from tushare_entitlement_hub.return80_research.basic_universe import verified_aliases

    panel, raw, _ = sources
    metadata = json.loads((panel / "panel.json").read_text())
    metadata["codes"][3] = "old.SZ"
    (panel / "panel.json").write_text(json.dumps(metadata))
    close = np.load(panel / "values-raw_close.npy")
    for name in ("raw_open", "raw_high", "raw_low", "raw_adj_factor"):
        np.save(panel / ("values-" + name + ".npy"), close)
    proof = tmp_path / "official.pdf"
    proof.write_bytes(b"official fixture")
    evidence = tmp_path / "aliases.json"
    evidence.write_text(
        json.dumps(
            {
                "schema": "return80-security-alias-evidence/v1",
                "aliases": [
                    {
                        "old_code": "old.SZ",
                        "canonical_code": "000001.SZ",
                        "official_file": proof.name,
                        "official_sha256": digest(proof),
                        "same_listing_entity": True,
                    }
                ],
            }
        )
    )
    assert verified_aliases(evidence, panel, metadata["codes"], {}) == {"old.SZ": "000001.SZ"}
    report = build_basic_universe(
        panel, raw, tmp_path / "aliased", end="20211022", identity_evidence=evidence
    )
    assert report["gap_counts"]["master_unknown"] == 0
    assert report["eligibility_sources_complete"]
    assert report["candidate_rows"] == 12
    close[0, 3] = 22
    np.save(panel / "values-raw_open.npy", close)
    with pytest.raises(ValueError, match="alias_market_history_conflict"):
        verified_aliases(evidence, panel, metadata["codes"], {})


@pytest.mark.parametrize("case", ["path", "schema", "api", "rows", "duplicate", "date", "missing"])
def test_source_contract_failures(tmp_path, case):
    """Corrupt page identity/count/date and escaped paths all fail before trusting ST negatives."""
    path = collection(
        tmp_path,
        "collection.json",
        [
            (
                "stock_st",
                {"trade_date": "20211022"},
                ["ts_code", "trade_date"],
                [["000001.SZ", "20211022"]],
            )
        ],
    )
    doc = json.loads(path.read_text())
    part = doc["partitions"][0]
    page_path = tmp_path / part["files"][0]["path"]
    envelope = json.loads(page_path.read_text())
    if case == "path":
        part["files"][0]["path"] = "../escape.json"
    elif case == "schema":
        doc["schema"] = "invalid"
    elif case == "api":
        envelope["api_name"] = "daily"
    elif case == "rows":
        part["rows"] = 99
    elif case == "duplicate":
        envelope["response"]["data"]["items"] *= 2
        part["rows"] = 2
    elif case == "date":
        envelope["response"]["data"]["items"][0][1] = "20211021"
    elif case == "missing":
        doc["partitions"] = []
    page_path.write_text(json.dumps(envelope))
    part["files"][0]["sha256"] = digest(page_path)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        verified_partitions(path, "stock_st", {})


def test_strict_price_age_and_amount_boundaries(sources, tmp_path):
    """Exactly one yuan and exactly thirty million are excluded, as are STAR securities."""
    panel, raw, _ = sources
    close = np.load(panel / "values-raw_close.npy")
    amount = np.load(panel / "values-raw_amount.npy")
    close[:, 0] = 1
    amount[:, 2] = 30000
    np.save(panel / "values-raw_close.npy", close)
    np.save(panel / "values-raw_amount.npy", amount)
    report = build_basic_universe(panel, raw, tmp_path / "strict", end="20211022")
    assert report["candidate_rows"] == 3
    assert report["weekly_rows"] == 0
    assert board("302132.SZ") == "ChiNext"


def test_installed_wheel_revision_without_git(tmp_path, monkeypatch):
    """A wheel outside every checkout gets its exact pinned commit from direct_url.json."""
    from types import SimpleNamespace

    from tushare_entitlement_hub.return80_research.basic_universe import code_origin

    source = tmp_path / "site-packages" / "package" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n")
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        lambda name: SimpleNamespace(
            read_text=lambda filename: json.dumps(
                {"url": "https://example.invalid/repo.git", "vcs_info": {"commit_id": "a" * 40}}
            )
        ),
    )
    monkeypatch.setattr(
        "subprocess.check_output", lambda *args, **kwargs: pytest.fail("git called")
    )
    result = code_origin(source)
    assert result["repository_revision"] == "a" * 40
    assert result["revision_source"] == "installed_direct_url"
    assert result["module_sha256"] == digest(source)


def test_unversioned_wheel_is_not_mislabeled(tmp_path, monkeypatch):
    """An unversioned wheel records missing revision rather than borrowing an unrelated cwd Git."""
    from types import SimpleNamespace

    from tushare_entitlement_hub.return80_research.basic_universe import code_origin

    source = tmp_path / "site-packages" / "package" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("pass\n")
    monkeypatch.setattr(
        "importlib.metadata.distribution",
        lambda name: SimpleNamespace(read_text=lambda filename: None),
    )
    monkeypatch.setattr(
        "subprocess.check_output", lambda *args, **kwargs: pytest.fail("git called")
    )
    result = code_origin(source)
    assert result["repository_revision"] is None
    assert result["revision_source"] == "unversioned_installation"
