"""Weekly feature reconstruction preserves unknowns and strict disclosure timing."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_return80_research_rebuild import envelope, fixture

from tushare_entitlement_hub.client import Client, FakeTransport
from tushare_entitlement_hub.raw import RawStore
from tushare_entitlement_hub.return80_research import basic_features as basic
from tushare_entitlement_hub.return80_research.rebuild import build_signals


def signal_proof_fixture(path, benchmark, signal_root):
    """Write a frozen execution fixture; e.g. one signal binds its bytes to CSI300 input."""
    path.mkdir()
    snapshot = path / "code-snapshot"
    snapshot.mkdir()
    producer = b"# frozen fixture producer\n"
    import hashlib

    digest = hashlib.sha256(producer).hexdigest()
    (snapshot / (digest + "-producer.py")).write_bytes(producer)
    arrays = {}
    for array in signal_root.glob("*.npy"):
        values = np.load(array)
        arrays[array.stem] = dict(
            new_sha256=basic.file_digest(array), shape=list(values.shape), dtype=str(values.dtype)
        )
    (path / "array-comparisons.json").write_text(json.dumps(arrays))
    (path / "raw-source-hashes.json").write_text(
        json.dumps({str(benchmark): basic.file_digest(benchmark)})
    )
    proof = path / "complete.json"
    proof.write_text(
        json.dumps(
            dict(
                complete=True,
                benchmark=str(benchmark),
                signal_arrays=len(arrays),
                code_hashes={"producer.py": digest},
                all_arrays_equal_nan=False,
            )
        )
    )
    return proof


def test_risk_boolean_and_unknown():
    """Existing risk is a Boolean, NaN stays unknown, and a pretend bitmask is rejected."""
    np.testing.assert_array_equal(
        basic.risk_features(np.array([0.0, 1.0, np.nan])), [[0, 0], [1, 0], [np.nan, 1]]
    )
    for values in [np.array([2.0]), np.array([np.inf]), np.array([[0.0]])]:
        with pytest.raises(ValueError, match="invalid_technical"):
            basic.risk_features(values)


def test_dated_records_hash_and_missing(tmp_path):
    """Explicit source bytes are audited, while changed manifest hashes fail closed."""
    raw = tmp_path / "raw.json"
    envelope(
        raw,
        "income_vip",
        [
            dict(
                ts_code="000001.SZ",
                end_date="20231231",
                ann_date="20240401",
                report_type="1",
                comp_type="1",
                n_income_attr_p=10,
            )
        ],
    )
    audit = []
    assert len(list(basic.dated_records([str(raw)], "income_vip", audit))) == 1
    assert len(audit) == 1
    assert list(basic.dated_records([], "income_vip", audit)) == []
    manifest = tmp_path / "collection.json"
    manifest.write_text(
        json.dumps(
            dict(
                partitions=[
                    dict(
                        request=dict(api="income_vip", parameters=dict(period="20231231")),
                        files=[dict(path="raw.json", sha256="bad")],
                    )
                ]
            )
        )
    )
    with pytest.raises(ValueError, match="raw_hash_mismatch"):
        list(basic.dated_records([str(manifest)], "income_vip", audit))


def test_weekly_patterns_keeps_unqualified_stocks(tmp_path):
    """Zero positive hits produce a zero pattern without deleting the requested origin day."""
    from stockops.screen.return80.signals import NAMES

    panel, source, roots, benchmark = fixture(tmp_path)
    signals, _ = build_signals(panel, source, roots, benchmark, tmp_path / "signals")
    for name in NAMES:
        signals[name][:] = False
    members = [dict(ts_code="000001.SZ", l2_code="801010.SI", in_date="20200101", out_date=None)]
    patterns, scores = basic.weekly_patterns(panel, signals, np.array([149]), members)
    assert patterns[149, 0] == 0
    assert np.isfinite(scores[149, 0])
    assert np.isnan(scores[:149]).all()


def test_additional_unknown_and_disclosure_lag(tmp_path):
    """No source stays unknown; a valid report cannot activate on its announcement date."""
    calendar = ["20240104", "20240105", "20240108"]
    panel = SimpleNamespace(
        calendar=calendar,
        codes=["000001.SZ"],
        values=dict(close=np.ones((3, 1)), sector_index=np.zeros((3, 1)), valid=np.ones((3, 1))),
    )
    indicator = tmp_path / "indicator.json"
    envelope(
        indicator,
        "fina_indicator_vip",
        [
            dict(
                ts_code="000001.SZ",
                ann_date="20240105",
                end_date="20230930",
                dt_netprofit_yoy=10.0,
                netprofit_yoy=20.0,
            )
        ],
    )
    output = tmp_path / "output"
    output.mkdir()
    result, audit = basic.additional_columns(
        panel,
        np.array([1, 2]),
        np.array([0, 0]),
        dict(fina_indicator_vip=[str(indicator)]),
        output,
        [],
    )
    assert np.isnan(result[0]).all()
    np.testing.assert_allclose(result[1, 8:10], [0.1, -0.1])
    assert np.isnan(result[:, :8]).all() and np.isnan(result[:, 10]).all()
    assert audit["forecasts"]["source_rows"] == 0


def test_financial_collection_preserves_conflicting_versions(tmp_path):
    """Economic duplicate identities survive ingestion and a short page proves completion."""
    fields = ["ts_code", "end_date", "ann_date", "n_income_attr_p", "report_type", "comp_type"]
    response = dict(
        code=0,
        data=dict(
            fields=fields,
            items=[
                ["000001.SZ", "20231231", "20240401", 10, "1", "1"],
                ["000001.SZ", "20231231", "20240401", 12, "1", "1"],
            ],
        ),
    )
    client = Client(FakeTransport([response]), RawStore(tmp_path / "raw"))
    receipt = basic.collect_financial_period(
        client, "income_vip", "20231231", fields, tmp_path / "receipt", page_size=3
    )
    rows = list(basic.dated_records([str(receipt)], "income_vip", []))
    assert len(rows) == 2 and rows[0]["n_income_attr_p"] != rows[1]["n_income_attr_p"]


def test_financial_collection_rejects_repeated_page(tmp_path):
    """An ignored offset cannot silently create a complete financial partition."""
    fields = ["ts_code", "end_date"]
    response = dict(code=0, data=dict(fields=fields, items=[["000001.SZ", "20231231"]]))
    client = Client(FakeTransport([response, response]), RawStore(tmp_path / "raw"))
    with pytest.raises(ValueError, match="repeated_full_page"):
        basic.collect_financial_period(
            client, "income_vip", "20231231", fields, tmp_path / "receipt", page_size=1
        )
    assert not (tmp_path / "receipt/collection.json").exists()


def test_build_weekly_artifact_and_manifest(tmp_path, monkeypatch):
    """Real formulas emit 47+2 columns on the supplied broader axis and hash every artifact."""
    from stockops.screen.return80.additional_features import ADDITIONAL_FEATURE_NAMES
    from stockops.screen.return80.signals import NAMES

    from tushare_entitlement_hub.strategy_discovery.nonlinear_scoring import NONLINEAR_FEATURE_NAMES

    panel, source, roots, benchmark = fixture(tmp_path)
    monkeypatch.setattr(basic, "load_panel", lambda path: panel)
    signal_root = tmp_path / "signals"
    signals, _ = build_signals(panel, source, roots, benchmark, signal_root)
    signals["technical_risk"][149, 0] = np.nan
    signals["technical_risk"].flush()
    signal_proof = signal_proof_fixture(tmp_path / "signal-proof", benchmark, signal_root)
    universe = tmp_path / "universe"
    universe.mkdir()
    for name, values in dict(
        weekly_candidate_day_indices=[148, 149],
        weekly_candidate_security_indices=[0, 0],
        weekly_origin_indices=[0, 1],
        calendar=panel.calendar,
        security_codes=panel.codes,
    ).items():
        np.save(universe / (name + ".npy"), values)
    members = tmp_path / "members.json"
    envelope(
        members,
        "index_member_all",
        [dict(ts_code="000001.SZ", l2_code="801010.SI", in_date="20200101", out_date=None)],
    )
    original = tmp_path / "original.json"
    names = [
        *NONLINEAR_FEATURE_NAMES,
        *["signal:" + name for name in NAMES],
        *ADDITIONAL_FEATURE_NAMES,
    ]
    original.write_text(json.dumps(dict(feature_names=names)))
    panel_root = tmp_path / "panel"
    panel_root.mkdir()
    (panel_root / "panel.json").write_text("{}")
    specification = tmp_path / "spec.json"
    specification.write_text(
        json.dumps(
            dict(
                panel=str(panel_root),
                universe=str(universe),
                signals=str(signal_root),
                signals_proof=str(signal_proof),
                feature_archives=str(source),
                members=[str(members)],
                original_dataset=str(original),
                raw_sources={},
            )
        )
    )
    output = tmp_path / "result"
    basic.build_basic_features(specification, output)
    values = np.load(output / "X.npy")
    assert values.shape == (2, 49) and not np.isinf(values).any()
    assert np.isnan(values[1, 47]) and values[1, 48] == 1
    complete = json.loads((output / "complete.json").read_text())
    assert complete["rows"] == 2 and complete["columns"] == 49
    for name, expected in complete["files"].items():
        assert basic.file_digest(output / name) == expected
    assert json.loads((output / "feature_names.json").read_text())[:47] == names
    with pytest.raises(ValueError, match="fresh_basic"):
        basic.build_basic_features(specification, output)


@pytest.mark.parametrize("defect", ["api", "status", "fields", "period", "rows", "more", "offset"])
def test_source_partition_contract_failures(tmp_path, defect):
    """Matching hashes cannot authorize wrong API, failed status, incomplete fields or pages."""
    raw = tmp_path / "raw.json"
    fields = ["ts_code", "ann_date", "end_date", "report_type", "comp_type", "n_income_attr_p"]
    envelope(
        raw,
        "income_vip",
        [dict(zip(fields, ["000001.SZ", "20240401", "20231231", "1", "1", 10], strict=True))],
        dict(period="20231231", offset=0, limit=2000),
    )
    document = json.loads(raw.read_text())
    partition = dict(
        request=dict(
            api="income_vip", parameters=dict(period="20231231"), fields=fields, page_size=2000
        ),
        rows=1,
        files=[dict(path="raw.json")],
    )
    if defect == "api":
        document["api_name"] = "daily"
    if defect == "status":
        document["response"]["code"] = 1
    if defect == "fields":
        partition["request"]["fields"] += ["missing_field"]
    if defect == "period":
        document["parameters"]["period"] = "20230930"
    if defect == "rows":
        partition["rows"] = 999
    if defect == "more":
        document["response"]["data"]["has_more"] = True
    if defect == "offset":
        document["parameters"]["offset"] = 2000
    raw.write_text(json.dumps(document))
    partition["files"][0]["sha256"] = basic.file_digest(raw)
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps(dict(partitions=[partition])))
    audit = []
    with pytest.raises(ValueError, match="basic_source"):
        list(basic.dated_records([str(manifest)], "income_vip", audit))
    assert audit == []


@pytest.mark.parametrize("defect", ["api", "status", "fields"])
def test_direct_raw_contract_failures(tmp_path, defect):
    """Direct raw inputs also require successful matching API and every consumed field."""
    raw = tmp_path / "raw.json"
    envelope(
        raw,
        "income_vip",
        [
            dict(
                ts_code="000001.SZ",
                ann_date="20240401",
                end_date="20231231",
                report_type="1",
                comp_type="1",
                n_income_attr_p=10,
            )
        ],
    )
    x = json.loads(raw.read_text())
    if defect == "api":
        x["api_name"] = "daily"
    if defect == "status":
        x["response"]["code"] = 1
    if defect == "fields":
        x["response"]["data"]["fields"][-1] = "wrong_field"
    raw.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="raw_identity"):
        list(basic.dated_records([str(raw)], "income_vip", []))


def test_nonempty_wrong_api_manifest_rejects(tmp_path):
    """A nonempty path list with no requested API cannot masquerade as a successful empty source."""
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps(dict(partitions=[dict(request=dict(api="daily"), files=[])])))
    with pytest.raises(ValueError, match="missing_api_partition"):
        list(basic.dated_records([str(manifest)], "income_vip", []))


@pytest.mark.parametrize("api,cap", list(basic.FINANCIAL_ROW_CAPS.items()))
def test_financial_silent_cap_and_real_versions(api, cap):
    """An oversized response at the real cap fails; revised values remain distinct versions."""
    with pytest.raises(ValueError, match="silent_row_cap"):
        basic.financial_page_keys(api, [{"value": 1}] * cap, cap + 1)
    assert len(basic.financial_page_keys(api, [{"value": 1}] * (cap - 1), cap + 1)) == 1
    assert len(basic.financial_page_keys(api, [{"value": 1}] * cap, cap)) == 1
    assert (
        len(
            basic.financial_page_keys(
                api,
                [{"value": 1, "update_flag": "0"}, {"value": 1, "update_flag": "1"}, {"value": 2}],
                2000,
            )
        )
        == 3
    )
    assert basic.financial_page_keys("daily_basic", [{"value": 1}], 1) == set()


@pytest.mark.parametrize("revision", ["exact", "value", "update_flag"])
def test_cross_page_economic_versions(tmp_path, revision):
    """Exact overlapping records fail; profit and update_flag revisions remain intact."""
    fields = [
        "ts_code",
        "end_date",
        "ann_date",
        "n_income_attr_p",
        "report_type",
        "comp_type",
        "update_flag",
    ]
    first = ["000001.SZ", "20231231", "20240401", 10, "1", "1", "0"]
    second = ["000002.SZ", "20231231", "20240401", 20, "1", "1", "0"]
    repeated = first.copy()
    if revision == "value":
        repeated[3] = 12
    if revision == "update_flag":
        repeated[-1] = "1"
    responses = [
        dict(code=0, data=dict(fields=fields, items=items))
        for items in [[first, second], [repeated]]
    ]
    client = Client(FakeTransport(responses), RawStore(tmp_path / "raw"))
    if revision != "exact":
        receipt = basic.collect_financial_period(
            client, "income_vip", "20231231", fields, tmp_path / "receipt", page_size=2
        )
        assert len(list(basic.dated_records([str(receipt)], "income_vip", []))) == 3
    else:
        with pytest.raises(ValueError, match="cross_page_economic_duplicate"):
            basic.collect_financial_period(
                client, "income_vip", "20231231", fields, tmp_path / "receipt", page_size=2
            )
        assert not (tmp_path / "receipt/collection.json").exists()
    files = []
    for index, items in enumerate([[first, second], [repeated]]):
        raw = tmp_path / f"page-{index}.json"
        envelope(
            raw,
            "income_vip",
            [dict(zip(fields, item, strict=True)) for item in items],
            dict(period="20231231", offset=index * 2, limit=2),
        )
        files.append(dict(path=str(raw), sha256=basic.file_digest(raw)))
    partition = dict(
        request=dict(
            api="income_vip", parameters=dict(period="20231231"), fields=fields, page_size=2
        ),
        files=files,
        rows=3,
    )
    if revision != "exact":
        assert (
            len(basic.validated_partition(tmp_path / "manifest.json", partition, "income_vip")) == 2
        )
    else:
        with pytest.raises(ValueError, match="cross_page_economic_duplicate"):
            basic.validated_partition(tmp_path / "manifest.json", partition, "income_vip")


@pytest.mark.parametrize(
    "defect", [None, "benchmark", "benchmark_hash", "array", "snapshot", "incomplete"]
)
def test_signal_execution_proof(tmp_path, defect):
    """Completed evidence binds raw CSI300 and output bytes; relabeling or tampering fails."""
    benchmark = tmp_path / "benchmark.json"
    envelope(benchmark, "index_daily", [dict(ts_code="000300.SH", trade_date="20240102", close=1)])
    signals = tmp_path / "signals"
    signals.mkdir()
    np.save(signals / "signal.npy", np.array([[1.0]]))
    proof = signal_proof_fixture(tmp_path / "proof", benchmark, signals)
    if defect == "benchmark":
        envelope(
            benchmark, "index_daily", [dict(ts_code="000985.CSI", trade_date="20240102", close=1)]
        )
        (proof.parent / "raw-source-hashes.json").write_text(
            json.dumps({str(benchmark): basic.file_digest(benchmark)})
        )
    if defect == "benchmark_hash":
        benchmark.write_text(benchmark.read_text() + " ")
    if defect == "array":
        np.save(signals / "signal.npy", np.array([[0.0]]))
    if defect == "snapshot":
        next((proof.parent / "code-snapshot").iterdir()).write_text("changed")
    if defect == "incomplete":
        value = json.loads(proof.read_text())
        value["complete"] = False
        proof.write_text(json.dumps(value))
    if defect is None:
        hashes = basic.validated_signals(proof, signals, ["20240102"], ("signal",))
        assert hashes[str(benchmark)] == basic.file_digest(benchmark)
    else:
        with pytest.raises(ValueError, match="basic_signals"):
            basic.validated_signals(proof, signals, ["20240102"], ("signal",))
