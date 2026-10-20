"""Failure boundaries for portable inputs, stage dispatch and source receipts."""

import json
from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pytest
from test_return80_research_collection import plan_fixture, response
from test_return80_research_dataset import arrays_fixture, block_fixture

from tushare_entitlement_hub.client import Client, FakeTransport
from tushare_entitlement_hub.raw import RawStore
from tushare_entitlement_hub.return80_research import cli, collection, dataset, features, sources
from tushare_entitlement_hub.strategy_discovery.replay_inputs import file_digest


@pytest.mark.parametrize(
    "command,module,function,arguments",
    [
        ("import-training", cli, "import_training_bundle", ["--bundle", "bundle"]),
        ("append-features", cli, "append_features", ["--dataset", "data", "--block", "block"]),
        (
            "map-predictions",
            cli,
            "map_predictions",
            ["--dataset", "data", "--predictions", "prediction"],
        ),
        ("export-replay", cli, "export_replay_inputs", ["--cache-root", "cache"]),
        ("make-plan", sources, "make_plan", ["--start", "20200101", "--end", "20261016"]),
        (
            "materialize-sources",
            sources,
            "materialize_sources",
            ["--bootstrap", "b", "--collection", "c"],
        ),
        ("prepare-sources", sources, "prepare_sources", ["--sources", "s"]),
    ],
)
def test_cli_dispatch_preserves_paths(tmp_path, monkeypatch, command, module, function, arguments):
    """Installed commands dispatch exactly once to the declared stage, preserving chosen outputs."""
    operation = Mock(return_value=tmp_path / "complete.json")
    monkeypatch.setattr(module, function, operation)
    cli.main([command, *arguments, "--output", str(tmp_path / "out")])
    operation.assert_called_once()


def test_cli_rebuild_dispatch_and_extra_argument_rejection(tmp_path, monkeypatch):
    """Rebuild is imported only on demand; stray arguments cannot be silently ignored."""
    from tushare_entitlement_hub.return80_research import rebuild

    operation = Mock(return_value=tmp_path / "complete")
    monkeypatch.setattr(rebuild, "rebuild_matrices", operation)
    cli.main(["rebuild-matrices", "--spec", "spec", "--output", "out"])
    operation.assert_called_once()
    with pytest.raises(SystemExit):
        cli.main(["export-replay", "--cache-root", "cache", "--output", "out", "--typo"])


def test_cli_explicit_collection_is_offline_under_fake_transport(tmp_path, monkeypatch):
    """Even the opted-in command is tested with a fake HTTPS boundary and no credentials."""
    plan = plan_fixture(tmp_path)
    monkeypatch.setattr(
        cli, "HttpTransport", lambda *a, **k: nullcontext(FakeTransport([response([])]))
    )
    cli.main(["collect", "--plan", str(plan), "--output", str(tmp_path / "out"), "--live"])
    assert (tmp_path / "out/collection.json").exists()


@pytest.mark.parametrize("kind", ["plan", "receipt", "fields", "schema", "api"])
def test_collection_conflicting_receipts_and_identity(tmp_path, kind):
    """Changed plans/receipts, missing requested fields and wrong API never pass as complete."""
    plan = plan_fixture(tmp_path)
    root = tmp_path / "out"
    if kind == "fields":
        fake = {"code": 0, "data": {"fields": ["ts_code", "trade_date"], "items": []}}
        with pytest.raises(ValueError, match="missing_requested_fields"):
            collection.collect_plan(
                Client(FakeTransport([fake]), RawStore(root / "raw")), plan, root
            )
        return
    collection.collect_plan(
        Client(FakeTransport([response([])]), RawStore(root / "raw")), plan, root
    )
    if kind == "plan":
        plan.write_text(plan.read_text() + " ")
        with pytest.raises(ValueError, match="plan_changed"):
            collection.collect_plan(Client(FakeTransport([]), RawStore(root / "raw")), plan, root)
    elif kind == "receipt":
        receipt = root / "partition-000000.json"
        value = json.loads(receipt.read_text())
        value["request"]["api"] = "different"
        receipt.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="receipt_mismatch"):
            collection.collect_plan(Client(FakeTransport([]), RawStore(root / "raw")), plan, root)
    else:
        manifest = root / "collection.json"
        value = json.loads(manifest.read_text())
        if kind == "schema":
            value["schema"] = "wrong"
        else:
            record = value["partitions"][0]["files"][0]
            path = root / record["path"]
            payload = json.loads(path.read_text())
            payload["api_name"] = "wrong"
            path.write_text(json.dumps(payload))
            record["sha256"] = file_digest(path)
        manifest.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="manifest" if kind == "schema" else "api_mismatch"):
            collection.read_collection(root, "daily_basic")


def test_bad_training_start_bundle_and_block_headers(tmp_path):
    """Unavailable training start, incomplete bundle and misdeclared CSV cannot be imported."""
    arrays = arrays_fixture()
    arrays["calendar"][0] = "20211017"
    arrays["candidate_dates"][0] = "20211017"
    with pytest.raises(ValueError, match="training_start"):
        dataset.validate_arrays(arrays)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SHA256SUMS").write_text("")
    with pytest.raises(ValueError, match="incomplete_bundle"):
        dataset.import_training_bundle(bundle, tmp_path / "out")
    np.savez(bundle / "training.npz", **arrays_fixture())
    (bundle / "original-protocol.json").write_text(
        json.dumps({"feature_names": ["wrong"], "candidate_rows": 3, "fit_rows": 2})
    )
    (bundle / "SHA256SUMS").write_text(
        "".join(
            file_digest(bundle / name) + "  " + name + "\n"
            for name in ("training.npz", "original-protocol.json")
        )
    )
    with pytest.raises(ValueError, match="protocol_mismatch"):
        dataset.import_training_bundle(bundle, tmp_path / "out")
    data = tmp_path / "data"
    dataset.save_dataset(data, arrays_fixture(), {})
    block = block_fixture(tmp_path, [])
    value = json.loads(block.read_text())
    csv = block.parent / value["file"]
    csv.write_text("wrong,header\n")
    value["sha256"] = file_digest(csv)
    block.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="csv_columns"):
        dataset.append_features(data, block, tmp_path / "out")


def test_daily_feature_missing_values_and_recipe_validation(tmp_path):
    """Off-axis rows are ignored, None stays unknown, and incomplete formula contracts fail."""
    values = features.daily_feature_values(
        [
            {"trade_date": "20211017", "ts_code": "A", "factor": 1},
            {"trade_date": "20211018", "ts_code": "A", "factor": None},
        ],
        ["20211018"],
        ["A"],
        field="factor",
    )
    assert np.isnan(values).all()
    data = tmp_path / "data"
    dataset.save_dataset(data, arrays_fixture(), {})
    recipe = tmp_path / "recipe.json"
    for spec, match in [
        ({}, "feature_spec"),
        (
            {"schema": "return80-daily-features/v1", "features": [{"name": "a"}, {"name": "a"}]},
            "duplicate_daily_feature_name",
        ),
        ({"schema": "return80-daily-features/v1", "features": [{"name": "a"}]}, "formula_or_unit"),
    ]:
        recipe.write_text(json.dumps(spec))
        with pytest.raises(ValueError, match=match):
            features.build_daily_block(data, tmp_path / "not-used", recipe, tmp_path / "out")
