"""冻结特征缓存的离线来源、完整性、内存映射及拒绝复用契约。"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import load_panel
from tushare_entitlement_hub.strategy_discovery import dataset
from tushare_entitlement_hub.strategy_optimization.features import compute_features


@pytest.fixture
def feature_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """创建两证券三日的公开小样本；输出源、Panel 和尚不存在的缓存目录。"""
    source, panel_path, output = (tmp_path / name for name in ("source", "panel", "output"))
    source.mkdir()
    panel_path.mkdir()
    calendar = ["20200102", "20200103", "20200106"]
    codes = ["000001.SZ", "600001.SH"]
    contract = {"calendar": calendar, "source_fingerprint": "a" * 64}
    (source / "dataset.json").write_text(json.dumps({**contract, "securities": codes}))
    (panel_path / "panel.json").write_text(
        json.dumps(
            {
                **contract,
                "codes": codes,
                "values": ["raw_close", "close"],
                "signals": [],
                "market": [],
            }
        )
    )
    for name in ("raw_close", "close"):
        np.save(panel_path / f"values-{name}.npy", np.full((3, 2), 10.0))
    for column, code in enumerate(codes):
        features = compute_features([], calendar)
        features["ma60"] = np.array([np.nan, column + 1.0, column + 2.0])
        np.savez(source / f"{code}.npz", **features)
    return source, panel_path, output


def _edit_manifest(path: Path, updates: dict[str, object]) -> None:
    """合并测试清单字段；例如 {'complete':False} 将缓存标为未完成。"""
    manifest = json.loads(path.read_text())
    manifest.update(updates)
    path.write_text(json.dumps(manifest))


def test_roundtrip_readonly_incremental_fields(feature_paths: tuple[Path, Path, Path]) -> None:
    """完整缓存逐列保留 NaN，复用已有数组，支持文件路径且拒绝写入。"""
    source, panel_path, output = feature_paths
    manifest_path = dataset.prepare_features(source / "dataset.json", panel_path, output)
    panel = load_panel(panel_path)
    features = dataset.load_features(manifest_path, panel)
    np.testing.assert_equal(features["ma60"], [[np.nan, np.nan], [1, 2], [2, 3]])
    assert features["close"] is panel.values["close"]
    assert not (output / "values-close.npy").exists()
    assert isinstance(features["ma60"], np.memmap)
    with pytest.raises(ValueError, match="read-only"):
        features["ma60"][0, 0] = 1
    assert dataset.prepare_features(source, panel_path, output) == manifest_path
    manifest = json.loads(manifest_path.read_text())
    assert manifest["feature_names"] == sorted(
        set(compute_features([], ["20200101"])) - {"close", "raw_close"}
    )
    assert manifest["source_fingerprint"] == "a" * 64
    assert len(manifest["input_fingerprint"]) == len(manifest["code_fingerprint"]) == 64


@pytest.mark.parametrize(
    "updates",
    [
        {"calendar": []},
        {"calendar": ["bad"]},
        {"calendar": ["20200103", "20200102"]},
        {"securities": []},
        {"securities": ["../secret"]},
        {"securities": ["000001.SZ", "000001.SZ"]},
        {"securities": ["600001.SH", "000001.SZ"]},
        {"source_fingerprint": "b" * 64},
        {"source_fingerprint": None},
    ],
)
def test_source_contract_rejected(
    feature_paths: tuple[Path, Path, Path],
    updates: dict[str, object],
) -> None:
    """日历、证券或来源无效时，在创建任何输出前拒绝转换。"""
    source, panel_path, output = feature_paths
    _edit_manifest(source / "dataset.json", updates)
    with pytest.raises(ValueError, match="source_panel_contract"):
        dataset.prepare_features(source, panel_path, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "names", [None, ["raw_close", "../secret"], ["close"], ["raw_close", "raw_close"]]
)
def test_panel_field_names_rejected(
    feature_paths: tuple[Path, Path, Path],
    names: object,
) -> None:
    """未知、重复或包含路径的 Panel 字段拒绝读取。"""
    source, panel_path, output = feature_paths
    _edit_manifest(panel_path / "panel.json", {"values": names})
    with pytest.raises(ValueError, match="invalid_panel_value_names"):
        dataset.prepare_features(source, panel_path, output)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"version": "old"}, "version_or_code"),
        ({"complete": False}, "version_or_code"),
        ({"code_fingerprint": "old"}, "version_or_code"),
        ({"input_fingerprint": "old"}, "contract_or_input"),
        ({"feature_names": ["../secret"]}, "contract_or_input"),
        ({"shape": [2, 3]}, "contract_or_input"),
        ({"dtype": "float32"}, "contract_or_input"),
        ({"source_fingerprint": "old"}, "contract_or_input"),
        ({"matrix_fingerprints": {}}, "contract_or_input"),
    ],
)
def test_cache_manifest_drift_rejected(
    feature_paths: tuple[Path, Path, Path],
    updates: dict[str, object],
    match: str,
) -> None:
    """旧代码、旧输入或篡改的清单不能被已有缓存复用路径绕过。"""
    source, panel_path, output = feature_paths
    manifest = dataset.prepare_features(source, panel_path, output)
    _edit_manifest(manifest, updates)
    with pytest.raises(ValueError, match=match):
        dataset.prepare_features(source, panel_path, output)


@pytest.mark.parametrize("change", ["content", "shape", "dtype", "missing", "extra"])
def test_matrix_corruption_rejected(
    feature_paths: tuple[Path, Path, Path],
    change: str,
) -> None:
    """矩阵值、维度、类型、缺文件和额外残留文件均导致拒绝加载。"""
    source, panel_path, output = feature_paths
    dataset.prepare_features(source, panel_path, output)
    matrix_path = output / "values-ma60.npy"
    if change == "missing":
        matrix_path.unlink()
    elif change == "extra":
        (output / "unfinished.tmp").touch()
    else:
        shape = (1, 2) if change == "shape" else (3, 2)
        dtype = np.float32 if change == "dtype" else np.float64
        np.save(matrix_path, np.ones(shape, dtype=dtype))
    with pytest.raises(ValueError, match="file_set|matrix_integrity"):
        dataset.load_features(output, load_panel(panel_path))


@pytest.mark.parametrize("change", ["missing", "shape", "dtype"])
def test_bad_source_features_leave_unreusable_output(
    feature_paths: tuple[Path, Path, Path],
    change: str,
) -> None:
    """输入特征缺失或维度类型错误使构建失败，未完成目录不能继续复用。"""
    source, panel_path, output = feature_paths
    features = compute_features([], ["20200102", "20200103", "20200106"])
    if change == "missing":
        del features["ma60"]
    else:
        features["ma60"] = np.ones(2) if change == "shape" else np.array(["x"] * 3)
    np.savez(source / "000001.SZ.npz", **features)
    with pytest.raises(ValueError, match="source_feature"):
        dataset.prepare_features(source, panel_path, output)
    assert not (output / "features.json").exists()
    with pytest.raises(ValueError, match="incomplete_features"):
        dataset.prepare_features(source, panel_path, output)
    with pytest.raises(ValueError, match="incomplete_features"):
        dataset.load_features(output, load_panel(panel_path))


def test_source_archive_drift_and_other_files_ignored(
    feature_paths: tuple[Path, Path, Path],
) -> None:
    """仅授权 NPZ 内容参与输入摘要；无关文件不读不哈希，行情修改必须拒绝复用。"""
    source, panel_path, output = feature_paths
    dataset.prepare_features(source, panel_path, output)
    (source / "unrelated-private-file").write_text("not part of the dataset")
    dataset.load_features(output, load_panel(panel_path))
    with (source / "000001.SZ.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="contract_or_input"):
        dataset.load_features(output, load_panel(panel_path))


@pytest.mark.parametrize("change", ["calendar", "codes", "keys", "shape", "writable", "origin"])
def test_unverified_panel_rejected(
    feature_paths: tuple[Path, Path, Path],
    change: str,
) -> None:
    """传入 Panel 的坐标、字段、只读属性和 raw_close 文件来源须可核验。"""
    source, panel_path, output = feature_paths
    dataset.prepare_features(source, panel_path, output)
    panel = load_panel(panel_path)
    if change == "calendar":
        panel.calendar = panel.calendar[::-1]
    elif change == "codes":
        panel.codes = panel.codes[::-1]
    elif change == "keys":
        del panel.values["close"]
    elif change == "shape":
        panel.values["close"] = panel.values["close"][:1]
    elif change == "writable":
        panel.values["close"] = np.ones((3, 2))
    else:
        values = np.ones((3, 2))
        values.flags.writeable = False
        panel.values["raw_close"] = values
    with pytest.raises(ValueError, match="feature_panel"):
        dataset.load_features(output, panel)


def test_manifest_origin_swap_rejected(feature_paths: tuple[Path, Path, Path]) -> None:
    """已有缓存宣称不同源目录时，在复用入口拒绝。"""
    source, panel_path, output = feature_paths
    manifest = dataset.prepare_features(source, panel_path, output)
    _edit_manifest(manifest, {"source": str(source.parent)})
    with pytest.raises(ValueError, match="origin_mismatch"):
        dataset.prepare_features(source, panel_path, output)


@pytest.mark.parametrize("contract_changed", [False, True])
def test_source_changes_during_preparation_rejected(
    feature_paths: tuple[Path, Path, Path],
    contract_changed: bool,
) -> None:
    """转换期间行情或源清单变化时不发布完整清单。"""
    source, panel_path, output = feature_paths
    if contract_changed:
        contract = dataset._read_contract(source, panel_path)
        context = patch.object(dataset, "_read_contract", side_effect=[contract, {}])
    else:
        context = patch.object(dataset, "_input_fingerprint", side_effect=["first", "second"])
    with context, pytest.raises(ValueError, match="changed_during_preparation"):
        dataset.prepare_features(source, panel_path, output)
    assert not (output / "features.json").exists()


def test_streaming_batch_flush_and_deterministic_hash(
    feature_paths: tuple[Path, Path, Path],
) -> None:
    """250证券触发分批flush，摘要不依赖矩阵内存顺序或JSON键顺序。"""
    source, panel_path, output = feature_paths
    codes = [f"{number:06d}.SZ" for number in range(250)]
    original = source / "000001.SZ.npz"
    for code in codes:
        target = source / f"{code}.npz"
        if target != original:
            target.symlink_to(original)
    _edit_manifest(source / "dataset.json", {"securities": codes})
    _edit_manifest(panel_path / "panel.json", {"codes": codes})
    for name in ("raw_close", "close"):
        np.save(panel_path / f"values-{name}.npy", np.ones((3, 250)))
    dataset.prepare_features(source, panel_path, output)
    assert dataset.load_features(output, load_panel(panel_path))["ma60"].shape == (3, 250)
    values = np.arange(12.0).reshape(3, 4)
    assert dataset._matrix_fingerprint(values) == dataset._matrix_fingerprint(
        np.asfortranarray(values)
    )
    assert dataset._digest_json({"a": 1, "b": 2}) == dataset._digest_json({"b": 2, "a": 1})
