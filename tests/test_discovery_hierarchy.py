"""独立L2成员缓存的离线历史区间、来源绑定和损坏拒绝测试。"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from tushare_entitlement_hub.regime_research.panel import load_panel
from tushare_entitlement_hub.strategy_discovery import hierarchy


def _member(**updates):
    """生成合法已退出成员；例如out_date=20200105表示5日起不再属于该L2。"""
    return {
        "ts_code": "000001.SZ",
        "l1_code": "801010.SI",
        "l2_code": "801011.SI",
        "in_date": "20200102",
        "out_date": "20200105",
        "is_new": "N",
    } | updates


def _write_members(source, rows, name="members-example.json"):
    """保存确定性公开成功包络；例如两条成员字典变为fields/items形式本地JSON。"""
    fields = ["ts_code", "l1_code", "l2_code", "in_date", "out_date", "is_new"]
    (source / name).write_text(
        json.dumps(
            {
                "response": {
                    "code": 0,
                    "data": {
                        "fields": fields,
                        "items": [[row.get(field) for field in fields] for row in rows],
                    },
                }
            }
        )
    )


def _edit_json(path, updates):
    """合并测试清单字段；例如complete=False将完整清单改成未完成状态。"""
    manifest = json.loads(path.read_text())
    manifest.update(updates)
    path.write_text(json.dumps(manifest))


@pytest.fixture
def hierarchy_paths(tmp_path):
    """创建六日两证券真实mmap面板及跨L2切换成员；输出源/面板/新输出三个路径。"""
    source, panel_path, output = (tmp_path / name for name in ("members", "panel", "hierarchy"))
    source.mkdir()
    panel_path.mkdir()
    _write_members(
        source,
        [
            _member(),
            _member(l2_code="801012.SI", in_date="20200105", out_date=None, is_new="Y"),
            _member(ts_code="000002.SZ", in_date="20200103", out_date=None, is_new="Y"),
        ],
    )
    (panel_path / "panel.json").write_text(
        json.dumps(
            {
                "calendar": [f"2020010{day}" for day in range(2, 8)],
                "codes": ["000001.SZ", "000002.SZ"],
                "source_fingerprint": "a" * 64,
                "values": ["raw_close", "sector_index"],
                "signals": [],
                "market": [],
            }
        )
    )
    np.save(panel_path / "values-raw_close.npy", np.full((6, 2), 10.0))
    np.save(panel_path / "values-sector_index.npy", np.zeros((6, 2)))
    return source, panel_path, output


def test_real_roundtrip_dates_unknown_and_readonly(hierarchy_paths):
    """纳入日含、退出日不含、当前成员不倒填；输出只读且不修改原Panel映射。"""
    source, panel_path, output = hierarchy_paths
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    panel = load_panel(panel_path)
    result = hierarchy.load_hierarchy(manifest_path, panel)
    assert set(result) == {"sector_level2_index"}
    assert set(panel.values) == {"raw_close", "sector_index"}
    matrix = result["sector_level2_index"]
    np.testing.assert_equal(matrix[:, 0], [0, 0, 0, 1, 1, 1])
    np.testing.assert_equal(matrix[:, 1], [np.nan, 0, 0, 0, 0, 0])
    assert isinstance(matrix, np.memmap) and matrix.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        matrix[0, 0] = 99
    assert hierarchy.prepare_hierarchy(source, panel_path, output) == manifest_path
    manifest = json.loads(manifest_path.read_text())
    assert manifest["historical_pit_certified"] is False
    assert manifest["level2_codes"] == ["801011.SI", "801012.SI"]
    assert manifest["source_fingerprint"] == "a" * 64
    assert manifest["quality"]["known_security_days"] == 11
    assert manifest["quality"]["unknown_security_days"] == 1


@pytest.mark.parametrize("different_parent", [False, True])
def test_duplicate_intervals_and_overlapping_conflicts(hierarchy_paths, different_parent):
    """相同记录去重，同L2重叠可共存，不同L2或不同父行业的重叠日均保持未知。"""
    source, panel_path, output = hierarchy_paths
    overlapping = _member(l2_code="801012.SI", in_date="20200104", out_date="20200106")
    if different_parent:
        overlapping.update(l1_code="801020.SI", l2_code="801011.SI")
    _write_members(
        source,
        [
            _member(),
            _member(),
            overlapping,
            _member(l2_code="801012.SI", in_date="20200105", out_date=None, is_new="Y"),
        ],
    )
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    matrix = hierarchy.load_hierarchy(output, load_panel(panel_path))[hierarchy.FIELD]
    np.testing.assert_equal(matrix[:, 0], [0, 0, np.nan, np.nan if different_parent else 1, 1, 1])
    quality = json.loads(manifest_path.read_text())["quality"]
    assert quality["duplicate_rows"] == 1
    assert quality["conflicting_security_days"] == (2 if different_parent else 1)
    assert np.isnan(matrix[:, 1]).all()


@pytest.mark.parametrize(
    "updates",
    [
        {"in_date": None},
        {"in_date": "20200230"},
        {"in_date": "2020-01-02"},
        {"out_date": None},
        {"out_date": "20200102"},
        {"out_date": "20191231"},
        {"out_date": "20200230"},
        {"is_new": None},
        {"is_new": "unknown"},
        {"is_new": "Y"},
        {"l1_code": None},
        {"l2_code": "../elsewhere"},
        {"ts_code": "920001.BJ"},
        {"ts_code": None},
    ],
)
def test_invalid_member_rows_do_not_invent_history(hierarchy_paths, updates):
    """缺日期、矛盾状态、错误代码或非法区间仅被诊断，不把未知记录当当前完整成员。"""
    source, panel_path, output = hierarchy_paths
    _write_members(source, [_member(), _member(**updates)])
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    quality = json.loads(manifest_path.read_text())["quality"]
    assert quality["invalid_rows"] == 1
    assert quality["unique_valid_intervals"] == 1
    matrix = hierarchy.load_hierarchy(output, load_panel(panel_path))[hierarchy.FIELD]
    assert np.isnan(matrix[3:, 0]).all()


def test_original_unknown_l1_days_and_outside_security(hierarchy_paths):
    """原L1未知或非法标签日仍保持未知；面板外有效证券计数但不扩大证券轴。"""
    source, panel_path, output = hierarchy_paths
    _write_members(source, [_member(out_date=None, is_new="Y"), _member(ts_code="600009.SH")])
    old_labels = np.zeros((6, 2))
    old_labels[:, 0] = [np.nan, -1, np.inf, 0.5, 0, 0]
    np.save(panel_path / "values-sector_index.npy", old_labels)
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    matrix = hierarchy.load_hierarchy(output, load_panel(panel_path))[hierarchy.FIELD]
    np.testing.assert_equal(matrix[:, 0], [np.nan, np.nan, np.nan, np.nan, 0, 0])
    quality = json.loads(manifest_path.read_text())["quality"]
    assert quality["source_securities_outside_panel"] == 1
    assert quality["level1_unknown_security_days"] == 4


def test_unrelated_files_not_read_or_hashed_and_sensitive_source_rejected(hierarchy_paths):
    """只处理明确members文件，无关凭据文件不读；成员包络含敏感字段则在哈希前拒绝。"""
    source, panel_path, output = hierarchy_paths
    (source / "unrelated.json").write_text('{"token":"not-real"}')
    manifest = hierarchy.prepare_hierarchy(source, panel_path, output)
    assert list(json.loads(manifest.read_text())["source_files"]) == ["members-example.json"]
    _edit_json(source / "members-example.json", {"token": "not-real"})
    with pytest.raises(ValueError, match="凭据"):
        hierarchy.load_hierarchy(output, load_panel(panel_path))


@pytest.mark.parametrize("kind", ["missing", "empty", "symlink", "unsafe_name"])
def test_invalid_source_files_rejected(hierarchy_paths, kind):
    """缺成员文件、无有效记录、符号链接或不安全文件名均不能构造可信缓存。"""
    source, panel_path, output = hierarchy_paths
    member_path = source / "members-example.json"
    if kind == "missing":
        member_path.unlink()
    elif kind == "empty":
        _write_members(source, [])
    elif kind == "unsafe_name":
        member_path.rename(source / "members-bad name.json")
    else:
        target = source / "other.json"
        member_path.rename(target)
        member_path.symlink_to(target)
    with pytest.raises(ValueError, match="hierarchy"):
        hierarchy.prepare_hierarchy(source, panel_path, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"calendar": []},
        {"calendar": ["20200230"]},
        {"calendar": ["20200103", "20200102"]},
        {"codes": []},
        {"codes": ["bad"]},
        {"codes": ["000001.SZ", "000001.SZ"]},
        {"source_fingerprint": None},
        {"source_fingerprint": "old"},
        {"values": ["raw_close"]},
    ],
)
def test_invalid_panel_contract_rejected(hierarchy_paths, updates):
    """面板坐标、来源摘要和所需原L1缺失时，在转换之前拒绝。"""
    source, panel_path, output = hierarchy_paths
    _edit_json(panel_path / "panel.json", updates)
    with pytest.raises(ValueError, match="panel_contract"):
        hierarchy.prepare_hierarchy(source, panel_path, output)


@pytest.mark.parametrize("change", ["axes", "shape", "array", "writable", "different_path"])
def test_loader_rejects_unbound_panel(hierarchy_paths, change):
    """传入面板日期错位、数组替换、可写或不同文件来源均不能冒充冻结Panel。"""
    source, panel_path, output = hierarchy_paths
    hierarchy.prepare_hierarchy(source, panel_path, output)
    panel = load_panel(panel_path)
    if change == "axes":
        panel.codes = panel.codes[::-1]
    elif change == "shape":
        panel.values["raw_close"] = panel.values["raw_close"][:1]
    elif change == "array":
        panel.values["raw_close"] = np.ones((6, 2))
    elif change == "writable":
        panel.values["raw_close"] = np.load(panel_path / "values-raw_close.npy", mmap_mode="r+")
    else:
        panel.values["raw_close"] = panel.values["sector_index"]
    with pytest.raises(ValueError, match="hierarchy_panel"):
        hierarchy.load_hierarchy(output, panel)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"version": "old"}, "version_or_code"),
        ({"complete": False}, "version_or_code"),
        ({"historical_pit_certified": True}, "version_or_code"),
        ({"code_fingerprint": "changed"}, "version_or_code"),
        ({"source_files": {}}, "contract_or_input"),
        ({"input_fingerprint": "old"}, "contract_or_input"),
        ({"calendar": []}, "contract_or_input"),
        ({"level2_codes": []}, "contract_or_input"),
        ({"feature_names": ["x"]}, "contract_or_input"),
        ({"shape": [1, 2]}, "contract_or_input"),
        ({"dtype": "int16"}, "contract_or_input"),
        ({"matrix_fingerprint": "bad"}, "matrix_integrity"),
    ],
)
def test_cache_manifest_drift_rejected_on_reuse(hierarchy_paths, updates, match):
    """旧版本、篡改坐标、标签映射或内容摘要均不能走已有缓存直接复用。"""
    source, panel_path, output = hierarchy_paths
    manifest = hierarchy.prepare_hierarchy(source, panel_path, output)
    _edit_json(manifest, updates)
    with pytest.raises(ValueError, match=match):
        hierarchy.prepare_hierarchy(source, panel_path, output)


@pytest.mark.parametrize("change", ["value", "shape", "dtype", "missing", "extra"])
def test_cache_matrix_corruption_rejected(hierarchy_paths, change):
    """矩阵值、尺寸、类型、缺文件或多文件都必须拒绝。"""
    source, panel_path, output = hierarchy_paths
    hierarchy.prepare_hierarchy(source, panel_path, output)
    matrix_path = output / f"values-{hierarchy.FIELD}.npy"
    if change == "missing":
        matrix_path.unlink()
    elif change == "extra":
        (output / "unfinished.tmp").touch()
    else:
        np.save(
            matrix_path,
            np.ones(
                (1, 2) if change == "shape" else (6, 2),
                dtype=np.float32 if change == "dtype" else np.float64,
            ),
        )
    with pytest.raises(ValueError, match="file_set|matrix_integrity"):
        hierarchy.load_hierarchy(output, load_panel(panel_path))


@pytest.mark.parametrize("change", ["source", "panel"])
def test_input_content_drift_rejected(hierarchy_paths, change):
    """成员内容或原L1值改变即使坐标不变，也不允许静默复用旧矩阵。"""
    source, panel_path, output = hierarchy_paths
    hierarchy.prepare_hierarchy(source, panel_path, output)
    if change == "source":
        _write_members(source, [_member(in_date="20200103")])
    else:
        np.save(panel_path / "values-sector_index.npy", np.ones((6, 2)))
    with pytest.raises(ValueError, match="contract_or_input"):
        hierarchy.load_hierarchy(output, load_panel(panel_path))


def test_incomplete_or_wrong_origin_cache_rejected(hierarchy_paths):
    """半成品输出不可覆盖且无法加载，完整清单的来源目录改变也拒绝。"""
    source, panel_path, output = hierarchy_paths
    output.mkdir()
    incomplete = output / "leftover.tmp"
    incomplete.touch()
    with pytest.raises(ValueError, match="incomplete_hierarchy"):
        hierarchy.prepare_hierarchy(source, panel_path, output)
    with pytest.raises(ValueError, match="incomplete_hierarchy"):
        hierarchy.load_hierarchy(output, load_panel(panel_path))
    incomplete.unlink()
    manifest = hierarchy.prepare_hierarchy(source, panel_path, output)
    _edit_json(manifest, {"source": str(source.parent)})
    with pytest.raises(ValueError, match="origin_mismatch"):
        hierarchy.prepare_hierarchy(source, panel_path, output)


@pytest.mark.parametrize("change_source", [False, True])
def test_input_changes_during_build_do_not_publish_manifest(hierarchy_paths, change_source):
    """转换时源文件或面板契约变化，不发布完成清单，防止混合两轮来源。"""
    source, panel_path, output = hierarchy_paths
    if change_source:
        intervals, fingerprints, diagnostics = hierarchy._source_intervals(source)
        context = patch.object(
            hierarchy,
            "_source_intervals",
            side_effect=[(intervals, fingerprints, diagnostics), (intervals, {}, diagnostics)],
        )
    else:
        contract = hierarchy._panel_contract(panel_path)
        context = patch.object(hierarchy, "_panel_contract", side_effect=[contract, {}])
    with context, pytest.raises(ValueError, match="changed_during_preparation"):
        hierarchy.prepare_hierarchy(source, panel_path, output)
    assert not (output / "hierarchy.json").exists()


def test_stable_metadata_and_array_fingerprints():
    """JSON键序与矩阵内存布局不影响摘要，合法闰日正确识别。"""
    assert hierarchy._digest({"a": 1, "b": 2}) == hierarchy._digest({"b": 2, "a": 1})
    values = np.arange(12.0).reshape(3, 4)
    assert hierarchy._matrix_fingerprint(values) == hierarchy._matrix_fingerprint(
        np.asfortranarray(values)
    )
    assert hierarchy._valid_day("20200229")


@pytest.mark.parametrize("in_panel", [False, True])
def test_verified_parent_mapping_blocks_conflicting_original_parent(hierarchy_paths, in_panel):
    """显式面板映射或同源features清单均可认证；源L1与当日Panel父级不一致则L2未知。"""
    source, panel_path, output = hierarchy_paths
    _write_members(source, [_member(out_date=None, is_new="Y")])
    panel_manifest = json.loads((panel_path / "panel.json").read_text())
    if in_panel:
        metadata_path = panel_path / "panel.json"
        _edit_json(metadata_path, {"sector_codes": ["801010.SI", "801020.SI"]})
    else:
        metadata_path = source.parent / "features" / "dataset.json"
        metadata_path.parent.mkdir()
        metadata_path.write_text(
            json.dumps(
                {
                    "calendar": panel_manifest["calendar"],
                    "securities": panel_manifest["codes"],
                    "source_fingerprint": panel_manifest["source_fingerprint"],
                    "sector_codes": ["801010.SI", "801020.SI"],
                }
            )
        )
    level1 = np.zeros((6, 2))
    level1[2:4, 0] = 1
    np.save(panel_path / "values-sector_index.npy", level1)
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["parent_mapping_verified"] is True
    assert manifest["parent_mapping_source"] == str(metadata_path)
    assert len(manifest["parent_mapping_fingerprint"]) == 64
    assert manifest["quality"]["parent_mismatch_security_days"] == 2
    matrix = hierarchy.load_hierarchy(output, load_panel(panel_path))[hierarchy.FIELD]
    np.testing.assert_equal(matrix[:, 0], [0, 0, np.nan, np.nan, 0, 0])
    _edit_json(metadata_path, {"additional_public_note": "changed"})
    with pytest.raises(ValueError, match="contract_or_input"):
        hierarchy.load_hierarchy(output, load_panel(panel_path))


@pytest.mark.parametrize("codes", [None, [], ["bad"], ["801010.SI", "801010.SI"]])
def test_invalid_explicit_parent_mapping_rejected(hierarchy_paths, codes):
    """父级映射存在却无效时不能退化为未认证后继续构建。"""
    source, panel_path, output = hierarchy_paths
    _edit_json(panel_path / "panel.json", {"sector_codes": codes})
    with pytest.raises(ValueError, match="parent_mapping_invalid"):
        hierarchy.prepare_hierarchy(source, panel_path, output)


def test_parent_dataset_wrong_origin_is_not_used(hierarchy_paths):
    """固定位置原特征清单的日历/证券/指纹不匹配时拒绝，不能用别轮元数据猜父级。"""
    source, panel_path, output = hierarchy_paths
    metadata_path = source.parent / "features" / "dataset.json"
    metadata_path.parent.mkdir()
    metadata_path.write_text(json.dumps({"sector_codes": ["801010.SI"]}))
    with pytest.raises(ValueError, match="parent_mapping_binding_mismatch"):
        hierarchy.prepare_hierarchy(source, panel_path, output)


def test_missing_parent_mapping_is_explicitly_uncertified(hierarchy_paths):
    """未找到明确映射时只保留原L1有效掩码，清单明确记为父类未认证，不猜标签排序。"""
    source, panel_path, output = hierarchy_paths
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["parent_mapping_verified"] is False
    assert manifest["parent_mapping_fingerprint"] is None
    assert manifest["parent_mapping_source"] is None
    assert manifest["parent_level1_codes"] == []
    assert "L1 parent identity is uncertified" in " ".join(manifest["limitations"])


def test_same_level2_multiple_global_parents_invalidates_every_member(hierarchy_paths):
    """两证券自身L1均匹配Panel，但同日L2跨两个父行业时，该L2全部成员必须未知。"""
    source, panel_path, output = hierarchy_paths
    _write_members(
        source,
        [
            _member(out_date=None, is_new="Y"),
            _member(ts_code="000002.SZ", l1_code="801020.SI", out_date=None, is_new="Y"),
        ],
    )
    _edit_json(panel_path / "panel.json", {"sector_codes": ["801010.SI", "801020.SI"]})
    np.save(panel_path / "values-sector_index.npy", np.tile([0.0, 1.0], (6, 1)))
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    matrix = hierarchy.load_hierarchy(output, load_panel(panel_path))[hierarchy.FIELD]
    assert np.isnan(matrix).all()
    quality = json.loads(manifest_path.read_text())["quality"]
    assert quality["conflicting_level2_days"] == 6
    assert quality["global_parent_conflict_security_days"] == 12
    assert quality["parent_mismatch_security_days"] == 0


def test_nonoverlapping_global_parent_reclassification_remains_valid(hierarchy_paths):
    """同L2历史父级变更在退出日交接而不重叠时，前后成员均有效，不按全历史父级数封禁。"""
    source, panel_path, output = hierarchy_paths
    _write_members(
        source,
        [
            _member(),
            _member(
                ts_code="000002.SZ",
                l1_code="801020.SI",
                in_date="20200105",
                out_date=None,
                is_new="Y",
            ),
        ],
    )
    _edit_json(panel_path / "panel.json", {"sector_codes": ["801010.SI", "801020.SI"]})
    np.save(panel_path / "values-sector_index.npy", np.tile([0.0, 1.0], (6, 1)))
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    matrix = hierarchy.load_hierarchy(output, load_panel(panel_path))[hierarchy.FIELD]
    np.testing.assert_equal(matrix[:, 0], [0, 0, 0, np.nan, np.nan, np.nan])
    np.testing.assert_equal(matrix[:, 1], [np.nan, np.nan, np.nan, 0, 0, 0])
    quality = json.loads(manifest_path.read_text())["quality"]
    assert quality["conflicting_level2_days"] == 0
    assert quality["global_parent_conflict_security_days"] == 0


@pytest.mark.parametrize("outside_panel", [False, True])
def test_global_conflict_only_invalidates_overlap_days_even_from_outside_panel(
    hierarchy_paths, outside_panel
):
    """跨父级冲突只污染实际重叠两日；面板外有效记录也能证明分类冲突，不忽略反证。"""
    source, panel_path, output = hierarchy_paths
    conflicting_code = "600009.SH" if outside_panel else "000002.SZ"
    _write_members(
        source,
        [
            _member(out_date=None, is_new="Y"),
            _member(
                ts_code=conflicting_code,
                l1_code="801020.SI",
                in_date="20200104",
                out_date="20200106",
            ),
            _member(
                ts_code="000002.SZ",
                l1_code="801020.SI",
                l2_code="801021.SI",
                in_date="20200106",
                out_date=None,
                is_new="Y",
            ),
        ],
    )
    _edit_json(panel_path / "panel.json", {"sector_codes": ["801010.SI", "801020.SI"]})
    np.save(panel_path / "values-sector_index.npy", np.tile([0.0, 1.0], (6, 1)))
    manifest_path = hierarchy.prepare_hierarchy(source, panel_path, output)
    matrix = hierarchy.load_hierarchy(output, load_panel(panel_path))[hierarchy.FIELD]
    np.testing.assert_equal(matrix[:, 0], [0, 0, np.nan, np.nan, 0, 0])
    np.testing.assert_equal(matrix[:, 1], [np.nan, np.nan, np.nan, np.nan, 1, 1])
    quality = json.loads(manifest_path.read_text())["quality"]
    assert quality["conflicting_level2_days"] == 2
    assert quality["global_parent_conflict_security_days"] == (2 if outside_panel else 4)


def test_global_parent_conflicts_do_not_use_future_intervals():
    """未来出现跨父级重叠时，不回写此前日期；完整矩阵与历史前缀结果一致。"""
    intervals = [
        ("000001.SZ", "801010.SI", "801011.SI", "20200102", "99999999"),
        ("000002.SZ", "801020.SI", "801011.SI", "20200106", "99999999"),
    ]
    calendar = [f"2020010{day}" for day in range(2, 8)]
    complete = hierarchy._level2_parent_conflicts(intervals, calendar)["801011.SI"]
    np.testing.assert_equal(complete, [False, False, False, False, True, True])
    prefix = hierarchy._level2_parent_conflicts(intervals, calendar[:4])["801011.SI"]
    np.testing.assert_equal(prefix, complete[:4])
