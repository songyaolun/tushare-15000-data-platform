"""从已有公开申万成员区间构建独立L2旁路缓存，不改变原研究面板。"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..raw import assert_public
from ..regime_research.panel import Panel
from ..research_history.collection import response_records
from ..strategy_optimization.features import FloatArray

FIELD = "sector_level2_index"
VERSION = "strategy-discovery-hierarchy/v1"
MemberInterval = tuple[str, str, str, str, str]


def _digest(value: Any) -> str:
    """对已选取的公开元数据生成规范摘要；例如同键异序字典产生相同SHA256。"""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _code_fingerprint() -> str:
    """绑定转换器源码与公共响应读取规则；例如区间解析代码变化使旧缓存失效。"""
    return _digest(
        [
            inspect.getsource(sys.modules[__name__]),
            inspect.getsource(response_records),
            inspect.getsource(assert_public),
        ]
    )


def _valid_day(value: Any) -> bool:
    """检查真实YYYYMMDD日期；例如20200229为真，20200230或空值为假。"""
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def _source_intervals(
    source: Path,
) -> tuple[list[MemberInterval], dict[str, str], dict[str, int]]:
    """只读取members-*.json公开包络，去重并验证证券、层级和历史有效期。

    例N记录[20200102,20200106)在6日已退出；Y缺out_date表示仍有效但不向纳入前回填。
    N缺退出、Y却有退出、缺纳入或is_new未知均计无效，不猜测日期。相同证券/L1/L2/
    区间的重复行去重，允许同L2不同三级行业记录，不读取目录内其他文件或凭据。
    """
    paths = sorted(source.glob("members-*.json"))
    if not paths:
        raise ValueError("hierarchy_member_sources_missing")
    intervals: set[MemberInterval] = set()
    fingerprints = {}
    diagnostics: Counter[str] = Counter(source_rows=0, invalid_rows=0, duplicate_rows=0)
    for path in paths:
        if not re.fullmatch(r"members-[A-Za-z0-9_.-]+\.json", path.name) or path.is_symlink():
            raise ValueError("hierarchy_member_source_path_invalid")
        payload = path.read_bytes()
        envelope = json.loads(payload)
        assert_public(envelope)
        rows = response_records(envelope["response"])
        fingerprints[path.name] = hashlib.sha256(payload).hexdigest()
        for row in rows:
            diagnostics["source_rows"] += 1
            code, level1, level2 = (row.get(name) for name in ("ts_code", "l1_code", "l2_code"))
            start, stop, current = (row.get(name) for name in ("in_date", "out_date", "is_new"))
            stop_missing = stop in (None, "")
            if (
                not isinstance(code, str)
                or not re.fullmatch(r"\d{6}\.(SH|SZ)", code)
                or any(
                    not isinstance(label, str) or not re.fullmatch(r"\d{6}\.SI", label)
                    for label in (level1, level2)
                )
                or not _valid_day(start)
                or current not in ("Y", "N")
                or (current == "N" and stop_missing)
                or (current == "Y" and not stop_missing)
                or (
                    not stop_missing
                    and (not _valid_day(stop) or cast(str, stop) <= cast(str, start))
                )
            ):
                diagnostics["invalid_rows"] += 1
                continue
            interval = (
                code,
                cast(str, level1),
                cast(str, level2),
                cast(str, start),
                "99999999" if stop_missing else cast(str, stop),
            )
            if interval in intervals:
                diagnostics["duplicate_rows"] += 1
            intervals.add(interval)
    diagnostics["unique_valid_intervals"] = len(intervals)
    if not intervals:
        raise ValueError("hierarchy_no_valid_member_intervals")
    return sorted(intervals), fingerprints, dict(diagnostics)


def _matrix_fingerprint(matrix: FloatArray) -> str:
    """逐证券摘要值内容，不复制整个矩阵；例如同值C序与F序矩阵得到相同摘要。"""
    digest = hashlib.sha256()
    for column in range(matrix.shape[1]):
        digest.update(np.ascontiguousarray(matrix[:, column]).tobytes())
    return digest.hexdigest()


def _panel_contract(panel_path: Path, panel: Panel | None = None) -> dict[str, Any]:
    """核对完整日期/证券绑定和L1来源，仅读取清单及两个明确面板数组。

    例传入证券倒序的Panel或人工替换raw_close数组会拒绝；合法load_panel返回值可用。
    原L1矩阵内容摘要进入契约，避免L2在不同原行业缺失范围上被静默复用。
    """
    manifest = json.loads((panel_path / "panel.json").read_text())
    assert_public(manifest)
    calendar, codes, fingerprint = (
        manifest.get(key) for key in ("calendar", "codes", "source_fingerprint")
    )
    if (
        not isinstance(calendar, list)
        or not calendar
        or not all(_valid_day(day) for day in calendar)
        or calendar != sorted(set(calendar))
        or not isinstance(codes, list)
        or not codes
        or any(
            not isinstance(code, str) or not re.fullmatch(r"\d{6}\.(SH|SZ)", code) for code in codes
        )
        or len(codes) != len(set(codes))
        or not isinstance(fingerprint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        or not {"raw_close", "sector_index"}.issubset(manifest.get("values", []))
    ):
        raise ValueError("hierarchy_panel_contract_invalid")
    if panel is None:
        panel = Panel(
            calendar,
            codes,
            {
                name: np.load(panel_path / f"values-{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in ("raw_close", "sector_index")
            },
            {},
            {},
        )
    if panel.calendar != calendar or panel.codes != codes:
        raise ValueError("hierarchy_panel_axes_mismatch")
    for name in ("raw_close", "sector_index"):
        values = panel.values.get(name)
        if (
            not isinstance(values, np.memmap)
            or Path(str(values.filename)).resolve() != panel_path / f"values-{name}.npy"
            or values.shape != (len(calendar), len(codes))
            or values.flags.writeable
        ):
            raise ValueError("hierarchy_panel_origin_or_shape_mismatch")
    return {
        "calendar": calendar,
        "securities": codes,
        "source_fingerprint": fingerprint,
        "level1_matrix_fingerprint": _matrix_fingerprint(panel.values["sector_index"]),
    }


def _parent_mapping(source: Path, panel_path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """从面板显式层级表或原特征清单认证L1整数映射，不凭当前成员排序猜测。

    例原features/dataset.json的calendar/securities/source_fingerprint全部一致时，
    sector_codes第0项认证Panel行业0；来源不存在则明确返回未认证并保留限制。
    只读这两个固定清单且只保存映射和绑定摘要，不读取凭据或扫描其他目录。
    """
    metadata_path = panel_path / "panel.json"
    metadata_payload = metadata_path.read_bytes()
    metadata = json.loads(metadata_payload)
    assert_public(metadata)
    if "sector_codes" not in metadata:
        metadata_path = source.parent / "features" / "dataset.json"
        if not metadata_path.is_file():
            return {
                "parent_mapping_verified": False,
                "parent_level1_codes": [],
                "parent_mapping_source": None,
                "parent_mapping_fingerprint": None,
            }
        metadata_payload = metadata_path.read_bytes()
        metadata = json.loads(metadata_payload)
        assert_public(metadata)
        if any(
            metadata.get(key) != contract.get(key)
            for key in ("calendar", "securities", "source_fingerprint")
        ):
            raise ValueError("hierarchy_parent_mapping_binding_mismatch")
    codes = metadata.get("sector_codes")
    if (
        not isinstance(codes, list)
        or not codes
        or any(not isinstance(code, str) or not re.fullmatch(r"\d{6}\.SI", code) for code in codes)
        or len(codes) != len(set(codes))
    ):
        raise ValueError("hierarchy_parent_mapping_invalid")
    return {
        "parent_mapping_verified": True,
        "parent_level1_codes": codes,
        "parent_mapping_source": str(metadata_path),
        "parent_mapping_fingerprint": hashlib.sha256(metadata_payload).hexdigest(),
    }


def _level2_parent_conflicts(
    intervals: list[MemberInterval],
    calendar: list[str],
) -> dict[str, NDArray[np.bool_]]:
    """按L2和有效日期核对全局父级唯一性，跨证券记录也参与，不用全历史唯一父级限制。

    例L2甲在2日至4日前属于L1甲、4日起属于L1乙，不冲突；若另一证券在3日
    已宣称L2甲属于L1乙，则仅3日冲突，所有当日L2甲成员都须保持未知。
    输入已经过滤无效区间；允许历史非重叠重分类，不使用未来区间污染过去日期。
    """
    days = np.asarray(calendar)
    parents = {label: index for index, label in enumerate(sorted({row[1] for row in intervals}))}
    assigned: dict[str, NDArray[np.int32]] = {}
    conflicts: dict[str, NDArray[np.bool_]] = {}
    for _, level1, level2, start, stop in intervals:
        if level2 not in assigned:
            assigned[level2] = np.full(len(calendar), -1, dtype=np.int32)
            conflicts[level2] = np.zeros(len(calendar), dtype=bool)
        active = (days >= start) & (days < stop)
        parent = parents[level1]
        conflicts[level2] |= active & (assigned[level2] >= 0) & (assigned[level2] != parent)
        assigned[level2][active & (assigned[level2] == -1)] = parent
    return conflicts


def _write_matrix(
    path: Path,
    intervals: list[MemberInterval],
    contract: dict[str, Any],
    panel_path: Path,
) -> tuple[str, dict[str, int], list[str]]:
    """逐证券写入整数标签；重叠不同L2或不同L1父类的日期保持NaN，退出日不包含。

    例A先属L2甲[2日,4日)、后属乙[3日,6日)，输出2日甲、3日NaN、4/5日乙。
    相同L2同L1重叠只是重复覆盖；同日同L2跨证券出现多个L1父级时全组未知。
    原Panel的L1未知日继续未知，绝不填零当作行业0。
    """
    calendar, codes = contract["calendar"], contract["securities"]
    labels = sorted({interval[2] for interval in intervals})
    level2_indices = {label: index for index, label in enumerate(labels)}
    level1_indices = {
        label: index for index, label in enumerate(sorted({item[1] for item in intervals}))
    }
    grouped: defaultdict[str, list[MemberInterval]] = defaultdict(list)
    for interval in intervals:
        grouped[interval[0]].append(interval)
    days = np.asarray(calendar)
    global_conflicts = _level2_parent_conflicts(intervals, calendar)
    old_labels = np.load(panel_path / "values-sector_index.npy", mmap_mode="r", allow_pickle=False)
    matrix = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        path,
        mode="w+",
        dtype=np.float64,
        shape=(len(calendar), len(codes)),
        fortran_order=True,
    )
    diagnostics = {
        "conflicting_security_days": 0,
        "level1_unknown_security_days": 0,
        "known_security_days": 0,
        "unknown_security_days": 0,
        "source_securities_outside_panel": len(set(grouped) - set(codes)),
        "parent_mismatch_security_days": 0,
        "conflicting_level2_days": sum(int(np.sum(values)) for values in global_conflicts.values()),
        "global_parent_conflict_security_days": 0,
    }
    for column, code in enumerate(codes):
        assigned = np.full(len(calendar), -1, dtype=np.int32)
        parents = np.full(len(calendar), -1, dtype=np.int32)
        conflicts = np.zeros(len(calendar), dtype=bool)
        global_parent_conflicts = np.zeros(len(calendar), dtype=bool)
        for _, level1, level2, start, stop in grouped[code]:
            active = (days >= start) & (days < stop)
            global_parent_conflicts |= active & global_conflicts[level2]
            label, parent = level2_indices[level2], level1_indices[level1]
            conflicts |= active & (assigned >= 0) & ((assigned != label) | (parents != parent))
            empty = active & (assigned == -1)
            assigned[empty], parents[empty] = label, parent
        previous = old_labels[:, column]
        level1_known = np.isfinite(previous) & (previous >= 0) & (previous == np.floor(previous))
        parent_matches = np.ones(len(calendar), dtype=bool)
        if contract["parent_mapping_verified"]:
            original_indices = {
                label: index for index, label in enumerate(contract["parent_level1_codes"])
            }
            expected_parents = np.array(
                [original_indices.get(label, -2) for label in level1_indices]
            )
            parent_matches = previous == expected_parents[np.maximum(parents, 0)]
            diagnostics["parent_mismatch_security_days"] += int(
                np.sum((assigned >= 0) & ~conflicts & level1_known & ~parent_matches)
            )
        known = (
            (assigned >= 0) & ~conflicts & ~global_parent_conflicts & level1_known & parent_matches
        )
        diagnostics["global_parent_conflict_security_days"] += int(np.sum(global_parent_conflicts))
        matrix[:, column] = np.where(known, assigned.astype(float), np.nan)
        diagnostics["conflicting_security_days"] += int(np.sum(conflicts))
        diagnostics["level1_unknown_security_days"] += int(np.sum((assigned >= 0) & ~level1_known))
        diagnostics["known_security_days"] += int(np.sum(known))
        diagnostics["unknown_security_days"] += int(np.sum(~known))
    matrix.flush()
    return _matrix_fingerprint(matrix), diagnostics, labels


def prepare_hierarchy(source: Path, panel_path: Path, output: Path) -> Path:
    """把明确本地成员目录转换为只读L2旁路矩阵，清单最后提交，旧目录必须完整核验。

    例prepare_hierarchy(MEMBERS,PANEL,OUT)返回OUT/hierarchy.json；不更改Panel或旧特征。
    相同代码输入可复用，代码/源文件/面板/内容变化或半成品目录均拒绝覆盖。
    """
    source, panel_path, output = source.resolve(), panel_path.resolve(), output.resolve()
    contract = _panel_contract(panel_path)
    contract |= _parent_mapping(source, panel_path, contract)
    manifest_path = output / "hierarchy.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source") != str(source) or manifest.get("panel_path") != str(panel_path):
            raise ValueError("hierarchy_cache_origin_mismatch")
        panel = Panel(
            contract["calendar"],
            contract["securities"],
            {
                name: np.load(panel_path / f"values-{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in ("raw_close", "sector_index")
            },
            {},
            {},
        )
        load_hierarchy(output, panel)
        return manifest_path
    if output.exists() and any(output.iterdir()):
        raise ValueError("incomplete_hierarchy_use_new_directory")
    intervals, sources, source_diagnostics = _source_intervals(source)
    output.mkdir(parents=True, exist_ok=True)
    matrix_fingerprint, matrix_diagnostics, labels = _write_matrix(
        output / f"values-{FIELD}.npy", intervals, contract, panel_path
    )
    latest_contract = _panel_contract(panel_path)
    latest_contract |= _parent_mapping(source, panel_path, latest_contract)
    if _source_intervals(source)[1] != sources or latest_contract != contract:
        raise ValueError("hierarchy_inputs_changed_during_preparation")
    manifest = {
        "version": VERSION,
        "complete": True,
        "historical_pit_certified": False,
        **contract,
        "source": str(source),
        "panel_path": str(panel_path),
        "source_files": sources,
        "input_fingerprint": _digest([contract, sources]),
        "code_fingerprint": _code_fingerprint(),
        "matrix_fingerprint": matrix_fingerprint,
        "feature_names": [FIELD],
        "shape": [len(contract["calendar"]), len(contract["securities"])],
        "dtype": "float64",
        "level2_codes": labels,
        "quality": source_diagnostics | matrix_diagnostics,
        "interval_semantics": "in_date_inclusive_out_date_exclusive",
        "limitations": [
            "SW2021 supplier reconstruction may restate historical taxonomy",
            "unknown intervals and conflicting memberships are not backfilled",
            "L1 parent identity is uncertified when parent_mapping_verified is false",
        ],
    }
    temporary = output / "hierarchy.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)
    return manifest_path


def load_hierarchy(path: Path, panel: Panel) -> dict[str, FloatArray]:
    """复核源文件、面板坐标、代码及矩阵内容后只读加载L2索引。

    例load_hierarchy(OUT,panel)['sector_level2_index'][0,0]返回整数值或NaN；
    字典只含新增字段，调用方可与原features合并，不改变panel.values。
    """
    path = path.resolve()
    if path.name == "hierarchy.json":
        path = path.parent
    manifest_path = path / "hierarchy.json"
    if not manifest_path.is_file():
        raise ValueError("incomplete_hierarchy_manifest_missing")
    manifest = json.loads(manifest_path.read_text())
    assert_public(manifest)
    if (
        manifest.get("version") != VERSION
        or manifest.get("complete") is not True
        or manifest.get("historical_pit_certified") is not False
        or manifest.get("code_fingerprint") != _code_fingerprint()
    ):
        raise ValueError("hierarchy_version_or_code_mismatch")
    contract = _panel_contract(Path(manifest["panel_path"]).resolve(), panel)
    contract |= _parent_mapping(
        Path(manifest["source"]).resolve(), Path(manifest["panel_path"]).resolve(), contract
    )
    intervals, sources, _ = _source_intervals(Path(manifest["source"]).resolve())
    labels = sorted({interval[2] for interval in intervals})
    shape = (len(panel.calendar), len(panel.codes))
    if (
        any(manifest.get(key) != value for key, value in contract.items())
        or manifest.get("source_files") != sources
        or manifest.get("input_fingerprint") != _digest([contract, sources])
        or manifest.get("level2_codes") != labels
        or manifest.get("feature_names") != [FIELD]
        or manifest.get("shape") != list(shape)
        or manifest.get("dtype") != "float64"
    ):
        raise ValueError("hierarchy_contract_or_input_mismatch")
    if {entry.name for entry in path.iterdir()} != {"hierarchy.json", f"values-{FIELD}.npy"}:
        raise ValueError("hierarchy_cache_file_set_mismatch")
    matrix = np.load(path / f"values-{FIELD}.npy", mmap_mode="r", allow_pickle=False)
    if (
        matrix.shape != shape
        or matrix.dtype != np.dtype("float64")
        or _matrix_fingerprint(matrix) != manifest.get("matrix_fingerprint")
    ):
        raise ValueError("hierarchy_matrix_integrity_mismatch")
    return {FIELD: matrix}
