"""把冻结的逐证券技术特征补齐为可核验、只读的日×证券矩阵。"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..regime_research.panel import Panel, load_panel
from ..strategy_optimization import features as feature_module
from ..strategy_optimization.features import FloatArray, compute_features

VERSION = "strategy-discovery-features/v1"


def _digest_json(value: Any) -> str:
    """对明确选取的公开元数据生成稳定摘要；例如同键异序的字典摘要相同。"""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _code_fingerprint() -> str:
    """哈希转换器和指标实现的源码字符串；例如修改 RSI 实现将改变输出摘要。"""
    return _digest_json(
        [inspect.getsource(sys.modules[__name__]), inspect.getsource(feature_module)]
    )


def _read_contract(source: Path, panel_path: Path) -> dict[str, Any]:
    """读取并严格对齐日历、证券顺序及来源；例如证券倒序会抛出 ValueError。"""
    source_manifest = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    panel_manifest = json.loads((panel_path / "panel.json").read_text(encoding="utf-8"))
    calendar = source_manifest.get("calendar")
    codes = source_manifest.get("securities")
    fingerprint = source_manifest.get("source_fingerprint")
    if (
        not isinstance(calendar, list)
        or not calendar
        or any(not isinstance(day, str) or not re.fullmatch(r"\d{8}", day) for day in calendar)
        or calendar != sorted(set(calendar))
        or not isinstance(codes, list)
        or not codes
        or any(
            not isinstance(code, str) or not re.fullmatch(r"\d{6}\.(SH|SZ)", code) for code in codes
        )
        or len(codes) != len(set(codes))
        or not isinstance(fingerprint, str)
        or not re.fullmatch(r"[a-f0-9]{64}", fingerprint)
        or calendar != panel_manifest.get("calendar")
        or codes != panel_manifest.get("codes")
        or fingerprint != panel_manifest.get("source_fingerprint")
    ):
        raise ValueError("feature_source_panel_contract_mismatch")
    value_names = panel_manifest.get("values")
    if (
        not isinstance(value_names, list)
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
            for name in value_names
        )
        or len(value_names) != len(set(value_names))
        or "raw_close" not in value_names
    ):
        raise ValueError("invalid_panel_value_names")
    return {
        "calendar": calendar,
        "securities": codes,
        "source_fingerprint": fingerprint,
        "panel_values": value_names,
    }


def _input_fingerprint(source: Path, contract: dict[str, Any]) -> str:
    """流式哈希清单列出的公开 NPZ，不扫描其他文件；例如单个行情变化使摘要变化。"""
    digest = hashlib.sha256(_digest_json(contract).encode())
    for code in contract["securities"]:
        digest.update(code.encode())
        file_digest = hashlib.sha256()
        with (source / f"{code}.npz").open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def _matrix_fingerprint(values: FloatArray) -> str:
    """按证券逐列摘要矩阵内容，限制临时内存；例如相同矩阵不同磁盘布局摘要相同。"""
    digest = hashlib.sha256()
    for column in range(values.shape[1]):
        digest.update(np.ascontiguousarray(values[:, column]).tobytes())
    return digest.hexdigest()


def _feature_names(contract: dict[str, Any]) -> list[str]:
    """推导指标函数提供且 Panel 缺少的字段；例如已有 close 则不再保存 close。"""
    return sorted(set(compute_features([], ["20000101"])) - set(contract["panel_values"]))


def _validate_panel(panel: Panel, contract: dict[str, Any], panel_path: Path) -> None:
    """校验传入面板与只读文件来源一致；例如手工替换 raw_close 会被拒绝。"""
    if (
        panel.calendar != contract["calendar"]
        or panel.codes != contract["securities"]
        or set(panel.values) != set(contract["panel_values"])
    ):
        raise ValueError("feature_panel_identity_mismatch")
    expected_shape = (len(panel.calendar), len(panel.codes))
    for values in panel.values.values():
        if values.shape != expected_shape or values.flags.writeable:
            raise ValueError("feature_panel_must_be_readonly_aligned")
    raw_close = panel.values["raw_close"]
    if (
        not isinstance(raw_close, np.memmap)
        or Path(str(raw_close.filename)).resolve() != panel_path / "values-raw_close.npy"
    ):
        raise ValueError("feature_panel_must_have_verified_origin")


def prepare_features(source: Path, panel_path: Path, output: Path) -> Path:
    """把已有公开 NPZ 的缺失技术字段转为矩阵，清单最后落盘，失败目录拒绝复用。

    输入源目录（或 dataset.json）、Panel 目录和空输出目录，返回 OUT/features.json。
    已有完整缓存也重新验证输入、代码、形状及内容，不请求网络或读取凭据。
    写入按证券逐列映射，内存不随证券数量增加；保留 NaN 与复权价格原值。
    """
    source = source.resolve()
    if source.name == "dataset.json":
        source = source.parent
    panel_path, output = panel_path.resolve(), output.resolve()
    contract = _read_contract(source, panel_path)
    panel = load_panel(panel_path)
    _validate_panel(panel, contract, panel_path)
    manifest_path = output / "features.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source") != str(source) or manifest.get("panel_path") != str(panel_path):
            raise ValueError("feature_cache_origin_mismatch")
        load_features(output, panel)
        return manifest_path
    if output.exists() and any(output.iterdir()):
        raise ValueError("incomplete_features_use_new_directory")
    fingerprint = _input_fingerprint(source, contract)
    names = _feature_names(contract)
    shape = (len(contract["calendar"]), len(contract["securities"]))
    output.mkdir(parents=True, exist_ok=True)
    matrices = {
        name: np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            output / f"values-{name}.npy",
            mode="w+",
            dtype=np.float64,
            shape=shape,
            fortran_order=True,
        )
        for name in names
    }
    for column, code in enumerate(contract["securities"]):
        with np.load(source / f"{code}.npz", allow_pickle=False) as archive:
            for name, matrix in matrices.items():
                if name not in archive:
                    raise ValueError("source_feature_missing")
                values = archive[name]
                if values.shape != (shape[0],) or values.dtype.kind not in "fiu":
                    raise ValueError("source_feature_shape_or_dtype_invalid")
                matrix[:, column] = values
        if (column + 1) % 250 == 0:
            for matrix in matrices.values():
                matrix.flush()
    for matrix in matrices.values():
        matrix.flush()
    if _input_fingerprint(source, contract) != fingerprint:
        raise ValueError("source_features_changed_during_preparation")
    if _read_contract(source, panel_path) != contract:
        raise ValueError("source_contract_changed_during_preparation")
    manifest = {
        "version": VERSION,
        "complete": True,
        **contract,
        "source": str(source),
        "panel_path": str(panel_path),
        "feature_names": names,
        "shape": list(shape),
        "dtype": "float64",
        "code_fingerprint": _code_fingerprint(),
        "input_fingerprint": fingerprint,
        "matrix_fingerprints": {
            name: _matrix_fingerprint(matrix) for name, matrix in matrices.items()
        },
    }
    temporary_manifest = output / "features.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest_path


def load_features(path: Path, panel: Panel) -> dict[str, FloatArray]:
    """复核来源、代码和矩阵后只读加载补充字段，返回 Panel.values 与新增字段的映射。

    例如结果['ma60'] 为只读日×证券 mmap，结果['close'] 复用 panel.values['close']。
    path 可为输出目录或 features.json；来源缺失、损坏、版本漂移均拒绝加载。
    """
    path = path.resolve()
    if path.name == "features.json":
        path = path.parent
    manifest_path = path / "features.json"
    if not manifest_path.is_file():
        raise ValueError("incomplete_features_manifest_missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("version") != VERSION
        or manifest.get("complete") is not True
        or manifest.get("code_fingerprint") != _code_fingerprint()
    ):
        raise ValueError("feature_cache_version_or_code_mismatch")
    source = Path(manifest["source"]).resolve()
    panel_path = Path(manifest["panel_path"]).resolve()
    contract = _read_contract(source, panel_path)
    _validate_panel(panel, contract, panel_path)
    names = _feature_names(contract)
    shape = (len(panel.calendar), len(panel.codes))
    if (
        any(manifest.get(key) != value for key, value in contract.items())
        or manifest.get("feature_names") != names
        or manifest.get("shape") != list(shape)
        or manifest.get("dtype") != "float64"
        or manifest.get("input_fingerprint") != _input_fingerprint(source, contract)
        or set(manifest.get("matrix_fingerprints", {})) != set(names)
    ):
        raise ValueError("feature_cache_contract_or_input_mismatch")
    expected_files = {"features.json", *(f"values-{name}.npy" for name in names)}
    if {entry.name for entry in path.iterdir()} != expected_files:
        raise ValueError("feature_cache_file_set_mismatch")
    result = dict(panel.values)
    for name in names:
        matrix = np.load(path / f"values-{name}.npy", mmap_mode="r", allow_pickle=False)
        if (
            matrix.shape != shape
            or matrix.dtype != np.dtype("float64")
            or _matrix_fingerprint(matrix) != manifest["matrix_fingerprints"][name]
        ):
            raise ValueError("feature_cache_matrix_integrity_mismatch")
        result[name] = matrix
    return result
