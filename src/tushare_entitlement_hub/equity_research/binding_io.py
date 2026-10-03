"""绑定构建的只读投影、容量边界和文件发布，不访问token或联网资源。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import duckdb

from ..product_foundation.config import validate_batch_size


@dataclass(frozen=True, kw_only=True)
class BuildLimits:
    """限制单批、单源、全库及实体规模；超界拒绝产出部分配置。"""

    batch_size: int = 128
    maximum_source_rows: int = 2_000_000
    maximum_total_rows: int = 5_000_000
    maximum_entities: int = 50_000
    maximum_row_bytes: int = 65_536

    def __post_init__(self) -> None:
        """验证每个边界为正整数；例：batch_size=0或maximum_entities=True均拒绝。

        输入合法限额得到不可变配置，批大小另外受公共512行上限约束。
        """
        validate_batch_size(self.batch_size)
        if any(
            type(value) is not int or value < 1
            for value in (
                self.maximum_source_rows,
                self.maximum_total_rows,
                self.maximum_entities,
                self.maximum_row_bytes,
            )
        ):
            raise ValueError("binding_limits_invalid")


def check_paths(reference: Path, *outputs: Path) -> None:
    """拒绝路径、软链和硬链冲突，保护输入及独立审计输出。

    输入同一个reference和output抛input_output_path_collision；两个独立路径返回None。
    检查在建目录或写文件之前进行。
    """
    if not reference.is_file():
        raise ValueError("reference_library_missing")
    if Path(str(reference) + ".wal").exists():
        raise ValueError("binding_reference_wal_present")
    paths = (reference, *outputs)
    for index, path in enumerate(paths):
        for other in paths[:index]:
            if path.resolve() == other.resolve() or (
                path.exists() and other.exists() and path.samefile(other)
            ):
                raise ValueError("input_output_path_collision")


def projected_rows(
    connection: duckdb.DuckDBPyConnection,
    api: str,
    fields: set[str],
    limits: BuildLimits,
) -> Iterator[dict[str, Any]]:
    """仅从内部白名单接口投影绑定所需列，按有限批读取并限制单行JSON大小。

    输入daily和{ts_code,source_units}，输出字典迭代器；source_fields和新闻正文不读取。
    SQL参数仅承载限额，标识符先按数据库转义；JSON元数据只解码声明的三个字段。
    """
    names = ",".join('"' + name.replace('"', '""') + '"' for name in sorted(fields))
    table = api.replace('"', '""')
    query = f'SELECT to_json(item) FROM (SELECT {names} FROM accepted."{table}") item'
    cursor = connection.execute(
        "SELECT CASE WHEN octet_length(encode(payload)) <= ? THEN payload END "
        f"FROM ({query}) projected(payload)",
        [limits.maximum_row_bytes],
    )
    while batch := cursor.fetchmany(min(limits.batch_size, 128)):
        for (payload,) in batch:
            if payload is None:
                raise ValueError("binding_row_capacity_exceeded")
            row = json.loads(payload)
            for name in ("source_units", "standard_units", "source_occurrences"):
                if isinstance(row.get(name), str):
                    row[name] = json.loads(row[name])
            yield row


def publish_files(outputs: tuple[tuple[Path, str], ...]) -> None:
    """先完整写入独立临时文件，再按报告、配置顺序原子替换，配置作为提交标记。

    输入两个路径及JSON文本，输出两个完整文件；写临时文件失败不会截断原输出。
    替换失败仍传播OSError，临时文件在成功与失败路径都清理，不遗留缓存。
    """
    temporaries: list[tuple[Path, Path]] = []
    try:
        for destination, payload in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=".binding-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                temporaries.append((temporary, destination))
                handle.write(payload)
        for temporary, destination in temporaries:
            temporary.replace(destination)
    finally:
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)
