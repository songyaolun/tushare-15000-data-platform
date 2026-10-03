"""上下文的常量空间多重集摘要与可溢写精确集合；不缓存全量Python对象。"""

from __future__ import annotations

import sqlite3

from ..store import digest

DIGEST_ALGORITHM = "sha256-multiset-sum-squares-v1"
DIGEST_MASK = (1 << 256) - 1
SET_CACHE_KIB = 2048
SET_COMMIT_INTERVAL = 512


class MultisetDigest:
    """仅保留计数、模2**256的和与平方和，状态大小不随输入数量增长。"""

    def __init__(self) -> None:
        """初始化空多重集；例如没有输入时count=0，摘要可确定性重建。"""
        self.count = 0
        self.total = 0
        self.squares = 0

    def add(self, fingerprint: str) -> None:
        """累加调用方提供的SHA256叶摘要，保留重复次数而不保存叶列表。

        例如先加入A再加入B，与B再A的结果相同；再加入A会改变计数和最终摘要。
        """
        value = int(fingerprint, 16)
        self.count += 1
        self.total = (self.total + value) & DIGEST_MASK
        self.squares = (self.squares + value * value) & DIGEST_MASK

    def hexdigest(self) -> str:
        """对算法标识及固定大小累计状态生成SHA256；例如同一多重集总返回同一64位值。"""
        return digest(
            [
                DIGEST_ALGORITHM,
                self.count,
                f"{self.total:064x}",
                f"{self.squares:064x}",
            ]
        )


class DiskKeySet:
    """私有临时SQLite精确集合，页缓存2MiB；关闭连接时SQLite自动删除临时库。"""

    def __init__(self) -> None:
        """创建非内存模式的临时库并启用页溢写，不访问reference或用户缓存。

        例如加入百万个不同键会增加临时磁盘页，而非创建百万个Python集合元素。
        初始化异常会先关闭已打开的连接；调用者最终须调用close。
        """
        try:
            self.connection = sqlite3.connect("", cached_statements=8)
        except sqlite3.Error:
            raise ValueError("research_context_spill_failed") from None
        self.pending = 0
        try:
            for statement in (
                f"PRAGMA cache_size=-{SET_CACHE_KIB}",
                "PRAGMA cache_spill=ON",
                "PRAGMA mmap_size=0",
                "PRAGMA temp_store=FILE",
                "PRAGMA journal_mode=OFF",
                "PRAGMA synchronous=OFF",
                "CREATE TABLE keys (namespace TEXT NOT NULL, value BLOB NOT NULL, "
                "PRIMARY KEY(namespace,value)) WITHOUT ROWID",
            ):
                self._execute(statement)
        except BaseException:
            self.connection.close()
            raise

    def _execute(
        self,
        statement: str,
        parameters: tuple[str | bytes, ...] = (),
    ) -> sqlite3.Cursor:
        """只执行内部固定SQL，磁盘或SQLite故障转为固定错误；例如磁盘写满不回显键内容。"""
        try:
            return self.connection.execute(statement, parameters)
        except sqlite3.Error:
            raise ValueError("research_context_spill_failed") from None

    def add(self, namespace: str, value: bytes) -> bool:
        """按完整字节精确去重，固定批次提交；例如同一命名空间两次加入b'A'返回True/False。

        引用使用完整规范JSON字节作键，不以摘要碰撞或概率结构替代精确去重。
        """
        added = (
            self._execute(
                "INSERT OR IGNORE INTO keys VALUES (?,?)",
                (namespace, value),
            ).rowcount
            == 1
        )
        self.pending += 1
        if self.pending == SET_COMMIT_INTERVAL:
            self._execute("COMMIT")
            self.pending = 0
        return added

    def contains(self, namespace: str, value: bytes) -> bool:
        """精确查询已有键且不取回全量集合；例如未登记的本源字段返回False。"""
        return (
            self._execute(
                "SELECT 1 FROM keys WHERE namespace=? AND value=?",
                (namespace, value),
            ).fetchone()
            is not None
        )

    def close(self) -> None:
        """释放私有临时库及页缓存，可重复调用；例如异常路径close后连接不能再查询。"""
        self.connection.close()
