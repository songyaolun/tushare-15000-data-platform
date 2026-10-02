"""初始化可观测计数；不输出记录内容、凭据或不可序列化对象。"""

from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]


class ProcessingProgress:
    """只累积处理量，最终计数可进入稳定摘要，实时事件交由调用方输出。"""

    def __init__(self, callback: ProgressCallback | None) -> None:
        """输入可选事件回调，输出从零开始的计数器；None 表示静默处理。"""
        self.callback = callback
        self.source_api: str | None = None
        self.counts = {
            "processed_apis": 0,
            "audit_only_apis": 0,
            "scanned_rows": 0,
            "source_batches": 0,
        }

    def emit(self, stage: str) -> None:
        """发送独立事件副本，不允许回调改变内部计数；输入 start，输出起始事件。

        回调为 None 时返回 None；事件仅含阶段、接口名称和处理计数。
        """
        if self.callback is not None:
            self.callback({"stage": stage, "source_api": self.source_api, **self.counts})

    def batch_completed(self, rows: int) -> None:
        """一批事实已写入事务后增加行数及批次数；输入64，输出累计行数加64的事件。"""
        self.counts["scanned_rows"] += rows
        self.counts["source_batches"] += 1
        self.emit("source_batch")

    def interface_completed(self, audit_only: bool) -> None:
        """完成一个接口的处理并报告进度；输入True，输出审计接口数与处理接口数各加1。"""
        self.counts["processed_apis"] += 1
        self.counts["audit_only_apis"] += int(audit_only)
        self.emit("coverage")
