"""私有研究库快照：提交后重开验证，再以文件替换原子发布，正式路径不暴露半成品。"""

from __future__ import annotations

import fcntl
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from ..product_foundation.progress import ProgressCallback
from .persistence_runtime import PersistenceProgress

# 独立进程持有POSIX记录锁，才能检测同一Python进程内DuckDB已有的写连接。
# 父进程关闭管道（包括异常退出）即释放锁；只传文件描述符，不传路径或数据。
_LOCK_PROGRAM = """
import fcntl, os, sys
try:
    fcntl.lockf(int(sys.argv[1]), fcntl.LOCK_SH | fcntl.LOCK_NB)
except OSError:
    sys.stdout.buffer.write(b'0')
    sys.stdout.buffer.flush()
else:
    sys.stdout.buffer.write(b'1')
    sys.stdout.buffer.flush()
    sys.stdin.buffer.read()
"""


def _version(path: Path) -> tuple[int, ...] | None:
    """记录普通单链接文件身份，拒绝符号/硬链接以免发布破坏别名；例：缺文件返回None。

    返回设备、inode、大小和纳秒修改时间，用于锁定后/发布前检测路径被替换或修改。
    原生数据库锁负责排斥SQL写者，路径锁负责串行化本工具的发布者。
    """
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise ValueError("research_output_path_unsafe")
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


@contextmanager
def _database_read_lock(path: Path) -> Iterator[subprocess.Popen[bytes]]:
    """由一个短小锁进程阻止原库写者，允许只读连接；例：已有DuckDB写连接时立即busy。

    不打开数据库、不联网、不读取环境或载荷。独立进程避免POSIX锁按进程合并的漏洞；
    上下文退出关闭管道并回收子进程，异常或键盘中断同样释放文件描述符与锁。
    """
    with (
        path.open("rb") as source,
        subprocess.Popen(
            [sys.executable, "-c", _LOCK_PROGRAM, str(source.fileno())],
            pass_fds=(source.fileno(),),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as guard,
    ):
        try:
            if guard.stdout.read(1) != b"1":  # type: ignore[union-attr]
                raise ValueError("research_output_busy")
            yield guard
        finally:
            guard.stdin.close()  # type: ignore[union-attr]
            guard.wait()


class ResearchSnapshot:
    """锁内私有副本，调用方完成提交、关闭及完整验证后才可发布。"""

    def __init__(
        self,
        output: Path,
        staged: Path,
        version: tuple[int, ...] | None,
        guard: subprocess.Popen[bytes] | None,
    ) -> None:
        """保存固定路径和原库身份；例：新库version=None，发布时仍要求正式路径不存在。"""
        self.output, self.path, self.version = output, staged, version
        self.guard = guard

    def publish(self, trace: PersistenceProgress) -> None:
        """将已验证且已checkpoint的单文件快照原子替换到正式路径；例：读者只见旧库或新库。

        不再次导入JSON或在大事务内回读；旧run及其他schema随原文件副本完整保留。
        fsync先保证快照内容落盘，替换后的目录同步失败仍是完整的新快照，不存在半个run。
        """
        trace.step("publish_snapshot")
        if self.guard is not None and self.guard.poll() is not None:
            raise ValueError("research_output_lock_lost")
        if _version(self.output) != self.version:
            raise ValueError("research_output_changed")
        if Path(str(self.output) + ".wal").exists() or Path(str(self.path) + ".wal").exists():
            raise ValueError("research_output_recovery_required")
        with self.path.open("rb") as source:
            os.fsync(source.fileno())
        # os.open也适用于POSIX目录；替换前打开，避免发布成功后才发现目录不可访问。
        directory = os.open(self.output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.replace(self.path, self.output)
            trace.published = True
            trace.operation = "sync_directory"
            os.fsync(directory)
        finally:
            os.close(directory)


@contextmanager
def research_snapshot(
    reference: Path,
    output: Path,
    *,
    run_id: str,
    progress: ProgressCallback | None = None,
) -> Iterator[ResearchSnapshot]:
    """锁定正式输出，在其同目录创建0700私有工作区，任何退出都清理副本/WAL/临时块。

    例：已有两个run时复制整个已关闭数据库，新增第三个run验证后一次替换，旧run不变。
    固定零字节路径锁不删除，避免等待者锁旧inode而新写者锁新inode的竞态；它不是数据缓存。
    未恢复WAL、安全路径或锁冲突均明确拒绝，不擅自修复、覆盖或删除已有数据。
    """
    trace = PersistenceProgress("stage", progress, run_id)
    try:
        output = output.parent.resolve() / output.name
        lock_path = output.with_name("." + output.name + ".research.lock")
        for candidate in (output, lock_path, Path(str(output) + ".wal")):
            if reference.resolve() == candidate.resolve() or (
                candidate.exists() and reference.samefile(candidate)
            ):
                raise ValueError("input_output_path_collision")
        output.parent.mkdir(parents=True, exist_ok=True)
        _version(lock_path)
        with ExitStack() as stack:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            stack.callback(os.close, descriptor)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("research_output_busy") from None
            version = _version(output)
            guard = (
                stack.enter_context(_database_read_lock(output)) if version is not None else None
            )
            if Path(str(output) + ".wal").exists():
                raise ValueError("research_output_recovery_required")
            temporary = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="." + output.name + ".stage-", dir=output.parent)
            )
            path = Path(temporary) / output.name
            trace.step("copy_snapshot")
            if version is not None:
                shutil.copy2(output, path)
            snapshot = ResearchSnapshot(output, path, version, guard)
            yield snapshot
    except Exception as error:
        # save/verify/commit错误已有精确阶段，不能被工作区的清理边界改写成stage。
        from .persistence_runtime import PersistenceError

        if isinstance(error, PersistenceError):
            raise
        raise trace.failure(error) from error
