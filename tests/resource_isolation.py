"""在独立且受限的 Python 进程执行完整 pytest 用例，严格核对 JUnit 与资源证据。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

CHILD_NODE_VARIABLE = "TUSHARE_ISOLATED_TEST_NODEID"
EVIDENCE_VARIABLE = "TUSHARE_TEST_ISOLATION_DIR"
MAXIMUM_VIRTUAL_BYTES = 4 * 1024**3


class IsolationError(RuntimeError):
    """子进程未完整通过指定用例，例如跳过一项也不能视为成功。"""


def bounded_address_limits(soft: int, hard: int) -> tuple[int, int]:
    """计算不超过 4 GiB 且不提高继承限制的软硬上限。

    例如输入 1 GiB/2 GiB 返回原值；输入无限/无限返回 4 GiB/4 GiB。
    """
    bounded_hard = (
        MAXIMUM_VIRTUAL_BYTES
        if hard == resource.RLIM_INFINITY
        else min(hard, MAXIMUM_VIRTUAL_BYTES)
    )
    bounded_soft = bounded_hard if soft == resource.RLIM_INFINITY else min(soft, bounded_hard)
    return bounded_soft, bounded_hard


def apply_child_limits(directory: Path, nodeid: str) -> None:
    """在加载 pytest 前约束当前子进程，并保存实际限制及 Python PID。

    例如继承 1 GiB 软上限时仍保持 1 GiB，不会为了执行用例提高内存上限。
    """
    inherited = resource.getrlimit(resource.RLIMIT_AS)
    limits = bounded_address_limits(*inherited)
    resource.setrlimit(resource.RLIMIT_AS, limits)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    report = {
        "nodeid": nodeid,
        "python_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "inherited_address_limits_bytes": inherited,
        "address_limits_bytes": resource.getrlimit(resource.RLIMIT_AS),
        "numeric_library_threads": 1,
    }
    (directory / "child-start.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


def process_memory(pid: int, proc_root: Path = Path("/proc")) -> dict[str, int] | None:
    """读取指定 Python PID 的内核内存指标，进程退出时返回 None。

    例如状态中的 VmHWM=1024 kB 返回相应整数；不会扫描启动器或其他任务进程。
    """
    try:
        lines = (proc_root / str(pid) / "status").read_text().splitlines()
    except FileNotFoundError:
        return None
    names = {"VmSize", "VmPeak", "VmRSS", "VmHWM", "Threads"}
    return {
        name: int(value.split()[0])
        for line in lines
        for name, _, value in [line.partition(":")]
        if name in names
    }


def wait_for_child(process: subprocess.Popen, directory: Path) -> dict:
    """等待直接启动的 Python 子进程，记录采样和该子进程独立的内核 RSS 峰值。

    例如一个正常退出的短进程返回 exit_code=0；父运行被中断时会回收子进程。
    """
    started = time.monotonic()
    first = None
    maxima = {"VmSize": 0, "VmPeak": 0, "VmRSS": 0, "VmHWM": 0}
    try:
        with (directory / "resources.jsonl").open("w") as stream:
            while True:
                ended, status, usage = os.wait4(process.pid, os.WNOHANG)
                if ended:
                    process.returncode = os.waitstatus_to_exitcode(status)
                    break
                values = process_memory(process.pid)
                if values is not None:
                    sample = {
                        "elapsed_seconds": time.monotonic() - started,
                        "python_pid": process.pid,
                        **values,
                    }
                    if first is None:
                        first = sample
                    for key in maxima:
                        maxima[key] = max(maxima[key], values.get(key, 0))
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                time.sleep(0.05)
    except BaseException:
        process.kill()
        process.wait()
        raise
    return {
        "python_pid": process.pid,
        "exit_code": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "first_observed_python_sample": first,
        "maximum_observed_memory_kib": maxima,
        "kernel_max_rss_kib": usage.ru_maxrss,
        "cpu_user_seconds": usage.ru_utime,
        "cpu_system_seconds": usage.ru_stime,
    }


def junit_outcome(path: Path, nodeid: str) -> dict:
    """严格读取单一指定用例的 JUnit，拒绝缺失、错位、失败、错误或跳过。

    例如指定 test_file.py::test_ok 时，零项、两项或一项 skipped 均返回 valid=False。
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        return {"valid": False, "reason": "junit_unreadable", "details": str(error)}
    cases = list(root.iter("testcase"))
    suites = list(root.iter("testsuite"))
    address = nodeid.split("::")
    classname = address[0].removesuffix(".py").replace("/", ".")
    classname = ".".join([classname, *address[1:-1]])
    counts = {
        name: sum(case.find(tag) is not None for case in cases)
        for name, tag in (("failed", "failure"), ("errors", "error"), ("skipped", "skipped"))
    }
    counts["total"] = len(cases)
    counts["passed"] = len(cases) - counts["failed"] - counts["errors"] - counts["skipped"]
    identity_matches = len(cases) == 1 and (
        cases[0].get("classname") == classname and cases[0].get("name") == address[-1]
    )
    counters_match = len(suites) == 1 and all(
        suites[0].get(key) == value
        for key, value in {"tests": "1", "failures": "0", "errors": "0", "skipped": "0"}.items()
    )
    details = "\n".join(
        child.text or child.get("message", "")
        for case in cases
        for child in case
        if child.tag in ("failure", "error", "skipped")
    )
    return {
        "valid": identity_matches and counters_match and counts["passed"] == 1,
        "nodeid": nodeid,
        "counts": counts,
        "identity_matches": identity_matches,
        "suite_counters_match": counters_match,
        "details": details,
    }


def run_isolated_test(root: Path, nodeid: str, evidence_root: Path) -> dict:
    """串行启动完整指定用例，只有进程成功且唯一 JUnit 用例通过才返回。

    例如 tests/isolation_scenarios.py::test_success 在独立 Python 中执行原 fixture 与断言。
    """
    if os.environ.get(CHILD_NODE_VARIABLE):
        raise IsolationError("isolated_child_recursion_refused")
    evidence_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
    with (evidence_root / "process.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory = Path(tempfile.mkdtemp(prefix=digest + "-", dir=evidence_root))
        environment = os.environ.copy()
        environment[CHILD_NODE_VARIABLE] = nodeid
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[name] = "1"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(root.resolve()),
            nodeid,
            str(directory),
        ]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (directory / "child.log").open("w") as stream:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            result = wait_for_child(process, directory)
        outcome = junit_outcome(directory / "results.xml", nodeid)
        result.update(nodeid=nodeid, evidence_directory=str(directory), junit=outcome)
        (directory / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        if result["exit_code"] != 0 or not outcome["valid"]:
            raise IsolationError(
                f"isolated_test_failed: {nodeid}; exit={result['exit_code']}; "
                f"JUnit={json.dumps(outcome, ensure_ascii=False)}; evidence={directory}"
            )
        return result


def child_main(arguments: list[str]) -> int:
    """在新解释器中施加资源限制后，精确运行一个 nodeid 并输出独立 JUnit。

    例如传入仓库、test_file.py::test_ok、证据目录时只执行该用例，标记防止递归隔离。
    """
    if len(arguments) != 3:
        return 2
    root, nodeid, destination = arguments
    directory = Path(destination)
    apply_child_limits(directory, nodeid)
    os.environ[CHILD_NODE_VARIABLE] = nodeid
    os.chdir(root)
    import pytest

    return int(
        pytest.main(
            [
                nodeid,
                "--rootdir=" + root,
                "--basetemp=" + str(directory / "temporary"),
                "--junitxml=" + str(directory / "results.xml"),
                "-o",
                "cache_dir=" + str(directory / "cache"),
                "-o",
                "junit_family=xunit2",
                "-o",
                "junit_logging=all",
                "-o",
                "junit_log_passing_tests=true",
                "-v",
                "--tb=short",
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(child_main(sys.argv[1:]))
