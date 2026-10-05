"""供隔离回归显式选择的真实场景；故意失败和跳过场景不参与默认全仓收集。"""

import importlib
import json
import os
import resource
import socket
import time
from pathlib import Path

import pytest
from resource_isolation import CHILD_NODE_VARIABLE, MAXIMUM_VIRTUAL_BYTES


@pytest.mark.resource_isolated
def test_success(policy, tmp_path, entry, store):
    """场景：独立进程运行原离线 fixture；预期：单次成功且资源受限。

    例：index_daily 政策可用，真实 socket 连接被拒绝，虚拟内存不超过 4 GiB。
    """
    assert policy.api_name == "index_daily"
    assert entry.api_name == policy.api_name
    assert store is not None
    assert os.environ[CHILD_NODE_VARIABLE].endswith("::test_success")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    assert 0 < soft <= hard <= MAXIMUM_VIRTUAL_BYTES
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[name] == "1"
    with pytest.raises(AssertionError, match="离线测试禁止真实网络"):
        socket.create_connection(("127.0.0.1", 9))
    (tmp_path / "executed.json").write_text(json.dumps({"python_pid": os.getpid()}))


@pytest.mark.resource_isolated
def test_failure():
    """场景：子用例断言失败；预期：父运行收到完整失败说明，例如 isolation_assertion。

    例：此场景只通过精确 nodeid 执行，失败不能被当成父用例成功。
    """
    pytest.fail("isolation_assertion")


@pytest.mark.resource_isolated
def test_skip():
    """场景：子用例主动跳过；预期：即使子 pytest 退出零也被拒绝。

    例：isolation_skip 在 JUnit 中为 skipped，父结果必须失败。
    """
    pytest.skip("isolation_skip")


@pytest.fixture
def broken_setup():
    """场景：用例准备阶段异常；预期：JUnit 记录 error，例如 isolation_setup_error。"""
    raise RuntimeError("isolation_setup_error")


@pytest.mark.resource_isolated
def test_setup_error(broken_setup):
    """场景：依赖失败 fixture；预期：测试体不执行且父运行拒绝 error。

    例：broken_setup 抛异常，断言行不会被运行。
    """
    assert broken_setup is None


@pytest.mark.resource_isolated
def test_serial_execution():
    """场景：竞争同一证据目录的两个请求；预期：子进程活动时间不重叠。

    例：每个子进程写起止事件，中间等待 0.15 秒，四个事件应依次开始/结束。
    """
    events = Path(os.environ["TUSHARE_ISOLATION_TEST_EVENTS"])
    with events.open("a") as stream:
        stream.write(json.dumps({"event": "start", "pid": os.getpid()}) + "\n")
        stream.flush()
        time.sleep(0.15)
        stream.write(json.dumps({"event": "end", "pid": os.getpid()}) + "\n")


@pytest.mark.resource_isolated
def test_coverage_piece(monkeypatch):
    """场景：只在子进程加载覆盖探针；预期：返回 42 且合并后包含探针执行行。

    例：父进程不导入 coverage_probe_target，子进程执行其 answer() 返回 42。
    """
    monkeypatch.syspath_prepend(os.environ["TUSHARE_ISOLATION_COVERAGE_PROBE_DIR"])
    target = importlib.import_module("coverage_probe_target")
    assert target.answer() == 42
