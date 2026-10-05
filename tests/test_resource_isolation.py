"""用小型真实 Python 子进程验证压力隔离、失败传播、资源证据和覆盖合并。"""

import json
import os
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import conftest
import pytest
import resource_isolation as isolation

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "tests/isolation_scenarios.py::"


@pytest.mark.parametrize(
    ("soft", "hard", "expected"),
    [
        (-1, -1, (4 * 1024**3, 4 * 1024**3)),
        (1024**3, 2 * 1024**3, (1024**3, 2 * 1024**3)),
        (-1, 2 * 1024**3, (2 * 1024**3, 2 * 1024**3)),
        (6 * 1024**3, 8 * 1024**3, (4 * 1024**3, 4 * 1024**3)),
    ],
)
def test_inherited_address_limits_are_never_relaxed(soft, hard, expected):
    """场景：无限、较低和较高继承值；预期：均不提高继承限制。

    例：1 GiB/2 GiB 保持原值，无限/无限收紧至 4 GiB/4 GiB。
    """
    assert isolation.bounded_address_limits(soft, hard) == expected


def test_real_child_runs_one_original_offline_case_with_resource_evidence(tmp_path):
    """场景：真实隔离成功场景；预期：唯一 JUnit 通过并记录直接 Python PID。

    例：child-start、资源采样、wait4 和实际用例写出的 PID 必须一致。
    """
    result = isolation.run_isolated_test(ROOT, SCENARIO + "test_success", tmp_path)
    directory = Path(result["evidence_directory"])
    start = json.loads((directory / "child-start.json").read_text())
    samples = [
        json.loads(line) for line in (directory / "resources.jsonl").read_text().splitlines()
    ]
    executions = list((directory / "temporary").rglob("executed.json"))
    assert result["exit_code"] == 0
    assert result["junit"]["counts"] == {
        "total": 1,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert start["python_pid"] == result["python_pid"] != os.getpid()
    assert start["parent_pid"] == os.getpid()
    assert len(executions) == 1
    assert json.loads(executions[0].read_text())["python_pid"] == result["python_pid"]
    assert samples and all(sample["python_pid"] == result["python_pid"] for sample in samples)
    assert result["kernel_max_rss_kib"] > 0
    assert result["maximum_observed_memory_kib"]["VmHWM"] > 0
    assert result["elapsed_seconds"] > 0
    assert json.loads((directory / "result.json").read_text()) == result


@pytest.mark.parametrize(
    ("case", "count", "detail", "exit_code"),
    [
        ("test_failure", "failed", "isolation_assertion", 1),
        ("test_skip", "skipped", "isolation_skip", 0),
        ("test_setup_error", "errors", "isolation_setup_error", 1),
    ],
)
def test_real_child_failure_skip_and_setup_error_are_rejected(
    tmp_path,
    case,
    count,
    detail,
    exit_code,
):
    """场景：子进程失败、跳过或准备异常；预期：父进程拒绝且原说明保留。

    例：skip 的真实退出码为零，但 JUnit skipped=1 仍抛 IsolationError。
    """
    with pytest.raises(isolation.IsolationError, match=detail):
        isolation.run_isolated_test(ROOT, SCENARIO + case, tmp_path)
    report = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert report["exit_code"] == exit_code
    assert report["junit"]["counts"][count] == 1
    assert report["junit"]["valid"] is False
    assert detail in next(tmp_path.glob("*/child.log")).read_text()


def test_real_child_preserves_a_lower_inherited_limit(tmp_path):
    """场景：bootstrap 前继承 1 GiB 软上限；预期：子测试仍成功且不提高上限。

    例：直接 Python 先 setrlimit 再调用 child_main，不使用 preexec_fn。
    """
    script = (
        "import resource, sys; "
        "resource.setrlimit(resource.RLIMIT_AS, (1073741824, 2147483648)); "
        "sys.path.insert(0, sys.argv[1]); "
        "from resource_isolation import child_main; "
        "raise SystemExit(child_main(sys.argv[2:]))"
    )
    with (tmp_path / "child.log").open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(ROOT / "tests"),
                str(ROOT),
                SCENARIO + "test_success",
                str(tmp_path),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        result = isolation.wait_for_child(process, tmp_path)
    assert result["exit_code"] == 0, (tmp_path / "child.log").read_text()
    report = json.loads((tmp_path / "child-start.json").read_text())
    assert report["inherited_address_limits_bytes"] == [1024**3, 2 * 1024**3]
    assert report["address_limits_bytes"] == [1024**3, 2 * 1024**3]
    assert isolation.junit_outcome(tmp_path / "results.xml", SCENARIO + "test_success")["valid"]


def test_competing_requests_only_run_one_isolated_child_at_a_time(tmp_path, monkeypatch):
    """场景：两个线程同时申请同一隔离目录；预期：实际子进程顺序执行。

    例：事件顺序为 start/end/start/end，不能出现 start/start。
    """
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("TUSHARE_ISOLATION_TEST_EVENTS", str(events))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                isolation.run_isolated_test, ROOT, SCENARIO + "test_serial_execution", tmp_path
            )
            for _ in range(2)
        ]
        results = [future.result() for future in futures]
    observed = [json.loads(line) for line in events.read_text().splitlines()]
    assert [event["event"] for event in observed] == ["start", "end", "start", "end"]
    assert observed[0]["pid"] == observed[1]["pid"]
    assert observed[2]["pid"] == observed[3]["pid"] != observed[0]["pid"]
    assert {event["pid"] for event in observed} == {result["python_pid"] for result in results}


def test_child_only_source_is_included_in_combined_coverage(tmp_path, monkeypatch):
    """场景：覆盖探针只在隔离子进程执行；预期：合并覆盖包含其全部语句。

    例：answer() 仅在子进程返回 42，最终 missing_lines 为空。
    """
    probe = tmp_path / "probe"
    probe.mkdir()
    target = probe / "coverage_probe_target.py"
    target.write_text(
        'def answer():\n    """Return the example answer: answer() -> 42."""\n    return 42\n'
    )
    monkeypatch.setenv("TUSHARE_ISOLATION_COVERAGE_PROBE_DIR", str(probe))
    configuration = tmp_path / "coverage.ini"
    configuration.write_text(
        f"[run]\nsource = {probe}\ndata_file = {tmp_path / '.coverage'}\n"
        "branch = true\npatch = subprocess\n[report]\nfail_under = 100\n"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent("""\
        import json
        import sys
        from pathlib import Path
        import coverage
        sys.path.insert(0, sys.argv[1])
        from resource_isolation import run_isolated_test
        destination = Path(sys.argv[3])
        collector = coverage.Coverage(config_file=str(destination / "coverage.ini"),
            data_file=str(destination / ".coverage"))
        collector.start()
        result = run_isolated_test(Path(sys.argv[2]),
            "tests/isolation_scenarios.py::test_coverage_piece", destination / "children")
        collector.stop()
        collector.save()
        collector.combine(strict=True)
        collector.save()
        collector.json_report(outfile=str(destination / "coverage.json"))
        (destination / "execution.json").write_text(json.dumps(result))
        """)
    )
    with (tmp_path / "driver.log").open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, str(driver), str(ROOT / "tests"), str(ROOT), str(tmp_path)],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        result = isolation.wait_for_child(process, tmp_path)
    assert result["exit_code"] == 0, (tmp_path / "driver.log").read_text()
    report = json.loads((tmp_path / "coverage.json").read_text())
    measured = next(value for path, value in report["files"].items() if path.endswith(target.name))
    assert measured["missing_lines"] == []
    assert measured["executed_lines"] == [1, 3]
    assert report["totals"]["percent_covered"] == 100
    assert json.loads((tmp_path / "execution.json").read_text())["junit"]["valid"]


def test_recursive_child_launch_is_refused_before_creating_evidence(tmp_path, monkeypatch):
    """场景：已在隔离子进程中再次调用启动器；预期：拒绝且不创建新目录。

    例：TUSHARE_ISOLATED_TEST_NODEID 已设置时抛 recursion_refused。
    """
    monkeypatch.setenv(isolation.CHILD_NODE_VARIABLE, SCENARIO + "test_success")
    with pytest.raises(isolation.IsolationError, match="recursion_refused"):
        isolation.run_isolated_test(ROOT, SCENARIO + "test_success", tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_direct_pid_memory_parser_ignores_unrelated_fields_and_handles_exit(tmp_path):
    """场景：指定 PID 含内存/线程字段或已退出；预期：只读目标字段。

    例：VmHWM: 1024 kB 返回 1024，PID 消失返回 None。
    """
    status = tmp_path / "73" / "status"
    status.parent.mkdir()
    status.write_text("Name:\tpython\nVmRSS:\t512 kB\nVmHWM:\t1024 kB\nThreads:\t1\n")
    assert isolation.process_memory(73, tmp_path) == {"VmRSS": 512, "VmHWM": 1024, "Threads": 1}
    assert isolation.process_memory(74, tmp_path) is None


def test_missing_samples_still_preserve_wait4_kernel_peak(tmp_path, monkeypatch):
    """场景：采样时状态文件均不可读；预期：启动样本为 None，内核峰值仍真实。

    例：不能把全零采样当实际零内存，wait4 峰值应大于零。
    """
    monkeypatch.setattr(isolation, "process_memory", Mock(return_value=None))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.15)"])
    result = isolation.wait_for_child(process, tmp_path)
    assert result["exit_code"] == 0
    assert result["first_observed_python_sample"] is None
    assert result["kernel_max_rss_kib"] > 0
    assert not (tmp_path / "resources.jsonl").read_text()


def test_interrupted_observation_kills_and_reaps_the_actual_child(tmp_path, monkeypatch):
    """场景：父观察器中断；预期：直接 Python 子进程被杀死并回收。

    例：KeyboardInterrupt 原样传播，子进程退出为 SIGKILL 且无法再次 waitpid。
    """
    monkeypatch.setattr(isolation, "process_memory", Mock(side_effect=KeyboardInterrupt))
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with pytest.raises(KeyboardInterrupt):
        isolation.wait_for_child(process, tmp_path)
    assert process.returncode == -9
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)


@pytest.mark.parametrize("contents", [None, "<broken>"])
def test_junit_missing_and_malformed_files_are_rejected(tmp_path, contents):
    """场景：子报告缺失或 XML 损坏；预期：明确标记不可读。

    例：不存在 results.xml 返回 valid=False 和 junit_unreadable。
    """
    path = tmp_path / "results.xml"
    if contents is not None:
        path.write_text(contents)
    assert isolation.junit_outcome(path, "tests/check.py::test_one")["reason"] == "junit_unreadable"


@pytest.mark.parametrize(
    ("cases", "counters", "expected"),
    [
        (
            '<testcase classname="tests.check.Group" name="test_one">'
            '<system-out>ok</system-out></testcase>',
            'tests="1" failures="0" errors="0" skipped="0"',
            True,
        ),
        (
            '<testcase classname="tests.check.Group" name="test_other"/>',
            'tests="1" failures="0" errors="0" skipped="0"',
            False,
        ),
        (
            '<testcase classname="tests.other.Group" name="test_one"/>',
            'tests="1" failures="0" errors="0" skipped="0"',
            False,
        ),
        (
            '<testcase classname="tests.check.Group" name="test_one"/>',
            'tests="2" failures="0" errors="0" skipped="0"',
            False,
        ),
        (
            '<testcase classname="tests.check.Group" name="test_one">'
            '<failure message="failure"/></testcase>',
            'tests="1" failures="0" errors="0" skipped="0"',
            False,
        ),
        (
            '<testcase classname="tests.check.Group" name="test_one">'
            '<error>error</error></testcase>',
            'tests="1" failures="0" errors="1" skipped="0"',
            False,
        ),
        (
            '<testcase classname="tests.check.Group" name="test_one">'
            '<skipped>skip</skipped></testcase>',
            'tests="1" failures="0" errors="0" skipped="1"',
            False,
        ),
        ("", 'tests="0" failures="0" errors="0" skipped="0"', False),
        ("<testcase/><testcase/>", 'tests="2" failures="0" errors="0" skipped="0"', False),
    ],
)
def test_junit_requires_the_exact_single_passing_case(tmp_path, cases, counters, expected):
    """场景：报告身份、数量、状态或套件计数不一致；预期：一律拒绝。

    例：唯一 Group::test_one 全通过返回 True；同名异文件或两项返回 False。
    """
    path = tmp_path / "results.xml"
    path.write_text(f"<testsuites><testsuite {counters}>{cases}</testsuite></testsuites>")
    result = isolation.junit_outcome(path, "tests/check.py::Group::test_one")
    assert result["valid"] is expected


@pytest.mark.parametrize(
    "xml", ["<testsuites/>", "<testsuites><testsuite/><testsuite/></testsuites>"]
)
def test_junit_requires_one_suite(tmp_path, xml):
    """场景：套件缺失或多于一个；预期：suite_counters_match 为 False。

    例：两个空套件不能被视为已执行唯一用例。
    """
    path = tmp_path / "results.xml"
    path.write_text(xml)
    assert (
        isolation.junit_outcome(path, "tests/check.py::test_one")["suite_counters_match"] is False
    )


def test_bootstrap_rejects_incomplete_arguments():
    """场景：bootstrap 缺少仓库、nodeid 或目录；预期：退出 2 且不启动 pytest。

    例：空参数列表返回 2。
    """
    assert isolation.child_main([]) == 2


def test_hook_routes_plain_matching_and_mismatched_nodes(monkeypatch):
    """场景：普通节点、子进程匹配节点与错位节点；预期：前两者直接执行，错位拒绝。

    例：子标记为 test_other 而实际为 test_one 时抛 pytest Failed。
    """
    item = SimpleNamespace(nodeid="test_one", get_closest_marker=Mock(return_value=None))
    assert conftest.pytest_pyfunc_call(item) is None
    item.get_closest_marker.return_value = object()
    monkeypatch.setenv(isolation.CHILD_NODE_VARIABLE, "test_one")
    assert conftest.pytest_pyfunc_call(item) is None
    monkeypatch.setenv(isolation.CHILD_NODE_VARIABLE, "test_other")
    with pytest.raises(pytest.fail.Exception, match="nodeid_mismatch"):
        conftest.pytest_pyfunc_call(item)


@pytest.mark.parametrize("explicit_directory", [True, False])
def test_hook_uses_selected_evidence_directory_and_propagates_failures(
    tmp_path,
    monkeypatch,
    explicit_directory,
):
    """场景：显式证据目录或 pytest cache；预期：传入准确 nodeid，失败不能吞掉。

    例：启动器抛 IsolationError 时父 hook 转为 pytest Failed。
    """
    monkeypatch.delenv(isolation.CHILD_NODE_VARIABLE, raising=False)
    if explicit_directory:
        monkeypatch.setenv(isolation.EVIDENCE_VARIABLE, str(tmp_path))
    else:
        monkeypatch.delenv(isolation.EVIDENCE_VARIABLE, raising=False)
    runner = Mock(return_value={})
    monkeypatch.setattr(conftest, "run_isolated_test", runner)
    config = SimpleNamespace(
        rootpath=ROOT, cache=SimpleNamespace(mkdir=Mock(return_value=tmp_path))
    )
    item = SimpleNamespace(
        nodeid="test_one", get_closest_marker=Mock(return_value=object()), config=config
    )
    assert conftest.pytest_pyfunc_call(item) is True
    runner.assert_called_once_with(ROOT, "test_one", tmp_path)
    runner.side_effect = isolation.IsolationError("preserved_failure")
    with pytest.raises(pytest.fail.Exception, match="preserved_failure"):
        conftest.pytest_pyfunc_call(item)
