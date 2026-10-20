"""Verify truthful phases, immediate output, liveness and unchanged exceptions."""

import json
import threading
import time

import pytest

from tushare_entitlement_hub.progress import progress_event, progress_phase


def test_phase_reports_ordered_start_done_and_elapsed(capsys):
    """A successful phase emits exactly start/done with public context and elapsed time."""
    with progress_phase("native_predict", case="half126"):
        assert json.loads(capsys.readouterr().out)["event"] == "native_predict_start"
    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "native_predict_done" and record["case"] == "half126"
    assert record["elapsed_seconds"] >= 0


def test_phase_preserves_failure_and_omits_exception_payload(capsys):
    """A failed phase rethrows the same exception and reports its type, never its payload."""
    error = RuntimeError("private input value")
    with pytest.raises(RuntimeError) as caught, progress_phase("account_audit"):
        raise error
    output = capsys.readouterr().out
    records = [json.loads(line) for line in output.splitlines()]
    assert caught.value is error and "private input value" not in output
    assert records[-1]["event"] == "account_audit_failed"
    assert records[-1]["error_type"] == "RuntimeError"


def test_disabled_progress_runs_work_without_output(capsys):
    """Quiet phases execute the unchanged body and produce no phase events."""
    with progress_phase("fit", enabled=False):
        result = 3 + 4
    assert result == 7 and capsys.readouterr().out == ""


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_heartbeat_interval_fails_before_work(interval):
    """Nonpositive or nonfinite heartbeat intervals reject before entering the body."""
    with pytest.raises(ValueError, match="positive_finite"), progress_phase(
        "fit", heartbeat_seconds=interval,
    ):
        pytest.fail("invalid interval entered work")


def test_heartbeat_proves_liveness_without_percent(monkeypatch):
    """A real worker emits heartbeat while work waits and stops before phase_done."""
    received = threading.Event()
    events = []

    def capture(event, **details):
        """Collect public event metadata; a heartbeat releases the bounded test wait."""
        events.append((event, details))
        if event == "fit_heartbeat":
            received.set()

    monkeypatch.setattr("tushare_entitlement_hub.progress.progress_event", capture)
    with progress_phase("fit", heartbeat_seconds=0.001):
        assert received.wait(1), "heartbeat missing"
    assert events[0][0] == "fit_start" and events[-1][0] == "fit_done"
    assert not any("percent" in details for _, details in events)


def test_event_flushes_and_retains_truthful_counts(monkeypatch):
    """An emitted account completion preserves counts and explicitly flushes stdout."""
    calls = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: calls.append((args, kwargs)))
    progress_event("account_done", started=time.monotonic(), completed=2, total=8)
    assert calls[0][1]["flush"] is True
    record = json.loads(calls[0][0][0])
    assert (record["completed"], record["total"]) == (2, 8)
