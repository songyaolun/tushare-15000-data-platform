"""Flush truthful phase events and heartbeats without touching research calculations."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

_OUTPUT_LOCK = threading.Lock()


def progress_event(event: str, *, started: float, **details: Any) -> None:
    """Print one atomic JSON event with elapsed seconds and immediate flushing.

    Input: event='account_done', started=time.monotonic(), completed=2, total=8.
    Output: one stdout line containing those fields plus a nonnegative elapsed value.
    Callers supply public metadata only; model values and credentials are unnecessary.
    """
    record = {**details, "event": event, "elapsed_seconds": max(0, time.monotonic() - started)}
    with _OUTPUT_LOCK:
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def _heartbeat(
    stopped: threading.Event, phase: str, started: float,
    interval: float, context: Mapping[str, Any],
) -> None:
    """Emit liveness until stopped, never claiming work percentage.

    Example: an unset event with interval=30 emits phase_heartbeat every 30 seconds;
    setting the event ends the worker without emitting a completion event.
    """
    while not stopped.wait(interval):
        progress_event(f"{phase}_heartbeat", started=started, **context)


@contextmanager
def progress_phase(
    phase: str, *, enabled: bool = True, heartbeat_seconds: float = 30,
    **context: Any,
) -> Iterator[None]:
    """Report start/done or failure and periodic liveness around unchanged work.

    Example: with progress_phase('native_predict', case='half126'): prediction = f().
    Output: flushed native_predict_start/done and elapsed seconds; exceptions retain
    their original identity and emit only their type, avoiding data in error messages.
    enabled=False emits nothing and starts no thread. Invalid intervals fail before work.
    """
    if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds_must_be_positive_finite")
    if not enabled:
        yield
        return
    started = time.monotonic()
    stopped = threading.Event()
    worker = threading.Thread(
        target=_heartbeat, args=(stopped, phase, started, heartbeat_seconds, context), daemon=True,
    )
    progress_event(f"{phase}_start", started=started, **context)
    worker.start()
    try:
        yield
    except BaseException as error:
        stopped.set()
        worker.join()
        progress_event(
            f"{phase}_failed", started=started, error_type=type(error).__name__, **context,
        )
        raise
    else:
        stopped.set()
        worker.join()
        progress_event(f"{phase}_done", started=started, **context)
