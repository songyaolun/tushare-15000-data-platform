"""Observe existing research processes and artifacts; never launch or stop work."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from tushare_entitlement_hub.progress import progress_event


def process_status(process_id: int, *, proc_root: Path = Path('/proc')) -> dict[str, Any]:
    """Read process state and CPU seconds without memory/locals access.

    Example: a running PID returns state='R' and cumulative cpu_seconds; an exited
    PID returns state='exited'. CPU time is not a training completion percentage.
    """
    try:
        text = (proc_root / str(process_id) / 'stat').read_text()
    except FileNotFoundError:
        return {'pid': process_id, 'state': 'exited'}
    fields = text[text.rindex(')') + 2:].split()
    ticks = os.sysconf('SC_CLK_TCK')
    return {'pid': process_id, 'state': fields[0],
            'cpu_seconds': (int(fields[11]) + int(fields[12])) / ticks}


def read_state(root: Path, process_ids: list[int]) -> dict[str, Any]:
    """Count only saved successful accounts/audits/windows, retaining partial failures.

    Example: two saved cost accounts and one failed module give accounts_completed=2
    and a failure entry; absent files contribute zero rather than estimated progress.
    """
    accounts = sum(len(json.loads(path.read_text()))
                   for path in (root / 'accounts').glob('*/results.json'))
    arms = ('equal', 'half126', 'half252', 'half504')
    audits = [row for arm in arms if (path := root / f'audit-results-{arm}.json').exists()
              for row in json.loads(path.read_text())]
    windows = [row for arm in arms if (path := root / f'period-sql-{arm}.json').exists()
               for row in json.loads(path.read_text())]
    failures = [path.name for path in root.glob('multicore-*-result.json')
                if json.loads(path.read_text()).get('exit_code') != 0]
    models = [path.parent.name for path in (root / 'models').glob('*/audit.json')]
    evidence = root / 'phase-sample.json'
    sampled = json.loads(evidence.read_text()) if evidence.exists() else {}
    return {'accounts_completed': accounts, 'accounts_total': 32,
            'audit_modules_completed': sum(row['exit_code'] == 0 for row in audits),
            'audit_modules_total': 24,
            'sql_windows_completed': sum(row['passed'] for row in windows),
            'sql_windows_total': 192, 'new_models_completed': models,
            'failures': failures, 'processes': [process_status(pid) for pid in process_ids],
            'last_sampled_phase': sampled.get('phase'),
            'phase_evidence': 'phase-sample.json; historical sample, not current percentage',
            'complete': ((root / 'final-integrity.json').is_file()
                         and json.loads((root / 'final-integrity.json').read_text())['passed'])}


def forward_new_events(root: Path, offsets: dict[Path, int]) -> None:
    """Forward newly flushed source log lines once, labeling their file.

    Example: a new fit_done JSON line is printed with source_log; already-read bytes
    are skipped and incomplete final lines remain pending until their newline arrives.
    """
    paths = list((root / 'multicore-logs').glob('*.log'))
    paths += [path for path in root.glob('postprocess-*.log')
              if path.name != 'postprocess-runner.log']
    for path in sorted(paths):
        with path.open() as stream:
            stream.seek(offsets.get(path, 0))
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line.endswith('\n'):
                    offsets[path] = position
                    break
                offsets[path] = stream.tell()
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                print(json.dumps({'source_log': path.name, 'source_event': record}), flush=True)


def main() -> None:
    """Emit existing task progress every five seconds until integrity is available.

    Example: --root /private/recent-weighting --pid 123 --once prints one snapshot.
    This observer does not mutate inputs, model files, account files or running tasks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pid', type=int, action='append', default=[])
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    offsets: dict[Path, int] = {}
    while True:
        forward_new_events(args.root, offsets)
        try:
            state = read_state(args.root, args.pid)
        except json.JSONDecodeError:
            progress_event('observer_snapshot_retry', started=started,
                           reason='artifact_write_in_progress')
        else:
            progress_event('research_progress', started=started, **state)
            if args.once or state['complete'] or state['failures']:
                return
        threading.Event().wait(5)


if __name__ == '__main__':
    main()
