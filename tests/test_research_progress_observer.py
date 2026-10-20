"""Exercise a read-only observer against synthetic processes and partial artifacts."""

import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def observer():
    """Load the standalone observer without starting it; return its callable namespace."""
    return runpy.run_path(str(Path(__file__).parents[1] /
                             'scripts/research/watch_return80_progress.py'))


def test_process_state_reports_cpu_and_exit_without_percentage(tmp_path):
    """Synthetic proc stat yields exact CPU seconds; an absent PID is reported exited."""
    directory = tmp_path / '12'
    directory.mkdir()
    fields = ['R'] + ['0'] * 20
    fields[11], fields[12] = '200', '100'
    (directory / 'stat').write_text('12 (worker name) ' + ' '.join(fields))
    query = observer()['process_status']
    assert query(12, proc_root=tmp_path) == {
        'pid': 12, 'state': 'R', 'cpu_seconds': 300 / os.sysconf('SC_CLK_TCK'),
    }
    assert query(13, proc_root=tmp_path) == {'pid': 13, 'state': 'exited'}


def test_saved_counts_include_only_finished_artifacts(tmp_path):
    """Partial costs and failed modules remain visible, with no inferred model progress."""
    group = tmp_path / 'accounts' / 'half126'
    group.mkdir(parents=True)
    (group / 'results.json').write_text('[{"cost": 1}]')
    (tmp_path / 'audit-results-equal.json').write_text('[{"exit_code":0},{"exit_code":1}]')
    (tmp_path / 'period-sql-old-command.json').write_text('{"command":"archived"}')
    (tmp_path / 'period-sql-equal.json').write_text('[{"passed":true},{"passed":false}]')
    (tmp_path / 'multicore-example-result.json').write_text('{"exit_code":1}')
    state = observer()['read_state'](tmp_path, [])
    assert (state['accounts_completed'], state['audit_modules_completed'],
            state['sql_windows_completed']) == (1, 1, 1)
    assert state['failures'] == ['multicore-example-result.json']
    assert state['last_sampled_phase'] is None and state['complete'] is False
    (tmp_path / 'phase-sample.json').write_text('{"phase":"native_prediction_after_fit"}')
    (tmp_path / 'final-integrity.json').write_text('{"passed":true}')
    assert observer()['read_state'](tmp_path, [])['complete'] is True


def test_source_lines_forward_once_and_wait_for_newline(tmp_path, capsys):
    """Malformed lines are skipped and incomplete JSON is forwarded only after newline."""
    directory = tmp_path / 'multicore-logs'
    directory.mkdir()
    path = directory / 'train.log'
    path.write_text('not json\n{"event":"fit_done"}\n{"event":"native_predict_start"}')
    offsets = {}
    forward = observer()['forward_new_events']
    forward(tmp_path, offsets)
    assert json.loads(capsys.readouterr().out)['source_event']['event'] == 'fit_done'
    with path.open('a') as stream:
        stream.write('\n')
    forward(tmp_path, offsets)
    assert json.loads(capsys.readouterr().out)['source_event']['event'] == 'native_predict_start'
    forward(tmp_path, offsets)
    assert capsys.readouterr().out == ''


def test_observer_retries_in_progress_json_then_exits(monkeypatch, tmp_path, capsys):
    """A concurrent artifact write triggers one retry, then validated completion exits."""
    main = observer()['main']
    monkeypatch.setattr('sys.argv', ['watch', '--root', str(tmp_path)])
    monkeypatch.setitem(main.__globals__, 'forward_new_events', Mock())
    monkeypatch.setitem(main.__globals__, 'read_state', Mock(side_effect=[
        json.JSONDecodeError('incomplete', '{', 1), {'complete': True, 'failures': []},
    ]))
    waiter = Mock()
    monkeypatch.setitem(main.__globals__, 'threading',
                        SimpleNamespace(Event=lambda: SimpleNamespace(wait=waiter)))
    main()
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row['event'] for row in records] == ['observer_snapshot_retry', 'research_progress']
    waiter.assert_called_once_with(5)
