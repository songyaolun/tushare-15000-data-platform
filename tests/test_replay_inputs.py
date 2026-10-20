"""Portable replay data must preserve the old context and reject altered dependencies."""

import json
import shutil
from dataclasses import asdict

import numpy as np
import pytest
from test_discovery_candidate_allocation import allocation_fixture

from tushare_entitlement_hub.strategy_discovery.replay_inputs import (
    export_replay_inputs,
    file_digest,
    load_replay_context,
    sector_capped_mask,
    verified_replay_manifest,
)


def cache_fixture(root):
    """Write eight synthetic sessions to the legacy layout; return panel and score matrix."""
    panel, definition = allocation_fixture()
    for name in ('index', 'total_index', 'amount'):
        panel.market[name] = np.arange(100, 108, dtype=float)
    for name in ('breadth', 'advancing', 'limit_down'):
        panel.market[name] = np.full(8, .5)
    panel_root = root / 'p03-regime-alpha-20261018/panel'
    panel_root.mkdir(parents=True)
    for group in ('values', 'signals', 'market'):
        for name, values in getattr(panel, group).items():
            np.save(panel_root / f'{group}-{name}.npy', values)
    (panel_root / 'panel.json').write_text(json.dumps({
        'calendar': panel.calendar, 'codes': panel.codes,
        **{group: list(getattr(panel, group)) for group in ('values', 'signals', 'market')},
    }))
    for relative, values in [
        ('stockops-pit-redesign-20261018/eligible.npy', definition.entry_mask),
        ('stockops-pit-redesign-20261018/scores.npy', definition.scores),
        ('stockops-pit-validation-20261018/matrices/technical_risk.npy',
         np.zeros(definition.scores.shape)),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values)
    return panel, definition.scores


def test_export_relocation_and_legacy_context_without_scripts(tmp_path):
    """Relocation preserves masks/policy/stages/factors and never needs cached Python files."""
    source, exported, relocated = (tmp_path / name for name in ('source', 'export', 'relocated'))
    panel, _ = cache_fixture(source)
    original = load_replay_context(source)
    export_replay_inputs(source, exported)
    shutil.move(exported, relocated)
    shutil.rmtree(source)
    actual = load_replay_context(relocated)
    assert actual[0].calendar == panel.calendar
    assert actual[0].codes == panel.codes
    assert asdict(actual[2]) == asdict(original[2])
    assert actual[4] == original[4]
    for index in (1, 5):
        np.testing.assert_array_equal(actual[index], original[index])
    for name in ('entry_mask', 'scores'):
        np.testing.assert_array_equal(getattr(actual[3], name), getattr(original[3], name))
    for name in original[6]:
        np.testing.assert_array_equal(actual[6][name], original[6][name])
    with pytest.raises(ValueError, match='fresh_replay_output'):
        export_replay_inputs(source, relocated)


@pytest.mark.parametrize('damage', ['schema', 'missing', 'escape', 'absolute', 'hash', 'unhashed'])
def test_manifest_rejects_incomplete_or_changed_data(tmp_path, damage):
    """Wrong schema, unpinned arrays, altered bytes and paths outside the bundle all fail."""
    source, output = tmp_path / 'source', tmp_path / 'output'
    cache_fixture(source)
    export_replay_inputs(source, output)
    path = output / 'replay.json'
    manifest = json.loads(path.read_text())
    if damage == 'schema':
        manifest['schema'] = 'unknown'
    elif damage == 'missing':
        del manifest['files']['eligible.npy']
    elif damage in ('escape', 'absolute'):
        manifest['files']['../outside' if damage == 'escape' else '/outside'] = 'bad'
    elif damage == 'hash':
        (output / 'eligible.npy').write_bytes(b'changed')
    else:
        del manifest['files']['panel/values-raw_close.npy']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verified_replay_manifest(output)


def test_invalid_legacy_axis_rejected(tmp_path):
    """A malformed candidate mask cannot be silently broadcast across a panel."""
    cache_fixture(tmp_path)
    np.save(tmp_path / 'stockops-pit-redesign-20261018/eligible.npy', np.ones(8))
    with pytest.raises(ValueError, match='invalid_replay_axes'):
        load_replay_context(tmp_path)


def test_sector_cap_stable_ties_unknown_group_and_missing_scores():
    """Ties keep column order; missing scores skip; unknown sectors share one slot."""
    scores = np.array([[9., 9., 8., 7., 6., np.nan]])
    sectors = np.array([[1., 1., np.nan, -1., 2., 3.]])
    eligible = np.ones(scores.shape, dtype=bool)
    assert sector_capped_mask(eligible, scores, sectors, 1, count=3).tolist() == [
        [True, False, True, False, True, False]]
    assert sector_capped_mask(eligible, scores, sectors, 1, count=10).sum() == 3
    with pytest.raises(ValueError, match='invalid_sector_cap'):
        sector_capped_mask(eligible, scores, sectors, 1, count=0)


def test_file_digest_empty_and_streamed(tmp_path):
    """Hashes match standard empty and multi-block SHA256, independent of chunking."""
    import hashlib
    path = tmp_path / 'file'
    for payload in (b'', b'x' * (1024 * 1024 + 1)):
        path.write_bytes(payload)
        assert file_digest(path) == hashlib.sha256(payload).hexdigest()


def test_explicit_export_rejects_existing_output_and_wrong_roles(tmp_path):
    """Explicit exports reject stale destinations and incomplete file-role mappings."""
    from tushare_entitlement_hub.strategy_discovery.replay_inputs import write_replay_inputs

    cache_fixture(tmp_path / "cache")
    panel = tmp_path / "cache/p03-regime-alpha-20261018/panel"
    with pytest.raises(ValueError, match="fresh_replay_output"):
        write_replay_inputs(panel, {}, [], tmp_path)
    with pytest.raises(ValueError, match="invalid_replay_file_roles"):
        write_replay_inputs(panel, {}, [], tmp_path / "out")
    assert not (tmp_path / "out").exists()
