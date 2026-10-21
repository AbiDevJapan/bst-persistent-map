"""Small correctness / controls tests for the minimal scaling harness."""
import json
import numpy as np
import pytest

from run_correction_scaling import run_scaling


def make_prepared(root, n=10, corrected=(2, 3)):
    root.mkdir(parents=True)
    (root / 'scans').mkdir()
    initial = []
    corrections = {}
    for i in range(n):
        pose = np.eye(4)
        pose[0, 3] = 0.5 * i
        target = pose.copy()
        if i in corrected:
            pose[0, 3] += 1.0
            corrections[f'scan_{i:03d}'] = target.tolist()
        # Fixed points partly overlap across different sources.
        points = np.asarray([[0, 0, 0], [0.5, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=float)
        np.savez_compressed(root / 'scans' / f'{i}.npz', points=points)
        initial.append({'source_id': f'scan_{i:03d}', 'robot_id': 'test_robot',
                        'observed_at': 100000 + i, 'npz': f'scans/{i}.npz',
                        'pose': pose.tolist()})
    (root / 'manifest_initial.json').write_text(json.dumps(initial))
    (root / 'corrections.json').write_text(json.dumps(corrections))
    return root


def test_minimal_scaling_exact_preserves_history_and_fixed_corrections(tmp_path):
    prepared = make_prepared(tmp_path / 'prepared')
    out = run_scaling(prepared, tmp_path / 'out', sizes=(4, 7, 10), repeats=1, batch_size=2)
    assert out['all_exact'] and out['all_previous_revisions_preserved']
    assert out['correction_ids_constant']
    assert [r['corrected_sources'] for r in out['summary']] == [2, 2, 2]
    assert all(r['affected_tiles'] > 0 for r in out['summary'])
    assert all(r['median_incremental_commit_ms'] > 0 for r in out['summary'])
    assert all(r['full_union_rebuild_ms_single'] > 0 for r in out['summary'])
    assert len((tmp_path / 'out' / 'trials.csv').read_text().splitlines()) == 4


def test_scaling_rejects_moving_correction_set(tmp_path):
    prepared = make_prepared(tmp_path / 'prepared')
    with pytest.raises(ValueError, match='smallest prefix excludes corrected scans'):
        run_scaling(prepared, tmp_path / 'bad', sizes=(3, 7), repeats=1)


def test_scaling_rejects_insufficient_prepared_scans(tmp_path):
    prepared = make_prepared(tmp_path / 'prepared')
    with pytest.raises(ValueError, match='Prepare more scans'):
        run_scaling(prepared, tmp_path / 'bad', sizes=(4, 20), repeats=1)
