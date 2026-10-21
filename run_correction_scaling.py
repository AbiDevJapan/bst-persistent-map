#!/usr/bin/env python3
"""Step 11.2: minimal, correctness-gated pose-correction locality check.

Usage:
  python run_correction_scaling.py --prepared results/step11_2_prepared \
      --output results/step11_2_smoke --sizes 64 128 256 --repeats 1

This is a diagnostic (NOT publication-grade). No extra algorithms, full-disk
rebuilds, or randomized benchmark matrices are introduced.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import hashlib
import itertools
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from versioned_map.correction import CorrectionMap
from versioned_map.correction.nclt_ingest import _get_records
from versioned_map.correction.voxel import Pose


def _digest(voxels: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(voxels, dtype='<i8', order='C').tobytes()).hexdigest()


def _full_bounds(*arrays):
    nonempty = [a for a in arrays if len(a)]
    if not nonempty:
        return (0, 1, 0, 1, 0, 1)
    low = np.minimum.reduce([a.min(axis=0) for a in nonempty])
    high = np.maximum.reduce([a.max(axis=0) for a in nonempty]) + 1
    return tuple(int(v) for v in (low[0], high[0], low[1], high[1], low[2], high[2]))


def _checkpoint(db: CorrectionMap) -> None:
    # Prevent WAL bytes from being mistaken for logical map compression.
    db.con.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()


def _get_sorted_sizes(sizes, total):
    if not sizes:
        raise ValueError('provide at least one --sizes value')
    selected = [int(i) for i in sizes]
    if any(i <= 0 or i > total for i in selected):
        raise ValueError(f'--sizes must contain positive prefixes <= {total} prepared sources. '
                         'Prepare more scans with run_nclt_correction.py prepare if needed.')
    if len(selected) != len(set(selected)) or selected != sorted(selected):
        raise ValueError('--sizes must be strictly increasing without duplicates')
    return selected


def run_scaling(prepared: str | Path, output: str | Path,
                *, sizes=(64, 128, 256), repeats=1, batch_size=16,
                voxel_size=0.5, origin=(0.0, 0.0, 0.0)) -> dict:
    """Test fixed-source pose corrections as an expanding map prefix is ingested.

    Corrections, data points, poses and correction timing boundary stay fixed.
    Ingestion and SQLite backup are *excluded* from correction time. One
    independent full voxel-union rebuild per prefix is measured for context.
    """
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if repeats < 1 or batch_size < 1 or voxel_size <= 0:
        raise ValueError('repeats, batch_size, voxel_size must be positive')
    initial_path = prepared / 'manifest_initial.json'
    corrections_path = prepared / 'corrections.json'
    manifest = json.loads(initial_path.read_text())
    changes = json.loads(corrections_path.read_text())
    if not isinstance(manifest, list) or not manifest or not isinstance(changes, dict) or not changes:
        raise ValueError('prepared manifests must contain sources and controlled pose corrections')
    source_ids = [item['source_id'] for item in manifest]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('duplicate source_id in manifest')
    missing = set(changes) - set(source_ids)
    if missing:
        raise ValueError(f'correction IDs absent from manifest: {sorted(missing)[:3]}')
    ordered_sizes = _get_sorted_sizes(sizes, len(manifest))
    # Enforce the research control: ALL sizes must contain the SAME corrections.
    earliest = set(source_ids[:ordered_sizes[0]])
    absent = set(changes) - earliest
    if absent:
        raise ValueError('smallest prefix excludes corrected scans; increase its size to include '
                         f'all {len(changes)} corrections (missing {len(absent)})')
    overrides = {sid: Pose(pose) for sid, pose in changes.items()}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output directory is nonempty: {output}. Use a new directory.')
    output.mkdir(parents=True, exist_ok=True)

    results = []
    with tempfile.TemporaryDirectory(prefix='step112_scale_') as temp:
        base_path = Path(temp) / 'base.sqlite'
        # Build larger prefixes by appending only the newly included sources.
        # This is NOT an incremental correction under measurement; it merely
        # saves repeated NCLT input parsing, and is outside timed correction.
        with CorrectionMap(base_path, voxel_size=voxel_size, origin=tuple(origin)) as base:
            previous = 0
            for size in ordered_sizes:
                selected = itertools.islice(_get_records(initial_path), previous, size)
                batch = []
                for rec in selected:
                    batch.append(rec)
                    if len(batch) >= batch_size:
                        base.add_sources(batch, note='Step 11.2 scaling-only ingestion')
                        batch = []
                if batch:
                    base.add_sources(batch, note='Step 11.2 scaling-only ingestion')
                previous = size
                before_revision = base.revision
                assert base.stats()['sources'] == size
                _checkpoint(base)

                # Corrected all-source rebuild is used as an independent oracle.
                start = time.perf_counter()
                expected = base.full_rebuild(pose_overrides=overrides)
                full_rebuild_ms = (time.perf_counter() - start) * 1000.0
                before = base.full_rebuild()
                bounds = _full_bounds(before, expected)
                expected_sha = _digest(expected)
                before_sha = _digest(before)
                active_tiles = int(base.stats()['current_nonempty_tiles'])

                for repeat in range(repeats):
                    trial_path = Path(temp) / f'trial_{size}_{repeat}.sqlite'
                    # Clone outside timing to produce the same committed input DB.
                    with closing(sqlite3.connect(str(base_path))) as src, closing(sqlite3.connect(str(trial_path))) as dest:
                        src.backup(dest)
                    with CorrectionMap(trial_path) as db:
                        if db.revision != before_revision:
                            raise AssertionError('SQLite backup revision changed')
                        start = time.perf_counter()
                        update = db.correct_poses(overrides, note='Step 11.2 locality check')
                        correction_ms = (time.perf_counter() - start) * 1000.0
                        result = db.query(bounds)
                        old = db.query(bounds, revision=before_revision)
                        exact = np.array_equal(result, expected)
                        preserved = np.array_equal(old, before)
                        if not exact or not preserved:
                            raise AssertionError(f'Incorrect map at size={size} repeat={repeat}: '
                                                 f'corrected={exact}, history={preserved}')
                        if update.sources_changed != len(changes):
                            raise AssertionError('correction count differs across sizes')
                        results.append({
                            'sources': size,
                            'repeat': repeat,
                            'corrected_sources': len(changes),
                            'active_tiles_before': active_tiles,
                            'affected_tiles': update.tiles_touched,
                            'modified_tiles': update.tiles_modified,
                            'affected_fraction': round(update.tiles_touched / active_tiles, 6) if active_tiles else 0,
                            'incremental_commit_ms': round(correction_ms, 4),
                            'full_rebuild_ms': round(full_rebuild_ms, 4),
                            'voxels_before': len(before),
                            'voxels_corrected': len(result),
                            'before_sha256': before_sha,
                            'corrected_sha256': expected_sha,
                            'exact': exact,
                            'previous_revision_preserved': preserved,
                        })
                    trial_path.unlink(missing_ok=True)
                    (trial_path.parent / (trial_path.name + '-wal')).unlink(missing_ok=True)
                    (trial_path.parent / (trial_path.name + '-shm')).unlink(missing_ok=True)

    with (output / 'trials.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    overview = []
    for size in ordered_sizes:
        rows = [r for r in results if r['sources'] == size]
        overview.append({
            'sources': size,
            'corrected_sources': len(changes),
            'initial_active_tiles': rows[0]['active_tiles_before'],
            'affected_tiles': rows[0]['affected_tiles'],
            'modified_tiles': rows[0]['modified_tiles'],
            'affected_fraction': rows[0]['affected_fraction'],
            'median_incremental_commit_ms': float(np.median([r['incremental_commit_ms'] for r in rows])),
            'full_union_rebuild_ms_single': rows[0]['full_rebuild_ms'],
            'exact': all(r['exact'] for r in rows),
            'historical_revision_preserved': all(r['previous_revision_preserved'] for r in rows),
        })
    first, last = overview[0], overview[-1]
    result = {
        'step': '11.2-light',
        'purpose': 'corrected-source locality as total source/map size grows',
        'prepared': str(prepared),
        'sizes': ordered_sizes,
        'repeats': repeats,
        'correction_ids_constant': True,
        'voxel_size_m': voxel_size,
        'summary': overview,
        'growth': {
            'map_sources_factor': last['sources'] / first['sources'],
            'active_tiles_factor': last['initial_active_tiles'] / first['initial_active_tiles'] if first['initial_active_tiles'] else None,
            'incremental_commit_factor': last['median_incremental_commit_ms'] / first['median_incremental_commit_ms'] if first['median_incremental_commit_ms'] else None,
            'full_rebuild_factor': last['full_union_rebuild_ms_single'] / first['full_union_rebuild_ms_single'] if first['full_union_rebuild_ms_single'] else None,
        },
        'all_exact': all(r['exact'] for r in results),
        'all_previous_revisions_preserved': all(r['previous_revision_preserved'] for r in results),
        'limitations': ('Diagnostic only: 1-2 repeats recommended, warmed processes, no run-order randomization. '
                        'The full rebuild is an in-memory occupancy calculation; the incremental operation '
                        'includes a committed SQLite transaction. Do not interpret their ratio as a '
                        'publication-grade speedup. Fixed scan prefix can increase overlap in corrected tiles.'),
    }
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prepared', required=True, help='Step 11.1 prepared source directory')
    p.add_argument('--output', required=True, help='new output directory')
    p.add_argument('--sizes', nargs='+', required=True, type=int, metavar='N',
                   help='increasing scan counts, e.g. 64 128 256; all must include the corrected scans')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--voxel-size', type=float, default=0.5)
    args = p.parse_args(argv)
    result = run_scaling(args.prepared, args.output, sizes=args.sizes, repeats=args.repeats,
                         batch_size=args.batch_size, voxel_size=args.voxel_size)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
