#!/usr/bin/env python3
"""Step 11.3: minimal real-database correction and rollback correctness smoke test.

Use a Step 11.1 corrected SQLite DB (a last pose-correction revision). The input
is NEVER modified. One cloned copy is reverted to pre-correction poses and then
restored; both states are checked against the independent full-rebuild oracle.

Example:
  python run_step113_smoke.py \\
      --database results/step11_1_experiment/corrected_map.sqlite \\
      --output results/step11_3_smoke.json

This tests only the updated engine, not a matched benchmark against old code.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from versioned_map.correction.engine import CorrectionMap


def run(database, output):
    database=Path(database)
    output=Path(output)
    if not database.is_file():
        raise FileNotFoundError(database)
    with tempfile.TemporaryDirectory(prefix='step113_') as tmp:
        cloned=Path(tmp)/'correction.sqlite'
        with sqlite3.connect(f'file:{database.resolve()}?mode=ro',uri=True) as original:
            with sqlite3.connect(str(cloned)) as duplicate:
                original.backup(duplicate)  # SQLite-native consistent copy; outside timing
        with CorrectionMap(cloned) as db:
            current=db.revision
            row=db.con.execute("SELECT MAX(revision) FROM revisions WHERE operation='pose_correction'").fetchone()
            last=int(row[0]) if row[0] is not None else None
            if last is None or last != current:
                raise ValueError('expected latest revision to be a pose_correction; supply Step 11.1 corrected_map.sqlite')
            source_ids=[r[0] for r in db.con.execute('SELECT source_id FROM pose_versions WHERE revision=? ORDER BY source_id',(last,))]
            if not source_ids:
                raise ValueError('no corrected sources in latest revision')
            before={sid:db._source_pose_asof(sid,last-1) for sid in source_ids}
            after={sid:db._source_pose_asof(sid,last) for sid in source_ids}
            bbox=db.con.execute('SELECT MIN(tx),MAX(tx),MIN(ty),MAX(ty),MIN(tz),MAX(tz) FROM tile_versions').fetchone()
            bounds=(int(bbox[0]*32),int((bbox[1]+1)*32),int(bbox[2]*32),int((bbox[3]+1)*32),int(bbox[4]*32),int((bbox[5]+1)*32))
            original_grid=db.query(bounds,last)
            source_shas=dict(db.con.execute('SELECT source_id,local_sha256 FROM sources'))

            t0=time.perf_counter()
            undone=db.correct_poses(before,note='Step 11.3 smoke: revert to initial poses')
            undo_ms=(time.perf_counter()-t0)*1000
            undo_valid=db.verify(undone.revision)
            persisted_old=db.query(bounds,last)
            old_preserved=np.array_equal(persisted_old,original_grid)

            t0=time.perf_counter()
            redone=db.correct_poses(after,note='Step 11.3 smoke: restore corrected poses')
            restore_ms=(time.perf_counter()-t0)*1000
            restore_valid=db.verify(redone.revision)
            restored_grid=db.query(bounds,redone.revision)
            same=np.array_equal(restored_grid,original_grid)
            hashes_unchanged=source_shas==dict(db.con.execute('SELECT source_id,local_sha256 FROM sources'))
            result={
                'step':'11.3-smoke','baseline_revision':last,'corrected_sources':len(source_ids),
                'revert_ms':round(undo_ms,3),'restore_ms':round(restore_ms,3),
                'revert_affected_tiles':undone.tiles_touched,'restore_affected_tiles':redone.tiles_touched,
                'revert_modified_tiles':undone.tiles_modified,'restore_modified_tiles':redone.tiles_modified,
                'revert_exact_vs_full_rebuild':bool(undo_valid),
                'restore_exact_vs_full_rebuild':bool(restore_valid),
                'restore_matches_previously_committed':bool(same),
                'original_revision_preserved':bool(old_preserved),
                'immutable_sources_unchanged':bool(hashes_unchanged),
                'overall_pass':bool(undo_valid and restore_valid and same and old_preserved and hashes_unchanged),
                'notes':'Single diagnostic run, cold/warm cache uncontrolled; SQLite backup excluded from timings; no cross-system or matched baseline comparison.'
            }
            if not result['overall_pass']:
                raise AssertionError(f'failed correctness gate: {result}')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--database',required=True)
    parser.add_argument('--output',default='results/step11_3_smoke.json')
    args=parser.parse_args()
    print(json.dumps(run(args.database,args.output),indent=2))


if __name__=='__main__':main()
