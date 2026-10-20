"""Paired full-rebuild vs incremental correction benchmark."""
from __future__ import annotations
import csv
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np
from .engine import CorrectionMap
from .voxel import Pose


def run_correction_benchmark(database: str | Path, output: str | Path, *,
                             repeats: int = 5, translation=(0.5,0.0,0.0), source_id=None):
    """Full map correctness comparison with paired timing.

    Each repetition clones the current DB OUTSIDE the timed correction operation.
    Reference computation scans all source observations and is deliberately unoptimized.
    The query and correction benchmark does not include clone/setup time.
    """
    if repeats <= 0: raise ValueError('repeats must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with CorrectionMap(database) as original:
        if source_id is None:
            r=original.con.execute('SELECT source_id FROM sources ORDER BY point_count DESC LIMIT 1').fetchone()
            if r is None: raise ValueError('benchmark requires ingested sources')
            source_id=r[0]
        base_pose=original._source_pose_asof(source_id,original.revision)
        new_mat=base_pose.matrix.copy();new_mat[:3,3]+=np.asarray(translation,dtype=np.float64)
        corrected=Pose(new_mat)
        for trial in range(repeats):
            t=time.perf_counter()
            ref=original.full_rebuild(pose_overrides={source_id:corrected})
            rebuild_ms=(time.perf_counter()-t)*1000
            with tempfile.TemporaryDirectory(prefix='correction_bench_') as d:
                dest=Path(d)/'clone.sqlite'
                clone=sqlite3.connect(str(dest))
                original.con.backup(clone)
                clone.close()
                with CorrectionMap(dest) as engine:
                    result=engine.correct_poses({source_id:corrected},note='paired benchmark')
                    t=time.perf_counter();
                    actual=engine.query(_full_bounds(ref,original.full_rebuild()))
                    query_ms=(time.perf_counter()-t)*1000
                    if not np.array_equal(actual,ref):
                        raise AssertionError('incremental correction != independent full rebuild')
                    stats=engine.stats()
                    rows.append(dict(trial=trial,source_id=source_id,full_rebuild_ms=rebuild_ms,
                                     incremental_ms=result.local_update_ms,query_ms=query_ms,
                                     tiles_touched=result.tiles_touched,tiles_modified=result.tiles_modified,
                                     updated_file_bytes=stats['physical_sqlite_bytes_including_wal'],exact_match=True))
    with (output/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    result={'repeats':repeats,'source_id':source_id,'reference':'independent all-sources rebuild',
            'full_rebuild_median_ms':float(np.median([r['full_rebuild_ms'] for r in rows])),
            'incremental_median_ms':float(np.median([r['incremental_ms'] for r in rows])),
            'incremental_query_median_ms':float(np.median([r['query_ms'] for r in rows])),
            'exact':True,'excluded':'database cloning and process startup',
            'caveat':'synthetic corrections on user-provided immutable local measurements; no localization validation'}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def _full_bounds(a:np.ndarray,b:np.ndarray):
    if len(a)==0 and len(b)==0: return (0,1,0,1,0,1)
    both=np.concatenate([c for c in [a,b] if len(c)],axis=0)
    lo=both.min(axis=0);hi=both.max(axis=0)+1
    return (int(lo[0]),int(hi[0]),int(lo[1]),int(hi[1]),int(lo[2]),int(hi[2]))
