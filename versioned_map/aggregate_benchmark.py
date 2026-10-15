"""Step 10.5A matched historical point and *genuine 4D range* benchmark."""
from __future__ import annotations
import csv
import json
import platform
from pathlib import Path
import time
import numpy as np

from .aggregate_4d import Merged4DIndex
from .interval_store import IntervalMap
from .store import VersionedStore


def _csv(path, rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def _summary(rows):
    out=[]
    for kind in sorted({r['kind'] for r in rows}):
        for method in sorted({r['method'] for r in rows}):
            vals=[r['elapsed_ms'] for r in rows if r['kind']==kind and r['method']==method]
            if not vals:continue
            v=np.asarray(vals)
            out.append({'kind':kind,'method':method,'n':len(v),
                        'median_ms':float(np.median(v)),
                        'mean_ms':float(v.mean()),
                        'p95_ms':float(np.percentile(v,95)),
                        'median_candidates':float(np.median([r['candidates'] for r in rows
                            if r['kind']==kind and r['method']==method]))})
    return out


def run_aggregate_benchmark(database, interval_root, output, *, queries=200, repeats=3, seed=10506):
    src=VersionedStore(database)
    root=Path(interval_root)
    bst=IntervalMap(root/'bst');morton=IntervalMap(root/'morton')
    merged=Merged4DIndex(bst,coalesced=True)
    raw=Merged4DIndex(bst,coalesced=False)
    reports={}
    for name,idx in (('bst_coalesced',merged),('bst_unmerged',raw)):
        if not idx.path.exists():reports[name]=idx.build()
        idx.load()
    bst._load_index();morton._load_index()
    sessions=list(src.all_versions());dims=np.asarray(src.meta()['dims'],dtype=np.int64)
    for sess in sessions:
        f=src.root/'indexes'/f'{sess}_morton_hash.npz'
        if not f.exists():src.build_spatial_index(sess,'morton_hash')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    samples=[]
    for i in range(queries):
        session=sessions[int(rng.integers(0,len(sessions)))]
        j=sessions.index(session);end=sessions[int(rng.integers(j,len(sessions)))]
        side=int(rng.choice([32,64,128]))
        if i%2==0:
            man=src.all_versions()[session]
            if man:
                keys=sorted(man)
                c=np.array([int(k) for k in keys[int(rng.integers(len(keys)))].split(',')])
                center=c*src.meta()['tile_side']+src.meta()['tile_side']//2
            else:center=rng.integers(0,dims)
        else:center=rng.integers(0,dims)
        low=center-side//2
        bounds=tuple(int(v) for d in range(3) for v in (low[d],low[d]+side))
        samples.append((session,end,bounds,side))
    methods=('snapshot_indexed','morton_interval','bst_interval','bst_unmerged','bst_coalesced')
    outputs=[]
    for rep in range(repeats):
        for qi,(session,end,bounds,side) in enumerate(samples):
            # This baseline includes the same SHA-256 validation but avoids
            # redundant canonical re-sorting of interval payloads.
            roll=(rep+qi)%len(methods)
            ordered=methods[roll:]+methods[:roll]
            result={};statistics={}
            for method in ordered:
                if method=='snapshot_indexed':
                    a,s=src.query(session,bounds,stats=True,index_mode='morton_hash')
                elif method=='morton_interval':
                    a,s=morton.query(session,bounds,stats=True,validation='hash')
                elif method=='bst_interval':
                    a,s=bst.query(session,bounds,stats=True,validation='hash')
                elif method=='bst_unmerged':
                    a,s=raw.query(session,bounds,stats=True)
                else:a,s=merged.query(session,bounds,stats=True)
                result[method]=a;statistics[method]=s
            reference=result['snapshot_indexed']
            for method in methods:
                if not np.array_equal(reference,result[method]):
                    raise AssertionError(f'point query mismatch {method}/{session}/{bounds}')
                s=statistics[method]
                outputs.append({'kind':'point','repeat':rep,'query_id':qi,'method':method,
                                'session':session,'to_session':'','side':side,
                                'voxels':len(reference),'elapsed_ms':s['latency_ms'],
                                'index_ms':s['index_ms'],'candidates':s.get('records_examined',s.get('keys_examined',0)),
                                'tiles_read':s['tiles_read'],'compressed_bytes_read':s['compressed_bytes_read']})
            if rep>0 and qi>=queries//2:
                continue  # 4D ranges: enough independent cases without duplicate expensive reads
            # Temporal *range*: retrieve ALL version-valid records intersecting
            # [session,end] and the requested XYZ box. No delta reconstruction.
            time_methods=('morton_interval','bst_interval','bst_unmerged','bst_coalesced')
            roll=(rep+qi)%len(time_methods)
            range_results={}
            for method in time_methods[roll:]+time_methods[:roll]:
                tic=time.perf_counter()
                if method=='morton_interval':rs,meta=morton.history_records(session,end,bounds)
                elif method=='bst_interval':rs,meta=bst.history_records(session,end,bounds)
                elif method=='bst_unmerged':rs,meta=raw.history_records(session,end,bounds)
                else:rs,meta=merged.history_records(session,end,bounds)
                elapsed=(time.perf_counter()-tic)*1000
                digest=tuple(sorted((r['key'],r['start'],r['stop'],r['hash']) for r in rs))
                range_results[method]=digest
                outputs.append({'kind':'4d_range','repeat':rep,'query_id':qi,'method':method,
                                'session':session,'to_session':end,'side':side,
                                'voxels':len(rs),'elapsed_ms':elapsed,'index_ms':elapsed,
                                'candidates':meta.get('records_examined',0),
                                'tiles_read':0,'compressed_bytes_read':0})
            if any(d != range_results['morton_interval'] for d in range_results.values()):
                raise AssertionError(f'4D range mismatch {session}/{end}/{bounds}')
    # A second set of records differing only by a changed payload is still
    # distinct for 4D range evaluation (not merely a union of tile coordinates).
    _csv(out/'trials.csv',outputs)
    result=_summary(outputs)
    _csv(out/'summary.csv',result)
    size={'bst_interval':bst.disk_stats(),'morton_interval':morton.disk_stats(),
          'bst_coalesced':merged.disk_stats(),'bst_unmerged':raw.disk_stats(),
          'snapshot':src.disk_stats(), 'build':reports}
    (out/'storage.json').write_text(json.dumps(size,indent=2))
    (out/'metadata.json').write_text(json.dumps({
        'step':'10.5A','queries':queries,'repeats':repeats,'seed':seed,'machine':platform.platform(),
        'index_frame':'BST-ID author Cell algebra over LOCAL integer tile grid, 21 bits x/y/z, UNIX t',
        'aggregation':'exact 4D TILE-PRESENCE cover; payload geometry NOT spatially aggregated',
        'benchmark_fairness':'hash + length verification; full canonical payload resorting omitted for both interval controls',
        'correctness':'exact voxel arrays for point queries and exact version-valid record sets for 4D ranges',
        'cache':'in-process warmed indexes, rotated methods, no decoded geometry cache',
        'caveat':'merged and unmerged share identical tile payloads; index bytes are additive sidecars to BST interval database',
    },indent=2))
    return result
