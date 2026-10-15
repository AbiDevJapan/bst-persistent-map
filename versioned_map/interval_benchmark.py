"""Paired exact ROI benchmark for Step 10.5 direct historical version queries."""
from __future__ import annotations
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
from .store import VersionedStore
from .interval_store import IntervalMap


def _write_csv(path,rows):
    with open(path,'w',newline='') as f:
        if not rows:raise ValueError('no benchmark rows')
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def _sum_latency(samples):
    a=np.asarray(samples,dtype=np.float64)
    return dict(n=len(a),median=float(np.median(a)),p95=float(np.percentile(a,95)),mean=float(a.mean()))


def run_interval_benchmark(database, interval_root, output, *, queries=200, repeats=3, seed=10505, verify_full=False):
    source=VersionedStore(database)
    stores={mode:IntervalMap(Path(interval_root)/mode) for mode in ('bst','morton')}
    for item in stores.values():item.validate_structure()
    sessions=list(source.all_versions())
    for item in stores.values():
        if item.meta['sessions']!=sessions:raise ValueError('interval db incomplete: append versions first')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Strong control: ensure Step 10.3 Morton indexed snapshots exist before
    # timing, rather than comparing BST against only an unindexed manifest scan.
    for session in sessions:
        path=source.root/'indexes'/f'{session}_morton_hash.npz'
        if not path.exists():source.build_spatial_index(session,'morton_hash')

    rng=np.random.default_rng(seed);dims=np.asarray(source.meta()['dims'],dtype=np.int64)
    samples=[]
    for i in range(queries):
        side=int(rng.choice([32,64,128]))
        # Half the queries centered on observed voxels, others uniform; unlike
        # always-uniform regions this also stresses nonempty ROI reconstruction.
        session=sessions[int(rng.integers(0,len(sessions)))]
        if i%2==0:
            manifest=source.all_versions()[session]
            if manifest:
                key=list(sorted(manifest))[int(rng.integers(0,len(manifest)))]
                tile=np.array([int(x) for x in key.split(',')]);center=tile*source.meta()['tile_side']+source.meta()['tile_side']//2
            else:center=rng.integers(0,dims)
        else:center=rng.integers(0,dims)
        lower=center-side//2
        b=tuple(int(v) for d in range(3) for v in (lower[d],lower[d]+side))
        samples.append((session,b,side))
    # Warm the indexes outside timing, and report their physical storage in metadata.
    rows=[]
    if verify_full:
        bb=tuple(int(v) for d in dims for v in (0,d))
        for session in sessions:
            expected=source.query(session,bb)
            for mode,item in stores.items():
                np.testing.assert_array_equal(expected,item.query(session,bb),err_msg=f'FULL MAP {mode}/{session}')
    # Rotate method order to reduce the method-always-first caching bias.
    for rep in range(repeats):
        for i,(session,b,side) in enumerate(samples):
            methods=('snapshot_scan','snapshot_indexed','bst','morton')
            methods=methods[(i+rep)%len(methods):]+methods[:(i+rep)%len(methods)]
            results={};st={}
            for mode in methods:
                if mode=='snapshot_scan':arr,info=source.query(session,b,stats=True,index_mode='scan')
                elif mode=='snapshot_indexed':arr,info=source.query(session,b,stats=True,index_mode='morton_hash')
                else:arr,info=stores[mode].query(session,b,stats=True)
                results[mode]=arr;st[mode]=info
            for mode in ('bst','morton'):
                if not np.array_equal(results['snapshot_scan'],results[mode]):
                    raise AssertionError(f'query mismatch {mode}, version={session}, bounds={b}')
            if not np.array_equal(results['snapshot_scan'],results['snapshot_indexed']):
                raise AssertionError('indexed snapshot control differs from manifest scan')
            for mode in ('snapshot_scan','snapshot_indexed','bst','morton'):
                t=st[mode]
                rows.append({'query_id':i,'repeat':rep,'session':session,
                             'region_side_voxels':side,'mode':mode,'voxels':len(results[mode]),
                             'query_ms':t['latency_ms'],'index_ms':t['index_ms'],
                             'tiles_read':t['tiles_read'],'compressed_bytes_read':t['compressed_bytes_read'],
                             'candidate_records':t.get('records_examined',t.get('keys_examined',0)),
                             'temporal_candidates':t.get('temporal_candidates',0),
                             'replayed_transitions':t.get('replayed_transitions',0),
                             'equal_snapshot':True})
    _write_csv(out/'trials.csv',rows)
    summary=[]
    for mode in ('snapshot_scan','snapshot_indexed','bst','morton'):
        for metric in ('query_ms','index_ms','compressed_bytes_read','tiles_read'):
            x=[r[metric] for r in rows if r['mode']==mode]
            summary.append({'mode':mode,'metric':metric,**_sum_latency(x)})
        for side in (32,64,128):
            x=[r['query_ms'] for r in rows if r['mode']==mode and r['region_side_voxels']==side]
            if x:summary.append({'mode':mode,'metric':f'query_ms_side_{side}',**_sum_latency(x)})
    _write_csv(out/'summary.csv',summary)
    size={mode:stores[mode].disk_stats() for mode in stores}
    size['source_snapshots']=source.disk_stats()
    (out/'storage.json').write_text(json.dumps(size,indent=2))
    (out/'metadata.json').write_text(json.dumps({
        'step':'10.5','sessions':sessions,'queries':queries,'repeats':repeats,'seed':seed,
        'verified_full_historical_maps':verify_full,'correctness':'exact canonical uint64 codes for all ROI queries',
        'timing':'warm-cache in-process, method order rotated, no decoded voxel caching, indexed snapshot control uses Step 10.3 Morton hash',
        'machine':platform.platform(),
        'semantics':'map version validity, not physical occupancy persistence',
        'attribution':'both interval databases store same complete immutable tile payloads, same validity runs; only index/identifiers differ'
    },indent=2))
    return summary
