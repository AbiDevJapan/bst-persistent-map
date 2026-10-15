"""Step 10.5B paired adaptive-geometry benchmark.

Snapshot and adaptive database use identical lossless voxel semantics; comparison
of BST vs Morton isolates the tile selector on the SAME shared block geometry.
No misleading claims about full BST native per-cell 4D trie or scan surfels.
"""
from __future__ import annotations
from pathlib import Path
import csv
import json
import platform
import time
import numpy as np
from .store import VersionedStore
from .geometry_105b import Geometry4D


def _queries(source, n, seed):
    rng=np.random.default_rng(seed)
    dims=source.meta()['dims'];sessions=list(source.all_versions())
    questions=[]
    versions=source.all_versions();tile_side=int(source.meta()['tile_side'])
    for i in range(n):
        side=int(rng.choice([16,32,64,128]))
        session=sessions[int(rng.integers(len(sessions)))]
        if i%2==0 and versions[session]:
            # A known occupied tile; prevents a benchmark dominated by empty ROIs.
            keys=sorted(versions[session]);k=keys[int(rng.integers(len(keys)))]
            origin=np.asarray([int(x) for x in k.split(',')])*tile_side
            center=[int(a+tile_side//2) for a in origin]
        else:
            center=[int(rng.integers(-10,int(d)+10)) for d in dims]
        bounds=tuple(v for c in center for v in (int(c-side//2),int(c+side//2)))
        questions.append((session,bounds,side))
    return questions


def verify_geometry(database,geometry,queries=200,seed=10507,full=False):
    source=VersionedStore(database);db=Geometry4D(geometry)
    sessions=db.meta['sessions'];source_s=list(source.all_versions())
    if sessions!=source_s[:len(sessions)]:raise ValueError('geometry and reference session order differs')
    full_count=0;verified=0
    if full:
        bounds=tuple(v for d in source.meta()['dims'] for v in (0,int(d)))
        for s in sessions:
            expected=source.query(s,bounds)
            for mode in ('bst','morton'):
                np.testing.assert_array_equal(db.query(s,bounds,mode=mode,max_output=max(10_000_000,len(expected)+1)), expected)
                verified+=1
            full_count+=1
    for s,b,_ in _queries(source,queries,seed):
        expected=source.query(s,b)
        for mode in ('bst','morton'):
            np.testing.assert_array_equal(db.query(s,b,mode=mode),expected,
                         err_msg=f'geometry mismatch: {mode}, {s}, {b}')
            verified+=1
    return {'verified_query_calls':verified,'full_maps_verified':full_count,
            'correctness':'bit-for-bit sorted uint64 observed endpoints',
            'semantics':'observation-history records, not free-space evidence'}


def _summary(rows):
    groups={}
    for r in rows:
        for metric in ('latency_ms','index_ms','tile_candidates','blocks_read','voxels'):
            groups.setdefault((r['mode'],r['roi_side'],metric),[]).append(float(r.get(metric,0)))
    out=[]
    for (mode,side,metric),a in sorted(groups.items()):
        z=np.asarray(a,dtype=float)
        out.append({'mode':mode,'roi_side':side,'metric':metric,'n':len(a),
                    'median':float(np.median(z)),'p95':float(np.percentile(z,95)),
                    'mean':float(np.mean(z))})
    return out


def run_geometry_benchmark(database,geometry,output,*,queries=200,repeats=3,seed=10507,unmerged=None):
    source=VersionedStore(database);db=Geometry4D(geometry)
    baseline=Geometry4D(unmerged) if unmerged else None
    if db.meta['sessions']!=list(source.all_versions()):
        raise ValueError('geometry benchmark needs all source versions')
    if baseline and baseline.meta['k']!=1:raise ValueError('unmerged geometry control must have K=1')
    # Compare against an actually INDEXED Step 10.3 snapshot control, not a
    # disadvantaged full manifest scan. Index construction is untimed.
    snapshot_index_built=[]
    for name in source.all_versions():
        filename=source.root/'indexes'/f'{name}_morton_hash.npz'
        if not filename.exists():
            source.build_spatial_index(name,'morton_hash')
            snapshot_index_built.append(name)
    tasks=_queries(source,queries,seed)
    # Preload modes outside warm-query timing; benchmark includes geometry decompression.
    db._load_index('bst');db._load_index('morton')
    if baseline:baseline._load_index('morton')
    rows=[];methods=['snapshot','morton','bst']+(['k1_morton'] if baseline else [])
    for repeat in range(repeats):
        for qi,(session,bounds,side) in enumerate(tasks):
            expected=source.query(session,bounds)
            order=methods[(qi+repeat)%len(methods):]+methods[:(qi+repeat)%len(methods)]
            for mode in order:
                if mode=='snapshot':
                    result,st=source.query(session,bounds,stats=True,index_mode='morton_hash')
                elif mode=='k1_morton':
                    result,st=baseline.query(session,bounds,stats=True,mode='morton')
                else:
                    result,st=db.query(session,bounds,stats=True,mode=mode)
                if not np.array_equal(result,expected):
                    raise AssertionError(f'query wrong: {mode} {session} {bounds}')
                rows.append({'query_id':qi,'repeat':repeat,'session':session,
                             'bounds':','.join(map(str,bounds)),'roi_side':side,
                             'mode':mode,'latency_ms':st['latency_ms'],
                             'index_ms':st['index_ms'],
                             'tile_candidates':st.get('tile_candidates',st.get('tiles_read',0)),
                             'blocks_read':st.get('blocks_read',0),'voxels':len(result),
                             'exact':1})
    # Primitive-mode benchmark isolates geometric savings from the cost of
    # expanding an adaptive representation back into fine voxel IDs.
    primitive_rows=[]
    for repeat in range(repeats):
        for qi,(session,bounds,side) in enumerate(tasks):
            expected_n=len(source.query(session,bounds))
            p_methods=['morton','bst']+(['k1_morton'] if baseline else [])
            order=p_methods[(qi+repeat)%len(p_methods):]+p_methods[:(qi+repeat)%len(p_methods)]
            for mode in order:
                obj=baseline if mode=='k1_morton' else db
                actual_mode='morton' if mode=='k1_morton' else mode
                blocks,info=obj.primitives(session,bounds,mode=actual_mode,stats=True)
                lo=np.maximum(blocks[:,:3],np.asarray(bounds[::2])[None,:])
                hi=np.minimum(blocks[:,:3]+blocks[:,3:],np.asarray(bounds[1::2])[None,:])
                occupied=int(np.maximum(hi-lo,0).prod(axis=1).sum())
                if occupied!=expected_n:
                    raise AssertionError(f'primitive cover count incorrect: {mode} {session} {bounds}')
                primitive_rows.append({'query_id':qi,'repeat':repeat,'session':session,
                    'bounds':','.join(map(str,bounds)),'roi_side':side,'mode':mode,
                    'latency_ms':info['latency_ms'],'index_ms':info['index_ms'],
                    'primitives':len(blocks),'voxels':expected_n,'exact_coverage':1})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=_summary(rows)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    with (output/'primitive_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(primitive_rows[0]));w.writeheader();w.writerows(primitive_rows)
    pg={}
    for r in primitive_rows:
        for metric in ('latency_ms','index_ms','primitives'):
            pg.setdefault((r['mode'],r['roi_side'],metric),[]).append(float(r[metric]))
    p_summary=[{'mode':mode,'roi_side':side,'metric':metric,'n':len(vals),
                'median':float(np.median(vals)),'p95':float(np.percentile(vals,95)),
                'mean':float(np.mean(vals))}
               for (mode,side,metric),vals in sorted(pg.items())]
    with (output/'primitive_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(p_summary[0]));w.writeheader();w.writerows(p_summary)
    sizes={'adaptive':db.storage(),'source_snapshot':source.disk_stats()}
    if baseline:sizes['k1_geometry']=baseline.storage()
    (output/'storage.json').write_text(json.dumps(sizes,indent=2))
    (output/'metadata.json').write_text(json.dumps({
        'step':'10.5B','snapshot_control':'indexed Morton hash Step 10.3','snapshot_indexes_built_before_timing':snapshot_index_built,'queries':queries,'repeats':repeats,'seed':seed,
        'sessions':db.meta['sessions'],'machine':platform.platform(),
        'correctness':'paired exact uint64 occupancy equality for every query and mode',
        'workload':'50% occupied-tile-centred and 50% random regions; sizes 16/32/64/128 voxels',
        'spatial_merging':'exact occupied dyadic siblings, K support cap; occupancy reversible',
        'temporal_merging':'same spatial cell stored once with session membership mask',
        'author_codec':'official BST-ID Cell 4D, local integer frame, unchanged upstream implementation',
        'claim_scope':'BST-ID tile-time selector not a fully optimized per-geometry-cell 4D trie',
        'benchmark_conditions':'warm in-process indexes, rotated order; data decompression on every read, no decoded block cache',
        'limitations':'no original point statistics; does not reproduce old planarity-gated K4 surfels; '
                      'not a separate hardware end-to-end benchmark'},indent=2))
    return summary
