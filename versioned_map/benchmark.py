"""Step 10.3 paired random-ROI benchmark over fixed historical manifests.
Results are measured from an actual built store; never manufactured timing.
"""
from __future__ import annotations
import csv,json,platform,statistics,time
from pathlib import Path
import numpy as np
from .store import VersionedStore
MODES=('scan','bst_prefix','morton_hash')

def _csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run_benchmark(database,output,*,query_count=200,repeats=3,seed=10303,
                  size_voxels=(32,64,128),build_indexes=False):
    if query_count<=0 or repeats<=0 or any(int(x)<=0 for x in size_voxels):
        raise ValueError('positive query count, repetitions and ROI sizes required')
    store=VersionedStore(database);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    sessions=list(store.all_versions())
    if not sessions:raise ValueError('no historical sessions in the database')
    modes=MODES if store.meta()['codec']=='official' else ('scan','morton_hash')
    builds=[]
    if build_indexes:
        for name in sessions:
            for mode in modes[1:]:builds.append(store.build_spatial_index(name,mode))
    else:
        for name in sessions:
            for mode in modes[1:]:
                # Fail before expensive benchmarking if a required persisted index is missing.
                store._matching_keys(store.all_versions()[name], (0,1,0,1,0,1),session=name,index_mode=mode)
    rng=np.random.default_rng(seed);dims=np.asarray(store.meta()['dims'],dtype=int)
    versions=store.all_versions();queries=[]
    for i in range(query_count):
        session=sessions[i%len(sessions)];tiles=list(versions[session]);side=store.meta()['tile_side']
        if tiles and i%5!=0:
            tile=tiles[int(rng.integers(len(tiles)))];cent=np.asarray(list(map(int,tile.split(','))))*side
            center=cent+rng.integers(0,side,size=3)
        else:
            center=rng.integers(0,np.maximum(dims,1))
        size=int(size_voxels[i%len(size_voxels)]);
        # Half-open query, includes boundary clipping and some no-hit regions.
        lo=np.maximum(0,center-size//2);hi=np.minimum(dims,lo+size);lo=np.maximum(0,hi-size)
        if np.any(hi<=lo):continue
        bounds=tuple(int(v) for j in range(3) for v in (lo[j],hi[j]))
        queries.append((i,session,bounds))
    trials=[]
    for rep in range(repeats):
        for qid,session,bounds in queries:
            # Round-robin method order to mitigate warm-cache ordering bias.
            mode_order=list(modes);rot=(qid+rep)%len(modes);mode_order=mode_order[rot:]+mode_order[:rot]
            # Warm the page cache uniformly and construct the exact reference
            # BEFORE the rotated measurement order. This is a warm-cache test.
            reference=store.query(session,bounds,index_mode='scan')
            control_keys=set(store._matching_keys(versions[session],bounds))
            for mode in mode_order:
                t0=time.perf_counter()
                values,info=store.query(session,bounds,stats=True,index_mode=mode)
                duration=(time.perf_counter()-t0)*1000
                keys=set(store._matching_keys(versions[session],bounds,session=session,index_mode=mode))
                if keys!=control_keys:
                    raise AssertionError(f'tile selection mismatch for {session} ROI={bounds} index={mode}: missing={control_keys-keys} extra={keys-control_keys}')
                if not np.array_equal(values,reference):
                    raise AssertionError(f'voxel mismatch for {session} ROI={bounds} index={mode}')
                trials.append({'repeat':rep,'query_id':qid,'session':session,'bounds':','.join(map(str,bounds)),
                              'mode':mode,'voxels':len(values),'query_ms':info['latency_ms'],
                              'end_to_end_call_ms':duration,'index_ms':info['index_ms'],
                              'keys_examined':info['keys_examined'],'index_probes':info['index_probes'],
                              'tiles_read':info['tiles_read'],'blob_bytes_read':info['compressed_bytes_read'],
                              'equivalent_to_scan':True})
    summary=[]
    for mode in modes:
        group=[r for r in trials if r['mode']==mode]
        for metric in ('query_ms','index_ms','keys_examined','tiles_read','blob_bytes_read'):
            nums=np.asarray([r[metric] for r in group],float)
            summary.append({'mode':mode,'metric':metric,'n':len(nums),'median':float(np.median(nums)),
                            'p95':float(np.percentile(nums,95)),'mean':float(np.mean(nums))})
    _csv_write(out/'trials.csv',trials);_csv_write(out/'summary.csv',summary)
    index_files=list((store.root/'indexes').glob('*.npz')) if (store.root/'indexes').exists() else []
    metadata={'step':'10.3','dataset':'existing Step 10 store','codec':store.meta()['codec'],
              'sessions':sessions,'query_count':len(queries),'repeats':repeats,'seed':seed,
              'size_voxels':list(size_voxels),'modes':modes,
              'indexes_built_this_run':builds,'index_files_bytes':sum(p.stat().st_size for p in index_files),
              'snapshot_store':store.disk_stats(), 'machine':platform.platform(),
              'correctness':'exact tile keys and exact sorted uint64 voxel query results',
              'caveat':'warm-cache mixed-order timing; author IDs prefix-index x/y/f only, temporal selection by manifest; prototype/localization unaffected'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return summary
