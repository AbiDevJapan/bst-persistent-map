"""Paired exact historical-ROI delta journal benchmark (Step 10.4)."""
from __future__ import annotations
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
from .store import VersionedStore
from .delta_journal import DeltaJournal, source_storage_controls, cumulative_storage_rows


def _csv(path, rows):
    with Path(path).open('w', newline='') as file:
        writer=csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def verify_against_store(store, journal, *, random_queries=100, seed=10404, full=False):
    versions=store.all_versions(); names=list(versions)
    if journal.meta['sessions'] != names:
        raise ValueError('session lists differ')
    if journal.meta['source_meta']['dims']!=store.meta()['dims']:
        raise ValueError('grid dimensions differ')
    from .region_index import manifest_hash
    for sess in names:
        if journal.meta['source_manifest_hashes'][sess] != manifest_hash(versions[sess]):
            raise ValueError('source manifest changed since journal build')
    rng=np.random.default_rng(seed)
    dims=np.asarray(store.meta()['dims'],dtype=int)
    side=int(store.meta()['tile_side'])
    count=0
    if full:
        b=tuple(int(v) for j in range(3) for v in (0,dims[j]))
        for sess in names:
            np.testing.assert_array_equal(store.query(sess,b),journal.reconstruct(sess))
            count+=1
    for i in range(random_queries):
        session=names[i % len(names)]
        tiles=list(versions[session]); size=int([1,16,32,64,128][i%5])
        if tiles and i%7:
            center=np.asarray([int(x) for x in tiles[int(rng.integers(len(tiles)))].split(',')])*side + rng.integers(0,side,size=3)
        else:center=rng.integers(0,dims)
        lo=np.maximum(0,center-size//2);hi=np.minimum(dims,lo+size);lo=np.maximum(0,hi-size)
        bounds=tuple(int(v) for j in range(3) for v in (lo[j],hi[j]))
        np.testing.assert_array_equal(store.query(session,bounds),journal.query(session,bounds))
        count+=1
    return {'checked_queries': count, 'full_historical_maps_checked':len(names) if full else 0,
            'correctness':'exact sorted uint64 observation-code equality'}


def run_delta_benchmark(database, journal_root, output, *, queries=100, repeats=3, seed=10404):
    if queries<=0 or repeats<=0:raise ValueError('query count and repeats must be positive')
    store=VersionedStore(database)
    journals={mode:DeltaJournal(Path(journal_root)/mode) for mode in ('morton','bst')}
    for journal in journals.values():
        journal.validate_structure()
        verify_against_store(store,journal,random_queries=min(20,queries),seed=seed)
    out=Path(output);out.mkdir(parents=True, exist_ok=True)
    names=list(store.all_versions())
    side=int(store.meta()['tile_side']);dims=np.asarray(store.meta()['dims'],dtype=int)
    rng=np.random.default_rng(seed);roi=[]
    for i in range(queries):
        session=names[i%len(names)]; tiles=list(store.all_versions()[session]);size=int([32,64,128][i%3])
        if tiles and i%5:
            center=np.asarray([int(x) for x in tiles[int(rng.integers(len(tiles)))].split(',')])*side + rng.integers(0,side,size=3)
        else:center=rng.integers(0,dims)
        lo=np.maximum(0,center-size//2);hi=np.minimum(dims,lo+size);lo=np.maximum(0,hi-size)
        if np.any(hi<=lo):continue
        b=tuple(int(v) for j in range(3) for v in (lo[j],hi[j]));roi.append((i,session,b))
    modes=['snapshot','morton_delta','bst_delta']
    rows=[]
    for rep in range(repeats):
        for qid,session,bounds in roi:
            # Source reference always obtained for correctness; not counted in timed path.
            expected=store.query(session,bounds)
            ordered=modes[(qid+rep)%3:]+modes[:(qid+rep)%3]
            for mode in ordered:
                start=time.perf_counter()
                if mode=='snapshot':
                    got,stat=store.query(session,bounds,stats=True)
                    byte_count=stat['compressed_bytes_read']
                    candidate=stat['tiles_read'];applied=0
                else:
                    got,stat=journals[mode.split('_')[0]].query(session,bounds,stats=True)
                    byte_count=stat['compressed_bytes_accessed']
                    candidate=stat['candidate_tiles'];applied=stat['transition_records_applied']
                elapsed=(time.perf_counter()-start)*1000
                if not np.array_equal(got,expected):
                    raise AssertionError(f'inexact {mode} reconstruction: query {qid} {session} {bounds}')
                rows.append({'repeat':rep,'query_id':qid,'session':session,
                             'bounds':','.join(map(str,bounds)), 'region_side_voxels':bounds[1]-bounds[0],
                             'mode':mode,'observed_voxels':len(got),'query_ms':elapsed,
                             'compressed_bytes_accessed':byte_count,'candidate_tiles':candidate,
                             'transition_records_applied':applied,'exact':True})
    summary=[]
    for mode in modes:
        group=[r for r in rows if r['mode']==mode]
        for metric in ('query_ms','compressed_bytes_accessed','candidate_tiles','transition_records_applied'):
            arr=np.asarray([float(x[metric]) for x in group])
            summary.append({'mode':mode,'metric':metric,'n':len(arr),'median':float(np.median(arr)),
                            'p95':float(np.percentile(arr,95)),'mean':float(np.mean(arr))})
    _csv(out/'trials.csv',rows);_csv(out/'summary.csv',summary)
    _csv(out/'growth.csv', cumulative_storage_rows(store,journals))
    storage={'controls':source_storage_controls(store),'journals':{mode:j.disk_stats() for mode,j in journals.items()},
             'build_transitions':{mode:j.meta['transitions'] for mode,j in journals.items()},
             'journal_build_ms':{mode:j.meta['build_ms'] for mode,j in journals.items()}}
    (out/'storage.json').write_text(json.dumps(storage,indent=2))
    metadata={'step':'10.4','queries':len(roi),'repeats':repeats,'seed':seed,'sessions':names,
              'region_sides_voxels':[32,64,128], 'machine':platform.platform(),
              'timing':'warm page cache, decoded blobs NOT cached by either journal; sequential replay; operations are exact',
              'semantics':'observation history; absence is not free-space evidence',
              'attribution':'BST-vs-Morton differs ONLY by stored tile-address labels, NOT candidate search; full temporal region algebra is not implemented',
              'correctness':'snapshot comparison exact for every query and mode'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return summary
