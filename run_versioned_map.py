#!/usr/bin/env python3
"""Step 10 versioned mapping. Does not import localization or neural libraries."""
import argparse,json,time
from pathlib import Path
import numpy as np
from versioned_map.store import VersionedStore,SESSIONS

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('build');a.add_argument('--database',required=True);a.add_argument('--data-root',required=True)
    a.add_argument('--cache-dir',default='.cache/step8_1');a.add_argument('--codec',choices=['official','prototype','morton'],default='official')
    a.add_argument('--sessions',nargs='+',default=SESSIONS);a.add_argument('--tile-side',type=int,default=32)
    a.add_argument('--voxel-size',type=float,default=0.5);a.add_argument('--scan-stride',type=int,default=10)
    a.add_argument('--min-range',type=float,default=2.0);a.add_argument('--max-range',type=float,default=80.0)
    a.add_argument('--max-scans-per-session',type=int,default=None)
    a=sub.add_parser('index');a.add_argument('--database',required=True);a.add_argument('--sessions',nargs='*');a.add_argument('--index-mode',choices=['bst_prefix','morton_hash','both'],default='both')
    a=sub.add_parser('benchmark');a.add_argument('--database',required=True);a.add_argument('--output',required=True);a.add_argument('--queries',type=int,default=200);a.add_argument('--repeats',type=int,default=3);a.add_argument('--seed',type=int,default=10303);a.add_argument('--build-indexes',action='store_true')
    a=sub.add_parser('query');a.add_argument('--database',required=True);a.add_argument('--session',required=True);a.add_argument('--bounds',type=int,nargs=6,required=True,metavar=('X0','X1','Y0','Y1','Z0','Z1'));a.add_argument('--save-npy');a.add_argument('--index-mode',choices=['scan','bst_prefix','morton_hash'],default='scan')
    a=sub.add_parser('changes');a.add_argument('--database',required=True);a.add_argument('--from-session',required=True);a.add_argument('--to-session',required=True);a.add_argument('--bounds',type=int,nargs=6,required=True);a.add_argument('--save-prefix');a.add_argument('--index-mode',choices=['scan','bst_prefix','morton_hash'],default='scan')
    a=sub.add_parser('sync');a.add_argument('--database',required=True);a.add_argument('--from-session',required=True);a.add_argument('--to-session',required=True);a.add_argument('--bounds',type=int,nargs=6,required=True);a.add_argument('--output',required=True);a.add_argument('--index-mode',choices=['scan','bst_prefix','morton_hash'],default='scan')
    a=sub.add_parser('history');a.add_argument('--database',required=True);a.add_argument('--from-session',required=True);a.add_argument('--to-session',required=True);a.add_argument('--bounds',type=int,nargs=6,required=True);a.add_argument('--index-mode',choices=['scan','bst_prefix','morton_hash'],default='scan')
    a=sub.add_parser('delta-build',help='Build two self-contained exact base+delta archives (Step 10.4)')
    a.add_argument('--database',required=True);a.add_argument('--journal-root',required=True)
    a.add_argument('--address-mode',choices=['both','morton','bst'],default='both')
    a.add_argument('--through-session',default=None,help='Build only through this session; append later sessions with delta-append')
    a=sub.add_parser('delta-append',help='Append next chronological session to existing delta archive')
    a.add_argument('--database',required=True);a.add_argument('--journal-root',required=True)
    a.add_argument('--session',required=True);a.add_argument('--address-mode',choices=['both','morton','bst'],default='both')
    a=sub.add_parser('delta-query',help='Exact historical region query via delta replay')
    a.add_argument('--journal',required=True);a.add_argument('--session',required=True)
    a.add_argument('--bounds',type=int,nargs=6,required=True);a.add_argument('--save-npy')
    a=sub.add_parser('delta-verify',help='Compare journal against source historical maps')
    a.add_argument('--database',required=True);a.add_argument('--journal-root',required=True)
    a.add_argument('--queries',type=int,default=100);a.add_argument('--seed',type=int,default=10404)
    a.add_argument('--full-maps',action='store_true')
    a=sub.add_parser('delta-stats',help='Compare independent snapshots, shared snapshots, and delta archives')
    a.add_argument('--database',required=True);a.add_argument('--journal-root',required=True)
    a=sub.add_parser('delta-benchmark',help='Timed paired benchmark, exact snapshot validation')
    a.add_argument('--database',required=True);a.add_argument('--journal-root',required=True)
    a.add_argument('--output',required=True);a.add_argument('--queries',type=int,default=100)
    a.add_argument('--repeats',type=int,default=3);a.add_argument('--seed',type=int,default=10404)
    a=sub.add_parser('interval-build',help='Build direct-read 4D BST-ID and matched Morton interval databases (Step 10.5)')
    a.add_argument('--database',required=True);a.add_argument('--interval-root',required=True)
    a.add_argument('--address-mode',choices=['both','bst','morton'],default='both')
    a.add_argument('--through-session',default=None)
    a.add_argument('--aggregate-cells',action='store_true',help='also build a lossless 4D tile-coverage index during interval-build')
    a=sub.add_parser('interval-append',help='Append the next chronological session, sharing unchanged interval runs')
    a.add_argument('--database',required=True);a.add_argument('--interval-root',required=True)
    a.add_argument('--session',required=True);a.add_argument('--address-mode',choices=['both','bst','morton'],default='both')
    a=sub.add_parser('interval-query',help='Direct historical region query without delta replay')
    a.add_argument('--interval',required=True);a.add_argument('--session',required=True)
    a.add_argument('--bounds',type=int,nargs=6,required=True);a.add_argument('--save-npy')
    a=sub.add_parser('interval-history',help='Query four-dimensional spatial-temporal range for validity records')
    a.add_argument('--interval',required=True);a.add_argument('--from-session',required=True)
    a.add_argument('--to-session',required=True);a.add_argument('--bounds',type=int,nargs=6,required=True)
    a=sub.add_parser('interval-stats',help='Count database blob, metadata, record, and 4D index bytes')
    a.add_argument('--interval-root',required=True)
    a=sub.add_parser('interval-verify',help='Check direct historical ROI and optional full map equality')
    a.add_argument('--database',required=True);a.add_argument('--interval-root',required=True)
    a.add_argument('--queries',type=int,default=100);a.add_argument('--seed',type=int,default=10505)
    a.add_argument('--full-maps',action='store_true')
    a=sub.add_parser('interval-benchmark',help='Paired BST4D/Morton/snapshot timings, exactness, storage')
    a.add_argument('--database',required=True);a.add_argument('--interval-root',required=True)
    a.add_argument('--output',required=True);a.add_argument('--queries',type=int,default=200)
    a.add_argument('--repeats',type=int,default=3);a.add_argument('--seed',type=int,default=10505)
    a.add_argument('--full-maps',action='store_true')
    a=sub.add_parser('stats');a.add_argument('--database',required=True)
    a=sub.add_parser('aggregate-build',help='Step 10.5A build merged 4D tile-presence index (plus unmerged control)')
    a.add_argument('--interval-root',required=True)
    a=sub.add_parser('aggregate-query',help='Direct historical query using merged BST 4D coverage')
    a.add_argument('--interval',required=True);a.add_argument('--session',required=True)
    a.add_argument('--bounds',type=int,nargs=6,required=True);a.add_argument('--unmerged',action='store_true')
    a.add_argument('--save-npy')
    a=sub.add_parser('aggregate-history',help='Genuine space-time range query over merged BST 4D coverage')
    a.add_argument('--interval',required=True);a.add_argument('--from-session',required=True)
    a.add_argument('--to-session',required=True);a.add_argument('--bounds',type=int,nargs=6,required=True)
    a.add_argument('--unmerged',action='store_true')
    a=sub.add_parser('aggregate-verify',help='Exact ROI and optional complete-history verification against snapshots')
    a.add_argument('--database',required=True);a.add_argument('--interval-root',required=True)
    a.add_argument('--queries',type=int,default=200);a.add_argument('--seed',type=int,default=10506)
    a.add_argument('--full-maps',action='store_true')
    a=sub.add_parser('aggregate-benchmark',help='Matched 4D range and point-query experiments')
    a.add_argument('--database',required=True);a.add_argument('--interval-root',required=True)
    a.add_argument('--output',required=True);a.add_argument('--queries',type=int,default=200)
    a.add_argument('--repeats',type=int,default=3);a.add_argument('--seed',type=int,default=10506)
    a=sub.add_parser('voxel4d-build',help='Optional region-bounded exact occupancy-cell XYZ+T compression')
    a.add_argument('--database',required=True);a.add_argument('--output',required=True)
    a.add_argument('--bounds',type=int,nargs=6,required=True)
    a.add_argument('--max-atoms',type=int,default=500000)
    a.add_argument('--unmerged',action='store_true')
    a=sub.add_parser('voxel4d-query',help='Reconstruct exact occupied-endpoint voxels from merged 4D cells')
    a.add_argument('--index',required=True);a.add_argument('--session',required=True)
    a.add_argument('--bounds',type=int,nargs=6,required=True)
    a.add_argument('--save-npy')
    a=sub.add_parser('geometry-build',help='Step 10.5B: build exact K-bounded adaptive 3D cells, merge version membership')
    a.add_argument('--database',required=True);a.add_argument('--geometry-root',required=True)
    a.add_argument('--k',type=int,default=4)
    a.add_argument('--through-session',default=None)
    a=sub.add_parser('geometry-append',help='Append one chronological version to Step 10.5B geometry history')
    a.add_argument('--database',required=True);a.add_argument('--geometry',required=True)
    a.add_argument('--session',required=True)
    a.add_argument('--gc',action='store_true',help='offline garbage collect unreferenced old geometry payloads')
    a=sub.add_parser('geometry-query',help='Exact historical ROI reconstructed from adaptive 4D geometry cells')
    a.add_argument('--geometry',required=True);a.add_argument('--session',required=True)
    a.add_argument('--bounds',type=int,nargs=6,required=True)
    a.add_argument('--mode',choices=['bst','morton'],default='bst')
    a.add_argument('--save-npy')
    a=sub.add_parser('geometry-primitives',help='Return compressed geometry blocks without expanding voxels')
    a.add_argument('--geometry',required=True);a.add_argument('--session',required=True)
    a.add_argument('--bounds',type=int,nargs=6,required=True)
    a.add_argument('--mode',choices=['bst','morton'],default='bst')
    a.add_argument('--save-npy')
    a=sub.add_parser('geometry-describe',help='Return centroids and covariance of exact filled 3D cells (not point surfels)')
    a.add_argument('--geometry',required=True);a.add_argument('--session',required=True)
    a.add_argument('--bounds',type=int,nargs=6,required=True)
    a.add_argument('--mode',choices=['bst','morton'],default='bst')
    a.add_argument('--limit',type=int,default=20)
    a=sub.add_parser('geometry-stats',help='Full adaptive geometry and index byte accounting')
    a.add_argument('--geometry',required=True)
    a=sub.add_parser('geometry-verify',help='Exact ROI and optional complete-map verification versus canonical snapshots')
    a.add_argument('--database',required=True);a.add_argument('--geometry',required=True)
    a.add_argument('--queries',type=int,default=200);a.add_argument('--seed',type=int,default=10507)
    a.add_argument('--full-maps',action='store_true')
    a=sub.add_parser('geometry-benchmark',help='Matched BST 4D vs Morton vs snapshot and K=1 baseline')
    a.add_argument('--database',required=True);a.add_argument('--geometry',required=True)
    a.add_argument('--unmerged-geometry',default=None,help='Optional K=1 database for geometry-only ablation')
    a.add_argument('--output',required=True);a.add_argument('--queries',type=int,default=200)
    a.add_argument('--repeats',type=int,default=3);a.add_argument('--seed',type=int,default=10507)
    a=p.parse_args()
    if a.cmd.startswith('geometry-'):
        from versioned_map.geometry_105b import Geometry4D
        if a.cmd=='geometry-build':
            db=Geometry4D.build(a.database,a.geometry_root,k=a.k,through_session=a.through_session)
            print(json.dumps(db.storage(),indent=2));return
        obj=Geometry4D(a.geometry)
        if a.cmd=='geometry-append':
            print(json.dumps(obj.append_from_store(a.database,a.session,gc=a.gc),indent=2));return
        if a.cmd=='geometry-stats':
            print(json.dumps(obj.storage(),indent=2));return
        if a.cmd=='geometry-query':
            arr,metrics=obj.query(a.session,tuple(a.bounds),mode=a.mode,stats=True)
            if a.save_npy:np.save(a.save_npy,arr)
            print(json.dumps({'voxels':len(arr),**metrics},indent=2));return
        if a.cmd=='geometry-primitives':
            cells,metrics=obj.primitives(a.session,tuple(a.bounds),mode=a.mode,stats=True)
            if a.save_npy:np.save(a.save_npy,cells)
            print(json.dumps(metrics,indent=2));return
        if a.cmd=='geometry-describe':
            blocks=obj.describe(a.session,tuple(a.bounds),mode=a.mode)
            print(json.dumps({'total_blocks':len(blocks),'shown':blocks[:a.limit]},indent=2));return
        if a.cmd=='geometry-verify':
            from versioned_map.geometry_benchmark_105b import verify_geometry
            print(json.dumps(verify_geometry(a.database,a.geometry,a.queries,a.seed,a.full_maps),indent=2));return
        if a.cmd=='geometry-benchmark':
            from versioned_map.geometry_benchmark_105b import run_geometry_benchmark
            rows=run_geometry_benchmark(a.database,a.geometry,a.output,queries=a.queries,
                  repeats=a.repeats,seed=a.seed,unmerged=a.unmerged_geometry)
            print(json.dumps({'output':a.output,'summary':rows},indent=2));return
    if a.cmd.startswith('aggregate-'):
        from versioned_map.aggregate_4d import Merged4DIndex
        if a.cmd=='aggregate-build':
            db=Path(a.interval_root)/'bst'
            result={}
            for compact in (True,False):
                index=Merged4DIndex(db,coalesced=compact)
                result['coalesced' if compact else 'unmerged']=index.build()
            print(json.dumps(result,indent=2));return
        if a.cmd in ('aggregate-query','aggregate-history'):
            index=Merged4DIndex(a.interval,coalesced=not a.unmerged)
            if a.cmd=='aggregate-query':
                arr,st=index.query(a.session,tuple(a.bounds),stats=True)
                if a.save_npy:np.save(a.save_npy,arr)
                print(json.dumps({'voxels':len(arr),**st},indent=2));return
            records,st=index.history_records(a.from_session,a.to_session,tuple(a.bounds))
            print(json.dumps({'records':records,'stats':st},indent=2));return
        if a.cmd=='aggregate-verify':
            source=VersionedStore(a.database);rng=np.random.default_rng(a.seed)
            idx=Merged4DIndex(Path(a.interval_root)/'bst');idx.load()
            sessions=list(source.all_versions());dims=source.meta()['dims']
            checked=0
            if a.full_maps:
                bounds=tuple(int(x) for d in dims for x in (0,d))
                for sess in sessions:
                    np.testing.assert_array_equal(source.query(sess,bounds),idx.query(sess,bounds))
                    checked+=1
            for _ in range(a.queries):
                sess=sessions[int(rng.integers(len(sessions)))];side=int(rng.choice([16,32,64,128]))
                center=[int(rng.integers(-20,d+20)) for d in dims]
                bounds=tuple(z for x in center for z in (x-side//2,x+side//2))
                np.testing.assert_array_equal(source.query(sess,bounds),idx.query(sess,bounds))
                checked+=1
            print(json.dumps({'verified_regions':checked,'verified_full_maps':len(sessions) if a.full_maps else 0},indent=2));return
        if a.cmd=='aggregate-benchmark':
            from versioned_map.aggregate_benchmark import run_aggregate_benchmark
            out=run_aggregate_benchmark(a.database,a.interval_root,a.output,
                  queries=a.queries,repeats=a.repeats,seed=a.seed)
            print(json.dumps({'output':a.output,'summary':out},indent=2));return
    if a.cmd.startswith('voxel4d-'):
        from versioned_map.voxel_4d import Voxel4D
        if a.cmd=='voxel4d-build':
            _,report=Voxel4D.build(a.database,a.output,tuple(a.bounds),
                        max_atoms=a.max_atoms,merged=not a.unmerged)
            print(json.dumps(report,indent=2));return
        if a.cmd=='voxel4d-query':
            data,stats=Voxel4D(a.index).query(a.session,tuple(a.bounds),stats=True)
            if a.save_npy:np.save(a.save_npy,data)
            print(json.dumps({'voxels':len(data),**stats},indent=2));return
    if a.cmd.startswith('interval-'):
        from versioned_map.interval_store import IntervalMap
        if a.cmd=='interval-query':
            obj=IntervalMap(a.interval)
            vox,metrics=obj.query(a.session,tuple(a.bounds),stats=True)
            if a.save_npy:np.save(a.save_npy,vox)
            print(json.dumps({'voxels':len(vox),**metrics},indent=2));return
        if a.cmd=='interval-history':
            obj=IntervalMap(a.interval)
            rows,metrics=obj.history_records(a.from_session,a.to_session,tuple(a.bounds))
            print(json.dumps({'records':rows,'stats':metrics},indent=2));return
        if a.cmd=='interval-stats':
            print(json.dumps({mode:IntervalMap(Path(a.interval_root)/mode).disk_stats()
                              for mode in ('bst','morton')},indent=2));return
        store=VersionedStore(a.database)
        if a.cmd=='interval-build':
            modes=('bst','morton') if a.address_mode=='both' else (a.address_mode,)
            result={}
            for mode in modes:
                db=IntervalMap.build(store,Path(a.interval_root)/mode,mode,
                                     through_session=a.through_session)
                result[mode]=db.disk_stats()
                if mode=='bst' and a.aggregate_cells:
                    from versioned_map.aggregate_4d import Merged4DIndex
                    result[mode]['aggregate_4d_build']=Merged4DIndex(db).build()
            print(json.dumps(result,indent=2));return
        if a.cmd=='interval-append':
            modes=('bst','morton') if a.address_mode=='both' else (a.address_mode,)
            result={mode:IntervalMap(Path(a.interval_root)/mode).append_from_store(store,a.session)
                    for mode in modes}
            print(json.dumps(result,indent=2));return
        if a.cmd=='interval-verify':
            rng=np.random.default_rng(a.seed);metadata=store.meta();dims=metadata['dims'];sessions=list(store.all_versions())
            result={}
            for mode in ('bst','morton'):
                obj=IntervalMap(Path(a.interval_root)/mode);obj.validate_structure()
                assert obj.meta['sessions']==sessions
                if a.full_maps:
                    bounds=tuple(v for dim in dims for v in (0,dim))
                    for session in sessions:
                        np.testing.assert_array_equal(store.query(session,bounds),obj.query(session,bounds))
                for _ in range(a.queries):
                    session=sessions[int(rng.integers(0,len(sessions)))];side=int(rng.choice([32,64,128]))
                    c=[int(rng.integers(0,d)) for d in dims]
                    b=tuple(v for p in c for v in (p-side//2,p+side//2))
                    np.testing.assert_array_equal(store.query(session,b),obj.query(session,b))
                result[mode]={'random_regions_verified':a.queries,'full_maps_verified':len(sessions) if a.full_maps else 0}
            print(json.dumps(result,indent=2));return
        if a.cmd=='interval-benchmark':
            from versioned_map.interval_benchmark import run_interval_benchmark
            data=run_interval_benchmark(a.database,a.interval_root,a.output,
                  queries=a.queries,repeats=a.repeats,seed=a.seed,verify_full=a.full_maps)
            print(json.dumps({'output':a.output,'summary':data},indent=2));return
    if a.cmd.startswith('delta-'):
        from versioned_map.delta_journal import DeltaJournal, source_storage_controls
        if a.cmd=='delta-query':
            b=tuple(a.bounds)
            arr,metrics=DeltaJournal(a.journal).query(a.session,b,stats=True)
            if a.save_npy:np.save(a.save_npy,arr)
            print(json.dumps({'voxels':int(len(arr)),**metrics},indent=2));return
        store=VersionedStore(a.database)
        if a.cmd=='delta-build':
            modes=('morton','bst') if a.address_mode=='both' else (a.address_mode,)
            result={}
            for mode in modes:
                journal=DeltaJournal.build(store,Path(a.journal_root)/mode,mode,through_session=a.through_session)
                result[mode]={'storage':journal.disk_stats(),'build_ms':journal.meta['build_ms'],
                              'transitions':journal.meta['transitions']}
            result['controls']=source_storage_controls(store)
            print(json.dumps(result,indent=2));return
        if a.cmd=='delta-append':
            modes=('morton','bst') if a.address_mode=='both' else (a.address_mode,)
            result={mode:DeltaJournal(Path(a.journal_root)/mode).append_from_store(store,a.session)
                    for mode in modes}
            print(json.dumps(result,indent=2));return
        if a.cmd=='delta-stats':
            result={'controls':source_storage_controls(store),'journals':{}}
            for mode in ('morton','bst'):
                journal=DeltaJournal(Path(a.journal_root)/mode)
                result['journals'][mode]=journal.disk_stats()
            print(json.dumps(result,indent=2));return
        if a.cmd=='delta-verify':
            from versioned_map.delta_benchmark import verify_against_store
            result={}
            for mode in ('morton','bst'):
                journal=DeltaJournal(Path(a.journal_root)/mode)
                journal.validate_structure()
                result[mode]=verify_against_store(store,journal,random_queries=a.queries,
                                                  seed=a.seed,full=a.full_maps)
            print(json.dumps(result,indent=2));return
        if a.cmd=='delta-benchmark':
            from versioned_map.delta_benchmark import run_delta_benchmark
            print(json.dumps({'output':a.output,'summary':run_delta_benchmark(a.database,a.journal_root,
                 a.output,queries=a.queries,repeats=a.repeats,seed=a.seed)},indent=2));return
    if a.cmd=='build':
        from realdata.nclt import NCLTIngestConfig
        from realdata.nclt_packed import prepare_nclt_packed
        config=NCLTIngestConfig(voxel_size_m=a.voxel_size,scan_stride=a.scan_stride,min_range_m=a.min_range,max_range_m=a.max_range,max_scans_per_session=a.max_scans_per_session)
        prepared=prepare_nclt_packed(a.data_root,a.sessions,config,a.cache_dir)
        store=VersionedStore(a.database);start=time.perf_counter();store.create(prepared,a.codec,a.tile_side)
        print(json.dumps({'codec':a.codec,'build_s':time.perf_counter()-start,'database':store.disk_stats()},indent=2))
        return
    store=VersionedStore(a.database)
    if a.cmd=='stats': print(json.dumps({'metadata':store.meta(),'storage':store.disk_stats()},indent=2));return
    if a.cmd=='index':
        names=a.sessions or list(store.all_versions())
        modes=['bst_prefix','morton_hash'] if a.index_mode=='both' and store.meta()['codec']=='official' else (['morton_hash'] if a.index_mode=='both' else [a.index_mode])
        print(json.dumps([store.build_spatial_index(s,m) for s in names for m in modes],indent=2));return
    if a.cmd=='benchmark':
        from versioned_map.benchmark import run_benchmark
        result=run_benchmark(a.database,a.output,query_count=a.queries,repeats=a.repeats,seed=a.seed,build_indexes=a.build_indexes)
        print(json.dumps({'output':a.output,'summary':result},indent=2));return
    b=tuple(a.bounds)
    if any(b[i]>=b[i+1] for i in (0,2,4)):raise ValueError('bounds are half-open: X0 X1 Y0 Y1 Z0 Z1')
    if a.cmd=='query':
        vox,stats=store.query(a.session,b,stats=True,index_mode=a.index_mode)
        if a.save_npy:np.save(a.save_npy,vox)
        print(json.dumps({'voxels':len(vox),**stats},indent=2))
    elif a.cmd=='changes':
        added,removed=store.changes(a.from_session,a.to_session,b,index_mode=a.index_mode)
        if a.save_prefix:
            np.save(a.save_prefix+'_added.npy',added);np.save(a.save_prefix+'_removed.npy',removed)
        print(json.dumps({'newly_observed':len(added),'no_longer_observed':len(removed),'note':'NOT physical object changes'},indent=2))
    elif a.cmd=='history':
        names=list(store.all_versions());start=names.index(a.from_session);end=names.index(a.to_session)
        if start>end:raise ValueError('history requires chronological from-session <= to-session')
        rows=[]
        for name in names[start:end+1]:
            vox,stats=store.query(name,b,stats=True,index_mode=a.index_mode)
            rows.append({'session':name,'voxels':len(vox),**stats})
        print(json.dumps(rows,indent=2))
    elif a.cmd=='sync':
        packet=store.sync_packet(a.from_session,a.to_session,b,index_mode=a.index_mode)
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(packet))
        print(json.dumps({'packet_file_bytes':Path(a.output).stat().st_size,'tile_entries':len(packet['entries']),'transferred_blobs':len(packet['blobs'])},indent=2))
if __name__=='__main__':main()
