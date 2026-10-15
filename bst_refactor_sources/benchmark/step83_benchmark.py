from __future__ import annotations

import csv, gc, json, math, platform, random, statistics, sys, time
from pathlib import Path
import numpy as np

from realdata.nclt_packed import PackedNCLTPrepared, linear_decode
from representations.packed_scale import PackedSnapshot, PackedOctreeShared, build_history_table
from representations.step83 import (
    AdaptiveBlockCore, ExplicitAdaptiveMap, PackedVoxelHistoryMap,
    DeltaVersionMap, SnapshotAdapter, OctreeAdapter,
)
from bst_id import BSTCompactMap, PrototypeBSTSpatialCodec


def _pctl(vals,q):
    s=sorted(vals); pos=(len(s)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return s[lo] if lo==hi else s[lo]*(hi-pos)+s[hi]*(pos-lo)


def _time_call(fn,warmups,repetitions):
    for _ in range(warmups): fn()
    vals=[]; enabled=gc.isenabled(); gc.disable()
    try:
        for _ in range(repetitions):
            t0=time.perf_counter_ns(); fn(); vals.append((time.perf_counter_ns()-t0)/1e6)
    finally:
        if enabled: gc.enable()
    return vals


def _write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def history_complexity(masks:np.ndarray,n_sessions:int):
    n=len(masks); transitions=np.zeros(n,dtype=np.uint8); occupied=np.zeros(n,dtype=np.uint8); prev=None
    for t in range(n_sessions):
        cur=((masks>>np.uint64(t))&1).astype(np.uint8); occupied += cur
        if prev is not None: transitions += (cur!=prev)
        prev=cur
    uniq,counts=np.unique(masks,return_counts=True)
    return {
        'ever_occupied_voxels':int(n),
        'mean_transitions_per_voxel':float(transitions.mean()) if n else 0.0,
        'median_transitions_per_voxel':float(np.median(transitions)) if n else 0.0,
        'fraction_zero_transitions':float(np.mean(transitions==0)) if n else 0.0,
        'fraction_le_2_transitions':float(np.mean(transitions<=2)) if n else 0.0,
        'mean_sessions_occupied':float(occupied.mean()) if n else 0.0,
        'unique_binary_histories':int(len(uniq)),
        'largest_history_group':int(counts.max()) if len(counts) else 0,
        'fraction_in_largest_history_group':float(counts.max()/n) if n else 0.0,
    }


def _subset(prepared:PackedNCLTPrepared,n:int)->PackedNCLTPrepared:
    return PackedNCLTPrepared(
        spec=prepared.spec,
        session_names=prepared.session_names[:n],
        session_codes=prepared.session_codes[:n],
        origin_local_m=prepared.origin_local_m,
        session_stats=prepared.session_stats[:n],
        config=prepared.config,
    )


def _centered_region(spec,x,y,size):
    half=size/2; cx=(float(x)+.5)*spec.voxel_size_m; cy=(float(y)+.5)*spec.voxel_size_m
    x0=min(max(0.0,cx-half),max(0.0,spec.size_x_m-size)); y0=min(max(0.0,cy-half),max(0.0,spec.size_y_m-size))
    return (x0,min(spec.size_x_m,x0+size),y0,min(spec.size_y_m,y0+size),0.0,spec.size_z_m)


def _random_region(spec,size,rng):
    x0=rng.uniform(0,max(0,spec.size_x_m-size)); y0=rng.uniform(0,max(0,spec.size_y_m-size))
    return (x0,min(spec.size_x_m,x0+size),y0,min(spec.size_y_m,y0+size),0.0,spec.size_z_m)


def make_queries(prepared,union_codes,count,seed,targeted_fraction=.75,include_full=False):
    rng=random.Random(seed); spec=prepared.spec; n=len(prepared.session_codes)
    ux,uy,_=linear_decode(union_codes,spec)
    regions=[]; changes=[]
    sizes=[('5m',min(5.0,spec.size_x_m,spec.size_y_m)),('20m',min(20.0,spec.size_x_m,spec.size_y_m))]
    if include_full: sizes.append(('full',None))
    change_pool_cache={}
    for label,size in sizes:
        for _ in range(count):
            t=rng.randrange(n)
            if size is None:
                reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m); mode='full'
            elif rng.random()<targeted_fraction and len(union_codes):
                j=rng.randrange(len(union_codes)); reg=_centered_region(spec,ux[j],uy[j],size); mode='occupied-centered'
            else:
                reg=_random_region(spec,size,rng); mode='random'
            regions.append((label,reg,t,mode))

        spans=[('short',1),('medium',max(1,n//3)),('long',max(1,n-1))]
        for span_name,span in spans:
            for _ in range(count):
                ta=0 if span>=n else rng.randrange(0,n-span); tb=min(n-1,ta+span)
                mode='random'
                if size is None:
                    reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m); mode='full'
                elif rng.random()<targeted_fraction:
                    pair=(ta,tb); changed=change_pool_cache.get(pair)
                    if changed is None:
                        changed=np.setxor1d(prepared.session_codes[ta],prepared.session_codes[tb],assume_unique=True)
                        change_pool_cache[pair]=changed
                    pool=changed if len(changed) else union_codes
                    if len(pool):
                        code=pool[rng.randrange(len(pool))]
                        x,y,_=linear_decode(np.asarray([code],dtype=np.uint64),spec)
                        reg=_centered_region(spec,x[0],y[0],size); mode='change-centered' if len(changed) else 'occupied-centered'
                    else: reg=_random_region(spec,size,rng)
                else: reg=_random_region(spec,size,rng)
                changes.append((label,span_name,reg,ta,tb,mode))
    return regions,changes


def _diag(rep,kind):
    st=rep.last_query_stats if kind=='region' else rep.last_change_stats
    for k in ('blocks_scanned','candidate_leaves','delta_records_scanned','records_scanned','nodes_visited'):
        if k in st:return k,float(st[k])
    return 'unknown',float('nan')


def _native_primitives(result):
    if isinstance(result,tuple): return sum(len(x) for x in result)
    return len(result)


def _native_voxel_equivalent(result,native_kind):
    def one(x):
        if native_kind=='blocks' and isinstance(x,np.ndarray) and x.ndim==2 and x.shape[1]>=6:
            return int(np.prod(x[:,3:6].astype(np.int64),axis=1).sum()) if len(x) else 0
        return len(x)
    if isinstance(result,tuple): return sum(one(x) for x in result)
    return one(result)


def _build_methods(prepared,progress=True):
    rows=[]

    # Snapshot and delta do not require the shared union/history precompute.
    snapshot=SnapshotAdapter(PackedSnapshot(prepared))
    t0=time.perf_counter_ns(); delta=DeltaVersionMap(prepared); delta_ms=(time.perf_counter_ns()-t0)/1e6

    if progress: print('[build] common packed voxel-history table',flush=True)
    t0=time.perf_counter_ns(); union,masks=build_history_table(prepared,progress=progress); common_ms=(time.perf_counter_ns()-t0)/1e6

    t0=time.perf_counter_ns(); voxel=PackedVoxelHistoryMap(prepared,union,masks); voxel_ms=(time.perf_counter_ns()-t0)/1e6

    if progress: print('[build] hardened sparse Morton OctreeShared',flush=True)
    t0=time.perf_counter_ns(); oct_impl=PackedOctreeShared(prepared,union,masks,query_mode='sparse'); oct_ms=(time.perf_counter_ns()-t0)/1e6
    octree=OctreeAdapter(oct_impl)

    if progress: print('[build] generic adaptive block core (BST-agnostic)',flush=True)
    t0=time.perf_counter_ns(); core=AdaptiveBlockCore(prepared,union,masks,progress=progress); adaptive_ms=(time.perf_counter_ns()-t0)/1e6
    explicit=ExplicitAdaptiveMap(core)
    bst=BSTCompactMap(core,PrototypeBSTSpatialCodec())

    methods=[snapshot,delta,voxel,octree,explicit,bst]
    method_times={
        snapshot.name:(0.0,0.0),
        delta.name:(delta_ms,0.0),
        voxel.name:(voxel_ms,common_ms),
        octree.name:(oct_ms,common_ms),
        explicit.name:(adaptive_ms,common_ms),
        bst.name:(adaptive_ms,common_ms),
    }
    for rep in methods:
        method_ms,shared_ms=method_times[rep.name]
        cb=rep.compact_bytes()
        rows.append({
            'method':rep.name,'native_kind':rep.native_kind,
            'method_build_ms':method_ms,'shared_history_ms':shared_ms,
            'standalone_total_build_ms':method_ms+shared_ms,
            'compact_bytes':cb,'compact_mib':cb/(1024**2),
            **{f'stat_{k}':v for k,v in rep.stats().items()},
        })
    return methods,union,masks,rows


def _benchmark_mode(methods,rq,cq,output_mode,warmups,repetitions,sessions):
    rows=[]
    # Native mode focuses on representations that can supply a directly usable
    # local geometric map without applying a version recipe. Snapshot is kept
    # as a fast reference; DeltaVersion and VoxelHistory remain in canonical
    # voxel comparisons but are omitted from native timing to avoid duplicating
    # the same materialized output under a misleading label.
    if output_mode=='native':
        allow={'snapshot','octree_shared_sparse','adaptive_explicit','bst_compact_bucket'}
        methods=[m for m in methods if m.name in allow]

    for rep in methods:
        for label in sorted({q[0] for q in rq}):
            group=[q for q in rq if q[0]==label]; vals=[];ins=[];cards=[];voxeq=[];modes=[];dname=''
            for _,reg,t,mode in group:
                vals += _time_call(lambda reg=reg,t=t:rep.region_query(reg,t,output=output_mode),warmups,repetitions)
                dname,dv=_diag(rep,'region');ins.append(dv)
                res=rep.region_query(reg,t,output=output_mode)
                cards.append(_native_primitives(res)); voxeq.append(_native_voxel_equivalent(res,rep.native_kind)); modes.append(mode)
            rows.append({'sessions':sessions,'method':rep.name,'native_kind':rep.native_kind,'output_mode':output_mode,'equal_output':output_mode=='voxels','query_kind':'region_at_time','region':label,'span':'','samples':len(vals),'median_ms':statistics.median(vals),'p95_ms':_pctl(vals,.95),'mean_ms':statistics.mean(vals),'diagnostic_name':dname,'mean_inspected':statistics.mean(ins),'mean_result_primitives':statistics.mean(cards),'mean_voxel_equivalent':statistics.mean(voxeq),'targeted_fraction_observed':sum(m!='random' for m in modes)/len(modes)})
        for label in sorted({q[0] for q in cq}):
            for span in ('short','medium','long'):
                group=[q for q in cq if q[0]==label and q[1]==span]
                if not group:continue
                vals=[];ins=[];cards=[];voxeq=[];modes=[];dname=''
                for _,_,reg,ta,tb,mode in group:
                    vals += _time_call(lambda reg=reg,ta=ta,tb=tb:rep.changes_query(reg,ta,tb,output=output_mode),warmups,repetitions)
                    dname,dv=_diag(rep,'change');ins.append(dv)
                    res=rep.changes_query(reg,ta,tb,output=output_mode)
                    cards.append(_native_primitives(res)); voxeq.append(_native_voxel_equivalent(res,rep.native_kind)); modes.append(mode)
                rows.append({'sessions':sessions,'method':rep.name,'native_kind':rep.native_kind,'output_mode':output_mode,'equal_output':output_mode=='voxels','query_kind':'changes','region':label,'span':span,'samples':len(vals),'median_ms':statistics.median(vals),'p95_ms':_pctl(vals,.95),'mean_ms':statistics.mean(vals),'diagnostic_name':dname,'mean_inspected':statistics.mean(ins),'mean_result_primitives':statistics.mean(cards),'mean_voxel_equivalent':statistics.mean(voxeq),'targeted_fraction_observed':sum(m!='random' for m in modes)/len(modes)})
    return rows


def benchmark_packed_step83(prepared_full:PackedNCLTPrepared,output_dir:Path,scales=(2,3,5),query_count=20,warmups=2,repetitions=5,query_seed=42083,targeted_fraction=.75,include_full=False,output_modes=('voxels','native'),progress=True):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    scales=tuple(sorted(set(int(n) for n in scales if int(n)>=2)))
    if not scales: raise ValueError('At least one scale >=2 is required')
    if max(scales)>len(prepared_full.session_codes): raise ValueError(f'scale {max(scales)} requested but only {len(prepared_full.session_codes)} sessions were prepared')
    if max(scales)>64: raise ValueError('Current packed uint64 temporal masks support at most 64 sessions')

    all_build=[]; all_queries=[]; all_complexity=[]
    for n in scales:
        if progress: print(f'\n========== Step 8.3 N={n} sessions ==========',flush=True)
        prepared=_subset(prepared_full,n)
        methods,union,masks,build=_build_methods(prepared,progress)
        for r in build:r['sessions']=n
        rq,cq=make_queries(prepared,union,query_count,query_seed+n*1009,targeted_fraction,include_full)

        # Canonical strict-equality correctness gate for every generated query.
        ref=next(m for m in methods if m.name=='snapshot')
        if progress: print(f'[verify N={n}] exact voxel outputs: {len(rq)} history + {len(cq)} change queries',flush=True)
        for qi,(_,reg,t,_) in enumerate(rq,1):
            expected=ref.region_query(reg,t,output='voxels')
            for rep in methods[1:]:
                got=rep.region_query(reg,t,output='voxels')
                if not np.array_equal(got,expected): raise AssertionError((n,rep.name,'region',qi,len(got),len(expected)))
        for qi,(_,_,reg,ta,tb,_) in enumerate(cq,1):
            ea,er=ref.changes_query(reg,ta,tb,output='voxels')
            for rep in methods[1:]:
                ga,gr=rep.changes_query(reg,ta,tb,output='voxels')
                if not np.array_equal(ga,ea) or not np.array_equal(gr,er): raise AssertionError((n,rep.name,'changes',qi,len(ga),len(ea),len(gr),len(er)))

        # Native adaptive output must cover exactly the same number of voxels as
        # canonical output, even though it returns fewer primitives.
        for rep in (m for m in methods if m.native_kind=='blocks'):
            for _,reg,t,_ in rq:
                native=rep.region_query(reg,t,output='native'); vox=rep.region_query(reg,t,output='voxels')
                if _native_voxel_equivalent(native,'blocks')!=len(vox): raise AssertionError((n,rep.name,'native coverage'))

        if progress: print(f'[benchmark N={n}] modes={",".join(output_modes)}',flush=True)
        for mode in output_modes:
            all_queries += _benchmark_mode(methods,rq,cq,mode,warmups,repetitions,n)
        all_build += build
        comp=history_complexity(masks,n); comp['sessions']=n; all_complexity.append(comp)

    _write_csv(out/'scaling_build_storage.csv',all_build)
    _write_csv(out/'scaling_queries.csv',all_queries)
    _write_csv(out/'history_complexity_scaling.csv',all_complexity)
    _write_csv(out/'session_stats.csv',[vars(x) for x in prepared_full.session_stats[:max(scales)]])

    metadata=prepared_full.metadata(); metadata.update({
        'step':'8.3','scales':list(scales),'query_count_per_group':query_count,'warmups':warmups,'repetitions':repetitions,
        'query_seed_base':query_seed,'targeted_fraction':targeted_fraction,'include_full_materialization':include_full,
        'output_modes':list(output_modes),
        'methods':['snapshot','delta_version','voxel_history','octree_shared_sparse','adaptive_explicit','bst_compact_bucket'],
        'native_timing_methods':['snapshot','octree_shared_sparse','adaptive_explicit','bst_compact_bucket'],
        'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'architecture_notes':[
            'AdaptiveBlockCore is BST-agnostic and owns block construction, temporal signatures, bucket indexing and query execution.',
            'BST-specific spatial encoding is isolated in bst_id/codec.py behind SpatialBlockCodec.',
            'BSTCompactMap is a thin wrapper over AdaptiveBlockCore; replacing the BST codec does not require rebuilding the adaptive algorithm.',
            'AdaptiveExplicitMap and BSTCompactMap share the exact same AdaptiveBlockCore and therefore identical blocks/query behavior.',
            'DeltaVersionMap uses latest-session base plus reverse positive/negative voxel deltas.',
            'PackedVoxelHistoryMap stores one packed voxel key plus a shared temporal-signature id per ever-observed voxel.',
            'output=voxels is the equal-output benchmark; output=native intentionally measures representation-native local-map retrieval and has unequal primitive types.',
            'Scaling prefixes use one common prepared NCLT coordinate grid to isolate temporal-history growth.',
        ],
        'scientific_caveat':'PrototypeBSTSpatialCodec remains a local dyadic-prefix research prototype, not verified upstream BST-ID. Future replacement is localized to bst_id/codec.py.',
    })
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return all_build,all_queries,all_complexity
