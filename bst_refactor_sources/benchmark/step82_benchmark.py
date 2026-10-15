from __future__ import annotations

import csv, gc, json, math, platform, random, statistics, sys, time
from pathlib import Path
import numpy as np

from realdata.nclt_packed import PackedNCLTPrepared, linear_decode
from representations.packed_scale import (
    PackedSnapshot, PackedOctreeShared, PackedAdaptiveBucket, build_history_table,
)


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
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def history_complexity(masks:np.ndarray,n_sessions:int):
    n=len(masks)
    transitions=np.zeros(n,dtype=np.uint8); occupied=np.zeros(n,dtype=np.uint8)
    prev=None
    for t in range(n_sessions):
        cur=((masks>>np.uint64(t))&1).astype(np.uint8)
        occupied += cur
        if prev is not None: transitions += (cur!=prev)
        prev=cur
    uniq,counts=np.unique(masks,return_counts=True)
    return {
        "ever_occupied_voxels":int(n),
        "mean_transitions_per_voxel":float(transitions.mean()) if n else 0.0,
        "median_transitions_per_voxel":float(np.median(transitions)) if n else 0.0,
        "fraction_zero_transitions":float(np.mean(transitions==0)) if n else 0.0,
        "fraction_le_2_transitions":float(np.mean(transitions<=2)) if n else 0.0,
        "mean_sessions_occupied":float(occupied.mean()) if n else 0.0,
        "unique_binary_histories":int(len(uniq)),
        "largest_history_group":int(counts.max()) if len(counts) else 0,
        "fraction_in_largest_history_group":float(counts.max()/n) if n else 0.0,
    }


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
    sizes=[("5m",min(5.0,spec.size_x_m,spec.size_y_m)),("20m",min(20.0,spec.size_x_m,spec.size_y_m))]
    if include_full: sizes.append(("full",None))
    change_pool_cache={}
    for label,size in sizes:
        for _ in range(count):
            t=rng.randrange(n)
            if size is None:
                reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m); mode="full"
            elif rng.random()<targeted_fraction and len(union_codes):
                j=rng.randrange(len(union_codes)); reg=_centered_region(spec,ux[j],uy[j],size); mode="occupied-centered"
            else:
                reg=_random_region(spec,size,rng); mode="random"
            regions.append((label,reg,t,mode))

        spans=[("short",1),("medium",max(1,n//3)),("long",max(1,n-1))]
        for span_name,span in spans:
            for _ in range(count):
                ta=0 if span>=n else rng.randrange(0,n-span); tb=min(n-1,ta+span)
                mode="random"
                if size is None:
                    reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m); mode="full"
                elif rng.random()<targeted_fraction:
                    pair=(ta,tb)
                    changed=change_pool_cache.get(pair)
                    if changed is None:
                        changed=np.setxor1d(prepared.session_codes[ta],prepared.session_codes[tb],assume_unique=True)
                        change_pool_cache[pair]=changed
                    pool=changed if len(changed) else union_codes
                    if len(pool):
                        code=pool[rng.randrange(len(pool))]; x,y,_=linear_decode(np.asarray([code],dtype=np.uint64),spec); reg=_centered_region(spec,x[0],y[0],size); mode="change-centered" if len(changed) else "occupied-centered"
                    else: reg=_random_region(spec,size,rng)
                else:
                    reg=_random_region(spec,size,rng)
                changes.append((label,span_name,reg,ta,tb,mode))
    return regions,changes


def _diag(rep,kind):
    st=rep.last_query_stats if kind=="region" else rep.last_change_stats
    for k in ("blocks_scanned","candidate_leaves","records_scanned","nodes_visited"):
        if k in st:return k,float(st[k])
    return "unknown",float("nan")


def benchmark_packed_step82(prepared:PackedNCLTPrepared,output_dir:Path,query_count=20,warmups=2,repetitions=5,query_seed=42081,targeted_fraction=.75,include_full=False,progress=True):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    build=[]

    if progress: print("[build] common history table",flush=True)
    t0=time.perf_counter_ns(); union,masks=build_history_table(prepared,progress=progress); common_ms=(time.perf_counter_ns()-t0)/1e6

    if progress: print("[build] packed Snapshot",flush=True)
    snapshot=PackedSnapshot(prepared)

    if progress: print("[build] sparse-traversal Morton OctreeShared",flush=True)
    t0=time.perf_counter_ns(); octree=PackedOctreeShared(prepared,union,masks,query_mode="sparse"); oct_ms=(time.perf_counter_ns()-t0)/1e6

    if progress: print("[build] adaptive dyadic blocks",flush=True)
    t0=time.perf_counter_ns(); adaptive=PackedAdaptiveBucket(prepared,union,masks,progress=progress); ada_ms=(time.perf_counter_ns()-t0)/1e6

    build += [
        {"method":"snapshot","build_ms_method":0.0,"common_history_ms":common_ms,"compact_bytes":snapshot.compact_bytes(),"compact_mib":snapshot.compact_bytes()/(1024**2),"internal_note":"packed sessions; no extra index"},
        {"method":"octree_shared_sparse","build_ms_method":oct_ms,"common_history_ms":common_ms,"compact_bytes":octree.compact_bytes(),"compact_mib":octree.compact_bytes()/(1024**2),**{f"stat_{k}":v for k,v in octree.stats().items()}},
        {"method":"adaptive_explicit","build_ms_method":ada_ms,"common_history_ms":common_ms,"compact_bytes":adaptive.explicit_bytes(),"compact_mib":adaptive.explicit_bytes()/(1024**2),**{f"stat_{k}":v for k,v in adaptive.stats().items()}},
        {"method":"bst_compact_bucket","build_ms_method":ada_ms,"common_history_ms":common_ms,"compact_bytes":adaptive.bst_compact_bytes(),"compact_mib":adaptive.bst_compact_bytes()/(1024**2),**{f"stat_{k}":v for k,v in adaptive.stats().items()}},
    ]

    rq,cq=make_queries(prepared,union,query_count,query_seed,targeted_fraction,include_full)
    methods=(("snapshot",snapshot),("octree_shared_sparse",octree),("adaptive_explicit",adaptive),("bst_compact_bucket",adaptive))

    if progress: print(f"[verify] exact local outputs: {len(rq)} history + {len(cq)} change queries",flush=True)
    for qi,(_,reg,t,_) in enumerate(rq,1):
        ref=snapshot.region_codes(reg,t)
        for name,rep in methods[1:]:
            got=rep.region_codes(reg,t)
            if not np.array_equal(got,ref): raise AssertionError((name,"region",qi,len(got),len(ref)))
    for qi,(_,_,reg,ta,tb,_) in enumerate(cq,1):
        ra,rr=snapshot.changes_codes(reg,ta,tb)
        for name,rep in methods[1:]:
            ga,gr=rep.changes_codes(reg,ta,tb)
            if not np.array_equal(ga,ra) or not np.array_equal(gr,rr): raise AssertionError((name,"changes",qi,len(ga),len(ra),len(gr),len(rr)))

    qrows=[]
    if progress: print("[benchmark] timed queries",flush=True)
    for name,rep in methods:
        for label in sorted({q[0] for q in rq}):
            group=[q for q in rq if q[0]==label]; vals=[];ins=[];cards=[];modes=[];dname=""
            for _,reg,t,mode in group:
                vals += _time_call(lambda reg=reg,t=t:rep.region_codes(reg,t),warmups,repetitions)
                dname,dv=_diag(rep,"region");ins.append(dv);cards.append(len(rep.region_codes(reg,t)));modes.append(mode)
            qrows.append({"method":name,"query_kind":"region_at_time","region":label,"span":"","samples":len(vals),"median_ms":statistics.median(vals),"p95_ms":_pctl(vals,.95),"mean_ms":statistics.mean(vals),"diagnostic_name":dname,"mean_inspected":statistics.mean(ins),"mean_result_voxels":statistics.mean(cards),"targeted_fraction_observed":sum(m!="random" for m in modes)/len(modes)})
        for label in sorted({q[0] for q in cq}):
            for span in ("short","medium","long"):
                group=[q for q in cq if q[0]==label and q[1]==span]
                if not group: continue
                vals=[];ins=[];cards=[];modes=[];dname=""
                for _,_,reg,ta,tb,mode in group:
                    vals += _time_call(lambda reg=reg,ta=ta,tb=tb:rep.changes_codes(reg,ta,tb),warmups,repetitions)
                    dname,dv=_diag(rep,"change");ins.append(dv);a,b=rep.changes_codes(reg,ta,tb);cards.append(len(a)+len(b));modes.append(mode)
                qrows.append({"method":name,"query_kind":"changes","region":label,"span":span,"samples":len(vals),"median_ms":statistics.median(vals),"p95_ms":_pctl(vals,.95),"mean_ms":statistics.mean(vals),"diagnostic_name":dname,"mean_inspected":statistics.mean(ins),"mean_result_voxels":statistics.mean(cards),"targeted_fraction_observed":sum(m!="random" for m in modes)/len(modes)})

    complexity=history_complexity(masks,len(prepared.session_codes))
    _write_csv(out/"real_build_storage.csv",build); _write_csv(out/"real_queries.csv",qrows); _write_csv(out/"session_stats.csv",[vars(x) for x in prepared.session_stats])
    (out/"history_complexity.json").write_text(json.dumps(complexity,indent=2))
    metadata=prepared.metadata(); metadata.update({
        "step":"8.2","query_count_per_group":query_count,"warmups":warmups,"repetitions":repetitions,"query_seed":query_seed,"targeted_fraction":targeted_fraction,"include_full_materialization":include_full,
        "methods":[x[0] for x in methods],"runtime":{"python":sys.version,"platform":platform.platform(),"machine":platform.machine()},
        "optimization_notes":[
            "Packed uint64 voxel arrays replace Python tuple/frozenset maps.",
            "Octree runtime traversal is sparse-node-aware: each existing Morton leaf slice is partitioned only into non-empty child octants.",
            "No persistent auxiliary Octree query index was added; logical Octree storage accounting is unchanged from Step 8.1.",
            "NCLT session preprocessing is disk-cached by ingest configuration.",
            "OctreeShared uses the same flat Morton leaf/history representation and logical node accounting as Step 8.1, with hardened sparse-only traversal.",
            "Adaptive merging is exact Step-5/6 independent dyadic x/y/z merging implemented with vectorized packed-key sibling matching.",
            "Full-map materialization is disabled by default because it primarily benchmarks allocation of millions of returned voxels; use --include-full explicitly.",
            "All generated local benchmark queries are exact-array-equivalence checked before timing.",
            "Metric query boxes use correct half-open [min,max) voxel conversion.",
        ],
        "scientific_caveat":"BST compact block encoding remains the local research prototype from Step 7, not verified upstream BST-ID.",
    })
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    return build,qrows,complexity
