from __future__ import annotations

import csv, gc, json, math, platform, random, statistics, sys, time
from pathlib import Path

from realdata.nclt import NCLTIngestConfig, prepare_nclt_dataset
from representations import (
    SnapshotMap,
    OctreeSharedSignatureMap,
    AdaptiveBlockExplicitMap,
    BSTAdaptiveIntervalMapPrototype,
)
from benchmark.storage import compact_storage_bytes


METHODS=(
    ("snapshot", SnapshotMap),
    ("octree_shared", OctreeSharedSignatureMap),
    ("adaptive_explicit", AdaptiveBlockExplicitMap),
    ("bst_compact_bucket", BSTAdaptiveIntervalMapPrototype),
)


def _pctl(xs,q):
    xs=sorted(xs)
    if not xs: return 0.0
    pos=(len(xs)-1)*q
    lo=math.floor(pos); hi=math.ceil(pos)
    return xs[lo] if lo==hi else xs[lo]*(hi-pos)+xs[hi]*(pos-lo)


def _time(fn,warmups,repetitions):
    for _ in range(warmups): fn()
    vals=[]; enabled=gc.isenabled(); gc.disable()
    try:
        for _ in range(repetitions):
            t0=time.perf_counter_ns(); fn(); t1=time.perf_counter_ns()
            vals.append((t1-t0)/1e6)
    finally:
        if enabled: gc.enable()
    return vals


def _write_csv(path, rows):
    fields=sorted({k for r in rows for k in r})
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def history_complexity(dataset):
    """
    Per-voxel occupied/not-occupied transition complexity over session maps.
    IMPORTANT: 'not occupied' is not interpreted as observed free space.
    """
    all_vox=set().union(*(s.occupied for s in dataset.sessions))
    transitions=[]
    occupied_sessions=[]
    signatures={}
    for v in all_vox:
        bits=tuple(1 if v in s.occupied else 0 for s in dataset.sessions)
        c=sum(bits[i]!=bits[i-1] for i in range(1,len(bits)))
        transitions.append(c)
        occupied_sessions.append(sum(bits))
        signatures[bits]=signatures.get(bits,0)+1

    n=len(all_vox)
    return {
        "ever_occupied_voxels": n,
        "mean_transitions_per_voxel": statistics.mean(transitions) if transitions else 0.0,
        "median_transitions_per_voxel": statistics.median(transitions) if transitions else 0.0,
        "fraction_zero_transitions": sum(x==0 for x in transitions)/n if n else 0.0,
        "fraction_le_2_transitions": sum(x<=2 for x in transitions)/n if n else 0.0,
        "mean_sessions_occupied": statistics.mean(occupied_sessions) if occupied_sessions else 0.0,
        "unique_binary_histories": len(signatures),
        "largest_history_group": max(signatures.values()) if signatures else 0,
        "fraction_in_largest_history_group": max(signatures.values())/n if n else 0.0,
    }


def _region_sizes(spec):
    # Full means the complete prepared NCLT crop.
    return (
        ("5m", min(5.0,spec.size_x_m,spec.size_y_m)),
        ("20m", min(20.0,spec.size_x_m,spec.size_y_m)),
        ("full", None),
    )


def _random_region(spec,size,rng):
    if size is None:
        return (0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m)
    x0=rng.uniform(0,max(0.0,spec.size_x_m-size))
    y0=rng.uniform(0,max(0.0,spec.size_y_m-size))
    return (x0,x0+size,y0,y0+size,0,spec.size_z_m)


def _queries(spec,n_sessions,count,seed):
    rng=random.Random(seed)
    regions=[]
    changes=[]
    for label,size in _region_sizes(spec):
        for _ in range(count):
            reg=_random_region(spec,size,rng)
            t=rng.randrange(n_sessions)
            regions.append((label,reg,t))
        for span_name,span in (
            ("short",1),
            ("medium",max(1,n_sessions//3)),
            ("long",max(1,n_sessions-1)),
        ):
            for _ in range(count):
                reg=_random_region(spec,size,rng)
                if span >= n_sessions:
                    ta,tb=0,n_sessions-1
                else:
                    ta=rng.randrange(0,n_sessions-span)
                    tb=ta+span
                changes.append((label,span_name,reg,ta,tb))
    return regions,changes


def _diag(rep,kind):
    st=rep.last_query_stats if kind=="region" else rep.last_change_stats
    for k in ("blocks_scanned","nodes_visited","records_scanned"):
        if k in st:return k,float(st[k])
    return "unknown",float("nan")


def benchmark_prepared(prepared, output_dir: Path, query_count=20, warmups=2,
                       repetitions=5, query_seed=42008):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    ds=prepared.dataset

    build_rows=[]
    reps={}
    for name,cls in METHODS:
        t0=time.perf_counter_ns(); rep=cls.from_dataset(ds); t1=time.perf_counter_ns()
        reps[name]=rep
        size=compact_storage_bytes(rep)
        build_rows.append({
            "method":name,
            "sessions":len(ds.sessions),
            "build_ms":(t1-t0)/1e6,
            "compact_bytes":size,
            "compact_mib":size/(1024*1024),
            **{f"stat_{k}":v for k,v in rep.stats().items()},
        })

    # Critical block equality control inherited from Step 7.
    assert tuple(reps["adaptive_explicit"].blocks)==tuple(reps["bst_compact_bucket"].blocks)

    rq,cq=_queries(ds.spec,len(ds.sessions),query_count,query_seed)

    # Exact equivalence for every timed query.
    for _,reg,t in rq:
        ref=reps["snapshot"].region_at_time(reg,t)
        for name,rep in reps.items():
            assert rep.region_at_time(reg,t)==ref,(name,"region",reg,t)
    for _,_,reg,ta,tb in cq:
        ref=reps["snapshot"].changes(reg,ta,tb)
        for name,rep in reps.items():
            assert rep.changes(reg,ta,tb)==ref,(name,"change",reg,ta,tb)

    qrows=[]
    for name,rep in reps.items():
        for label,_ in _region_sizes(ds.spec):
            group=[q for q in rq if q[0]==label]
            vals=[];ins=[];dname=""
            cards=[]
            for _,reg,t in group:
                vals += _time(lambda reg=reg,t=t:rep.region_at_time(reg,t),warmups,repetitions)
                dname,dv=_diag(rep,"region");ins.append(dv)
                cards.append(len(rep.region_at_time(reg,t)))
            qrows.append({
                "method":name,"query_kind":"region_at_time","region":label,"span":"",
                "samples":len(vals),"median_ms":statistics.median(vals),
                "p95_ms":_pctl(vals,.95),"mean_ms":statistics.mean(vals),
                "diagnostic_name":dname,"mean_inspected":statistics.mean(ins),
                "mean_result_voxels":statistics.mean(cards),
            })

        for label,_ in _region_sizes(ds.spec):
            for span_name in ("short","medium","long"):
                group=[q for q in cq if q[0]==label and q[1]==span_name]
                vals=[];ins=[];dname="";cards=[]
                for _,_,reg,ta,tb in group:
                    vals += _time(lambda reg=reg,ta=ta,tb=tb:rep.changes(reg,ta,tb),warmups,repetitions)
                    dname,dv=_diag(rep,"change");ins.append(dv)
                    a,b=rep.changes(reg,ta,tb); cards.append(len(a)+len(b))
                qrows.append({
                    "method":name,"query_kind":"changes","region":label,"span":span_name,
                    "samples":len(vals),"median_ms":statistics.median(vals),
                    "p95_ms":_pctl(vals,.95),"mean_ms":statistics.mean(vals),
                    "diagnostic_name":dname,"mean_inspected":statistics.mean(ins),
                    "mean_result_voxels":statistics.mean(cards),
                })

    complexity=history_complexity(ds)
    _write_csv(out/"real_build_storage.csv",build_rows)
    _write_csv(out/"real_queries.csv",qrows)
    _write_csv(out/"session_stats.csv",[vars(x) for x in prepared.session_stats])
    (out/"history_complexity.json").write_text(json.dumps(complexity,indent=2))
    metadata=prepared.metadata()
    metadata.update({
        "query_count_per_group":query_count,
        "warmups":warmups,
        "repetitions":repetitions,
        "query_seed":query_seed,
        "methods":[x[0] for x in METHODS],
        "runtime":{"python":sys.version,"platform":platform.platform(),"machine":platform.machine()},
        "critical_controls":[
            "NCLT reference/ground-truth poses are used; this step does not evaluate SLAM.",
            "velodyne_sync scans are treated as motion-compensated Segway-body-frame scans.",
            "adaptive_explicit and bst_compact_bucket use identical adaptive RectBlocks.",
            "all benchmark queries are exact-equivalence checked against Snapshot before timing.",
            "missing occupancy is not called free space; change results are observation-history differences.",
        ],
    })
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2))
    return build_rows,qrows,complexity
