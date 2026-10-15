from __future__ import annotations

import csv, gc, json, math, platform, random, statistics, sys, time
from dataclasses import asdict
from pathlib import Path

from synthetic.scenarios import AdversarialScenarioGenerator, ScenarioConfig
from representations import (
    SnapshotMap, OctreeSharedSignatureMap,
    AdaptiveBlockExplicitMap, BSTAdaptiveIntervalMapPrototype,
    BSTPrefixIndexedAdaptiveMapPrototype,
)
from benchmark.storage import compact_storage_bytes


METHODS=(
    ("snapshot",SnapshotMap),
    ("octree_shared",OctreeSharedSignatureMap),
    ("adaptive_explicit",AdaptiveBlockExplicitMap),
    ("bst_compact_bucket",BSTAdaptiveIntervalMapPrototype),
    ("bst_prefix_index",BSTPrefixIndexedAdaptiveMapPrototype),
)

DEFAULT_SCENARIOS=(
    ScenarioConfig("baseline",target_static_fraction=.8,change_probability=.15,coherence="mixed"),
    ScenarioConfig("static95",target_static_fraction=.95,change_probability=.15,coherence="mixed"),
    ScenarioConfig("static50",target_static_fraction=.50,change_probability=.15,coherence="mixed"),
    ScenarioConfig("change_high",target_static_fraction=.8,change_probability=.60,coherence="mixed"),
    ScenarioConfig("fragmented",target_static_fraction=.8,change_probability=.15,coherence="fragmented"),
    ScenarioConfig("voxel_025m",target_static_fraction=.8,change_probability=.15,coherence="mixed",voxel_size_m=.25),
)


def percentile(vals,q):
    s=sorted(vals);pos=(len(s)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return s[lo] if lo==hi else s[lo]*(hi-pos)+s[hi]*(pos-lo)


def time_call(fn,warmups,repetitions):
    for _ in range(warmups): fn()
    vals=[];was=gc.isenabled();gc.disable()
    try:
        for _ in range(repetitions):
            t0=time.perf_counter_ns();fn();t1=time.perf_counter_ns()
            vals.append((t1-t0)/1e6)
    finally:
        if was:gc.enable()
    return vals


def region_sizes(spec):
    return (("5m",min(5.0,spec.size_x_m)),("20m",min(20.0,spec.size_x_m)),("full",spec.size_x_m))


def make_region_queries(spec,n,count,seed):
    rng=random.Random(seed+7001+n+round(spec.voxel_size_m*1000));out=[]
    for label,size in region_sizes(spec):
        for _ in range(count):
            if label=="full":
                reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m)
            else:
                x0=rng.uniform(0,max(0,spec.size_x_m-size))
                y0=rng.uniform(0,max(0,spec.size_y_m-size))
                reg=(x0,x0+size,y0,y0+size,0,spec.size_z_m)
            t=rng.randrange(n)
            out.append((label,reg,t))
    return out


def make_change_queries(spec,n,count,seed):
    rng=random.Random(seed+9001+n+round(spec.voxel_size_m*1000));out=[]
    spans=(("short",1),("medium",max(1,n//10)),("long",max(1,n//2)))
    for label,size in region_sizes(spec):
        for sl,span in spans:
            for _ in range(count):
                if label=="full":
                    reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m)
                else:
                    x0=rng.uniform(0,max(0,spec.size_x_m-size))
                    y0=rng.uniform(0,max(0,spec.size_y_m-size))
                    reg=(x0,x0+size,y0,y0+size,0,spec.size_z_m)
                latest=max(0,n-1-span)
                ta=rng.randrange(latest+1) if latest else 0
                tb=min(n-1,ta+span)
                out.append((label,sl,reg,ta,tb))
    return out


def diagnostic(rep,kind):
    st=rep.last_query_stats if kind=="region" else rep.last_change_stats
    for k in ("blocks_scanned","nodes_visited","records_scanned"):
        if k in st:return k,float(st[k])
    return "unknown",float("nan")


def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def benchmark_one(config,n,seed,query_count,warmups,repetitions):
    ds,metrics=AdversarialScenarioGenerator(config,seed).generate(n)
    scenario={f"scenario_{k}":v for k,v in asdict(config).items()}
    observed={f"observed_{k}":v for k,v in asdict(metrics).items()}
    reps={};build=[]

    for name,cls in METHODS:
        t0=time.perf_counter_ns();rep=cls.from_dataset(ds);t1=time.perf_counter_ns()
        reps[name]=rep
        size=compact_storage_bytes(rep)
        build.append({
            **scenario,**observed,"seed":seed,"sessions":n,"method":name,
            "build_ms":(t1-t0)/1e6,"compact_bytes":size,"compact_mib":size/(1024*1024),
            **{f"stat_{k}":v for k,v in rep.stats().items()},
        })

    # Critical Step-7 control: all three adaptive methods MUST contain exactly
    # the same spatial blocks.
    a=tuple(reps["adaptive_explicit"].blocks)
    b=tuple(reps["bst_compact_bucket"].blocks)
    c=tuple(reps["bst_prefix_index"].blocks)
    assert a==b==c, (config.name,seed,"adaptive block sets differ")

    rq=make_region_queries(ds.spec,n,query_count,seed)
    cq=make_change_queries(ds.spec,n,query_count,seed)

    # Exact-output gate for every query.
    for _,reg,t in rq:
        ref=reps["snapshot"].region_at_time(reg,t)
        for name,rep in reps.items():
            assert rep.region_at_time(reg,t)==ref,(config.name,seed,name,"region",reg,t)
    for _,_,reg,ta,tb in cq:
        ref=reps["snapshot"].changes(reg,ta,tb)
        for name,rep in reps.items():
            assert rep.changes(reg,ta,tb)==ref,(config.name,seed,name,"changes",reg,ta,tb)

    rows=[]
    for method,rep in reps.items():
        for label,_ in region_sizes(ds.spec):
            group=[q for q in rq if q[0]==label];vals=[];di=[];dname=""
            for _,reg,t in group:
                vals += time_call(lambda reg=reg,t=t:rep.region_at_time(reg,t),warmups,repetitions)
                dname,dv=diagnostic(rep,"region");di.append(dv)
            rows.append({
                **scenario,**observed,"seed":seed,"sessions":n,"method":method,
                "query_kind":"region_at_time","region":label,"span":"",
                "samples":len(vals),"median_ms":statistics.median(vals),
                "p95_ms":percentile(vals,.95),"mean_ms":statistics.mean(vals),
                "diagnostic_name":dname,"mean_inspected":statistics.mean(di),
            })

        for rl,_ in region_sizes(ds.spec):
            for sl in ("short","medium","long"):
                group=[q for q in cq if q[0]==rl and q[1]==sl];vals=[];di=[];dname=""
                for _,_,reg,ta,tb in group:
                    vals += time_call(lambda reg=reg,ta=ta,tb=tb:rep.changes(reg,ta,tb),warmups,repetitions)
                    dname,dv=diagnostic(rep,"change");di.append(dv)
                rows.append({
                    **scenario,**observed,"seed":seed,"sessions":n,"method":method,
                    "query_kind":"changes","region":rl,"span":sl,
                    "samples":len(vals),"median_ms":statistics.median(vals),
                    "p95_ms":percentile(vals,.95),"mean_ms":statistics.mean(vals),
                    "diagnostic_name":dname,"mean_inspected":statistics.mean(di),
                })
    return build,rows


def make_summary(build,queries):
    # Aggregate per scenario/method across seeds so seed replicates are not
    # incorrectly counted as independent scenario types.
    rows=[]
    scenarios=sorted({r["scenario_name"] for r in build})
    for sc in scenarios:
        for method in ("octree_shared","adaptive_explicit","bst_compact_bucket","bst_prefix_index"):
            br=[r for r in build if r["scenario_name"]==sc and r["method"]==method]
            if not br:continue
            rows.append({
                "scenario":sc,"method":method,"metric":"compact_mib",
                "median":statistics.median(float(r["compact_mib"]) for r in br),
                "min":min(float(r["compact_mib"]) for r in br),
                "max":max(float(r["compact_mib"]) for r in br),
            })
        for region in ("5m","20m","full"):
            for method in ("octree_shared","adaptive_explicit","bst_compact_bucket","bst_prefix_index"):
                qr=[r for r in queries if r["scenario_name"]==sc and r["method"]==method
                    and r["query_kind"]=="region_at_time" and r["region"]==region]
                if qr:
                    rows.append({
                        "scenario":sc,"method":method,"metric":f"region_{region}_median_ms",
                        "median":statistics.median(float(r["median_ms"]) for r in qr),
                        "min":min(float(r["median_ms"]) for r in qr),
                        "max":max(float(r["median_ms"]) for r in qr),
                    })
    return rows


def run_benchmark(output_dir:Path,scenarios=DEFAULT_SCENARIOS,seeds=(1,7,21,42,84),
                  sessions=250,query_count=10,warmups=1,repetitions=3):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    build=[];queries=[]
    total=len(scenarios)*len(seeds);i=0
    for cfg in scenarios:
        for seed in seeds:
            i+=1
            print(f"[Step 7] {i}/{total} {cfg.name} seed={seed}",flush=True)
            b,q=benchmark_one(cfg,sessions,seed,query_count,warmups,repetitions)
            build+=b;queries+=q

    summary=make_summary(build,queries)
    write_csv(output_dir/"isolation_build.csv",build)
    write_csv(output_dir/"isolation_queries.csv",queries)
    write_csv(output_dir/"isolation_summary.csv",summary)

    meta={
        "sessions":sessions,"seeds":list(seeds),"query_count_per_group":query_count,
        "warmups":warmups,"repetitions":repetitions,
        "methods":[x[0] for x in METHODS],
        "scenarios":[asdict(x) for x in scenarios],
        "critical_controls":[
            "adaptive_explicit, bst_compact_bucket and bst_prefix_index use exactly identical RectBlocks",
            "adaptive_explicit and bst_compact_bucket use the same coarse bucket query index",
            "bst_prefix_index changes only the spatial candidate index to hierarchical dyadic prefix postings",
            "all queries are exact-equivalence checked against snapshot before timing",
        ],
        "runtime":{"python":sys.version,"platform":platform.platform(),"machine":platform.machine()},
        "scientific_caveat":"bst_prefix_index is a local hierarchical-prefix prototype, not verified upstream BST-ID.",
    }
    (output_dir/"metadata.json").write_text(json.dumps(meta,indent=2))
    return build,queries,summary
