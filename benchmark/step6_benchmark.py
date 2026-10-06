from __future__ import annotations

import csv
import gc
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from synthetic.scenarios import (
    AdversarialScenarioGenerator,
    ScenarioConfig,
    default_scenarios,
    quick_scenarios,
)
from representations import (
    SnapshotMap,
    OctreeTemporalMap,
    OctreeIntervalMap,
    OctreeSharedSignatureMap,
    BSTIntervalMapPrototype,
    BSTAdaptiveIntervalMapPrototype,
)
from benchmark.storage import compact_storage_bytes


METHODS=(
    ("snapshot",SnapshotMap),
    ("octree_time",OctreeTemporalMap),
    ("octree_interval",OctreeIntervalMap),
    ("octree_shared",OctreeSharedSignatureMap),
    ("bst_interval",BSTIntervalMapPrototype),
    ("bst_adaptive",BSTAdaptiveIntervalMapPrototype),
)


@dataclass(frozen=True)
class RegionQuery:
    region_label:str
    region:tuple[float,float,float,float,float,float]
    t:int


@dataclass(frozen=True)
class ChangeQuery:
    region_label:str
    region:tuple[float,float,float,float,float,float]
    span_label:str
    t_a:int
    t_b:int


def percentile(vals,q):
    s=sorted(vals)
    pos=(len(s)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return s[lo] if lo==hi else s[lo]*(hi-pos)+s[hi]*(pos-lo)


def _region_sizes(spec):
    # Keep 5m/20m when possible; shrink gracefully for the 50m world.
    return [("5m",min(5.0,spec.size_x_m)),("20m",min(20.0,spec.size_x_m)),("full",spec.size_x_m)]


def make_region_queries(spec,n,count,seed):
    rng=random.Random(seed+101*n+round(spec.size_x_m*10)+round(spec.voxel_size_m*1000));out=[]
    for label,size in _region_sizes(spec):
        for _ in range(count):
            if label=="full":reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m)
            else:
                x0=rng.uniform(0,max(0,spec.size_x_m-size));y0=rng.uniform(0,max(0,spec.size_y_m-size))
                reg=(x0,x0+size,y0,y0+size,0,spec.size_z_m)
            bucket=rng.choice(("old","middle","recent"))
            if bucket=="old":t=rng.randrange(0,max(1,n//3))
            elif bucket=="middle":
                lo=n//3;hi=max(lo+1,2*n//3);t=rng.randrange(lo,hi)
            else:t=rng.randrange(2*n//3,n)
            out.append(RegionQuery(label,reg,t))
    return out


def make_change_queries(spec,n,count,seed):
    rng=random.Random(seed+202*n+round(spec.size_x_m*10)+round(spec.voxel_size_m*1000));out=[]
    spans=(("short",1),("medium",max(1,n//10)),("long",max(1,n//2)))
    for rl,size in _region_sizes(spec):
        for sl,span in spans:
            for _ in range(count):
                if rl=="full":reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m)
                else:
                    x0=rng.uniform(0,max(0,spec.size_x_m-size));y0=rng.uniform(0,max(0,spec.size_y_m-size))
                    reg=(x0,x0+size,y0,y0+size,0,spec.size_z_m)
                latest=max(0,n-1-span);ta=rng.randrange(0,latest+1) if latest else 0;tb=min(n-1,ta+span)
                out.append(ChangeQuery(rl,reg,sl,ta,tb))
    return out


def time_call(fn,warmups,repetitions):
    for _ in range(warmups):fn()
    vals=[];was=gc.isenabled();gc.disable()
    try:
        for _ in range(repetitions):
            t0=time.perf_counter_ns();fn();t1=time.perf_counter_ns();vals.append((t1-t0)/1e6)
    finally:
        if was:gc.enable()
    return vals


def diagnostic(rep,kind):
    st=rep.last_query_stats if kind=="region" else rep.last_change_stats
    for k in ("records_scanned","nodes_visited","events_scanned","blocks_scanned","leaves_tested"):
        if k in st:return k,float(st[k])
    return "unknown",float("nan")


def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def benchmark_scenario(config:ScenarioConfig,n,seed,query_count,warmups,repetitions):
    ds,scenario_metrics=AdversarialScenarioGenerator(config,seed).generate(n)
    reps={};build=[]
    scenario_fields={f"scenario_{k}":v for k,v in asdict(config).items()}
    metric_fields={f"observed_{k}":v for k,v in asdict(scenario_metrics).items()}

    for name,cls in METHODS:
        t0=time.perf_counter_ns();rep=cls.from_dataset(ds);t1=time.perf_counter_ns();reps[name]=rep
        size=compact_storage_bytes(rep)
        build.append({
            **scenario_fields,**metric_fields,"sessions":n,"method":name,
            "build_ms":(t1-t0)/1e6,"compact_bytes":size,"compact_mib":size/(1024*1024),
            **{f"stat_{k}":v for k,v in rep.stats().items()},
        })

    rq=make_region_queries(ds.spec,n,query_count,seed);cq=make_change_queries(ds.spec,n,query_count,seed)

    # Reviewer-grade correctness: every method must match the snapshot for all
    # generated benchmark queries, not merely one representative query.
    for q in rq:
        ref=reps["snapshot"].region_at_time(q.region,q.t)
        for name,rep in reps.items():
            assert rep.region_at_time(q.region,q.t)==ref,(config.name,name,"region",q)
    for q in cq:
        ref=reps["snapshot"].changes(q.region,q.t_a,q.t_b)
        for name,rep in reps.items():
            assert rep.changes(q.region,q.t_a,q.t_b)==ref,(config.name,name,"change",q)

    rows=[]
    for method,rep in reps.items():
        for label,_ in _region_sizes(ds.spec):
            group=[q for q in rq if q.region_label==label];ts=[];di=[];dname=""
            for q in group:
                ts+=time_call(lambda q=q:rep.region_at_time(q.region,q.t),warmups,repetitions)
                dname,dv=diagnostic(rep,"region");di.append(dv)
            rows.append({
                **scenario_fields,**metric_fields,"sessions":n,"method":method,
                "query_kind":"region_at_time","region":label,"span":"","samples":len(ts),
                "median_ms":statistics.median(ts),"p95_ms":percentile(ts,.95),"mean_ms":statistics.mean(ts),
                "diagnostic_name":dname,"mean_inspected":statistics.mean(di),
            })
        for rl,_ in _region_sizes(ds.spec):
            for sl in ("short","medium","long"):
                group=[q for q in cq if q.region_label==rl and q.span_label==sl];ts=[];di=[];dname=""
                for q in group:
                    ts+=time_call(lambda q=q:rep.changes(q.region,q.t_a,q.t_b),warmups,repetitions)
                    dname,dv=diagnostic(rep,"change");di.append(dv)
                rows.append({
                    **scenario_fields,**metric_fields,"sessions":n,"method":method,
                    "query_kind":"changes","region":rl,"span":sl,"samples":len(ts),
                    "median_ms":statistics.median(ts),"p95_ms":percentile(ts,.95),"mean_ms":statistics.mean(ts),
                    "diagnostic_name":dname,"mean_inspected":statistics.mean(di),
                })
    return build,rows


def run_benchmark(output_dir:Path,scenarios:list[ScenarioConfig],sessions=250,seed=42,query_count=8,warmups=1,repetitions=3):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);build=[];queries=[]
    for i,cfg in enumerate(scenarios,1):
        print(f"[Step 6] {i}/{len(scenarios)} {cfg.name}",flush=True)
        b,q=benchmark_scenario(cfg,sessions,seed,query_count,warmups,repetitions);build+=b;queries+=q
    write_csv(output_dir/"scenario_build.csv",build);write_csv(output_dir/"scenario_queries.csv",queries)
    meta={
        "sessions":sessions,"seed":seed,"query_count_per_group":query_count,
        "warmups":warmups,"repetitions":repetitions,
        "methods":[name for name,_ in METHODS],
        "scenarios":[asdict(x) for x in scenarios],
        "runtime":{"python":sys.version,"platform":platform.platform(),"machine":platform.machine()},
        "notes":[
            "OctreeInterval gets the same direct dyadic validity-prefix idea as BST interval but duplicates histories at leaves.",
            "OctreeShared deduplicates temporal signatures globally, matching the BST temporal-sharing advantage.",
            "BSTAdaptive is unchanged from Step 5; Step 6 changes baselines and datasets, not the proposed method.",
            "All generated benchmark queries are exact-equivalence checked before timing.",
            "BST variants remain local-metric research prototypes, not verified upstream BST-ID.",
        ],
    }
    (output_dir/"metadata.json").write_text(json.dumps(meta,indent=2))
    return build,queries
