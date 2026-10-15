from __future__ import annotations
import csv,gc,json,math,platform,random,statistics,sys,time
from dataclasses import dataclass
from pathlib import Path
from synthetic import SyntheticLifelongGenerator,WorldSpec
from representations import (
    SnapshotMap,OctreeTemporalMap,BSTTemporalMapPrototype,
    BSTIntervalMapPrototype,BSTAdaptiveIntervalMapPrototype,
)
from benchmark.storage import compact_storage_bytes

BASE_METHODS=(
    ('snapshot',SnapshotMap),
    ('octree_time',OctreeTemporalMap),
    ('bst_interval',BSTIntervalMapPrototype),
    ('bst_adaptive',BSTAdaptiveIntervalMapPrototype),
)
EVENT_METHOD=('bst_events',BSTTemporalMapPrototype)

@dataclass(frozen=True)
class RegionQuery:
    region_label:str; region:tuple[float,float,float,float,float,float]; t:int
@dataclass(frozen=True)
class ChangeQuery:
    region_label:str; region:tuple[float,float,float,float,float,float]; span_label:str; t_a:int; t_b:int

def percentile(vals,q):
    s=sorted(vals); pos=(len(s)-1)*q; lo=math.floor(pos); hi=math.ceil(pos)
    return s[lo] if lo==hi else s[lo]*(hi-pos)+s[hi]*(pos-lo)

def make_region_queries(spec,n,count,seed):
    rng=random.Random(seed+1000*n);out=[]
    for label,size in [('5m',5.0),('20m',20.0),('full',spec.size_x_m)]:
        for _ in range(count):
            if label=='full': reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m)
            else:
                x0=rng.uniform(0,spec.size_x_m-size);y0=rng.uniform(0,spec.size_y_m-size)
                reg=(x0,x0+size,y0,y0+size,0,spec.size_z_m)
            bucket=rng.choice(['old','middle','recent'])
            if bucket=='old': t=rng.randrange(0,max(1,n//3))
            elif bucket=='middle':
                lo=n//3;hi=max(lo+1,2*n//3);t=rng.randrange(lo,hi)
            else:t=rng.randrange(2*n//3,n)
            out.append(RegionQuery(label,reg,t))
    return out

def make_change_queries(spec,n,count,seed):
    rng=random.Random(seed+2000*n);out=[]
    spans=[('short',1),('medium',max(1,n//10)),('long',max(1,n//2))]
    for rl,size in [('5m',5.0),('20m',20.0),('full',spec.size_x_m)]:
        for sl,span in spans:
            for _ in range(count):
                if rl=='full':reg=(0,spec.size_x_m,0,spec.size_y_m,0,spec.size_z_m)
                else:
                    x0=rng.uniform(0,spec.size_x_m-size);y0=rng.uniform(0,spec.size_y_m-size)
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
    st=rep.last_query_stats if kind=='region' else rep.last_change_stats
    for k in ('records_scanned','nodes_visited','events_scanned','blocks_scanned'):
        if k in st:return k,float(st[k])
    return 'unknown',float('nan')

def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def benchmark_one_n(n,seed,query_count,warmups,repetitions,include_events=False):
    spec=WorldSpec(100,100,5,0.5);ds=SyntheticLifelongGenerator(spec,seed).generate(n)
    reps={};build=[]
    methods=list(BASE_METHODS)
    if include_events: methods.insert(2, EVENT_METHOD)
    for name,cls in methods:
        t0=time.perf_counter_ns();rep=cls.from_dataset(ds);t1=time.perf_counter_ns();reps[name]=rep
        size=compact_storage_bytes(rep)
        build.append({'sessions':n,'method':name,'build_ms':(t1-t0)/1e6,'compact_bytes':size,'compact_mib':size/(1024*1024),**{f'stat_{k}':v for k,v in rep.stats().items()}})
    rq=make_region_queries(spec,n,query_count,seed);cq=make_change_queries(spec,n,query_count,seed)

    # exact equivalence gates for every method
    for label in ('5m','20m','full'):
        q=next(x for x in rq if x.region_label==label);ref=reps['snapshot'].region_at_time(q.region,q.t)
        for name,rep in reps.items():
            assert rep.region_at_time(q.region,q.t)==ref,(name,label,q.t)
    for rl in ('5m','20m','full'):
        for sl in ('short','medium','long'):
            q=next(x for x in cq if x.region_label==rl and x.span_label==sl);ref=reps['snapshot'].changes(q.region,q.t_a,q.t_b)
            for name,rep in reps.items():
                assert rep.changes(q.region,q.t_a,q.t_b)==ref,(name,rl,sl,q.t_a,q.t_b)

    rows=[]
    for method,rep in reps.items():
        for label in ('5m','20m','full'):
            group=[q for q in rq if q.region_label==label];ts=[];di=[];dname=''
            for q in group:
                ts+=time_call(lambda q=q:rep.region_at_time(q.region,q.t),warmups,repetitions)
                dname,dv=diagnostic(rep,'region');di.append(dv)
            rows.append({'sessions':n,'method':method,'query_kind':'region_at_time','region':label,'span':'','samples':len(ts),'median_ms':statistics.median(ts),'p95_ms':percentile(ts,.95),'mean_ms':statistics.mean(ts),'diagnostic_name':dname,'mean_inspected':statistics.mean(di)})
        for rl in ('5m','20m','full'):
            for sl in ('short','medium','long'):
                group=[q for q in cq if q.region_label==rl and q.span_label==sl];ts=[];di=[];dname=''
                for q in group:
                    ts+=time_call(lambda q=q:rep.changes(q.region,q.t_a,q.t_b),warmups,repetitions)
                    dname,dv=diagnostic(rep,'change');di.append(dv)
                rows.append({'sessions':n,'method':method,'query_kind':'changes','region':rl,'span':sl,'samples':len(ts),'median_ms':statistics.median(ts),'p95_ms':percentile(ts,.95),'mean_ms':statistics.mean(ts),'diagnostic_name':dname,'mean_inspected':statistics.mean(di)})
    return build,rows

def run_benchmark(output_dir,session_counts,seed=42,query_count=8,warmups=1,repetitions=3,include_events=False):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);build=[];query=[]
    for n in session_counts:
        print(f'[Step 5] N={n}',flush=True);b,q=benchmark_one_n(n,seed,query_count,warmups,repetitions,include_events=include_events);build+=b;query+=q
    write_csv(output_dir/'build_storage.csv',build);write_csv(output_dir/'queries.csv',query)
    meta={'session_counts':session_counts,'seed':seed,'query_count_per_group':query_count,'warmups':warmups,'repetitions':repetitions,'world':{'size_m':[100,100,5],'voxel_size_m':.5},'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},'include_step4_event_baseline':include_events,'notes':['Step 5 focuses on temporal-interval and adaptive space-time variants; the Step-4 event baseline can be included with --include-events.','Temporal validity is encoded by exact dyadic time-prefix covers.','Adaptive spatial blocks merge independently along x/y/z only when temporal histories are identical.','BST variants remain local-metric research prototypes, not verified upstream BST-ID.']}
    (output_dir/'metadata.json').write_text(json.dumps(meta,indent=2));return build,query
