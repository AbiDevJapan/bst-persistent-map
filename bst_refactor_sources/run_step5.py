from argparse import ArgumentParser
from pathlib import Path
from benchmark.step5_benchmark import run_benchmark

def main():
    p=ArgumentParser();p.add_argument('--sessions',default='10,25,50,100');p.add_argument('--seed',type=int,default=42);p.add_argument('--queries',type=int,default=8);p.add_argument('--warmups',type=int,default=1);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--output',default='results/step5');p.add_argument('--include-events',action='store_true',help='also rerun the slow Step-4 bst_events baseline')
    a=p.parse_args();counts=[int(x) for x in a.sessions.split(',') if x.strip()]
    build,query=run_benchmark(Path(a.output),counts,a.seed,a.queries,a.warmups,a.repetitions,include_events=a.include_events);n=max(counts)
    print(f'\nStorage at N={n}')
    for r in build:
        if r['sessions']==n:print(f"  {r['method']:13s} {r['compact_mib']:.4f} MiB  build={r['build_ms']:.1f} ms")
    print(f'\n5m historical query at N={n}')
    for r in query:
        if r['sessions']==n and r['query_kind']=='region_at_time' and r['region']=='5m':print(f"  {r['method']:13s} median={r['median_ms']:.4f} ms p95={r['p95_ms']:.4f} {r['diagnostic_name']}={r['mean_inspected']:.1f}")
    print(f'\n5m short-change query at N={n}')
    for r in query:
        if r['sessions']==n and r['query_kind']=='changes' and r['region']=='5m' and r['span']=='short':print(f"  {r['method']:13s} median={r['median_ms']:.4f} ms p95={r['p95_ms']:.4f} {r['diagnostic_name']}={r['mean_inspected']:.1f}")
    print(f'\nResults written to {a.output}')
if __name__=='__main__':main()
