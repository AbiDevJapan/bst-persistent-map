from argparse import ArgumentParser
from pathlib import Path

from benchmark.step6_benchmark import run_benchmark
from synthetic.scenarios import default_scenarios, quick_scenarios


def main():
    p=ArgumentParser()
    p.add_argument("--suite",choices=("quick","full"),default="quick")
    p.add_argument("--sessions",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--queries",type=int,default=4)
    p.add_argument("--warmups",type=int,default=1)
    p.add_argument("--repetitions",type=int,default=3)
    p.add_argument("--output",default="results/step6")
    a=p.parse_args()
    scenarios=quick_scenarios() if a.suite=="quick" else default_scenarios()
    build,queries=run_benchmark(Path(a.output),scenarios,a.sessions,a.seed,a.queries,a.warmups,a.repetitions)
    print(f"\nWrote {a.output}/scenario_build.csv")
    print(f"Wrote {a.output}/scenario_queries.csv")
    print(f"Wrote {a.output}/metadata.json")

    # Compact headline comparing the strongest fair Octree and BST baselines.
    print("\nHeadline storage ratios (BST adaptive / Octree shared):")
    by={(r["scenario_name"],r["method"]):r for r in build}
    for cfg in scenarios:
        o=by[(cfg.name,"octree_shared")]["compact_bytes"]
        b=by[(cfg.name,"bst_adaptive")]["compact_bytes"]
        print(f"  {cfg.name:14s} {b/o:.3f}x")

if __name__=="__main__": main()
