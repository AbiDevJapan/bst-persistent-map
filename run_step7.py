from argparse import ArgumentParser
from pathlib import Path
from benchmark.step7_benchmark import run_benchmark, DEFAULT_SCENARIOS
from synthetic.scenarios import ScenarioConfig

def main():
    p=ArgumentParser()
    p.add_argument("--suite",choices=["quick","full"],default="quick")
    p.add_argument("--sessions",type=int,default=None)
    p.add_argument("--queries",type=int,default=None)
    p.add_argument("--warmups",type=int,default=1)
    p.add_argument("--repetitions",type=int,default=3)
    p.add_argument("--output",default="results/step7")
    a=p.parse_args()

    if a.suite=="quick":
        scenarios=(
            ScenarioConfig("baseline",target_static_fraction=.8,change_probability=.15,coherence="mixed"),
            ScenarioConfig("fragmented",target_static_fraction=.8,change_probability=.15,coherence="fragmented"),
        )
        seeds=(7,42)
        sessions=a.sessions or 100
        queries=a.queries or 4
    else:
        scenarios=DEFAULT_SCENARIOS
        seeds=(1,7,21,42,84)
        sessions=a.sessions or 250
        queries=a.queries or 20

    build,q,summary=run_benchmark(
        Path(a.output),scenarios=scenarios,seeds=seeds,sessions=sessions,
        query_count=queries,warmups=a.warmups,repetitions=a.repetitions,
    )
    print(f"\nWrote {a.output}/isolation_build.csv")
    print(f"Wrote {a.output}/isolation_queries.csv")
    print(f"Wrote {a.output}/isolation_summary.csv")
    print(f"Wrote {a.output}/metadata.json")

if __name__=="__main__":
    main()
