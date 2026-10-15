from synthetic.scenarios import ScenarioConfig, AdversarialScenarioGenerator
from representations import (
    SnapshotMap, AdaptiveBlockExplicitMap, BSTAdaptiveIntervalMapPrototype,
    BSTPrefixIndexedAdaptiveMapPrototype,
)
from benchmark.storage import compact_storage_bytes
from benchmark.step7_benchmark import run_benchmark


def make_ds(n=20):
    return AdversarialScenarioGenerator(
        ScenarioConfig("test",target_static_fraction=.8,change_probability=.2),
        seed=42,
    ).generate(n)[0]


def test_identical_adaptive_blocks():
    ds=make_ds()
    a=AdaptiveBlockExplicitMap.from_dataset(ds)
    b=BSTAdaptiveIntervalMapPrototype.from_dataset(ds)
    c=BSTPrefixIndexedAdaptiveMapPrototype.from_dataset(ds)
    assert tuple(a.blocks)==tuple(b.blocks)==tuple(c.blocks)


def test_prefix_index_exact_equivalence():
    ds=make_ds()
    ref=SnapshotMap.from_dataset(ds)
    p=BSTPrefixIndexedAdaptiveMapPrototype.from_dataset(ds)
    regions=[(0,5,0,5,0,5),(20,40,20,40,0,5),(0,100,0,100,0,5)]
    for t in (0,5,10,19):
        for r in regions:
            assert p.region_at_time(r,t)==ref.region_at_time(r,t)
    for r in regions:
        for ta,tb in ((0,1),(2,15),(18,4)):
            assert p.changes(r,ta,tb)==ref.changes(r,ta,tb)


def test_storage_isolates_encoding():
    ds=make_ds()
    explicit=AdaptiveBlockExplicitMap.from_dataset(ds)
    compact=BSTAdaptiveIntervalMapPrototype.from_dataset(ds)
    prefix=BSTPrefixIndexedAdaptiveMapPrototype.from_dataset(ds)
    assert compact_storage_bytes(explicit)>0
    assert compact_storage_bytes(compact)>0
    assert compact_storage_bytes(prefix)>0


def test_quick_step7(tmp_path):
    cfg=(ScenarioConfig("baseline"),)
    build,queries,summary=run_benchmark(
        tmp_path/"s7",scenarios=cfg,seeds=(1,),sessions=10,
        query_count=1,warmups=0,repetitions=1,
    )
    assert build and queries and summary
    assert (tmp_path/"s7"/"isolation_build.csv").exists()
    assert (tmp_path/"s7"/"isolation_queries.csv").exists()
    assert (tmp_path/"s7"/"isolation_summary.csv").exists()
