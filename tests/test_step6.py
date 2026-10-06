from synthetic.scenarios import ScenarioConfig, AdversarialScenarioGenerator
from representations import (
    SnapshotMap, OctreeTemporalMap, OctreeIntervalMap, OctreeSharedSignatureMap,
    BSTIntervalMapPrototype, BSTAdaptiveIntervalMapPrototype,
)
from benchmark.storage import compact_storage_bytes
from benchmark.step6_benchmark import run_benchmark


def make_dataset(n=24, config=None):
    cfg=config or ScenarioConfig("test",target_static_fraction=.8,change_probability=.2)
    return AdversarialScenarioGenerator(cfg,42).generate(n)


def test_strengthened_baselines_exact_equivalence():
    ds,_=make_dataset(24)
    reps=[
        SnapshotMap.from_dataset(ds),OctreeTemporalMap.from_dataset(ds),
        OctreeIntervalMap.from_dataset(ds),OctreeSharedSignatureMap.from_dataset(ds),
        BSTIntervalMapPrototype.from_dataset(ds),BSTAdaptiveIntervalMapPrototype.from_dataset(ds),
    ]
    regions=[(0,5,0,5,0,5),(55,85,10,40,0,5),(0,100,0,100,0,5)]
    for t in (0,5,12,23):
        ref=reps[0].snapshot(t)
        for r in reps[1:]: assert r.snapshot(t)==ref
        for region in regions:
            ref=reps[0].region_at_time(region,t)
            for r in reps[1:]: assert r.region_at_time(region,t)==ref
    for region in regions:
        for ta,tb in ((0,1),(2,12),(5,23),(20,4)):
            ref=reps[0].changes(region,ta,tb)
            for r in reps[1:]: assert r.changes(region,ta,tb)==ref


def test_static_fraction_sweep_moves_in_expected_direction():
    _,m95=make_dataset(8,ScenarioConfig("s95",target_static_fraction=.95))
    _,m50=make_dataset(8,ScenarioConfig("s50",target_static_fraction=.50))
    assert m95.initial_static_fraction > m50.initial_static_fraction
    assert m95.dynamic_object_count < m50.dynamic_object_count


def test_fragmented_has_more_dynamic_objects_than_coherent():
    _,mc=make_dataset(8,ScenarioConfig("c",coherence="coherent"))
    _,mf=make_dataset(8,ScenarioConfig("f",coherence="fragmented"))
    assert mf.dynamic_object_count > mc.dynamic_object_count


def test_storage_metrics_include_new_baselines():
    ds,_=make_dataset(12)
    for rep in (OctreeIntervalMap.from_dataset(ds),OctreeSharedSignatureMap.from_dataset(ds)):
        assert compact_storage_bytes(rep) > 0


def test_quick_step6_benchmark(tmp_path):
    cfgs=[
        ScenarioConfig("s95",target_static_fraction=.95),
        ScenarioConfig("frag",coherence="fragmented"),
    ]
    build,queries=run_benchmark(tmp_path/"s6",cfgs,sessions=12,query_count=1,warmups=0,repetitions=1)
    assert len(build)==len(cfgs)*6
    assert queries
    assert (tmp_path/"s6"/"scenario_build.csv").exists()
    assert (tmp_path/"s6"/"scenario_queries.csv").exists()
