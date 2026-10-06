from synthetic import SyntheticLifelongGenerator,WorldSpec
from representations import SnapshotMap,BSTIntervalMapPrototype,BSTAdaptiveIntervalMapPrototype
from representations.interval_utils import build_occupied_intervals,dyadic_cover,bits_required,prefix_active,tokens_by_depth
from benchmark.step5_benchmark import run_benchmark

def ds(n=20):return SyntheticLifelongGenerator(WorldSpec(100,100,5,.5),42).generate(n)

def test_dyadic_cover_exact_membership():
    tb=bits_required(100)
    for a,b in [(0,100),(3,19),(17,18),(50,99)]:
        toks=dyadic_cover(a,b,tb);idx=tokens_by_depth(toks)
        for t in range(100):assert prefix_active(idx,t,tb)==(a<=t<b)

def test_interval_reconstructs_snapshots():
    d=ds(20);ref=SnapshotMap.from_dataset(d);m=BSTIntervalMapPrototype.from_dataset(d)
    for t in [0,1,5,10,19]:assert m.snapshot(t)==ref.snapshot(t)

def test_adaptive_reconstructs_snapshots():
    d=ds(20);ref=SnapshotMap.from_dataset(d);m=BSTAdaptiveIntervalMapPrototype.from_dataset(d)
    for t in [0,1,5,10,19]:assert m.snapshot(t)==ref.snapshot(t)

def test_adaptive_changes_exact():
    d=ds(20);ref=SnapshotMap.from_dataset(d);m=BSTAdaptiveIntervalMapPrototype.from_dataset(d);region=(45,95,5,95,0,5)
    for a,b in [(0,1),(1,10),(3,19),(19,5)]:assert m.changes(region,a,b)==ref.changes(region,a,b)

def test_spatial_merging_occurs():
    d=ds(20);m=BSTAdaptiveIntervalMapPrototype.from_dataset(d)
    assert len(m.blocks)<m.unmerged_voxels

def test_quick_step5_benchmark(tmp_path):
    b,q=run_benchmark(tmp_path/'out',[6],query_count=1,warmups=0,repetitions=1)
    assert {x['method'] for x in b}=={'snapshot','octree_time','bst_interval','bst_adaptive'}
    assert q
