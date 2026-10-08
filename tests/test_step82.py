import numpy as np

from synthetic.generator import SyntheticLifelongGenerator
from realdata.nclt import NCLTIngestConfig, NCLTSessionStats
from realdata.nclt_packed import PackedNCLTPrepared, linear_encode
from representations.packed_scale import PackedOctreeShared, PackedSnapshot, build_history_table
from benchmark.step82_benchmark import benchmark_packed_step82


def _packed_from_generated(ds):
    codes=[]
    for s in ds.sessions:
        if s.occupied:
            a=np.asarray(sorted(s.occupied),dtype=np.int64)
            c=linear_encode(a[:,0],a[:,1],a[:,2],ds.spec)
        else:
            c=np.empty(0,dtype=np.uint64)
        codes.append(np.unique(c))
    return PackedNCLTPrepared(
        ds.spec,[str(i) for i in range(len(codes))],codes,(0,0,0),
        [NCLTSessionStats(str(i)) for i in range(len(codes))],
        NCLTIngestConfig(voxel_size_m=ds.spec.voxel_size_m),
    )


def test_sparse_octree_exactly_matches_geometric_reference():
    ds=SyntheticLifelongGenerator(seed=42).generate(12)
    p=_packed_from_generated(ds)
    union,masks=build_history_table(p,progress=False)
    sparse=PackedOctreeShared(p,union,masks,query_mode="sparse")
    geom=PackedOctreeShared(p,union,masks,query_mode="geometric")
    snap=PackedSnapshot(p)

    assert sparse.node_count==geom.node_count
    assert sparse.leaf_count==geom.leaf_count
    assert sparse.compact_bytes()==geom.compact_bytes()

    regs=((0,5,0,5,0,5),(20,40,20,40,0,5),(0,100,0,100,0,5))
    for reg in regs:
        for t in (0,5,11):
            ref=snap.region_codes(reg,t)
            a=sparse.region_codes(reg,t); b=geom.region_codes(reg,t)
            assert np.array_equal(a,ref)
            assert np.array_equal(b,ref)
            assert sparse.last_query_stats["nodes_visited"] <= geom.last_query_stats["nodes_visited"]
        a1,a2=sparse.changes_codes(reg,1,11)
        b1,b2=geom.changes_codes(reg,1,11)
        r1,r2=snap.changes_codes(reg,1,11)
        assert np.array_equal(a1,r1) and np.array_equal(a2,r2)
        assert np.array_equal(b1,r1) and np.array_equal(b2,r2)
        assert sparse.last_change_stats["nodes_visited"] <= geom.last_change_stats["nodes_visited"]


def test_sparse_octree_skips_empty_children():
    ds=SyntheticLifelongGenerator(seed=7).generate(8)
    p=_packed_from_generated(ds)
    union,masks=build_history_table(p,progress=False)
    sparse=PackedOctreeShared(p,union,masks,query_mode="sparse")
    sparse.region_codes((0,5,0,5,0,5),3)
    assert sparse.last_query_stats["octree_query_mode"]=="sparse"
    assert sparse.last_query_stats["empty_children_skipped"] > 0


def test_step82_benchmark_smoke(tmp_path):
    ds=SyntheticLifelongGenerator(seed=3).generate(6)
    p=_packed_from_generated(ds)
    build,queries,h=benchmark_packed_step82(
        p,tmp_path/"out",query_count=1,warmups=0,repetitions=1,
        query_seed=5,targeted_fraction=1.0,include_full=False,progress=False,
    )
    methods={r["method"] for r in build}
    assert "octree_shared_sparse" in methods
    assert "bst_compact_bucket" in methods
    assert (tmp_path/"out"/"metadata.json").exists()
