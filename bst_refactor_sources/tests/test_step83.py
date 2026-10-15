from __future__ import annotations

import inspect
import numpy as np

from synthetic.world import WorldSpec
from realdata.nclt import NCLTIngestConfig, NCLTSessionStats
from realdata.nclt_packed import PackedNCLTPrepared, linear_encode
from representations.packed_scale import PackedSnapshot, PackedOctreeShared, build_history_table
from representations.step83 import AdaptiveBlockCore, ExplicitAdaptiveMap, PackedVoxelHistoryMap, DeltaVersionMap, SnapshotAdapter, OctreeAdapter
from bst_id import BSTCompactMap, PrototypeBSTSpatialCodec
from benchmark.step83_benchmark import benchmark_packed_step83


def _codes(spec,coords):
    if not coords:return np.empty(0,dtype=np.uint64)
    a=np.asarray(sorted(coords),dtype=np.int64)
    return np.unique(linear_encode(a[:,0],a[:,1],a[:,2],spec))


def _prepared5():
    spec=WorldSpec(8,8,2,1)
    base={(x,y,0) for x in range(1,7) for y in range(1,7)}
    ss=[]
    for t in range(5):
        cur=set(base)
        if t>=1: cur |= {(0,0,0),(0,1,0)}
        if t>=2: cur -= {(2,2,0),(2,3,0)}
        if t>=3: cur |= {(7,7,0),(7,6,0)}
        if t==4: cur -= {(5,5,0)}
        ss.append(_codes(spec,cur))
    return PackedNCLTPrepared(spec,[f's{t}' for t in range(5)],ss,(0,0,0),[NCLTSessionStats(f's{t}') for t in range(5)],NCLTIngestConfig(voxel_size_m=1))


def test_new_non_octree_baselines_exact():
    p=_prepared5(); union,masks=build_history_table(p,progress=False); snap=SnapshotAdapter(PackedSnapshot(p))
    voxel=PackedVoxelHistoryMap(p,union,masks); delta=DeltaVersionMap(p)
    regs=((0,4,0,4,0,2),(0,8,0,8,0,2))
    for reg in regs:
        for t in range(5):
            ref=snap.region_query(reg,t)
            assert np.array_equal(voxel.region_query(reg,t),ref)
            assert np.array_equal(delta.region_query(reg,t),ref)
        for ta,tb in ((0,1),(0,4),(3,1)):
            ra,rr=snap.changes_query(reg,ta,tb)
            for rep in (voxel,delta):
                a,r=rep.changes_query(reg,ta,tb)
                assert np.array_equal(a,ra) and np.array_equal(r,rr)


def test_adaptive_core_is_bst_agnostic_and_codec_swappable():
    p=_prepared5(); union,masks=build_history_table(p,progress=False); core=AdaptiveBlockCore(p,union,masks,progress=False)
    explicit=ExplicitAdaptiveMap(core); bst=BSTCompactMap(core,PrototypeBSTSpatialCodec())
    assert explicit.core is bst.core
    # Generic adaptive core must not import or reference bst_id.
    src=inspect.getsource(type(core))
    assert 'bst_id' not in src.lower()
    reg=(0,8,0,8,0,2)
    assert np.array_equal(explicit.region_query(reg,3),bst.region_query(reg,3))
    native=bst.region_query(reg,3,output='native')
    vox=bst.region_query(reg,3,output='voxels')
    assert int(np.prod(native[:,3:6].astype(np.int64),axis=1).sum())==len(vox)
    assert bst.compact_bytes() < explicit.compact_bytes()


def test_prototype_codec_encodes_aligned_block():
    p=_prepared5(); union,masks=build_history_table(p,progress=False); core=AdaptiveBlockCore(p,union,masks,progress=False)
    codec=PrototypeBSTSpatialCodec()
    code=codec.encode_block(core.blocks[0],p.spec)
    assert code.x.length>=0 and code.y.length>=0 and code.z.length>=0
    assert codec.spatial_table_bytes(core.blocks,p.spec)>0


def test_step83_scaling_smoke(tmp_path):
    p=_prepared5()
    b,q,h=benchmark_packed_step83(
        p,tmp_path/'out',scales=(2,3,5),query_count=1,warmups=0,repetitions=1,
        query_seed=9,targeted_fraction=1.0,include_full=False,output_modes=('voxels','native'),progress=False,
    )
    methods={r['method'] for r in b}
    assert {'delta_version','voxel_history','octree_shared_sparse','adaptive_explicit','bst_compact_bucket','snapshot'} <= methods
    assert {r['sessions'] for r in b}=={2,3,5}
    assert any(r['output_mode']=='native' and r['method']=='bst_compact_bucket' for r in q)
    assert any(r['output_mode']=='voxels' and r['method']=='delta_version' for r in q)
    assert (tmp_path/'out'/'scaling_build_storage.csv').exists()
    assert (tmp_path/'out'/'scaling_queries.csv').exists()
    assert (tmp_path/'out'/'history_complexity_scaling.csv').exists()
