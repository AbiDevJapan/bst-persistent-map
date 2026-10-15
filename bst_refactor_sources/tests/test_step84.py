from __future__ import annotations

import inspect
import numpy as np

from synthetic.world import WorldSpec
from realdata.nclt import NCLTIngestConfig, NCLTSessionStats
from realdata.nclt_packed import PackedNCLTPrepared, linear_encode
from representations.packed_scale import PackedSnapshot, build_history_table
from representations.step83 import (
    AdaptiveBlockCore, TimeLocalAdaptiveCore, TimeLocalExplicitMap,
    SnapshotAdapter, build_session_spatial_blocks,
)
from bst_id import BSTTimeLocalMap, PrototypeBSTSpatialCodec
from benchmark.step84_benchmark import benchmark_packed_step84


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


def _prepared_history_mismatch():
    # Adjacent A/B share occupancy at t0,t1 but have different complete histories.
    # Full-history construction cannot merge them; time-local construction can.
    spec=WorldSpec(4,2,1,1)
    A=(0,0,0); B=(1,0,0)
    ss=[_codes(spec,{A,B}),_codes(spec,{A,B}),_codes(spec,{A})]
    return PackedNCLTPrepared(spec,['s0','s1','s2'],ss,(0,0,0),[NCLTSessionStats(f's{i}') for i in range(3)],NCLTIngestConfig(voxel_size_m=1))


def test_time_local_exact_region_and_change_outputs():
    p=_prepared5(); union,masks=build_history_table(p,progress=False)
    sb,_=build_session_spatial_blocks(p,progress=False)
    core=TimeLocalAdaptiveCore(p,sb,union,progress=False)
    rep=TimeLocalExplicitMap(core); snap=SnapshotAdapter(PackedSnapshot(p))
    regs=((0,4,0,4,0,2),(0,8,0,8,0,2))
    for reg in regs:
        for t in range(5):
            ref=snap.region_query(reg,t)
            got=rep.region_query(reg,t)
            assert np.array_equal(got,ref)
            native=rep.region_query(reg,t,output='native')
            assert int(np.prod(native[:,3:6].astype(np.int64),axis=1).sum())==len(ref)
        for ta,tb in ((0,1),(0,4),(3,1)):
            ea,er=snap.changes_query(reg,ta,tb)
            ga,gr=rep.changes_query(reg,ta,tb)
            assert np.array_equal(ga,ea) and np.array_equal(gr,er)


def test_time_local_merges_neighbors_despite_different_lifetime_histories():
    p=_prepared_history_mismatch(); union,masks=build_history_table(p,progress=False)
    old=AdaptiveBlockCore(p,union,masks,progress=False)
    sb,_=build_session_spatial_blocks(p,progress=False)
    new=TimeLocalAdaptiveCore(p,sb,union,progress=False)
    reg=(0,4,0,2,0,1)
    old_native=old.region_native(reg,0)
    new_native=new.region_native(reg,0)
    assert len(old_native)==2
    assert len(new_native)==1
    assert int(np.prod(new_native[:,3:6].astype(np.int64),axis=1).sum())==2


def test_time_local_bst_isolation_and_codec_swap_boundary():
    p=_prepared5(); union,_=build_history_table(p,progress=False)
    sb,_=build_session_spatial_blocks(p,progress=False)
    core=TimeLocalAdaptiveCore(p,sb,union,progress=False)
    bst=BSTTimeLocalMap(core,PrototypeBSTSpatialCodec())
    src=inspect.getsource(type(core))
    assert 'bst_id' not in src.lower()
    reg=(0,8,0,8,0,2)
    native=bst.region_query(reg,3,output='native')
    vox=bst.region_query(reg,3,output='voxels')
    assert int(np.prod(native[:,3:6].astype(np.int64),axis=1).sum())==len(vox)
    assert bst.compact_bytes()>0
    try:
        bst.changes_query(reg,0,4,output='native')
        assert False, 'native change should be explicitly unsupported for time-varying tilings'
    except NotImplementedError:
        pass


def test_step84_scaling_smoke(tmp_path):
    p=_prepared5()
    b,q,h=benchmark_packed_step84(
        p,tmp_path/'out',scales=(2,3,5),query_count=1,warmups=0,repetitions=1,
        query_seed=9,targeted_fraction=1.0,include_full=False,output_modes=('voxels','native'),progress=False,
    )
    methods={r['method'] for r in b}
    assert {'bst_compact_bucket','bst_time_local','time_local_explicit','voxel_history','delta_version'} <= methods
    assert {r['sessions'] for r in b}=={2,3,5}
    assert any(r['output_mode']=='native' and r['method']=='bst_time_local' and r['query_kind']=='region_at_time' for r in q)
    assert not any(r['output_mode']=='native' and r['method']=='bst_time_local' and r['query_kind']=='changes' for r in q)
    assert any(r['output_mode']=='voxels' and r['method']=='bst_time_local' and r['query_kind']=='changes' for r in q)
    assert (tmp_path/'out'/'scaling_build_storage.csv').exists()
    assert (tmp_path/'out'/'scaling_queries.csv').exists()
