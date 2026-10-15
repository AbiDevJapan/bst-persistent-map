from pathlib import Path
import numpy as np

from synthetic.world import WorldSpec
from synthetic.generator import SyntheticLifelongGenerator
from realdata.nclt import NCLTIngestConfig, NCLTSessionStats
from realdata.nclt_packed import PackedNCLTPrepared, linear_encode, prepare_nclt_packed
from representations import OctreeSharedSignatureMap, BSTAdaptiveIntervalMapPrototype
from representations.packed_scale import (
    PackedSnapshot, PackedOctreeShared, PackedAdaptiveBucket,
    build_history_table, merge_coords_fast,
)
from benchmark.storage import octree_shared_bytes, bst_adaptive_bytes
from benchmark.step81_benchmark import benchmark_packed

DT=np.dtype([("x","<u2"),("y","<u2"),("z","<u2"),("intensity","u1"),("laser","u1")])

def _encode_scan(path,xyz):
    arr=np.zeros(len(xyz),dtype=DT)
    raw=np.rint((np.asarray(xyz)+100.0)/0.005).astype(np.uint16)
    arr["x"]=raw[:,0];arr["y"]=raw[:,1];arr["z"]=raw[:,2];arr["intensity"]=20;arr["laser"]=1
    arr.tofile(path)

def _fake_session(root,name,dx):
    sd=root/name/"velodyne_sync";sd.mkdir(parents=True)
    gt=[]
    for ts in (1_000_000,1_100_000,1_200_000):
        gt.append([ts,dx,0,0,0,0,0])
        _encode_scan(sd/f"{ts}.bin",[(5,0,0),(5.1,0,0),(6,1,0),(7,-1,.5)])
    np.savetxt(root/name/f"groundtruth_{name}.csv",np.asarray(gt),delimiter=",")

def _packed_from_generated(ds):
    codes=[]
    for s in ds.sessions:
        if s.occupied:
            a=np.asarray(sorted(s.occupied),dtype=np.int64)
            c=linear_encode(a[:,0],a[:,1],a[:,2],ds.spec)
        else:c=np.empty(0,dtype=np.uint64)
        codes.append(np.unique(c))
    return PackedNCLTPrepared(ds.spec,[str(i) for i in range(len(codes))],codes,(0,0,0),[NCLTSessionStats(str(i)) for i in range(len(codes))],NCLTIngestConfig(voxel_size_m=ds.spec.voxel_size_m))

def test_fast_merge_matches_legacy_random():
    rng=np.random.default_rng(2); spec=WorldSpec(8,8,4,1)
    vox=np.argwhere(rng.random((8,8,4))<.45).astype(np.int32)
    got=merge_coords_fast(vox,(8,8,4),None)
    ds=SyntheticLifelongGenerator(spec=spec,seed=1).generate(1)
    # Use legacy merger directly on our chosen voxels.
    from representations.bst_adaptive import merge_signature_voxels
    old=merge_signature_voxels({tuple(map(int,v)) for v in vox},0,(8,8,4))
    a={(int(r[0]),int(r[1]),int(r[2]),int(r[3]),int(r[4]),int(r[5])) for r in got}
    b={(x.x0,x.y0,x.z0,x.sx,x.sy,x.sz) for x in old}
    assert a==b

def test_packed_octree_matches_legacy_and_storage():
    ds=SyntheticLifelongGenerator(seed=42).generate(8)
    p=_packed_from_generated(ds); union,masks=build_history_table(p,progress=False)
    packed=PackedOctreeShared(p,union,masks); legacy=OctreeSharedSignatureMap.from_dataset(ds)
    assert packed.node_count==legacy.node_count
    assert packed.leaf_count==legacy.leaf_count
    assert packed.compact_bytes()==octree_shared_bytes(legacy)
    snap=PackedSnapshot(p)
    for reg in ((0,5,0,5,0,5),(20,40,20,40,0,5),(0,100,0,100,0,5)):
        for t in (0,3,7):
            assert np.array_equal(packed.region_codes(reg,t),snap.region_codes(reg,t))

def test_packed_adaptive_matches_legacy_storage_and_queries():
    ds=SyntheticLifelongGenerator(seed=42).generate(8)
    p=_packed_from_generated(ds); union,masks=build_history_table(p,progress=False)
    packed=PackedAdaptiveBucket(p,union,masks,progress=False); legacy=BSTAdaptiveIntervalMapPrototype.from_dataset(ds)
    assert len(packed.blocks)==len(legacy.blocks)
    assert packed.bst_compact_bytes()==bst_adaptive_bytes(legacy)
    snap=PackedSnapshot(p)
    for reg in ((0,5,0,5,0,5),(20,40,20,40,0,5)):
        for t in (0,3,7): assert np.array_equal(packed.region_codes(reg,t),snap.region_codes(reg,t))
        a,b=packed.changes_codes(reg,1,7); ra,rb=snap.changes_codes(reg,1,7)
        assert np.array_equal(a,ra) and np.array_equal(b,rb)

def test_packed_nclt_cache_and_benchmark(tmp_path):
    _fake_session(tmp_path,"2012-01-15",0); _fake_session(tmp_path,"2012-02-02",1)
    cfg=NCLTIngestConfig(voxel_size_m=.5,scan_stride=1,min_range_m=0,max_range_m=20,min_points_per_voxel=1,max_pose_gap_us=1)
    cache=tmp_path/"cache"
    p=prepare_nclt_packed(tmp_path,["2012-01-15","2012-02-02"],cfg,cache,flush_scans=1,progress_every=99)
    assert len(list(cache.glob("*.npy")))==2
    # Second construction must use the cache and preserve exact data.
    q=prepare_nclt_packed(tmp_path,["2012-01-15","2012-02-02"],cfg,cache,flush_scans=1,progress_every=99)
    assert all(np.array_equal(a,b) for a,b in zip(p.session_codes,q.session_codes))
    build,queries,h=benchmark_packed(p,tmp_path/"out",query_count=1,warmups=0,repetitions=1,query_seed=3,targeted_fraction=1.0,include_full=False,progress=False)
    assert build and queries and h
    assert (tmp_path/"out"/"real_build_storage.csv").exists()
