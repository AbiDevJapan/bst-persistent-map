from pathlib import Path
import numpy as np

from realdata.nclt import (
    NCLTIngestConfig, read_velodyne_sync_bin, prepare_nclt_dataset,
)
from benchmark.step8_benchmark import history_complexity, benchmark_prepared


DT=np.dtype([
    ("x","<u2"),("y","<u2"),("z","<u2"),("intensity","u1"),("laser","u1")
])


def encode_points(path: Path, xyz):
    arr=np.zeros(len(xyz),dtype=DT)
    raw=np.rint((np.asarray(xyz)+100.0)/0.005).astype(np.uint16)
    arr["x"]=raw[:,0];arr["y"]=raw[:,1];arr["z"]=raw[:,2]
    arr["intensity"]=20;arr["laser"]=1
    arr.tofile(path)


def make_fake_session(root:Path,name:str,dx:float):
    sd=root/name/"velodyne_sync";sd.mkdir(parents=True)
    gt=[]
    for i,ts in enumerate((1_000_000,1_100_000,1_200_000)):
        # Body at x=dx in native local frame; zero rotation.
        gt.append([ts,dx,0,0,0,0,0])
        encode_points(sd/f"{ts}.bin",[
            (5,0,0),(5.1,0,0),(6,1,0),(7,-1,0.5)
        ])
    np.savetxt(root/name/f"groundtruth_{name}.csv",np.asarray(gt),delimiter=",")


def test_nclt_decoder(tmp_path):
    p=tmp_path/"x.bin"
    encode_points(p,[(1,2,3),(-4,5,-6)])
    q=read_velodyne_sync_bin(p)
    assert np.allclose(q,np.asarray([(1,2,3),(-4,5,-6)]),atol=.003)


def test_prepare_fake_nclt(tmp_path):
    make_fake_session(tmp_path,"2012-01-15",0)
    make_fake_session(tmp_path,"2012-02-02",1)
    cfg=NCLTIngestConfig(
        voxel_size_m=.5,scan_stride=1,min_range_m=0,max_range_m=20,
        min_points_per_voxel=1,max_pose_gap_us=1,
    )
    p=prepare_nclt_dataset(tmp_path,["2012-01-15","2012-02-02"],cfg)
    assert len(p.dataset.sessions)==2
    assert p.dataset.sessions[0].occupied
    assert p.dataset.sessions[0].occupied != p.dataset.sessions[1].occupied
    p.dataset.validate()


def test_history_complexity_fake(tmp_path):
    make_fake_session(tmp_path,"2012-01-15",0)
    make_fake_session(tmp_path,"2012-02-02",0)
    cfg=NCLTIngestConfig(
        voxel_size_m=.5,scan_stride=1,min_range_m=0,max_range_m=20,
        min_points_per_voxel=1,max_pose_gap_us=1,
    )
    p=prepare_nclt_dataset(tmp_path,["2012-01-15","2012-02-02"],cfg)
    h=history_complexity(p.dataset)
    assert h["fraction_zero_transitions"]==1.0


def test_step8_benchmark_fake(tmp_path):
    make_fake_session(tmp_path,"2012-01-15",0)
    make_fake_session(tmp_path,"2012-02-02",1)
    cfg=NCLTIngestConfig(
        voxel_size_m=.5,scan_stride=1,min_range_m=0,max_range_m=20,
        min_points_per_voxel=1,max_pose_gap_us=1,
    )
    p=prepare_nclt_dataset(tmp_path,["2012-01-15","2012-02-02"],cfg)
    b,q,h=benchmark_prepared(
        p,tmp_path/"out",query_count=1,warmups=0,repetitions=1,query_seed=1
    )
    assert b and q and h
    assert (tmp_path/"out"/"real_build_storage.csv").exists()
    assert (tmp_path/"out"/"real_queries.csv").exists()
    assert (tmp_path/"out"/"history_complexity.json").exists()
