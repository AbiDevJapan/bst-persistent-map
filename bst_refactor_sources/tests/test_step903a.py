import numpy as np

from benchmark.step903a_benchmark import (
    parse_resolutions, scaled_map_pyramid, bucket_voxels_for_resolution,
)
from localization.step902 import DEFAULT_PYRAMID


def test_resolution_parser_and_scaled_map_pyramid():
    assert parse_resolutions("0.25,0.5,1.0") == (0.25, 0.5, 1.0)
    p025 = scaled_map_pyramid(0.25)
    p050 = scaled_map_pyramid(0.5)
    p100 = scaled_map_pyramid(1.0)
    assert np.allclose([x.map_voxel_m for x in p025], [0.5, 0.375, 0.25])
    assert np.allclose([x.map_voxel_m for x in p050], [1.0, 0.75, 0.5])
    assert np.allclose([x.map_voxel_m for x in p100], [2.0, 1.5, 1.0])
    # Scan-side resolution and physical correspondence geometry are frozen.
    assert [x.scan_voxel_m for x in p025] == [x.scan_voxel_m for x in DEFAULT_PYRAMID]
    assert [x.max_correspondence_m for x in p025] == [x.max_correspondence_m for x in DEFAULT_PYRAMID]
    assert [x.normal_max_radius_m for x in p100] == [x.normal_max_radius_m for x in DEFAULT_PYRAMID]


def test_bucket_width_is_constant_in_metric_space():
    for vs in (0.25, 0.5, 1.0):
        bv = bucket_voxels_for_resolution(vs, 8.0)
        assert abs(bv * vs - 8.0) <= vs / 2 + 1e-12


def test_prepare_map_only_reuse_only_does_not_write_new_cache(tmp_path):
    from pathlib import Path
    from realdata.nclt import NCLTIngestConfig
    from benchmark.step903a_benchmark import prepare_map_only

    dt = np.dtype([("x","<u2"),("y","<u2"),("z","<u2"),("intensity","u1"),("laser","u1")])
    name = "2012-01-15"
    sd = tmp_path / name / "velodyne_sync"
    sd.mkdir(parents=True)
    gt=[]
    for i,ts in enumerate((1_000_000,1_100_000,1_200_000)):
        gt.append([ts,0,0,0,0,0,0])
        xyz=np.asarray([(5,0,0),(5.25,0,0),(6,1,0),(7,-1,.5)],dtype=float)
        arr=np.zeros(len(xyz),dtype=dt)
        raw=np.rint((xyz+100.0)/0.005).astype(np.uint16)
        arr["x"]=raw[:,0]; arr["y"]=raw[:,1]; arr["z"]=raw[:,2]; arr["intensity"]=20; arr["laser"]=1
        arr.tofile(sd/f"{ts}.bin")
    np.savetxt(tmp_path/name/f"groundtruth_{name}.csv",np.asarray(gt),delimiter=",")
    cache=tmp_path/"cache"
    cfg=NCLTIngestConfig(voxel_size_m=.25,scan_stride=1,min_range_m=0,max_range_m=20,min_points_per_voxel=1,max_pose_gap_us=1)
    p=prepare_map_only(tmp_path,name,cfg,cache,"reuse-only")
    assert len(p.map_codes)>0
    assert not p.cache_hit
    assert not cache.exists() or not list(cache.iterdir())
