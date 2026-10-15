import math
import numpy as np

from localization.step90 import (
    VoxelLocalMapIndex, ICP3DOFConfig, Pose3DOF,
    pose_error_3dof, scan_to_map_icp_3dof,
)
from realdata.nclt_packed import linear_encode
from synthetic.world import WorldSpec


def test_voxel_local_map_index_matches_metric_box_and_centers():
    spec = WorldSpec(20.0, 20.0, 4.0, 0.5)
    xyz = np.array([
        [0, 0, 0], [2, 3, 1], [10, 10, 2], [20, 20, 3], [30, 4, 1]
    ], dtype=np.uint64)
    codes = linear_encode(xyz[:,0], xyz[:,1], xyz[:,2], spec)
    origin = (-10.0, -10.0, -1.0)
    idx = VoxelLocalMapIndex(codes, spec, origin, bucket_voxels=4)
    pts = idx.query_radius_centers(-4.75, -4.75, 1.0)
    assert len(pts) == 1
    # voxel (10,10,2) center
    assert np.allclose(pts[0], [-4.75, -4.75, 0.25])


def test_icp_3dof_recovers_controlled_pose():
    rng = np.random.default_rng(7)
    # Distinctive 3-D urban-like geometry: two walls plus scattered poles.
    y = np.linspace(-12, 12, 240)
    z = np.tile(np.array([-1.0, 0.0, 1.0, 2.0]), 60)
    wall1 = np.column_stack([np.full(240, 8.0), y, z])
    x = np.linspace(-10, 14, 240)
    wall2 = np.column_stack([x, np.full(240, -6.0), np.roll(z, 17)])
    poles = np.column_stack([
        rng.uniform(-8, 12, 180), rng.uniform(-5, 10, 180), rng.uniform(-1, 3, 180)
    ])
    map_pts = np.vstack([wall1, wall2, poles])

    gt = Pose3DOF(2.5, -1.5, math.radians(18.0))
    c, s = math.cos(gt.yaw), math.sin(gt.yaw)
    R = np.array([[c, -s], [s, c]])
    # world = body @ R.T + t -> body = (world-t) @ R
    body_xy = (map_pts[:, :2] - np.array([gt.x, gt.y])) @ R
    body = np.column_stack([body_xy, map_pts[:, 2]])
    # Use a subset to avoid exact one-to-one triviality while retaining overlap.
    body = body[::2]
    init = Pose3DOF(gt.x + 0.9, gt.y - 0.7, gt.yaw + math.radians(4.0))
    cfg = ICP3DOFConfig(
        max_iterations=40, max_correspondence_m=2.0, trim_fraction=.85,
        min_correspondences=100, max_step_translation_m=.7, max_step_yaw_deg=2.0,
    )
    res = scan_to_map_icp_3dof(body, map_pts, init, 0.0, 0.0, 0.0, cfg)
    te, ye = pose_error_3dof(res.pose, gt)
    assert te < 0.25, (res, te, ye)
    assert ye < 1.0, (res, te, ye)
    assert res.correspondences >= 100


def test_pose_error_wraps_yaw():
    a = Pose3DOF(0, 0, math.radians(179))
    b = Pose3DOF(0, 0, math.radians(-179))
    te, ye = pose_error_3dof(a, b)
    assert te == 0
    assert abs(ye - 2.0) < 1e-9
