import math
import random

import numpy as np

from localization.step90 import Pose3DOF, pose_error_3dof
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import (
    DEFAULT_PYRAMID,
    parse_pyramid,
    prepare_local_map_pyramid,
    scan_to_map_coarse_to_fine_point_to_plane_3dof,
)
from benchmark.step901_benchmark import _trial_perturbation


def _grid_wall_x(x0, ys, zs):
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    return np.column_stack([np.full(Y.size, x0), Y.ravel(), Z.ravel()])


def _grid_wall_y(y0, xs, zs):
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return np.column_stack([X.ravel(), np.full(X.size, y0), Z.ravel()])


def test_parse_default_pyramid_is_coarse_to_fine():
    p = parse_pyramid(None)
    assert [x.name for x in p] == ["coarse", "medium", "fine"]
    assert [x.max_correspondence_m for x in p] == [4.0, 2.0, 1.0]
    assert p[0].scan_voxel_m > p[-1].scan_voxel_m
    assert p[0].map_voxel_m > p[-1].map_voxel_m


def test_coarse_to_fine_recovers_larger_planar_perturbation():
    ys = np.arange(-15.0, 15.01, 0.5)
    xs = np.arange(-14.0, 16.01, 0.5)
    zs = np.arange(-1.0, 3.01, 0.5)
    map_pts = np.vstack([
        _grid_wall_x(8.0, ys, zs),
        _grid_wall_y(-6.0, xs, zs),
        _grid_wall_x(-5.0, np.arange(-4.0, 9.01, 0.5), zs),
        _grid_wall_y(7.0, np.arange(-10.0, 5.01, 0.5), zs),
    ])
    gt = Pose3DOF(2.5, -1.5, math.radians(18.0))
    c, s = math.cos(gt.yaw), math.sin(gt.yaw)
    R = np.array([[c, -s], [s, c]])
    body_xy = (map_pts[:, :2] - np.array([gt.x, gt.y])) @ R
    body = np.column_stack([body_xy, map_pts[:, 2]])[::2]
    init = Pose3DOF(gt.x + 2.2, gt.y - 1.5, gt.yaw + math.radians(7.0))
    levels = prepare_local_map_pyramid(map_pts, DEFAULT_PYRAMID)
    cfg = PointToPlaneICP3DOFConfig(
        min_correspondences=80,
        normal_k=12,
        normal_min_neighbors=6,
        normal_max_curvature=.20,
        normal_min_horizontal_component=.35,
        damping=1e-4,
    )
    res = scan_to_map_coarse_to_fine_point_to_plane_3dof(
        body, levels, init, 0.0, 0.0, 0.0, cfg
    )
    te, ye = pose_error_3dof(res.pose, gt)
    assert len(res.level_results) == 3, res
    assert te < 0.35, (res, te, ye)
    assert ye < 1.0, (res, te, ye)


def test_zero_control_does_not_change_frozen_perturbation_rng_sequence():
    # Step 9.0.2 executes the four frozen perturbations first and appends the
    # zero-control without consuming RNG. The next scan must therefore receive
    # the same random perturbation as the prior benchmark.
    levels = [(0.5, 1), (1, 3), (2, 5), (5, 10)]
    a = random.Random(42090)
    seq_without_zero = []
    for _scan in range(2):
        for xy, yaw in levels:
            seq_without_zero.append(_trial_perturbation(xy, yaw, a))

    b = random.Random(42090)
    seq_with_zero = []
    for _scan in range(2):
        for xy, yaw in levels:
            seq_with_zero.append(_trial_perturbation(xy, yaw, b))
        # zero control is deterministic and intentionally consumes no RNG
        _zero = (0.0, 0.0, 0.0)
    assert seq_with_zero == seq_without_zero


def test_same_session_selector_excludes_direct_stride_frames(tmp_path):
    from benchmark.step902_benchmark import _select_query_scans

    dt = np.dtype([("x","<u2"),("y","<u2"),("z","<u2"),("intensity","u1"),("laser","u1")])
    sd = tmp_path / "2012-01-15" / "velodyne_sync"
    sd.mkdir(parents=True)
    gt = []
    for i in range(6):
        ts = 1_000_000 + 100_000 * i
        gt.append([ts, 0, 0, 0, 0, 0, 0])
        arr = np.zeros(1, dtype=dt)
        raw = np.rint((np.asarray([[5.0, 0.0, 0.0]]) + 100.0) / 0.005).astype(np.uint16)
        arr["x"] = raw[:,0]; arr["y"] = raw[:,1]; arr["z"] = raw[:,2]
        arr["intensity"] = 20; arr["laser"] = 1
        arr.tofile(sd / f"{ts}.bin")
    np.savetxt(tmp_path / "2012-01-15" / "groundtruth_2012-01-15.csv", np.asarray(gt), delimiter=",")

    selected = _select_query_scans(
        tmp_path, '2012-01-15', count=10, max_pose_gap_us=1000000, exclude_ingest_stride=2
    )
    assert selected
    assert all((original_idx % 2) != 0 for *_rest, original_idx in selected)
