import math
import numpy as np

from localization.step90 import Pose3DOF, pose_error_3dof
from localization.step901 import (
    PointToPlaneICP3DOFConfig,
    scan_to_map_point_to_plane_icp_3dof,
)


def _grid_wall_x(x0, ys, zs):
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    return np.column_stack([np.full(Y.size, x0), Y.ravel(), Z.ravel()])


def _grid_wall_y(y0, xs, zs):
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return np.column_stack([X.ravel(), np.full(X.size, y0), Z.ravel()])


def test_point_to_plane_recovers_planar_pose():
    ys = np.arange(-12.0, 12.01, 0.5)
    xs = np.arange(-10.0, 14.01, 0.5)
    zs = np.arange(-1.0, 3.01, 0.5)
    wall1 = _grid_wall_x(8.0, ys, zs)
    wall2 = _grid_wall_y(-6.0, xs, zs)
    # A third oblique-ish structure made from a vertical wall segment helps yaw.
    wall3 = _grid_wall_x(-4.0, np.arange(-2.0, 8.01, 0.5), zs)
    map_pts = np.vstack([wall1, wall2, wall3])

    gt = Pose3DOF(2.5, -1.5, math.radians(18.0))
    c, s = math.cos(gt.yaw), math.sin(gt.yaw)
    R = np.array([[c, -s], [s, c]])
    body_xy = (map_pts[:, :2] - np.array([gt.x, gt.y])) @ R
    body = np.column_stack([body_xy, map_pts[:, 2]])
    body = body[::3]

    init = Pose3DOF(gt.x + 1.0, gt.y - 0.8, gt.yaw + math.radians(4.0))
    cfg = PointToPlaneICP3DOFConfig(
        max_iterations=40,
        max_correspondence_m=2.5,
        trim_fraction=.85,
        min_correspondences=100,
        max_step_translation_m=.75,
        max_step_yaw_deg=2.5,
        normal_k=12,
        normal_max_radius_m=1.25,
        normal_min_neighbors=6,
        normal_max_curvature=.15,
        normal_min_horizontal_component=.5,
    )
    res = scan_to_map_point_to_plane_icp_3dof(body, map_pts, init, 0.0, 0.0, 0.0, cfg)
    te, ye = pose_error_3dof(res.pose, gt)
    assert res.normal_correspondences >= 100, res
    assert te < 0.15, (res, te, ye)
    assert ye < 0.5, (res, te, ye)


def test_point_to_plane_rejects_ground_only_for_planar_motion():
    xs = np.arange(-5.0, 5.01, 0.5)
    ys = np.arange(-5.0, 5.01, 0.5)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ground = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    cfg = PointToPlaneICP3DOFConfig(
        min_correspondences=20,
        normal_k=12,
        normal_max_radius_m=1.25,
        normal_min_neighbors=6,
        normal_min_horizontal_component=.3,
    )
    res = scan_to_map_point_to_plane_icp_3dof(
        ground[::2], ground, Pose3DOF(.5, .5, math.radians(2)), 0.0, 0.0, 0.0, cfg
    )
    assert not res.converged
    assert res.failure_reason == "too_few_normal_correspondences"
