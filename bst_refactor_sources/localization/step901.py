from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from localization.step90 import (
    Pose3DOF,
    VoxelLocalMapIndex,
    level_body_points,
    pose_error_3dof,
    transform_leveled,
    wrap_angle,
)

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class PointToPlaneICP3DOFConfig:
    max_iterations: int = 30
    max_correspondence_m: float = 2.0
    trim_fraction: float = 0.75
    min_correspondences: int = 80
    translation_tolerance_m: float = 0.01
    yaw_tolerance_deg: float = 0.05
    max_step_translation_m: float = 1.0
    max_step_yaw_deg: float = 3.0
    normal_k: int = 16
    normal_max_radius_m: float = 1.5
    normal_min_neighbors: int = 6
    normal_max_curvature: float = 0.20
    normal_min_horizontal_component: float = 0.20
    damping: float = 1e-4


@dataclass
class PointToPlaneICP3DOFResult:
    pose: Pose3DOF
    converged: bool
    iterations: int
    correspondences: int
    rmse_m: float
    match_ms: float
    failure_reason: str = ""
    normal_correspondences: int = 0
    mean_abs_plane_residual_m: float = float("inf")


def _limit_increment(delta: np.ndarray, cfg: PointToPlaneICP3DOFConfig) -> np.ndarray:
    d = np.asarray(delta, dtype=np.float64).copy()
    trans = math.hypot(float(d[0]), float(d[1]))
    if cfg.max_step_translation_m > 0 and trans > cfg.max_step_translation_m:
        d[:2] *= cfg.max_step_translation_m / trans
    max_yaw = math.radians(cfg.max_step_yaw_deg)
    if max_yaw > 0:
        d[2] = np.clip(d[2], -max_yaw, max_yaw)
    return d


def _estimate_normals_for_indices(
    map_points: np.ndarray,
    tree,
    indices: np.ndarray,
    cfg: PointToPlaneICP3DOFConfig,
    cache_normals: np.ndarray,
    cache_valid: np.ndarray,
) -> None:
    """Estimate PCA normals lazily for requested map-point indices.

    cache_valid values: 0 unknown, 1 valid normal, -1 rejected normal.
    """
    if len(indices) == 0:
        return
    idx = np.unique(np.asarray(indices, dtype=np.int64))
    todo = idx[cache_valid[idx] == 0]
    if len(todo) == 0:
        return

    k = max(int(cfg.normal_k), int(cfg.normal_min_neighbors))
    d, nn = tree.query(
        map_points[todo],
        k=k,
        distance_upper_bound=float(cfg.normal_max_radius_m),
        workers=1,
    )
    if k == 1:
        d = d[:, None]
        nn = nn[:, None]

    n_map = len(map_points)
    for row, map_idx in enumerate(todo):
        neigh = nn[row]
        dist = d[row]
        good = np.isfinite(dist) & (neigh < n_map)
        neigh = neigh[good]
        if len(neigh) < cfg.normal_min_neighbors:
            cache_valid[map_idx] = -1
            continue
        pts = map_points[neigh]
        center = pts.mean(axis=0)
        X = pts - center
        C = (X.T @ X) / max(1, len(X))
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            cache_valid[map_idx] = -1
            continue
        total = float(np.sum(eigvals))
        if total <= 1e-12:
            cache_valid[map_idx] = -1
            continue
        curvature = float(max(0.0, eigvals[0]) / total)
        normal = eigvecs[:, 0]
        h = float(math.hypot(float(normal[0]), float(normal[1])))
        if curvature > cfg.normal_max_curvature or h < cfg.normal_min_horizontal_component:
            cache_valid[map_idx] = -1
            continue
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        cache_normals[map_idx] = normal
        cache_valid[map_idx] = 1


def scan_to_map_point_to_plane_icp_3dof(
    points_body: np.ndarray,
    map_points_local: np.ndarray,
    initial_pose: Pose3DOF,
    fixed_z: float,
    fixed_roll: float,
    fixed_pitch: float,
    config: PointToPlaneICP3DOFConfig | None = None,
    map_tree=None,
) -> PointToPlaneICP3DOFResult:
    """3-DOF scan-to-map point-to-plane ICP over x/y/yaw.

    Map normals are estimated lazily from local voxel-center neighborhoods using
    PCA and cached for the duration of one localization trial. Nearly horizontal
    normals (ground/roof) are rejected because they provide almost no x/y/yaw
    constraint when z/roll/pitch are fixed.
    """
    if cKDTree is None:
        raise RuntimeError("Step 9.0.1 requires scipy.spatial.cKDTree") from _SCIPY_IMPORT_ERROR
    cfg = config or PointToPlaneICP3DOFConfig()
    pts = np.asarray(points_body, dtype=np.float64)
    mp = np.asarray(map_points_local, dtype=np.float64)
    if len(pts) == 0:
        return PointToPlaneICP3DOFResult(initial_pose, False, 0, 0, float("inf"), 0.0, "empty_scan")
    if len(mp) < cfg.min_correspondences:
        return PointToPlaneICP3DOFResult(initial_pose, False, 0, 0, float("inf"), 0.0, "local_map_too_small")

    tree = map_tree if map_tree is not None else cKDTree(mp)
    leveled = level_body_points(pts, fixed_roll, fixed_pitch)
    normals = np.zeros((len(mp), 3), dtype=np.float64)
    normal_valid = np.zeros(len(mp), dtype=np.int8)

    pose = initial_pose
    converged = False
    reason = "max_iterations"
    last_corr = 0
    last_normal_corr = 0
    last_rmse = float("inf")
    last_abs_plane = float("inf")
    t0 = time.perf_counter_ns()

    for it in range(1, cfg.max_iterations + 1):
        world = transform_leveled(leveled, pose, fixed_z)
        dist, idx = tree.query(world, k=1, workers=1)
        keep = np.isfinite(dist) & (dist <= cfg.max_correspondence_m)
        ids = np.flatnonzero(keep)
        if len(ids) < cfg.min_correspondences:
            reason = "too_few_correspondences"
            last_corr = int(len(ids))
            break

        # Estimate normals only where the current correspondences need them.
        _estimate_normals_for_indices(mp, tree, idx[ids], cfg, normals, normal_valid)
        valid_normal = normal_valid[idx[ids]] == 1
        ids = ids[valid_normal]
        if len(ids) < cfg.min_correspondences:
            reason = "too_few_normal_correspondences"
            last_corr = int(np.count_nonzero(keep))
            last_normal_corr = int(len(ids))
            break

        target_idx = idx[ids]
        targets = mp[target_idx]
        ns = normals[target_idx]
        residual = np.einsum("ij,ij->i", ns, world[ids] - targets)

        if 0.0 < cfg.trim_fraction < 1.0 and len(ids) > cfg.min_correspondences:
            take = max(cfg.min_correspondences, int(math.ceil(len(ids) * cfg.trim_fraction)))
            sel = np.argpartition(np.abs(residual), take - 1)[:take]
            ids = ids[sel]
            target_idx = target_idx[sel]
            targets = targets[sel]
            ns = ns[sel]
            residual = residual[sel]

        # Linearize point-to-plane residual wrt an incremental world-frame
        # translation and yaw update at the current pose.
        q = leveled[ids]
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        dpx_dyaw = -s * q[:, 0] - c * q[:, 1]
        dpy_dyaw =  c * q[:, 0] - s * q[:, 1]
        J = np.empty((len(ids), 3), dtype=np.float64)
        J[:, 0] = ns[:, 0]
        J[:, 1] = ns[:, 1]
        J[:, 2] = ns[:, 0] * dpx_dyaw + ns[:, 1] * dpy_dyaw

        A = J.T @ J
        b = -(J.T @ residual)
        A.flat[::4] += float(cfg.damping)
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(A, b, rcond=None)
        if not np.all(np.isfinite(delta)):
            reason = "nonfinite_update"
            break
        delta = _limit_increment(delta, cfg)

        new_pose = Pose3DOF(
            pose.x + float(delta[0]),
            pose.y + float(delta[1]),
            wrap_angle(pose.yaw + float(delta[2])),
        )
        dtrans = math.hypot(float(delta[0]), float(delta[1]))
        dyaw_deg = abs(math.degrees(float(delta[2])))
        last_corr = int(np.count_nonzero(keep))
        last_normal_corr = int(len(ids))
        last_rmse = float(math.sqrt(np.mean(np.square(residual))))
        last_abs_plane = float(np.mean(np.abs(residual)))
        pose = new_pose

        if dtrans <= cfg.translation_tolerance_m and dyaw_deg <= cfg.yaw_tolerance_deg:
            converged = True
            reason = ""
            break

    match_ms = (time.perf_counter_ns() - t0) / 1e6
    return PointToPlaneICP3DOFResult(
        pose=pose,
        converged=converged,
        iterations=it if "it" in locals() else 0,
        correspondences=last_corr,
        rmse_m=last_rmse,
        match_ms=match_ms,
        failure_reason=reason,
        normal_correspondences=last_normal_corr,
        mean_abs_plane_residual_m=last_abs_plane,
    )


__all__ = [
    "Pose3DOF",
    "VoxelLocalMapIndex",
    "PointToPlaneICP3DOFConfig",
    "PointToPlaneICP3DOFResult",
    "scan_to_map_point_to_plane_icp_3dof",
    "pose_error_3dof",
]
