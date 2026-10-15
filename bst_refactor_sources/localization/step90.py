from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np

from realdata.nclt_packed import linear_decode
from synthetic.world import WorldSpec

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover - exercised only when dependency missing
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


def wrap_angle(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose3DOF:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ICP3DOFConfig:
    max_iterations: int = 30
    max_correspondence_m: float = 2.0
    trim_fraction: float = 0.75
    min_correspondences: int = 80
    translation_tolerance_m: float = 0.01
    yaw_tolerance_deg: float = 0.05
    max_step_translation_m: float = 1.0
    max_step_yaw_deg: float = 3.0


@dataclass
class ICP3DOFResult:
    pose: Pose3DOF
    converged: bool
    iterations: int
    correspondences: int
    rmse_m: float
    match_ms: float
    failure_reason: str = ""


def _rot_rp(roll: float, pitch: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Ry(pitch) @ Rx(roll). Yaw is applied separately in XY.
    return np.array([
        [cp, sp * sr, sp * cr],
        [0.0, cr, -sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def level_body_points(points_body: np.ndarray, roll: float, pitch: float) -> np.ndarray:
    pts = np.asarray(points_body, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_body must be Nx3")
    return pts @ _rot_rp(float(roll), float(pitch)).T


def transform_leveled(points_leveled: np.ndarray, pose: Pose3DOF, z: float) -> np.ndarray:
    q = np.asarray(points_leveled, dtype=np.float64)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    out = np.empty_like(q)
    out[:, 0] = c * q[:, 0] - s * q[:, 1] + pose.x
    out[:, 1] = s * q[:, 0] + c * q[:, 1] + pose.y
    out[:, 2] = q[:, 2] + float(z)
    return out


def _estimate_planar_transform(src_xy: np.ndarray, dst_xy: np.ndarray) -> Pose3DOF:
    """Absolute least-squares SE(2) transform mapping src_xy -> dst_xy."""
    if len(src_xy) < 2:
        raise ValueError("Need >=2 correspondences")
    cs = src_xy.mean(axis=0)
    cd = dst_xy.mean(axis=0)
    X = src_xy - cs
    Y = dst_xy - cd
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Row-vector convention: dst ~= src @ R.T + t.
    t = cd - cs @ R.T
    yaw = math.atan2(R[1, 0], R[0, 0])
    return Pose3DOF(float(t[0]), float(t[1]), wrap_angle(yaw))


def _limit_update(current: Pose3DOF, target: Pose3DOF, cfg: ICP3DOFConfig) -> Pose3DOF:
    dx = target.x - current.x
    dy = target.y - current.y
    d = math.hypot(dx, dy)
    if d > cfg.max_step_translation_m > 0:
        scale = cfg.max_step_translation_m / d
        dx *= scale; dy *= scale
    dyaw = wrap_angle(target.yaw - current.yaw)
    max_yaw = math.radians(cfg.max_step_yaw_deg)
    if max_yaw > 0:
        dyaw = max(-max_yaw, min(max_yaw, dyaw))
    return Pose3DOF(current.x + dx, current.y + dy, wrap_angle(current.yaw + dyaw))


def scan_to_map_icp_3dof(
    points_body: np.ndarray,
    map_points_local: np.ndarray,
    initial_pose: Pose3DOF,
    fixed_z: float,
    fixed_roll: float,
    fixed_pitch: float,
    config: ICP3DOFConfig | None = None,
    map_tree=None,
) -> ICP3DOFResult:
    """Point-to-point scan-to-map ICP over x/y/yaw.

    z, roll and pitch are held at their reference values in Step 9.0. This is a
    harness-validation localizer, not yet the final native-block matcher.
    Correspondence search is 3-D; the least-squares update is planar SE(2).
    """
    if cKDTree is None:
        raise RuntimeError("Step 9.0 requires scipy (scipy.spatial.cKDTree)") from _SCIPY_IMPORT_ERROR
    cfg = config or ICP3DOFConfig()
    pts = np.asarray(points_body, dtype=np.float64)
    mp = np.asarray(map_points_local, dtype=np.float64)
    if len(pts) == 0:
        return ICP3DOFResult(initial_pose, False, 0, 0, float("inf"), 0.0, "empty_scan")
    if len(mp) < cfg.min_correspondences:
        return ICP3DOFResult(initial_pose, False, 0, 0, float("inf"), 0.0, "local_map_too_small")

    leveled = level_body_points(pts, fixed_roll, fixed_pitch)
    tree = map_tree if map_tree is not None else cKDTree(mp)
    pose = initial_pose
    converged = False
    reason = "max_iterations"
    last_corr = 0
    last_rmse = float("inf")
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

        if 0.0 < cfg.trim_fraction < 1.0 and len(ids) > cfg.min_correspondences:
            take = max(cfg.min_correspondences, int(math.ceil(len(ids) * cfg.trim_fraction)))
            # argpartition avoids globally sorting thousands of distances.
            loc = np.argpartition(dist[ids], take - 1)[:take]
            ids = ids[loc]

        targets = mp[idx[ids]]
        target_pose = _estimate_planar_transform(leveled[ids, :2], targets[:, :2])
        new_pose = _limit_update(pose, target_pose, cfg)
        dtrans = math.hypot(new_pose.x - pose.x, new_pose.y - pose.y)
        dyaw = abs(math.degrees(wrap_angle(new_pose.yaw - pose.yaw)))
        last_corr = int(len(ids))
        last_rmse = float(math.sqrt(np.mean(np.square(dist[ids]))))
        pose = new_pose
        if dtrans <= cfg.translation_tolerance_m and dyaw <= cfg.yaw_tolerance_deg:
            converged = True
            reason = ""
            break

    match_ms = (time.perf_counter_ns() - t0) / 1e6
    return ICP3DOFResult(pose, converged, it if 'it' in locals() else 0, last_corr, last_rmse, match_ms, reason)


def pose_error_3dof(estimate: Pose3DOF, reference: Pose3DOF) -> tuple[float, float]:
    trans = math.hypot(estimate.x - reference.x, estimate.y - reference.y)
    yaw_deg = abs(math.degrees(wrap_angle(estimate.yaw - reference.yaw)))
    return float(trans), float(yaw_deg)


class VoxelLocalMapIndex:
    """Counted, conventional coarse-bucket index over one voxel map session.

    Step 9.0 intentionally uses a conventional voxel map. This class is not a
    BST component. It returns voxel centers in the native NCLT local frame.
    """

    def __init__(
        self,
        linear_codes: np.ndarray,
        spec: WorldSpec,
        origin_local_m: tuple[float, float, float],
        bucket_voxels: int = 16,
    ):
        self.codes = np.asarray(linear_codes, dtype=np.uint64)
        self.spec = spec
        self.origin = np.asarray(origin_local_m, dtype=np.float64)
        self.bucket_voxels = int(bucket_voxels)
        if self.bucket_voxels <= 0:
            raise ValueError("bucket_voxels must be >0")
        x, y, _ = linear_decode(self.codes, spec)
        bx = x.astype(np.int64) // self.bucket_voxels
        by = y.astype(np.int64) // self.bucket_voxels
        self.nby = int(math.ceil(spec.ny / self.bucket_voxels))
        key = bx * self.nby + by
        order = np.argsort(key, kind="stable")
        self._keys_per_code = key[order]
        self._codes_by_bucket = self.codes[order]
        self.bucket_keys, self.bucket_starts, self.bucket_counts = np.unique(
            self._keys_per_code, return_index=True, return_counts=True
        )

    def query_box(self, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
        vs = self.spec.voxel_size_m
        # Metric local coordinates -> grid half-open integer bounds.
        gx0 = max(0, int(math.floor((xmin - self.origin[0]) / vs)))
        gx1 = min(self.spec.nx, int(math.ceil((xmax - self.origin[0]) / vs)))
        gy0 = max(0, int(math.floor((ymin - self.origin[1]) / vs)))
        gy1 = min(self.spec.ny, int(math.ceil((ymax - self.origin[1]) / vs)))
        if gx0 >= gx1 or gy0 >= gy1:
            return np.empty(0, dtype=np.uint64)
        bx0, bx1 = gx0 // self.bucket_voxels, (gx1 - 1) // self.bucket_voxels
        by0, by1 = gy0 // self.bucket_voxels, (gy1 - 1) // self.bucket_voxels
        parts = []
        for bx in range(bx0, bx1 + 1):
            for by in range(by0, by1 + 1):
                k = bx * self.nby + by
                j = int(np.searchsorted(self.bucket_keys, k))
                if j < len(self.bucket_keys) and int(self.bucket_keys[j]) == k:
                    a = int(self.bucket_starts[j]); b = a + int(self.bucket_counts[j])
                    parts.append(self._codes_by_bucket[a:b])
        if not parts:
            return np.empty(0, dtype=np.uint64)
        cand = np.concatenate(parts) if len(parts) > 1 else parts[0]
        x, y, _ = linear_decode(cand, self.spec)
        keep = (x >= gx0) & (x < gx1) & (y >= gy0) & (y < gy1)
        return np.asarray(cand[keep], dtype=np.uint64)

    def query_radius_centers(self, cx: float, cy: float, radius_m: float) -> np.ndarray:
        codes = self.query_box(cx - radius_m, cx + radius_m, cy - radius_m, cy + radius_m)
        if not len(codes):
            return np.empty((0, 3), dtype=np.float64)
        x, y, z = linear_decode(codes, self.spec)
        vs = self.spec.voxel_size_m
        pts = np.column_stack((x, y, z)).astype(np.float64)
        pts = self.origin + (pts + 0.5) * vs
        # Circular crop after box/bucket retrieval.
        r2 = np.square(pts[:, 0] - cx) + np.square(pts[:, 1] - cy)
        return pts[r2 <= radius_m * radius_m]
