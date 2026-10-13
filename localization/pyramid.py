# Consolidated localization module (moved without re-tuning from step902.py).
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from localization.pose import Pose3DOF, pose_error_3dof
from localization.point_to_plane import (
    PointToPlaneICP3DOFConfig,
    PointToPlaneICP3DOFResult,
    scan_to_map_point_to_plane_icp_3dof,
)

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class PyramidLevelConfig:
    name: str
    scan_voxel_m: float
    map_voxel_m: float
    max_correspondence_m: float
    max_iterations: int
    trim_fraction: float
    normal_max_radius_m: float
    max_step_translation_m: float
    max_step_yaw_deg: float


DEFAULT_PYRAMID: tuple[PyramidLevelConfig, ...] = (
    PyramidLevelConfig(
        name="coarse",
        scan_voxel_m=1.50,
        map_voxel_m=1.00,
        max_correspondence_m=4.00,
        max_iterations=15,
        trim_fraction=0.60,
        normal_max_radius_m=2.50,
        max_step_translation_m=1.50,
        max_step_yaw_deg=5.0,
    ),
    PyramidLevelConfig(
        name="medium",
        scan_voxel_m=1.00,
        map_voxel_m=0.75,
        max_correspondence_m=2.00,
        max_iterations=15,
        trim_fraction=0.70,
        normal_max_radius_m=2.00,
        max_step_translation_m=1.00,
        max_step_yaw_deg=3.0,
    ),
    PyramidLevelConfig(
        name="fine",
        scan_voxel_m=0.75,
        map_voxel_m=0.50,
        max_correspondence_m=1.00,
        max_iterations=20,
        trim_fraction=0.75,
        normal_max_radius_m=1.50,
        max_step_translation_m=0.50,
        max_step_yaw_deg=1.5,
    ),
)


@dataclass
class PreparedPyramidLevel:
    config: PyramidLevelConfig
    map_points: np.ndarray
    tree: object


@dataclass
class CoarseToFinePointToPlaneResult:
    pose: Pose3DOF
    converged: bool
    iterations: int
    correspondences: int
    normal_correspondences: int
    rmse_m: float
    mean_abs_plane_residual_m: float
    match_ms: float
    failure_reason: str
    level_results: list[tuple[str, PointToPlaneICP3DOFResult]]


def voxel_downsample_first(points: np.ndarray, voxel_m: float) -> np.ndarray:
    """Deterministic first-point voxel downsampling in the point coordinate frame."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0 or voxel_m <= 0:
        return pts
    keys = np.floor(pts / float(voxel_m)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def parse_pyramid(spec: str | None) -> tuple[PyramidLevelConfig, ...]:
    """Parse `name:scan:map:corr:iters,...`.

    Only scale/correspondence fields are encoded in the CLI string; robust
    defaults for trimming, normal radius and update limits are inferred by
    level order. Passing None/empty returns DEFAULT_PYRAMID.
    """
    if spec is None or not str(spec).strip():
        return DEFAULT_PYRAMID
    raw = [x.strip() for x in str(spec).split(",") if x.strip()]
    if not raw:
        return DEFAULT_PYRAMID
    out: list[PyramidLevelConfig] = []
    n = len(raw)
    for i, token in enumerate(raw):
        p = token.split(":")
        if len(p) != 5:
            raise ValueError("pyramid must be name:scan_voxel:map_voxel:max_corr:iterations comma-separated")
        name = p[0]
        scan_v, map_v, corr = map(float, p[1:4])
        iters = int(p[4])
        # Interpolate conservative-to-fine defaults by normalized level order.
        u = 0.0 if n == 1 else i / (n - 1)
        trim = 0.60 + 0.15 * u
        normal_r = max(1.5, 2.5 - 1.0 * u)
        max_trans = 1.5 - 1.0 * u
        max_yaw = 5.0 - 3.5 * u
        out.append(PyramidLevelConfig(
            name=name,
            scan_voxel_m=scan_v,
            map_voxel_m=map_v,
            max_correspondence_m=corr,
            max_iterations=iters,
            trim_fraction=trim,
            normal_max_radius_m=normal_r,
            max_step_translation_m=max_trans,
            max_step_yaw_deg=max_yaw,
        ))
    return tuple(out)


def prepare_local_map_pyramid(
    map_points_local: np.ndarray,
    levels: tuple[PyramidLevelConfig, ...] = DEFAULT_PYRAMID,
) -> list[PreparedPyramidLevel]:
    if cKDTree is None:
        raise RuntimeError("Step 9.0.2 requires scipy.spatial.cKDTree") from _SCIPY_IMPORT_ERROR
    mp = np.asarray(map_points_local, dtype=np.float64)
    prepared: list[PreparedPyramidLevel] = []
    for level in levels:
        pts = voxel_downsample_first(mp, level.map_voxel_m)
        tree = cKDTree(pts) if len(pts) else None
        prepared.append(PreparedPyramidLevel(level, pts, tree))
    return prepared


def scan_to_map_coarse_to_fine_point_to_plane_3dof(
    points_body: np.ndarray,
    prepared_levels: list[PreparedPyramidLevel],
    initial_pose: Pose3DOF,
    fixed_z: float,
    fixed_roll: float,
    fixed_pitch: float,
    base_config: PointToPlaneICP3DOFConfig | None = None,
) -> CoarseToFinePointToPlaneResult:
    """Run point-to-plane ICP from coarse to fine over x/y/yaw.

    Each level uses a separately downsampled scan/map and its own correspondence
    radius. The final pose of one level initializes the next. Map normals remain
    lazy and level-local via the Step-9.0.1 matcher.
    """
    cfg0 = base_config or PointToPlaneICP3DOFConfig()
    pts = np.asarray(points_body, dtype=np.float64)
    pose = initial_pose
    level_results: list[tuple[str, PointToPlaneICP3DOFResult]] = []
    total_iterations = 0
    total_match_ms = 0.0
    final: PointToPlaneICP3DOFResult | None = None

    for prepared in prepared_levels:
        lev = prepared.config
        scan = voxel_downsample_first(pts, lev.scan_voxel_m)
        if prepared.tree is None or len(prepared.map_points) < cfg0.min_correspondences:
            final = PointToPlaneICP3DOFResult(
                pose, False, 0, 0, float("inf"), 0.0, "local_map_too_small"
            )
            level_results.append((lev.name, final))
            break
        cfg = replace(
            cfg0,
            max_iterations=int(lev.max_iterations),
            max_correspondence_m=float(lev.max_correspondence_m),
            trim_fraction=float(lev.trim_fraction),
            normal_max_radius_m=float(lev.normal_max_radius_m),
            max_step_translation_m=float(lev.max_step_translation_m),
            max_step_yaw_deg=float(lev.max_step_yaw_deg),
        )
        res = scan_to_map_point_to_plane_icp_3dof(
            scan,
            prepared.map_points,
            pose,
            fixed_z,
            fixed_roll,
            fixed_pitch,
            cfg,
            map_tree=prepared.tree,
        )
        level_results.append((lev.name, res))
        total_iterations += int(res.iterations)
        total_match_ms += float(res.match_ms)
        pose = res.pose
        final = res
        # A coarse level with too few usable constraints cannot safely seed a
        # finer one. max_iterations is not fatal: the finer stage may refine.
        if res.failure_reason in {
            "empty_scan", "local_map_too_small", "too_few_correspondences",
            "too_few_normal_correspondences", "nonfinite_update",
        }:
            break

    if final is None:
        final = PointToPlaneICP3DOFResult(initial_pose, False, 0, 0, float("inf"), 0.0, "no_levels")
    completed_all = len(level_results) == len(prepared_levels)
    # For a pyramid, reaching the fine stage is more important than the exact
    # stopping flag of earlier levels. The final-level convergence flag is kept.
    reason = final.failure_reason
    return CoarseToFinePointToPlaneResult(
        pose=pose,
        converged=bool(final.converged and completed_all),
        iterations=total_iterations,
        correspondences=int(final.correspondences),
        normal_correspondences=int(final.normal_correspondences),
        rmse_m=float(final.rmse_m),
        mean_abs_plane_residual_m=float(final.mean_abs_plane_residual_m),
        match_ms=total_match_ms,
        failure_reason=reason,
        level_results=level_results,
    )


__all__ = [
    "Pose3DOF",
    "PointToPlaneICP3DOFConfig",
    "PyramidLevelConfig",
    "PreparedPyramidLevel",
    "CoarseToFinePointToPlaneResult",
    "DEFAULT_PYRAMID",
    "parse_pyramid",
    "prepare_local_map_pyramid",
    "scan_to_map_coarse_to_fine_point_to_plane_3dof",
    "pose_error_3dof",
    "voxel_downsample_first",
]
