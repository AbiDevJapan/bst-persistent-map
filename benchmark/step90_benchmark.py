from __future__ import annotations

import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from localization.step90 import (
    VoxelLocalMapIndex,
    ICP3DOFConfig,
    Pose3DOF,
    pose_error_3dof,
    scan_to_map_icp_3dof,
)
from realdata.nclt import (
    NCLTIngestConfig,
    _filter_body_points,
    _list_scan_files,
    _timestamp_from_scan,
    discover_nclt_session,
    load_ground_truth,
    read_velodyne_sync_bin,
)
from realdata.nclt_packed import PackedNCLTPrepared, _nearest_pose_rows


def _write_csv(path: Path, rows: list[dict]):
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _pctl(vals, q):
    if not vals:
        return float("nan")
    s = np.sort(np.asarray(vals, dtype=float))
    return float(np.quantile(s, q))


def _voxel_downsample_first(points: np.ndarray, voxel_m: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0 or voxel_m <= 0:
        return pts
    keys = np.floor(pts / voxel_m).astype(np.int32)
    # np.unique sorts keys, but return_index keeps one original point per cell.
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def _cap_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    # Deterministic uniform subsampling across file order.
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[idx]


def _select_query_scans(
    data_root: str | Path,
    session: str,
    count: int,
    max_pose_gap_us: int,
):
    paths = discover_nclt_session(data_root, session)
    scans = _list_scan_files(paths.scan_dir)
    gt = load_ground_truth(paths.gt_csv)
    ts = np.asarray([_timestamp_from_scan(p) for p in scans], dtype=np.int64)
    pose_rows, gaps = _nearest_pose_rows(gt, ts)
    valid = np.flatnonzero(gaps <= int(max_pose_gap_us))
    if not len(valid):
        raise RuntimeError(f"No query scans in {session} pass max pose gap {max_pose_gap_us} us")
    n = min(int(count), len(valid))
    take = valid[np.linspace(0, len(valid) - 1, n, dtype=np.int64)]
    return [(scans[i], pose_rows[i], int(gaps[i])) for i in take]


def parse_perturbations(spec: str) -> list[tuple[str, float, float]]:
    """Parse `0.5:1,1:3,...` as (label, xy_magnitude, yaw_deg)."""
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError("perturbations must be xy_m:yaw_deg comma-separated, e.g. 0.5:1,1:3")
        xy, yaw = float(parts[0]), float(parts[1])
        out.append((f"xy{xy:g}_yaw{yaw:g}", xy, yaw))
    if not out:
        raise ValueError("At least one perturbation level is required")
    return out


def _trial_perturbation(xy_m: float, yaw_deg: float, rng: random.Random):
    angle = rng.uniform(-math.pi, math.pi)
    dx = xy_m * math.cos(angle)
    dy = xy_m * math.sin(angle)
    signed_yaw = yaw_deg if rng.random() < 0.5 else -yaw_deg
    return dx, dy, signed_yaw


def _summaries(rows: list[dict]) -> list[dict]:
    groups = {}
    for r in rows:
        groups.setdefault(r["perturbation"], []).append(r)
    out = []
    for label, g in groups.items():
        def vals(k): return [float(x[k]) for x in g]
        successes = [bool(x["success"]) for x in g]
        converged = [bool(x["converged"]) for x in g]
        out.append({
            "perturbation": label,
            "trials": len(g),
            "success_rate": float(np.mean(successes)),
            "optimizer_convergence_rate": float(np.mean(converged)),
            "initial_translation_error_m": statistics.mean(vals("initial_translation_error_m")),
            "initial_yaw_error_deg": statistics.mean(vals("initial_yaw_error_deg")),
            "median_final_translation_error_m": statistics.median(vals("final_translation_error_m")),
            "p95_final_translation_error_m": _pctl(vals("final_translation_error_m"), .95),
            "median_final_yaw_error_deg": statistics.median(vals("final_yaw_error_deg")),
            "p95_final_yaw_error_deg": _pctl(vals("final_yaw_error_deg"), .95),
            "median_retrieval_ms": statistics.median(vals("retrieval_ms")),
            "median_map_prep_ms": statistics.median(vals("map_prep_ms")),
            "median_match_ms": statistics.median(vals("match_ms")),
            "median_total_ms": statistics.median(vals("total_ms")),
            "p95_total_ms": _pctl(vals("total_ms"), .95),
            "mean_query_points": statistics.mean(vals("query_points")),
            "mean_local_map_voxels": statistics.mean(vals("local_map_voxels")),
            "mean_final_correspondences": statistics.mean(vals("correspondences")),
            "median_final_rmse_m": statistics.median(vals("final_rmse_m")),
        })
    return sorted(out, key=lambda x: x["initial_translation_error_m"])


def benchmark_step90(
    prepared: PackedNCLTPrepared,
    data_root: str | Path,
    output_dir: str | Path,
    map_session: str,
    query_session: str,
    query_count: int = 20,
    perturbations: str = "0.5:1,1:3,2:5,5:10",
    perturb_seed: int = 42090,
    localization_max_range_m: float = 30.0,
    scan_downsample_m: float = 0.75,
    max_scan_points: int = 8000,
    local_map_radius_m: float = 40.0,
    map_bucket_voxels: int = 16,
    success_translation_m: float = 0.5,
    success_yaw_deg: float = 1.0,
    icp_config: ICP3DOFConfig | None = None,
):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if map_session == query_session:
        raise ValueError("Step 9.0 defaults to cross-session localization; use distinct map/query sessions")
    if prepared.session_names[:2] != [map_session, query_session]:
        raise ValueError("prepared sessions must be [map_session, query_session] in that order")

    cfg = icp_config or ICP3DOFConfig()
    print(f"[9.0] building conventional voxel local-map index from {map_session}", flush=True)
    t0 = time.perf_counter_ns()
    map_index = VoxelLocalMapIndex(
        prepared.session_codes[0], prepared.spec, prepared.origin_local_m, bucket_voxels=map_bucket_voxels
    )
    map_index_build_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"[9.0] map voxels={len(prepared.session_codes[0]):,}; index={map_index_build_ms:.1f} ms", flush=True)

    selected = _select_query_scans(data_root, query_session, query_count, prepared.config.max_pose_gap_us)
    levels = parse_perturbations(perturbations)
    rng = random.Random(int(perturb_seed))
    trial_rows = []
    scan_rows = []

    for si, (scan_path, gt, gap) in enumerate(selected, start=1):
        body = read_velodyne_sync_bin(scan_path)
        body = _filter_body_points(body, prepared.config)
        if len(body):
            r = np.linalg.norm(body, axis=1)
            body = body[r <= float(localization_max_range_m)]
        raw_after_range = len(body)
        body = _voxel_downsample_first(body, scan_downsample_m)
        body = _cap_points(body, max_scan_points)

        _, gx, gy, gz, roll, pitch, gyaw = [float(v) for v in gt]
        gt_pose = Pose3DOF(gx, gy, gyaw)
        scan_rows.append({
            "scan_index": si - 1,
            "scan_file": scan_path.name,
            "timestamp_us": int(_timestamp_from_scan(scan_path)),
            "pose_gap_us": int(gap),
            "points_after_range": int(raw_after_range),
            "points_used": int(len(body)),
            "gt_x": gx, "gt_y": gy, "gt_z": gz,
            "gt_roll": roll, "gt_pitch": pitch, "gt_yaw": gyaw,
        })
        print(f"[9.0] scan {si}/{len(selected)} {scan_path.name}: {len(body):,} query points", flush=True)

        for label, xy_mag, yaw_mag in levels:
            dx, dy, dyaw_deg = _trial_perturbation(xy_mag, yaw_mag, rng)
            init = Pose3DOF(gx + dx, gy + dy, gyaw + math.radians(dyaw_deg))
            init_te, init_ye = pose_error_3dof(init, gt_pose)

            total0 = time.perf_counter_ns()
            tr0 = time.perf_counter_ns()
            local_map = map_index.query_radius_centers(init.x, init.y, local_map_radius_m)
            retrieval_ms = (time.perf_counter_ns() - tr0) / 1e6

            tp0 = time.perf_counter_ns()
            tree = cKDTree(local_map) if len(local_map) else None
            map_prep_ms = (time.perf_counter_ns() - tp0) / 1e6

            if tree is None:
                from localization.step90 import ICP3DOFResult
                result = ICP3DOFResult(init, False, 0, 0, float("inf"), 0.0, "empty_local_map")
            else:
                result = scan_to_map_icp_3dof(
                    body, local_map, init, gz, roll, pitch, cfg, map_tree=tree
                )
            total_ms = (time.perf_counter_ns() - total0) / 1e6
            final_te, final_ye = pose_error_3dof(result.pose, gt_pose)
            success = bool(final_te <= success_translation_m and final_ye <= success_yaw_deg)

            trial_rows.append({
                "scan_index": si - 1,
                "scan_file": scan_path.name,
                "timestamp_us": int(_timestamp_from_scan(scan_path)),
                "perturbation": label,
                "perturb_xy_m": xy_mag,
                "perturb_yaw_deg": yaw_mag,
                "actual_dx_m": dx,
                "actual_dy_m": dy,
                "actual_dyaw_deg": dyaw_deg,
                "initial_translation_error_m": init_te,
                "initial_yaw_error_deg": init_ye,
                "final_translation_error_m": final_te,
                "final_yaw_error_deg": final_ye,
                "success": success,
                "converged": bool(result.converged),
                "iterations": int(result.iterations),
                "correspondences": int(result.correspondences),
                "final_rmse_m": float(result.rmse_m),
                "failure_reason": result.failure_reason,
                "query_points": int(len(body)),
                "local_map_voxels": int(len(local_map)),
                "retrieval_ms": retrieval_ms,
                "map_prep_ms": map_prep_ms,
                "match_ms": float(result.match_ms),
                "total_ms": total_ms,
                "initial_x": init.x, "initial_y": init.y, "initial_yaw": init.yaw,
                "estimated_x": result.pose.x, "estimated_y": result.pose.y, "estimated_yaw": result.pose.yaw,
                "gt_x": gx, "gt_y": gy, "gt_yaw": gyaw,
            })

    summary = _summaries(trial_rows)
    _write_csv(out / "localization_trials.csv", trial_rows)
    _write_csv(out / "localization_summary.csv", summary)
    _write_csv(out / "query_scans.csv", scan_rows)

    metadata = prepared.metadata()
    metadata.update({
        "step": "9.0",
        "purpose": "Conventional voxel scan-to-map localization harness validation before native BST/block matching.",
        "map_session": map_session,
        "query_session": query_session,
        "query_count": len(selected),
        "perturbations": [
            {"label": label, "xy_m": xy, "yaw_deg": yaw} for label, xy, yaw in levels
        ],
        "perturb_seed": int(perturb_seed),
        "localization_max_range_m": float(localization_max_range_m),
        "scan_downsample_m": float(scan_downsample_m),
        "max_scan_points": int(max_scan_points),
        "local_map_radius_m": float(local_map_radius_m),
        "map_bucket_voxels": int(map_bucket_voxels),
        "success_thresholds": {
            "translation_m": float(success_translation_m),
            "yaw_deg": float(success_yaw_deg),
        },
        "icp_config": asdict(cfg),
        "map_index_build_ms": float(map_index_build_ms),
        "map_voxels": int(len(prepared.session_codes[0])),
        "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()},
        "localization_scope": {
            "optimized_dof": ["x", "y", "yaw"],
            "fixed_from_reference_pose": ["z", "roll", "pitch"],
            "correspondence": "3-D nearest voxel-center using scipy.spatial.cKDTree",
            "update": "trimmed point-to-point planar SE(2) least squares",
            "retrieval_center": "perturbed initial pose, not ground truth",
            "cross_session": True,
        },
        "scientific_caveat": (
            "Step 9.0 validates the localization benchmark harness only. z/roll/pitch are held at reference values, "
            "the map is conventional voxel occupancy, and no native BST geometry is consumed yet."
        ),
    })
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return trial_rows, summary
