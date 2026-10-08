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

from localization.step90 import Pose3DOF, VoxelLocalMapIndex, pose_error_3dof
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import (
    DEFAULT_PYRAMID,
    PyramidLevelConfig,
    CoarseToFinePointToPlaneResult,
    prepare_local_map_pyramid,
    scan_to_map_coarse_to_fine_point_to_plane_3dof,
    voxel_downsample_first,
)
from benchmark.step901_benchmark import (
    _write_csv,
    _pctl,
    _cap_points,
    parse_perturbations,
    _trial_perturbation,
)
from realdata.nclt import (
    _filter_body_points,
    _list_scan_files,
    _timestamp_from_scan,
    discover_nclt_session,
    load_ground_truth,
    read_velodyne_sync_bin,
)
from realdata.nclt_packed import PackedNCLTPrepared, _nearest_pose_rows


def _select_query_scans(
    data_root: str | Path,
    session: str,
    count: int,
    max_pose_gap_us: int,
    exclude_ingest_stride: int | None = None,
):
    paths = discover_nclt_session(data_root, session)
    scans = _list_scan_files(paths.scan_dir)
    gt = load_ground_truth(paths.gt_csv)
    ts = np.asarray([_timestamp_from_scan(p) for p in scans], dtype=np.int64)
    pose_rows, gaps = _nearest_pose_rows(gt, ts)
    valid = np.flatnonzero(gaps <= int(max_pose_gap_us))
    if exclude_ingest_stride is not None and int(exclude_ingest_stride) > 1:
        stride = int(exclude_ingest_stride)
        valid = valid[(valid % stride) != 0]
    if not len(valid):
        raise RuntimeError(f"No query scans in {session} pass selection filters")
    n = min(int(count), len(valid))
    take = valid[np.linspace(0, len(valid) - 1, n, dtype=np.int64)]
    return [(scans[i], pose_rows[i], int(gaps[i]), int(i)) for i in take]


def _replay_query_scans(
    data_root: str | Path,
    session: str,
    csv_path: str | Path,
    count: int,
    max_pose_gap_us: int,
):
    with Path(csv_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    paths = discover_nclt_session(data_root, session)
    scan_list = _list_scan_files(paths.scan_dir)
    scans = {p.name: (p, i) for i, p in enumerate(scan_list)}
    gt_all = load_ground_truth(paths.gt_csv)
    chosen = []
    for row in rows:
        name = row["scan_file"]
        if name not in scans:
            raise FileNotFoundError(f"Replay scan {name} not found in {paths.scan_dir}")
        sp, original_index = scans[name]
        ts1 = np.asarray([_timestamp_from_scan(sp)], dtype=np.int64)
        prow, gap = _nearest_pose_rows(gt_all, ts1)
        if int(gap[0]) > int(max_pose_gap_us):
            raise RuntimeError(f"Replay scan {name} exceeds max pose gap")
        chosen.append((sp, prow[0], int(gap[0]), int(original_index)))
    return chosen[: int(count)]


def _summaries(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["perturbation"]), []).append(r)
    out = []
    for (scenario, label), g in groups.items():
        def vals(k): return [float(x[k]) for x in g]
        successes = [bool(x["success"]) for x in g]
        converged = [bool(x["converged"]) for x in g]
        out.append({
            "scenario": scenario,
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
            "median_pose_shift_from_initial_m": statistics.median(vals("pose_shift_from_initial_m")),
            "median_yaw_shift_from_initial_deg": statistics.median(vals("yaw_shift_from_initial_deg")),
            "median_retrieval_ms": statistics.median(vals("retrieval_ms")),
            "median_map_prep_ms": statistics.median(vals("map_prep_ms")),
            "median_match_ms": statistics.median(vals("match_ms")),
            "median_total_ms": statistics.median(vals("total_ms")),
            "p95_total_ms": _pctl(vals("total_ms"), .95),
            "mean_query_points": statistics.mean(vals("query_points")),
            "mean_local_map_voxels": statistics.mean(vals("local_map_voxels")),
            "mean_final_correspondences": statistics.mean(vals("correspondences")),
            "mean_normal_correspondences": statistics.mean(vals("normal_correspondences")),
            "median_final_plane_rmse_m": statistics.median(vals("final_plane_rmse_m")),
            "median_abs_plane_residual_m": statistics.median(vals("mean_abs_plane_residual_m")),
            "median_total_iterations": statistics.median(vals("iterations")),
        })
    scenario_order = {"cross_session": 0, "same_session_control": 1}
    return sorted(out, key=lambda x: (scenario_order.get(x["scenario"], 9), x["initial_translation_error_m"]))


def _load_and_prepare_scan(scan_path, prepared, localization_max_range_m, scan_downsample_m, max_scan_points):
    body = read_velodyne_sync_bin(scan_path)
    body = _filter_body_points(body, prepared.config)
    if len(body):
        r = np.linalg.norm(body, axis=1)
        body = body[r <= float(localization_max_range_m)]
    raw_after_range = len(body)
    # Preserve the Step-9.0.1 fine-resolution scan preparation. The pyramid
    # only downsamples further at coarser levels.
    body = voxel_downsample_first(body, scan_downsample_m)
    body = _cap_points(body, max_scan_points)
    return body, raw_after_range


def _run_trial(
    *,
    scenario: str,
    scan_index: int,
    scan_path,
    body: np.ndarray,
    gt,
    gap: int,
    original_scan_index: int,
    init: Pose3DOF,
    perturbation_label: str,
    xy_mag: float,
    yaw_mag: float,
    dx: float,
    dy: float,
    dyaw_deg: float,
    map_index: VoxelLocalMapIndex,
    local_map_radius_m: float,
    pyramid: tuple[PyramidLevelConfig, ...],
    base_icp_config: PointToPlaneICP3DOFConfig,
    success_translation_m: float,
    success_yaw_deg: float,
):
    _, gx, gy, gz, roll, pitch, gyaw = [float(v) for v in gt]
    gt_pose = Pose3DOF(gx, gy, gyaw)
    init_te, init_ye = pose_error_3dof(init, gt_pose)

    total0 = time.perf_counter_ns()
    tr0 = time.perf_counter_ns()
    local_map = map_index.query_radius_centers(init.x, init.y, local_map_radius_m)
    retrieval_ms = (time.perf_counter_ns() - tr0) / 1e6

    tp0 = time.perf_counter_ns()
    prepared_levels = prepare_local_map_pyramid(local_map, pyramid) if len(local_map) else []
    map_prep_ms = (time.perf_counter_ns() - tp0) / 1e6

    if not prepared_levels:
        result = CoarseToFinePointToPlaneResult(
            pose=init, converged=False, iterations=0, correspondences=0,
            normal_correspondences=0, rmse_m=float("inf"),
            mean_abs_plane_residual_m=float("inf"), match_ms=0.0,
            failure_reason="empty_local_map", level_results=[]
        )
    else:
        result = scan_to_map_coarse_to_fine_point_to_plane_3dof(
            body, prepared_levels, init, gz, roll, pitch, base_icp_config
        )
    total_ms = (time.perf_counter_ns() - total0) / 1e6
    final_te, final_ye = pose_error_3dof(result.pose, gt_pose)
    shift_m, shift_yaw = pose_error_3dof(result.pose, init)
    success = bool(final_te <= success_translation_m and final_ye <= success_yaw_deg)

    row = {
        "scenario": scenario,
        "scan_index": int(scan_index),
        "original_scan_index": int(original_scan_index),
        "scan_file": scan_path.name,
        "timestamp_us": int(_timestamp_from_scan(scan_path)),
        "pose_gap_us": int(gap),
        "matcher": "coarse_to_fine_point_to_plane",
        "perturbation": perturbation_label,
        "perturb_xy_m": float(xy_mag),
        "perturb_yaw_deg": float(yaw_mag),
        "actual_dx_m": float(dx),
        "actual_dy_m": float(dy),
        "actual_dyaw_deg": float(dyaw_deg),
        "initial_translation_error_m": init_te,
        "initial_yaw_error_deg": init_ye,
        "final_translation_error_m": final_te,
        "final_yaw_error_deg": final_ye,
        "pose_shift_from_initial_m": shift_m,
        "yaw_shift_from_initial_deg": shift_yaw,
        "success": success,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "correspondences": int(result.correspondences),
        "normal_correspondences": int(result.normal_correspondences),
        "final_rmse_m": float(result.rmse_m),
        "final_plane_rmse_m": float(result.rmse_m),
        "mean_abs_plane_residual_m": float(result.mean_abs_plane_residual_m),
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
    }
    for level_name, lev in result.level_results:
        prefix = f"level_{level_name}_"
        row[prefix + "converged"] = bool(lev.converged)
        row[prefix + "iterations"] = int(lev.iterations)
        row[prefix + "correspondences"] = int(lev.correspondences)
        row[prefix + "normal_correspondences"] = int(lev.normal_correspondences)
        row[prefix + "plane_rmse_m"] = float(lev.rmse_m)
        row[prefix + "mean_abs_plane_residual_m"] = float(lev.mean_abs_plane_residual_m)
        row[prefix + "match_ms"] = float(lev.match_ms)
        row[prefix + "failure_reason"] = lev.failure_reason
    return row


def benchmark_step902(
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
    icp_config: PointToPlaneICP3DOFConfig | None = None,
    pyramid: tuple[PyramidLevelConfig, ...] = DEFAULT_PYRAMID,
    query_scans_csv: str | Path | None = None,
    include_zero_control: bool = True,
    include_same_session_control: bool = True,
    same_session_query_count: int | None = None,
):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if map_session == query_session:
        raise ValueError("Step 9.0.2 main benchmark requires distinct map/query sessions")
    if prepared.session_names[:2] != [map_session, query_session]:
        raise ValueError("prepared sessions must be [map_session, query_session] in that order")

    cfg = icp_config or PointToPlaneICP3DOFConfig()
    print(f"[9.0.2] building conventional voxel local-map index from {map_session}", flush=True)
    t0 = time.perf_counter_ns()
    map_index = VoxelLocalMapIndex(
        prepared.session_codes[0], prepared.spec, prepared.origin_local_m, bucket_voxels=map_bucket_voxels
    )
    map_index_build_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"[9.0.2] map voxels={len(prepared.session_codes[0]):,}; index={map_index_build_ms:.1f} ms", flush=True)

    if query_scans_csv is None:
        cross_selected = _select_query_scans(
            data_root, query_session, query_count, prepared.config.max_pose_gap_us
        )
    else:
        cross_selected = _replay_query_scans(
            data_root, query_session, query_scans_csv, query_count, prepared.config.max_pose_gap_us
        )

    same_count = int(query_count if same_session_query_count is None else same_session_query_count)
    same_selected = []
    if include_same_session_control:
        same_selected = _select_query_scans(
            data_root,
            map_session,
            same_count,
            prepared.config.max_pose_gap_us,
            exclude_ingest_stride=prepared.config.scan_stride,
        )

    levels = parse_perturbations(perturbations)
    cross_rng = random.Random(int(perturb_seed))  # frozen to Step 9.0/9.0.1 sequence
    same_rng = random.Random(int(perturb_seed) + 902)
    trial_rows: list[dict] = []
    scan_rows: list[dict] = []

    def run_scenario(scenario: str, selected, rng: random.Random, preserve_cross_rng: bool):
        for si, (scan_path, gt, gap, original_scan_index) in enumerate(selected, start=1):
            body, raw_after_range = _load_and_prepare_scan(
                scan_path, prepared, localization_max_range_m, scan_downsample_m, max_scan_points
            )
            _, gx, gy, gz, roll, pitch, gyaw = [float(v) for v in gt]
            scan_rows.append({
                "scenario": scenario,
                "scan_index": si - 1,
                "original_scan_index": int(original_scan_index),
                "scan_file": scan_path.name,
                "timestamp_us": int(_timestamp_from_scan(scan_path)),
                "pose_gap_us": int(gap),
                "points_after_range": int(raw_after_range),
                "points_used": int(len(body)),
                "gt_x": gx, "gt_y": gy, "gt_z": gz,
                "gt_roll": roll, "gt_pitch": pitch, "gt_yaw": gyaw,
                "directly_ingested_scan": bool((original_scan_index % prepared.config.scan_stride) == 0),
            })
            print(f"[9.0.2] {scenario} scan {si}/{len(selected)} {scan_path.name}: {len(body):,} points", flush=True)

            # Main perturbations are executed first. For cross-session this keeps
            # exactly the same RNG consumption/order as Step 9.0 and 9.0.1.
            for label, xy_mag, yaw_mag in levels:
                dx, dy, dyaw_deg = _trial_perturbation(xy_mag, yaw_mag, rng)
                init = Pose3DOF(gx + dx, gy + dy, gyaw + math.radians(dyaw_deg))
                trial_rows.append(_run_trial(
                    scenario=scenario,
                    scan_index=si - 1,
                    scan_path=scan_path,
                    body=body,
                    gt=gt,
                    gap=gap,
                    original_scan_index=original_scan_index,
                    init=init,
                    perturbation_label=label,
                    xy_mag=xy_mag,
                    yaw_mag=yaw_mag,
                    dx=dx,
                    dy=dy,
                    dyaw_deg=dyaw_deg,
                    map_index=map_index,
                    local_map_radius_m=local_map_radius_m,
                    pyramid=pyramid,
                    base_icp_config=cfg,
                    success_translation_m=success_translation_m,
                    success_yaw_deg=success_yaw_deg,
                ))

            # Zero-control does not touch RNG, preserving frozen perturbations.
            if include_zero_control:
                init = Pose3DOF(gx, gy, gyaw)
                trial_rows.append(_run_trial(
                    scenario=scenario,
                    scan_index=si - 1,
                    scan_path=scan_path,
                    body=body,
                    gt=gt,
                    gap=gap,
                    original_scan_index=original_scan_index,
                    init=init,
                    perturbation_label="zero_control",
                    xy_mag=0.0,
                    yaw_mag=0.0,
                    dx=0.0,
                    dy=0.0,
                    dyaw_deg=0.0,
                    map_index=map_index,
                    local_map_radius_m=local_map_radius_m,
                    pyramid=pyramid,
                    base_icp_config=cfg,
                    success_translation_m=success_translation_m,
                    success_yaw_deg=success_yaw_deg,
                ))

    run_scenario("cross_session", cross_selected, cross_rng, True)
    if include_same_session_control:
        run_scenario("same_session_control", same_selected, same_rng, False)

    summary = _summaries(trial_rows)
    _write_csv(out / "localization_trials.csv", trial_rows)
    _write_csv(out / "localization_summary.csv", summary)
    _write_csv(out / "query_scans.csv", scan_rows)

    metadata = prepared.metadata()
    metadata.update({
        "step": "9.0.2",
        "purpose": "Coarse-to-fine point-to-plane localization with cross-session zero control and same-session off-stride sanity control.",
        "map_session": map_session,
        "query_session": query_session,
        "query_count": len(cross_selected),
        "same_session_query_count": len(same_selected),
        "perturbations": [
            {"label": label, "xy_m": xy, "yaw_deg": yaw} for label, xy, yaw in levels
        ],
        "include_zero_control": bool(include_zero_control),
        "include_same_session_control": bool(include_same_session_control),
        "same_session_query_selection": (
            "valid map-session scans with scan_index % map_ingest_stride != 0; therefore query scan is not directly one of the scans aggregated into the map"
            if include_same_session_control else None
        ),
        "perturb_seed": int(perturb_seed),
        "same_session_perturb_seed": int(perturb_seed) + 902,
        "localization_max_range_m": float(localization_max_range_m),
        "scan_downsample_m": float(scan_downsample_m),
        "max_scan_points": int(max_scan_points),
        "local_map_radius_m": float(local_map_radius_m),
        "map_bucket_voxels": int(map_bucket_voxels),
        "success_thresholds": {
            "translation_m": float(success_translation_m),
            "yaw_deg": float(success_yaw_deg),
        },
        "base_icp_config": asdict(cfg),
        "pyramid": [asdict(x) for x in pyramid],
        "map_index_build_ms": float(map_index_build_ms),
        "query_scans_replay_csv": str(query_scans_csv) if query_scans_csv is not None else None,
        "map_voxels": int(len(prepared.session_codes[0])),
        "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()},
        "localization_scope": {
            "optimized_dof": ["x", "y", "yaw"],
            "fixed_from_reference_pose": ["z", "roll", "pitch"],
            "correspondence": "3-D nearest voxel-center at a three-level coarse-to-fine map/scan pyramid; PCA map normals estimated lazily per level",
            "update": "trimmed point-to-plane Gauss-Newton over planar SE(2)",
            "retrieval_center": "initial pose for each trial; ground truth only for zero-control where initial==GT",
            "main_benchmark_cross_session": True,
            "same_session_control_is_optimistic": True,
        },
        "diagnostic_interpretation": {
            "cross_zero_control": "Measures the cross-session registration/GT consistency floor: how far the objective moves a scan starting at reference pose.",
            "same_session_control": "Tests matcher implementation on easier same-traversal geometry while excluding query frames directly ingested by the stride-sampled map.",
            "coarse_to_fine": "Tests whether capture range, rather than representation, caused the remaining Step-9.0.1 error.",
        },
        "scientific_caveat": (
            "Step 9.0.2 still uses conventional voxel occupancy and consumes no native BST geometry. "
            "The same-session control is deliberately diagnostic/optimistic and is not a final localization baseline."
        ),
    })
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return trial_rows, summary


__all__ = ["benchmark_step902", "_select_query_scans"]
