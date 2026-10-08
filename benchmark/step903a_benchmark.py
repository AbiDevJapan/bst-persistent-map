from __future__ import annotations

import csv
import gc
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark.step901_benchmark import _trial_perturbation, parse_perturbations, _write_csv, _pctl
from benchmark.step902_benchmark import (
    _select_query_scans,
    _replay_query_scans,
    _load_and_prepare_scan,
    _run_trial,
)
from localization.step90 import Pose3DOF, VoxelLocalMapIndex
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID, PyramidLevelConfig
from realdata.nclt import NCLTIngestConfig, discover_nclt_session, _timestamp_from_scan
from realdata.nclt_packed import (
    _session_absolute_codes,
    _cache_paths,
    unpack_signed_xyz,
    linear_encode,
)
from synthetic.world import WorldSpec


def parse_resolutions(spec: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in str(spec).split(",") if x.strip())
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("resolutions must be positive comma-separated meters")
    if len(set(vals)) != len(vals):
        raise ValueError("resolutions must not contain duplicates")
    return vals


def scaled_map_pyramid(voxel_size_m: float, reference_voxel_m: float = 0.5) -> tuple[PyramidLevelConfig, ...]:
    """Keep scan-side sampling/physical thresholds fixed, scale map-side resolution only."""
    scale = float(voxel_size_m) / float(reference_voxel_m)
    return tuple(PyramidLevelConfig(
        name=p.name,
        scan_voxel_m=p.scan_voxel_m,
        map_voxel_m=p.map_voxel_m * scale,
        max_correspondence_m=p.max_correspondence_m,
        max_iterations=p.max_iterations,
        trim_fraction=p.trim_fraction,
        normal_max_radius_m=p.normal_max_radius_m,
        max_step_translation_m=p.max_step_translation_m,
        max_step_yaw_deg=p.max_step_yaw_deg,
    ) for p in DEFAULT_PYRAMID)


def bucket_voxels_for_resolution(voxel_size_m: float, bucket_width_m: float = 8.0) -> int:
    return max(1, int(round(float(bucket_width_m) / float(voxel_size_m))))


def _cache_for_policy(cache_dir: str | Path | None, session: str, cfg: NCLTIngestConfig, policy: str):
    if policy == "off" or cache_dir is None:
        return None, False
    cache = Path(cache_dir)
    npy, js = _cache_paths(cache, session, cfg)
    exists = npy.exists() and js.exists()
    if exists:
        return cache, True
    if policy == "read-write":
        return cache, False
    if policy == "reuse-only":
        return None, False
    raise ValueError("cache_policy must be one of reuse-only, read-write, off")


def prepare_map_only(
    data_root: str | Path,
    map_session: str,
    cfg: NCLTIngestConfig,
    cache_dir: str | Path | None,
    cache_policy: str = "reuse-only",
):
    paths = discover_nclt_session(data_root, map_session)
    selected_cache, cache_hit = _cache_for_policy(cache_dir, map_session, cfg, cache_policy)
    t0 = time.perf_counter_ns()
    abs_codes, stats = _session_absolute_codes(paths, cfg, selected_cache)
    ingest_ms = (time.perf_counter_ns() - t0) / 1e6
    if not len(abs_codes):
        raise RuntimeError(f"No occupied voxels for {map_session} at {cfg.voxel_size_m} m")
    x, y, z = unpack_signed_xyz(abs_codes)
    mins = np.asarray([x.min(), y.min(), z.min()], dtype=np.int64)
    maxs = np.asarray([x.max(), y.max(), z.max()], dtype=np.int64)
    dims = (maxs - mins + 1).astype(int)
    spec = WorldSpec(
        float(dims[0] * cfg.voxel_size_m),
        float(dims[1] * cfg.voxel_size_m),
        float(dims[2] * cfg.voxel_size_m),
        float(cfg.voxel_size_m),
    )
    codes = np.unique(linear_encode(x - mins[0], y - mins[1], z - mins[2], spec))
    origin = tuple(float(v * cfg.voxel_size_m) for v in mins)
    return SimpleNamespace(
        spec=spec,
        origin_local_m=origin,
        map_codes=codes,
        config=cfg,
        session_stats=stats,
        ingest_ms=float(ingest_ms),
        cache_hit=bool(cache_hit),
    )


def _filter_replay_csv_for_cross_session(path: str | Path, out_path: Path) -> Path:
    """Accept Step-9.0/9.0.1 cross-only CSVs or Step-9.0.2 mixed scenario CSVs."""
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("query_scans_csv is empty")
    if "scenario" not in rows[0]:
        return Path(path)
    rows = [r for r in rows if r.get("scenario") == "cross_session"]
    if not rows:
        raise ValueError("query_scans_csv contains no cross_session rows")
    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    return out_path


def _summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[float, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((float(r["voxel_size_m"]), r["scenario"], r["perturbation"]), []).append(r)
    out = []
    for (vs, scenario, label), g in groups.items():
        def vals(k): return [float(x[k]) for x in g]
        out.append({
            "voxel_size_m": vs,
            "scenario": scenario,
            "perturbation": label,
            "trials": len(g),
            "success_rate": float(np.mean([bool(x["success"]) for x in g])),
            "optimizer_convergence_rate": float(np.mean([bool(x["converged"]) for x in g])),
            "initial_translation_error_m": statistics.mean(vals("initial_translation_error_m")),
            "initial_yaw_error_deg": statistics.mean(vals("initial_yaw_error_deg")),
            "median_final_translation_error_m": statistics.median(vals("final_translation_error_m")),
            "p95_final_translation_error_m": _pctl(vals("final_translation_error_m"), .95),
            "median_final_yaw_error_deg": statistics.median(vals("final_yaw_error_deg")),
            "p95_final_yaw_error_deg": _pctl(vals("final_yaw_error_deg"), .95),
            "median_pose_shift_from_initial_m": statistics.median(vals("pose_shift_from_initial_m")),
            "median_yaw_shift_from_initial_deg": statistics.median(vals("yaw_shift_from_initial_deg")),
            "median_total_ms": statistics.median(vals("total_ms")),
            "median_match_ms": statistics.median(vals("match_ms")),
            "median_map_prep_ms": statistics.median(vals("map_prep_ms")),
            "mean_local_map_voxels": statistics.mean(vals("local_map_voxels")),
            "mean_normal_correspondences": statistics.mean(vals("normal_correspondences")),
            "median_final_plane_rmse_m": statistics.median(vals("final_plane_rmse_m")),
        })
    order = {"cross_session": 0, "same_session_control": 1}
    return sorted(out, key=lambda r: (r["voxel_size_m"], order.get(r["scenario"], 9), r["initial_translation_error_m"]))


def benchmark_step903a(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    map_session: str,
    query_session: str,
    resolutions: tuple[float, ...] = (0.25, 0.5, 1.0),
    scan_stride: int = 10,
    min_range_m: float = 2.0,
    max_range_m: float = 80.0,
    min_points_per_voxel: int = 1,
    max_pose_gap_us: int = 100000,
    cache_dir: str | Path | None = ".cache/step8_1",
    cache_policy: str = "reuse-only",
    query_count: int = 10,
    same_session_query_count: int = 10,
    query_scans_csv: str | Path | None = None,
    perturbations: str = "0.5:1,1:3",
    perturb_seed: int = 42090,
    localization_max_range_m: float = 30.0,
    scan_downsample_m: float = 0.75,
    max_scan_points: int = 8000,
    local_map_radius_m: float = 40.0,
    bucket_width_m: float = 8.0,
    success_translation_m: float = 0.5,
    success_yaw_deg: float = 1.0,
    icp_config: PointToPlaneICP3DOFConfig | None = None,
    include_same_session_zero: bool = True,
    same_session_perturbations: bool = False,
):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if map_session == query_session:
        raise ValueError("Use distinct map/query sessions for the cross-session diagnostic")
    base_cfg = icp_config or PointToPlaneICP3DOFConfig()
    levels = parse_perturbations(perturbations)

    # Select/replay scan identities once. Every resolution uses exactly the same scans.
    replay_path = None
    if query_scans_csv is not None:
        replay_path = _filter_replay_csv_for_cross_session(query_scans_csv, out / "_cross_replay.csv")
        cross_selected = _replay_query_scans(data_root, query_session, replay_path, query_count, max_pose_gap_us)
    else:
        cross_selected = _select_query_scans(data_root, query_session, query_count, max_pose_gap_us)
    same_selected = _select_query_scans(
        data_root, map_session, same_session_query_count, max_pose_gap_us, exclude_ingest_stride=scan_stride
    ) if include_same_session_zero else []

    trial_rows: list[dict] = []
    scan_rows: list[dict] = []
    map_rows: list[dict] = []
    scan_meta_written = set()

    for vs in resolutions:
        print(f"\n[9.0.3A] === voxel size {vs:g} m ===", flush=True)
        cfg = NCLTIngestConfig(
            voxel_size_m=float(vs), scan_stride=int(scan_stride), min_range_m=float(min_range_m),
            max_range_m=float(max_range_m), min_points_per_voxel=int(min_points_per_voxel),
            max_pose_gap_us=int(max_pose_gap_us),
        )
        prepared = prepare_map_only(data_root, map_session, cfg, cache_dir, cache_policy)
        bucket_voxels = bucket_voxels_for_resolution(vs, bucket_width_m)
        t0 = time.perf_counter_ns()
        map_index = VoxelLocalMapIndex(prepared.map_codes, prepared.spec, prepared.origin_local_m, bucket_voxels)
        index_ms = (time.perf_counter_ns() - t0) / 1e6
        pyramid = scaled_map_pyramid(vs)
        index_bytes = int(
            map_index.codes.nbytes + map_index._keys_per_code.nbytes + map_index._codes_by_bucket.nbytes +
            map_index.bucket_keys.nbytes + map_index.bucket_starts.nbytes + map_index.bucket_counts.nbytes
        )
        map_rows.append({
            "voxel_size_m": float(vs),
            "map_voxels": int(len(prepared.map_codes)),
            "map_codes_mib": float(prepared.map_codes.nbytes / (1024**2)),
            "index_arrays_mib": float(index_bytes / (1024**2)),
            "ingest_or_cache_load_ms": prepared.ingest_ms,
            "index_build_ms": float(index_ms),
            "cache_hit": prepared.cache_hit,
            "cache_policy": cache_policy,
            "bucket_width_target_m": float(bucket_width_m),
            "bucket_voxels": int(bucket_voxels),
            "bucket_width_actual_m": float(bucket_voxels * vs),
            "pyramid_map_voxels_m": ";".join(f"{p.map_voxel_m:g}" for p in pyramid),
            "pyramid_scan_voxels_m": ";".join(f"{p.scan_voxel_m:g}" for p in pyramid),
        })
        print(f"[9.0.3A] map voxels={len(prepared.map_codes):,}; bucket={bucket_voxels} vox ({bucket_voxels*vs:g}m); cache_hit={prepared.cache_hit}", flush=True)

        # Thin wrapper accepted by the frozen scan-preparation helper.
        prepared_scan = SimpleNamespace(config=cfg)

        def run_selected(scenario: str, selected, perturb: bool, rng_seed: int):
            rng = random.Random(int(rng_seed))
            for si, (scan_path, gt, gap, original_idx) in enumerate(selected, start=1):
                body, raw_after_range = _load_and_prepare_scan(
                    scan_path, prepared_scan, localization_max_range_m, scan_downsample_m, max_scan_points
                )
                _, gx, gy, gz, roll, pitch, gyaw = [float(v) for v in gt]
                key = (scenario, scan_path.name)
                if key not in scan_meta_written:
                    scan_meta_written.add(key)
                    scan_rows.append({
                        "scenario": scenario, "scan_index": si-1, "original_scan_index": int(original_idx),
                        "scan_file": scan_path.name, "timestamp_us": int(_timestamp_from_scan(scan_path)),
                        "pose_gap_us": int(gap), "points_after_range": int(raw_after_range), "points_used": int(len(body)),
                        "gt_x": gx, "gt_y": gy, "gt_z": gz, "gt_roll": roll, "gt_pitch": pitch, "gt_yaw": gyaw,
                        "directly_ingested_scan": bool((original_idx % scan_stride) == 0),
                    })

                # Zero control is the primary resolution-floor measurement.
                init = Pose3DOF(gx, gy, gyaw)
                row = _run_trial(
                    scenario=scenario, scan_index=si-1, scan_path=scan_path, body=body, gt=gt, gap=gap,
                    original_scan_index=original_idx, init=init, perturbation_label="zero_control",
                    xy_mag=0.0, yaw_mag=0.0, dx=0.0, dy=0.0, dyaw_deg=0.0,
                    map_index=map_index, local_map_radius_m=local_map_radius_m, pyramid=pyramid,
                    base_icp_config=base_cfg, success_translation_m=success_translation_m, success_yaw_deg=success_yaw_deg,
                )
                row["voxel_size_m"] = float(vs); trial_rows.append(row)

                if perturb:
                    for label, xy_mag, yaw_mag in levels:
                        dx, dy, dyaw_deg = _trial_perturbation(xy_mag, yaw_mag, rng)
                        init = Pose3DOF(gx + dx, gy + dy, gyaw + math.radians(dyaw_deg))
                        row = _run_trial(
                            scenario=scenario, scan_index=si-1, scan_path=scan_path, body=body, gt=gt, gap=gap,
                            original_scan_index=original_idx, init=init, perturbation_label=label,
                            xy_mag=xy_mag, yaw_mag=yaw_mag, dx=dx, dy=dy, dyaw_deg=dyaw_deg,
                            map_index=map_index, local_map_radius_m=local_map_radius_m, pyramid=pyramid,
                            base_icp_config=base_cfg, success_translation_m=success_translation_m, success_yaw_deg=success_yaw_deg,
                        )
                        row["voxel_size_m"] = float(vs); trial_rows.append(row)

        run_selected("cross_session", cross_selected, True, perturb_seed)
        if include_same_session_zero:
            run_selected("same_session_control", same_selected, same_session_perturbations, perturb_seed + 902)

        # Checkpoint after every resolution so a memory/time failure at a finer
        # resolution does not discard already-completed diagnostic results.
        _write_csv(out / "localization_trials.csv", trial_rows)
        _write_csv(out / "localization_summary.csv", _summary(trial_rows))
        _write_csv(out / "map_resolution_stats.csv", map_rows)
        _write_csv(out / "query_scans.csv", scan_rows)

        # Deliberately release the potentially large resolution-specific map before the next one.
        del map_index, prepared
        gc.collect()

    summary = _summary(trial_rows)
    _write_csv(out / "localization_trials.csv", trial_rows)
    _write_csv(out / "localization_summary.csv", summary)
    _write_csv(out / "map_resolution_stats.csv", map_rows)
    _write_csv(out / "query_scans.csv", scan_rows)
    tmp = out / "_cross_replay.csv"
    if tmp.exists():
        tmp.unlink()

    metadata = {
        "step": "9.0.3A",
        "purpose": "Voxel-size diagnostic for the conventional occupancy-center localization baseline before surfel/native-block localization.",
        "map_session": map_session,
        "query_session": query_session,
        "resolutions_m": list(map(float, resolutions)),
        "query_count": len(cross_selected),
        "same_session_query_count": len(same_selected),
        "perturbations": [{"label": l, "xy_m": x, "yaw_deg": y} for l, x, y in levels],
        "cross_session_always_includes_zero_control": True,
        "same_session_zero_control": bool(include_same_session_zero),
        "same_session_perturbations": bool(same_session_perturbations),
        "query_scans_replay_csv": str(query_scans_csv) if query_scans_csv is not None else None,
        "cache_policy": cache_policy,
        "cache_behavior": "reuse-only reuses an already-existing exact ingest cache but never writes a new resolution cache",
        "scan_stride": int(scan_stride),
        "localization_max_range_m": float(localization_max_range_m),
        "scan_downsample_m": float(scan_downsample_m),
        "max_scan_points": int(max_scan_points),
        "local_map_radius_m": float(local_map_radius_m),
        "bucket_width_m": float(bucket_width_m),
        "success_thresholds": {"translation_m": success_translation_m, "yaw_deg": success_yaw_deg},
        "base_icp_config": asdict(base_cfg),
        "pyramid_policy": {
            "scan_side": "fixed across voxel resolutions",
            "map_side": "9.0.2 map pyramid scaled linearly by tested base voxel size relative to 0.5 m",
            "correspondence_and_normal_radii": "fixed in physical meters across resolutions",
        },
        "map_resolution_stats": map_rows,
        "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()},
        "scientific_caveat": (
            "Step 9.0.3A remains a diagnostic of occupancy-center geometry. It changes map voxel resolution and therefore map size; "
            "it does not alter the Step-8 representation experiments and does not yet store sub-voxel surfel geometry or consume native BST blocks."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return trial_rows, summary, map_rows


__all__ = [
    "benchmark_step903a", "parse_resolutions", "scaled_map_pyramid",
    "bucket_voxels_for_resolution", "prepare_map_only",
]
