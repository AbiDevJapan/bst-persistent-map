"""Replay frozen NCLT queries and build/load immutable reference keyframes."""
from __future__ import annotations
import csv
import math
import time
from pathlib import Path
import numpy as np
from localization.pyramid import voxel_downsample_first
from localization.global_frontend import ScanContextConfig, make_scan_context
from realdata.nclt import (NCLTIngestConfig, discover_nclt_session, load_ground_truth,
    _list_scan_files, _timestamp_from_scan, read_velodyne_sync_bin, _filter_body_points)
from realdata.nclt_packed import _nearest_pose_rows
from benchmark.localization_helpers import _cap_points, _select_query_scans_excluding, _union_excluded

def _angle_diff_deg(a: float, b: float) -> float:
    d = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(d))

def _finite_pose(row) -> bool:
    return bool(np.all(np.isfinite(np.asarray(row[1:7], dtype=np.float64))))

def _load_body(scan_path: Path, cfg: NCLTIngestConfig, max_range_m: float | None = None) -> np.ndarray:
    pts = read_velodyne_sync_bin(scan_path)
    pts = _filter_body_points(pts, cfg)
    if len(pts) and max_range_m is not None:
        r = np.linalg.norm(pts, axis=1)
        pts = pts[r <= float(max_range_m)]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    return pts

def _prep_global_points(body: np.ndarray, voxel_m: float, max_points: int) -> np.ndarray:
    pts = voxel_downsample_first(np.asarray(body, dtype=np.float64), float(voxel_m))
    return np.asarray(_cap_points(pts, int(max_points)), dtype=np.float64)

def _reference_cache_path(cache_dir: str | Path, session: str, sc: ScanContextConfig,
                          scan_stride: int, min_translation_m: float, min_yaw_deg: float,
                          max_time_s: float, descriptor_voxel_m: float) -> Path:
    root = Path(cache_dir) / 'step9_5_scan_context'
    root.mkdir(parents=True, exist_ok=True)
    key = f'{session}_r{sc.num_rings}_s{sc.num_sectors}_R{sc.max_radius_m:g}_stride{scan_stride}_d{min_translation_m:g}_y{min_yaw_deg:g}_t{max_time_s:g}_v{descriptor_voxel_m:g}.npz'
    return root / key

def build_or_load_reference_database(
    data_root, session: str, cfg: NCLTIngestConfig, cache_dir,
    sc_config: ScanContextConfig, *, reference_scan_stride: int = 5,
    min_translation_m: float = 5.0, min_yaw_deg: float = 15.0, max_time_s: float = 5.0,
    descriptor_voxel_m: float = 0.5, max_keyframes: int = 0, progress: bool = True,
):
    cache = _reference_cache_path(cache_dir, session, sc_config, reference_scan_stride,
                                  min_translation_m, min_yaw_deg, max_time_s, descriptor_voxel_m)
    paths = discover_nclt_session(data_root, session)
    scans = _list_scan_files(paths.scan_dir)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        out = {
            'descriptors': np.asarray(z['descriptors'], dtype=np.float32),
            'ring_keys': np.asarray(z['ring_keys'], dtype=np.float32),
            'scan_indices': np.asarray(z['scan_indices'], dtype=np.int64),
            'pose_rows': np.asarray(z['pose_rows'], dtype=np.float64),
            'timestamps': np.asarray(z['timestamps'], dtype=np.int64),
            'scans': scans, 'cache_hit': True, 'build_ms': 0.0,
            'cache_path': str(cache),
        }
        return out

    t0 = time.perf_counter_ns()
    gt = load_ground_truth(paths.gt_csv)
    candidate_indices = np.arange(0, len(scans), max(1, int(reference_scan_stride)), dtype=np.int64)
    ts = np.asarray([_timestamp_from_scan(scans[int(i)]) for i in candidate_indices], dtype=np.int64)
    poses, gaps = _nearest_pose_rows(gt, ts)
    valid = (gaps <= int(cfg.max_pose_gap_us)) & np.all(np.isfinite(poses[:, 1:7]), axis=1)
    candidate_indices = candidate_indices[valid]
    poses = poses[valid]
    ts = ts[valid]

    chosen = []
    chosen_poses = []
    chosen_ts = []
    last_pose = None
    last_ts = None
    min_yaw = float(min_yaw_deg)
    for idx, pose, stamp in zip(candidate_indices, poses, ts):
        take = last_pose is None
        if not take:
            d = math.hypot(float(pose[1]-last_pose[1]), float(pose[2]-last_pose[2]))
            y = _angle_diff_deg(float(pose[6]), float(last_pose[6]))
            dt = abs(float(stamp-last_ts)) * 1e-6
            take = d >= float(min_translation_m) or y >= min_yaw or dt >= float(max_time_s)
        if take:
            chosen.append(int(idx)); chosen_poses.append(np.asarray(pose, dtype=np.float64)); chosen_ts.append(int(stamp))
            last_pose = pose; last_ts = stamp
            if max_keyframes > 0 and len(chosen) >= int(max_keyframes):
                break
    if not chosen:
        raise RuntimeError(f'no valid Scan Context reference keyframes for {session}')

    descriptors = []
    keep_indices = []
    keep_poses = []
    keep_ts = []
    for j, (idx, pose, stamp) in enumerate(zip(chosen, chosen_poses, chosen_ts)):
        body = _load_body(scans[idx], cfg, sc_config.max_radius_m)
        body = voxel_downsample_first(body, float(descriptor_voxel_m))
        if len(body) < 50:
            continue
        descriptors.append(make_scan_context(body, sc_config))
        keep_indices.append(idx); keep_poses.append(pose); keep_ts.append(stamp)
        if progress and (j + 1) % 250 == 0:
            print(f'  [9.5 ScanContext {session}] {j+1}/{len(chosen)} keyframes', flush=True)
    if not descriptors:
        raise RuntimeError(f'no non-empty Scan Context descriptors for {session}')
    desc = np.asarray(descriptors, dtype=np.float32)
    keys = np.mean(desc, axis=2, dtype=np.float32)
    arr_idx = np.asarray(keep_indices, dtype=np.int64)
    arr_pose = np.asarray(keep_poses, dtype=np.float64)
    arr_ts = np.asarray(keep_ts, dtype=np.int64)
    np.savez_compressed(cache, descriptors=desc, ring_keys=keys, scan_indices=arr_idx, pose_rows=arr_pose, timestamps=arr_ts)
    return {
        'descriptors': desc, 'ring_keys': keys, 'scan_indices': arr_idx, 'pose_rows': arr_pose,
        'timestamps': arr_ts, 'scans': scans, 'cache_hit': False,
        'build_ms': (time.perf_counter_ns()-t0)/1e6, 'cache_path': str(cache),
    }

def _replay_multisession(data_root, csv_path, query_sessions, count_per_session, max_pose_gap_us):
    with Path(csv_path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    out = {}
    for session in query_sessions:
        names = [r['scan_file'] for r in rows if r.get('query_session') == session and r.get('scenario','cross_session') == 'cross_session']
        if not names:
            raise ValueError(f'query replay CSV contains no cross-session rows for {session}')
        paths = discover_nclt_session(data_root, session)
        scan_list = _list_scan_files(paths.scan_dir)
        by_name = {p.name:(p,i) for i,p in enumerate(scan_list)}
        gt = load_ground_truth(paths.gt_csv)
        chosen=[]
        for name in names[:int(count_per_session)]:
            if name not in by_name:
                raise FileNotFoundError(f'replay scan {name} missing from {session}')
            sp, oi = by_name[name]
            prow, gap = _nearest_pose_rows(gt, np.asarray([_timestamp_from_scan(sp)], dtype=np.int64))
            if int(gap[0]) > int(max_pose_gap_us) or not _finite_pose(prow[0]):
                raise RuntimeError(f'replay scan {name} has invalid pose')
            chosen.append((sp, prow[0], int(gap[0]), int(oi)))
        out[session]=chosen
    return out

def _select_queries(data_root, query_sessions, count_per_session, max_pose_gap_us,
                    query_scans_csv=None, exclude_query_scans_csv=()):
    if query_scans_csv and exclude_query_scans_csv:
        raise ValueError('use query_scans_csv replay OR exclusion CSVs, not both')
    if query_scans_csv:
        return _replay_multisession(data_root, query_scans_csv, query_sessions, count_per_session, max_pose_gap_us)
    excluded = _union_excluded(exclude_query_scans_csv) if exclude_query_scans_csv else set()
    return {s:_select_query_scans_excluding(data_root,s,count_per_session,max_pose_gap_us,excluded) for s in query_sessions}

