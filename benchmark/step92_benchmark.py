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
from benchmark.step902_benchmark import _select_query_scans, _load_and_prepare_scan
from benchmark.step91_benchmark import _run_trial
from benchmark.step913_benchmark import (
    FROZEN_PROFILE,
    FROZEN_K,
    _read_excluded_scan_names,
    _select_query_scans_excluding,
    _runtime_voxel_geometry_bytes,
)
from localization.step90 import Pose3DOF
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID
from localization.step903b import SurfelLocalMapIndex
from localization.step91 import align_voxel_surfels_to_prepared
from localization.step912 import build_native_surfel_hierarchy, build_hierarchical_lod_view
from localization.step913 import materialize_lod_view, MaterializedLODBSTIndex, verify_materialization
from realdata.nclt import NCLTIngestConfig, _timestamp_from_scan
from realdata.nclt_packed import prepare_nclt_packed
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec


MAIN_PERTURBATION = 'xy0.5_yaw1'


def _session_block_timing_metadata(values) -> dict:
    per_session = [float(x) for x in values]
    return {
        'session_block_build_ms': float(sum(per_session)),
        'session_block_build_ms_per_session': per_session,
    }


def _session_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r['map_geometry'], r['scenario'], r.get('query_session', ''), r['perturbation'])
        groups.setdefault(key, []).append(r)
    out = []
    for (geom, scenario, query_session, label), g in groups.items():
        def vals(k): return [float(x[k]) for x in g]
        out.append({
            'map_geometry': geom,
            'scenario': scenario,
            'query_session': query_session,
            'perturbation': label,
            'trials': len(g),
            'success_rate': float(np.mean([bool(x['success']) for x in g])),
            'optimizer_convergence_rate': float(np.mean([bool(x['converged']) for x in g])),
            'median_final_translation_error_m': statistics.median(vals('final_translation_error_m')),
            'p95_final_translation_error_m': _pctl(vals('final_translation_error_m'), .95),
            'median_final_yaw_error_deg': statistics.median(vals('final_yaw_error_deg')),
            'p95_final_yaw_error_deg': _pctl(vals('final_yaw_error_deg'), .95),
            'median_retrieval_ms': statistics.median(vals('retrieval_ms')),
            'median_map_prep_ms': statistics.median(vals('map_prep_ms')),
            'median_match_ms': statistics.median(vals('match_ms')),
            'median_total_ms': statistics.median(vals('total_ms')),
            'p95_total_ms': _pctl(vals('total_ms'), .95),
            'mean_query_points': statistics.mean(vals('query_points')),
            'mean_local_map_primitives': statistics.mean(vals('local_map_primitives')),
            'mean_final_correspondences': statistics.mean(vals('correspondences')),
            'mean_normal_correspondences': statistics.mean(vals('normal_correspondences')),
            'median_final_plane_rmse_m': statistics.median(vals('final_plane_rmse_m')),
            'median_abs_plane_residual_m': statistics.median(vals('mean_abs_plane_residual_m')),
            'median_total_iterations': statistics.median(vals('iterations')),
        })
    return sorted(out, key=lambda r: (r['scenario'], r['query_session'], r['perturbation'], r['map_geometry']))


def _pooled_summary(rows: list[dict]) -> list[dict]:
    # Pool only cross-session rows across query sessions. Same-session control is
    # already one common Jan->Jan diagnostic and is reported session-wise.
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r['scenario'] != 'cross_session':
            continue
        key = (r['map_geometry'], r['perturbation'])
        groups.setdefault(key, []).append(r)
    out = []
    for (geom, label), g in groups.items():
        def vals(k): return [float(x[k]) for x in g]
        out.append({
            'map_geometry': geom,
            'scenario': 'cross_session_pooled',
            'query_sessions': len(set(x['query_session'] for x in g)),
            'perturbation': label,
            'trials': len(g),
            'success_rate': float(np.mean([bool(x['success']) for x in g])),
            'optimizer_convergence_rate': float(np.mean([bool(x['converged']) for x in g])),
            'median_final_translation_error_m': statistics.median(vals('final_translation_error_m')),
            'p95_final_translation_error_m': _pctl(vals('final_translation_error_m'), .95),
            'median_final_yaw_error_deg': statistics.median(vals('final_yaw_error_deg')),
            'p95_final_yaw_error_deg': _pctl(vals('final_yaw_error_deg'), .95),
            'median_retrieval_ms': statistics.median(vals('retrieval_ms')),
            'median_map_prep_ms': statistics.median(vals('map_prep_ms')),
            'median_match_ms': statistics.median(vals('match_ms')),
            'median_total_ms': statistics.median(vals('total_ms')),
            'p95_total_ms': _pctl(vals('total_ms'), .95),
            'mean_local_map_primitives': statistics.mean(vals('local_map_primitives')),
            'median_final_plane_rmse_m': statistics.median(vals('final_plane_rmse_m')),
        })
    return sorted(out, key=lambda r: (r['perturbation'], r['map_geometry']))


def _exact_mcnemar_p(voxel_only: int, bst_only: int) -> float:
    # Exact two-sided sign/binomial test on discordant matched pairs.
    n = int(voxel_only + bst_only)
    if n == 0:
        return 1.0
    k = min(int(voxel_only), int(bst_only))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def _paired_outcomes(rows: list[dict]) -> list[dict]:
    # One row per query session and perturbation, plus a pooled row.
    cross = [r for r in rows if r['scenario'] == 'cross_session']
    pairs = sorted(set((r['query_session'], r['perturbation']) for r in cross))
    out = []
    for session, perturb in pairs + [('ALL', p) for p in sorted(set(r['perturbation'] for r in cross))]:
        q = [r for r in cross if r['perturbation'] == perturb and (session == 'ALL' or r['query_session'] == session)]
        key = lambda r: (r['query_session'], int(r['original_scan_index']), r['perturbation'])
        va = {key(r): bool(r['success']) for r in q if r['map_geometry'] == 'voxel_surfel_0.5m'}
        ba = {key(r): bool(r['success']) for r in q if r['map_geometry'] == 'bst_lod_k4_compact'}
        if set(va) != set(ba):
            raise AssertionError(f'paired outcome keys differ for {session}/{perturb}')
        both = sum(va[k] and ba[k] for k in va)
        voxel_only = sum(va[k] and not ba[k] for k in va)
        bst_only = sum((not va[k]) and ba[k] for k in va)
        neither = sum((not va[k]) and (not ba[k]) for k in va)
        out.append({
            'query_session': session,
            'perturbation': perturb,
            'paired_trials': len(va),
            'both_success': both,
            'voxel_only_success': voxel_only,
            'bst_only_success': bst_only,
            'neither_success': neither,
            'net_bst_success_difference': bst_only - voxel_only,
            'exact_mcnemar_p': _exact_mcnemar_p(voxel_only, bst_only),
        })
    return out


def _validation_gates(pooled: list[dict], *, main_perturbation: str = MAIN_PERTURBATION) -> list[dict]:
    rows = [r for r in pooled if r['perturbation'] == main_perturbation]
    by = {r['map_geometry']: r for r in rows}
    v = by.get('voxel_surfel_0.5m')
    b = by.get('bst_lod_k4_compact')
    if v is None or b is None:
        return []
    success_ratio = float(b['success_rate'] / v['success_rate']) if v['success_rate'] > 0 else float('nan')
    primitive_reduction = 1.0 - float(b['mean_local_map_primitives'] / v['mean_local_map_primitives'])
    speedup = float(v['median_total_ms'] / b['median_total_ms']) if b['median_total_ms'] > 0 else float('inf')
    return [{
        'perturbation': main_perturbation,
        'voxel_success_rate': v['success_rate'],
        'bst_success_rate': b['success_rate'],
        'success_ratio_bst_to_voxel': success_ratio,
        'meets_success_retention_0.90': bool(success_ratio >= 0.90),
        'primitive_reduction_vs_voxel': primitive_reduction,
        'meets_primitive_reduction_0.40': bool(primitive_reduction >= 0.40),
        'median_total_speedup_vs_voxel': speedup,
        'meets_speedup_1.20': bool(speedup >= 1.20),
        'overall_gate_pass': bool(success_ratio >= 0.90 and primitive_reduction >= 0.40 and speedup >= 1.20),
    }]


def benchmark_step92(
    *, data_root, output_dir, history_sessions,
    map_session='2012-01-15', query_sessions=('2012-02-02','2012-03-17','2012-05-11','2012-06-15'),
    voxel_size_m=0.5, scan_stride=10, min_range_m=2.0, max_range_m=80.0,
    min_points_per_voxel=1, max_pose_gap_us=100000, cache_dir='.cache/step8_1',
    query_count_per_session=50, same_session_query_count=50,
    exclude_query_scans_csv=None,
    perturbations='0.5:1', perturb_seed=42090,
    localization_max_range_m=30.0, scan_downsample_m=0.75, max_scan_points=8000,
    local_map_radius_m=40.0, bucket_width_m=8.0,
    success_translation_m=0.5, success_yaw_deg=1.0, icp_config=None,
    include_same_session_zero=True, progress=True,
):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    names = list(history_sessions)
    query_sessions = list(query_sessions)
    if map_session not in names:
        raise ValueError('map_session must appear in history_sessions')
    if not query_sessions:
        raise ValueError('at least one query session is required')
    if map_session in query_sessions:
        raise ValueError('query_sessions must be cross-session and exclude map_session')
    missing = [s for s in query_sessions if s not in names]
    if missing:
        raise ValueError(f'query sessions must appear in history_sessions: {missing}')
    if abs(float(voxel_size_m) - 0.5) > 1e-9:
        raise ValueError('Step 9.2 keeps the frozen Step-8 0.5m canonical lattice')

    map_t = names.index(map_session)
    cfg = NCLTIngestConfig(
        voxel_size_m=float(voxel_size_m), scan_stride=int(scan_stride),
        min_range_m=float(min_range_m), max_range_m=float(max_range_m),
        min_points_per_voxel=int(min_points_per_voxel), max_pose_gap_us=int(max_pose_gap_us),
    )
    base_cfg = icp_config or PointToPlaneICP3DOFConfig()
    levels = parse_perturbations(perturbations)
    pyramid = DEFAULT_PYRAMID
    if not any(l == MAIN_PERTURBATION for l, _, _ in levels):
        raise ValueError(f'Step 9.2 main validation perturbation {MAIN_PERTURBATION!r} must be included')

    tuning_cross_names = _read_excluded_scan_names(exclude_query_scans_csv, 'cross_session') if exclude_query_scans_csv else set()
    tuning_same_names = _read_excluded_scan_names(exclude_query_scans_csv, 'same_session_control') if exclude_query_scans_csv else set()

    selected_by_session = {}
    selection_roles = {}
    for session in query_sessions:
        selected = _select_query_scans_excluding(
            data_root, session, query_count_per_session, max_pose_gap_us, tuning_cross_names,
        )
        selected_by_session[session] = selected
        selection_roles[session] = 'held-out scan selection; session pair unseen during tuning' if session != '2012-02-02' else (
            'held-out scans excluding Step-9.1.2 tuning CSV' if tuning_cross_names else 'deterministic scans; no exclusion CSV supplied'
        )

    same = []
    if include_same_session_zero:
        same = _select_query_scans_excluding(
            data_root, map_session, same_session_query_count, max_pose_gap_us,
            tuning_same_names, exclude_ingest_stride=scan_stride,
        )

    prep_scan = SimpleNamespace(config=cfg)
    loaded = {}
    scan_rows = []
    for session, selected in selected_by_session.items():
        for si, (sp, gt, gap, oi) in enumerate(selected):
            body, raw = _load_and_prepare_scan(sp, prep_scan, localization_max_range_m, scan_downsample_m, max_scan_points)
            loaded[('cross_session', session, si)] = (sp, gt, gap, oi, body)
            _, gx, gy, gz, roll, pitch, gyaw = [float(v) for v in gt]
            scan_rows.append({
                'scenario': 'cross_session', 'map_session': map_session, 'query_session': session,
                'scan_index': si, 'original_scan_index': int(oi), 'scan_file': sp.name,
                'timestamp_us': int(_timestamp_from_scan(sp)), 'pose_gap_us': int(gap),
                'points_after_range': int(raw), 'points_used': int(len(body)),
                'gt_x': gx, 'gt_y': gy, 'gt_z': gz, 'gt_roll': roll, 'gt_pitch': pitch, 'gt_yaw': gyaw,
                'directly_ingested_scan': bool((oi % scan_stride) == 0),
                'excluded_tuning_name_match': bool(sp.name in tuning_cross_names),
                'selection_role': selection_roles[session],
            })
    for si, (sp, gt, gap, oi) in enumerate(same):
        body, raw = _load_and_prepare_scan(sp, prep_scan, localization_max_range_m, scan_downsample_m, max_scan_points)
        loaded[('same_session_control', map_session, si)] = (sp, gt, gap, oi, body)
        _, gx, gy, gz, roll, pitch, gyaw = [float(v) for v in gt]
        scan_rows.append({
            'scenario': 'same_session_control', 'map_session': map_session, 'query_session': map_session,
            'scan_index': si, 'original_scan_index': int(oi), 'scan_file': sp.name,
            'timestamp_us': int(_timestamp_from_scan(sp)), 'pose_gap_us': int(gap),
            'points_after_range': int(raw), 'points_used': int(len(body)),
            'gt_x': gx, 'gt_y': gy, 'gt_z': gz, 'gt_roll': roll, 'gt_pitch': pitch, 'gt_yaw': gyaw,
            'directly_ingested_scan': bool((oi % scan_stride) == 0),
            'excluded_tuning_name_match': bool(sp.name in tuning_same_names),
            'selection_role': 'held-out off-stride same-session control',
        })

    if any(r['excluded_tuning_name_match'] for r in scan_rows):
        raise AssertionError('held-out query selection included a tuning scan')
    if any(r['scenario'] == 'same_session_control' and r['directly_ingested_scan'] for r in scan_rows):
        raise AssertionError('same-session control contains a directly ingested map scan')
    # Fail early, before the expensive five-session map/hierarchy build, if a
    # future selector/replay path ever admits a non-finite reference pose.
    bad_pose_rows = [
        r for r in scan_rows
        if not np.all(np.isfinite([r['gt_x'], r['gt_y'], r['gt_z'], r['gt_roll'], r['gt_pitch'], r['gt_yaw']]))
    ]
    if bad_pose_rows:
        bad = bad_pose_rows[0]
        raise ValueError(
            'non-finite query ground-truth pose survived selection: '
            f"session={bad['query_session']} scan={bad['scan_file']} "
            f"pose=({bad['gt_x']},{bad['gt_y']},{bad['gt_z']},{bad['gt_roll']},{bad['gt_pitch']},{bad['gt_yaw']})"
        )

    if progress:
        print('[9.2] loading five-session packed history once from existing Step-8 cache...', flush=True)
    prepared = prepare_nclt_packed(data_root, names, cfg, cache_dir, progress_every=250)
    if progress:
        print('[9.2] building exact Step-8.4 session-local tilings once...', flush=True)
    session_blocks, session_block_ms = build_session_spatial_blocks(prepared, len(names), progress=progress)
    if progress:
        print('[9.2] finalizing frozen Step-8.5 time-aware BST map once...', flush=True)
    t0 = time.perf_counter_ns()
    core = TimeLocalAdaptiveCore(prepared, session_blocks, progress=progress)
    core_ms = (time.perf_counter_ns() - t0) / 1e6
    bst = BSTTimeLocalTimeIndexedMap(core, PrototypeBSTSpatialCodec())

    if progress:
        print('[9.2] rebuilding Jan-15 0.5m surfel sufficient statistics once...', flush=True)
    raw_surf = build_nclt_surfel_map(
        data_root, map_session, cfg, cache_dir,
        normal_min_points=base_cfg.normal_min_neighbors,
        normal_max_curvature=base_cfg.normal_max_curvature,
        normal_min_horizontal_component=base_cfg.normal_min_horizontal_component,
    )
    aligned = align_voxel_surfels_to_prepared(raw_surf, prepared, map_t)
    t0 = time.perf_counter_ns(); voxel_index = SurfelLocalMapIndex(aligned, bucket_width_m); voxel_index_ms = (time.perf_counter_ns() - t0) / 1e6

    if progress:
        print('[9.2] constructing frozen balanced K=4 hierarchy once, then materializing compact runtime...', flush=True)
    hierarchy = build_native_surfel_hierarchy(
        aligned, prepared, session_blocks, core, bst, map_t,
        base_cfg.normal_min_neighbors, base_cfg.normal_max_curvature,
        base_cfg.normal_min_horizontal_component, progress=progress,
    )
    view = build_hierarchical_lod_view(hierarchy, FROZEN_PROFILE, FROZEN_K)
    compact = materialize_lod_view(view)

    # Verify materialization at centers spanning every query session before
    # releasing the construction hierarchy.
    query_centers = []
    for qsi, session in enumerate(query_sessions):
        rng = random.Random(int(perturb_seed) + 1000 * qsi)
        for si, _ in enumerate(selected_by_session[session]):
            _, gt, _, _, _ = loaded[('cross_session', session, si)]
            _, gx, gy, _, _, _, _ = [float(v) for v in gt]
            query_centers.append((gx, gy))
            for _, xy, yaw in levels:
                dx, dy, _ = _trial_perturbation(xy, yaw, rng)
                query_centers.append((gx + dx, gy + dy))
    for si, _ in enumerate(same):
        _, gt, _, _, _ = loaded[('same_session_control', map_session, si)]
        _, gx, gy, _, _, _, _ = [float(v) for v in gt]
        query_centers.append((gx, gy))

    verification = verify_materialization(
        view, compact, bst, map_t, query_centers, local_map_radius_m,
        pyramid, base_cfg, max_pyramid_checks=min(8, max(1, len(query_sessions) * 2)),
    )
    if not verification.passed:
        raise AssertionError(f'Step 9.2 materialization verification failed: {verification}')

    compact_index = MaterializedLODBSTIndex(compact, bst, map_t)
    voxel_runtime_geom = _runtime_voxel_geometry_bytes(aligned)
    voxel_deployed = voxel_runtime_geom + voxel_index.nbytes()
    compact_geom = compact.geometry_payload_nbytes()
    compact_dir = compact.directory_nbytes()
    compact_total_with_bst = compact.runtime_nbytes() + bst.compact_bytes()
    construction_internal_nodes = int(hierarchy.internal_count)
    construction_payload_mib = hierarchy.internal_payload_nbytes() / (1024**2)
    construction_build_ms = float(hierarchy.build_ms)
    lod_decision_ms = float(view.decision_ms)

    map_rows = [
        {
            'map_geometry': 'voxel_surfel_0.5m', 'map_elements': int(len(aligned.linear_codes)),
            'runtime_geometry_mib': voxel_runtime_geom / (1024**2),
            'runtime_directory_or_index_mib': voxel_index.nbytes() / (1024**2),
            'bst_representation_mib': 0.0, 'deployed_runtime_total_mib': voxel_deployed / (1024**2),
            'construction_hierarchy_retained_at_runtime': False,
            'mean_voxels_per_primitive': 1.0,
        },
        {
            'map_geometry': 'bst_lod_k4_compact', 'map_elements': int(compact.map_elements),
            'runtime_geometry_mib': compact_geom / (1024**2),
            'runtime_directory_or_index_mib': compact_dir / (1024**2),
            'bst_representation_mib': bst.compact_bytes() / (1024**2),
            'deployed_runtime_total_mib': compact_total_with_bst / (1024**2),
            'construction_hierarchy_retained_at_runtime': False,
            'mean_voxels_per_primitive': float(hierarchy.leaf_count / compact.map_elements),
        },
    ]

    # Release all construction-only hierarchy references before the actual
    # validation runs. This mirrors deployment and prevents accidental use.
    hierarchy = None; view = None; raw_surf = None
    gc.collect()

    rows = []
    def run_geometry(geom, index):
        if progress:
            print(f'[9.2] validating {geom} across {len(query_sessions)} query sessions...', flush=True)
        for qsi, session in enumerate(query_sessions):
            rng = random.Random(int(perturb_seed) + 1000 * qsi)
            for si, _ in enumerate(selected_by_session[session]):
                sp, gt, gap, oi, body = loaded[('cross_session', session, si)]
                _, gx, gy, _, _, _, gyaw = [float(v) for v in gt]
                specs = [('zero_control', 0., 0., 0., 0., 0.)]
                for label, xy, yaw in levels:
                    dx, dy, dyaw = _trial_perturbation(xy, yaw, rng)
                    specs.append((label, xy, yaw, dx, dy, dyaw))
                for label, xy, yaw, dx, dy, dyaw in specs:
                    init = Pose3DOF(gx + dx, gy + dy, gyaw + math.radians(dyaw))
                    row = _run_trial(
                        map_geometry=geom, scenario='cross_session', scan_index=si, scan_path=sp,
                        body=body, gt=gt, gap=gap, original_scan_index=oi, init=init,
                        perturbation_label=label, xy_mag=xy, yaw_mag=yaw, dx=dx, dy=dy, dyaw_deg=dyaw,
                        map_index=index, local_map_radius_m=local_map_radius_m, pyramid=pyramid,
                        base_icp_config=base_cfg, success_translation_m=success_translation_m,
                        success_yaw_deg=success_yaw_deg,
                    )
                    row['map_session'] = map_session; row['query_session'] = session; row['session_pair'] = f'{map_session}->{session}'
                    rows.append(row)
        if include_same_session_zero:
            for si, _ in enumerate(same):
                sp, gt, gap, oi, body = loaded[('same_session_control', map_session, si)]
                _, gx, gy, _, _, _, gyaw = [float(v) for v in gt]
                init = Pose3DOF(gx, gy, gyaw)
                row = _run_trial(
                    map_geometry=geom, scenario='same_session_control', scan_index=si, scan_path=sp,
                    body=body, gt=gt, gap=gap, original_scan_index=oi, init=init,
                    perturbation_label='zero_control', xy_mag=0., yaw_mag=0., dx=0., dy=0., dyaw_deg=0.,
                    map_index=index, local_map_radius_m=local_map_radius_m, pyramid=pyramid,
                    base_icp_config=base_cfg, success_translation_m=success_translation_m,
                    success_yaw_deg=success_yaw_deg,
                )
                row['map_session'] = map_session; row['query_session'] = map_session; row['session_pair'] = f'{map_session}->{map_session}'
                rows.append(row)

    run_geometry('voxel_surfel_0.5m', voxel_index)
    run_geometry('bst_lod_k4_compact', compact_index)

    session_summary = _session_summary(rows)
    pooled = _pooled_summary(rows)
    paired = _paired_outcomes(rows)
    gates = _validation_gates(pooled)

    _write_csv(out / 'localization_trials.csv', rows)
    _write_csv(out / 'session_summary.csv', session_summary)
    _write_csv(out / 'pooled_summary.csv', pooled)
    _write_csv(out / 'paired_outcomes.csv', paired)
    _write_csv(out / 'validation_gates.csv', gates)
    _write_csv(out / 'query_scans.csv', scan_rows)
    _write_csv(out / 'deployment_memory.csv', map_rows)
    _write_csv(out / 'materialization_verification.csv', [{
        'payload_exact': verification.payload_exact,
        'directory_exact': verification.directory_exact,
        'query_exact': verification.query_exact,
        'pyramid_exact': verification.pyramid_exact,
        'query_checks': verification.checks,
    }])

    metadata = prepared.metadata()
    metadata.update({
        'step': '9.2',
        'purpose': 'Frozen multi-session held-out localization validation of compact balanced K=4 BST LOD versus 0.5m voxel surfels.',
        'history_sessions': names,
        'map_session': map_session,
        'map_session_index': map_t,
        'query_sessions': query_sessions,
        'query_count_per_session': int(query_count_per_session),
        'total_cross_session_scans': int(sum(len(v) for v in selected_by_session.values())),
        'same_session_query_count': len(same),
        'query_selection_roles': selection_roles,
        'excluded_tuning_query_scans_csv': str(exclude_query_scans_csv) if exclude_query_scans_csv else None,
        'excluded_cross_session_scan_names': len(tuning_cross_names),
        'frozen_geometry_profile': asdict(FROZEN_PROFILE),
        'frozen_k': FROZEN_K,
        'main_validation_perturbation': MAIN_PERTURBATION,
        'perturbations': [{'label': l, 'xy_m': x, 'yaw_deg': y} for l, x, y in levels],
        'perturb_seed_policy': 'base seed + 1000*query_session_index; identical perturbations across compared map geometries within each session',
        'localization_max_range_m': localization_max_range_m,
        'scan_downsample_m': scan_downsample_m,
        'max_scan_points': max_scan_points,
        'local_map_radius_m': local_map_radius_m,
        'success_thresholds': {'translation_m': success_translation_m, 'yaw_deg': success_yaw_deg},
        'base_icp_config': asdict(base_cfg),
        'pyramid': [asdict(x) for x in pyramid],
        'methods': ['voxel_surfel_0.5m', 'bst_lod_k4_compact'],
        'map_built_once_for_all_query_sessions': True,
        'construction_only': {
            'hierarchy_internal_nodes': construction_internal_nodes,
            'hierarchy_payload_mib': construction_payload_mib,
            'hierarchy_build_ms': construction_build_ms,
            'lod_decision_ms': lod_decision_ms,
            'materialize_ms': float(compact.materialize_ms),
            'retained_in_deployed_runtime': False,
            **_session_block_timing_metadata(session_block_ms),
            'adaptive_core_build_ms': float(core_ms),
            'voxel_index_build_ms': float(voxel_index_ms),
        },
        'materialization_verification': {
            'payload_exact': verification.payload_exact,
            'directory_exact': verification.directory_exact,
            'query_exact': verification.query_exact,
            'pyramid_exact': verification.pyramid_exact,
            'query_checks': verification.checks,
        },
        'deployment_memory': map_rows,
        'validation_gates': gates,
        'paired_outcomes': paired,
        'runtime': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine()},
        'scientific_caveat': 'Step 9.2 is frozen validation, not tuning. The balanced gate and K=4 are unchanged from Step 9.1.2. PrototypeBSTSpatialCodec remains a local research codec, not verified upstream BST-ID.',
    })
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return rows, session_summary, pooled, paired, gates, map_rows


__all__ = [
    'benchmark_step92', '_session_summary', '_pooled_summary', '_paired_outcomes',
    '_validation_gates', '_exact_mcnemar_p', '_session_block_timing_metadata', 'MAIN_PERTURBATION',
]
