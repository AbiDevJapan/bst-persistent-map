from __future__ import annotations

import csv
import gc
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmark.step901_benchmark import _trial_perturbation, parse_perturbations, _write_csv
from benchmark.step902_benchmark import _select_query_scans, _replay_query_scans, _load_and_prepare_scan
from benchmark.step903a_benchmark import _filter_replay_csv_for_cross_session
from benchmark.step91_benchmark import _run_trial, _summary
from localization.step90 import Pose3DOF
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID
from localization.step903b import SurfelLocalMapIndex
from localization.step91 import align_voxel_surfels_to_prepared
from localization.step911 import GeometryGateProfile
from localization.step912 import (
    build_native_surfel_hierarchy,
    build_hierarchical_lod_view,
    HierarchicalLODBSTIndex,
)
from localization.step913 import (
    materialize_lod_view,
    MaterializedLODBSTIndex,
    verify_materialization,
)
from realdata.nclt import (
    NCLTIngestConfig,
    _timestamp_from_scan,
    discover_nclt_session,
    load_ground_truth,
    _list_scan_files,
)
from realdata.nclt_packed import prepare_nclt_packed, _nearest_pose_rows
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec


FROZEN_PROFILE = GeometryGateProfile('balanced', 0.050, 0.025, 0.950, 2.0)
FROZEN_K = 4


def _read_excluded_scan_names(csv_path: str | Path | None, scenario: str = 'cross_session') -> set[str]:
    if csv_path is None:
        return set()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = set()
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            if 'scenario' in r and r['scenario'] and r['scenario'] != scenario:
                continue
            if r.get('scan_file'):
                out.add(r['scan_file'])
    return out




def _finite_pose_mask(pose_rows: np.ndarray) -> np.ndarray:
    """True for GT rows whose x/y/z/roll/pitch/yaw fields are all finite."""
    a = np.asarray(pose_rows, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError('pose_rows must be Nx7 or wider')
    return np.all(np.isfinite(a[:, 1:7]), axis=1)

def _select_query_scans_excluding(
    data_root: str | Path,
    session: str,
    count: int,
    max_pose_gap_us: int,
    excluded_names: set[str],
    exclude_ingest_stride: int | None = None,
):
    paths = discover_nclt_session(data_root, session)
    scans = _list_scan_files(paths.scan_dir)
    gt = load_ground_truth(paths.gt_csv)
    ts = np.asarray([_timestamp_from_scan(p) for p in scans], dtype=np.int64)
    pose_rows, gaps = _nearest_pose_rows(gt, ts)
    # A small number of NCLT ground-truth rows can contain non-finite pose
    # fields even though their timestamps are valid.  Timestamp-gap filtering
    # alone therefore is not sufficient for localization query selection.
    # Reject those rows here, before expensive map/hierarchy construction.
    finite_pose = _finite_pose_mask(pose_rows)
    valid = np.flatnonzero((gaps <= int(max_pose_gap_us)) & finite_pose)
    if exclude_ingest_stride is not None and int(exclude_ingest_stride) > 1:
        stride = int(exclude_ingest_stride)
        valid = valid[(valid % stride) != 0]
    if excluded_names:
        keep = np.asarray([scans[int(i)].name not in excluded_names for i in valid], dtype=bool)
        valid = valid[keep]
    if not len(valid):
        raise RuntimeError(f'No held-out query scans in {session} pass selection filters')
    n = min(int(count), len(valid))
    take = valid[np.linspace(0, len(valid) - 1, n, dtype=np.int64)]
    return [(scans[i], pose_rows[i], int(gaps[i]), int(i)) for i in take]


def _runtime_voxel_geometry_bytes(aligned) -> int:
    # Exact arrays consumed by prepare_surfel_pyramid at runtime.
    return int(aligned.counts.nbytes + aligned.centroids.nbytes + aligned.cov6.nbytes)


def _compare_reference_and_compact(rows: list[dict]) -> dict:
    ref = [r for r in rows if r['map_geometry'] == 'bst_lod_k4_reference']
    new = [r for r in rows if r['map_geometry'] == 'bst_lod_k4_compact']
    key = lambda r: (r['scenario'], int(r['scan_index']), r['perturbation'])
    a = {key(r): r for r in ref}; b = {key(r): r for r in new}
    if set(a) != set(b):
        raise AssertionError('reference/compact localization trial keys differ')

    exact_fields = [
        'success', 'converged', 'iterations', 'correspondences', 'normal_correspondences',
        'failure_reason', 'query_points', 'local_map_primitives',
    ]
    float_fields = [
        'final_translation_error_m', 'final_yaw_error_deg', 'pose_shift_from_initial_m',
        'yaw_shift_from_initial_deg', 'final_rmse_m', 'mean_abs_plane_residual_m',
        'estimated_x', 'estimated_y', 'estimated_yaw',
    ]
    max_abs = 0.0
    for k in sorted(a):
        ra, rb = a[k], b[k]
        for f in exact_fields:
            if ra.get(f) != rb.get(f):
                raise AssertionError(f'compact/reference mismatch {k} field={f}: {ra.get(f)!r} != {rb.get(f)!r}')
        for f in float_fields:
            xa, xb = float(ra[f]), float(rb[f])
            d = abs(xa - xb)
            max_abs = max(max_abs, d)
            if not np.isclose(xa, xb, rtol=0.0, atol=1e-12, equal_nan=True):
                raise AssertionError(f'compact/reference mismatch {k} field={f}: {xa} != {xb}')
        for lev in ('coarse', 'medium', 'fine'):
            for f in ('iterations', 'correspondences', 'normal_correspondences', 'failure_reason'):
                kk = f'level_{lev}_{f}'
                if ra.get(kk) != rb.get(kk):
                    raise AssertionError(f'compact/reference mismatch {k} field={kk}')
            for f in ('plane_rmse_m', 'mean_abs_plane_residual_m'):
                kk = f'level_{lev}_{f}'
                xa, xb = float(ra.get(kk, np.nan)), float(rb.get(kk, np.nan))
                if not np.isclose(xa, xb, rtol=0.0, atol=1e-12, equal_nan=True):
                    raise AssertionError(f'compact/reference mismatch {k} field={kk}: {xa} != {xb}')
    return {
        'reference_trials': len(ref),
        'compact_trials': len(new),
        'trial_outputs_equivalent': True,
        'max_abs_deterministic_float_difference': float(max_abs),
    }


def benchmark_step913(
    *, data_root, output_dir, history_sessions,
    map_session='2012-01-15', query_session='2012-02-02',
    voxel_size_m=0.5, scan_stride=10, min_range_m=2.0, max_range_m=80.0,
    min_points_per_voxel=1, max_pose_gap_us=100000, cache_dir='.cache/step8_1',
    query_count=20, same_session_query_count=20,
    query_scans_csv=None, exclude_query_scans_csv=None,
    perturbations='0.5:1', perturb_seed=42090,
    localization_max_range_m=30.0, scan_downsample_m=0.75, max_scan_points=8000,
    local_map_radius_m=40.0, bucket_width_m=8.0,
    success_translation_m=0.5, success_yaw_deg=1.0, icp_config=None,
    include_same_session_zero=True, include_reference=True, progress=True,
):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    names = list(history_sessions)
    if map_session not in names:
        raise ValueError('map_session must appear in history_sessions')
    if query_session == map_session:
        raise ValueError('cross-session query_session must differ from map_session')
    if abs(float(voxel_size_m) - 0.5) > 1e-9:
        raise ValueError('Step 9.1.3 keeps the frozen Step-8 0.5m canonical lattice')
    if query_scans_csv is not None and exclude_query_scans_csv is not None:
        raise ValueError('use either query_scans_csv replay OR exclude_query_scans_csv held-out selection, not both')

    map_t = names.index(map_session)
    cfg = NCLTIngestConfig(
        voxel_size_m=float(voxel_size_m), scan_stride=int(scan_stride),
        min_range_m=float(min_range_m), max_range_m=float(max_range_m),
        min_points_per_voxel=int(min_points_per_voxel), max_pose_gap_us=int(max_pose_gap_us),
    )
    base_cfg = icp_config or PointToPlaneICP3DOFConfig()
    levels = parse_perturbations(perturbations)
    pyramid = DEFAULT_PYRAMID

    if query_scans_csv is not None:
        replay = _filter_replay_csv_for_cross_session(query_scans_csv, out / '_cross_replay.csv')
        cross = _replay_query_scans(data_root, query_session, replay, query_count, max_pose_gap_us)
        selection_role = 'replay/equivalence'
        excluded_names = set()
    else:
        excluded_names = _read_excluded_scan_names(exclude_query_scans_csv, 'cross_session')
        if excluded_names:
            cross = _select_query_scans_excluding(
                data_root, query_session, query_count, max_pose_gap_us, excluded_names,
            )
            selection_role = 'held-out from tuning CSV'
        else:
            cross = _select_query_scans(data_root, query_session, query_count, max_pose_gap_us)
            selection_role = 'deterministic development selection'

    same_excluded = _read_excluded_scan_names(exclude_query_scans_csv, 'same_session_control') if exclude_query_scans_csv else set()
    if include_same_session_zero:
        if same_excluded:
            same = _select_query_scans_excluding(
                data_root, map_session, same_session_query_count, max_pose_gap_us,
                same_excluded, exclude_ingest_stride=scan_stride,
            )
        else:
            same = _select_query_scans(
                data_root, map_session, same_session_query_count, max_pose_gap_us,
                exclude_ingest_stride=scan_stride,
            )
    else:
        same = []

    prep_scan = SimpleNamespace(config=cfg)
    scan_rows = []; loaded = {}
    for scenario, selected in [('cross_session', cross), ('same_session_control', same)]:
        for si, (sp, gt, gap, oi) in enumerate(selected):
            body, raw = _load_and_prepare_scan(
                sp, prep_scan, localization_max_range_m, scan_downsample_m, max_scan_points,
            )
            loaded[(scenario, si)] = (sp, gt, gap, oi, body)
            _, gx, gy, gz, roll, pitch, gyaw = [float(v) for v in gt]
            scan_rows.append({
                'scenario': scenario, 'scan_index': si, 'original_scan_index': int(oi),
                'scan_file': sp.name, 'timestamp_us': int(_timestamp_from_scan(sp)),
                'pose_gap_us': int(gap), 'points_after_range': int(raw),
                'points_used': int(len(body)), 'gt_x': gx, 'gt_y': gy, 'gt_z': gz,
                'gt_roll': roll, 'gt_pitch': pitch, 'gt_yaw': gyaw,
                'directly_ingested_scan': bool((oi % scan_stride) == 0),
                'excluded_by_tuning_csv': bool(sp.name in excluded_names or sp.name in same_excluded),
            })

    if progress:
        print('[9.1.3] loading five-session packed history from existing Step-8 cache...', flush=True)
    prepared = prepare_nclt_packed(data_root, names, cfg, cache_dir, progress_every=250)
    if progress:
        print('[9.1.3] building exact Step-8.4 session-local tilings...', flush=True)
    session_blocks, session_block_ms = build_session_spatial_blocks(prepared, len(names), progress=progress)
    if progress:
        print('[9.1.3] finalizing unchanged Step-8.5 time-aware BST map...', flush=True)
    t0 = time.perf_counter_ns()
    core = TimeLocalAdaptiveCore(prepared, session_blocks, progress=progress)
    core_ms = (time.perf_counter_ns() - t0) / 1e6
    bst = BSTTimeLocalTimeIndexedMap(core, PrototypeBSTSpatialCodec())

    if progress:
        print('[9.1.3] rebuilding 0.5m map-session surfel sufficient statistics in memory...', flush=True)
    raw_surf = build_nclt_surfel_map(
        data_root, map_session, cfg, cache_dir,
        normal_min_points=base_cfg.normal_min_neighbors,
        normal_max_curvature=base_cfg.normal_max_curvature,
        normal_min_horizontal_component=base_cfg.normal_min_horizontal_component,
    )
    aligned = align_voxel_surfels_to_prepared(raw_surf, prepared, map_t)
    t0 = time.perf_counter_ns()
    voxel_index = SurfelLocalMapIndex(aligned, bucket_width_m)
    voxel_index_ms = (time.perf_counter_ns() - t0) / 1e6

    if progress:
        print('[9.1.3] constructing temporary merge hierarchy and frozen balanced K=4 LOD...', flush=True)
    hierarchy = build_native_surfel_hierarchy(
        aligned, prepared, session_blocks, core, bst, map_t,
        base_cfg.normal_min_neighbors, base_cfg.normal_max_curvature,
        base_cfg.normal_min_horizontal_component, progress=progress,
    )
    view = build_hierarchical_lod_view(hierarchy, FROZEN_PROFILE, FROZEN_K)
    compact = materialize_lod_view(view)

    # Verify on all GT centers plus the deterministic perturbed centers used by trials.
    query_centers = []
    rng = random.Random(int(perturb_seed))
    for si, _ in enumerate(cross):
        _, gt, _, _, _ = loaded[('cross_session', si)]
        _, gx, gy, _, _, _, _ = [float(v) for v in gt]
        query_centers.append((gx, gy))
        for _, xy, yaw in levels:
            dx, dy, _ = _trial_perturbation(xy, yaw, rng)
            query_centers.append((gx + dx, gy + dy))
    for si, _ in enumerate(same):
        _, gt, _, _, _ = loaded[('same_session_control', si)]
        _, gx, gy, _, _, _, _ = [float(v) for v in gt]
        query_centers.append((gx, gy))

    verification = verify_materialization(
        view, compact, bst, map_t, query_centers, local_map_radius_m,
        pyramid, base_cfg, max_pyramid_checks=3,
    )
    if not verification.passed:
        raise AssertionError(f'materialized K=4 verification failed: {verification}')

    ref_index = HierarchicalLODBSTIndex(view, bst, map_t) if include_reference else None
    compact_index = MaterializedLODBSTIndex(compact, bst, map_t)

    voxel_runtime_geom = _runtime_voxel_geometry_bytes(aligned)
    voxel_deployed = voxel_runtime_geom + voxel_index.nbytes()
    compact_geom = compact.geometry_payload_nbytes()
    compact_dir = compact.directory_nbytes()
    compact_lod_runtime = compact.runtime_nbytes()
    compact_total_with_bst = compact_lod_runtime + bst.compact_bytes()

    map_rows = [
        {
            'map_geometry': 'voxel_surfel_0.5m',
            'map_elements': int(len(aligned.linear_codes)),
            'runtime_geometry_mib': voxel_runtime_geom / (1024**2),
            'runtime_directory_or_index_mib': voxel_index.nbytes() / (1024**2),
            'bst_representation_mib': 0.0,
            'deployed_runtime_total_mib': voxel_deployed / (1024**2),
            'construction_hierarchy_mib': 0.0,
            'construction_hierarchy_retained_at_runtime': False,
            'materialize_ms': 0.0,
            'mean_voxels_per_primitive': 1.0,
        },
        {
            'map_geometry': 'bst_lod_k4_reference',
            'map_elements': int(view.map_elements),
            'runtime_geometry_mib': view.selected_payload_nbytes(False) / (1024**2),
            'runtime_directory_or_index_mib': (view.directory_nbytes() + hierarchy.block_map.global_block_ids.nbytes) / (1024**2),
            'bst_representation_mib': bst.compact_bytes() / (1024**2),
            'deployed_runtime_total_mib': float('nan'),
            'construction_hierarchy_mib': hierarchy.internal_payload_nbytes() / (1024**2),
            'construction_hierarchy_retained_at_runtime': True,
            'materialize_ms': 0.0,
            'mean_voxels_per_primitive': float(hierarchy.leaf_count / view.map_elements),
        },
        {
            'map_geometry': 'bst_lod_k4_compact',
            'map_elements': int(compact.map_elements),
            'runtime_geometry_mib': compact_geom / (1024**2),
            'runtime_directory_or_index_mib': compact_dir / (1024**2),
            'bst_representation_mib': bst.compact_bytes() / (1024**2),
            'deployed_runtime_total_mib': compact_total_with_bst / (1024**2),
            'construction_hierarchy_mib': hierarchy.internal_payload_nbytes() / (1024**2),
            'construction_hierarchy_retained_at_runtime': False,
            'materialize_ms': float(compact.materialize_ms),
            'mean_voxels_per_primitive': float(hierarchy.leaf_count / compact.map_elements),
        },
    ]

    rows = []
    def run_geometry(geom, index):
        if progress:
            print(f'[9.1.3] localizing with {geom}...', flush=True)
        def scenario_run(scenario, selected, perturb, seed):
            rng2 = random.Random(int(seed))
            for si, _ in enumerate(selected):
                sp, gt, gap, oi, body = loaded[(scenario, si)]
                _, gx, gy, _, _, _, gyaw = [float(v) for v in gt]
                specs = [('zero_control', 0., 0., 0., 0., 0.)]
                if perturb:
                    for label, xy, yaw in levels:
                        dx, dy, dyaw = _trial_perturbation(xy, yaw, rng2)
                        specs.append((label, xy, yaw, dx, dy, dyaw))
                for label, xy, yaw, dx, dy, dyaw in specs:
                    init = Pose3DOF(gx + dx, gy + dy, gyaw + math.radians(dyaw))
                    rows.append(_run_trial(
                        map_geometry=geom, scenario=scenario, scan_index=si, scan_path=sp,
                        body=body, gt=gt, gap=gap, original_scan_index=oi, init=init,
                        perturbation_label=label, xy_mag=xy, yaw_mag=yaw,
                        dx=dx, dy=dy, dyaw_deg=dyaw, map_index=index,
                        local_map_radius_m=local_map_radius_m, pyramid=pyramid,
                        base_icp_config=base_cfg, success_translation_m=success_translation_m,
                        success_yaw_deg=success_yaw_deg,
                    ))
        scenario_run('cross_session', cross, True, perturb_seed)
        if include_same_session_zero:
            scenario_run('same_session_control', same, False, perturb_seed + 902)

    run_geometry('voxel_surfel_0.5m', voxel_index)
    if include_reference:
        run_geometry('bst_lod_k4_reference', ref_index)

    # Freeze all construction-only numbers before releasing the hierarchy.  The
    # compact localization run below then has no Python reference to the merge
    # forest or selected node-ID table, matching the intended deployed design.
    construction_internal_nodes = int(hierarchy.internal_count)
    construction_payload_mib = hierarchy.internal_payload_nbytes() / (1024**2)
    construction_build_ms = float(hierarchy.build_ms)
    lod_decision_ms = float(view.decision_ms)
    ref_index = None; view = None; hierarchy = None; raw_surf = None
    gc.collect()

    run_geometry('bst_lod_k4_compact', compact_index)

    localization_equivalence = _compare_reference_and_compact(rows) if include_reference else {
        'reference_trials': 0, 'compact_trials': len([r for r in rows if r['map_geometry']=='bst_lod_k4_compact']),
        'trial_outputs_equivalent': None, 'max_abs_deterministic_float_difference': None,
    }

    summary = _summary(rows)
    _write_csv(out / 'localization_trials.csv', rows)
    _write_csv(out / 'localization_summary.csv', summary)
    _write_csv(out / 'query_scans.csv', scan_rows)
    _write_csv(out / 'deployment_memory.csv', map_rows)
    _write_csv(out / 'materialization_verification.csv', [{
        'payload_exact': verification.payload_exact,
        'directory_exact': verification.directory_exact,
        'query_exact': verification.query_exact,
        'pyramid_exact': verification.pyramid_exact,
        'query_checks': verification.checks,
        **localization_equivalence,
    }])
    tmp = out / '_cross_replay.csv'
    if tmp.exists():
        tmp.unlink()

    metadata = prepared.metadata()
    metadata.update({
        'step': '9.1.3',
        'purpose': 'Materialize the frozen Step-9.1.2 balanced K=4 localization LOD into a standalone hierarchy-free runtime table and count actual deployed arrays.',
        'history_sessions': names,
        'map_session': map_session,
        'map_session_index': map_t,
        'query_session': query_session,
        'query_count': len(cross),
        'same_session_query_count': len(same),
        'query_selection_role': selection_role,
        'query_scans_replay_csv': str(query_scans_csv) if query_scans_csv else None,
        'excluded_tuning_query_scans_csv': str(exclude_query_scans_csv) if exclude_query_scans_csv else None,
        'excluded_cross_session_scan_names': len(excluded_names),
        'frozen_geometry_profile': asdict(FROZEN_PROFILE),
        'frozen_k': FROZEN_K,
        'perturbations': [{'label': l, 'xy_m': x, 'yaw_deg': y} for l, x, y in levels],
        'localization_max_range_m': localization_max_range_m,
        'scan_downsample_m': scan_downsample_m,
        'max_scan_points': max_scan_points,
        'local_map_radius_m': local_map_radius_m,
        'success_thresholds': {'translation_m': success_translation_m, 'yaw_deg': success_yaw_deg},
        'base_icp_config': asdict(base_cfg),
        'pyramid': [asdict(x) for x in pyramid],
        'materialized_runtime': {
            'stores': ['uint32 point count', 'float32 centroid xyz', 'float32 covariance cov6', 'global BST block id per root', 'root start', 'root count'],
            'does_not_store': ['construction merge forest', 'unselected hierarchy nodes', 'selected hierarchy node IDs', 'precomputed normals/curvature'],
            'reason_covariance_is_counted': 'the unchanged coarse-to-fine matcher re-aggregates selected surfels at each map pyramid level and therefore consumes count+centroid+covariance',
            'online_path': 'Step-8.5 time-aware root query -> compact root directory -> materialized surfel rows -> radius filter -> unchanged matcher',
        },
        'construction_only': {
            'hierarchy_internal_nodes': construction_internal_nodes,
            'hierarchy_payload_mib': construction_payload_mib,
            'hierarchy_build_ms': construction_build_ms,
            'lod_decision_ms': lod_decision_ms,
            'materialize_ms': float(compact.materialize_ms),
            'retained_in_deployed_runtime': False,
        },
        'materialization_verification': {
            'payload_exact': verification.payload_exact,
            'directory_exact': verification.directory_exact,
            'query_exact': verification.query_exact,
            'pyramid_exact': verification.pyramid_exact,
            'query_checks': verification.checks,
            **localization_equivalence,
        },
        'deployment_memory': map_rows,
        'runtime': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine()},
        'scientific_caveat': 'Step 9.1.3 freezes the balanced gate and K=4 selected on development data. Replaying those scans is for equivalence only; final performance claims require held-out scans/session pairs. PrototypeBSTSpatialCodec remains a local research codec, not verified upstream BST-ID.',
    })
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    return rows, summary, map_rows, metadata['materialization_verification']


__all__ = ['benchmark_step913', 'FROZEN_PROFILE', 'FROZEN_K']
