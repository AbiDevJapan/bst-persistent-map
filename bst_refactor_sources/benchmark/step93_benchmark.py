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

from benchmark.step901_benchmark import _trial_perturbation, parse_perturbations, _write_csv
from benchmark.step902_benchmark import _load_and_prepare_scan
from benchmark.step91_benchmark import _run_trial
from benchmark.step913_benchmark import (
    FROZEN_PROFILE, FROZEN_K, _read_excluded_scan_names,
    _select_query_scans_excluding, _runtime_voxel_geometry_bytes,
)
from benchmark.step92_benchmark import (
    MAIN_PERTURBATION, _session_summary, _pooled_summary, _exact_mcnemar_p,
    _session_block_timing_metadata,
)
from localization.step90 import Pose3DOF, pose_error_3dof
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID
from localization.step903b import SurfelLocalMapIndex, CoarseToFineSurfelResult
from localization.step91 import align_voxel_surfels_to_prepared
from localization.step912 import build_native_surfel_hierarchy, build_hierarchical_lod_view
from localization.step913 import materialize_lod_view, MaterializedLODBSTIndex, verify_materialization
from localization.step93 import (
    build_causal_persistence, PersistenceWeightedLODBSTIndex,
    scan_to_map_coarse_to_fine_persistent_3dof,
)
from realdata.nclt import NCLTIngestConfig, _timestamp_from_scan
from realdata.nclt_packed import prepare_nclt_packed
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec


UNWEIGHTED = 'bst_lod_k4_compact'
PERSISTENT = 'bst_lod_k4_persistence'
VOXEL = 'voxel_surfel_0.5m'


def _union_excluded(paths, scenario):
    out = set()
    for p in paths or []:
        if p:
            out |= _read_excluded_scan_names(p, scenario)
    return out


def _run_persistence_trial(*, map_geometry, scenario, scan_index, scan_path, body, gt, gap, original_scan_index,
                           init, perturbation_label, xy_mag, yaw_mag, dx, dy, dyaw_deg, map_index,
                           local_map_radius_m, pyramid, base_icp_config, success_translation_m, success_yaw_deg):
    _, gx, gy, gz, roll, pitch, gyaw = [float(v) for v in gt]
    gt_pose = Pose3DOF(gx, gy, gyaw)
    init_te, init_ye = pose_error_3dof(init, gt_pose)
    total0 = time.perf_counter_ns()
    tr0 = time.perf_counter_ns(); ids = map_index.query_radius_ids(init.x, init.y, local_map_radius_m); retrieval_ms = (time.perf_counter_ns()-tr0)/1e6
    tp0 = time.perf_counter_ns(); prepared = map_index.prepare_pyramid(ids, pyramid, base_icp_config) if len(ids) else []; map_prep_ms = (time.perf_counter_ns()-tp0)/1e6
    if not prepared:
        result = CoarseToFineSurfelResult(init, False, 0, 0, 0, float('inf'), float('inf'), 0.0, 'empty_local_map', [])
    else:
        result = scan_to_map_coarse_to_fine_persistent_3dof(body, prepared, init, gz, roll, pitch, base_icp_config)
    total_ms = (time.perf_counter_ns()-total0)/1e6
    fte, fye = pose_error_3dof(result.pose, gt_pose); sm, sy = pose_error_3dof(result.pose, init)
    row = {
        'map_geometry': map_geometry, 'scenario': scenario, 'scan_index': int(scan_index), 'original_scan_index': int(original_scan_index),
        'scan_file': scan_path.name, 'timestamp_us': int(_timestamp_from_scan(scan_path)), 'pose_gap_us': int(gap),
        'matcher': 'coarse_to_fine_surfel_point_to_plane_persistence_weighted', 'perturbation': perturbation_label,
        'perturb_xy_m': float(xy_mag), 'perturb_yaw_deg': float(yaw_mag), 'actual_dx_m': float(dx), 'actual_dy_m': float(dy),
        'actual_dyaw_deg': float(dyaw_deg), 'initial_translation_error_m': init_te, 'initial_yaw_error_deg': init_ye,
        'final_translation_error_m': fte, 'final_yaw_error_deg': fye, 'pose_shift_from_initial_m': sm, 'yaw_shift_from_initial_deg': sy,
        'success': bool(fte <= success_translation_m and fye <= success_yaw_deg), 'converged': bool(result.converged),
        'iterations': int(result.iterations), 'correspondences': int(result.correspondences), 'normal_correspondences': int(result.normal_correspondences),
        'final_rmse_m': float(result.rmse_m), 'final_plane_rmse_m': float(result.rmse_m),
        'mean_abs_plane_residual_m': float(result.mean_abs_plane_residual_m), 'failure_reason': result.failure_reason,
        'query_points': int(len(body)), 'local_map_primitives': int(len(ids)), 'retrieval_ms': retrieval_ms,
        'map_prep_ms': map_prep_ms, 'match_ms': float(result.match_ms), 'total_ms': total_ms,
        'initial_x': init.x, 'initial_y': init.y, 'initial_yaw': init.yaw,
        'estimated_x': result.pose.x, 'estimated_y': result.pose.y, 'estimated_yaw': result.pose.yaw,
        'gt_x': gx, 'gt_y': gy, 'gt_yaw': gyaw,
    }
    for k, v in (getattr(map_index, 'last_stats', {}) or {}).items(): row[f'retrieval_{k}'] = v
    for name, lev in result.level_results:
        p = f'level_{name}_'; row[p+'converged'] = bool(lev.converged); row[p+'iterations'] = int(lev.iterations)
        row[p+'correspondences'] = int(lev.correspondences); row[p+'normal_correspondences'] = int(lev.normal_correspondences)
        row[p+'plane_rmse_m'] = float(lev.rmse_m); row[p+'mean_abs_plane_residual_m'] = float(lev.mean_abs_plane_residual_m)
        row[p+'match_ms'] = float(lev.match_ms); row[p+'failure_reason'] = lev.failure_reason
    return row


def _paired_history_outcomes(rows):
    cross = [r for r in rows if r['scenario'] == 'cross_session' and r['map_geometry'] in {UNWEIGHTED, PERSISTENT}]
    pairs = sorted(set((r['query_session'], r['perturbation']) for r in cross))
    out = []
    for session, perturb in pairs + [('LATER', p) for p in sorted(set(r['perturbation'] for r in cross))] + [('ALL', p) for p in sorted(set(r['perturbation'] for r in cross))]:
        if session == 'LATER':
            sessions = sorted(set(r['query_session'] for r in cross))[1:]
            q = [r for r in cross if r['perturbation'] == perturb and r['query_session'] in sessions]
        else:
            q = [r for r in cross if r['perturbation'] == perturb and (session == 'ALL' or r['query_session'] == session)]
        key = lambda r: (r['query_session'], int(r['original_scan_index']), r['perturbation'])
        ua = {key(r): bool(r['success']) for r in q if r['map_geometry'] == UNWEIGHTED}
        pa = {key(r): bool(r['success']) for r in q if r['map_geometry'] == PERSISTENT}
        if set(ua) != set(pa): raise AssertionError(f'persistence paired keys differ for {session}/{perturb}')
        both = sum(ua[k] and pa[k] for k in ua); uonly = sum(ua[k] and not pa[k] for k in ua)
        ponly = sum((not ua[k]) and pa[k] for k in ua); neither = sum((not ua[k]) and (not pa[k]) for k in ua)
        out.append({
            'query_session': session, 'perturbation': perturb, 'paired_trials': len(ua), 'both_success': both,
            'unweighted_only_success': uonly, 'persistence_only_success': ponly, 'neither_success': neither,
            'net_persistence_success_difference': ponly-uonly, 'exact_mcnemar_p': _exact_mcnemar_p(uonly, ponly),
        })
    return out


def _verify_null_equivalence(rows, first_query_session, *, atol=0.0):
    q = [r for r in rows if r['scenario'] == 'cross_session' and r['query_session'] == first_query_session and r['map_geometry'] in {UNWEIGHTED, PERSISTENT}]
    key = lambda r: (int(r['original_scan_index']), r['perturbation'])
    a = {key(r): r for r in q if r['map_geometry'] == UNWEIGHTED}; b = {key(r): r for r in q if r['map_geometry'] == PERSISTENT}
    if set(a) != set(b): raise AssertionError('first-session null-control trial keys differ')
    exact_fields = ['success','converged','iterations','correspondences','normal_correspondences','failure_reason','local_map_primitives']
    float_fields = ['final_translation_error_m','final_yaw_error_deg','estimated_x','estimated_y','estimated_yaw','final_plane_rmse_m','mean_abs_plane_residual_m']
    max_abs = 0.0
    for k in a:
        for f in exact_fields:
            if a[k].get(f) != b[k].get(f): return {'passed': False, 'field': f, 'key': str(k), 'max_abs_float_difference': max_abs}
        for f in float_fields:
            x, y = float(a[k][f]), float(b[k][f]); d = abs(x-y); max_abs=max(max_abs,d)
            if not np.isclose(x,y,rtol=0.0,atol=atol,equal_nan=True): return {'passed': False, 'field': f, 'key': str(k), 'max_abs_float_difference': max_abs}
    return {'passed': True, 'field': '', 'key': '', 'max_abs_float_difference': max_abs, 'trials_per_method': len(a)}


def _persistence_gates(session_summary, paired, first_query_session):
    main = [r for r in session_summary if r['scenario']=='cross_session' and r['perturbation']==MAIN_PERTURBATION]
    later_sessions = sorted(set(r['query_session'] for r in main if r['query_session'] != first_query_session))
    u = [r for r in main if r['map_geometry']==UNWEIGHTED and r['query_session'] in later_sessions]
    p = [r for r in main if r['map_geometry']==PERSISTENT and r['query_session'] in later_sessions]
    if not u or not p: return []
    # Equal trial counts by construction; use trial-count-weighted success.
    su = sum(r['success_rate']*r['trials'] for r in u)/sum(r['trials'] for r in u)
    sp = sum(r['success_rate']*r['trials'] for r in p)/sum(r['trials'] for r in p)
    tu = statistics.median([r['median_total_ms'] for r in u]); tp = statistics.median([r['median_total_ms'] for r in p])
    pu = statistics.mean([r['mean_local_map_primitives'] for r in u]); pp = statistics.mean([r['mean_local_map_primitives'] for r in p])
    pair = next((r for r in paired if r['query_session']=='LATER' and r['perturbation']==MAIN_PERTURBATION), None)
    ratio = sp/su if su>0 else float('nan'); time_ratio = tp/tu if tu>0 else float('nan'); prim_ratio=pp/pu if pu>0 else float('nan')
    net = int(pair['net_persistence_success_difference']) if pair else 0
    row = {
        'perturbation': MAIN_PERTURBATION, 'later_sessions': ';'.join(later_sessions),
        'unweighted_success_rate': su, 'persistence_success_rate': sp, 'success_ratio': ratio,
        'net_persistence_success_difference': net, 'exact_mcnemar_p': pair['exact_mcnemar_p'] if pair else float('nan'),
        'median_total_time_ratio_persistence_to_unweighted': time_ratio,
        'primitive_ratio_persistence_to_unweighted': prim_ratio,
        'meets_success_retention_0.95': bool(np.isfinite(ratio) and ratio>=0.95),
        'nonnegative_paired_net_gain': bool(net>=0),
        'strict_success_gain_observed': bool(net>0),
        'meets_runtime_overhead_1.10': bool(np.isfinite(time_ratio) and time_ratio<=1.10),
        'primitive_count_preserved': bool(np.isclose(prim_ratio,1.0,rtol=0,atol=1e-12)),
    }
    row['safety_gate_pass'] = bool(row['meets_success_retention_0.95'] and row['meets_runtime_overhead_1.10'] and row['primitive_count_preserved'])
    row['temporal_gain_gate_pass'] = bool(row['safety_gate_pass'] and row['nonnegative_paired_net_gain'])
    return [row]


def benchmark_step93(
    *, data_root, output_dir, history_sessions,
    map_session='2012-01-15', query_sessions=('2012-02-02','2012-03-17','2012-05-11','2012-06-15'),
    voxel_size_m=0.5, scan_stride=10, min_range_m=2.0, max_range_m=80.0,
    min_points_per_voxel=1, max_pose_gap_us=100000, cache_dir='.cache/step8_1',
    query_count_per_session=50, same_session_query_count=50, exclude_query_scans_csv=(),
    perturbations='0.5:1', perturb_seed=93090, localization_max_range_m=30.0,
    scan_downsample_m=0.75, max_scan_points=8000, local_map_radius_m=40.0,
    bucket_width_m=8.0, success_translation_m=0.5, success_yaw_deg=1.0,
    icp_config=None, include_same_session_zero=True, include_voxel_baseline=False, progress=True,
):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    names=list(history_sessions); query_sessions=list(query_sessions)
    if map_session not in names: raise ValueError('map_session must appear in history_sessions')
    if not query_sessions: raise ValueError('at least one query session is required')
    if any(s not in names for s in query_sessions): raise ValueError('all query sessions must appear in history_sessions')
    map_t=names.index(map_session); qidx=[names.index(s) for s in query_sessions]
    if any(q<=map_t for q in qidx): raise ValueError('Step 9.3 causal validation requires query sessions after map_session in history_sessions order')
    if qidx != sorted(qidx): raise ValueError('query_sessions must be chronological in history_sessions order')
    if abs(float(voxel_size_m)-0.5)>1e-9: raise ValueError('Step 9.3 keeps frozen 0.5m canonical lattice')
    cfg=NCLTIngestConfig(voxel_size_m=float(voxel_size_m),scan_stride=int(scan_stride),min_range_m=float(min_range_m),max_range_m=float(max_range_m),min_points_per_voxel=int(min_points_per_voxel),max_pose_gap_us=int(max_pose_gap_us))
    base_cfg=icp_config or PointToPlaneICP3DOFConfig(); levels=parse_perturbations(perturbations); pyramid=DEFAULT_PYRAMID
    if not any(l==MAIN_PERTURBATION for l,_,_ in levels): raise ValueError(f'main perturbation {MAIN_PERTURBATION} must be included')

    excluded_cross=_union_excluded(exclude_query_scans_csv,'cross_session'); excluded_same=_union_excluded(exclude_query_scans_csv,'same_session_control')
    selected_by_session={s:_select_query_scans_excluding(data_root,s,query_count_per_session,max_pose_gap_us,excluded_cross) for s in query_sessions}
    same=_select_query_scans_excluding(data_root,map_session,same_session_query_count,max_pose_gap_us,excluded_same,exclude_ingest_stride=scan_stride) if include_same_session_zero else []
    prep_scan=SimpleNamespace(config=cfg); loaded={}; scan_rows=[]
    for session,selected in selected_by_session.items():
        for si,(sp,gt,gap,oi) in enumerate(selected):
            body,raw=_load_and_prepare_scan(sp,prep_scan,localization_max_range_m,scan_downsample_m,max_scan_points); loaded[('cross_session',session,si)]=(sp,gt,gap,oi,body)
            _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
            scan_rows.append({'scenario':'cross_session','map_session':map_session,'query_session':session,'scan_index':si,'original_scan_index':int(oi),'scan_file':sp.name,'timestamp_us':int(_timestamp_from_scan(sp)),'pose_gap_us':int(gap),'points_after_range':int(raw),'points_used':int(len(body)),'gt_x':gx,'gt_y':gy,'gt_z':gz,'gt_roll':roll,'gt_pitch':pitch,'gt_yaw':gyaw,'excluded_prior_name_match':bool(sp.name in excluded_cross)})
    for si,(sp,gt,gap,oi) in enumerate(same):
        body,raw=_load_and_prepare_scan(sp,prep_scan,localization_max_range_m,scan_downsample_m,max_scan_points); loaded[('same_session_control',map_session,si)]=(sp,gt,gap,oi,body)
        _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
        scan_rows.append({'scenario':'same_session_control','map_session':map_session,'query_session':map_session,'scan_index':si,'original_scan_index':int(oi),'scan_file':sp.name,'timestamp_us':int(_timestamp_from_scan(sp)),'pose_gap_us':int(gap),'points_after_range':int(raw),'points_used':int(len(body)),'gt_x':gx,'gt_y':gy,'gt_z':gz,'gt_roll':roll,'gt_pitch':pitch,'gt_yaw':gyaw,'excluded_prior_name_match':bool(sp.name in excluded_same)})
    if any(r['excluded_prior_name_match'] for r in scan_rows): raise AssertionError('Step 9.3 selected a previously excluded scan')
    for r in scan_rows:
        if not np.all(np.isfinite([r['gt_x'],r['gt_y'],r['gt_z'],r['gt_roll'],r['gt_pitch'],r['gt_yaw']])): raise ValueError(f"non-finite query pose: {r['query_session']} {r['scan_file']}")

    if progress: print('[9.3] loading packed five-session history once...',flush=True)
    prepared=prepare_nclt_packed(data_root,names,cfg,cache_dir,progress_every=250)
    if progress: print('[9.3] building frozen Step-8.4/8.5 map once...',flush=True)
    session_blocks,session_block_ms=build_session_spatial_blocks(prepared,len(names),progress=progress)
    t0=time.perf_counter_ns(); core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=progress); core_ms=(time.perf_counter_ns()-t0)/1e6
    bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())
    if progress: print('[9.3] rebuilding frozen Jan map surfel sufficient statistics...',flush=True)
    raw_surf=build_nclt_surfel_map(data_root,map_session,cfg,cache_dir,normal_min_points=base_cfg.normal_min_neighbors,normal_max_curvature=base_cfg.normal_max_curvature,normal_min_horizontal_component=base_cfg.normal_min_horizontal_component)
    aligned=align_voxel_surfels_to_prepared(raw_surf,prepared,map_t)
    t0=time.perf_counter_ns(); voxel_index=SurfelLocalMapIndex(aligned,bucket_width_m); voxel_index_ms=(time.perf_counter_ns()-t0)/1e6

    if progress: print('[9.3] building frozen balanced K=4 LOD and causal persistence scores...',flush=True)
    hierarchy=build_native_surfel_hierarchy(aligned,prepared,session_blocks,core,bst,map_t,base_cfg.normal_min_neighbors,base_cfg.normal_max_curvature,base_cfg.normal_min_horizontal_component,progress=progress)
    view=build_hierarchical_lod_view(hierarchy,FROZEN_PROFILE,FROZEN_K); compact=materialize_lod_view(view)
    causal=build_causal_persistence(view,prepared,map_t,qidx)

    # Small equivalence sample before construction hierarchy is released.
    centers=[]
    for s in query_sessions:
        for si in range(min(2,len(selected_by_session[s]))):
            _,gt,_,_,_=loaded[('cross_session',s,si)]; centers.append((float(gt[1]),float(gt[2])))
    verification=verify_materialization(view,compact,bst,map_t,centers,local_map_radius_m,pyramid,base_cfg,max_pyramid_checks=min(4,len(centers)))
    if not verification.passed: raise AssertionError('Step 9.3 compact materialization verification failed')

    compact_index=MaterializedLODBSTIndex(compact,bst,map_t)
    persistence_indices={q:PersistenceWeightedLODBSTIndex(compact,bst,map_t,causal.scores_q8[q],q-map_t) for q in qidx}
    all_one=np.full(compact.map_elements,255,dtype=np.uint8)
    same_persistence_index=PersistenceWeightedLODBSTIndex(compact,bst,map_t,all_one,1)

    voxel_runtime_geom=_runtime_voxel_geometry_bytes(aligned); voxel_total=voxel_runtime_geom+voxel_index.nbytes()
    compact_total=compact.runtime_nbytes()+bst.compact_bytes(); score_bytes=causal.score_bytes_per_active_session()
    memory=[
        {'map_geometry':UNWEIGHTED,'runtime_geometry_mib':compact.geometry_payload_nbytes()/(1024**2),'runtime_directory_or_index_mib':compact.directory_nbytes()/(1024**2),'persistence_score_mib':0.0,'bst_representation_mib':bst.compact_bytes()/(1024**2),'deployed_runtime_total_mib':compact_total/(1024**2)},
        {'map_geometry':PERSISTENT,'runtime_geometry_mib':compact.geometry_payload_nbytes()/(1024**2),'runtime_directory_or_index_mib':compact.directory_nbytes()/(1024**2),'persistence_score_mib':score_bytes/(1024**2),'bst_representation_mib':bst.compact_bytes()/(1024**2),'deployed_runtime_total_mib':(compact_total+score_bytes)/(1024**2)},
    ]
    if include_voxel_baseline: memory.insert(0,{'map_geometry':VOXEL,'runtime_geometry_mib':voxel_runtime_geom/(1024**2),'runtime_directory_or_index_mib':voxel_index.nbytes()/(1024**2),'persistence_score_mib':0.0,'bst_representation_mib':0.0,'deployed_runtime_total_mib':voxel_total/(1024**2)})
    construction={'hierarchy_internal_nodes':int(hierarchy.internal_count),'hierarchy_payload_mib':hierarchy.internal_payload_nbytes()/(1024**2),'hierarchy_build_ms':float(hierarchy.build_ms),'lod_decision_ms':float(view.decision_ms),'materialize_ms':float(compact.materialize_ms),'persistence_build_ms':float(causal.build_ms),'persistence_leaf_expansion_ms':float(causal.expansion_ms),'persistence_membership_ms':float(causal.membership_ms),'retained_at_runtime':False,**_session_block_timing_metadata(session_block_ms),'adaptive_core_build_ms':float(core_ms),'voxel_index_build_ms':float(voxel_index_ms)}
    hierarchy=None; view=None; raw_surf=None; gc.collect()

    rows=[]
    def run_unweighted(geom,index):
        if progress: print(f'[9.3] validating {geom}...',flush=True)
        for qsi,session in enumerate(query_sessions):
            rng=random.Random(int(perturb_seed)+1000*qsi)
            for si,_ in enumerate(selected_by_session[session]):
                sp,gt,gap,oi,body=loaded[('cross_session',session,si)]; _,gx,gy,_,_,_,gyaw=[float(v) for v in gt]
                specs=[('zero_control',0.,0.,0.,0.,0.)]
                for label,xy,yaw in levels:
                    dx,dy,dyaw=_trial_perturbation(xy,yaw,rng); specs.append((label,xy,yaw,dx,dy,dyaw))
                for label,xy,yaw,dx,dy,dyaw in specs:
                    init=Pose3DOF(gx+dx,gy+dy,gyaw+math.radians(dyaw)); row=_run_trial(map_geometry=geom,scenario='cross_session',scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,original_scan_index=oi,init=init,perturbation_label=label,xy_mag=xy,yaw_mag=yaw,dx=dx,dy=dy,dyaw_deg=dyaw,map_index=index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg)
                    row['map_session']=map_session; row['query_session']=session; row['session_pair']=f'{map_session}->{session}'; row['causal_prior_sessions']=qidx[qsi]-map_t; rows.append(row)
        if include_same_session_zero:
            for si,_ in enumerate(same):
                sp,gt,gap,oi,body=loaded[('same_session_control',map_session,si)]; _,gx,gy,_,_,_,gyaw=[float(v) for v in gt]; init=Pose3DOF(gx,gy,gyaw)
                row=_run_trial(map_geometry=geom,scenario='same_session_control',scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,original_scan_index=oi,init=init,perturbation_label='zero_control',xy_mag=0.,yaw_mag=0.,dx=0.,dy=0.,dyaw_deg=0.,map_index=index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg)
                row['map_session']=map_session; row['query_session']=map_session; row['session_pair']=f'{map_session}->{map_session}'; row['causal_prior_sessions']=1; rows.append(row)

    def run_persistent():
        if progress: print(f'[9.3] validating {PERSISTENT} with causal prior-only weights...',flush=True)
        for qsi,session in enumerate(query_sessions):
            q=qidx[qsi]; index=persistence_indices[q]; rng=random.Random(int(perturb_seed)+1000*qsi)
            for si,_ in enumerate(selected_by_session[session]):
                sp,gt,gap,oi,body=loaded[('cross_session',session,si)]; _,gx,gy,_,_,_,gyaw=[float(v) for v in gt]
                specs=[('zero_control',0.,0.,0.,0.,0.)]
                for label,xy,yaw in levels:
                    dx,dy,dyaw=_trial_perturbation(xy,yaw,rng); specs.append((label,xy,yaw,dx,dy,dyaw))
                for label,xy,yaw,dx,dy,dyaw in specs:
                    init=Pose3DOF(gx+dx,gy+dy,gyaw+math.radians(dyaw)); row=_run_persistence_trial(map_geometry=PERSISTENT,scenario='cross_session',scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,original_scan_index=oi,init=init,perturbation_label=label,xy_mag=xy,yaw_mag=yaw,dx=dx,dy=dy,dyaw_deg=dyaw,map_index=index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg)
                    row['map_session']=map_session; row['query_session']=session; row['session_pair']=f'{map_session}->{session}'; row['causal_prior_sessions']=q-map_t; rows.append(row)
        if include_same_session_zero:
            for si,_ in enumerate(same):
                sp,gt,gap,oi,body=loaded[('same_session_control',map_session,si)]; _,gx,gy,_,_,_,gyaw=[float(v) for v in gt]; init=Pose3DOF(gx,gy,gyaw)
                row=_run_persistence_trial(map_geometry=PERSISTENT,scenario='same_session_control',scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,original_scan_index=oi,init=init,perturbation_label='zero_control',xy_mag=0.,yaw_mag=0.,dx=0.,dy=0.,dyaw_deg=0.,map_index=same_persistence_index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg)
                row['map_session']=map_session; row['query_session']=map_session; row['session_pair']=f'{map_session}->{map_session}'; row['causal_prior_sessions']=1; rows.append(row)

    if include_voxel_baseline: run_unweighted(VOXEL,voxel_index)
    run_unweighted(UNWEIGHTED,compact_index); run_persistent()

    session_summary=_session_summary(rows); pooled=_pooled_summary(rows); paired=_paired_history_outcomes(rows)
    null=_verify_null_equivalence(rows,query_sessions[0],atol=0.0); gates=_persistence_gates(session_summary,paired,query_sessions[0])
    if not null['passed']: raise AssertionError(f'causal no-history control is not exact: {null}')
    # Geometry workload must be exactly unchanged trial-by-trial between the two BST methods.
    uk={(r['scenario'],r['query_session'],int(r['original_scan_index']),r['perturbation']):int(r['local_map_primitives']) for r in rows if r['map_geometry']==UNWEIGHTED}
    pk={(r['scenario'],r['query_session'],int(r['original_scan_index']),r['perturbation']):int(r['local_map_primitives']) for r in rows if r['map_geometry']==PERSISTENT}
    if uk != pk: raise AssertionError('persistence weighting changed retrieved primitive counts')

    _write_csv(out/'localization_trials.csv',rows); _write_csv(out/'session_summary.csv',session_summary); _write_csv(out/'pooled_summary.csv',pooled)
    _write_csv(out/'paired_history_outcomes.csv',paired); _write_csv(out/'persistence_gates.csv',gates); _write_csv(out/'persistence_stats.csv',causal.stats)
    _write_csv(out/'query_scans.csv',scan_rows); _write_csv(out/'deployment_memory.csv',memory); _write_csv(out/'null_control_verification.csv',[null])
    _write_csv(out/'materialization_verification.csv',[{'payload_exact':verification.payload_exact,'directory_exact':verification.directory_exact,'query_exact':verification.query_exact,'pyramid_exact':verification.pyramid_exact,'query_checks':verification.checks}])

    metadata=prepared.metadata(); metadata.update({
        'step':'9.3','purpose':'Causal persistence-aware local registration over the frozen compact balanced K=4 BST LOD; temporal history changes correspondence weights but not geometry/retrieval.',
        'history_sessions':names,'map_session':map_session,'map_session_index':map_t,'query_sessions':query_sessions,'query_session_indices':qidx,
        'query_count_per_session':int(query_count_per_session),'total_cross_session_scans':int(sum(len(v) for v in selected_by_session.values())),'same_session_query_count':len(same),
        'excluded_prior_query_csvs':[str(x) for x in exclude_query_scans_csv or []],'excluded_cross_session_scan_names':len(excluded_cross),
        'frozen_geometry_profile':asdict(FROZEN_PROFILE),'frozen_k':FROZEN_K,'perturbations':[{'label':l,'xy_m':x,'yaw_deg':y} for l,x,y in levels],
        'causal_persistence':{
            'definition':'For query session t, each Jan/map voxel score is fraction of sessions from map_session through t-1 in which that canonical voxel was occupied; selected LOD score is the voxel-support mean.',
            'query_session_and_future_excluded':True,'absence_semantics':'absence is not treated as observed free; score is a repeated-occupancy observation proxy','weight_formula':'ICP normal-equation weight w = persistence fraction','hard_rejection':False,'quantization':'uint8, score/255 at runtime','first_query_session_exact_null_control':True,
        },
        'methods':([VOXEL] if include_voxel_baseline else [])+[UNWEIGHTED,PERSISTENT],
        'localization_max_range_m':localization_max_range_m,'scan_downsample_m':scan_downsample_m,'max_scan_points':max_scan_points,'local_map_radius_m':local_map_radius_m,
        'success_thresholds':{'translation_m':success_translation_m,'yaw_deg':success_yaw_deg},'base_icp_config':asdict(base_cfg),'pyramid':[asdict(x) for x in pyramid],
        'construction_only':construction,'persistence_stats':causal.stats,'deployment_memory':memory,'null_control_verification':null,'persistence_gates':gates,'paired_history_outcomes':paired,
        'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'scientific_caveat':'Step 9.3 tests a causal repeated-occupancy proxy, not semantic permanence or observed-free probability. Scores are materialized offline from exact per-session occupancy histories and soft-weight ICP; PrototypeBSTSpatialCodec remains a local research codec, not verified upstream BST-ID.',
    })
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return rows,session_summary,pooled,paired,gates,memory,causal.stats,null


__all__=['benchmark_step93','_paired_history_outcomes','_verify_null_equivalence','_persistence_gates',UNWEIGHTED,PERSISTENT,VOXEL]
