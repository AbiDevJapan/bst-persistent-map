from __future__ import annotations

import csv
import gc
import json
import re
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
from benchmark.step902_benchmark import _load_and_prepare_scan
from benchmark.step91_benchmark import _run_trial
from benchmark.step913_benchmark import (
    FROZEN_PROFILE, FROZEN_K, _select_query_scans_excluding,
    _runtime_voxel_geometry_bytes,
)
from benchmark.step92_benchmark import _exact_mcnemar_p, _session_block_timing_metadata
from benchmark.step93_benchmark import _union_excluded
from localization.step90 import Pose3DOF
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID
from localization.step903b import SurfelLocalMapIndex
from localization.step91 import align_voxel_surfels_to_prepared
from localization.step912 import build_native_surfel_hierarchy, build_hierarchical_lod_view
from localization.step913 import materialize_lod_view, MaterializedLODBSTIndex, verify_materialization
from localization.step94 import causal_latest_prior_plan, historical_catalog_totals
from realdata.nclt import NCLTIngestConfig, _timestamp_from_scan
from realdata.nclt_packed import prepare_nclt_packed
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec


STATIC = 'static_jan_voxel'
LATEST_VOXEL = 'latest_prior_voxel'
BST_LATEST = 'bst_latest_prior_k4'
MAIN_PERTURBATION = 'xy0.5_yaw1'


def _parse_checkpoint_scalar(v):
    if v is None:
        return None
    if not isinstance(v, str):
        return v
    x=v.strip()
    if x == '':
        return ''
    if x == 'True': return True
    if x == 'False': return False
    if re.fullmatch(r'[+-]?\d+', x):
        try: return int(x)
        except ValueError: pass
    try:
        return float(x)
    except ValueError:
        return v


def _read_checkpoint_csv(path: Path):
    with path.open(newline='') as f:
        return [{k:_parse_checkpoint_scalar(v) for k,v in row.items()} for row in csv.DictReader(f)]



def _summary_rows(rows, pooled=False):
    groups = {}
    for r in rows:
        if pooled and r['scenario'] != 'cross_session':
            continue
        if pooled:
            key = (r['map_geometry'], r['perturbation'])
        else:
            key = (r['map_geometry'], r['scenario'], r['query_session'], r['perturbation'])
        groups.setdefault(key, []).append(r)
    out = []
    for key, g in groups.items():
        def vals(k): return [float(x[k]) for x in g]
        base = {
            'map_geometry': key[0],
            'perturbation': key[-1],
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
            'mean_final_correspondences': statistics.mean(vals('correspondences')),
            'median_final_plane_rmse_m': statistics.median(vals('final_plane_rmse_m')),
        }
        if pooled:
            base.update({'scenario': 'cross_session_pooled', 'query_sessions': len(set(x['query_session'] for x in g))})
        else:
            base.update({
                'scenario': key[1], 'query_session': key[2],
                'historical_map_sessions': ';'.join(sorted(set(x.get('historical_map_session','') for x in g))),
            })
        out.append(base)
    if pooled:
        return sorted(out, key=lambda r: (r['perturbation'], r['map_geometry']))
    return sorted(out, key=lambda r: (r['scenario'], r['query_session'], r['perturbation'], r['map_geometry']))


def _paired_compare(rows, method_a, method_b, *, label_a, label_b, later_only=False, first_query_session=None):
    cross = [r for r in rows if r['scenario'] == 'cross_session' and r['map_geometry'] in {method_a, method_b}]
    sessions = sorted(set(r['query_session'] for r in cross))
    perturbations = sorted(set(r['perturbation'] for r in cross))
    specs = [(s,p) for s in sessions for p in perturbations]
    specs += [('LATER',p) for p in perturbations]
    specs += [('ALL',p) for p in perturbations]
    out=[]
    for session, perturb in specs:
        q=[r for r in cross if r['perturbation']==perturb]
        if session == 'LATER':
            if first_query_session is None:
                continue
            q=[r for r in q if r['query_session'] != first_query_session]
        elif session != 'ALL':
            q=[r for r in q if r['query_session']==session]
        if not q:
            continue
        key=lambda r:(r['query_session'],int(r['original_scan_index']),r['perturbation'])
        aa={key(r):bool(r['success']) for r in q if r['map_geometry']==method_a}
        bb={key(r):bool(r['success']) for r in q if r['map_geometry']==method_b}
        if set(aa)!=set(bb):
            raise AssertionError(f'paired keys differ: {method_a} vs {method_b} {session}/{perturb}')
        both=sum(aa[k] and bb[k] for k in aa)
        aonly=sum(aa[k] and not bb[k] for k in aa)
        bonly=sum((not aa[k]) and bb[k] for k in aa)
        neither=sum((not aa[k]) and (not bb[k]) for k in aa)
        out.append({
            'comparison':f'{method_a}_vs_{method_b}','query_session':session,'perturbation':perturb,
            'paired_trials':len(aa),'both_success':both,
            f'{label_a}_only_success':aonly,f'{label_b}_only_success':bonly,'neither_success':neither,
            f'net_{label_b}_success_difference':bonly-aonly,'exact_mcnemar_p':_exact_mcnemar_p(aonly,bonly),
        })
    return out


def _find_pooled(pooled, geom, perturb=MAIN_PERTURBATION):
    return next((r for r in pooled if r['map_geometry']==geom and r['perturbation']==perturb), None)


def _find_pooled_later(rows, geom, first_query_session, perturb=MAIN_PERTURBATION):
    g=[r for r in rows if r['scenario']=='cross_session' and r['map_geometry']==geom and r['query_session']!=first_query_session and r['perturbation']==perturb]
    if not g: return None
    def vals(k): return [float(x[k]) for x in g]
    return {
        'success_rate':float(np.mean([bool(x['success']) for x in g])),
        'median_total_ms':statistics.median(vals('total_ms')),
        'mean_local_map_primitives':statistics.mean(vals('local_map_primitives')),
        'trials':len(g),
    }


def _historical_gates(rows, pooled, storage_totals, first_query_session):
    static=_find_pooled_later(rows,STATIC,first_query_session)
    latest=_find_pooled_later(rows,LATEST_VOXEL,first_query_session)
    bst=_find_pooled_later(rows,BST_LATEST,first_query_session)
    if not (static and latest and bst): return []
    temporal_ratio=latest['success_rate']/static['success_rate'] if static['success_rate']>0 else float('nan')
    temporal_gain=latest['success_rate']-static['success_rate']
    bst_ratio=bst['success_rate']/latest['success_rate'] if latest['success_rate']>0 else float('nan')
    primitive_reduction=1.0-bst['mean_local_map_primitives']/latest['mean_local_map_primitives']
    speedup=latest['median_total_ms']/bst['median_total_ms'] if bst['median_total_ms']>0 else float('inf')
    storage_reduction=float(storage_totals['catalog_storage_reduction_vs_snapshots'])
    return [{
        'perturbation':MAIN_PERTURBATION,'later_sessions_excluding_first':True,
        'static_success_rate':static['success_rate'],'latest_prior_voxel_success_rate':latest['success_rate'],
        'temporal_selection_success_difference':temporal_gain,'temporal_selection_success_ratio':temporal_ratio,
        'nonnegative_temporal_selection_gain':bool(temporal_gain>=0),'strict_temporal_selection_gain_observed':bool(temporal_gain>0),
        'bst_latest_prior_success_rate':bst['success_rate'],'bst_success_ratio_to_latest_prior_voxel':bst_ratio,
        'meets_bst_success_retention_0.90':bool(bst_ratio>=0.90),
        'bst_primitive_reduction_vs_latest_prior_voxel':primitive_reduction,'meets_primitive_reduction_0.40':bool(primitive_reduction>=0.40),
        'bst_median_total_speedup_vs_latest_prior_voxel':speedup,'meets_speedup_1.20':bool(speedup>=1.20),
        'historical_catalog_storage_reduction_vs_independent_snapshots':storage_reduction,
        'meets_catalog_storage_reduction_0.20':bool(storage_reduction>=0.20),
        'bst_representation_gate_pass':bool(bst_ratio>=0.90 and primitive_reduction>=0.40 and speedup>=1.20),
        'historical_storage_gate_pass':bool(storage_reduction>=0.20),
        'temporal_selection_gain_gate_pass':bool(temporal_gain>0),
    }]


def benchmark_step94(
    *, data_root, output_dir, history_sessions,
    map_session='2012-01-15', query_sessions=('2012-02-02','2012-03-17','2012-05-11','2012-06-15'),
    voxel_size_m=0.5, scan_stride=10, min_range_m=2.0, max_range_m=80.0,
    min_points_per_voxel=1, max_pose_gap_us=100000, cache_dir='.cache/step8_1',
    query_count_per_session=50, exclude_query_scans_csv=(), perturbations='0.5:1', perturb_seed=94090,
    localization_max_range_m=30.0, scan_downsample_m=0.75, max_scan_points=8000,
    local_map_radius_m=40.0, bucket_width_m=8.0, success_translation_m=0.5, success_yaw_deg=1.0,
    icp_config=None, progress=True, resume_checkpoint=False,
):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    names=list(history_sessions); query_sessions=list(query_sessions)
    if abs(float(voxel_size_m)-0.5)>1e-9: raise ValueError('Step 9.4 keeps the frozen 0.5m canonical lattice')
    plan=causal_latest_prior_plan(names,map_session,query_sessions)
    map_t=names.index(map_session); first_query=query_sessions[0]
    cfg=NCLTIngestConfig(voxel_size_m=float(voxel_size_m),scan_stride=int(scan_stride),min_range_m=float(min_range_m),max_range_m=float(max_range_m),min_points_per_voxel=int(min_points_per_voxel),max_pose_gap_us=int(max_pose_gap_us))
    base_cfg=icp_config or PointToPlaneICP3DOFConfig(); levels=parse_perturbations(perturbations); pyramid=DEFAULT_PYRAMID
    if not any(l==MAIN_PERTURBATION for l,_,_ in levels): raise ValueError(f'{MAIN_PERTURBATION} must be included')

    excluded=_union_excluded(exclude_query_scans_csv,'cross_session')
    selected={p.query_session:_select_query_scans_excluding(data_root,p.query_session,query_count_per_session,max_pose_gap_us,excluded) for p in plan}
    prep_scan=SimpleNamespace(config=cfg); loaded={}; scan_rows=[]
    for p in plan:
        for si,(sp,gt,gap,oi) in enumerate(selected[p.query_session]):
            body,raw=_load_and_prepare_scan(sp,prep_scan,localization_max_range_m,scan_downsample_m,max_scan_points)
            loaded[(p.query_session,si)]=(sp,gt,gap,oi,body)
            _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
            scan_rows.append({'scenario':'cross_session','base_map_session':map_session,'query_session':p.query_session,
                'latest_prior_session':p.latest_prior_session,'query_session_index':p.query_index,'latest_prior_index':p.latest_prior_index,
                'scan_index':si,'original_scan_index':int(oi),'scan_file':sp.name,'timestamp_us':int(_timestamp_from_scan(sp)),
                'pose_gap_us':int(gap),'points_after_range':int(raw),'points_used':int(len(body)),
                'gt_x':gx,'gt_y':gy,'gt_z':gz,'gt_roll':roll,'gt_pitch':pitch,'gt_yaw':gyaw,
                'excluded_prior_name_match':bool(sp.name in excluded)})
    if any(r['excluded_prior_name_match'] for r in scan_rows): raise AssertionError('Step 9.4 selected a previously excluded scan')
    if any(not np.all(np.isfinite([r['gt_x'],r['gt_y'],r['gt_z'],r['gt_roll'],r['gt_pitch'],r['gt_yaw']])) for r in scan_rows): raise ValueError('non-finite query pose selected')

    # Freeze perturbations once so all three map strategies see exactly the same initial poses.
    trial_specs={}
    for qsi,p in enumerate(plan):
        rng=random.Random(int(perturb_seed)+1000*qsi)
        for si in range(len(selected[p.query_session])):
            specs=[('zero_control',0.,0.,0.,0.,0.)]
            for label,xy,yaw in levels:
                dx,dy,dyaw=_trial_perturbation(xy,yaw,rng); specs.append((label,xy,yaw,dx,dy,dyaw))
            trial_specs[(p.query_session,si)]=specs

    if progress: print('[9.4] loading packed five-session history once...',flush=True)
    prepared=prepare_nclt_packed(data_root,names,cfg,cache_dir,progress_every=250)
    if progress: print('[9.4] building Step-8.4/8.5 historical BST once...',flush=True)
    session_blocks,session_block_ms=build_session_spatial_blocks(prepared,len(names),progress=progress)
    t0=time.perf_counter_ns(); core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=progress); core_ms=(time.perf_counter_ns()-t0)/1e6
    bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())

    rows=[]; map_stats=[]; ver_rows=[]; snapshot_bytes={}; lod_bytes={}
    completed_prior_sessions=set()
    if resume_checkpoint:
        cp_rows=out/'checkpoint_localization_trials.csv'
        cp_stats=out/'checkpoint_historical_map_stats.csv'
        if not cp_rows.exists() or not cp_stats.exists():
            raise FileNotFoundError(
                'resume requested but Step 9.4 checkpoint files are missing: '
                f'{cp_rows} / {cp_stats}'
            )
        rows=_read_checkpoint_csv(cp_rows)
        map_stats=_read_checkpoint_csv(cp_stats)
        completed_prior_sessions={str(r['historical_map_session']) for r in map_stats}
        valid_prior_sessions={names[p.latest_prior_index] for p in plan}
        if not completed_prior_sessions.issubset(valid_prior_sessions):
            raise ValueError(
                'checkpoint contains historical-map sessions incompatible with this causal plan: '
                f'{sorted(completed_prior_sessions-valid_prior_sessions)}'
            )
        for r in map_stats:
            sess=str(r['historical_map_session'])
            snapshot_bytes[sess]=int(round(float(r['voxel_snapshot_total_mib'])*(1024**2)))
            lod_bytes[sess]=int(round(float(r['bst_lod_overlay_total_mib'])*(1024**2)))
            # The checkpoint is written only after verify_materialization(...).passed
            # has been asserted.  Old checkpoints did not persist ver_rows, so
            # preserve that proven pass state without fabricating a query count.
            ver_rows.append({
                'historical_map_session':sess,'session_index':int(r['session_index']),
                'payload_exact':True,'directory_exact':True,'query_exact':True,'pyramid_exact':True,
                'query_checks':-1,'verification_source':'checkpoint_implies_pass',
            })
        if progress:
            print(
                '[9.4] resume checkpoint: completed prior maps = '+
                (', '.join(sorted(completed_prior_sessions)) if completed_prior_sessions else '(none)'),
                flush=True,
            )

    def run_for_session(geom,index,p, historical_map_session):
        if progress: print(f'[9.4] {geom}: {historical_map_session}->{p.query_session}',flush=True)
        for si in range(len(selected[p.query_session])):
            sp,gt,gap,oi,body=loaded[(p.query_session,si)]; _,gx,gy,_,_,_,gyaw=[float(v) for v in gt]
            for label,xy,yaw,dx,dy,dyaw in trial_specs[(p.query_session,si)]:
                init=Pose3DOF(gx+dx,gy+dy,gyaw+math.radians(dyaw))
                r=_run_trial(map_geometry=geom,scenario='cross_session',scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,
                    original_scan_index=oi,init=init,perturbation_label=label,xy_mag=xy,yaw_mag=yaw,dx=dx,dy=dy,dyaw_deg=dyaw,
                    map_index=index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,
                    success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg)
                r['base_map_session']=map_session; r['historical_map_session']=historical_map_session; r['query_session']=p.query_session
                r['session_pair']=f'{historical_map_session}->{p.query_session}'
                rows.append(r)

    # Jan map is special: it serves the static baseline for every query session and
    # is also the exact latest-prior map for Feb.
    unique_prior_indices=[]
    for p in plan:
        if p.latest_prior_index not in unique_prior_indices: unique_prior_indices.append(p.latest_prior_index)

    for prior_i in unique_prior_indices:
        prior_session=names[prior_i]
        if prior_session in completed_prior_sessions:
            if progress: print(f'[9.4] resume: skipping completed causal prior session {prior_session}',flush=True)
            continue
        if progress: print(f'[9.4] building localization geometry for causal prior session {prior_session}...',flush=True)
        raw=build_nclt_surfel_map(data_root,prior_session,cfg,cache_dir,normal_min_points=base_cfg.normal_min_neighbors,
            normal_max_curvature=base_cfg.normal_max_curvature,normal_min_horizontal_component=base_cfg.normal_min_horizontal_component)
        aligned=align_voxel_surfels_to_prepared(raw,prepared,prior_i)
        t0=time.perf_counter_ns(); voxel_index=SurfelLocalMapIndex(aligned,bucket_width_m); voxel_index_ms=(time.perf_counter_ns()-t0)/1e6

        hierarchy=build_native_surfel_hierarchy(aligned,prepared,session_blocks,core,bst,prior_i,base_cfg.normal_min_neighbors,
            base_cfg.normal_max_curvature,base_cfg.normal_min_horizontal_component,progress=progress)
        view=build_hierarchical_lod_view(hierarchy,FROZEN_PROFILE,FROZEN_K); compact=materialize_lod_view(view)
        compact_index=MaterializedLODBSTIndex(compact,bst,prior_i)

        associated=[p for p in plan if p.latest_prior_index==prior_i]
        centers=[]
        for p in associated:
            for si in range(min(3,len(selected[p.query_session]))):
                _,gt,_,_,_=loaded[(p.query_session,si)]; centers.append((float(gt[1]),float(gt[2])))
        if prior_i==map_t:
            # Static Jan is used by all query sessions; sample their centers too.
            for p in plan:
                if p not in associated and len(selected[p.query_session]):
                    _,gt,_,_,_=loaded[(p.query_session,0)]; centers.append((float(gt[1]),float(gt[2])))
        verification=verify_materialization(view,compact,bst,prior_i,centers,local_map_radius_m,pyramid,base_cfg,max_pyramid_checks=min(6,max(1,len(centers))))
        if not verification.passed: raise AssertionError(f'Step 9.4 materialization verification failed for {prior_session}')
        ver_rows.append({'historical_map_session':prior_session,'session_index':prior_i,'payload_exact':verification.payload_exact,
            'directory_exact':verification.directory_exact,'query_exact':verification.query_exact,'pyramid_exact':verification.pyramid_exact,
            'query_checks':verification.checks,'verification_source':'live'})

        voxel_geom=_runtime_voxel_geometry_bytes(aligned); voxel_total=voxel_geom+voxel_index.nbytes(); compact_total=compact.runtime_nbytes()
        snapshot_bytes[prior_session]=int(voxel_total); lod_bytes[prior_session]=int(compact_total)
        map_stats.append({'historical_map_session':prior_session,'session_index':prior_i,'voxel_elements':int(len(aligned.linear_codes)),
            'voxel_runtime_geometry_mib':voxel_geom/(1024**2),'voxel_runtime_index_mib':voxel_index.nbytes()/(1024**2),
            'voxel_snapshot_total_mib':voxel_total/(1024**2),'bst_lod_elements':int(compact.map_elements),
            'bst_lod_geometry_mib':compact.geometry_payload_nbytes()/(1024**2),'bst_lod_directory_mib':compact.directory_nbytes()/(1024**2),
            'bst_lod_overlay_total_mib':compact_total/(1024**2),'mean_voxels_per_bst_primitive':float(hierarchy.leaf_count/compact.map_elements),
            'surfel_build_ms':float(raw.stats_build_ms),'voxel_index_build_ms':float(voxel_index_ms),'hierarchy_build_ms':float(hierarchy.build_ms),
            'lod_decision_ms':float(view.decision_ms),'materialize_ms':float(compact.materialize_ms)})

        # Static Jan voxel baseline across every query session.
        if prior_i==map_t:
            for p in plan: run_for_session(STATIC,voxel_index,p,map_session)

        # Causal latest-prior voxel and compact BST for the sessions that use this prior map.
        for p in associated:
            run_for_session(LATEST_VOXEL,voxel_index,p,prior_session)
            run_for_session(BST_LATEST,compact_index,p,prior_session)

        # Cheap recovery checkpoint after each expensive historical-map build.
        # A late reporting failure therefore does not erase already-completed localization trials.
        _write_csv(out/'checkpoint_localization_trials.csv',rows)
        _write_csv(out/'checkpoint_historical_map_stats.csv',map_stats)
        _write_csv(out/'checkpoint_materialization_verification.csv',ver_rows)

        hierarchy=None; view=None; compact_index=None; voxel_index=None; aligned=None; raw=None; compact=None; gc.collect()

    session_summary=_summary_rows(rows,pooled=False); pooled=_summary_rows(rows,pooled=True)
    temporal_pairs=_paired_compare(rows,STATIC,LATEST_VOXEL,label_a='static',label_b='latest_prior',first_query_session=first_query)
    bst_pairs=_paired_compare(rows,LATEST_VOXEL,BST_LATEST,label_a='latest_prior_voxel',label_b='bst_latest_prior',first_query_session=first_query)

    # Feb is an exact temporal-selection null control: latest prior is Jan, so the
    # two voxel methods must be numerically identical trial-by-trial.
    null_a=[r for r in rows if r['map_geometry']==STATIC and r['query_session']==first_query]
    null_b=[r for r in rows if r['map_geometry']==LATEST_VOXEL and r['query_session']==first_query]
    key=lambda r:(int(r['original_scan_index']),r['perturbation'])
    da={key(r):r for r in null_a}; db={key(r):r for r in null_b}
    if set(da)!=set(db): raise AssertionError('Step 9.4 first-session null keys differ')
    max_diff=0.0
    for k in da:
        for f in ('final_translation_error_m','final_yaw_error_deg','estimated_x','estimated_y','estimated_yaw','final_plane_rmse_m'):
            d=abs(float(da[k][f])-float(db[k][f])); max_diff=max(max_diff,d)
            if not np.isclose(float(da[k][f]),float(db[k][f]),rtol=0,atol=1e-12,equal_nan=True): raise AssertionError(f'null mismatch {k} {f}')
        for f in ('success','converged','iterations','correspondences','normal_correspondences','failure_reason','local_map_primitives'):
            if da[k].get(f)!=db[k].get(f): raise AssertionError(f'null mismatch {k} {f}')
    null={'passed':True,'query_session':first_query,'historical_map_session':map_session,'trials_per_method':len(da),'max_abs_float_difference':float(max_diff)}

    storage=historical_catalog_totals(snapshot_bytes,lod_bytes,bst.compact_bytes())
    storage_rows=[
        {'strategy':'independent_latest_prior_voxel_snapshots','snapshot_catalog_mib':storage['independent_snapshot_catalog_bytes']/(1024**2),'bst_history_mib':0.0,'compact_lod_overlay_catalog_mib':0.0,'historical_catalog_total_mib':storage['independent_snapshot_catalog_bytes']/(1024**2),'storage_reduction_vs_independent_snapshots':0.0},
        {'strategy':'bst_history_plus_compact_k4_overlays','snapshot_catalog_mib':0.0,'bst_history_mib':storage['bst_history_bytes']/(1024**2),'compact_lod_overlay_catalog_mib':storage['compact_lod_overlay_catalog_bytes']/(1024**2),'historical_catalog_total_mib':storage['bst_historical_catalog_bytes']/(1024**2),'storage_reduction_vs_independent_snapshots':storage['catalog_storage_reduction_vs_snapshots']},
    ]
    gates=_historical_gates(rows,pooled,storage,first_query)

    _write_csv(out/'localization_trials.csv',rows); _write_csv(out/'session_summary.csv',session_summary); _write_csv(out/'pooled_summary.csv',pooled)
    _write_csv(out/'temporal_selection_outcomes.csv',temporal_pairs); _write_csv(out/'bst_historical_outcomes.csv',bst_pairs)
    _write_csv(out/'historical_map_stats.csv',map_stats); _write_csv(out/'historical_storage.csv',storage_rows); _write_csv(out/'historical_gates.csv',gates)
    _write_csv(out/'materialization_verification.csv',ver_rows); _write_csv(out/'null_control_verification.csv',[null]); _write_csv(out/'query_scans.csv',scan_rows)

    metadata=prepared.metadata(); metadata.update({
        'step':'9.4','purpose':'Causal historical map selection: static Jan map vs immediately latest-prior voxel snapshot vs compact BST K=4 latest-prior historical retrieval.',
        'history_sessions':names,'base_map_session':map_session,'query_sessions':query_sessions,
        'causal_latest_prior_plan':[asdict(x) for x in plan],'query_count_per_session':int(query_count_per_session),'total_cross_session_scans':int(sum(len(v) for v in selected.values())),
        'excluded_prior_query_csvs':[str(x) for x in exclude_query_scans_csv or []],'excluded_cross_session_scan_names':len(excluded),
        'frozen_geometry_profile':asdict(FROZEN_PROFILE),'frozen_k':FROZEN_K,'perturbations':[{'label':l,'xy_m':x,'yaw_deg':y} for l,x,y in levels],
        'methods':[STATIC,LATEST_VOXEL,BST_LATEST],
        'interpretation_contract':{
            'static_vs_latest_prior_voxel':'isolates temporal map-selection effect without BST representation changes',
            'latest_prior_voxel_vs_bst_latest_prior_k4':'isolates compact BST representation effect at the same causal map date',
            'no_future_leakage':True,'latest_prior_rule':'immediately preceding available history session only','first_query_session_exact_temporal_null_control':True,
        },
        'localization_max_range_m':localization_max_range_m,'scan_downsample_m':scan_downsample_m,'max_scan_points':max_scan_points,'local_map_radius_m':local_map_radius_m,
        'success_thresholds':{'translation_m':success_translation_m,'yaw_deg':success_yaw_deg},'base_icp_config':asdict(base_cfg),'pyramid':[asdict(x) for x in pyramid],
        'construction_once':{**_session_block_timing_metadata(session_block_ms),'adaptive_core_build_ms':float(core_ms),'bst_history_built_once':True},
        'historical_map_stats':map_stats,'historical_storage':storage_rows,'historical_gates':gates,'null_control_verification':null,'materialization_verification':ver_rows,
        'resume_checkpoint_used':bool(resume_checkpoint),'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'scientific_caveat':'Latest-prior voxel accuracy gains are a causal lifelong-map selection effect, not by themselves a BST contribution. The BST contribution tested here is retention of that latest-prior localization capability with compact hierarchical LOD/history storage. PrototypeBSTSpatialCodec remains a local research codec, not verified upstream BST-ID.',
    })
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return rows,session_summary,pooled,temporal_pairs,bst_pairs,gates,storage_rows,map_stats,null


__all__=['benchmark_step94',STATIC,LATEST_VOXEL,BST_LATEST,'_paired_compare','_historical_gates']
