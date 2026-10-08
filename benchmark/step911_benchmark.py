from __future__ import annotations

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
from localization.step91 import align_voxel_surfels_to_prepared, NativeBlockSurfelIndex
from localization.step911 import (
    parse_geometry_profiles,
    build_native_block_surfel_map_with_leaf_owners,
    build_native_geometry_descriptors,
    build_geometry_aware_virtual_view,
    GeometryAwareBSTVirtualIndex,
)
from realdata.nclt import NCLTIngestConfig, _timestamp_from_scan
from realdata.nclt_packed import prepare_nclt_packed
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec


def _selected_profile_stats(view, desc, block_surf, bst, decision_ms, descriptor_ms):
    keep = view.keep_native
    valid = int(np.count_nonzero(block_surf.normal_valid[keep]) + np.sum(desc.normal_votes[~keep], dtype=np.int64))
    elements = int(view.map_elements)
    voxel_total = int(desc.voxel_volume.sum(dtype=np.int64))
    return {
        'map_geometry': f'bst_geom_{view.profile.name}',
        'profile': view.profile.name,
        'map_elements': elements,
        'valid_normal_elements': valid,
        'valid_normal_fraction': float(valid / elements) if elements else 0.0,
        'geometry_payload_mib': view.selected_payload_nbytes(False) / (1024**2),
        'minimal_geometry_payload_mib': view.selected_payload_nbytes(True) / (1024**2),
        'query_index_mib': view.standalone_refinement_directory_nbytes() / (1024**2),
        'bst_representation_mib': bst.compact_bytes() / (1024**2),
        'geometry_build_ms': float(block_surf.build_ms),
        'descriptor_build_ms': float(descriptor_ms),
        'profile_decision_ms': float(decision_ms),
        'voxel_equivalent': voxel_total,
        'mean_voxels_per_primitive': float(voxel_total / elements) if elements else 0.0,
        'kept_native_blocks': int(view.kept_native_blocks),
        'fallback_parent_blocks': int(view.fallback_parent_blocks),
        'fallback_voxels': int(view.fallback_voxels),
        'native_keep_fraction': float(view.kept_native_blocks / len(keep)) if len(keep) else 0.0,
        'max_plane_sigma_m': float(view.profile.max_plane_sigma_m),
        'max_relative_planarity': float(view.profile.max_relative_planarity),
        'min_normal_coherence': float(view.profile.min_normal_coherence),
        'max_extent_m': float(view.profile.max_extent_m),
    }


def _pareto_rows(summary, map_rows, target_perturbation='xy0.5_yaw1'):
    stat = {r['map_geometry']: r for r in map_rows}
    target = [r for r in summary if r['scenario']=='cross_session' and r['perturbation']==target_perturbation]
    if not target:
        return []
    by = {r['map_geometry']: r for r in target}
    base = by.get('voxel_surfel_0.5m')
    if base is None:
        return []
    rows=[]
    for geom, s in by.items():
        if geom not in stat:
            continue
        m=stat[geom]
        rows.append({
            'map_geometry': geom,
            'profile': m.get('profile','baseline'),
            'success_rate': float(s['success_rate']),
            'success_ratio_to_voxel': float(s['success_rate']/base['success_rate']) if base['success_rate']>0 else float('nan'),
            'meets_90pct_voxel_success_gate': bool(s['success_rate'] >= 0.9*base['success_rate']),
            'median_final_translation_error_m': float(s['median_final_translation_error_m']),
            'median_final_yaw_error_deg': float(s['median_final_yaw_error_deg']),
            'mean_local_map_primitives': float(s['mean_local_map_primitives']),
            'primitive_reduction_vs_voxel': float(1.0 - s['mean_local_map_primitives']/base['mean_local_map_primitives']) if base['mean_local_map_primitives'] else 0.0,
            'median_total_ms': float(s['median_total_ms']),
            'total_speedup_vs_voxel': float(base['median_total_ms']/s['median_total_ms']) if s['median_total_ms'] else float('inf'),
            'minimal_geometry_payload_mib': float(m['minimal_geometry_payload_mib']),
            'bst_representation_mib': float(m['bst_representation_mib']),
            'query_index_mib': float(m['query_index_mib']),
            'map_elements': int(m['map_elements']),
            'mean_voxels_per_primitive_global': float(m['mean_voxels_per_primitive']),
        })
    for i,r in enumerate(rows):
        dom_p=False; dom_t=False
        for j,q in enumerate(rows):
            if i==j: continue
            if (q['success_rate']>=r['success_rate'] and q['mean_local_map_primitives']<=r['mean_local_map_primitives'] and
                (q['success_rate']>r['success_rate'] or q['mean_local_map_primitives']<r['mean_local_map_primitives'])):
                dom_p=True
            if (q['success_rate']>=r['success_rate'] and q['median_total_ms']<=r['median_total_ms'] and
                (q['success_rate']>r['success_rate'] or q['median_total_ms']<r['median_total_ms'])):
                dom_t=True
        r['pareto_success_vs_primitives']=not dom_p
        r['pareto_success_vs_total_ms']=not dom_t
    return sorted(rows,key=lambda r:(r['map_geometry']!='voxel_surfel_0.5m',r['mean_local_map_primitives']))


def benchmark_step911(
    *, data_root, output_dir, history_sessions,
    map_session='2012-01-15', query_session='2012-02-02',
    voxel_size_m=0.5, scan_stride=10, min_range_m=2.0, max_range_m=80.0,
    min_points_per_voxel=1, max_pose_gap_us=100000, cache_dir='.cache/step8_1',
    query_count=20, same_session_query_count=20, query_scans_csv=None,
    perturbations='0.5:1', perturb_seed=42090,
    localization_max_range_m=30.0, scan_downsample_m=0.75, max_scan_points=8000,
    local_map_radius_m=40.0, bucket_width_m=8.0,
    success_translation_m=0.5, success_yaw_deg=1.0,
    icp_config=None, include_same_session_zero=True,
    geometry_profiles=None, include_step91_native=True, progress=True,
):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    names=list(history_sessions)
    if map_session not in names: raise ValueError('map_session must appear in history_sessions')
    if query_session==map_session: raise ValueError('cross-session query_session must differ from map_session')
    if abs(float(voxel_size_m)-0.5)>1e-9: raise ValueError('Step 9.1.1 keeps the frozen Step-8 0.5m canonical lattice')
    map_t=names.index(map_session)
    cfg=NCLTIngestConfig(voxel_size_m=float(voxel_size_m),scan_stride=int(scan_stride),min_range_m=float(min_range_m),max_range_m=float(max_range_m),min_points_per_voxel=int(min_points_per_voxel),max_pose_gap_us=int(max_pose_gap_us))
    base_cfg=icp_config or PointToPlaneICP3DOFConfig()
    levels=parse_perturbations(perturbations); pyramid=DEFAULT_PYRAMID
    profiles=parse_geometry_profiles(geometry_profiles)

    if query_scans_csv is not None:
        replay=_filter_replay_csv_for_cross_session(query_scans_csv,out/'_cross_replay.csv')
        cross=_replay_query_scans(data_root,query_session,replay,query_count,max_pose_gap_us)
    else:
        cross=_select_query_scans(data_root,query_session,query_count,max_pose_gap_us)
    same=_select_query_scans(data_root,map_session,same_session_query_count,max_pose_gap_us,exclude_ingest_stride=scan_stride) if include_same_session_zero else []
    prep_scan=SimpleNamespace(config=cfg)
    scan_rows=[]; loaded={}
    for scenario,selected in [('cross_session',cross),('same_session_control',same)]:
        for si,(sp,gt,gap,oi) in enumerate(selected):
            body,raw=_load_and_prepare_scan(sp,prep_scan,localization_max_range_m,scan_downsample_m,max_scan_points)
            loaded[(scenario,si)]=(sp,gt,gap,oi,body)
            _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
            scan_rows.append({'scenario':scenario,'scan_index':si,'original_scan_index':int(oi),'scan_file':sp.name,'timestamp_us':int(_timestamp_from_scan(sp)),'pose_gap_us':int(gap),'points_after_range':int(raw),'points_used':int(len(body)),'gt_x':gx,'gt_y':gy,'gt_z':gz,'gt_roll':roll,'gt_pitch':pitch,'gt_yaw':gyaw,'directly_ingested_scan':bool((oi%scan_stride)==0)})

    if progress: print('[9.1.1] loading five-session packed history from existing Step-8 cache...',flush=True)
    prepared=prepare_nclt_packed(data_root,names,cfg,cache_dir,progress_every=250)
    if progress: print('[9.1.1] building exact Step-8.4 session-local tilings...',flush=True)
    session_blocks,session_block_ms=build_session_spatial_blocks(prepared,len(names),progress=progress)
    if progress: print('[9.1.1] finalizing unchanged Step-8.5 time-aware BST map...',flush=True)
    t0=time.perf_counter_ns(); core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=progress); core_ms=(time.perf_counter_ns()-t0)/1e6
    bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())

    if progress: print('[9.1.1] rebuilding 0.5m map-session surfel sufficient statistics in memory...',flush=True)
    raw_surf=build_nclt_surfel_map(data_root,map_session,cfg,cache_dir,normal_min_points=base_cfg.normal_min_neighbors,normal_max_curvature=base_cfg.normal_max_curvature,normal_min_horizontal_component=base_cfg.normal_min_horizontal_component)
    aligned=align_voxel_surfels_to_prepared(raw_surf,prepared,map_t)
    t0=time.perf_counter_ns(); voxel_index=SurfelLocalMapIndex(aligned,bucket_width_m); voxel_index_ms=(time.perf_counter_ns()-t0)/1e6

    if progress: print('[9.1.1] reproducing native blocks and building the one-time voxel->native-parent map...',flush=True)
    block_surf, leaf_parent_gids = build_native_block_surfel_map_with_leaf_owners(
        aligned,prepared,session_blocks,core,bst,map_t,
        base_cfg.normal_min_neighbors,base_cfg.normal_max_curvature,
        base_cfg.normal_min_horizontal_component,progress=progress,
    )
    block_index=NativeBlockSurfelIndex(block_surf)
    if progress: print('[9.1.1] precomputing geometry descriptors once for all threshold profiles...',flush=True)
    desc=build_native_geometry_descriptors(aligned,block_surf,leaf_parent_gids)

    voxel_payload=int(sum(a.nbytes for a in (aligned.counts,aligned.centroids,aligned.cov6,aligned.normals,aligned.curvature,aligned.normal_valid)))
    voxel_min=int(sum(a.nbytes for a in (aligned.counts,aligned.centroids,aligned.normals,aligned.curvature,aligned.normal_valid)))
    map_rows=[{
        'map_geometry':'voxel_surfel_0.5m','profile':'baseline','map_elements':len(aligned.linear_codes),
        'valid_normal_elements':int(np.count_nonzero(aligned.normal_valid)),'valid_normal_fraction':float(np.mean(aligned.normal_valid.astype(bool))),
        'geometry_payload_mib':voxel_payload/(1024**2),'minimal_geometry_payload_mib':voxel_min/(1024**2),
        'query_index_mib':voxel_index.nbytes()/(1024**2),'bst_representation_mib':0.0,
        'geometry_build_ms':float(raw_surf.stats_build_ms),'descriptor_build_ms':0.0,'profile_decision_ms':0.0,
        'voxel_equivalent':len(aligned.linear_codes),'mean_voxels_per_primitive':1.0,
        'kept_native_blocks':0,'fallback_parent_blocks':0,'fallback_voxels':0,'native_keep_fraction':0.0,
    }]
    if include_step91_native:
        map_rows.append({
            'map_geometry':'bst_native_block_surfel','profile':'step9.1_native','map_elements':len(block_surf.blocks),
            'valid_normal_elements':int(np.count_nonzero(block_surf.normal_valid)),'valid_normal_fraction':float(np.mean(block_surf.normal_valid.astype(bool))),
            'geometry_payload_mib':block_surf.payload_nbytes()/(1024**2),'minimal_geometry_payload_mib':block_surf.minimal_localization_payload_nbytes()/(1024**2),
            'query_index_mib':0.0,'bst_representation_mib':bst.compact_bytes()/(1024**2),
            'geometry_build_ms':float(block_surf.build_ms),'descriptor_build_ms':float(desc.build_ms),'profile_decision_ms':0.0,
            'voxel_equivalent':int(desc.voxel_volume.sum(dtype=np.int64)),
            'mean_voxels_per_primitive':float(np.mean(desc.voxel_volume)),'kept_native_blocks':len(block_surf.blocks),
            'fallback_parent_blocks':0,'fallback_voxels':0,'native_keep_fraction':1.0,
        })

    rows=[]
    def run_geometry(geom,index):
        if progress: print(f'[9.1.1] localizing with {geom}...',flush=True)
        def scenario_run(scenario,selected,perturb,seed):
            rng=random.Random(int(seed))
            for si,_ in enumerate(selected):
                sp,gt,gap,oi,body=loaded[(scenario,si)]
                _,gx,gy,_,_,_,gyaw=[float(v) for v in gt]
                specs=[('zero_control',0.,0.,0.,0.,0.)]
                if perturb:
                    for label,xy,yaw in levels:
                        dx,dy,dyaw=_trial_perturbation(xy,yaw,rng); specs.append((label,xy,yaw,dx,dy,dyaw))
                for label,xy,yaw,dx,dy,dyaw in specs:
                    init=Pose3DOF(gx+dx,gy+dy,gyaw+math.radians(dyaw))
                    rows.append(_run_trial(map_geometry=geom,scenario=scenario,scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,original_scan_index=oi,init=init,perturbation_label=label,xy_mag=xy,yaw_mag=yaw,dx=dx,dy=dy,dyaw_deg=dyaw,map_index=index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg))
        scenario_run('cross_session',cross,True,perturb_seed)
        if include_same_session_zero: scenario_run('same_session_control',same,False,perturb_seed+902)

    run_geometry('voxel_surfel_0.5m',voxel_index)
    if include_step91_native: run_geometry('bst_native_block_surfel',block_index)

    profile_meta=[]
    for prof in profiles:
        view, decision_ms = build_geometry_aware_virtual_view(aligned,block_surf,desc,prof)
        index=GeometryAwareBSTVirtualIndex(view,bst,map_t)
        stat=_selected_profile_stats(view,desc,block_surf,bst,decision_ms,desc.build_ms)
        map_rows.append(stat)
        profile_meta.append(asdict(prof))
        if progress:
            print(f"[9.1.1] profile {prof.name}: keep={100*stat['native_keep_fraction']:.1f}% native parents, elements={stat['map_elements']:,}, mean={stat['mean_voxels_per_primitive']:.2f} vox/primitive",flush=True)
        run_geometry(stat['map_geometry'],index)
        del index, view

    summary=_summary(rows)
    pareto=_pareto_rows(summary,map_rows,'xy0.5_yaw1')
    _write_csv(out/'localization_trials.csv',rows)
    _write_csv(out/'localization_summary.csv',summary)
    _write_csv(out/'query_scans.csv',scan_rows)
    _write_csv(out/'geometry_profile_stats.csv',map_rows)
    _write_csv(out/'pareto_summary.csv',pareto)
    tmp=out/'_cross_replay.csv'
    if tmp.exists(): tmp.unlink()

    metadata=prepared.metadata(); metadata.update({
        'step':'9.1.1',
        'purpose':'Geometry-aware dynamic localization resolution over the unchanged Step-8.5 BST map; Pareto sweep between native block surfels and exact 0.5m voxel-surfel fallback.',
        'history_sessions':names,'map_session':map_session,'map_session_index':map_t,'query_session':query_session,
        'query_count':len(cross),'same_session_query_count':len(same),
        'query_scans_replay_csv':str(query_scans_csv) if query_scans_csv else None,
        'perturbations':[{'label':l,'xy_m':x,'yaw_deg':y} for l,x,y in levels],
        'localization_max_range_m':localization_max_range_m,'scan_downsample_m':scan_downsample_m,'max_scan_points':max_scan_points,'local_map_radius_m':local_map_radius_m,
        'success_thresholds':{'translation_m':success_translation_m,'yaw_deg':success_yaw_deg},
        'base_icp_config':asdict(base_cfg),'pyramid':[asdict(x) for x in pyramid],
        'geometry_profiles':profile_meta,
        'dynamic_resolution_policy':{
            'online_decision':False,
            'underlying_bst_changed':False,
            'query_path':'unchanged Step-8.5 counted (bucket,session) time-aware BST index',
            'two_level_choice':'one existing native block surfel OR the exact 0.5m voxel surfels inside that same parent block',
            'gate_signals':['absolute plane sigma sqrt(lambda_min)','relative planarity lambda_min/trace(covariance)','sign-invariant base-voxel normal coherence','maximum native-block extent'],
            'normal_coherence':'largest eigenvalue of mean n*n^T across valid base-voxel normals; sign invariant',
            'process_time_rationale':'all gate descriptors are precomputed once offline; each profile assignment is a vectorized O(number_of_native_blocks) mask; localization performs no PCA/split search/decompression decision',
        },
        'sweep_runtime_shared_mapping_mib':desc.index_nbytes()/(1024**2),
        'map_geometry_stats':map_rows,
        'threshold_selection_protocol':{
            'role':'development/tuning set',
            'cross_session_scans':len(cross),
            'warning':'Do not use these same scans for the final claimed threshold performance. Select the operating point here, then validate it on a larger held-out scan set and additional session pairs.',
        },
        'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'scientific_caveat':'Step 9.1.1 is a process-efficient two-level localization overlay, not a new lifelong-map representation. It preserves the Step-8.5 BST/history/index unchanged and refines complex map-session parents to the frozen 0.5m voxel-surfel level. PrototypeBSTSpatialCodec remains a local research codec, not verified upstream BST-ID.',
    })
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return rows,summary,map_rows,pareto


__all__=['benchmark_step911']
