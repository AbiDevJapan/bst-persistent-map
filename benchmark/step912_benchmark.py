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
from localization.step912 import (
    parse_geometry_profiles, build_native_surfel_hierarchy,
    build_hierarchical_lod_view, HierarchicalLODBSTIndex,
)
from realdata.nclt import NCLTIngestConfig, _timestamp_from_scan
from realdata.nclt_packed import prepare_nclt_packed
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec


def parse_k_values(text):
    vals=[int(x.strip()) for x in str(text).split(',') if x.strip()]
    if not vals: raise ValueError('no K values provided')
    for k in vals:
        if k<1 or (k&(k-1))!=0: raise ValueError('K values must be powers of two >=1')
    return sorted(set(vals))


def _lod_stat(view,hierarchy,bst):
    ids=view.selected_node_ids
    valid=int(np.count_nonzero(hierarchy.node_valid(ids)))
    elements=len(ids)
    rc=view.root_counts.astype(np.int64)
    leaf=int(np.count_nonzero(ids<hierarchy.leaf_count))
    return {
        'map_geometry':f'bst_lod_k{view.max_primitives_per_root}',
        'profile':view.profile.name,
        'max_primitives_per_root':int(view.max_primitives_per_root),
        'map_elements':int(elements),
        'valid_normal_elements':valid,
        'valid_normal_fraction':float(valid/elements) if elements else 0.0,
        'geometry_payload_mib':view.selected_payload_nbytes(False)/(1024**2),
        'minimal_geometry_payload_mib':view.selected_payload_nbytes(True)/(1024**2),
        'query_index_mib':view.directory_nbytes()/(1024**2),
        'bst_representation_mib':bst.compact_bytes()/(1024**2),
        'hierarchy_build_ms':float(hierarchy.build_ms),
        'lod_decision_ms':float(view.decision_ms),
        'voxel_equivalent':int(hierarchy.leaf_count),
        'mean_voxels_per_primitive':float(hierarchy.leaf_count/elements) if elements else 0.0,
        'roots_refined':int(np.count_nonzero(rc>1)),
        'root_refine_fraction':float(np.mean(rc>1)) if len(rc) else 0.0,
        'mean_primitives_per_root':float(np.mean(rc)) if len(rc) else 0.0,
        'p95_primitives_per_root':float(np.percentile(rc,95)) if len(rc) else 0.0,
        'max_observed_primitives_per_root':int(rc.max()) if len(rc) else 0,
        'selected_leaf_voxels':leaf,
        'selected_internal_nodes':int(elements-leaf),
        'selected_leaf_fraction':float(leaf/elements) if elements else 0.0,
        'max_plane_sigma_m':float(view.profile.max_plane_sigma_m),
        'max_relative_planarity':float(view.profile.max_relative_planarity),
        'min_normal_coherence':float(view.profile.min_normal_coherence),
        'max_extent_m':float(view.profile.max_extent_m),
    }


def _pareto_rows(summary,map_rows,target='xy0.5_yaw1'):
    stat={r['map_geometry']:r for r in map_rows}
    rows0=[r for r in summary if r['scenario']=='cross_session' and r['perturbation']==target]
    by={r['map_geometry']:r for r in rows0}; base=by.get('voxel_surfel_0.5m')
    if base is None: return []
    out=[]
    for geom,s in by.items():
        if geom not in stat: continue
        m=stat[geom]
        out.append({
            'map_geometry':geom,'profile':m.get('profile','baseline'),
            'max_primitives_per_root':m.get('max_primitives_per_root',1 if geom=='bst_native_block_surfel' else ''),
            'success_rate':float(s['success_rate']),
            'success_ratio_to_voxel':float(s['success_rate']/base['success_rate']) if base['success_rate'] else float('nan'),
            'meets_90pct_voxel_success_gate':bool(s['success_rate']>=0.9*base['success_rate']),
            'median_final_translation_error_m':float(s['median_final_translation_error_m']),
            'median_final_yaw_error_deg':float(s['median_final_yaw_error_deg']),
            'mean_local_map_primitives':float(s['mean_local_map_primitives']),
            'primitive_reduction_vs_voxel':float(1-s['mean_local_map_primitives']/base['mean_local_map_primitives']) if base['mean_local_map_primitives'] else 0.0,
            'median_total_ms':float(s['median_total_ms']),
            'total_speedup_vs_voxel':float(base['median_total_ms']/s['median_total_ms']) if s['median_total_ms'] else float('inf'),
            'minimal_geometry_payload_mib':float(m['minimal_geometry_payload_mib']),
            'bst_representation_mib':float(m['bst_representation_mib']),
            'query_index_mib':float(m['query_index_mib']),
            'map_elements':int(m['map_elements']),
            'mean_voxels_per_primitive_global':float(m['mean_voxels_per_primitive']),
        })
    for i,r in enumerate(out):
        dp=dt=False
        for j,q in enumerate(out):
            if i==j: continue
            if q['success_rate']>=r['success_rate'] and q['mean_local_map_primitives']<=r['mean_local_map_primitives'] and (q['success_rate']>r['success_rate'] or q['mean_local_map_primitives']<r['mean_local_map_primitives']): dp=True
            if q['success_rate']>=r['success_rate'] and q['median_total_ms']<=r['median_total_ms'] and (q['success_rate']>r['success_rate'] or q['median_total_ms']<r['median_total_ms']): dt=True
        r['pareto_success_vs_primitives']=not dp; r['pareto_success_vs_total_ms']=not dt
    return sorted(out,key=lambda r:(r['map_geometry']!='voxel_surfel_0.5m',r['mean_local_map_primitives']))


def benchmark_step912(
    *,data_root,output_dir,history_sessions,map_session='2012-01-15',query_session='2012-02-02',
    voxel_size_m=0.5,scan_stride=10,min_range_m=2.0,max_range_m=80.0,min_points_per_voxel=1,max_pose_gap_us=100000,cache_dir='.cache/step8_1',
    query_count=20,same_session_query_count=20,query_scans_csv=None,perturbations='0.5:1',perturb_seed=42090,
    localization_max_range_m=30.0,scan_downsample_m=0.75,max_scan_points=8000,local_map_radius_m=40.0,bucket_width_m=8.0,
    success_translation_m=0.5,success_yaw_deg=1.0,icp_config=None,include_same_session_zero=True,
    geometry_profile='balanced:0.050:0.025:0.950:2.0',k_values='2,4,8',progress=True,
):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    names=list(history_sessions)
    if map_session not in names: raise ValueError('map_session must appear in history_sessions')
    if query_session==map_session: raise ValueError('cross-session query_session must differ from map_session')
    if abs(float(voxel_size_m)-0.5)>1e-9: raise ValueError('Step 9.1.2 keeps the frozen Step-8 0.5m canonical lattice')
    profs=parse_geometry_profiles(geometry_profile)
    if len(profs)!=1: raise ValueError('Step 9.1.2 uses one frozen geometry gate profile while sweeping K')
    profile=profs[0]; kvals=parse_k_values(k_values)
    map_t=names.index(map_session)
    cfg=NCLTIngestConfig(voxel_size_m=float(voxel_size_m),scan_stride=int(scan_stride),min_range_m=float(min_range_m),max_range_m=float(max_range_m),min_points_per_voxel=int(min_points_per_voxel),max_pose_gap_us=int(max_pose_gap_us))
    base_cfg=icp_config or PointToPlaneICP3DOFConfig(); levels=parse_perturbations(perturbations); pyramid=DEFAULT_PYRAMID

    if query_scans_csv is not None:
        replay=_filter_replay_csv_for_cross_session(query_scans_csv,out/'_cross_replay.csv')
        cross=_replay_query_scans(data_root,query_session,replay,query_count,max_pose_gap_us)
    else: cross=_select_query_scans(data_root,query_session,query_count,max_pose_gap_us)
    same=_select_query_scans(data_root,map_session,same_session_query_count,max_pose_gap_us,exclude_ingest_stride=scan_stride) if include_same_session_zero else []
    prep_scan=SimpleNamespace(config=cfg); scan_rows=[]; loaded={}
    for scenario,selected in [('cross_session',cross),('same_session_control',same)]:
        for si,(sp,gt,gap,oi) in enumerate(selected):
            body,raw=_load_and_prepare_scan(sp,prep_scan,localization_max_range_m,scan_downsample_m,max_scan_points)
            loaded[(scenario,si)]=(sp,gt,gap,oi,body)
            _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
            scan_rows.append({'scenario':scenario,'scan_index':si,'original_scan_index':int(oi),'scan_file':sp.name,'timestamp_us':int(_timestamp_from_scan(sp)),'pose_gap_us':int(gap),'points_after_range':int(raw),'points_used':int(len(body)),'gt_x':gx,'gt_y':gy,'gt_z':gz,'gt_roll':roll,'gt_pitch':pitch,'gt_yaw':gyaw,'directly_ingested_scan':bool((oi%scan_stride)==0)})

    if progress: print('[9.1.2] loading five-session packed history from existing Step-8 cache...',flush=True)
    prepared=prepare_nclt_packed(data_root,names,cfg,cache_dir,progress_every=250)
    if progress: print('[9.1.2] building exact Step-8.4 session-local tilings...',flush=True)
    session_blocks,session_block_ms=build_session_spatial_blocks(prepared,len(names),progress=progress)
    if progress: print('[9.1.2] finalizing unchanged Step-8.5 time-aware BST map...',flush=True)
    t0=time.perf_counter_ns(); core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=progress); core_ms=(time.perf_counter_ns()-t0)/1e6
    bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())

    if progress: print('[9.1.2] rebuilding 0.5m map-session surfel sufficient statistics in memory...',flush=True)
    raw_surf=build_nclt_surfel_map(data_root,map_session,cfg,cache_dir,normal_min_points=base_cfg.normal_min_neighbors,normal_max_curvature=base_cfg.normal_max_curvature,normal_min_horizontal_component=base_cfg.normal_min_horizontal_component)
    aligned=align_voxel_surfels_to_prepared(raw_surf,prepared,map_t)
    t0=time.perf_counter_ns(); voxel_index=SurfelLocalMapIndex(aligned,bucket_width_m); voxel_index_ms=(time.perf_counter_ns()-t0)/1e6

    if progress: print('[9.1.2] building compact native merge hierarchy with internal surfel statistics...',flush=True)
    hierarchy=build_native_surfel_hierarchy(aligned,prepared,session_blocks,core,bst,map_t,base_cfg.normal_min_neighbors,base_cfg.normal_max_curvature,base_cfg.normal_min_horizontal_component,progress=progress)
    native_index=NativeBlockSurfelIndex(hierarchy.block_map)

    voxel_payload=int(sum(a.nbytes for a in (aligned.counts,aligned.centroids,aligned.cov6,aligned.normals,aligned.curvature,aligned.normal_valid)))
    voxel_min=int(sum(a.nbytes for a in (aligned.counts,aligned.centroids,aligned.normals,aligned.curvature,aligned.normal_valid)))
    bm=hierarchy.block_map
    map_rows=[{
        'map_geometry':'voxel_surfel_0.5m','profile':'baseline','max_primitives_per_root':'',
        'map_elements':len(aligned.linear_codes),'valid_normal_elements':int(np.count_nonzero(aligned.normal_valid)),'valid_normal_fraction':float(np.mean(aligned.normal_valid.astype(bool))),
        'geometry_payload_mib':voxel_payload/(1024**2),'minimal_geometry_payload_mib':voxel_min/(1024**2),'query_index_mib':voxel_index.nbytes()/(1024**2),'bst_representation_mib':0.0,
        'hierarchy_build_ms':0.0,'lod_decision_ms':0.0,'voxel_equivalent':len(aligned.linear_codes),'mean_voxels_per_primitive':1.0,
        'roots_refined':0,'root_refine_fraction':0.0,'mean_primitives_per_root':1.0,'p95_primitives_per_root':1.0,'max_observed_primitives_per_root':1,
        'selected_leaf_voxels':len(aligned.linear_codes),'selected_internal_nodes':0,'selected_leaf_fraction':1.0,
    },{
        'map_geometry':'bst_native_block_surfel','profile':'native_k1','max_primitives_per_root':1,
        'map_elements':len(bm.blocks),'valid_normal_elements':int(np.count_nonzero(bm.normal_valid)),'valid_normal_fraction':float(np.mean(bm.normal_valid.astype(bool))),
        'geometry_payload_mib':bm.payload_nbytes()/(1024**2),'minimal_geometry_payload_mib':bm.minimal_localization_payload_nbytes()/(1024**2),'query_index_mib':0.0,'bst_representation_mib':bst.compact_bytes()/(1024**2),
        'hierarchy_build_ms':float(hierarchy.build_ms),'lod_decision_ms':0.0,'voxel_equivalent':hierarchy.leaf_count,'mean_voxels_per_primitive':float(hierarchy.leaf_count/len(bm.blocks)),
        'roots_refined':0,'root_refine_fraction':0.0,'mean_primitives_per_root':1.0,'p95_primitives_per_root':1.0,'max_observed_primitives_per_root':1,
        'selected_leaf_voxels':int(np.count_nonzero(hierarchy.root_node_ids<hierarchy.leaf_count)),'selected_internal_nodes':int(np.count_nonzero(hierarchy.root_node_ids>=hierarchy.leaf_count)),
        'selected_leaf_fraction':float(np.mean(hierarchy.root_node_ids<hierarchy.leaf_count)),
    }]

    views=[]
    for k in kvals:
        if k==1: continue
        view=build_hierarchical_lod_view(hierarchy,profile,k); views.append(view); map_rows.append(_lod_stat(view,hierarchy,bst))
        if progress:
            st=map_rows[-1]
            print(f"[9.1.2] K={k}: elements={st['map_elements']:,}, mean={st['mean_voxels_per_primitive']:.2f} vox/primitive, refined roots={100*st['root_refine_fraction']:.1f}%",flush=True)

    rows=[]
    def run_geometry(geom,index):
        if progress: print(f'[9.1.2] localizing with {geom}...',flush=True)
        def scenario_run(scenario,selected,perturb,seed):
            rng=random.Random(int(seed))
            for si,_ in enumerate(selected):
                sp,gt,gap,oi,body=loaded[(scenario,si)]; _,gx,gy,_,_,_,gyaw=[float(v) for v in gt]
                specs=[('zero_control',0.,0.,0.,0.,0.)]
                if perturb:
                    for label,xy,yaw in levels:
                        dx,dy,dyaw=_trial_perturbation(xy,yaw,rng); specs.append((label,xy,yaw,dx,dy,dyaw))
                for label,xy,yaw,dx,dy,dyaw in specs:
                    init=Pose3DOF(gx+dx,gy+dy,gyaw+math.radians(dyaw))
                    rows.append(_run_trial(map_geometry=geom,scenario=scenario,scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,original_scan_index=oi,init=init,perturbation_label=label,xy_mag=xy,yaw_mag=yaw,dx=dx,dy=dy,dyaw_deg=dyaw,map_index=index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg))
        scenario_run('cross_session',cross,True,perturb_seed)
        if include_same_session_zero: scenario_run('same_session_control',same,False,perturb_seed+902)

    run_geometry('voxel_surfel_0.5m',voxel_index); run_geometry('bst_native_block_surfel',native_index)
    for view in views:
        idx=HierarchicalLODBSTIndex(view,bst,map_t); run_geometry(f'bst_lod_k{view.max_primitives_per_root}',idx)

    summary=_summary(rows); pareto=_pareto_rows(summary,map_rows,'xy0.5_yaw1')
    _write_csv(out/'localization_trials.csv',rows); _write_csv(out/'localization_summary.csv',summary); _write_csv(out/'query_scans.csv',scan_rows)
    _write_csv(out/'lod_profile_stats.csv',map_rows); _write_csv(out/'pareto_summary.csv',pareto)
    tmp=out/'_cross_replay.csv'
    if tmp.exists(): tmp.unlink()

    metadata=prepared.metadata(); metadata.update({
        'step':'9.1.2','purpose':'Hierarchical multi-resolution native block-surfels: refine rejected BST roots through their binary merge hierarchy with capped K surfels per original root.',
        'history_sessions':names,'map_session':map_session,'map_session_index':map_t,'query_session':query_session,'query_count':len(cross),'same_session_query_count':len(same),
        'query_scans_replay_csv':str(query_scans_csv) if query_scans_csv else None,'perturbations':[{'label':l,'xy_m':x,'yaw_deg':y} for l,x,y in levels],
        'localization_max_range_m':localization_max_range_m,'scan_downsample_m':scan_downsample_m,'max_scan_points':max_scan_points,'local_map_radius_m':local_map_radius_m,
        'success_thresholds':{'translation_m':success_translation_m,'yaw_deg':success_yaw_deg},'base_icp_config':asdict(base_cfg),'pyramid':[asdict(x) for x in pyramid],
        'geometry_profile':asdict(profile),'k_values':[1]+[k for k in kvals if k!=1],
        'hierarchical_lod_policy':{
            'online_decision':False,'underlying_bst_changed':False,'query_path':'unchanged Step-8.5 counted (bucket,session) time-aware BST index',
            'tree':'binary dyadic merge forest retained while exactly reproducing the Step-8.4 map-session native roots',
            'selection':'start at each native root; if gate fails, descend to its two children; stop early when a node passes or the log2(K) depth cap is reached',
            'max_primitives_per_root':'K in {1,2,4,8} by default; K is an upper bound, not a forced count',
            'gate_signals':['absolute plane sigma sqrt(lambda_min)','relative planarity lambda_min/trace(covariance)','sign-invariant immediate-child normal agreement','maximum node extent'],
            'normal_coherence':'abs(dot(n_child_a,n_child_b)) when both child normals are valid; otherwise no disagreement penalty',
            'runtime':'all subdivision choices and root->LOD tables are precomputed offline; localization performs BST root query + ragged table lookup + radius filtering only',
        },
        'hierarchy_internal_nodes':int(hierarchy.internal_count),'hierarchy_runtime_payload_mib':hierarchy.internal_payload_nbytes()/(1024**2),
        'map_geometry_stats':map_rows,
        'threshold_selection_protocol':{'role':'development/tuning set','cross_session_scans':len(cross),'warning':'The balanced gate came from Step 9.1.1 development data and K is tuned here on the same development scan set. Validate the selected operating point on new held-out scans and additional session pairs before any final performance claim.'},
        'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'scientific_caveat':'Step 9.1.2 is a localization LOD overlay over the unchanged Step-8.5 lifelong BST/history map. It does not alter stored occupancy/history semantics. PrototypeBSTSpatialCodec remains a local research codec, not verified upstream BST-ID.',
    })
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return rows,summary,map_rows,pareto


__all__=['benchmark_step912','parse_k_values']
