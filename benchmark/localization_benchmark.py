"""Unified causal lifelong localization benchmark.

One frozen mapping/registration loop for BEVPlace++ and Scan Context.
Voxels and BST K=4 consume the exact same global pose hypothesis.
"""
from __future__ import annotations
import csv
import gc
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from benchmark.localization_inputs import (_load_body, _prep_global_points, _select_queries, build_or_load_reference_database)
from benchmark.localization_helpers import (FROZEN_PROFILE, FROZEN_K, _runtime_voxel_geometry_bytes, _write_csv)
from benchmark.localization_reports import _summary, _session_summary, _bevplace_summary, _scan_context_summary
from localization.pyramid import voxel_downsample_first
from localization.surfel_icp import SurfelLocalMapIndex
from localization.surfels import align_voxel_surfels_to_prepared
from localization.lod import build_native_surfel_hierarchy, build_hierarchical_lod_view
from localization.compact_map import materialize_lod_view, MaterializedLODBSTIndex, IndexedMaterializedLODBSTIndex
from localization.history import causal_latest_prior_plan
from localization.retrieval import HistoricalRetriever
from benchmark.orientation_diagnostics import make_pose_audit
from localization.native_surfel_registration import NativeSurfelConfig, align_native_surfels
from localization.native_surfel_prepared import PreparedSurfelMap
from localization.surfel_6dof import align_surfel_6dof, Surfel6DConfig
from localization.bev_local_pose import LocalFeatureStore, estimate_pose
from localization.bev_hybrid import HybridBEVPolicy
from localization.frontend_adaptive import EarlyExitPolicy, GeometryVerificationPolicy, verified_candidate, verify_alignment_geometry
from localization.global_frontend import (ScanContextConfig, IsolatedKISSMatcherBackend, SmallGICPBackend, make_scan_context, scan_context_align, pose_row_to_matrix, pose_errors_6d, scan_context_pose_guess, compose_world_query, require_frontend_dependencies)
from realdata.nclt import (NCLTIngestConfig, _timestamp_from_scan)
from realdata.nclt_packed import prepare_nclt_packed
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec

VOXEL_METHOD = 'bevplace_kiss_gicp_latest_prior_voxel'
BST_METHOD = 'bevplace_kiss_gicp_latest_prior_bst_k4'

def method_names(retriever):
    if retriever == 'bevplace':
        return VOXEL_METHOD, BST_METHOD
    if retriever == 'scan_context':
        return 'sc_kiss_gicp_latest_prior_voxel', 'sc_kiss_gicp_latest_prior_bst_k4'
    raise ValueError(f'unsupported retriever: {retriever}')

def benchmark_localization(
    *, data_root, output_dir, history_sessions,
    map_session='2012-01-15', query_sessions=('2012-02-02','2012-03-17','2012-05-11','2012-06-15'),
    query_count_per_session=50, query_scans_csv=None, exclude_query_scans_csv=(),
    voxel_size_m=0.5, scan_stride=10, min_range_m=2.0, max_range_m=80.0,
    min_points_per_voxel=1, max_pose_gap_us=100000, cache_dir='.cache/step8_1',
    sc_config=None, sc_top_k=5, sc_candidate_pool=30, reference_scan_stride=5,
    reference_min_translation_m=5.0, reference_min_yaw_deg=15.0, reference_max_time_s=5.0,
    descriptor_voxel_m=0.5, global_point_voxel_m=0.30, global_max_points=20000,
    kiss_voxel_m=0.30, scan_gicp_downsample_m=0.30, scan_gicp_max_corr_m=2.0,
    map_gicp_downsample_m=0.40, map_gicp_max_corr_m=2.0, gicp_threads=4,
    local_map_radius_m=50.0, bucket_width_m=8.0,
    fine_success_translation_m=0.5, fine_success_yaw_deg=1.0,
    strict_6d_translation_m=0.5, strict_6d_rotation_deg=2.0,
    global_success_translation_m=2.0, global_success_rotation_deg=5.0,
    include_voxel_baseline=True, progress=True, descriptor_cache_dir=None, retriever="bevplace",
    orientation_diagnostics=False, local_solver="gicp", native_yaw_multistart=False,
    frontend_pose_cache=None, save_frontend_poses=False,
    frontend_strategy="exhaustive", frontend_min_inliers=350, frontend_min_inlier_fraction=0.025,
    frontend_geo_radius_m=1.0, frontend_geo_min_overlap=0.55, frontend_geo_max_rmse_m=0.65,
    bev_local_feature_dir=None, hybrid_bev_min_inliers=28, hybrid_bev_min_ratio=0.07, hybrid_bev_max_candidates=2,
    bst_xy_index=False, surfel6d_fast_tree=False, surfel6d_nn_workers=1, surfel6d_leafsize=16,
):
    if frontend_strategy not in ('exhaustive','adaptive','adaptive_geo','bev_local','bev_hybrid'):
        raise ValueError('frontend_strategy must be exhaustive or adaptive')
    if hybrid_bev_min_inliers < 1 or not (0 < hybrid_bev_min_ratio <= 1) or hybrid_bev_max_candidates < 1:
        raise ValueError("Invalid hybrid BEV confidence configuration")
    hybrid_policy=HybridBEVPolicy(hybrid_bev_min_inliers,hybrid_bev_min_ratio,hybrid_bev_max_candidates)
    early_policy=EarlyExitPolicy(frontend_min_inliers,frontend_min_inlier_fraction)
    geo_policy=GeometryVerificationPolicy(overlap_radius_m=frontend_geo_radius_m,min_overlap=frontend_geo_min_overlap,max_trimmed_rmse_m=frontend_geo_max_rmse_m)
    candidate_trace=[]
    local_features=LocalFeatureStore(bev_local_feature_dir) if frontend_strategy in ('bev_local','bev_hybrid') and bev_local_feature_dir else None
    if frontend_strategy in ('bev_local','bev_hybrid') and (retriever!='bevplace' or local_features is None):
        raise ValueError('bev_local requires BEVPlace retrieval and local features')
    if local_solver not in ('gicp','native_surfel','surfel_6dof'):
        raise ValueError('local_solver must be gicp or native_surfel')
    pose_cache={}
    if frontend_pose_cache:
        with np.load(frontend_pose_cache,allow_pickle=False) as z:
            keys=np.asarray(z['keys']); mats=np.asarray(z['poses'])
            if len(keys)!=len(mats) or mats.shape!=(len(keys),4,4) or len(set(keys.tolist()))!=len(keys):
                raise ValueError('Invalid frontend pose cache')
            pose_cache=dict(zip(keys.tolist(),mats))
    output_frontend_poses={}
    native_cfg=NativeSurfelConfig(candidate_offsets_deg=(0.0,-1.5,1.5) if native_yaw_multistart else (0.0,))
    provider = HistoricalRetriever(retriever, cache_dir=descriptor_cache_dir, sc_config=sc_config or ScanContextConfig(max_radius_m=max_range_m),
        top_k=sc_top_k, candidate_pool=sc_candidate_pool, descriptor_voxel_m=descriptor_voxel_m)
    method_voxel, method_bst = method_names(retriever)
    if frontend_strategy in ('bev_local','bev_hybrid'):
        method_voxel='bevplace_'+frontend_strategy+'_native_latest_prior_voxel'
        method_bst='bevplace_'+frontend_strategy+'_native_latest_prior_bst_k4'
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    names=list(history_sessions); qs=list(query_sessions)
    if abs(float(voxel_size_m)-0.5)>1e-9:
        raise ValueError('Step 9.5 keeps the frozen 0.5m canonical lattice')
    if map_session not in names: raise ValueError('map_session missing from history_sessions')
    plan=causal_latest_prior_plan(names,map_session,qs)
    deps=require_frontend_dependencies(frontend_strategy!='bev_local',local_solver=='gicp' or frontend_strategy!='bev_local')  # fail before expensive map construction
    sc=sc_config or ScanContextConfig(max_radius_m=max_range_m)
    sc.validate()
    cfg=NCLTIngestConfig(voxel_size_m=float(voxel_size_m),scan_stride=int(scan_stride),min_range_m=float(min_range_m),
        max_range_m=float(max_range_m),min_points_per_voxel=int(min_points_per_voxel),max_pose_gap_us=int(max_pose_gap_us))
    selected=_select_queries(data_root,qs,query_count_per_session,max_pose_gap_us,query_scans_csv,exclude_query_scans_csv)

    query_rows=[]; query_data={}
    for p in plan:
        for si,(sp,gt,gap,oi) in enumerate(selected[p.query_session]):
            body_sc=_load_body(sp,cfg,sc.max_radius_m)
            body_global=_prep_global_points(body_sc,global_point_voxel_m,global_max_points)
            query_data[(p.query_session,si)]=(sp,np.asarray(gt,dtype=np.float64),gap,oi,body_sc,body_global)
            query_rows.append({'query_session':p.query_session,'historical_map_session':p.latest_prior_session,'scan_index':si,
                'original_scan_index':int(oi),'scan_file':sp.name,'timestamp_us':int(_timestamp_from_scan(sp)),
                'pose_gap_us':int(gap),'scan_context_points':int(len(body_sc)),'global_registration_points':int(len(body_global))})

    if progress: print('[9.5] loading packed five-session history once...',flush=True)
    prepared=prepare_nclt_packed(data_root,names,cfg,cache_dir,progress_every=250)
    if progress: print('[9.5] building frozen Step-8.4/8.5 historical BST once...',flush=True)
    session_blocks,session_block_ms=build_session_spatial_blocks(prepared,len(names),progress=progress)
    t0=time.perf_counter_ns(); core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=progress); core_ms=(time.perf_counter_ns()-t0)/1e6
    bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())

    # Only instantiate registration backends actually used by the selected route.
    # In particular, bev_local + surfel_6dof requires neither KISS nor small_gicp.
    kiss = IsolatedKISSMatcherBackend(kiss_voxel_m) if frontend_strategy != 'bev_local' else None
    scan_gicp = (SmallGICPBackend(
        downsampling_resolution_m=scan_gicp_downsample_m,
        max_correspondence_distance_m=scan_gicp_max_corr_m,
        num_threads=gicp_threads, module=deps['small_gicp'])
        if frontend_strategy != 'bev_local' else None)
    map_gicp = (SmallGICPBackend(
        downsampling_resolution_m=map_gicp_downsample_m,
        max_correspondence_distance_m=map_gicp_max_corr_m,
        num_threads=gicp_threads, module=deps['small_gicp'])
        if local_solver == 'gicp' else None)

    rows=[]; frontend_rows=[]; ref_stats=[]; map_stats=[]
    orientation_audit_rows=[]
    unique_prior=[]
    for p in plan:
        if p.latest_prior_index not in unique_prior: unique_prior.append(p.latest_prior_index)

    for prior_i in unique_prior:
        ref_scan_cache={}
        prior_session=names[prior_i]
        assoc=[p for p in plan if p.latest_prior_index==prior_i]
        if progress: print(f'[9.5] building/loading Scan Context database for {prior_session}...',flush=True)
        ref=build_or_load_reference_database(data_root,prior_session,cfg,cache_dir,sc,
            reference_scan_stride=reference_scan_stride,min_translation_m=reference_min_translation_m,
            min_yaw_deg=reference_min_yaw_deg,max_time_s=reference_max_time_s,
            descriptor_voxel_m=descriptor_voxel_m,progress=progress)
        # Preserve exact historical reference scan order for both retrievers.
        bev_bytes = provider.load_reference(prior_session,[q.query_session for q in assoc],ref)
        ref_stats.append({'historical_map_session':prior_session,'keyframes':len(ref['scan_indices']),
            'cache_hit':ref['cache_hit'],'build_ms':ref['build_ms'],'descriptor_bytes':int(ref['descriptors'].nbytes+ref['ring_keys'].nbytes),**({'bevplace_descriptor_bytes':bev_bytes} if retriever == 'bevplace' else {}),
            'cache_path':ref['cache_path']})

        if progress: print(f'[9.5] building latest-prior voxel + compact BST K=4 map for {prior_session}...',flush=True)
        raw=build_nclt_surfel_map(data_root,prior_session,cfg,cache_dir,normal_min_points=6,
            normal_max_curvature=0.20,normal_min_horizontal_component=0.20)
        aligned=align_voxel_surfels_to_prepared(raw,prepared,prior_i)
        voxel_index=SurfelLocalMapIndex(aligned,bucket_width_m)
        hierarchy=build_native_surfel_hierarchy(aligned,prepared,session_blocks,core,bst,prior_i,6,0.20,0.20,progress=progress)
        view=build_hierarchical_lod_view(hierarchy,FROZEN_PROFILE,FROZEN_K); compact=materialize_lod_view(view)
        bst_index=(IndexedMaterializedLODBSTIndex if bst_xy_index else MaterializedLODBSTIndex)(compact,bst,prior_i)
        # Precompute frozen covariance eigenvectors once per historical map, not per query.
        prepared_native = (
            ((PreparedSurfelMap.build(aligned.cov6, aligned.counts, native_cfg) if include_voxel_baseline else None),
             PreparedSurfelMap.build(compact.cov6, compact.counts, native_cfg))
            if local_solver == "native_surfel" else (None, None)
        )
        if local_solver == "surfel_6dof":
            # Prepare ALL surface orientations (ground included) once per map.
            sixdof_cfg=NativeSurfelConfig(min_horizontal_normal=0.0)
            prepared_native=(
                PreparedSurfelMap.build(aligned.cov6,aligned.counts,sixdof_cfg) if include_voxel_baseline else None,
                PreparedSurfelMap.build(compact.cov6,compact.counts,sixdof_cfg),
            )
        map_stats.append({'historical_map_session':prior_session,'voxel_elements':len(aligned.centroids),
            'bst_k4_elements':len(compact.centroids),'voxel_geometry_mib':_runtime_voxel_geometry_bytes(aligned)/(1024**2),
            'bst_k4_geometry_mib':compact.geometry_payload_nbytes()/(1024**2),'bst_k4_directory_mib':compact.directory_nbytes()/(1024**2),
            'bst_xy_index_extra_mib':getattr(bst_index,'extra_index_nbytes_estimate',0)/(1024**2),
            'bst_xy_index_prepare_ms':getattr(bst_index,'prepare_ms',0.0)})

        for p in assoc:
            if progress: print(f'[9.5] global relocalization {prior_session}->{p.query_session} ({len(selected[p.query_session])} scans)...',flush=True)
            for si in range(len(selected[p.query_session])):
                sp,gt,gap,oi,body_sc,body_global=query_data[(p.query_session,si)]
                T_gt=pose_row_to_matrix(gt)
                ft0=time.perf_counter_ns()
                matches=provider.query(p.query_session, sp.name, int(oi), body_sc)
                if not matches:
                    continue
                gt_xy=T_gt[:2,3]
                pos_errs=[]
                for m in matches:
                    rp=ref['pose_rows'][m.reference_index]
                    pos_errs.append(math.hypot(float(rp[1]-gt_xy[0]),float(rp[2]-gt_xy[1])))
                retrieval_ms=(time.perf_counter_ns()-ft0)/1e6

                pose_key=f"{p.query_session}|{prior_session}|{int(oi)}|{sp.name}"
                if frontend_pose_cache:
                    if pose_key not in pose_cache:
                        raise KeyError(f'Frontend replay key absent: {pose_key}')
                    T_front=np.asarray(pose_cache[pose_key],dtype=np.float64).copy()
                    if T_front.shape!=(4,4) or not np.all(np.isfinite(T_front)):
                        raise ValueError(f'Invalid frontend transform: {pose_key}')
                    m_best=matches[0];best_ref_i=m_best.reference_index
                    k_best=None;g_best=None;frontend_reason='cached_frontend_pose';used_kiss=False
                    frontend_ms=0.0 # replay is not an online latency measurement
                else:
                    candidates=[]
                    hybrid_accepted=False
                    if frontend_strategy in ('bev_local','bev_hybrid'):
                        q_xy,q_feat=local_features.get('query',p.query_session,si,sp.name)
                        bev_matches=matches[:hybrid_bev_max_candidates] if frontend_strategy=='bev_hybrid' else matches
                        for candidate_rank,m in enumerate(bev_matches,1):
                            attempt_start=time.perf_counter_ns()
                            ref_db_i=int(m.reference_index)
                            scan_i=int(ref['scan_indices'][ref_db_i])
                            r_xy,r_feat=local_features.get('reference',prior_session,ref_db_i,ref['scans'][scan_i].name)
                            local=estimate_pose(q_xy,q_feat,r_xy,r_feat)
                            candidate_trace.append({'query_session':p.query_session,'historical_map_session':prior_session,
                                'original_scan_index':int(oi),'scan_file':sp.name,'candidate_rank':candidate_rank,
                                'reference_index':ref_db_i,'reference_scan_index':scan_i,
                                'descriptor_distance':float(m.distance),'kiss_converged':False,'kiss_inliers':0,
                                'scan_gicp_ran':False,'scan_gicp_converged':False,'scan_gicp_inliers':0,
                                'verified_for_early_exit':bool(local.converged),'inlier_gate_pass':False,
                                'geometry_gate_pass':False,'geometry_overlap':float('nan'),
                                'geometry_trimmed_rmse_m':float('nan'),
                                'attempt_wall_ms':(time.perf_counter_ns()-attempt_start)/1e6,
                                'kiss_reported_ms':0.0,'gicp_reported_ms':0.0,
                                'bev_local_inliers':int(local.inliers),'bev_local_matching_ms':float(local.runtime_ms),
                                'bev_local_failure_reason':local.failure_reason})
                            bev_ok=hybrid_policy.accepts(local,len(q_xy),len(r_xy))
                            candidate_trace[-1]['hybrid_bev_confident']=bev_ok
                            candidate_trace[-1]['verified_for_early_exit']=bev_ok if frontend_strategy=='bev_hybrid' else bool(local.converged)
                            candidate_trace[-1]['backend']='bev_local'
                            if not local.converged:continue
                            if frontend_strategy=='bev_hybrid' and not bev_ok:continue
                            Twq=compose_world_query(ref['pose_rows'][ref_db_i],local.T_target_source)
                            candidates.append((int(local.inliers),0,-float(m.distance),Twq,m,None,None,
                                               float(local.runtime_ms),ref_db_i))
                            hybrid_accepted=True
                            break
                    # BEV-first: on any untrusted or missing planar pose, use the original
                    # adaptive KISS scan-registration sequence from candidate rank one.
                    run_kiss=(frontend_strategy not in ('bev_local','bev_hybrid') or (frontend_strategy=='bev_hybrid' and not hybrid_accepted))
                    for candidate_rank,m in (enumerate(matches,1) if run_kiss else []):
                        attempt_start=time.perf_counter_ns()
                        ref_db_i=m.reference_index
                        scan_i=int(ref['scan_indices'][ref_db_i])
                        key=(prior_session,scan_i)
                        if key not in ref_scan_cache:
                            rb=_load_body(ref['scans'][scan_i],cfg,sc.max_radius_m)
                            ref_scan_cache[key]=_prep_global_points(rb,global_point_voxel_m,global_max_points)
                        ref_body=ref_scan_cache[key]
                        kres=kiss.estimate(body_global,ref_body)
                        if kres.converged:
                            gres=scan_gicp.align(ref_body,body_global,kres.T_target_source)
                            T_ref_query=gres.T_target_source if gres.converged else kres.T_target_source
                        else:
                            gres=None; T_ref_query=None
                        verified = verified_candidate(kres,gres,len(body_global),early_policy) if T_ref_query is not None else False
                        geom_ok=False; geom_overlap=float('nan');geom_rmse=float('nan')
                        if frontend_strategy == 'adaptive_geo' and verified and gres is not None:
                            geom_ok,geom_overlap,geom_rmse=verify_alignment_geometry(body_global,ref_body,gres.T_target_source,geo_policy)
                        early_stop = verified if frontend_strategy in ('adaptive','bev_hybrid') else (verified and geom_ok if frontend_strategy == 'adaptive_geo' else False)
                        candidate_trace.append({
                            'backend':'kiss', 'query_session':p.query_session,'historical_map_session':prior_session,
                            'original_scan_index':int(oi),'scan_file':sp.name,
                            'candidate_rank':candidate_rank,'reference_index':int(ref_db_i),
                            'reference_scan_index':scan_i,'descriptor_distance':float(m.distance),
                            'kiss_converged':bool(kres.converged),
                            'kiss_inliers':int(kres.inliers),
                            'scan_gicp_ran':bool(gres is not None),
                            'scan_gicp_converged':bool(gres.converged) if gres is not None else False,
                            'scan_gicp_inliers':int(gres.inliers) if gres is not None else 0,
                            'verified_for_early_exit':bool(early_stop),
                            'inlier_gate_pass':bool(verified),'geometry_gate_pass':bool(geom_ok),
                            'geometry_overlap':geom_overlap,'geometry_trimmed_rmse_m':geom_rmse,
                            'attempt_wall_ms':(time.perf_counter_ns()-attempt_start)/1e6,
                            'kiss_reported_ms':float(kres.runtime_ms),
                            'gicp_reported_ms':float(gres.runtime_ms) if gres is not None else 0.0,
                        })
                        if T_ref_query is None:
                            continue
                        Twq=compose_world_query(ref['pose_rows'][ref_db_i],T_ref_query)
                        scan_inliers=int(gres.inliers if gres is not None and gres.converged else 0)
                        cand_runtime=float(kres.runtime_ms+(gres.runtime_ms if gres is not None else 0.0))
                        candidates.append((scan_inliers,int(kres.inliers),-float(m.distance),Twq,m,kres,gres,cand_runtime,ref_db_i))
                        if early_stop:
                            break

                    if candidates:
                        candidates.sort(key=lambda x:(x[0],x[1],x[2]),reverse=True)
                        _,_,_,T_front,m_best,k_best,g_best,cand_ms,best_ref_i=candidates[0]
                        frontend_reason=''
                        used_kiss=bool(run_kiss)
                    else:
                        m_best=matches[0]; best_ref_i=m_best.reference_index
                        if retriever == 'bevplace':
                            # Retain original SC-only fallback yaw (no learned yaw).
                            sqdesc=make_scan_context(voxel_downsample_first(body_sc,descriptor_voxel_m),sc)
                            ref_desc=ref['descriptors'][best_ref_i]
                            _, _, yaw=scan_context_align(sqdesc,ref_desc)
                        else:
                            yaw=m_best.relative_yaw_rad
                        T_front=scan_context_pose_guess(ref['pose_rows'][best_ref_i],yaw)
                        k_best=None; g_best=None; cand_ms=0.0; frontend_reason=('hybrid_both_failed_sc_yaw_fallback' if frontend_strategy=='bev_hybrid' else ('bev_local_failed_sc_yaw_fallback' if frontend_strategy=='bev_local' else ('kiss_failed_all_candidates_sc_yaw_fallback' if retriever == 'bevplace' else 'kiss_failed_all_candidates'))); used_kiss=False
                    # Include failed KISS attempts, lazy reference loads, and fallback.
                    # Offline BEV neural extraction time is NOT included here.
                    frontend_ms=(time.perf_counter_ns()-ft0)/1e6
                output_frontend_poses[pose_key]=np.array(T_front,copy=True)
                ferr=pose_errors_6d(T_front,T_gt)
                fr={
                    'query_session':p.query_session,'historical_map_session':prior_session,'scan_index':si,
                    'original_scan_index':int(oi),'scan_file':sp.name,
                    **({'bevplace_top1_position_error_m':float(pos_errs[0]),'bevplace_topk_best_position_error_m':float(min(pos_errs))} if retriever == 'bevplace' else {'sc_top1_position_error_m':float(pos_errs[0]),'sc_topk_best_position_error_m':float(min(pos_errs))}),
                    **({'bevplace_top1_descriptor_distance':float(matches[0].distance),'bevplace_topk':int(len(matches))} if retriever == 'bevplace' else {'sc_top1_distance':float(matches[0].distance),'sc_topk':int(len(matches))}),
                    'selected_reference_rank':int(next(i for i,x in enumerate(matches) if x.reference_index==best_ref_i)+1),
                    'selected_reference_scan_index':int(ref['scan_indices'][best_ref_i]),
                    'selected_reference_position_error_m':math.hypot(float(ref['pose_rows'][best_ref_i,1]-gt[1]),float(ref['pose_rows'][best_ref_i,2]-gt[2])),
                    'frontend_replayed':bool(frontend_pose_cache),'kiss_used':bool(used_kiss),'kiss_final_inliers':int(k_best.inliers if k_best else 0),
                    'kiss_rotation_inliers':int(k_best.rotation_inliers if k_best else 0),
                    'scan_gicp_inliers':int(g_best.inliers if g_best else 0),'scan_gicp_converged':bool(g_best.converged if g_best else False),
                    'frontend_translation_xy_m':ferr['translation_xy_m'],'frontend_translation_3d_m':ferr['translation_3d_m'],
                    'frontend_yaw_error_deg':ferr['yaw_deg'],'frontend_rotation_error_deg':ferr['rotation_deg'],
                    **({'bevplace_descriptor_retrieval_ms':float(retrieval_ms)} if retriever == 'bevplace' else {'scan_context_ms':float(retrieval_ms)}),
                    'frontend_ms':float(frontend_ms),'frontend_failure_reason':frontend_reason,
                    'frontend_strategy':frontend_strategy,
                    'frontend_candidates_attempted':sum(1 for r in candidate_trace if r['query_session']==p.query_session and r['historical_map_session']==prior_session and r['original_scan_index']==int(oi) and r['scan_file']==sp.name) if not frontend_pose_cache else 0,
                }
                frontend_rows.append(fr)

                for method,index,target_map,prepared_map in (
                    (method_voxel,voxel_index,aligned,prepared_native[0]),
                    (method_bst,bst_index,compact,prepared_native[1]),
                ):
                    if method==method_voxel and not include_voxel_baseline: continue
                    total0=time.perf_counter_ns()
                    center=T_front[:2,3]
                    tr0=time.perf_counter_ns(); ids=index.query_radius_ids(float(center[0]),float(center[1]),local_map_radius_m); retrieval_ms=(time.perf_counter_ns()-tr0)/1e6
                    target=np.asarray(target_map.centroids[np.asarray(ids,dtype=np.int64)],dtype=np.float64) if len(ids) else np.empty((0,3),dtype=np.float64)
                    if local_solver == 'surfel_6dof':
                        selected_ids=np.asarray(ids,dtype=np.int64)
                        lres=align_surfel_6dof(target,target_map.cov6[selected_ids],target_map.counts[selected_ids],body_global,T_front,
                            prepared=prepared_map.select(selected_ids), fast_tree=surfel6d_fast_tree,
                            config=Surfel6DConfig(nn_workers=surfel6d_nn_workers, tree_leafsize=surfel6d_leafsize))
                    elif local_solver == 'native_surfel':
                        selected_ids=np.asarray(ids,dtype=np.int64)
                        lres=align_native_surfels(target,target_map.cov6[selected_ids],target_map.counts[selected_ids],body_global,T_front,native_cfg,
                            prepared=prepared_map.select(selected_ids))
                    else:
                        lres=map_gicp.align(target,body_global,T_front)
                    T_final=lres.T_target_source if lres.converged else T_front
                    err=pose_errors_6d(T_final,T_gt)
                    total_ms=frontend_ms+(time.perf_counter_ns()-total0)/1e6
                    rows.append({
                        'map_geometry':method,'query_session':p.query_session,'historical_map_session':prior_session,
                        'scan_index':si,'original_scan_index':int(oi),'scan_file':sp.name,
                        'success':bool(err['translation_xy_m']<=fine_success_translation_m and err['yaw_deg']<=fine_success_yaw_deg),
                        'global_2m5deg_success':bool(err['translation_3d_m']<=global_success_translation_m and err['rotation_deg']<=global_success_rotation_deg),
                        'strict_6d_success':bool(err['translation_3d_m']<=strict_6d_translation_m and err['rotation_deg']<=strict_6d_rotation_deg),
                        'final_translation_xy_m':err['translation_xy_m'],'final_translation_3d_m':err['translation_3d_m'],
                        'final_yaw_error_deg':err['yaw_deg'],'final_rotation_error_deg':err['rotation_deg'],
                        'frontend_translation_xy_m':ferr['translation_xy_m'],'frontend_translation_3d_m':ferr['translation_3d_m'],
                        'frontend_yaw_error_deg':ferr['yaw_deg'],'frontend_rotation_error_deg':ferr['rotation_deg'],
                        **({'bevplace_top1_position_error_m':float(pos_errs[0]),'bevplace_topk_best_position_error_m':float(min(pos_errs))} if retriever == 'bevplace' else {'sc_top1_position_error_m':float(pos_errs[0]),'sc_topk_best_position_error_m':float(min(pos_errs))}),
                        'kiss_final_inliers':int(k_best.inliers if k_best else 0),'scan_gicp_inliers':int(g_best.inliers if g_best else 0),
                        'frontend_ms':float(frontend_ms),'map_retrieval_ms':float(retrieval_ms),'local_gicp_ms':float(lres.runtime_ms),
                        'total_ms':float(total_ms),'local_map_primitives':int(len(ids)),'local_gicp_inliers':int(lres.inliers),
                        **({'surfel6d_'+k:float(v) for k,v in lres.profile_ms.items()} if local_solver=='surfel_6dof' else {}),
                        'local_gicp_converged':bool(lres.converged),'failure_reason':lres.failure_reason,
                    })
                    # Step 9.7A instrumentation occurs OUTSIDE reported query timing.
                    # The selection T_final and all frozen CSV fields are unchanged.
                    if orientation_diagnostics:
                        orientation_audit_rows.append(make_pose_audit(
                            map_geometry=method, query_session=p.query_session,
                            scan_file=sp.name, original_scan_index=int(oi),
                            T_gt=T_gt, T_front=T_front, T_final=T_final,
                            gicp_result=lres, target_points=target,
                            source_points=body_global,
                            max_correspondence_m=map_gicp_max_corr_m))

        del prepared_native, hierarchy,view,compact,bst_index,voxel_index,aligned,raw
        gc.collect()

    if candidate_trace:
        _write_csv(out/"frontend_candidate_trials.csv",candidate_trace)
    # Explicitly stop the isolated KISS worker before final reporting.
    if kiss is not None:
        kiss.close()
    summary=_summary(rows); session_summary=_session_summary(rows); place_summary=(_bevplace_summary if retriever == 'bevplace' else _scan_context_summary)(frontend_rows)
    # Fair representation gate: same frontend, only final historical map representation differs.
    gates=[]
    if include_voxel_baseline:
        a=next(r for r in summary if r['map_geometry']==method_voxel); b=next(r for r in summary if r['map_geometry']==method_bst)
        retain=b['fine_success_rate']/a['fine_success_rate'] if a['fine_success_rate']>0 else float('nan')
        prim_red=1.0-b['mean_local_map_primitives']/a['mean_local_map_primitives']
        local_speed=a['median_local_gicp_ms']/b['median_local_gicp_ms'] if b['median_local_gicp_ms']>0 else float('inf')
        gates=[{'bst_fine_success_retention':retain,'meets_success_retention_0.90':bool(retain>=0.90),
            'bst_primitive_reduction':prim_red,'meets_primitive_reduction_0.40':bool(prim_red>=0.40),
            'bst_local_gicp_speedup':local_speed,'representation_gate_pass':bool(retain>=0.90 and prim_red>=0.40)}]

    if save_frontend_poses:
        keys=list(output_frontend_poses)
        np.savez_compressed(out/'frontend_poses.npz',keys=np.asarray(keys),poses=np.stack([output_frontend_poses[k] for k in keys]))
    if orientation_diagnostics:
        _write_csv(out/'orientation_pose_audit.csv' , orientation_audit_rows)
    _write_csv(out/'localization_trials.csv',rows); _write_csv(out/'localization_summary.csv',summary)
    _write_csv(out/'session_summary.csv',session_summary); _write_csv(out/'frontend_trials.csv',frontend_rows)
    _write_csv(out/'place_recognition_summary.csv',place_summary); _write_csv(out/'reference_database_stats.csv',ref_stats)
    _write_csv(out/'map_geometry_stats.csv',map_stats); _write_csv(out/'representation_gates.csv',gates); _write_csv(out/'query_scans.csv',query_rows)
    metadata=prepared.metadata(); metadata.update({
        'step':('9.6_BEVPlace++' if retriever == 'bevplace' else '9.5'),'purpose':'Same frozen Step 9.5 downstream stack, BEVPlace++ official pretrained descriptor retrieval replaces Scan Context ranking.',
        'history_sessions':names,'map_session':map_session,'query_sessions':qs,'causal_latest_prior_plan':[asdict(x) for x in plan],
        'query_count_per_session':int(query_count_per_session),'query_scans_csv':str(query_scans_csv) if query_scans_csv else None,
        'no_pose_initialization':True,'frozen_k':FROZEN_K,'frozen_geometry_profile':asdict(FROZEN_PROFILE),
        'scan_context_fallback_config':asdict(sc),'bevplace_retrieval_top_k':int(sc_top_k),'unused_legacy_sc_candidate_pool':int(sc_candidate_pool),
        'reference_keyframes':{'scan_stride':int(reference_scan_stride),'min_translation_m':float(reference_min_translation_m),
            'min_yaw_deg':float(reference_min_yaw_deg),'max_time_s':float(reference_max_time_s),'descriptor_voxel_m':float(descriptor_voxel_m)},
        'global_registration':{'backend':'KISS-Matcher isolated worker','voxel_size_m':float(kiss_voxel_m),'candidate_refinement':'small_gicp GICP',
            'scan_gicp_downsample_m':float(scan_gicp_downsample_m),'scan_gicp_max_correspondence_m':float(scan_gicp_max_corr_m)},
        'step97i':{'bst_xy_index':bool(bst_xy_index),'surfel6d_fast_tree':bool(surfel6d_fast_tree),'surfel6d_nn_workers':int(surfel6d_nn_workers),'surfel6d_leafsize':int(surfel6d_leafsize)},
        'local_refinement':{'backend':('native_surfel_xy_yaw' if local_solver=='native_surfel' else ('native_surfel_6dof' if local_solver=='surfel_6dof' else 'small_gicp GICP')),'map_gicp_downsample_m':float(map_gicp_downsample_m),
            'map_gicp_max_correspondence_m':float(map_gicp_max_corr_m),'local_map_radius_m':float(local_map_radius_m)},
        'success_thresholds':{'fine_xy_m':float(fine_success_translation_m),'fine_yaw_deg':float(fine_success_yaw_deg),
            'strict_6d_translation_m':float(strict_6d_translation_m),'strict_6d_rotation_deg':float(strict_6d_rotation_deg),
            'global_translation_m':float(global_success_translation_m),'global_rotation_deg':float(global_success_rotation_deg)},
        'dependency_versions':{
            'kiss-matcher':importlib.metadata.version('kiss-matcher'),
            'small-gicp':importlib.metadata.version('small-gicp'),
        },
        'retriever':('BEVPlace++ REIN NetVLAD global descriptor, frozen pretrained checkpoint' if retriever == 'bevplace' else 'Scan Context'),
        'descriptor_cache_dir':str(descriptor_cache_dir),
        'retriever_provenance':provider.provenance,
        'online_neural_inference_timed':False,
        'timing_note':'Step 9.6 frontend_ms includes retrieval lookup and all registration attempts, but EXCLUDES offline BEV preprocessing/network extraction; do not interpret total_ms as online end-to-end latency.',
        'step97g':{'hybrid_bev_min_inliers':int(hybrid_bev_min_inliers),'hybrid_bev_min_ratio':float(hybrid_bev_min_ratio),'hybrid_bev_max_candidates':int(hybrid_bev_max_candidates),'frontend_strategy':frontend_strategy},
        'step97f':{'bev_local_feature_dir':str(bev_local_feature_dir) if bev_local_feature_dir else None,'online_neural_inference_timed':False,'local_solver':local_solver},
        'step97b':{'local_solver':local_solver,'native_yaw_multistart':bool(native_yaw_multistart),'frontend_pose_cache':str(frontend_pose_cache) if frontend_pose_cache else None,'reported_total_ms_comparable':not bool(frontend_pose_cache)},
        'construction_once':{'session_block_build_ms':float(sum(session_block_ms)),'adaptive_core_build_ms':float(core_ms)},
        'reference_database_stats':ref_stats,'map_geometry_stats':map_stats,'representation_gates':gates,
        'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'scientific_caveat':'BEVPlace++ pretrained global descriptors replace Scan Context retrieval only. SC yaw fallback remains when KISS fails. Frozen K=4 BST geometry and original prototype codec are unchanged. Neural network extraction was done offline; measured total_ms excludes it. Author BST-ID integration remains pending.',
    })
    if retriever == 'scan_context':
        metadata.update({
            'purpose':'Scan Context retrieval -> frozen KISS/scan GICP -> latest-prior voxel/BST K=4 map GICP.',
            'scan_context_config':asdict(sc), 'scan_context_top_k':int(sc_top_k),
            'scan_context_candidate_pool':int(sc_candidate_pool),
            'timing_note':'Includes attempted registrations; not directly comparable to legacy Step 9.5 which omitted failed KISS attempts.',
            'scientific_caveat':'Frozen Scan Context global retrieval and prototype BST spatial codec; query GT is evaluation-only.'
        })
        for old_key in ('bevplace_retrieval_top_k','unused_legacy_sc_candidate_pool','scan_context_fallback_config','descriptor_cache_dir','retriever_provenance'):
            metadata.pop(old_key,None)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return rows,summary,session_summary,frontend_rows,place_summary,gates

