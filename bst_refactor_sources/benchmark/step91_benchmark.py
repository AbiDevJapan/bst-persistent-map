from __future__ import annotations

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
from benchmark.step902_benchmark import _select_query_scans, _replay_query_scans, _load_and_prepare_scan
from benchmark.step903a_benchmark import _filter_replay_csv_for_cross_session
from localization.step90 import Pose3DOF, pose_error_3dof
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID
from localization.step903b import SurfelLocalMapIndex, prepare_surfel_pyramid, scan_to_map_coarse_to_fine_surfel_3dof, CoarseToFineSurfelResult
from localization.step91 import (
    align_voxel_surfels_to_prepared, build_native_block_surfel_map,
    NativeBlockSurfelIndex, BSTDecompressedSurfelIndex,
)
from realdata.nclt import NCLTIngestConfig, _timestamp_from_scan
from realdata.nclt_packed import prepare_nclt_packed
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec


def _summary(rows):
    groups={}
    for r in rows:
        groups.setdefault((r['map_geometry'],r['scenario'],r['perturbation']),[]).append(r)
    out=[]
    for (geom,scenario,label),g in groups.items():
        def vals(k): return [float(x[k]) for x in g]
        out.append({
            'map_geometry':geom,'scenario':scenario,'perturbation':label,'trials':len(g),
            'success_rate':float(np.mean([bool(x['success']) for x in g])),
            'optimizer_convergence_rate':float(np.mean([bool(x['converged']) for x in g])),
            'initial_translation_error_m':statistics.mean(vals('initial_translation_error_m')),
            'initial_yaw_error_deg':statistics.mean(vals('initial_yaw_error_deg')),
            'median_final_translation_error_m':statistics.median(vals('final_translation_error_m')),
            'p95_final_translation_error_m':_pctl(vals('final_translation_error_m'),.95),
            'median_final_yaw_error_deg':statistics.median(vals('final_yaw_error_deg')),
            'p95_final_yaw_error_deg':_pctl(vals('final_yaw_error_deg'),.95),
            'median_retrieval_ms':statistics.median(vals('retrieval_ms')),
            'median_map_prep_ms':statistics.median(vals('map_prep_ms')),
            'median_match_ms':statistics.median(vals('match_ms')),
            'median_total_ms':statistics.median(vals('total_ms')),
            'p95_total_ms':_pctl(vals('total_ms'),.95),
            'mean_query_points':statistics.mean(vals('query_points')),
            'mean_local_map_primitives':statistics.mean(vals('local_map_primitives')),
            'mean_final_correspondences':statistics.mean(vals('correspondences')),
            'mean_normal_correspondences':statistics.mean(vals('normal_correspondences')),
            'median_final_plane_rmse_m':statistics.median(vals('final_plane_rmse_m')),
            'median_abs_plane_residual_m':statistics.median(vals('mean_abs_plane_residual_m')),
            'median_total_iterations':statistics.median(vals('iterations')),
        })
    so={'cross_session':0,'same_session_control':1}
    go={'voxel_surfel_0.5m':0,'bst_decompressed_voxel_surfel':1,'bst_native_block_surfel':2}
    return sorted(out,key=lambda r:(so.get(r['scenario'],9),r['initial_translation_error_m'],go.get(r['map_geometry'],9)))


def _run_trial(*,map_geometry,scenario,scan_index,scan_path,body,gt,gap,original_scan_index,init,perturbation_label,xy_mag,yaw_mag,dx,dy,dyaw_deg,map_index,local_map_radius_m,pyramid,base_icp_config,success_translation_m,success_yaw_deg):
    _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
    gt_pose=Pose3DOF(gx,gy,gyaw)
    init_te,init_ye=pose_error_3dof(init,gt_pose)
    total0=time.perf_counter_ns()
    tr0=time.perf_counter_ns(); ids=map_index.query_radius_ids(init.x,init.y,local_map_radius_m); retrieval_ms=(time.perf_counter_ns()-tr0)/1e6
    tp0=time.perf_counter_ns()
    if len(ids):
        if hasattr(map_index, 'prepare_pyramid'):
            prepared=map_index.prepare_pyramid(ids,pyramid,base_icp_config)
        else:
            prepared=prepare_surfel_pyramid(map_index.map,ids,pyramid,base_icp_config)
    else:
        prepared=[]
    map_prep_ms=(time.perf_counter_ns()-tp0)/1e6
    if not prepared:
        result=CoarseToFineSurfelResult(init,False,0,0,0,float('inf'),float('inf'),0.0,'empty_local_map',[])
    else:
        result=scan_to_map_coarse_to_fine_surfel_3dof(body,prepared,init,gz,roll,pitch,base_icp_config)
    total_ms=(time.perf_counter_ns()-total0)/1e6
    fte,fye=pose_error_3dof(result.pose,gt_pose); sm,sy=pose_error_3dof(result.pose,init)
    row={
        'map_geometry':map_geometry,'scenario':scenario,'scan_index':int(scan_index),'original_scan_index':int(original_scan_index),
        'scan_file':scan_path.name,'timestamp_us':int(_timestamp_from_scan(scan_path)),'pose_gap_us':int(gap),
        'matcher':'coarse_to_fine_surfel_point_to_plane','perturbation':perturbation_label,'perturb_xy_m':float(xy_mag),'perturb_yaw_deg':float(yaw_mag),
        'actual_dx_m':float(dx),'actual_dy_m':float(dy),'actual_dyaw_deg':float(dyaw_deg),
        'initial_translation_error_m':init_te,'initial_yaw_error_deg':init_ye,'final_translation_error_m':fte,'final_yaw_error_deg':fye,
        'pose_shift_from_initial_m':sm,'yaw_shift_from_initial_deg':sy,'success':bool(fte<=success_translation_m and fye<=success_yaw_deg),
        'converged':bool(result.converged),'iterations':int(result.iterations),'correspondences':int(result.correspondences),
        'normal_correspondences':int(result.normal_correspondences),'final_rmse_m':float(result.rmse_m),'final_plane_rmse_m':float(result.rmse_m),
        'mean_abs_plane_residual_m':float(result.mean_abs_plane_residual_m),'failure_reason':result.failure_reason,
        'query_points':int(len(body)),'local_map_primitives':int(len(ids)),'retrieval_ms':retrieval_ms,'map_prep_ms':map_prep_ms,
        'match_ms':float(result.match_ms),'total_ms':total_ms,'initial_x':init.x,'initial_y':init.y,'initial_yaw':init.yaw,
        'estimated_x':result.pose.x,'estimated_y':result.pose.y,'estimated_yaw':result.pose.yaw,'gt_x':gx,'gt_y':gy,'gt_yaw':gyaw,
    }
    stats=getattr(map_index,'last_stats',{}) or {}
    for k,v in stats.items(): row[f'retrieval_{k}']=v
    for name,lev in result.level_results:
        p=f'level_{name}_'; row[p+'converged']=bool(lev.converged); row[p+'iterations']=int(lev.iterations); row[p+'correspondences']=int(lev.correspondences)
        row[p+'normal_correspondences']=int(lev.normal_correspondences); row[p+'plane_rmse_m']=float(lev.rmse_m); row[p+'mean_abs_plane_residual_m']=float(lev.mean_abs_plane_residual_m)
        row[p+'match_ms']=float(lev.match_ms); row[p+'failure_reason']=lev.failure_reason
    return row


def benchmark_step91(*,data_root,output_dir,history_sessions,map_session='2012-01-15',query_session='2012-02-02',voxel_size_m=0.5,scan_stride=10,min_range_m=2.0,max_range_m=80.0,min_points_per_voxel=1,max_pose_gap_us=100000,cache_dir='.cache/step8_1',query_count=10,same_session_query_count=10,query_scans_csv=None,perturbations='0.5:1',perturb_seed=42090,localization_max_range_m=30.0,scan_downsample_m=0.75,max_scan_points=8000,local_map_radius_m=40.0,bucket_width_m=8.0,success_translation_m=0.5,success_yaw_deg=1.0,icp_config=None,include_same_session_zero=True,include_decompressed_control=True,progress=True):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    names=list(history_sessions)
    if map_session not in names: raise ValueError('map_session must appear in history_sessions')
    if query_session==map_session: raise ValueError('cross-session query_session must differ from map_session')
    if abs(float(voxel_size_m)-0.5)>1e-9: raise ValueError('Step 9.1 keeps the frozen Step-8 0.5m canonical lattice')
    map_t=names.index(map_session)
    cfg=NCLTIngestConfig(voxel_size_m=float(voxel_size_m),scan_stride=int(scan_stride),min_range_m=float(min_range_m),max_range_m=float(max_range_m),min_points_per_voxel=int(min_points_per_voxel),max_pose_gap_us=int(max_pose_gap_us))
    base_cfg=icp_config or PointToPlaneICP3DOFConfig()
    levels=parse_perturbations(perturbations); pyramid=DEFAULT_PYRAMID

    replay=None
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

    if progress: print('[9.1] loading five-session packed history from existing Step-8 cache...',flush=True)
    prepared=prepare_nclt_packed(data_root,names,cfg,cache_dir,progress_every=250)
    if progress: print('[9.1] building exact Step-8.4 session-local tilings...',flush=True)
    session_blocks,session_block_ms=build_session_spatial_blocks(prepared,len(names),progress=progress)
    if progress: print('[9.1] finalizing Step-8.5 time-aware BST map...',flush=True)
    t0=time.perf_counter_ns(); core=TimeLocalAdaptiveCore(prepared,session_blocks,progress=progress); core_ms=(time.perf_counter_ns()-t0)/1e6
    bst=BSTTimeLocalTimeIndexedMap(core,PrototypeBSTSpatialCodec())

    if progress: print('[9.1] rebuilding 0.5m map-session surfel sufficient statistics in memory...',flush=True)
    raw_surf=build_nclt_surfel_map(data_root,map_session,cfg,cache_dir,normal_min_points=base_cfg.normal_min_neighbors,normal_max_curvature=base_cfg.normal_max_curvature,normal_min_horizontal_component=base_cfg.normal_min_horizontal_component)
    aligned=align_voxel_surfels_to_prepared(raw_surf,prepared,map_t)
    t0=time.perf_counter_ns(); voxel_index=SurfelLocalMapIndex(aligned,bucket_width_m); voxel_index_ms=(time.perf_counter_ns()-t0)/1e6
    if progress: print('[9.1] propagating surfel statistics through the exact native block merges...',flush=True)
    block_surf=build_native_block_surfel_map(aligned,prepared,session_blocks,core,bst,map_t,base_cfg.normal_min_neighbors,base_cfg.normal_max_curvature,base_cfg.normal_min_horizontal_component,progress=progress)
    block_index=NativeBlockSurfelIndex(block_surf)
    decomp_index=BSTDecompressedSurfelIndex(bst,aligned,map_t)

    # Strict compatibility gate: direct voxel-surfel retrieval and BST-decompressed
    # retrieval must select exactly the same voxel-surfel rows for all initial poses.
    if include_decompressed_control:
        if progress: print('[9.1] verifying BST-decompressed surfel retrieval exactly matches direct voxel-surfel retrieval...',flush=True)
        rng=random.Random(int(perturb_seed))
        for si,_ in enumerate(cross):
            sp,gt,gap,oi,body=loaded[('cross_session',si)]
            _,gx,gy,_,_,_,gyaw=[float(v) for v in gt]
            inits=[Pose3DOF(gx,gy,gyaw)]
            for _,xy,yaw in levels:
                dx,dy,dyaw=_trial_perturbation(xy,yaw,rng); inits.append(Pose3DOF(gx+dx,gy+dy,gyaw+math.radians(dyaw)))
            for init in inits:
                a=np.sort(voxel_index.query_radius_ids(init.x,init.y,local_map_radius_m))
                b=np.sort(decomp_index.query_radius_ids(init.x,init.y,local_map_radius_m))
                if not np.array_equal(a,b): raise AssertionError(('BST decompressed surfel compatibility',si,len(a),len(b)))

    map_rows=[]
    voxel_payload=int(sum(a.nbytes for a in (aligned.linear_codes,aligned.counts,aligned.centroids,aligned.cov6,aligned.normals,aligned.curvature,aligned.normal_valid)))
    voxel_min=int(sum(a.nbytes for a in (aligned.linear_codes,aligned.counts,aligned.centroids,aligned.normals,aligned.curvature,aligned.normal_valid)))
    map_rows.append({'map_geometry':'voxel_surfel_0.5m','map_elements':len(aligned.linear_codes),'valid_normal_elements':int(np.count_nonzero(aligned.normal_valid)),'valid_normal_fraction':float(np.mean(aligned.normal_valid.astype(bool))),'geometry_payload_mib':voxel_payload/(1024**2),'minimal_geometry_payload_mib':voxel_min/(1024**2),'query_index_mib':voxel_index.nbytes()/(1024**2),'bst_representation_mib':0.0,'geometry_build_ms':float(raw_surf.stats_build_ms),'index_build_ms':float(voxel_index_ms),'voxel_equivalent':int(len(aligned.linear_codes)),'mean_voxels_per_primitive':1.0})
    if include_decompressed_control:
        map_rows.append({'map_geometry':'bst_decompressed_voxel_surfel','map_elements':len(aligned.linear_codes),'valid_normal_elements':int(np.count_nonzero(aligned.normal_valid)),'valid_normal_fraction':float(np.mean(aligned.normal_valid.astype(bool))),'geometry_payload_mib':voxel_payload/(1024**2),'minimal_geometry_payload_mib':voxel_min/(1024**2),'query_index_mib':0.0,'bst_representation_mib':bst.compact_bytes()/(1024**2),'geometry_build_ms':float(raw_surf.stats_build_ms),'index_build_ms':float(core_ms),'voxel_equivalent':int(len(aligned.linear_codes)),'mean_voxels_per_primitive':1.0})
    map_rows.append({'map_geometry':'bst_native_block_surfel','map_elements':len(block_surf.blocks),'valid_normal_elements':int(np.count_nonzero(block_surf.normal_valid)),'valid_normal_fraction':float(np.mean(block_surf.normal_valid.astype(bool))) if len(block_surf.normal_valid) else 0.0,'geometry_payload_mib':block_surf.payload_nbytes()/(1024**2),'minimal_geometry_payload_mib':block_surf.minimal_localization_payload_nbytes()/(1024**2),'query_index_mib':0.0,'bst_representation_mib':bst.compact_bytes()/(1024**2),'geometry_build_ms':float(block_surf.build_ms),'index_build_ms':float(core_ms),'session_block_build_ms':float(sum(session_block_ms)),'voxel_equivalent':int(np.prod(block_surf.blocks[:,3:6].astype(np.int64),axis=1).sum()),'mean_voxels_per_primitive':float(np.prod(block_surf.blocks[:,3:6].astype(np.int64),axis=1).mean()) if len(block_surf.blocks) else 0.0})

    geometries=[('voxel_surfel_0.5m',voxel_index)]
    if include_decompressed_control: geometries.append(('bst_decompressed_voxel_surfel',decomp_index))
    geometries.append(('bst_native_block_surfel',block_index))
    rows=[]
    for geom,index in geometries:
        if progress: print(f'[9.1] localizing with {geom}...',flush=True)
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

    summary=_summary(rows)
    _write_csv(out/'localization_trials.csv',rows); _write_csv(out/'localization_summary.csv',summary); _write_csv(out/'query_scans.csv',scan_rows); _write_csv(out/'map_geometry_stats.csv',map_rows)
    tmp=out/'_cross_replay.csv'
    if tmp.exists(): tmp.unlink()
    metadata=prepared.metadata(); metadata.update({
        'step':'9.1','purpose':'Native BST block-surface localization feasibility against the validated 0.5m voxel-surfel baseline.',
        'history_sessions':names,'map_session':map_session,'map_session_index':map_t,'query_session':query_session,
        'query_count':len(cross),'same_session_query_count':len(same),'query_scans_replay_csv':str(query_scans_csv) if query_scans_csv else None,
        'perturbations':[{'label':l,'xy_m':x,'yaw_deg':y} for l,x,y in levels],
        'localization_max_range_m':localization_max_range_m,'scan_downsample_m':scan_downsample_m,'max_scan_points':max_scan_points,'local_map_radius_m':local_map_radius_m,
        'success_thresholds':{'translation_m':success_translation_m,'yaw_deg':success_yaw_deg},'base_icp_config':asdict(base_cfg),'pyramid':[asdict(x) for x in pyramid],
        'methods':[x[0] for x in geometries],
        'native_block_geometry':{'construction':'surfel sufficient statistics propagated through the exact Step-8.4 map-session dyadic merges','payload':'block bounds remain in BST map; localization payload stores count, centroid, covariance, derived normal/curvature/validity','no_hidden_decompression':True,'map_query':'Step-8.5 counted (bucket,session) time-aware index','matcher':'same coarse-to-fine point-to-plane matcher as voxel-surfel baseline'},
        'compatibility_control':'bst_decompressed_voxel_surfel must retrieve exactly the same voxel-surfel rows as direct voxel_surfel_0.5m for every benchmark initialization.',
        'map_geometry_stats':map_rows,'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'scientific_caveat':'Step 9.1 tests local registration within the established ICP basin; it does not claim global relocalization. PrototypeBSTSpatialCodec remains a local research codec, not verified upstream BST-ID.',
    })
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return rows,summary,map_rows


__all__=['benchmark_step91']
