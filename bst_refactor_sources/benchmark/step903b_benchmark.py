from __future__ import annotations

import csv
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
    _run_trial as _run_occupancy_trial,
)
from benchmark.step903a_benchmark import prepare_map_only, _filter_replay_csv_for_cross_session
from localization.step90 import Pose3DOF, VoxelLocalMapIndex, pose_error_3dof
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID, prepare_local_map_pyramid
from localization.step903b import SurfelLocalMapIndex, prepare_surfel_pyramid, scan_to_map_coarse_to_fine_surfel_3dof
from realdata.nclt import NCLTIngestConfig, _timestamp_from_scan
from realdata.nclt_surfel import build_nclt_surfel_map


def _summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str,str,str], list[dict]] = {}
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
            'median_pose_shift_from_initial_m':statistics.median(vals('pose_shift_from_initial_m')),
            'median_yaw_shift_from_initial_deg':statistics.median(vals('yaw_shift_from_initial_deg')),
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
    go={'occupancy_center_0.5m':0,'voxel_surfel_0.5m':1}
    return sorted(out,key=lambda r:(so.get(r['scenario'],9),r['initial_translation_error_m'],go.get(r['map_geometry'],9)))


def _run_surfel_trial(*,scenario,scan_index,scan_path,body,gt,gap,original_scan_index,init,perturbation_label,xy_mag,yaw_mag,dx,dy,dyaw_deg,surfel_index,local_map_radius_m,pyramid,base_icp_config,success_translation_m,success_yaw_deg):
    _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
    gt_pose=Pose3DOF(gx,gy,gyaw); init_te,init_ye=pose_error_3dof(init,gt_pose)
    total0=time.perf_counter_ns(); tr0=time.perf_counter_ns()
    ids=surfel_index.query_radius_ids(init.x,init.y,local_map_radius_m)
    retrieval_ms=(time.perf_counter_ns()-tr0)/1e6
    tp0=time.perf_counter_ns()
    prepared=prepare_surfel_pyramid(surfel_index.map,ids,pyramid,base_icp_config) if len(ids) else []
    map_prep_ms=(time.perf_counter_ns()-tp0)/1e6
    if not prepared:
        from localization.step903b import CoarseToFineSurfelResult
        result=CoarseToFineSurfelResult(init,False,0,0,0,float('inf'),float('inf'),0.0,'empty_local_map',[])
    else:
        result=scan_to_map_coarse_to_fine_surfel_3dof(body,prepared,init,gz,roll,pitch,base_icp_config)
    total_ms=(time.perf_counter_ns()-total0)/1e6
    fte,fye=pose_error_3dof(result.pose,gt_pose); sm,sy=pose_error_3dof(result.pose,init)
    row={
        'map_geometry':'voxel_surfel_0.5m','scenario':scenario,'scan_index':int(scan_index),'original_scan_index':int(original_scan_index),
        'scan_file':scan_path.name,'timestamp_us':int(_timestamp_from_scan(scan_path)),'pose_gap_us':int(gap),'matcher':'coarse_to_fine_surfel_point_to_plane',
        'perturbation':perturbation_label,'perturb_xy_m':float(xy_mag),'perturb_yaw_deg':float(yaw_mag),'actual_dx_m':float(dx),'actual_dy_m':float(dy),'actual_dyaw_deg':float(dyaw_deg),
        'initial_translation_error_m':init_te,'initial_yaw_error_deg':init_ye,'final_translation_error_m':fte,'final_yaw_error_deg':fye,
        'pose_shift_from_initial_m':sm,'yaw_shift_from_initial_deg':sy,
        'success':bool(fte<=success_translation_m and fye<=success_yaw_deg),'converged':bool(result.converged),'iterations':int(result.iterations),
        'correspondences':int(result.correspondences),'normal_correspondences':int(result.normal_correspondences),'final_rmse_m':float(result.rmse_m),
        'final_plane_rmse_m':float(result.rmse_m),'mean_abs_plane_residual_m':float(result.mean_abs_plane_residual_m),'failure_reason':result.failure_reason,
        'query_points':int(len(body)),'local_map_voxels':int(len(ids)),'local_map_primitives':int(len(ids)),
        'retrieval_ms':retrieval_ms,'map_prep_ms':map_prep_ms,'match_ms':float(result.match_ms),'total_ms':total_ms,
        'initial_x':init.x,'initial_y':init.y,'initial_yaw':init.yaw,'estimated_x':result.pose.x,'estimated_y':result.pose.y,'estimated_yaw':result.pose.yaw,
        'gt_x':gx,'gt_y':gy,'gt_yaw':gyaw,
    }
    for name,lev in result.level_results:
        p=f'level_{name}_'; row[p+'converged']=bool(lev.converged); row[p+'iterations']=int(lev.iterations); row[p+'correspondences']=int(lev.correspondences)
        row[p+'normal_correspondences']=int(lev.normal_correspondences); row[p+'plane_rmse_m']=float(lev.rmse_m); row[p+'mean_abs_plane_residual_m']=float(lev.mean_abs_plane_residual_m)
        row[p+'match_ms']=float(lev.match_ms); row[p+'failure_reason']=lev.failure_reason
    return row


def benchmark_step903b(*,data_root:str|Path,output_dir:str|Path,map_session:str,query_session:str,voxel_size_m:float=0.5,scan_stride:int=10,min_range_m:float=2.0,max_range_m:float=80.0,min_points_per_voxel:int=1,max_pose_gap_us:int=100000,cache_dir:str|Path|None='.cache/step8_1',query_count:int=10,same_session_query_count:int=10,query_scans_csv:str|Path|None=None,perturbations:str='0.5:1,1:3',perturb_seed:int=42090,localization_max_range_m:float=30.0,scan_downsample_m:float=0.75,max_scan_points:int=8000,local_map_radius_m:float=40.0,bucket_width_m:float=8.0,success_translation_m:float=0.5,success_yaw_deg:float=1.0,icp_config:PointToPlaneICP3DOFConfig|None=None,include_same_session_zero:bool=True,include_occupancy_baseline:bool=True):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    if abs(float(voxel_size_m)-0.5)>1e-9:
        raise ValueError('Step 9.0.3B intentionally fixes the indexing lattice at 0.5 m')
    if map_session==query_session: raise ValueError('Use distinct sessions for cross-session benchmark')
    cfg=NCLTIngestConfig(voxel_size_m=float(voxel_size_m),scan_stride=int(scan_stride),min_range_m=float(min_range_m),max_range_m=float(max_range_m),min_points_per_voxel=int(min_points_per_voxel),max_pose_gap_us=int(max_pose_gap_us))
    base_cfg=icp_config or PointToPlaneICP3DOFConfig(); levels=parse_perturbations(perturbations); pyramid=DEFAULT_PYRAMID

    replay=None
    if query_scans_csv is not None:
        replay=_filter_replay_csv_for_cross_session(query_scans_csv,out/'_cross_replay.csv')
        cross=_replay_query_scans(data_root,query_session,replay,query_count,max_pose_gap_us)
    else: cross=_select_query_scans(data_root,query_session,query_count,max_pose_gap_us)
    same=_select_query_scans(data_root,map_session,same_session_query_count,max_pose_gap_us,exclude_ingest_stride=scan_stride) if include_same_session_zero else []

    prep_scan=SimpleNamespace(config=cfg)
    scan_rows=[]; loaded={}
    for scenario,selected in [('cross_session',cross),('same_session_control',same)]:
        for si,(sp,gt,gap,oi) in enumerate(selected):
            body,raw=_load_and_prepare_scan(sp,prep_scan,localization_max_range_m,scan_downsample_m,max_scan_points)
            loaded[(scenario,si)]=(sp,gt,gap,oi,body)
            _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
            scan_rows.append({'scenario':scenario,'scan_index':si,'original_scan_index':int(oi),'scan_file':sp.name,'timestamp_us':int(_timestamp_from_scan(sp)),'pose_gap_us':int(gap),'points_after_range':int(raw),'points_used':int(len(body)),'gt_x':gx,'gt_y':gy,'gt_z':gz,'gt_roll':roll,'gt_pitch':pitch,'gt_yaw':gyaw,'directly_ingested_scan':bool((oi%scan_stride)==0)})

    print('[9.0.3B] loading 0.5m occupancy map...',flush=True)
    occ=prepare_map_only(data_root,map_session,cfg,cache_dir,'reuse-only')
    bucket_vox=max(1,int(round(bucket_width_m/voxel_size_m)))
    t=time.perf_counter_ns(); occ_index=VoxelLocalMapIndex(occ.map_codes,occ.spec,occ.origin_local_m,bucket_vox); occ_index_ms=(time.perf_counter_ns()-t)/1e6
    occ_index_bytes=int(sum(a.nbytes for a in (occ_index.codes,occ_index._keys_per_code,occ_index._codes_by_bucket,occ_index.bucket_keys,occ_index.bucket_starts,occ_index.bucket_counts)))

    print('[9.0.3B] building 0.5m sub-voxel surfel payload from raw map-session returns...',flush=True)
    surf=build_nclt_surfel_map(data_root,map_session,cfg,cache_dir,normal_min_points=base_cfg.normal_min_neighbors,normal_max_curvature=base_cfg.normal_max_curvature,normal_min_horizontal_component=base_cfg.normal_min_horizontal_component)
    t=time.perf_counter_ns(); surf_index=SurfelLocalMapIndex(surf,bucket_width_m); surf_index_ms=(time.perf_counter_ns()-t)/1e6

    map_rows=[{
        'map_geometry':'occupancy_center_0.5m','voxel_size_m':voxel_size_m,'map_elements':int(len(occ.map_codes)),'valid_normal_elements':'lazy-PCA',
        'payload_mib':float(occ.map_codes.nbytes/(1024**2)),'minimal_localization_payload_mib':float(occ.map_codes.nbytes/(1024**2)),
        'index_arrays_mib':float(occ_index_bytes/(1024**2)),'occupancy_load_ms':float(occ.ingest_ms),'geometry_build_ms':0.0,'index_build_ms':float(occ_index_ms),'occupancy_cache_hit':bool(occ.cache_hit),
    },{
        'map_geometry':'voxel_surfel_0.5m','voxel_size_m':voxel_size_m,'map_elements':int(len(surf.linear_codes)),'valid_normal_elements':int(np.count_nonzero(surf.normal_valid)),
        'valid_normal_fraction':float(np.mean(surf.normal_valid.astype(bool))),'payload_mib':float(surf.payload_nbytes()/(1024**2)),'minimal_localization_payload_mib':float(surf.minimal_localization_payload_nbytes()/(1024**2)),
        'index_arrays_mib':float(surf_index.nbytes()/(1024**2)),'occupancy_load_ms':float(surf.occupancy_load_ms),'geometry_build_ms':float(surf.stats_build_ms),'index_build_ms':float(surf_index_ms),'occupancy_cache_hit':bool(surf.occupancy_cache_hit),
    }]

    rows=[]
    def run_geometry(name:str):
        def run_scenario(scenario,selected,perturb,seed):
            rng=random.Random(int(seed))
            for si,_ in enumerate(selected):
                sp,gt,gap,oi,body=loaded[(scenario,si)]
                _,gx,gy,gz,roll,pitch,gyaw=[float(v) for v in gt]
                specs=[('zero_control',0.,0.,0.,0.,0.)]
                if perturb:
                    for label,xy,yaw in levels:
                        dx,dy,dyaw=_trial_perturbation(xy,yaw,rng); specs.append((label,xy,yaw,dx,dy,dyaw))
                for label,xy,yaw,dx,dy,dyaw in specs:
                    init=Pose3DOF(gx+dx,gy+dy,gyaw+math.radians(dyaw))
                    if name=='occupancy_center_0.5m':
                        r=_run_occupancy_trial(scenario=scenario,scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,original_scan_index=oi,init=init,perturbation_label=label,xy_mag=xy,yaw_mag=yaw,dx=dx,dy=dy,dyaw_deg=dyaw,map_index=occ_index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg)
                        r['map_geometry']=name; r['local_map_primitives']=r['local_map_voxels']
                    else:
                        r=_run_surfel_trial(scenario=scenario,scan_index=si,scan_path=sp,body=body,gt=gt,gap=gap,original_scan_index=oi,init=init,perturbation_label=label,xy_mag=xy,yaw_mag=yaw,dx=dx,dy=dy,dyaw_deg=dyaw,surfel_index=surf_index,local_map_radius_m=local_map_radius_m,pyramid=pyramid,base_icp_config=base_cfg,success_translation_m=success_translation_m,success_yaw_deg=success_yaw_deg)
                    rows.append(r)
        run_scenario('cross_session',cross,True,perturb_seed)
        if include_same_session_zero: run_scenario('same_session_control',same,False,perturb_seed+902)
    if include_occupancy_baseline: run_geometry('occupancy_center_0.5m')
    run_geometry('voxel_surfel_0.5m')

    summary=_summary(rows)
    _write_csv(out/'localization_trials.csv',rows); _write_csv(out/'localization_summary.csv',summary); _write_csv(out/'query_scans.csv',scan_rows); _write_csv(out/'surfel_map_stats.csv',map_rows)
    tmp=out/'_cross_replay.csv'
    if tmp.exists(): tmp.unlink()
    metadata={
        'step':'9.0.3B','purpose':'0.5m voxel-surfel localization: keep the 0.5m index lattice but replace voxel-center targets with sub-voxel point statistics.',
        'map_session':map_session,'query_session':query_session,'voxel_size_m':voxel_size_m,'query_count':len(cross),'same_session_query_count':len(same),
        'perturbations':[{'label':l,'xy_m':x,'yaw_deg':y} for l,x,y in levels],'cross_session_always_includes_zero_control':True,'same_session_zero_control':bool(include_same_session_zero),
        'query_scans_replay_csv':str(query_scans_csv) if query_scans_csv is not None else None,'scan_stride':scan_stride,'localization_max_range_m':localization_max_range_m,'scan_downsample_m':scan_downsample_m,
        'max_scan_points':max_scan_points,'local_map_radius_m':local_map_radius_m,'bucket_width_m':bucket_width_m,'success_thresholds':{'translation_m':success_translation_m,'yaw_deg':success_yaw_deg},
        'base_icp_config':asdict(base_cfg),'pyramid':[asdict(x) for x in pyramid],
        'surfel_geometry':{'base_cell_m':0.5,'centroid':'mean of actual transformed LiDAR returns within each occupied voxel','covariance':'within-voxel 3x3 covariance from raw returns','normal':'smallest-eigenvalue eigenvector of covariance','coarse_levels':'weighted statistical merge of base surfels; no fallback to voxel centers','surfel_cache':'none; built in memory to avoid additional disk use'},
        'map_stats':map_rows,'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'scientific_caveat':'Step 9.0.3B changes only the localization geometry payload. Step-8 map representation/storage results remain on the frozen 0.5m occupancy lattice. Native BST blocks are not consumed yet.'
    }
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return rows,summary,map_rows

__all__=['benchmark_step903b']
