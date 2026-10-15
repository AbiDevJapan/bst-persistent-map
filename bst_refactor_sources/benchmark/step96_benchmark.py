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

from benchmark.step901_benchmark import _cap_points, _write_csv, _pctl
from benchmark.step913_benchmark import FROZEN_PROFILE, FROZEN_K, _runtime_voxel_geometry_bytes, _select_query_scans_excluding
from benchmark.step93_benchmark import _union_excluded
from localization.step902 import voxel_downsample_first
from localization.step903b import SurfelLocalMapIndex
from localization.step91 import align_voxel_surfels_to_prepared
from localization.step912 import build_native_surfel_hierarchy, build_hierarchical_lod_view
from localization.step913 import materialize_lod_view, MaterializedLODBSTIndex
from localization.step94 import causal_latest_prior_plan
from localization.step96_bevplace import BEVPlaceDescriptorCache
from localization.step95 import (
    ScanContextConfig, ScanContextDatabase, IsolatedKISSMatcherBackend, SmallGICPBackend,
    make_scan_context, scan_context_align, pose_row_to_matrix, pose_errors_6d, scan_context_pose_guess,
    compose_world_query, require_frontend_dependencies,
)
from realdata.nclt import (
    NCLTIngestConfig, discover_nclt_session, load_ground_truth, _list_scan_files,
    _timestamp_from_scan, read_velodyne_sync_bin, _filter_body_points,
)
from realdata.nclt_packed import prepare_nclt_packed, _nearest_pose_rows
from realdata.nclt_surfel import build_nclt_surfel_map
from representations.step83 import TimeLocalAdaptiveCore, build_session_spatial_blocks
from bst_id import BSTTimeLocalTimeIndexedMap, PrototypeBSTSpatialCodec


VOXEL_METHOD = 'bevplace_kiss_gicp_latest_prior_voxel'
BST_METHOD = 'bevplace_kiss_gicp_latest_prior_bst_k4'


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


def _summary(rows):
    out=[]
    for method in sorted(set(r['map_geometry'] for r in rows)):
        g=[r for r in rows if r['map_geometry']==method]
        def vals(k): return [float(x[k]) for x in g]
        out.append({
            'map_geometry':method,'trials':len(g),
            'fine_success_rate':float(np.mean([bool(x['success']) for x in g])),
            'global_2m5deg_success_rate':float(np.mean([bool(x['global_2m5deg_success']) for x in g])),
            'strict_6d_success_rate':float(np.mean([bool(x['strict_6d_success']) for x in g])),
            'median_final_translation_xy_m':statistics.median(vals('final_translation_xy_m')),
            'median_final_translation_3d_m':statistics.median(vals('final_translation_3d_m')),
            'median_final_yaw_error_deg':statistics.median(vals('final_yaw_error_deg')),
            'median_final_rotation_error_deg':statistics.median(vals('final_rotation_error_deg')),
            'p95_final_translation_3d_m':_pctl(vals('final_translation_3d_m'),.95),
            'p95_final_rotation_error_deg':_pctl(vals('final_rotation_error_deg'),.95),
            'median_frontend_ms':statistics.median(vals('frontend_ms')),
            'median_map_retrieval_ms':statistics.median(vals('map_retrieval_ms')),
            'median_local_gicp_ms':statistics.median(vals('local_gicp_ms')),
            'median_total_ms':statistics.median(vals('total_ms')),
            'p95_total_ms':_pctl(vals('total_ms'),.95),
            'mean_local_map_primitives':statistics.mean(vals('local_map_primitives')),
        })
    return out


def _session_summary(rows):
    out=[]
    sessions=sorted(set(r['query_session'] for r in rows))
    methods=sorted(set(r['map_geometry'] for r in rows))
    for s in sessions:
        for m in methods:
            g=[r for r in rows if r['query_session']==s and r['map_geometry']==m]
            if not g: continue
            out.append({
                'query_session':s,'historical_map_session':g[0]['historical_map_session'],'map_geometry':m,'trials':len(g),
                'fine_success_rate':float(np.mean([bool(x['success']) for x in g])),
                'global_2m5deg_success_rate':float(np.mean([bool(x['global_2m5deg_success']) for x in g])),
                'median_translation_xy_m':statistics.median(float(x['final_translation_xy_m']) for x in g),
                'median_yaw_error_deg':statistics.median(float(x['final_yaw_error_deg']) for x in g),
                'median_rotation_error_deg':statistics.median(float(x['final_rotation_error_deg']) for x in g),
                'median_total_ms':statistics.median(float(x['total_ms']) for x in g),
                'mean_local_map_primitives':statistics.mean(float(x['local_map_primitives']) for x in g),
            })
    return out


def _place_summary(frontend_rows):
    if not frontend_rows: return []
    out=[]
    for s in sorted(set(r['query_session'] for r in frontend_rows)) + ['ALL']:
        g=frontend_rows if s=='ALL' else [r for r in frontend_rows if r['query_session']==s]
        if not g: continue
        out.append({
            'query_session':s,'trials':len(g),
            'bevplace_top1_recall_5m':float(np.mean([float(r['bevplace_top1_position_error_m'])<=5 for r in g])),
            'bevplace_top1_recall_10m':float(np.mean([float(r['bevplace_top1_position_error_m'])<=10 for r in g])),
            'bevplace_top1_recall_20m':float(np.mean([float(r['bevplace_top1_position_error_m'])<=20 for r in g])),
            'bevplace_topk_recall_5m':float(np.mean([float(r['bevplace_topk_best_position_error_m'])<=5 for r in g])),
            'bevplace_topk_recall_10m':float(np.mean([float(r['bevplace_topk_best_position_error_m'])<=10 for r in g])),
            'bevplace_topk_recall_20m':float(np.mean([float(r['bevplace_topk_best_position_error_m'])<=20 for r in g])),
            'kiss_candidate_fine_rate':float(np.mean([float(r['frontend_translation_xy_m'])<=0.5 and float(r['frontend_yaw_error_deg'])<=1.0 for r in g])),
            'kiss_candidate_global_2m5deg_rate':float(np.mean([float(r['frontend_translation_3d_m'])<=2.0 and float(r['frontend_rotation_error_deg'])<=5.0 for r in g])),
            'median_frontend_translation_3d_m':statistics.median(float(r['frontend_translation_3d_m']) for r in g),
            'median_frontend_rotation_error_deg':statistics.median(float(r['frontend_rotation_error_deg']) for r in g),
            'median_frontend_ms':statistics.median(float(r['frontend_ms']) for r in g),
        })
    return out


def benchmark_step96(
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
    include_voxel_baseline=True, progress=True, descriptor_cache_dir=None,
):
    if descriptor_cache_dir is None:
        raise ValueError("Step 9.6 requires --descriptor-cache-dir with official model features")
    descriptor_cache=BEVPlaceDescriptorCache(descriptor_cache_dir)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    names=list(history_sessions); qs=list(query_sessions)
    if abs(float(voxel_size_m)-0.5)>1e-9:
        raise ValueError('Step 9.5 keeps the frozen 0.5m canonical lattice')
    if map_session not in names: raise ValueError('map_session missing from history_sessions')
    plan=causal_latest_prior_plan(names,map_session,qs)
    deps=require_frontend_dependencies(True,True)  # fail before expensive map construction
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

    kiss=IsolatedKISSMatcherBackend(kiss_voxel_m)
    scan_gicp=SmallGICPBackend(downsampling_resolution_m=scan_gicp_downsample_m,
        max_correspondence_distance_m=scan_gicp_max_corr_m,num_threads=gicp_threads,module=deps['small_gicp'])
    map_gicp=SmallGICPBackend(downsampling_resolution_m=map_gicp_downsample_m,
        max_correspondence_distance_m=map_gicp_max_corr_m,num_threads=gicp_threads,module=deps['small_gicp'])

    rows=[]; frontend_rows=[]; ref_stats=[]; map_stats=[]
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
        # Check EXACT historical scans and indices. BEV descriptors only change
        # candidate retrieval; all map history and representation remain frozen.
        pair_cache={}
        for assoc_p in assoc:
            db,pair=descriptor_cache.load_pair(prior_session,assoc_p.query_session,
                ref['scan_indices'],[ref['scans'][int(i)].name for i in ref['scan_indices']])
            pair_cache[assoc_p.query_session]=(db,pair)
        ref_stats.append({'historical_map_session':prior_session,'keyframes':len(ref['scan_indices']),
            'cache_hit':ref['cache_hit'],'build_ms':ref['build_ms'],'descriptor_bytes':int(ref['descriptors'].nbytes+ref['ring_keys'].nbytes),'bevplace_descriptor_bytes':int(db.descriptors.nbytes),
            'cache_path':ref['cache_path']})

        if progress: print(f'[9.5] building latest-prior voxel + compact BST K=4 map for {prior_session}...',flush=True)
        raw=build_nclt_surfel_map(data_root,prior_session,cfg,cache_dir,normal_min_points=6,
            normal_max_curvature=0.20,normal_min_horizontal_component=0.20)
        aligned=align_voxel_surfels_to_prepared(raw,prepared,prior_i)
        voxel_index=SurfelLocalMapIndex(aligned,bucket_width_m)
        hierarchy=build_native_surfel_hierarchy(aligned,prepared,session_blocks,core,bst,prior_i,6,0.20,0.20,progress=progress)
        view=build_hierarchical_lod_view(hierarchy,FROZEN_PROFILE,FROZEN_K); compact=materialize_lod_view(view)
        bst_index=MaterializedLODBSTIndex(compact,bst,prior_i)
        map_stats.append({'historical_map_session':prior_session,'voxel_elements':len(aligned.centroids),
            'bst_k4_elements':len(compact.centroids),'voxel_geometry_mib':_runtime_voxel_geometry_bytes(aligned)/(1024**2),
            'bst_k4_geometry_mib':compact.geometry_payload_nbytes()/(1024**2),'bst_k4_directory_mib':compact.directory_nbytes()/(1024**2)})

        for p in assoc:
            if progress: print(f'[9.5] global relocalization {prior_session}->{p.query_session} ({len(selected[p.query_session])} scans)...',flush=True)
            for si in range(len(selected[p.query_session])):
                sp,gt,gap,oi,body_sc,body_global=query_data[(p.query_session,si)]
                T_gt=pose_row_to_matrix(gt)
                ft0=time.perf_counter_ns()
                db,pair=pair_cache[p.query_session]
                qdesc=descriptor_cache.query_descriptor(pair['query_by_name'],sp.name,int(oi))
                matches=db.query_descriptor(qdesc,top_k=sc_top_k)
                if not matches:
                    continue
                gt_xy=T_gt[:2,3]
                pos_errs=[]
                for m in matches:
                    rp=ref['pose_rows'][m.reference_index]
                    pos_errs.append(math.hypot(float(rp[1]-gt_xy[0]),float(rp[2]-gt_xy[1])))
                retrieval_ms=(time.perf_counter_ns()-ft0)/1e6

                candidates=[]
                for m in matches:
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
                    if T_ref_query is None:
                        continue
                    Twq=compose_world_query(ref['pose_rows'][ref_db_i],T_ref_query)
                    scan_inliers=int(gres.inliers if gres is not None and gres.converged else 0)
                    cand_runtime=float(kres.runtime_ms+(gres.runtime_ms if gres is not None else 0.0))
                    candidates.append((scan_inliers,int(kres.inliers),-float(m.distance),Twq,m,kres,gres,cand_runtime,ref_db_i))

                if candidates:
                    candidates.sort(key=lambda x:(x[0],x[1],x[2]),reverse=True)
                    _,_,_,T_front,m_best,k_best,g_best,cand_ms,best_ref_i=candidates[0]
                    frontend_reason=''
                    used_kiss=True
                else:
                    m_best=matches[0]; best_ref_i=m_best.reference_index
                    # Preserve Step 9.5 SC-yaw fallback when KISS fails. BEVPlace
                    # selects the place, SC supplies fallback yaw only on failure.
                    sqdesc=make_scan_context(voxel_downsample_first(body_sc,descriptor_voxel_m),sc)
                    _,_,fallback_yaw=scan_context_align(sqdesc,ref['descriptors'][best_ref_i])
                    T_front=scan_context_pose_guess(ref['pose_rows'][best_ref_i],fallback_yaw)
                    k_best=None; g_best=None; cand_ms=0.0; frontend_reason='kiss_failed_all_candidates_sc_yaw_fallback'; used_kiss=False
                # Include failed KISS attempts, lazy reference loads, and fallback.
                # Offline BEV neural extraction time is NOT included here.
                frontend_ms=(time.perf_counter_ns()-ft0)/1e6
                ferr=pose_errors_6d(T_front,T_gt)
                fr={
                    'query_session':p.query_session,'historical_map_session':prior_session,'scan_index':si,
                    'original_scan_index':int(oi),'scan_file':sp.name,
                    'bevplace_top1_position_error_m':float(pos_errs[0]),'bevplace_topk_best_position_error_m':float(min(pos_errs)),
                    'bevplace_top1_descriptor_distance':float(matches[0].distance),'bevplace_topk':int(len(matches)),
                    'selected_reference_rank':int(next(i for i,x in enumerate(matches) if x.reference_index==best_ref_i)+1),
                    'selected_reference_scan_index':int(ref['scan_indices'][best_ref_i]),
                    'selected_reference_position_error_m':math.hypot(float(ref['pose_rows'][best_ref_i,1]-gt[1]),float(ref['pose_rows'][best_ref_i,2]-gt[2])),
                    'kiss_used':bool(used_kiss),'kiss_final_inliers':int(k_best.inliers if k_best else 0),
                    'kiss_rotation_inliers':int(k_best.rotation_inliers if k_best else 0),
                    'scan_gicp_inliers':int(g_best.inliers if g_best else 0),'scan_gicp_converged':bool(g_best.converged if g_best else False),
                    'frontend_translation_xy_m':ferr['translation_xy_m'],'frontend_translation_3d_m':ferr['translation_3d_m'],
                    'frontend_yaw_error_deg':ferr['yaw_deg'],'frontend_rotation_error_deg':ferr['rotation_deg'],
                    'bevplace_descriptor_retrieval_ms':float(retrieval_ms),'frontend_ms':float(frontend_ms),'frontend_failure_reason':frontend_reason,
                }
                frontend_rows.append(fr)

                for method,index,target_centroids in (
                    (VOXEL_METHOD,voxel_index,aligned.centroids),
                    (BST_METHOD,bst_index,compact.centroids),
                ):
                    if method==VOXEL_METHOD and not include_voxel_baseline: continue
                    total0=time.perf_counter_ns()
                    center=T_front[:2,3]
                    tr0=time.perf_counter_ns(); ids=index.query_radius_ids(float(center[0]),float(center[1]),local_map_radius_m); retrieval_ms=(time.perf_counter_ns()-tr0)/1e6
                    target=np.asarray(target_centroids[np.asarray(ids,dtype=np.int64)],dtype=np.float64) if len(ids) else np.empty((0,3),dtype=np.float64)
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
                        'bevplace_top1_position_error_m':float(pos_errs[0]),'bevplace_topk_best_position_error_m':float(min(pos_errs)),
                        'kiss_final_inliers':int(k_best.inliers if k_best else 0),'scan_gicp_inliers':int(g_best.inliers if g_best else 0),
                        'frontend_ms':float(frontend_ms),'map_retrieval_ms':float(retrieval_ms),'local_gicp_ms':float(lres.runtime_ms),
                        'total_ms':float(total_ms),'local_map_primitives':int(len(ids)),'local_gicp_inliers':int(lres.inliers),
                        'local_gicp_converged':bool(lres.converged),'failure_reason':lres.failure_reason,
                    })

        del hierarchy,view,compact,bst_index,voxel_index,aligned,raw
        gc.collect()

    # Explicitly stop the isolated KISS worker before final reporting.
    kiss.close()
    summary=_summary(rows); session_summary=_session_summary(rows); place_summary=_place_summary(frontend_rows)
    # Fair representation gate: same frontend, only final historical map representation differs.
    gates=[]
    if include_voxel_baseline:
        a=next(r for r in summary if r['map_geometry']==VOXEL_METHOD); b=next(r for r in summary if r['map_geometry']==BST_METHOD)
        retain=b['fine_success_rate']/a['fine_success_rate'] if a['fine_success_rate']>0 else float('nan')
        prim_red=1.0-b['mean_local_map_primitives']/a['mean_local_map_primitives']
        local_speed=a['median_local_gicp_ms']/b['median_local_gicp_ms'] if b['median_local_gicp_ms']>0 else float('inf')
        gates=[{'bst_fine_success_retention':retain,'meets_success_retention_0.90':bool(retain>=0.90),
            'bst_primitive_reduction':prim_red,'meets_primitive_reduction_0.40':bool(prim_red>=0.40),
            'bst_local_gicp_speedup':local_speed,'representation_gate_pass':bool(retain>=0.90 and prim_red>=0.40)}]

    _write_csv(out/'localization_trials.csv',rows); _write_csv(out/'localization_summary.csv',summary)
    _write_csv(out/'session_summary.csv',session_summary); _write_csv(out/'frontend_trials.csv',frontend_rows)
    _write_csv(out/'place_recognition_summary.csv',place_summary); _write_csv(out/'reference_database_stats.csv',ref_stats)
    _write_csv(out/'map_geometry_stats.csv',map_stats); _write_csv(out/'representation_gates.csv',gates); _write_csv(out/'query_scans.csv',query_rows)
    metadata=prepared.metadata(); metadata.update({
        'step':'9.6_BEVPlace++','purpose':'Same frozen Step 9.5 downstream stack, BEVPlace++ official pretrained descriptor retrieval replaces Scan Context ranking.',
        'history_sessions':names,'map_session':map_session,'query_sessions':qs,'causal_latest_prior_plan':[asdict(x) for x in plan],
        'query_count_per_session':int(query_count_per_session),'query_scans_csv':str(query_scans_csv) if query_scans_csv else None,
        'no_pose_initialization':True,'frozen_k':FROZEN_K,'frozen_geometry_profile':asdict(FROZEN_PROFILE),
        'scan_context_fallback_config':asdict(sc),'bevplace_retrieval_top_k':int(sc_top_k),'unused_legacy_sc_candidate_pool':int(sc_candidate_pool),
        'reference_keyframes':{'scan_stride':int(reference_scan_stride),'min_translation_m':float(reference_min_translation_m),
            'min_yaw_deg':float(reference_min_yaw_deg),'max_time_s':float(reference_max_time_s),'descriptor_voxel_m':float(descriptor_voxel_m)},
        'global_registration':{'backend':'KISS-Matcher isolated worker','voxel_size_m':float(kiss_voxel_m),'candidate_refinement':'small_gicp GICP',
            'scan_gicp_downsample_m':float(scan_gicp_downsample_m),'scan_gicp_max_correspondence_m':float(scan_gicp_max_corr_m)},
        'local_refinement':{'backend':'small_gicp GICP','map_gicp_downsample_m':float(map_gicp_downsample_m),
            'map_gicp_max_correspondence_m':float(map_gicp_max_corr_m),'local_map_radius_m':float(local_map_radius_m)},
        'success_thresholds':{'fine_xy_m':float(fine_success_translation_m),'fine_yaw_deg':float(fine_success_yaw_deg),
            'strict_6d_translation_m':float(strict_6d_translation_m),'strict_6d_rotation_deg':float(strict_6d_rotation_deg),
            'global_translation_m':float(global_success_translation_m),'global_rotation_deg':float(global_success_rotation_deg)},
        'dependency_versions':{
            'kiss-matcher':importlib.metadata.version('kiss-matcher'),
            'small-gicp':importlib.metadata.version('small-gicp'),
        },
        'retriever':'BEVPlace++ REIN NetVLAD global descriptor, frozen pretrained checkpoint',
        'descriptor_cache_dir':str(descriptor_cache_dir),
        'retriever_provenance':next(iter(pair_cache.values()))[1]['provenance'],
        'online_neural_inference_timed':False,
        'timing_note':'Step 9.6 frontend_ms includes retrieval lookup and all registration attempts, but EXCLUDES offline BEV preprocessing/network extraction; do not interpret total_ms as online end-to-end latency.',
        'construction_once':{'session_block_build_ms':float(sum(session_block_ms)),'adaptive_core_build_ms':float(core_ms)},
        'reference_database_stats':ref_stats,'map_geometry_stats':map_stats,'representation_gates':gates,
        'runtime':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine()},
        'scientific_caveat':'BEVPlace++ pretrained global descriptors replace Scan Context retrieval only. SC yaw fallback remains when KISS fails. Frozen K=4 BST geometry and original prototype codec are unchanged. Neural network extraction was done offline; measured total_ms excludes it. Author BST-ID integration remains pending.',
    })
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return rows,summary,session_summary,frontend_rows,place_summary,gates


__all__=['benchmark_step96','VOXEL_METHOD','BST_METHOD','build_or_load_reference_database']
