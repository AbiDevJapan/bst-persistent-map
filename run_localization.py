from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from benchmark.localization_benchmark import benchmark_localization
from localization.global_frontend import ScanContextConfig

DEFAULT_HISTORY='2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15'
DEFAULT_QUERIES='2012-02-02,2012-03-17,2012-05-11,2012-06-15'


def main():
    p=ArgumentParser(description='Unified lifelong localization: BEVPlace++ or Scan Context -> frozen KISS/GICP -> voxel/BST K=4')
    p.add_argument('--data-root',required=True)
    p.add_argument('--retriever',choices=('bevplace','scan_context'),default='bevplace')
    p.add_argument('--descriptor-cache-dir',default=None,help='Required for BEVPlace++; not used for Scan Context')
    p.add_argument('--history-sessions',default=DEFAULT_HISTORY)
    p.add_argument('--map-session',default='2012-01-15')
    p.add_argument('--query-sessions',default=DEFAULT_QUERIES)
    p.add_argument('--query-count-per-session',type=int,default=50)
    p.add_argument('--query-scans-csv',default=None,help='Replay an existing multi-session query_scans.csv, recommended for paired comparison with Step 9.4.')
    p.add_argument('--exclude-query-scans-csv',action='append',default=[],help='Prior query_scans.csv to exclude when selecting a fresh set; may be repeated.')
    p.add_argument('--cache-dir',default='.cache/step8_1')
    p.add_argument('--voxel-size',type=float,default=0.5)
    p.add_argument('--scan-stride',type=int,default=10)
    p.add_argument('--min-range',type=float,default=2.0)
    p.add_argument('--max-range',type=float,default=80.0)
    p.add_argument('--max-pose-gap-us',type=int,default=100000)

    p.add_argument('--sc-rings',type=int,default=20)
    p.add_argument('--sc-sectors',type=int,default=60)
    p.add_argument('--sc-max-radius',type=float,default=80.0)
    p.add_argument('--sc-top-k','--top-k',dest='sc_top_k',type=int,default=5,help='Retrieval top-K for either method')
    p.add_argument('--sc-candidate-pool',type=int,default=30,help='Unused for BEVPlace++ ranking; only applies to original Scan Context baseline')
    p.add_argument('--reference-scan-stride',type=int,default=5)
    p.add_argument('--reference-min-translation',type=float,default=5.0)
    p.add_argument('--reference-min-yaw-deg',type=float,default=15.0)
    p.add_argument('--reference-max-time-s',type=float,default=5.0)
    p.add_argument('--descriptor-voxel',type=float,default=0.5)

    p.add_argument('--global-point-voxel',type=float,default=0.30)
    p.add_argument('--global-max-points',type=int,default=20000)
    p.add_argument('--kiss-voxel',type=float,default=0.30)
    p.add_argument('--scan-gicp-downsample',type=float,default=0.30)
    p.add_argument('--scan-gicp-max-corr',type=float,default=2.0)
    p.add_argument('--map-gicp-downsample',type=float,default=0.40)
    p.add_argument('--map-gicp-max-corr',type=float,default=2.0)
    p.add_argument('--gicp-threads',type=int,default=4)
    p.add_argument('--local-map-radius',type=float,default=50.0)
    p.add_argument('--map-bucket-m',type=float,default=8.0)

    p.add_argument('--fine-success-translation',type=float,default=0.5)
    p.add_argument('--fine-success-yaw-deg',type=float,default=1.0)
    p.add_argument('--strict-6d-translation',type=float,default=0.5)
    p.add_argument('--strict-6d-rotation-deg',type=float,default=2.0)
    p.add_argument('--global-success-translation',type=float,default=2.0)
    p.add_argument('--global-success-rotation-deg',type=float,default=5.0)
    p.add_argument('--no-voxel-baseline',action='store_true')
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--frontend-strategy',choices=('exhaustive','adaptive','adaptive_geo','bev_local','bev_hybrid'),default='exhaustive',help='Exhaustive frozen top-5 control, or geometrically verified early-exit cascade')
    p.add_argument('--bev-local-feature-dir',default=None)
    p.add_argument('--hybrid-bev-min-inliers',type=int,default=28)
    p.add_argument('--hybrid-bev-min-ratio',type=float,default=0.07)
    p.add_argument('--hybrid-bev-max-candidates',type=int,default=2)
    p.add_argument('--frontend-geo-radius-m',type=float,default=1.0,help='Independent NN verification inlier distance threshold')
    p.add_argument('--frontend-geo-min-overlap',type=float,default=0.55,help='Minimum independent nearest-neighbor overlap')
    p.add_argument('--frontend-geo-max-rmse-m',type=float,default=0.65,help='Maximum 80%% trimmed NN RMSE among overlapping points')
    p.add_argument('--frontend-min-inliers',type=int,default=350,help='Adaptive early exit: minimum converged scan-GICP inliers')
    p.add_argument('--frontend-min-inlier-fraction',type=float,default=0.025,help='Adaptive early exit: scan-GICP inliers / query input points')
    p.add_argument('--frontend-pose-cache',default=None,help='Replay identical frontend SE(3) poses, skipping KISS attempts; timing excludes frontend')
    p.add_argument('--save-frontend-poses',action='store_true',help='Write frontend_poses.npz for paired solver comparison')
    p.add_argument('--local-solver',choices=('gicp','native_surfel','surfel_6dof'),default='gicp')
    p.add_argument('--native-yaw-multistart',action='store_true',help='Try fixed ±1.5 degree starts; native_surfel only')
    p.add_argument('--orientation-diagnostics',action='store_true',help='Write Step 9.7A signed-yaw/geometry audit without altering localization')
    p.add_argument('--output',default='results/step9_6_bevplace_global_to_local')
    a=p.parse_args()
    if a.retriever == "bevplace" and not a.descriptor_cache_dir:
        p.error("--descriptor-cache-dir is required with --retriever bevplace")

    if a.query_scans_csv and a.exclude_query_scans_csv:
        p.error('use --query-scans-csv or --exclude-query-scans-csv, not both')
    if a.query_scans_csv and not Path(a.query_scans_csv).exists():
        raise FileNotFoundError(a.query_scans_csv)
    for x in a.exclude_query_scans_csv:
        if not Path(x).exists(): raise FileNotFoundError(x)

    names=[x.strip() for x in a.history_sessions.split(',') if x.strip()]
    qs=[x.strip() for x in a.query_sessions.split(',') if x.strip()]
    sc=ScanContextConfig(a.sc_rings,a.sc_sectors,a.sc_max_radius,2.0)
    _,summary,session_summary,_,place_summary,gates=benchmark_localization(
        data_root=a.data_root,output_dir=Path(a.output),history_sessions=names,map_session=a.map_session,query_sessions=qs,
        query_count_per_session=a.query_count_per_session,query_scans_csv=a.query_scans_csv,exclude_query_scans_csv=a.exclude_query_scans_csv,
        voxel_size_m=a.voxel_size,scan_stride=a.scan_stride,min_range_m=a.min_range,max_range_m=a.max_range,
        max_pose_gap_us=a.max_pose_gap_us,cache_dir=a.cache_dir,sc_config=sc,sc_top_k=a.sc_top_k,sc_candidate_pool=a.sc_candidate_pool,
        reference_scan_stride=a.reference_scan_stride,reference_min_translation_m=a.reference_min_translation,
        reference_min_yaw_deg=a.reference_min_yaw_deg,reference_max_time_s=a.reference_max_time_s,descriptor_voxel_m=a.descriptor_voxel,
        global_point_voxel_m=a.global_point_voxel,global_max_points=a.global_max_points,kiss_voxel_m=a.kiss_voxel,
        scan_gicp_downsample_m=a.scan_gicp_downsample,scan_gicp_max_corr_m=a.scan_gicp_max_corr,
        map_gicp_downsample_m=a.map_gicp_downsample,map_gicp_max_corr_m=a.map_gicp_max_corr,gicp_threads=a.gicp_threads,
        local_map_radius_m=a.local_map_radius,bucket_width_m=a.map_bucket_m,
        fine_success_translation_m=a.fine_success_translation,fine_success_yaw_deg=a.fine_success_yaw_deg,
        strict_6d_translation_m=a.strict_6d_translation,strict_6d_rotation_deg=a.strict_6d_rotation_deg,
        global_success_translation_m=a.global_success_translation,global_success_rotation_deg=a.global_success_rotation_deg,
        include_voxel_baseline=not a.no_voxel_baseline,progress=not a.quiet,descriptor_cache_dir=a.descriptor_cache_dir,retriever=a.retriever,
        orientation_diagnostics=a.orientation_diagnostics,local_solver=a.local_solver,native_yaw_multistart=a.native_yaw_multistart,frontend_pose_cache=a.frontend_pose_cache,save_frontend_poses=a.save_frontend_poses,frontend_strategy=a.frontend_strategy,frontend_min_inliers=a.frontend_min_inliers,frontend_min_inlier_fraction=a.frontend_min_inlier_fraction,frontend_geo_radius_m=a.frontend_geo_radius_m,frontend_geo_min_overlap=a.frontend_geo_min_overlap,frontend_geo_max_rmse_m=a.frontend_geo_max_rmse_m,bev_local_feature_dir=a.bev_local_feature_dir,hybrid_bev_min_inliers=a.hybrid_bev_min_inliers,hybrid_bev_min_ratio=a.hybrid_bev_min_ratio,hybrid_bev_max_candidates=a.hybrid_bev_max_candidates)

    print('\nStep 9.6 BEVPlace++ global-to-local summary')
    for r in summary:
        print(f"  {r['map_geometry']:38s}: fine={100*r['fine_success_rate']:.1f}% 2m/5deg={100*r['global_2m5deg_success_rate']:.1f}% "
              f"med={r['median_final_translation_3d_m']:.3f}m/{r['median_final_rotation_error_deg']:.2f}deg total={r['median_total_ms']:.1f}ms prim={r['mean_local_map_primitives']:.0f}")
    if place_summary:
        r=next(x for x in place_summary if x['query_session']=='ALL')
        print(f'\n{a.retriever} retrieval/global frontend')
        print(f"  Top-1 recall: 5m={100*r[('bevplace_top1_recall_5m' if a.retriever=='bevplace' else 'scan_context_top1_recall_5m')]:.1f}% 10m={100*r[('bevplace_top1_recall_10m' if a.retriever=='bevplace' else 'scan_context_top1_recall_10m')]:.1f}% 20m={100*r[('bevplace_top1_recall_20m' if a.retriever=='bevplace' else 'scan_context_top1_recall_20m')]:.1f}%")
        print(f"  Top-{a.sc_top_k} recall: 5m={100*r[('bevplace_topk_recall_5m' if a.retriever=='bevplace' else 'scan_context_topk_recall_5m')]:.1f}% 10m={100*r[('bevplace_topk_recall_10m' if a.retriever=='bevplace' else 'scan_context_topk_recall_10m')]:.1f}% 20m={100*r[('bevplace_topk_recall_20m' if a.retriever=='bevplace' else 'scan_context_topk_recall_20m')]:.1f}%")
        print(f"  KISS+scan-GICP candidate: fine={100*r['kiss_candidate_fine_rate']:.1f}% 2m/5deg={100*r['kiss_candidate_global_2m5deg_rate']:.1f}% med={r['median_frontend_translation_3d_m']:.2f}m/{r['median_frontend_rotation_error_deg']:.2f}deg")
    if gates:
        g=gates[0]
        print('\nBST representation gate under the same global frontend')
        print(f"  success retention={100*g['bst_fine_success_retention']:.1f}% primitive reduction={100*g['bst_primitive_reduction']:.1f}% local-GICP speedup={g['bst_local_gicp_speedup']:.2f}x pass={g['representation_gate_pass']}")
    print('NOTE: in BEVPlace++ mode, total_ms excludes offline descriptor extraction; it is not full online latency.' if a.retriever=='bevplace' else 'NOTE: timing definitions follow the consolidated frontend and include all attempted KISS registrations.')
    print(f'Wrote unified localization results to {a.output}')


if __name__=='__main__': main()
