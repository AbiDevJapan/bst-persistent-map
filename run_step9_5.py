from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from benchmark.step95_benchmark import benchmark_step95, VOXEL_METHOD, BST_METHOD
from localization.step95 import ScanContextConfig

DEFAULT_HISTORY='2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15'
DEFAULT_QUERIES='2012-02-02,2012-03-17,2012-05-11,2012-06-15'


def main():
    p=ArgumentParser(description='Step 9.5: global-to-local lifelong relocalization (Scan Context -> KISS-Matcher -> GICP -> latest-prior voxel/BST K=4)')
    p.add_argument('--data-root',required=True)
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
    p.add_argument('--sc-top-k',type=int,default=5)
    p.add_argument('--sc-candidate-pool',type=int,default=30)
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
    p.add_argument('--output',default='results/step9_5_global_to_local')
    a=p.parse_args()

    if a.query_scans_csv and a.exclude_query_scans_csv:
        p.error('use --query-scans-csv or --exclude-query-scans-csv, not both')
    if a.query_scans_csv and not Path(a.query_scans_csv).exists():
        raise FileNotFoundError(a.query_scans_csv)
    for x in a.exclude_query_scans_csv:
        if not Path(x).exists(): raise FileNotFoundError(x)

    names=[x.strip() for x in a.history_sessions.split(',') if x.strip()]
    qs=[x.strip() for x in a.query_sessions.split(',') if x.strip()]
    sc=ScanContextConfig(a.sc_rings,a.sc_sectors,a.sc_max_radius,2.0)
    _,summary,session_summary,_,place_summary,gates=benchmark_step95(
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
        include_voxel_baseline=not a.no_voxel_baseline,progress=not a.quiet)

    print('\nStep 9.5 global-to-local summary')
    for r in summary:
        print(f"  {r['map_geometry']:38s}: fine={100*r['fine_success_rate']:.1f}% 2m/5deg={100*r['global_2m5deg_success_rate']:.1f}% "
              f"med={r['median_final_translation_3d_m']:.3f}m/{r['median_final_rotation_error_deg']:.2f}deg total={r['median_total_ms']:.1f}ms prim={r['mean_local_map_primitives']:.0f}")
    if place_summary:
        r=next(x for x in place_summary if x['query_session']=='ALL')
        print('\nPlace/global frontend')
        print(f"  Scan Context top-1 recall: 5m={100*r['scan_context_top1_recall_5m']:.1f}% 10m={100*r['scan_context_top1_recall_10m']:.1f}% 20m={100*r['scan_context_top1_recall_20m']:.1f}%")
        print(f"  Scan Context top-{a.sc_top_k} recall: 5m={100*r['scan_context_topk_recall_5m']:.1f}% 10m={100*r['scan_context_topk_recall_10m']:.1f}% 20m={100*r['scan_context_topk_recall_20m']:.1f}%")
        print(f"  KISS+scan-GICP candidate: fine={100*r['kiss_candidate_fine_rate']:.1f}% 2m/5deg={100*r['kiss_candidate_global_2m5deg_rate']:.1f}% med={r['median_frontend_translation_3d_m']:.2f}m/{r['median_frontend_rotation_error_deg']:.2f}deg")
    if gates:
        g=gates[0]
        print('\nBST representation gate under the same global frontend')
        print(f"  success retention={100*g['bst_fine_success_retention']:.1f}% primitive reduction={100*g['bst_primitive_reduction']:.1f}% local-GICP speedup={g['bst_local_gicp_speedup']:.2f}x pass={g['representation_gate_pass']}")
    print(f'Wrote Step 9.5 results to {a.output}')


if __name__=='__main__': main()
