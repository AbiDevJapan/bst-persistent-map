from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path

from benchmark.step911_benchmark import benchmark_step911
from localization.step901 import PointToPlaneICP3DOFConfig

DEFAULT_HISTORY='2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15'
DEFAULT_PROFILES='strict:0.025:0.010:0.985:1.0;balanced:0.050:0.025:0.950:2.0;moderate:0.080:0.050:0.900:4.0;permissive:0.120:0.080:0.800:8.0'


def main():
    p=ArgumentParser(description='Step 9.1.1: geometry-aware dynamic-resolution BST localization Pareto sweep')
    p.add_argument('--data-root',required=True)
    p.add_argument('--history-sessions',default=DEFAULT_HISTORY)
    p.add_argument('--map-session',default='2012-01-15'); p.add_argument('--query-session',default='2012-02-02')
    p.add_argument('--voxel-size',type=float,default=0.5); p.add_argument('--scan-stride',type=int,default=10)
    p.add_argument('--min-range',type=float,default=2.0); p.add_argument('--max-range',type=float,default=80.0)
    p.add_argument('--min-points-per-voxel',type=int,default=1); p.add_argument('--max-pose-gap-us',type=int,default=100000)
    p.add_argument('--cache-dir',default='.cache/step8_1')
    p.add_argument('--query-count',type=int,default=20); p.add_argument('--same-session-query-count',type=int,default=20)
    p.add_argument('--query-scans-csv',default=None); p.add_argument('--perturbations',default='0.5:1')
    p.add_argument('--perturb-seed',type=int,default=42090)
    p.add_argument('--localization-max-range',type=float,default=30.0); p.add_argument('--scan-downsample',type=float,default=0.75)
    p.add_argument('--max-scan-points',type=int,default=8000); p.add_argument('--local-map-radius',type=float,default=40.0); p.add_argument('--map-bucket-m',type=float,default=8.0)
    p.add_argument('--success-translation',type=float,default=0.5); p.add_argument('--success-yaw-deg',type=float,default=1.0)
    p.add_argument('--geometry-profiles',default=DEFAULT_PROFILES,help='semicolon-separated name:sigma_m:relative:min_normal_coherence:max_extent_m')
    p.add_argument('--no-same-session-zero',action='store_true'); p.add_argument('--skip-step91-native',action='store_true')
    p.add_argument('--icp-min-correspondences',type=int,default=80); p.add_argument('--normal-min-neighbors',type=int,default=6)
    p.add_argument('--normal-max-curvature',type=float,default=0.20); p.add_argument('--normal-min-horizontal',type=float,default=0.20); p.add_argument('--damping',type=float,default=1e-4)
    p.add_argument('--quiet',action='store_true'); p.add_argument('--output',default='results/step9_1_1_geometry_aware')
    a=p.parse_args()
    cfg=PointToPlaneICP3DOFConfig(min_correspondences=a.icp_min_correspondences,normal_min_neighbors=a.normal_min_neighbors,normal_max_curvature=a.normal_max_curvature,normal_min_horizontal_component=a.normal_min_horizontal,damping=a.damping)
    names=[x.strip() for x in a.history_sessions.split(',') if x.strip()]
    _,summary,map_rows,pareto=benchmark_step911(
        data_root=a.data_root,output_dir=Path(a.output),history_sessions=names,map_session=a.map_session,query_session=a.query_session,
        voxel_size_m=a.voxel_size,scan_stride=a.scan_stride,min_range_m=a.min_range,max_range_m=a.max_range,min_points_per_voxel=a.min_points_per_voxel,max_pose_gap_us=a.max_pose_gap_us,
        cache_dir=a.cache_dir,query_count=a.query_count,same_session_query_count=a.same_session_query_count,query_scans_csv=a.query_scans_csv,perturbations=a.perturbations,perturb_seed=a.perturb_seed,
        localization_max_range_m=a.localization_max_range,scan_downsample_m=a.scan_downsample,max_scan_points=a.max_scan_points,local_map_radius_m=a.local_map_radius,bucket_width_m=a.map_bucket_m,
        success_translation_m=a.success_translation,success_yaw_deg=a.success_yaw_deg,icp_config=cfg,include_same_session_zero=not a.no_same_session_zero,
        geometry_profiles=a.geometry_profiles,include_step91_native=not a.skip_step91_native,progress=not a.quiet,
    )
    print('\nStep 9.1.1 geometry profiles',flush=True)
    for r in map_rows:
        print(f"  {r['map_geometry']:28s}: elements={int(r['map_elements']):,}; mean={float(r['mean_voxels_per_primitive']):.2f} vox/prim; geom={float(r['minimal_geometry_payload_mib']):.1f} MiB; idx={float(r['query_index_mib']):.1f} MiB",flush=True)
    print('\nStep 9.1.1 localization summary',flush=True)
    for r in summary:
        if r['scenario']=='cross_session':
            print(f"  {r['map_geometry']:28s} {r['perturbation']:14s}: success={100*r['success_rate']:.1f}% med={r['median_final_translation_error_m']:.3f}m/{r['median_final_yaw_error_deg']:.2f}deg primitives={r['mean_local_map_primitives']:.0f} total={r['median_total_ms']:.1f}ms",flush=True)
    print('\nStep 9.1.1 Pareto target xy0.5_yaw1',flush=True)
    for r in pareto:
        print(f"  {r['map_geometry']:28s}: success={100*r['success_rate']:.1f}% primitives={r['mean_local_map_primitives']:.0f} speedup={r['total_speedup_vs_voxel']:.2f}x gate={r['meets_90pct_voxel_success_gate']} paretoP={r['pareto_success_vs_primitives']}",flush=True)
    print(f'Wrote Step 9.1.1 results to {a.output}',flush=True)

if __name__=='__main__': main()
