from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from benchmark.step94_benchmark import benchmark_step94, STATIC, LATEST_VOXEL, BST_LATEST
from localization.step901 import PointToPlaneICP3DOFConfig

DEFAULT_HISTORY='2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15'
DEFAULT_QUERIES='2012-02-02,2012-03-17,2012-05-11,2012-06-15'


def main():
    p=ArgumentParser(description='Step 9.4: causal latest-prior historical map selection with compact BST K=4 retrieval')
    p.add_argument('--data-root',required=True)
    p.add_argument('--history-sessions',default=DEFAULT_HISTORY)
    p.add_argument('--map-session',default='2012-01-15')
    p.add_argument('--query-sessions',default=DEFAULT_QUERIES)
    p.add_argument('--voxel-size',type=float,default=0.5)
    p.add_argument('--scan-stride',type=int,default=10)
    p.add_argument('--min-range',type=float,default=2.0)
    p.add_argument('--max-range',type=float,default=80.0)
    p.add_argument('--min-points-per-voxel',type=int,default=1)
    p.add_argument('--max-pose-gap-us',type=int,default=100000)
    p.add_argument('--cache-dir',default='.cache/step8_1')
    p.add_argument('--query-count-per-session',type=int,default=50)
    p.add_argument('--exclude-query-scans-csv',action='append',default=[],help='Prior query_scans.csv to exclude; may be repeated.')
    p.add_argument('--perturbations',default='0.5:1')
    p.add_argument('--perturb-seed',type=int,default=94090)
    p.add_argument('--localization-max-range',type=float,default=30.0)
    p.add_argument('--scan-downsample',type=float,default=0.75)
    p.add_argument('--max-scan-points',type=int,default=8000)
    p.add_argument('--local-map-radius',type=float,default=40.0)
    p.add_argument('--map-bucket-m',type=float,default=8.0)
    p.add_argument('--success-translation',type=float,default=0.5)
    p.add_argument('--success-yaw-deg',type=float,default=1.0)
    p.add_argument('--icp-min-correspondences',type=int,default=80)
    p.add_argument('--normal-min-neighbors',type=int,default=6)
    p.add_argument('--normal-max-curvature',type=float,default=0.20)
    p.add_argument('--normal-min-horizontal',type=float,default=0.20)
    p.add_argument('--damping',type=float,default=1e-4)
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--resume-checkpoint',action='store_true',help='Resume from checkpoint_localization_trials.csv and checkpoint_historical_map_stats.csv in --output.')
    p.add_argument('--output',default='results/step9_4_historical_selection')
    a=p.parse_args()

    names=[x.strip() for x in a.history_sessions.split(',') if x.strip()]
    qs=[x.strip() for x in a.query_sessions.split(',') if x.strip()]
    excludes=[]
    for x in a.exclude_query_scans_csv:
        if not Path(x).exists(): raise FileNotFoundError(f'exclusion CSV not found: {x}')
        excludes.append(x)
    cfg=PointToPlaneICP3DOFConfig(min_correspondences=a.icp_min_correspondences,normal_min_neighbors=a.normal_min_neighbors,
        normal_max_curvature=a.normal_max_curvature,normal_min_horizontal_component=a.normal_min_horizontal,damping=a.damping)

    _,session_summary,pooled,temporal_pairs,bst_pairs,gates,storage,map_stats,null=benchmark_step94(
        data_root=a.data_root,output_dir=Path(a.output),history_sessions=names,map_session=a.map_session,query_sessions=qs,
        voxel_size_m=a.voxel_size,scan_stride=a.scan_stride,min_range_m=a.min_range,max_range_m=a.max_range,
        min_points_per_voxel=a.min_points_per_voxel,max_pose_gap_us=a.max_pose_gap_us,cache_dir=a.cache_dir,
        query_count_per_session=a.query_count_per_session,exclude_query_scans_csv=excludes,perturbations=a.perturbations,
        perturb_seed=a.perturb_seed,localization_max_range_m=a.localization_max_range,scan_downsample_m=a.scan_downsample,
        max_scan_points=a.max_scan_points,local_map_radius_m=a.local_map_radius,bucket_width_m=a.map_bucket_m,
        success_translation_m=a.success_translation,success_yaw_deg=a.success_yaw_deg,icp_config=cfg,progress=not a.quiet,
        resume_checkpoint=a.resume_checkpoint)

    print('\nStep 9.4 causal map plan',flush=True)
    for q in qs:
        rr=next(r for r in session_summary if r['query_session']==q and r['map_geometry']==LATEST_VOXEL and r['perturbation']=='xy0.5_yaw1')
        prior=rr['historical_map_sessions']
        print(f'  {q}: latest prior = {prior}',flush=True)
    print('\nStep 9.4 per-session main comparison',flush=True)
    for q in qs:
        for geom in (STATIC,LATEST_VOXEL,BST_LATEST):
            rr=next(r for r in session_summary if r['query_session']==q and r['map_geometry']==geom and r['perturbation']=='xy0.5_yaw1')
            print(f"  {q} {geom:24s}: success={100*rr['success_rate']:.1f}% med={rr['median_final_translation_error_m']:.3f}m/{rr['median_final_yaw_error_deg']:.2f}deg total={rr['median_total_ms']:.1f}ms prim={rr['mean_local_map_primitives']:.0f}",flush=True)
    print('\nStep 9.4 temporal null control',flush=True)
    print(f"  first-session static/latest exact={null['passed']} max-float-diff={null['max_abs_float_difference']}",flush=True)
    if gates:
        g=gates[0]
        print('\nStep 9.4 later-session gates',flush=True)
        print(f"  temporal selection: static={100*g['static_success_rate']:.1f}% latest-voxel={100*g['latest_prior_voxel_success_rate']:.1f}% delta={100*g['temporal_selection_success_difference']:+.1f}pp",flush=True)
        print(f"  BST retention={100*g['bst_success_ratio_to_latest_prior_voxel']:.1f}% primitive reduction={100*g['bst_primitive_reduction_vs_latest_prior_voxel']:.1f}% speedup={g['bst_median_total_speedup_vs_latest_prior_voxel']:.2f}x",flush=True)
        print(f"  catalog storage reduction={100*g['historical_catalog_storage_reduction_vs_independent_snapshots']:.1f}% representation gate={g['bst_representation_gate_pass']} temporal-gain={g['temporal_selection_gain_gate_pass']}",flush=True)
    print('\nStep 9.4 historical catalog storage',flush=True)
    for r in storage: print(f"  {r['strategy']:42s}: {r['historical_catalog_total_mib']:.1f} MiB",flush=True)
    print(f'Wrote Step 9.4 results to {a.output}',flush=True)


if __name__=='__main__': main()
