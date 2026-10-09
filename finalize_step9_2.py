from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from benchmark.step92_recover import recover_step92_metadata

DEFAULT_HISTORY='2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15'
DEFAULT_QUERIES='2012-02-02,2012-03-17,2012-05-11,2012-06-15'


def main():
    p=ArgumentParser(description='Finalize an existing Step 9.2 result directory after the late metadata serialization bug. No map/localization rebuild is performed.')
    p.add_argument('--output',default='results/step9_2_multisession')
    p.add_argument('--history-sessions',default=DEFAULT_HISTORY)
    p.add_argument('--map-session',default='2012-01-15')
    p.add_argument('--query-sessions',default=DEFAULT_QUERIES)
    p.add_argument('--query-count-per-session',type=int,default=50)
    p.add_argument('--same-session-query-count',type=int,default=50)
    p.add_argument('--exclude-query-scans-csv',default='results/step9_1_2_hierarchical_lod/query_scans.csv')
    p.add_argument('--perturbations',default='0.5:1')
    p.add_argument('--perturb-seed',type=int,default=42090)
    p.add_argument('--localization-max-range',type=float,default=30.0)
    p.add_argument('--scan-downsample',type=float,default=0.75)
    p.add_argument('--max-scan-points',type=int,default=8000)
    p.add_argument('--local-map-radius',type=float,default=40.0)
    p.add_argument('--success-translation',type=float,default=0.5)
    p.add_argument('--success-yaw-deg',type=float,default=1.0)
    a=p.parse_args()

    meta=recover_step92_metadata(
        Path(a.output),
        history_sessions=[x.strip() for x in a.history_sessions.split(',') if x.strip()],
        map_session=a.map_session,
        query_sessions=[x.strip() for x in a.query_sessions.split(',') if x.strip()],
        query_count_per_session=a.query_count_per_session,
        same_session_query_count=a.same_session_query_count,
        excluded_tuning_query_scans_csv=a.exclude_query_scans_csv,
        perturbations=a.perturbations,
        perturb_seed=a.perturb_seed,
        localization_max_range_m=a.localization_max_range,
        scan_downsample_m=a.scan_downsample,
        max_scan_points=a.max_scan_points,
        local_map_radius_m=a.local_map_radius,
        success_translation_m=a.success_translation,
        success_yaw_deg=a.success_yaw_deg,
    )
    print(f"Recovered Step 9.2 metadata from existing CSV outputs: {Path(a.output)/'metadata.json'}")
    if meta.get('validation_gates'):
        g=meta['validation_gates'][0]
        print(f"Frozen gate overall_pass={g.get('overall_gate_pass')} success_retention={g.get('success_ratio_bst_to_voxel')} primitive_reduction={g.get('primitive_reduction_vs_voxel')} speedup={g.get('median_total_speedup_vs_voxel')}")


if __name__=='__main__':
    main()
