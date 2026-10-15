from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from benchmark.step903a_benchmark import benchmark_step903a, parse_resolutions
from localization.step901 import PointToPlaneICP3DOFConfig


def main():
    p = ArgumentParser(description="Step 9.0.3A: voxel-size localization diagnostic")
    p.add_argument("--data-root", required=True)
    p.add_argument("--map-session", default="2012-01-15")
    p.add_argument("--query-session", default="2012-02-02")
    p.add_argument("--resolutions", default="0.5,1.0,0.25", help="Map occupancy voxel sizes in meters")
    p.add_argument("--scan-stride", type=int, default=10)
    p.add_argument("--min-range", type=float, default=2.0)
    p.add_argument("--max-range", type=float, default=80.0)
    p.add_argument("--min-points-per-voxel", type=int, default=1)
    p.add_argument("--max-pose-gap-us", type=int, default=100000)
    p.add_argument("--cache-dir", default=".cache/step8_1")
    p.add_argument("--cache-policy", choices=["reuse-only", "read-write", "off"], default="reuse-only")

    p.add_argument("--query-count", type=int, default=10)
    p.add_argument("--same-session-query-count", type=int, default=10)
    p.add_argument("--query-scans-csv", default=None, help="Replay prior query_scans.csv; mixed 9.0.2 CSVs are filtered to cross_session")
    p.add_argument("--perturbations", default="0.5:1,1:3")
    p.add_argument("--perturb-seed", type=int, default=42090)
    p.add_argument("--localization-max-range", type=float, default=30.0)
    p.add_argument("--scan-downsample", type=float, default=0.75)
    p.add_argument("--max-scan-points", type=int, default=8000)
    p.add_argument("--local-map-radius", type=float, default=40.0)
    p.add_argument("--map-bucket-m", type=float, default=8.0, help="Keep coarse query buckets approximately constant in physical size")
    p.add_argument("--success-translation", type=float, default=0.5)
    p.add_argument("--success-yaw-deg", type=float, default=1.0)
    p.add_argument("--no-same-session-zero", action="store_true")
    p.add_argument("--same-session-perturbations", action="store_true")

    p.add_argument("--icp-min-correspondences", type=int, default=80)
    p.add_argument("--normal-k", type=int, default=16)
    p.add_argument("--normal-min-neighbors", type=int, default=6)
    p.add_argument("--normal-max-curvature", type=float, default=0.20)
    p.add_argument("--normal-min-horizontal", type=float, default=0.20)
    p.add_argument("--damping", type=float, default=1e-4)
    p.add_argument("--output", default="results/step9_0_3a_voxel_size")
    a = p.parse_args()

    if a.map_session == a.query_session:
        raise ValueError("Use distinct map/query sessions for the cross-session diagnostic")
    cfg = PointToPlaneICP3DOFConfig(
        min_correspondences=a.icp_min_correspondences,
        normal_k=a.normal_k,
        normal_min_neighbors=a.normal_min_neighbors,
        normal_max_curvature=a.normal_max_curvature,
        normal_min_horizontal_component=a.normal_min_horizontal,
        damping=a.damping,
    )
    resolutions = parse_resolutions(a.resolutions)
    _, summary, map_rows = benchmark_step903a(
        data_root=a.data_root, output_dir=Path(a.output), map_session=a.map_session, query_session=a.query_session,
        resolutions=resolutions, scan_stride=a.scan_stride, min_range_m=a.min_range, max_range_m=a.max_range,
        min_points_per_voxel=a.min_points_per_voxel, max_pose_gap_us=a.max_pose_gap_us,
        cache_dir=a.cache_dir, cache_policy=a.cache_policy, query_count=a.query_count,
        same_session_query_count=a.same_session_query_count, query_scans_csv=a.query_scans_csv,
        perturbations=a.perturbations, perturb_seed=a.perturb_seed,
        localization_max_range_m=a.localization_max_range, scan_downsample_m=a.scan_downsample,
        max_scan_points=a.max_scan_points, local_map_radius_m=a.local_map_radius,
        bucket_width_m=a.map_bucket_m, success_translation_m=a.success_translation,
        success_yaw_deg=a.success_yaw_deg, icp_config=cfg,
        include_same_session_zero=not a.no_same_session_zero,
        same_session_perturbations=a.same_session_perturbations,
    )
    print("\nStep 9.0.3A map sizes", flush=True)
    for r in map_rows:
        print(f"  {r['voxel_size_m']:g}m: {r['map_voxels']:,} voxels; cache_hit={r['cache_hit']}", flush=True)
    print("\nStep 9.0.3A localization summary", flush=True)
    for r in summary:
        print(
            f"  {r['voxel_size_m']:g}m {r['scenario']:20s} {r['perturbation']:14s}: "
            f"success={100*r['success_rate']:.1f}% "
            f"med={r['median_final_translation_error_m']:.3f}m/{r['median_final_yaw_error_deg']:.2f}deg "
            f"total={r['median_total_ms']:.1f}ms",
            flush=True,
        )
    print(f"Wrote Step 9.0.3A results to {a.output}", flush=True)


if __name__ == "__main__":
    main()
