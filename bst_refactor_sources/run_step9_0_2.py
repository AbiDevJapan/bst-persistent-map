from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from benchmark.step902_benchmark import benchmark_step902
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import parse_pyramid
from realdata.nclt import NCLTIngestConfig
from realdata.nclt_packed import prepare_nclt_packed


def main():
    p = ArgumentParser(description="Step 9.0.2: controlled coarse-to-fine conventional-voxel localization diagnostics")
    p.add_argument("--data-root", required=True)
    p.add_argument("--map-session", default="2012-01-15")
    p.add_argument("--query-session", default="2012-02-02")
    p.add_argument("--voxel-size", type=float, default=.5)
    p.add_argument("--scan-stride", type=int, default=10, help="Map ingestion stride; keep equal to Step 8.x")
    p.add_argument("--min-range", type=float, default=2.0)
    p.add_argument("--max-range", type=float, default=80.0)
    p.add_argument("--min-points-per-voxel", type=int, default=1)
    p.add_argument("--max-pose-gap-us", type=int, default=100000)
    p.add_argument("--cache-dir", default=".cache/step8_1")
    p.add_argument("--no-cache", action="store_true")

    p.add_argument("--query-count", type=int, default=20)
    p.add_argument("--same-session-query-count", type=int, default=None)
    p.add_argument("--query-scans-csv", default=None, help="Replay exact prior cross-session query_scans.csv")
    p.add_argument("--perturbations", default="0.5:1,1:3,2:5,5:10")
    p.add_argument("--perturb-seed", type=int, default=42090)
    p.add_argument("--localization-max-range", type=float, default=30.0)
    p.add_argument("--scan-downsample", type=float, default=.75)
    p.add_argument("--max-scan-points", type=int, default=8000)
    p.add_argument("--local-map-radius", type=float, default=40.0)
    p.add_argument("--map-bucket-voxels", type=int, default=16)
    p.add_argument("--success-translation", type=float, default=.5)
    p.add_argument("--success-yaw-deg", type=float, default=1.0)
    p.add_argument("--no-zero-control", action="store_true")
    p.add_argument("--no-same-session-control", action="store_true")

    p.add_argument(
        "--pyramid",
        default="coarse:1.5:1.0:4.0:15,medium:1.0:0.75:2.0:15,fine:0.75:0.5:1.0:20",
        help="name:scan_voxel:map_voxel:max_corr:iterations comma-separated",
    )
    p.add_argument("--icp-min-correspondences", type=int, default=80)
    p.add_argument("--normal-k", type=int, default=16)
    p.add_argument("--normal-min-neighbors", type=int, default=6)
    p.add_argument("--normal-max-curvature", type=float, default=.20)
    p.add_argument("--normal-min-horizontal", type=float, default=.20)
    p.add_argument("--damping", type=float, default=1e-4)
    p.add_argument("--output", default="results/step9_0_2_controls")
    a = p.parse_args()

    if a.map_session == a.query_session:
        raise ValueError("Use distinct map/query sessions for the Step 9.0.2 main cross-session benchmark")
    ingest = NCLTIngestConfig(
        voxel_size_m=a.voxel_size,
        scan_stride=a.scan_stride,
        min_range_m=a.min_range,
        max_range_m=a.max_range,
        min_points_per_voxel=a.min_points_per_voxel,
        max_pose_gap_us=a.max_pose_gap_us,
    )
    print(f"Step 9.0.2 map={a.map_session}; cross-query={a.query_session}", flush=True)
    prepared = prepare_nclt_packed(
        a.data_root,
        [a.map_session, a.query_session],
        ingest,
        None if a.no_cache else a.cache_dir,
    )
    base_icp = PointToPlaneICP3DOFConfig(
        min_correspondences=a.icp_min_correspondences,
        normal_k=a.normal_k,
        normal_min_neighbors=a.normal_min_neighbors,
        normal_max_curvature=a.normal_max_curvature,
        normal_min_horizontal_component=a.normal_min_horizontal,
        damping=a.damping,
    )
    pyramid = parse_pyramid(a.pyramid)
    _, summary = benchmark_step902(
        prepared=prepared,
        data_root=a.data_root,
        output_dir=Path(a.output),
        map_session=a.map_session,
        query_session=a.query_session,
        query_count=a.query_count,
        same_session_query_count=a.same_session_query_count,
        query_scans_csv=a.query_scans_csv,
        perturbations=a.perturbations,
        perturb_seed=a.perturb_seed,
        localization_max_range_m=a.localization_max_range,
        scan_downsample_m=a.scan_downsample,
        max_scan_points=a.max_scan_points,
        local_map_radius_m=a.local_map_radius,
        map_bucket_voxels=a.map_bucket_voxels,
        success_translation_m=a.success_translation,
        success_yaw_deg=a.success_yaw_deg,
        icp_config=base_icp,
        pyramid=pyramid,
        include_zero_control=not a.no_zero_control,
        include_same_session_control=not a.no_same_session_control,
    )
    print("\nStep 9.0.2 summary", flush=True)
    for r in summary:
        print(
            f"  {r['scenario']:20s} {r['perturbation']:16s}: "
            f"success={100*r['success_rate']:.1f}% "
            f"med_err={r['median_final_translation_error_m']:.3f}m/{r['median_final_yaw_error_deg']:.2f}deg "
            f"shift={r['median_pose_shift_from_initial_m']:.3f}m/{r['median_yaw_shift_from_initial_deg']:.2f}deg "
            f"total={r['median_total_ms']:.1f}ms",
            flush=True,
        )
    print(f"Wrote Step 9.0.2 results to {a.output}", flush=True)


if __name__ == "__main__":
    main()
