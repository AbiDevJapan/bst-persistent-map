from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from realdata.nclt import NCLTIngestConfig, prepare_nclt_dataset, save_prepared
from benchmark.step8_benchmark import benchmark_prepared


DEFAULT_PILOT = "2012-01-15,2012-02-02,2012-04-29,2012-05-26,2012-08-04"


def parse_roi(s):
    if not s:
        return None
    vals=tuple(float(x) for x in s.split(","))
    if len(vals)!=6:
        raise ValueError("--roi needs xmin,xmax,ymin,ymax,zmin,zmax in native NCLT local meters")
    return vals


def main():
    p=ArgumentParser(description="Step 8: real NCLT repeated-session representation benchmark")
    p.add_argument("--data-root",required=True,
                   help="Root containing NCLT sessions/velodyne_data and ground_truth")
    p.add_argument("--sessions",default=DEFAULT_PILOT,
                   help="Comma-separated NCLT dates")
    p.add_argument("--voxel-size",type=float,default=0.5)
    p.add_argument("--scan-stride",type=int,default=10,
                   help="Use every Nth synchronized LiDAR scan. Pilot default=10; final experiment should study stride sensitivity.")
    p.add_argument("--min-range",type=float,default=2.0)
    p.add_argument("--max-range",type=float,default=80.0)
    p.add_argument("--min-points-per-voxel",type=int,default=1)
    p.add_argument("--max-pose-gap-us",type=int,default=100000)
    p.add_argument("--max-scans-per-session",type=int,default=None,
                   help="Useful only for smoke tests.")
    p.add_argument("--roi",default=None,
                   help="Optional native NCLT local ROI: xmin,xmax,ymin,ymax,zmin,zmax")
    p.add_argument("--queries",type=int,default=20)
    p.add_argument("--warmups",type=int,default=2)
    p.add_argument("--repetitions",type=int,default=5)
    p.add_argument("--query-seed",type=int,default=42008)
    p.add_argument("--save-prepared",action="store_true")
    p.add_argument("--output",default="results/step8_nclt")
    a=p.parse_args()

    cfg=NCLTIngestConfig(
        voxel_size_m=a.voxel_size,
        scan_stride=a.scan_stride,
        min_range_m=a.min_range,
        max_range_m=a.max_range,
        min_points_per_voxel=a.min_points_per_voxel,
        max_pose_gap_us=a.max_pose_gap_us,
        max_scans_per_session=a.max_scans_per_session,
        roi_local_m=parse_roi(a.roi),
    )
    names=[x.strip() for x in a.sessions.split(",") if x.strip()]
    print("Preparing NCLT:",", ".join(names),flush=True)
    prepared=prepare_nclt_dataset(a.data_root,names,cfg)

    print("Prepared grid:",
          f"{prepared.dataset.spec.nx} x {prepared.dataset.spec.ny} x {prepared.dataset.spec.nz}",
          "voxels; origin(local)=",prepared.origin_local_m,flush=True)
    for st in prepared.session_stats:
        print(
            st.session_name,
            f"scans {st.scans_used}/{st.scans_selected}",
            f"occupied={st.occupied_world_voxels:,}",
            f"pose_gap_p95={st.pose_gap_us_p95/1000:.1f} ms",
            flush=True,
        )

    out=Path(a.output)
    if a.save_prepared:
        save_prepared(prepared,out/"prepared")
    build,q,complexity=benchmark_prepared(
        prepared,out,query_count=a.queries,warmups=a.warmups,
        repetitions=a.repetitions,query_seed=a.query_seed,
    )
    print("\nHistory complexity:")
    for k,v in complexity.items():
        print(f"  {k}: {v}")
    print(f"\nWrote Step-8 results to {out}")


if __name__=="__main__":
    main()
