#!/usr/bin/env python3
"""Step 11.1 real NCLT provenance and controlled pose-correction experiment."""
from __future__ import annotations
import argparse
import json
from versioned_map.correction.nclt_ingest import NCLTSourceConfig,prepare_nclt,benchmark_nclt


def main(argv=None):
    parser=argparse.ArgumentParser(description='Step 11.1: NCLT scan provenance and correction baseline')
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare',help='generate true scan-by-scan local XYZ and GT pose manifests')
    a.add_argument('--data-root',required=True)
    a.add_argument('--sessions',nargs='+',required=True)
    a.add_argument('--output',required=True)
    a.add_argument('--scan-stride',type=int,default=20)
    a.add_argument('--max-scans-per-session',type=int,default=64)
    a.add_argument('--min-range-m',type=float,default=2.)
    a.add_argument('--max-range-m',type=float,default=80.)
    a.add_argument('--max-pose-gap-us',type=int,default=100_000)
    a.add_argument('--max-points-per-scan',type=int,default=0)
    a.add_argument('--corrupt-from',type=int,default=0)
    a.add_argument('--corrupt-count',type=int,default=8)
    a.add_argument('--translation',nargs=3,type=float,default=[1.,0.,0.])
    a.add_argument('--yaw-deg',type=float,default=0.)
    a.add_argument('--robot-prefix',default='nclt')
    a.add_argument('--overwrite',action='store_true')
    a=sub.add_parser('experiment',help='benchmark ground-truth pose restoration against full rebuild')
    a.add_argument('--prepared',required=True)
    a.add_argument('--output',required=True)
    a.add_argument('--voxel-size',type=float,default=.5)
    a.add_argument('--origin',nargs=3,type=float,default=[0.,0.,0.])
    a.add_argument('--batch-size',type=int,default=16)
    a.add_argument('--repeats',type=int,default=3)
    a.add_argument('--skip-full-disk-rebuild',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='prepare':
        conf=NCLTSourceConfig(scan_stride=args.scan_stride,max_scans_per_session=args.max_scans_per_session,
                              min_range_m=args.min_range_m,max_range_m=args.max_range_m,
                              max_pose_gap_us=args.max_pose_gap_us,max_points_per_scan=args.max_points_per_scan,
                              corrupt_from=args.corrupt_from,corrupt_count=args.corrupt_count,
                              corrupt_translation_m=tuple(args.translation),corrupt_yaw_deg=args.yaw_deg,
                              robot_prefix=args.robot_prefix)
        result=prepare_nclt(args.data_root,args.sessions,args.output,conf,overwrite=args.overwrite)
    else:
        result=benchmark_nclt(args.prepared,args.output,voxel_size=args.voxel_size,origin=args.origin,
                              batch_size=args.batch_size,repeats=args.repeats,
                              full_disk_rebuild=not args.skip_full_disk_rebuild)
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':main()
