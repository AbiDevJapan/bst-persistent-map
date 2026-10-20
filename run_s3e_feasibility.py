#!/usr/bin/env python3
"""Three-real-robot S3E ROS2 bag -> BST-ID feasibility benchmark.

First run `inspect` to discover PointCloud2 topics, then `build` with confirmed
T_pose_lidar extrinsics, then `benchmark`. See README_S3E_FEASIBILITY.md.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collab_feasibility.s3e import list_topics,build_s3e,s3e_request_tasks,ROBOTS
from collab_feasibility.benchmark import benchmark
from versioned_map.store import VersionedStore
from versioned_map.geometry_105b import Geometry4D


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    insp=sub.add_parser('inspect',help='Inspect LiDAR PointCloud2 topic names and sqlite data counts; no rosbags required')
    insp.add_argument('--db3',required=True)
    build=sub.add_parser('build',help='Convert S3E to 3 separate chronological voxel/K4 geometry stores')
    build.add_argument('--db3',required=True)
    build.add_argument('--gt-dir',required=True)
    build.add_argument('--output',required=True)
    build.add_argument('--voxel-size',type=float,default=.75)
    build.add_argument('--epochs',type=int,default=3)
    build.add_argument('--scan-stride',type=int,default=10)
    build.add_argument('--max-scans-per-robot',type=int,default=280)
    build.add_argument('--max-range',type=float,default=40.)
    build.add_argument('--tile-side',type=int,default=16)
    build.add_argument('--topics-json',default=None)
    build.add_argument('--extrinsics-json',default=None)
    build.add_argument('--assume-identity-extrinsics',action='store_true')
    build.add_argument('--max-pose-gap',type=float,default=1.5)
    trial=sub.add_parser('benchmark',help='Replay real robot GPS-centered on-demand ROI requests')
    trial.add_argument('--store-root',required=True)
    trial.add_argument('--output',required=True)
    trial.add_argument('--requests',type=int,default=90)
    trial.add_argument('--gps-sigma',type=float,default=3.)
    trial.add_argument('--alignment-sigma',type=float,default=1.)
    trial.add_argument('--local-radius',type=float,default=20.)
    trial.add_argument('--bandwidth-mbps',type=float,default=2.)
    trial.add_argument('--seed',type=int,default=20261016)
    a=p.parse_args()
    if a.action=='inspect':
        topics=list_topics(Path(a.db3))
        for t in topics:
            if t['type'].endswith('/PointCloud2'):
                print(f"[{t['id']:>3d}] {t['name']} ({t['type']}, {t['serialization_format']})")
        print(f'All topics: {len(topics)}')
    elif a.action=='build':
        sources,meta=build_s3e(a.db3,a.gt_dir,a.output,voxel_size=a.voxel_size,
                   epochs=a.epochs,scan_stride=a.scan_stride,max_scans_per_robot=a.max_scans_per_robot,
                   max_range=a.max_range,tile_side=a.tile_side,topics_json=a.topics_json,
                   extrinsics_json=a.extrinsics_json,assume_identity_extrinsics=a.assume_identity_extrinsics,
                   max_pose_gap=a.max_pose_gap)
        print('Built S3E map stores at',a.output)
        for robot in ROBOTS: print(robot,meta['scan_stats'][robot])
        print('Sessions:',meta['sessions'])
    else:
        root=Path(a.store_root)
        meta=json.loads((root/'ingestion.json').read_text())
        sources=[(VersionedStore(root/'stores'/r/'voxel'),Geometry4D(root/'stores'/r/'geometry')) for r in ROBOTS]
        tasks=s3e_request_tasks(sources,meta,a.requests,a.gps_sigma,a.local_radius,a.alignment_sigma,a.seed)
        summary,info=benchmark(sources,a.output,requests=a.requests,gps_sigma_m=a.gps_sigma,
                  alignment_sigma_m=a.alignment_sigma,local_radius_m=a.local_radius,
                  seed=a.seed,bandwidth_mbps=a.bandwidth_mbps,tasks=tasks,dataset_meta=meta)
        for r in summary:
            print(f"{r['method']:16s} {r['total_wire_bytes']:>11,d} bytes "
                  f"saved={100*r['wire_saving_vs_voxel_tiles']:+5.1f}% exact={r['all_correct']}")
        print(f"Nonempty real-data map requests: {100*info['nonempty_requested_region_fraction']:.1f}%")
        print('Results:',a.output)

if __name__=='__main__':main()
