#!/usr/bin/env python3
"""Coarse-GPS distributed BST-ID mapping: reproducible first-stage feasibility test.

Examples:
  python run_collab_feasibility.py demo --output results/collab_demo --robots 3 --requests 36
  python run_collab_feasibility.py existing --database /path/to/step10_store \\
    --geometry /path/to/step10_k4 --output results/collab_nclt --requests 100
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collab_feasibility.synthetic import build_synthetic
from collab_feasibility.benchmark import benchmark
from versioned_map.store import VersionedStore
from versioned_map.geometry_105b import Geometry4D


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    for name in ('demo','existing'):
        a=sub.add_parser(name)
        a.add_argument('--output',required=True)
        a.add_argument('--requests',type=int,default=36)
        a.add_argument('--gps-sigma',type=float,default=3.)
        a.add_argument('--alignment-sigma',type=float,default=1.)
        a.add_argument('--local-radius',type=float,default=6.)
        a.add_argument('--bandwidth-mbps',type=float,default=2.)
        a.add_argument('--seed',type=int,default=20261015)
        if name=='demo':
            a.add_argument('--robots',type=int,default=3)
            a.add_argument('--epochs',type=int,default=3)
            a.add_argument('--tile-side',type=int,default=16)
        else:
            a.add_argument('--database',required=True)
            a.add_argument('--geometry',required=True)
    a=p.parse_args()
    if a.action=='demo':
        stores=build_synthetic(Path(a.output)/'stores',robots=a.robots,epochs=a.epochs,tile_side=a.tile_side)
    else:
        stores=[(VersionedStore(a.database),Geometry4D(a.geometry))]
    rows,meta=benchmark(stores,Path(a.output)/'benchmark',requests=a.requests,
         gps_sigma_m=a.gps_sigma,alignment_sigma_m=a.alignment_sigma,
         local_radius_m=a.local_radius,bandwidth_mbps=a.bandwidth_mbps,seed=a.seed)
    print('\nExact ROI map exchange (wire includes framing and metadata):')
    for x in rows:
        print(f"{x['method']:13s} bytes={x['total_wire_bytes']:>9,d} vs-tiles={100*x['wire_saving_vs_voxel_tiles']:+6.1f}% "
              f"local-median={x['median_end_to_end_local_ms']:7.2f}ms exact={x['all_correct']}")
    print('GPS true-ROI enclosure:',round(100*meta['gps_target_roi_enclosed_fraction'],1),'%')
    print('Results:',Path(a.output)/'benchmark')

if __name__=='__main__':main()
