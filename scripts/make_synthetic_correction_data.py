#!/usr/bin/env python3
"""Produce an explicit synthetic multi-robot manifest for reproducible smoke tests.

No simulated measurements are represented as real NCLT/real sensors.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--sources',type=int,default=64);p.add_argument('--points',type=int,default=250)
    p.add_argument('--seed',type=int,default=11001)
    args=p.parse_args();rng=np.random.default_rng(args.seed)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    records=[]
    for n in range(args.sources):
        robot=f'robot-{n%4}'
        local=rng.uniform(-2,2,size=(args.points,3)).astype(np.float64)
        pose=np.eye(4,dtype=np.float64)
        pose[:3,3]=[12*(n%8),12*(n//8),float((n%3)*2)]
        path=root/f'scan_{n:04d}.npz';np.savez_compressed(path,points=local,pose=pose)
        records.append({'source_id':f'scan_{n:04d}','robot_id':robot,
                        'observed_at':1550000000+n,'npz':path.name})
    (root/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    pose=np.eye(4);pose[:3,3]=[.5,0.,0.]
    (root/'example_corrections.json').write_text(json.dumps({'scan_0000':pose.tolist()},indent=2)+'\n')
    print(json.dumps({'manifest':str(root/'manifest.json'),'sources':len(records),'seed':args.seed}))

if __name__=='__main__':main()
