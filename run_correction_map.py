#!/usr/bin/env python3
"""Step 11 correction-aware lifelong mapping CLI. Run --help."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from versioned_map.correction import CorrectionMap,Pose
from versioned_map.correction.benchmark import run_correction_benchmark


def arguments():
    p=argparse.ArgumentParser(description='Correction-aware versioned global map research engine')
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('init');a.add_argument('--database',required=True);a.add_argument('--voxel-size',type=float,default=.5)
    a.add_argument('--origin',type=float,nargs=3,default=[0,0,0])
    a=sub.add_parser('ingest');a.add_argument('--database',required=True);a.add_argument('--npz',required=True)
    a.add_argument('--source-id',required=True);a.add_argument('--robot-id',required=True)
    a.add_argument('--observed-at',type=int,required=True);a.add_argument('--note',default='')
    a=sub.add_parser('ingest-manifest');a.add_argument('--database',required=True)
    a.add_argument('--manifest',required=True);a.add_argument('--batch-size',type=int,default=16)
    a=sub.add_parser('correct-batch');a.add_argument('--database',required=True)
    a.add_argument('--corrections-json',required=True);a.add_argument('--note',default='')
    a=sub.add_parser('correct');a.add_argument('--database',required=True);a.add_argument('--source-id',required=True)
    g=a.add_mutually_exclusive_group(required=True);g.add_argument('--pose-npy');g.add_argument('--translation',nargs=3,type=float)
    a.add_argument('--note',default='')
    for name in ('query','observed-history'):
        a=sub.add_parser(name);a.add_argument('--database',required=True)
        a.add_argument('--bounds',type=int,nargs=6,required=True,metavar=('X0','X1','Y0','Y1','Z0','Z1'))
        a.add_argument('--revision',type=int);a.add_argument('--output-npy')
        if name=='observed-history': a.add_argument('--observed-until',type=int,required=True)
    a=sub.add_parser('diff');a.add_argument('--database',required=True);a.add_argument('--bounds',nargs=6,type=int,required=True)
    a.add_argument('--left',type=int,required=True);a.add_argument('--right',type=int,required=True)
    for name in ('stats','log','verify'):
        a=sub.add_parser(name);a.add_argument('--database',required=True)
        if name=='verify':a.add_argument('--revision',type=int)
    a=sub.add_parser('demo');a.add_argument('--database',required=True)
    a=sub.add_parser('benchmark');a.add_argument('--database',required=True);a.add_argument('--output',required=True)
    a.add_argument('--source-id');a.add_argument('--repeats',type=int,default=5)
    a.add_argument('--translation',nargs=3,type=float,default=[.5,0,0])
    return p.parse_args()


def main():
    a=arguments()
    if a.action=='init':
        with CorrectionMap(a.database,voxel_size=a.voxel_size,origin=tuple(a.origin)) as engine:
            print(json.dumps({'initialized':str(a.database),'grid':engine.config},indent=2));return
    if a.action=='benchmark':
        print(json.dumps(run_correction_benchmark(a.database,a.output,repeats=a.repeats,
                   translation=a.translation,source_id=a.source_id),indent=2));return
    with CorrectionMap(a.database) as db:
        if a.action=='ingest':
            with np.load(a.npz,allow_pickle=False) as blob:
                if 'points' not in blob:raise ValueError('NPZ must contain local-frame points array')
                pose=Pose(blob['pose']) if 'pose' in blob else Pose.identity()
                source={'source_id':a.source_id,'robot_id':a.robot_id,'observed_at':a.observed_at,
                        'points':blob['points'],'pose':pose}
            out=db.add_sources([source],a.note).as_dict()
        elif a.action=='ingest-manifest':
            if a.batch_size<=0:raise ValueError('batch-size must be positive')
            manifest_path=Path(a.manifest).resolve()
            manifest=json.loads(manifest_path.read_text())
            if not isinstance(manifest,list):raise ValueError('manifest must be a JSON list')
            records=[];out=[]
            for item in manifest:
                npz_path=Path(item['npz'])
                if not npz_path.is_absolute():npz_path=manifest_path.parent/npz_path
                with np.load(npz_path,allow_pickle=False) as chunk:
                    if 'points' not in chunk:raise ValueError(f'{npz_path} lacks local-frame points')
                    mat=np.asarray(item['pose'],dtype=np.float64) if 'pose' in item else (
                        chunk['pose'] if 'pose' in chunk else None)
                    if mat is None:raise ValueError('manifest ingestion requires an explicit source-to-global pose')
                    records.append({'source_id':item['source_id'],'robot_id':item['robot_id'],
                                    'observed_at':int(item['observed_at']),
                                    'points':chunk['points'].copy(),'pose':Pose(mat)})
                if len(records)>=a.batch_size:
                    out.append(db.add_sources(records,note='batch source ingest').as_dict());records=[]
            if records:out.append(db.add_sources(records,note='batch source ingest').as_dict())
        elif a.action=='correct-batch':
            mapping=json.loads(Path(a.corrections_json).read_text())
            if not isinstance(mapping,dict) or not mapping:raise ValueError('corrections must be a nonempty source_id -> 4x4 map')
            out=db.correct_poses({str(sid):Pose(matrix) for sid,matrix in mapping.items()},a.note).as_dict()
        elif a.action=='correct':
            pose=Pose(np.load(a.pose_npy,allow_pickle=False)) if a.pose_npy else None
            if pose is None:
                old=db._source_pose_asof(a.source_id,db.revision)
                m=old.matrix.copy();m[:3,3]+=np.asarray(a.translation,dtype=float)
                pose=Pose(m)
            out=db.correct_poses({a.source_id:pose},a.note).as_dict()
        elif a.action=='query' or a.action=='observed-history':
            if a.action=='query': result=db.query(a.bounds,a.revision)
            else: result=db.query_observed_history(a.bounds,a.observed_until,a.revision)
            if a.output_npy:
                Path(a.output_npy).parent.mkdir(parents=True,exist_ok=True)
                np.save(a.output_npy,result)
            out={'voxels':len(result),'sample':result[:10].tolist(),'revision':a.revision if a.revision is not None else db.revision,
                 'output_npy':a.output_npy}
        elif a.action=='diff': out=db.diff(a.bounds,a.left,a.right)
        elif a.action=='stats':out=db.stats()
        elif a.action=='log':out=db.revision_log()
        elif a.action=='verify':out={'verified':db.verify(a.revision)}
        elif a.action=='demo':
            pts=np.array([[.1,.1,.1],[.6,.1,.1],[1.1,.1,.1],[-.4,-.1,.1]])
            r1=db.add_sources([dict(source_id='A-001',robot_id='robot-A',observed_at=1000,
                                    points=pts,pose=Pose.identity()),
                               dict(source_id='B-001',robot_id='robot-B',observed_at=1100,
                                    points=pts[:2],pose=Pose.identity())],note='two overlapping robots')
            before=db.query((-10,10,-10,10,-10,10),r1.revision)
            r2=db.correct_poses({'A-001':Pose.translation(1.0,0.,0.)},note='delayed loop closure')
            after=db.query((-10,10,-10,10,-10,10),r2.revision)
            out={'ingest':r1.as_dict(),'correction':r2.as_dict(),
                 'before':before.tolist(),'after':after.tolist(),
                 'historical_r1_preserved':np.array_equal(before,db.query((-10,10,-10,10,-10,10),r1.revision)),
                 'verified':db.verify()}
        else:raise ValueError(a.action)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
