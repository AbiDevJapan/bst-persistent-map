"""S3E v1 ROS2 .db3 -> three causal, robot-owned Step-10 map histories.

Uses ground-truth *only* to align raw LiDAR scans for this oracle-frame
representation benchmark; the GPS noise is introduced separately at request
construction. This is NOT a standalone SLAM or trajectory-estimation method.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import math
import sqlite3
import struct

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from versioned_map.store import VersionedStore
from versioned_map.geometry_105b import Geometry4D
from .benchmark import _bounds

ROBOTS = ('alpha','bob','carol')


def gt_file(path: Path) -> np.ndarray:
    """Timestamp, common-metric-frame xyz, xyzw quaternion (S3E *_gt.txt)."""
    lines=[]
    for line in Path(path).read_text().splitlines():
        s=line.strip().replace(',', ' ')
        if not s or s.startswith('#'):continue
        parts=s.split()
        try: vals=[float(x) for x in parts[:8]]
        except ValueError:continue  # optional header
        if len(vals)==8 and np.all(np.isfinite(vals)) and np.linalg.norm(vals[4:])>0:
            lines.append(vals)
    if len(lines)<2:raise ValueError(f'need at least two finite poses: {path}')
    a=np.array(lines,dtype=float)
    a=a[np.argsort(a[:,0])]
    a=a[np.r_[True,np.diff(a[:,0])>0]]
    if len(a)<2:raise ValueError('ground-truth timestamps not increasing')
    return a


@dataclass
class PoseInterpolator:
    gt: np.ndarray
    max_gap_seconds: float = 1.5

    def __post_init__(self):
        t=self.gt[:,0]
        q=self.gt[:,4:8]/np.linalg.norm(self.gt[:,4:8],axis=1,keepdims=True)
        self._slerp=Slerp(t,Rotation.from_quat(q))

    def at(self,t:float):
        stamps=self.gt[:,0]
        if t<stamps[0] or t>stamps[-1]:return None
        i=int(np.searchsorted(stamps,t))
        near=min(abs(t-stamps[min(i,len(stamps)-1)]),abs(t-stamps[max(0,i-1)]))
        if near>self.max_gap_seconds:return None
        xyz=np.array([np.interp(t,stamps,self.gt[:,j]) for j in (1,2,3)])
        return xyz,self._slerp([t]).as_matrix()[0]


def pointcloud_xyz(msg) -> np.ndarray:
    """Read XYZ fields by PointField offsets, honoring endianness and row padding."""
    fields={f.name.lower():f for f in msg.fields}
    if not all(k in fields for k in ('x','y','z')):raise ValueError('PointCloud2 missing XYZ fields')
    typemap={7:'f4',8:'f8'}
    if any(int(fields[k].datatype) not in typemap for k in ('x','y','z')):
        raise ValueError('XYZ must be float32 or float64 PointCloud2 fields')
    point_step=int(msg.point_step); width=int(msg.width);height=int(msg.height)
    row_step=int(msg.row_step)
    if point_step<=0 or row_step<width*point_step or width<=0 or height<=0:
        raise ValueError('invalid PointCloud2 layout')
    raw=bytes(msg.data)
    if len(raw)<row_step*height:raise ValueError('truncated PointCloud2 data')
    endian='>' if msg.is_bigendian else '<'
    # Struct/NumPy dtype with explicit field offsets; buffer may contain padding.
    names=['x','y','z']
    dtype=np.dtype({'names':names,'formats':[endian+typemap[int(fields[k].datatype)] for k in names],
                    'offsets':[int(fields[k].offset) for k in names], 'itemsize':point_step})
    if max(int(fields[k].offset)+np.dtype(typemap[int(fields[k].datatype)]).itemsize for k in names)>point_step:
        raise ValueError('PointCloud2 field outside point_step')
    arr=np.ndarray((height,width),dtype=dtype,buffer=raw,strides=(row_step,point_step))
    points=np.column_stack([arr[k].reshape(-1).astype(float) for k in names])
    return points[np.all(np.isfinite(points),axis=1)]


def list_topics(db_path: Path):
    with sqlite3.connect(f'file:{Path(db_path).resolve()}?mode=ro',uri=True) as con:
        return [dict(id=int(r[0]),name=str(r[1]),type=str(r[2]),serialization_format=str(r[3]))
                for r in con.execute('SELECT id,name,type,serialization_format FROM topics ORDER BY id')]


def select_lidar_topics(topics,overrides=None):
    overrides=overrides or {}
    selected={}
    for robot in ROBOTS:
        matches=[t for t in topics if t['type'].endswith('/PointCloud2')
                 and (robot in t['name'].lower().replace('blob','bob').split('/')
                      or f'/{robot}_' in t['name'].lower().replace('blob','bob'))]
        if robot in overrides:
            matches=[t for t in topics if t['name']==overrides[robot]]
            if len(matches)!=1 or not matches[0]['type'].endswith('/PointCloud2'):
                raise ValueError(f'invalid {robot} override: {overrides[robot]}')
        if len(matches)!=1:
            raise ValueError(f'{robot}: expected exactly one LiDAR PointCloud2 topic; found {[x["name"] for x in matches]}. Use --topics-json')
        selected[robot]=matches[0]
    return selected


def load_extrinsics(path=None,assume_identity=False):
    """Return T_pose_lidar; NO implicit identity calibration in real mode."""
    if path and assume_identity:raise ValueError('provide --extrinsics-json OR --assume-identity-extrinsics')
    if not path and not assume_identity:
        raise ValueError('Provide calibrated --extrinsics-json (T_pose_lidar) or explicitly opt into --assume-identity-extrinsics for non-validated smoke tests')
    v={} if not path else json.loads(Path(path).read_text())
    matrices={}
    for r in ROBOTS:
        a=np.eye(4) if assume_identity else np.asarray(v[r],float)
        if a.shape!=(4,4) or not np.all(np.isfinite(a)) or not np.allclose(a[3],[0,0,0,1]):
            raise ValueError(f'invalid T_pose_lidar for {r}')
        if not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-3) or np.linalg.det(a[:3,:3])<0.99:
            raise ValueError(f'nonrigid T_pose_lidar for {r}')
        matrices[r]=a
    return matrices


def _frame_bounds(groundtruth,voxel_size,max_range):
    allpos=np.concatenate([a[:,1:4] for a in groundtruth.values()])
    low=np.floor((allpos.min(axis=0)-(max_range+5))/voxel_size)*voxel_size
    high=np.ceil((allpos.max(axis=0)+(max_range+5))/voxel_size)*voxel_size
    dims=np.ceil((high-low)/voxel_size).astype(int)+1
    if max(dims)>=2**21:raise ValueError('world bounds beyond 21-bit BST tile capacity')
    return low,dims


def decode_scans(db, selected, typestore,*,stride,max_scans):
    """Yield time-ordered (robot,time,XYZ) scans from ROS2 SQLite CDR."""
    ids={t['id']:r for r,t in selected.items()}
    seen={r:0 for r in ROBOTS};kept={r:0 for r in ROBOTS}
    with sqlite3.connect(f'file:{Path(db).resolve()}?mode=ro',uri=True) as con:
        for topic_id,timestamp,raw in con.execute('SELECT topic_id,timestamp,data FROM messages WHERE topic_id IN ('+','.join('?' for _ in ids)+') ORDER BY timestamp',tuple(ids)):
            r=ids[int(topic_id)]
            ix=seen[r];seen[r]+=1
            if ix%stride or (max_scans and kept[r]>=max_scans):continue
            typ=selected[r]['type']
            msg=typestore.deserialize_cdr(raw,typ)
            pts=pointcloud_xyz(msg)
            kept[r]+=1
            yield r,float(timestamp)*1e-9,pts


def _make_typestore(topics):
    try:
        from rosbags.typesys import Stores,get_typestore
    except ImportError as e:
        raise ImportError('S3E .db3 decoding requires: pip install rosbags') from e
    # PointCloud2 is part of the standard ROS2 message set.
    typestore=get_typestore(Stores.ROS2_HUMBLE)
    for t in topics:
        if t['serialization_format']!='cdr':raise ValueError(f'non-CDR topic: {t["name"]}')
        if t['type'] not in typestore.types and t['type'].endswith('/PointCloud2'):
            raise ValueError('PointCloud2 type missing from typestore')
    return typestore


def build_s3e(db3,gt_dir,output,*,voxel_size=0.75,epochs=3,
              scan_stride=10,max_scans_per_robot=280,max_range=40.,
              tile_side=16,topics_json=None,extrinsics_json=None,
              assume_identity_extrinsics=False,min_poses=2,max_pose_gap=1.5,
              refresh=False):
    if not (0<voxel_size<=5 and 2<=epochs<=8 and scan_stride>=1 and max_scans_per_robot>0
            and max_range>0):raise ValueError('invalid dataset ingestion configuration')
    db3=Path(db3); gt_dir=Path(gt_dir);output=Path(output)
    if not db3.is_file():raise FileNotFoundError(db3)
    if (output/'ingestion.json').exists() and not refresh:
        m=json.loads((output/'ingestion.json').read_text())
        return [(VersionedStore(output/'stores'/r/'voxel'),Geometry4D(output/'stores'/r/'geometry')) for r in ROBOTS],m
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output is nonempty; choose a new --output (never overwrite previous results)')
    output.mkdir(parents=True,exist_ok=True)
    gt={r:gt_file(gt_dir/f'{r}_gt.txt') for r in ROBOTS}
    ext=load_extrinsics(extrinsics_json,assume_identity_extrinsics)
    topics=list_topics(db3)
    overrides=json.loads(Path(topics_json).read_text()) if topics_json else {}
    selected=select_lidar_topics(topics,overrides)
    typestore=_make_typestore(list(selected.values()))
    start=max(x[0,0] for x in gt.values());stop=min(x[-1,0] for x in gt.values())
    if stop<=start+2*epochs:raise ValueError('robot GT timelines lack shared temporal overlap')
    edges=np.linspace(start,stop,epochs+1)
    origin,dims=_frame_bounds(gt,voxel_size,max_range)
    spec=WorldSpec(*(tuple(int(n)*voxel_size for n in dims)+(voxel_size,)))
    sessions=[datetime.fromtimestamp(float(t),timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') for t in edges[1:]]
    if len(set(sessions))!=epochs:raise ValueError('epoch timestamps collapsed: use fewer epochs')
    interpolators={r:PoseInterpolator(gt[r],max_pose_gap) for r in ROBOTS}
    voxel_hist={r:[[] for _ in range(epochs)] for r in ROBOTS}
    scan_stats={r:{'eligible':0,'accepted':0,'pose_missing':0,'out_of_range':0,'points':0} for r in ROBOTS}
    # Coordinates of receiver trajectory sampled at accepted scans, for actual
    # location-based peer requests rather than random synthetic tile centers.
    loc={r:[[] for _ in range(epochs)] for r in ROBOTS}
    gt_origin=np.asarray(origin)
    for robot,t,points in decode_scans(db3,selected,typestore,stride=scan_stride,max_scans=max_scans_per_robot):
        if t<start or t>=stop:continue
        epoch=min(epochs-1,int(np.searchsorted(edges,t,side='right')-1))
        st=scan_stats[robot];st['eligible']+=1
        pose=interpolators[robot].at(t)
        if pose is None:st['pose_missing']+=1;continue
        p,R=pose
        d=np.linalg.norm(points,axis=1)
        points=points[np.isfinite(d)&(d<=max_range)&(d>=1.5)]
        if not len(points):continue
        T=ext[robot]
        body=points@T[:3,:3].T+T[:3,3]
        world=body@R.T+p
        ijk=np.floor((world-gt_origin)/voxel_size).astype(np.int64)
        good=np.all((ijk>=0)&(ijk<dims),axis=1)
        xyz=ijk[good]
        if len(xyz):
            vox=np.unique(linear_encode(xyz[:,0],xyz[:,1],xyz[:,2],spec))
            voxel_hist[robot][epoch].append(vox)
            loc[robot][epoch].append([float(t),*(p-gt_origin)])
            st['accepted']+=1;st['points']+=len(vox)
            st['out_of_range']+=int((~good).sum())
    generated=[]
    for robot in ROBOTS:
        if scan_stats[robot]['accepted']<min_poses:
            raise ValueError(f'{robot}: too few accepted scans: {scan_stats[robot]}. Check GT timestamp format and topic names')
        # Cumulative snapshot up to each epoch; no future scan observations.
        codes=[];past=np.empty(0,dtype=np.uint64)
        for time_bucket in voxel_hist[robot]:
            if time_bucket:past=np.union1d(past,np.unique(np.concatenate(time_bucket)))
            codes.append(past.copy())
        prep=SimpleNamespace(spec=spec,origin_local_m=tuple(float(v) for v in origin),
                             session_names=sessions,session_codes=codes)
        db_root=output/'stores'/robot/'voxel';geo_root=output/'stores'/robot/'geometry'
        store=VersionedStore(db_root);store.create(prep,codec='morton',tile_side=tile_side)
        for version in sessions:store.build_spatial_index(version,'morton_hash')
        geom=Geometry4D.build(store,geo_root,k=4)
        generated.append((store,geom))
    traces={r:[list(x) for x in loc[r]] for r in ROBOTS}
    meta={'dataset':'S3Ev1/S3E_Playground_2','source_db3':str(db3),'groundtruth_dir':str(gt_dir),
          'labels':list(ROBOTS),'sessions':sessions,'edges_unix_seconds':edges.tolist(),
          'frame':'S3E GT shared UTM metric coordinates minus a common origin; NOT GPS-only estimated map alignment',
          'origin_utm_like_m':origin.tolist(),'voxel_size_m':voxel_size,'dims':dims.tolist(),
          'topics':{r:x['name'] for r,x in selected.items()},'scan_stats':scan_stats,'pose_traces':traces,
          'scan_stride':scan_stride,'max_scans_per_robot':max_scans_per_robot,
          'max_sensor_range_m':max_range,'extrinsic_mode':'identity_opt_in_unvalidated' if assume_identity_extrinsics else 'supplied_T_pose_lidar',
          'limitations':['GT trajectories are used to construct maps (oracle shared frame); not a SLAM evaluation',
                         'GT may be intermittent; missing scans are dropped',
                         'map version epochs are causal but geometry is built offline from completed epochs',
                         'endpoint voxels only, not freespace or surfels']}
    (output/'ingestion.json').write_text(json.dumps(meta,indent=2))
    return generated,meta


def s3e_request_tasks(sources,meta,count,gps_sigma_m,local_radius_m,alignment_sigma_m,seed):
    """GPS requests at *actual* receiver trajectories, peer chosen using prior
    receiver/source track proximity. Ground-truth used as oracle for evaluation.
    """
    rng=np.random.default_rng(seed)
    n=len(sources);sessions=meta['sessions'];epochs=len(sessions)
    points={i:[np.asarray(x,dtype=float).reshape(-1,4) for x in meta['pose_traces'][r]] for i,r in enumerate(ROBOTS)}
    candidates=[]
    for epoch in range(epochs):
        for dst in range(n):
            srcpaths=[]
            rpath=points[dst][epoch]
            if not len(rpath):continue
            for src in range(n):
                if src==dst:continue
                sp=np.concatenate([v for v in points[src][:epoch+1] if len(v)],axis=0) if any(len(v) for v in points[src][:epoch+1]) else np.empty((0,4))
                if not len(sp):continue
                srcpaths.append((src,sp))
            if not srcpaths:continue
            for t, *pos in rpath:
                # The sender must have mapped the area by the request time.
                available=[(src,track[track[:,0]<=t]) for src,track in srcpaths]
                available=[(src,track) for src,track in available if len(track)]
                if not available:continue
                near=min(available,key=lambda x:float(np.min(np.linalg.norm(x[1][:,1:3]-pos[:2],axis=1))))
                candidates.append((epoch,dst,near[0],np.asarray(pos,float)))
    if not candidates:raise ValueError('no temporally valid multi-robot request locations')
    rng.shuffle(candidates)
    selected=[candidates[i%len(candidates)] for i in range(count)]
    out=[]
    for i,(epoch,dst,src,true_pos) in enumerate(selected):
        db,_=sources[src]
        vs=db.meta()['voxel_size_m']
        noisy=true_pos.copy();noisy[:2]+=rng.normal(0,gps_sigma_m,2)
        noisy[2]+=rng.normal(0,alignment_sigma_m)
        # Existing _bounds uses the same local metric origin as all three stores.
        margin=3*math.hypot(gps_sigma_m,alignment_sigma_m)
        req=_bounds(noisy,local_radius_m+margin,db.meta()['dims'],vs)
        target=_bounds(true_pos,local_radius_m,db.meta()['dims'],vs)
        encloses=all(req[j]<=target[j] and req[j+1]>=target[j+1] for j in (0,2,4))
        out.append(dict(request_id=i,source_robot=src,receiver_robot=dst,epoch=epoch,
             version=sessions[epoch],bounds=req,target_bounds=target,
             gps_offset_m=float(np.linalg.norm(noisy[:2]-true_pos[:2])),
             gps_roi_encloses_target=bool(encloses),reused_center=False))
    return out
