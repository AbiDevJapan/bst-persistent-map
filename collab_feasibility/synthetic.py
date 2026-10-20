"""Reproducible *simulated* multi-robot observed-LiDAR endpoint histories.

This is an occupancy-endpoint store, NOT raycasting, a SLAM simulation, or
navigation-quality occupancy. Geometry is deliberately a mixture of coherent
walls and sparse objects. Robot maps overlap as coverage expands over epochs.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from synthetic.world import WorldSpec
from realdata.nclt_packed import linear_encode
from versioned_map.store import VersionedStore
from versioned_map.geometry_105b import Geometry4D

DATES=['2012-01-15','2012-02-02','2012-03-17','2012-05-11','2012-06-15']

def world_points():
    pts=set()
    def wall_x(x,yrange):
        for y in yrange:
            for z in range(2,12):pts.add((x,y,z))
    def wall_y(y,xrange):
        for x in xrange:
            for z in range(2,12):pts.add((x,y,z))
    for x in (10,85):wall_x(x,range(10,86))
    for y in (10,85):wall_y(y,range(10,86))
    for x in (31,55,75):wall_x(x,range(14,82))
    for y in (34,57):wall_y(y,range(14,82))
    for x,y in [(18,21),(25,70),(42,45),(68,22),(79,71),(60,45),(43,17)]:
        for dx in range(2):
            for dy in range(2):
                for z in range(1,14):pts.add((x+dx,y+dy,z))
    # sparse endpoints test irregular geometry where coalescing has no advantage
    for x in range(14,85,7):
        for y in range(15,83,11):pts.add((x,y,1))
    return np.asarray(sorted(pts),dtype=np.int64)


def _robot_centers(robot,epoch):
    # Overlapping but independently grown maps. Centers are in voxel coordinates.
    centers=[[(24,27),(35,30),(46,40),(60,45),(75,60)],
             [(71,35),(60,37),(49,44),(38,53),(27,60)],
             [(45,70),(50,58),(55,45),(55,32),(45,23)],
             [(24,65),(35,59),(46,51),(58,43),(70,30)]]
    seq=centers[robot%len(centers)]
    return seq[:epoch+1]


def build_synthetic(root,robots=3,epochs=3,tile_side=16,k=4):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if robots<2 or robots>4 or epochs<2 or epochs>5:raise ValueError('robots 2..4 and epochs 2..5')
    base=world_points();spec=WorldSpec(48,48,8,0.5)
    generated=[]
    for robot in range(robots):
        dbroot=root/f'robot_{robot}'/'voxel';georoot=root/f'robot_{robot}'/'geometry'
        if dbroot.joinpath('metadata.json').exists() and georoot.joinpath('metadata.json').exists():
            generated.append((VersionedStore(dbroot),Geometry4D(georoot)));continue
        sessions=[]
        for epoch in range(epochs):
            centers=np.asarray(_robot_centers(robot,epoch),dtype=float)
            distances=np.sum((base[:,None,:2]-centers[None,:,:])**2,axis=2)
            keep=np.any(distances<=24**2,axis=1)
            xyz=base[keep]
            # A time-varying observed object that may disappear from endpoint sets.
            extra=np.asarray([(49+2*epoch,46+robot,z) for z in range(1,10)],dtype=np.int64)
            xyz=np.unique(np.vstack((xyz,extra)),axis=0)
            code=linear_encode(xyz[:,0],xyz[:,1],xyz[:,2],spec)
            sessions.append(code)
        prepared=SimpleNamespace(spec=spec,origin_local_m=(0.,0.,0.),
                                 session_names=DATES[:epochs],session_codes=sessions)
        db=VersionedStore(dbroot)
        db.create(prepared,codec='morton',tile_side=tile_side)
        for name in DATES[:epochs]:db.build_spatial_index(name,'morton_hash')
        geom=Geometry4D.build(db,georoot,k=k)
        generated.append((db,geom))
    return generated
