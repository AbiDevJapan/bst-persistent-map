from __future__ import annotations
"""Interchangeable *index labels*. Exact geometry ALWAYS uses integer voxel keys.
The upstream author's geospatial semantics are respected: x=longitude, y=latitude,
f=elevation, t=Unix seconds. Local NCLT is NOT intrinsically georeferenced.
The origin is an explicitly declared synthetic anchor unless surveyed coordinates supplied.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Frame:
    origin_longitude:float=0.0
    origin_latitude:float=0.0
    origin_elevation:float=0.0
    metric_origin_x:float=0.0
    metric_origin_y:float=0.0
    metric_origin_z:float=0.0
    voxel_size:float=0.5
    georeferenced:bool=False
    def coordinates(self,x:float,y:float,z:float):
        # local planar approximation, NOT a replacement for surveyed geodesy
        lat=self.origin_latitude + (y-self.metric_origin_y)/111_320.0
        lon=self.origin_longitude + (x-self.metric_origin_x)/(111_320.0*math.cos(math.radians(self.origin_latitude)))
        h=self.origin_elevation + z-self.metric_origin_z
        if not (-85.0511 < lat < 85.0511) or not (-180 < lon < 180):
            raise ValueError('Coordinate outside author codec Web Mercator domain')
        if not (-16383 <= h <= 16384):
            raise ValueError('Altitude outside author codec range')
        return lon,lat,h

class Codec:
    def __init__(self,kind='official',frame=None):
        if kind not in ('official','prototype','morton'):raise ValueError(kind)
        self.kind=kind
        self.frame=frame or Frame()
        if kind=='official':
            from .author import author_module
            self._encoder=author_module('encoder').BSTIDEncoder
    def label(self,ix:int,iy:int,iz:int,session_unix:int):
        if self.kind=='prototype':
            # Original prototype's per-axis prefix concept; index-label only.
            return f'prototype:{ix}:{iy}:{iz}:{session_unix}'
        if self.kind=='morton':
            def spread(x):
                if x<0 or x>=1<<21:raise ValueError('out-of-range voxel key')
                v=0
                for b in range(21):v|=((x>>b)&1)<<(3*b)
                return v
            return f'morton:{spread(ix)|(spread(iy)<<1)|(spread(iz)<<2):016x}:{session_unix}'
        f=self.frame
        # A stable metadata key for a tile anchor, NOT an exact voxel address.
        lon,lat,h=f.coordinates(f.metric_origin_x+(ix+0.5)*f.voxel_size,
                                f.metric_origin_y+(iy+0.5)*f.voxel_size,
                                f.metric_origin_z+(iz+0.5)*f.voxel_size)
        result,bits=self._encoder.encode(lon,lat,h,int(session_unix), 28,28,24,32)
        if result < 0 or result >= 1<<bits:raise ValueError('invalid upstream ID')
        return f'official:{bits}:{result:0{(bits+3)//4}x}'
