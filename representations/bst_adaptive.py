from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

from synthetic.generator import GeneratedDataset
from synthetic.world import Voxel
from .common import MetricBox, metric_box_to_voxel_bounds
from .interval_utils import (
    Signature, bits_required, build_occupied_intervals,
    signature_prefixes, tokens_by_depth, prefix_active,
)


@dataclass(frozen=True, order=True)
class RectBlock:
    x0: int; y0: int; z0: int
    sx: int; sy: int; sz: int
    signature_id: int

    @property
    def voxel_count(self): return self.sx*self.sy*self.sz


def _merge_axis(blocks: set[RectBlock], axis: str, limits: tuple[int,int,int]):
    lookup=set(blocks); consumed=set(); result=set(); changed=False
    for b in sorted(blocks):
        if b in consumed: continue
        if axis=='x':
            size=b.sx; origin=b.x0
            if origin % (2*size): continue
            sib=RectBlock(b.x0+size,b.y0,b.z0,b.sx,b.sy,b.sz,b.signature_id)
            parent=RectBlock(b.x0,b.y0,b.z0,b.sx*2,b.sy,b.sz,b.signature_id)
            valid=b.x0+b.sx*2 <= limits[0]
        elif axis=='y':
            size=b.sy; origin=b.y0
            if origin % (2*size): continue
            sib=RectBlock(b.x0,b.y0+size,b.z0,b.sx,b.sy,b.sz,b.signature_id)
            parent=RectBlock(b.x0,b.y0,b.z0,b.sx,b.sy*2,b.sz,b.signature_id)
            valid=b.y0+b.sy*2 <= limits[1]
        else:
            size=b.sz; origin=b.z0
            if origin % (2*size): continue
            sib=RectBlock(b.x0,b.y0,b.z0+size,b.sx,b.sy,b.sz,b.signature_id)
            parent=RectBlock(b.x0,b.y0,b.z0,b.sx,b.sy,b.sz*2,b.signature_id)
            valid=b.z0+b.sz*2 <= limits[2]
        if valid and sib in lookup and sib not in consumed:
            consumed.add(b); consumed.add(sib); result.add(parent); changed=True
    result.update(b for b in blocks if b not in consumed)
    return result, changed


def merge_signature_voxels(voxels: set[Voxel], signature_id: int, limits):
    """
    Independent dyadic x/y/z merging. Unlike an octree, a thin wall can merge
    along x and z without requiring refinement/merging along y.
    """
    blocks={RectBlock(x,y,z,1,1,1,signature_id) for x,y,z in voxels}
    changed=True
    while changed:
        changed=False
        for axis in ('x','y','z'):
            blocks,c=_merge_axis(blocks,axis,limits)
            changed = changed or c
    return blocks


class BSTAdaptiveIntervalMapPrototype:
    """
    Step-5B proposed representation.

    1) Convert event history to exact occupied validity intervals.
    2) Represent those intervals as dyadic temporal prefixes.
    3) Group voxels with identical temporal histories.
    4) Merge adjacent voxels independently along x/y/z into aligned dyadic
       rectangular spatial prefixes.
    5) Query endpoint changes in one pass over candidate space-time blocks.

    This explicitly tests the original hypothesis: heterogeneous spatial and
    temporal resolution, rather than one full x/y/z/t key per event.
    """
    name='BSTAdaptiveIntervalMapPrototype'

    def __init__(self,dataset:GeneratedDataset,bucket_voxels:int=8):
        self.dataset=dataset; self.bucket_voxels=bucket_voxels
        self.time_bits=bits_required(len(dataset.sessions))
        self.signatures: list[Signature]=[]
        self.signature_tokens=[]; self.signature_depth_index=[]
        self.blocks: list[RectBlock]=[]
        self._buckets: dict[tuple[int,int,int], list[int]]={}
        self.bucket_references=0
        self.unmerged_voxels=0
        self.last_query_stats={'blocks_scanned':0,'bucket_refs':0,'voxels_emitted':0}
        self.last_change_stats={'blocks_scanned':0,'bucket_refs':0,'voxels_emitted':0}

    @classmethod
    def from_dataset(cls,dataset:GeneratedDataset,bucket_voxels:int=8):
        obj=cls(dataset,bucket_voxels)
        voxel_intervals=build_occupied_intervals(dataset)
        obj.unmerged_voxels=len(voxel_intervals)
        grouped: dict[Signature,set[Voxel]]=defaultdict(set)
        for v,sig in voxel_intervals.items(): grouped[sig].add(v)

        limits=(dataset.spec.nx,dataset.spec.ny,dataset.spec.nz)
        for sig,voxels in grouped.items():
            sid=len(obj.signatures)
            obj.signatures.append(sig)
            tok=signature_prefixes(sig,len(dataset.sessions))
            obj.signature_tokens.append(tok)
            obj.signature_depth_index.append(tokens_by_depth(tok))
            merged=merge_signature_voxels(voxels,sid,limits)
            obj.blocks.extend(sorted(merged))

        # Coarse spatial directory. A variable-resolution block can appear in
        # multiple directory buckets; references are counted in storage.
        bsz=bucket_voxels
        for rid,b in enumerate(obj.blocks):
            for bx in range(b.x0//bsz,(b.x0+b.sx-1)//bsz+1):
                for by in range(b.y0//bsz,(b.y0+b.sy-1)//bsz+1):
                    for bz in range(b.z0//bsz,(b.z0+b.sz-1)//bsz+1):
                        obj._buckets.setdefault((bx,by,bz),[]).append(rid)
                        obj.bucket_references += 1
        return obj

    def _candidate_ids(self,bounds):
        x0,x1,y0,y1,z0,z1=bounds; b=self.bucket_voxels
        seen=set(); ids=[]; refs=0
        for bx in range(x0//b,x1//b+1):
            for by in range(y0//b,y1//b+1):
                for bz in range(z0//b,z1//b+1):
                    vals=self._buckets.get((bx,by,bz),()); refs += len(vals)
                    for rid in vals:
                        if rid not in seen: seen.add(rid); ids.append(rid)
        return ids,refs

    @staticmethod
    def _intersection(b:RectBlock,bounds):
        x0,x1,y0,y1,z0,z1=bounds
        ax0=max(x0,b.x0); ax1=min(x1,b.x0+b.sx-1)
        ay0=max(y0,b.y0); ay1=min(y1,b.y0+b.sy-1)
        az0=max(z0,b.z0); az1=min(z1,b.z0+b.sz-1)
        if ax0>ax1 or ay0>ay1 or az0>az1: return None
        return ax0,ax1,ay0,ay1,az0,az1

    def _active(self,sid,t):
        return prefix_active(self.signature_depth_index[sid],t,self.time_bits)

    @staticmethod
    def _emit(bounds,out:set[Voxel]):
        x0,x1,y0,y1,z0,z1=bounds
        for x in range(x0,x1+1):
            for y in range(y0,y1+1):
                for z in range(z0,z1+1): out.add((x,y,z))
        return (x1-x0+1)*(y1-y0+1)*(z1-z0+1)

    def region_at_time(self,region:MetricBox,t:int):
        bounds=metric_box_to_voxel_bounds(self.dataset,region)
        ids,refs=self._candidate_ids(bounds); out=set(); scanned=0; emitted=0
        for rid in ids:
            b=self.blocks[rid]; scanned += 1
            inter=self._intersection(b,bounds)
            if inter is None or not self._active(b.signature_id,t): continue
            emitted += self._emit(inter,out)
        self.last_query_stats={'blocks_scanned':scanned,'bucket_refs':refs,'voxels_emitted':emitted}
        return frozenset(out)

    def snapshot(self,t):
        s=self.dataset.spec
        return self.region_at_time((0,s.size_x_m,0,s.size_y_m,0,s.size_z_m),t)

    def changes(self,region:MetricBox,t_a:int,t_b:int):
        if t_b<t_a: t_a,t_b=t_b,t_a
        bounds=metric_box_to_voxel_bounds(self.dataset,region)
        ids,refs=self._candidate_ids(bounds); added=set(); removed=set(); scanned=0; emitted=0
        for rid in ids:
            b=self.blocks[rid]; scanned += 1
            inter=self._intersection(b,bounds)
            if inter is None: continue
            a=self._active(b.signature_id,t_a); c=self._active(b.signature_id,t_b)
            if a==c: continue
            target=added if c else removed
            emitted += self._emit(inter,target)
        self.last_change_stats={'blocks_scanned':scanned,'bucket_refs':refs,'voxels_emitted':emitted}
        return frozenset(added),frozenset(removed)

    def stats(self):
        return {
            'blocks':len(self.blocks),
            'unmerged_voxels':self.unmerged_voxels,
            'spatial_compression_ratio': (len(self.blocks)/self.unmerged_voxels if self.unmerged_voxels else 0.0),
            'unique_temporal_signatures':len(self.signatures),
            'temporal_prefixes':sum(len(x) for x in self.signature_tokens),
            'bucket_references':self.bucket_references,
            'bucket_voxels':self.bucket_voxels,
            'time_bits':self.time_bits,
        }
