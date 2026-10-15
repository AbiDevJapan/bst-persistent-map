from __future__ import annotations

import math
import numpy as np

from representations.packed_scale import (
    temporal_signature_bytes, merge_coords_fast, metric_box_to_bounds,
)
from realdata.nclt_packed import linear_decode


def _ceil_log2(n:int):
    return max(1, math.ceil(math.log2(max(1,n))))


class AdaptiveBlockCore:
    """BST-agnostic adaptive spatial/temporal map core.

    This class owns only:
      * exact adaptive dyadic block construction,
      * shared temporal signatures,
      * coarse spatial bucket index,
      * native block selection,
      * optional voxel materialization.

    It contains no BST-ID encoding logic.  Spatial encoding/storage policy is
    supplied by wrapper representations rather than by this core.
    """
    def __init__(self, prepared, union_codes, masks, bucket_voxels=8, progress=True):
        self.prepared=prepared
        self.spec=prepared.spec
        self.n_sessions=len(prepared.session_codes)
        self.bucket_voxels=bucket_voxels
        self.union_codes=union_codes
        self.history_masks=masks
        self.unique_masks=np.unique(masks)
        self.sig_bytes,self.temporal_prefixes=temporal_signature_bytes(self.unique_masks,self.n_sessions)
        self.sid_to_mask=self.unique_masks.astype(np.uint64)

        blocks=[]
        order=np.argsort(masks,kind='stable')
        sm=masks[order]; su=union_codes[order]
        starts=np.r_[0,1+np.flatnonzero(sm[1:]!=sm[:-1])]
        ends=np.r_[starts[1:],len(sm)]
        limits=(self.spec.nx,self.spec.ny,self.spec.nz)
        for gi,(a,b) in enumerate(zip(starts,ends)):
            mask=int(sm[a])
            sid=int(np.searchsorted(self.unique_masks,np.uint64(mask)))
            codes=su[a:b]
            x,y,z=linear_decode(codes,self.spec)
            coords=np.column_stack((x,y,z)).astype(np.int32,copy=False)
            label=f'sig {gi+1}/{len(starts)} vox={len(coords):,}' if progress else None
            merged=merge_coords_fast(coords,limits,label)
            sidcol=np.full((len(merged),1),sid,dtype=np.int32)
            blocks.append(np.concatenate([merged,sidcol],axis=1))
        self.blocks=np.concatenate(blocks,axis=0) if blocks else np.empty((0,7),dtype=np.int32)
        self._build_buckets(progress)
        self.last_query_stats={}
        self.last_change_stats={}

    def _build_buckets(self,progress):
        b=self.bucket_voxels; B=self.blocks; spec=self.spec
        nbx=(spec.nx+b-1)//b; nby=(spec.ny+b-1)//b; nbz=(spec.nz+b-1)//b
        self._bucket_dims=(nbx,nby,nbz)
        refs_k=[]; refs_i=[]
        if len(B):
            bx0=B[:,0]//b; by0=B[:,1]//b; bz0=B[:,2]//b
            bx1=(B[:,0]+B[:,3]-1)//b; by1=(B[:,1]+B[:,4]-1)//b; bz1=(B[:,2]+B[:,5]-1)//b
            one=(bx0==bx1)&(by0==by1)&(bz0==bz1)
            ids=np.flatnonzero(one).astype(np.int32)
            keys=((bx0[one].astype(np.uint64)*np.uint64(nby)+by0[one].astype(np.uint64))*np.uint64(nbz)+bz0[one].astype(np.uint64))
            refs_k.append(keys); refs_i.append(ids)
            multi=np.flatnonzero(~one)
            for rid in multi:
                xs=np.arange(bx0[rid],bx1[rid]+1,dtype=np.uint64)
                ys=np.arange(by0[rid],by1[rid]+1,dtype=np.uint64)
                zs=np.arange(bz0[rid],bz1[rid]+1,dtype=np.uint64)
                kk=((xs[:,None,None]*np.uint64(nby)+ys[None,:,None])*np.uint64(nbz)+zs[None,None,:]).ravel()
                refs_k.append(kk); refs_i.append(np.full(len(kk),rid,dtype=np.int32))
        allk=np.concatenate(refs_k) if refs_k else np.empty(0,dtype=np.uint64)
        alli=np.concatenate(refs_i) if refs_i else np.empty(0,dtype=np.int32)
        if len(allk):
            o=np.argsort(allk,kind='stable'); self.ref_keys=allk[o]; self.ref_ids=alli[o]
            self.bucket_keys,self.bucket_starts,self.bucket_counts=np.unique(self.ref_keys,return_index=True,return_counts=True)
        else:
            self.ref_keys=np.empty(0,dtype=np.uint64); self.ref_ids=np.empty(0,dtype=np.int32)
            self.bucket_keys=np.empty(0,dtype=np.uint64); self.bucket_starts=np.empty(0,dtype=np.int64); self.bucket_counts=np.empty(0,dtype=np.int64)
        self.bucket_references=len(self.ref_ids)
        if progress:
            print(f'[adaptive] bucket index: {len(self.bucket_keys):,} buckets, {self.bucket_references:,} refs',flush=True)

    def _candidates(self,bounds):
        x0,x1,y0,y1,z0,z1=bounds; b=self.bucket_voxels; _,nby,nbz=self._bucket_dims
        xs=np.arange(x0//b,x1//b+1,dtype=np.uint64)
        ys=np.arange(y0//b,y1//b+1,dtype=np.uint64)
        zs=np.arange(z0//b,z1//b+1,dtype=np.uint64)
        q=((xs[:,None,None]*np.uint64(nby)+ys[None,:,None])*np.uint64(nbz)+zs[None,None,:]).ravel()
        p=np.searchsorted(self.bucket_keys,q)
        valid=p<len(self.bucket_keys); p=p[valid]; q=q[valid]
        hit=p[self.bucket_keys[p]==q]
        if not len(hit): return np.empty(0,dtype=np.int32),0
        parts=[self.ref_ids[s:s+c] for s,c in zip(self.bucket_starts[hit],self.bucket_counts[hit])]
        refs=sum(len(x) for x in parts)
        ids=np.unique(np.concatenate(parts)) if len(parts)>1 else np.unique(parts[0])
        return ids,refs

    def _select(self,bounds,t=None,ta=None,tb=None):
        ids,refs=self._candidates(bounds)
        if not len(ids): return ids,np.empty(0,dtype=np.int8),refs,0
        B=self.blocks[ids]; x0,x1,y0,y1,z0,z1=bounds
        inter=(B[:,0]<=x1)&(B[:,0]+B[:,3]-1>=x0)&(B[:,1]<=y1)&(B[:,1]+B[:,4]-1>=y0)&(B[:,2]<=z1)&(B[:,2]+B[:,5]-1>=z0)
        ids=ids[inter]; B=self.blocks[ids]
        scanned=len(ids)
        masks=self.sid_to_mask[B[:,6]]
        if t is not None:
            active=((masks>>np.uint64(t))&1).astype(bool)
            return ids[active],np.ones(np.count_nonzero(active),dtype=np.int8),refs,scanned
        A=((masks>>np.uint64(ta))&1).astype(bool); C=((masks>>np.uint64(tb))&1).astype(bool)
        changed=A!=C; direction=np.where(C,1,-1).astype(np.int8)
        return ids[changed],direction[changed],refs,scanned

    def _clip_blocks(self,ids,bounds):
        """Return native clipped blocks as int32 [x0,y0,z0,sx,sy,sz,sid]."""
        if not len(ids): return np.empty((0,7),dtype=np.int32)
        B=self.blocks[ids]
        x0,x1,y0,y1,z0,z1=bounds
        out=B.copy()
        ax0=np.maximum(B[:,0],x0); ax1=np.minimum(B[:,0]+B[:,3]-1,x1)
        ay0=np.maximum(B[:,1],y0); ay1=np.minimum(B[:,1]+B[:,4]-1,y1)
        az0=np.maximum(B[:,2],z0); az1=np.minimum(B[:,2]+B[:,5]-1,z1)
        out[:,0]=ax0; out[:,1]=ay0; out[:,2]=az0
        out[:,3]=ax1-ax0+1; out[:,4]=ay1-ay0+1; out[:,5]=az1-az0+1
        return out

    def _materialize(self,ids,bounds):
        if not len(ids): return np.empty(0,dtype=np.uint64)
        x0,x1,y0,y1,z0,z1=bounds; out=[]; spec=self.spec
        for rid in ids:
            b=self.blocks[rid]
            ax0=max(x0,int(b[0])); ax1=min(x1,int(b[0]+b[3]-1))
            ay0=max(y0,int(b[1])); ay1=min(y1,int(b[1]+b[4]-1))
            az0=max(z0,int(b[2])); az1=min(z1,int(b[2]+b[5]-1))
            xs=np.arange(ax0,ax1+1,dtype=np.uint64)
            ys=np.arange(ay0,ay1+1,dtype=np.uint64)
            zs=np.arange(az0,az1+1,dtype=np.uint64)
            codes=(((xs[:,None,None]*np.uint64(spec.ny)+ys[None,:,None])*np.uint64(spec.nz))+zs[None,None,:]).ravel()
            out.append(codes)
        return np.sort(np.concatenate(out)) if out else np.empty(0,dtype=np.uint64)

    def region_native(self,region,t):
        bounds=metric_box_to_bounds(self.spec,region)
        ids,_,refs,scanned=self._select(bounds,t=t)
        out=self._clip_blocks(ids,bounds)
        self.last_query_stats={'blocks_scanned':scanned,'active_blocks':len(ids),'bucket_refs':refs,'native_primitives':len(out),'native_kind':'blocks'}
        return out

    def changes_native(self,region,ta,tb):
        if tb<ta: ta,tb=tb,ta
        bounds=metric_box_to_bounds(self.spec,region)
        ids,direction,refs,scanned=self._select(bounds,ta=ta,tb=tb)
        added=self._clip_blocks(ids[direction>0],bounds)
        removed=self._clip_blocks(ids[direction<0],bounds)
        self.last_change_stats={'blocks_scanned':scanned,'changed_blocks':len(ids),'bucket_refs':refs,'native_primitives':len(added)+len(removed),'native_kind':'blocks'}
        return added,removed

    def region_voxels(self,region,t):
        bounds=metric_box_to_bounds(self.spec,region)
        ids,_,refs,scanned=self._select(bounds,t=t)
        out=self._materialize(ids,bounds)
        self.last_query_stats={'blocks_scanned':scanned,'active_blocks':len(ids),'bucket_refs':refs,'result_voxels':len(out),'native_kind':'blocks'}
        return out

    def changes_voxels(self,region,ta,tb):
        if tb<ta: ta,tb=tb,ta
        bounds=metric_box_to_bounds(self.spec,region)
        ids,direction,refs,scanned=self._select(bounds,ta=ta,tb=tb)
        added=self._materialize(ids[direction>0],bounds)
        removed=self._materialize(ids[direction<0],bounds)
        self.last_change_stats={'blocks_scanned':scanned,'changed_blocks':len(ids),'bucket_refs':refs,'result_voxels':len(added)+len(removed),'native_kind':'blocks'}
        return added,removed

    def common_storage_bytes(self):
        directory=12*len(self.bucket_keys)+4*self.bucket_references
        return 24+directory+self.sig_bytes

    def stats(self):
        return {
            'blocks':len(self.blocks),
            'unmerged_voxels':len(self.union_codes),
            'spatial_compression_ratio':len(self.blocks)/len(self.union_codes) if len(self.union_codes) else 0.0,
            'unique_temporal_signatures':len(self.unique_masks),
            'temporal_prefixes':self.temporal_prefixes,
            'bucket_references':self.bucket_references,
            'bucket_count':len(self.bucket_keys),
            'bucket_voxels':self.bucket_voxels,
        }
