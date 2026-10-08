from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from synthetic.world import WorldSpec
from representations.interval_utils import bits_required, signature_prefixes
from realdata.nclt_packed import linear_decode, linear_encode, PackedNCLTPrepared

MetricBox = tuple[float,float,float,float,float,float]


def metric_box_to_bounds(spec: WorldSpec, region: MetricBox):
    """Correct half-open metric box [min,max) -> inclusive voxel bounds."""
    xmin,xmax,ymin,ymax,zmin,zmax = region
    r=spec.voxel_size_m
    def one(lo,hi,n):
        a=max(0, int(math.floor(lo/r)))
        b=min(n-1, int(math.ceil(hi/r)-1))
        return a,b
    x0,x1=one(xmin,xmax,spec.nx); y0,y1=one(ymin,ymax,spec.ny); z0,z1=one(zmin,zmax,spec.nz)
    return x0,x1,y0,y1,z0,z1


def build_history_table(prepared: PackedNCLTPrepared, progress=True):
    """Return sorted ever-occupied voxel ids and uint64 occupancy-history masks."""
    union=np.empty(0,dtype=np.uint64)
    for i,codes in enumerate(prepared.session_codes):
        t0=time.perf_counter()
        union = codes.copy() if not len(union) else np.union1d(union,codes)
        if progress:
            print(f"[history] union {i+1}/{len(prepared.session_codes)}: {len(union):,} voxels ({time.perf_counter()-t0:.2f}s)",flush=True)
    masks=np.zeros(len(union),dtype=np.uint64)
    for i,codes in enumerate(prepared.session_codes):
        pos=np.searchsorted(union,codes)
        masks[pos] |= np.uint64(1) << np.uint64(i)
    return union,masks


def mask_to_signature(mask:int,n_sessions:int):
    out=[]; start=None
    for t in range(n_sessions):
        active=(mask>>t)&1
        if active and start is None: start=t
        elif not active and start is not None:
            out.append((start,t)); start=None
    if start is not None: out.append((start,n_sessions))
    return tuple(out)


def temporal_signature_bytes(unique_masks: np.ndarray, n_sessions:int) -> tuple[int,int]:
    total=4; token_count=0
    for m in unique_masks:
        tokens=signature_prefixes(mask_to_signature(int(m),n_sessions),n_sessions)
        total += 2
        for depth,_ in tokens:
            total += 1 + math.ceil(depth/8)
        token_count += len(tokens)
    return total,token_count


def _query_linear_rows(codes: np.ndarray, spec:WorldSpec, bounds):
    x0,x1,y0,y1,z0,z1=bounds
    if x0==0 and x1==spec.nx-1 and y0==0 and y1==spec.ny-1 and z0==0 and z1==spec.nz-1:
        return codes.copy()
    xs=np.arange(x0,x1+1,dtype=np.uint64)
    ys=np.arange(y0,y1+1,dtype=np.uint64)
    bases=((xs[:,None]*np.uint64(spec.ny)+ys[None,:])*np.uint64(spec.nz)).ravel()
    lows=bases+np.uint64(z0); highs=bases+np.uint64(z1+1)
    lo=np.searchsorted(codes,lows); hi=np.searchsorted(codes,highs)
    lengths=hi-lo
    if not lengths.sum(): return np.empty(0,dtype=np.uint64)
    parts=[codes[a:b] for a,b in zip(lo,hi) if b>a]
    return np.concatenate(parts) if len(parts)>1 else parts[0].copy()


class PackedSnapshot:
    def __init__(self,prepared:PackedNCLTPrepared):
        self.prepared=prepared; self.spec=prepared.spec
        self.last_query_stats={}; self.last_change_stats={}

    def region_codes(self,region:MetricBox,t:int):
        bounds=metric_box_to_bounds(self.spec,region)
        out=_query_linear_rows(self.prepared.session_codes[t],self.spec,bounds)
        self.last_query_stats={"records_scanned":len(out),"result_voxels":len(out)}
        return out

    def changes_codes(self,region:MetricBox,ta:int,tb:int):
        if tb<ta: ta,tb=tb,ta
        a=self.region_codes(region,ta); b=self.region_codes(region,tb)
        added=np.setdiff1d(b,a,assume_unique=True); removed=np.setdiff1d(a,b,assume_unique=True)
        self.last_change_stats={"records_scanned":len(a)+len(b),"result_voxels":len(added)+len(removed)}
        return added,removed

    def compact_bytes(self):
        return 4+sum(8+6*len(x) for x in self.prepared.session_codes)


# ---------- Flat sparse Octree equivalent ----------

def _root_depth(spec:WorldSpec):
    m=max(spec.nx,spec.ny,spec.nz); d=0; size=1
    while size<m: size*=2; d+=1
    return d,size


def _morton_encode(x,y,z,depth):
    x=np.asarray(x,dtype=np.uint64); y=np.asarray(y,dtype=np.uint64); z=np.asarray(z,dtype=np.uint64)
    out=np.zeros(len(x),dtype=np.uint64)
    for b in range(depth):
        out |= ((x>>b)&1) << (3*b)
        out |= ((y>>b)&1) << (3*b+1)
        out |= ((z>>b)&1) << (3*b+2)
    return out


def _octree_cover(bounds,root_size,depth):
    x0,x1,y0,y1,z0,z1=bounds
    out=[]; visited=0
    stack=[(0,0,0,root_size,0,0)]
    while stack:
        nx,ny,nz,size,d,prefix=stack.pop(); visited+=1
        ex=nx+size-1; ey=ny+size-1; ez=nz+size-1
        if ex<x0 or nx>x1 or ey<y0 or ny>y1 or ez<z0 or nz>z1:
            continue
        if x0<=nx and ex<=x1 and y0<=ny and ey<=y1 and z0<=nz and ez<=z1:
            out.append((d,prefix)); continue
        if size==1:
            out.append((d,prefix)); continue
        h=size//2; nd=d+1
        # Push all 8. Octant bits match Morton high-level x|y<<1|z<<2.
        for zb in (0,1):
            for yb in (0,1):
                for xb in (0,1):
                    octant=xb | (yb<<1) | (zb<<2)
                    stack.append((nx+xb*h,ny+yb*h,nz+zb*h,h,nd,(prefix<<3)|octant))
    return out,visited


class PackedOctreeShared:
    """Sparse Octree semantics, flat Morton-prefix implementation for scale."""
    def __init__(self,prepared,union_codes,masks):
        self.prepared=prepared; self.spec=prepared.spec; self.n_sessions=len(prepared.session_codes)
        self.depth,self.root_size=_root_depth(self.spec)
        x,y,z=linear_decode(union_codes,self.spec)
        morton=_morton_encode(x,y,z,self.depth)
        order=np.argsort(morton,kind="stable")
        self.morton=morton[order]; self.linear=union_codes[order]; self.masks=masks[order]
        self.leaf_count=len(union_codes)
        self.unique_masks=np.unique(masks)
        self.sig_bytes,self.temporal_prefixes=temporal_signature_bytes(self.unique_masks,self.n_sessions)
        self.node_count=self._count_nodes()
        self.last_query_stats={}; self.last_change_stats={}

    def _count_nodes(self):
        if not len(self.morton): return 1
        total=1
        for d in range(1,self.depth+1):
            p=self.morton >> np.uint64(3*(self.depth-d))
            total += 1 + int(np.count_nonzero(p[1:]!=p[:-1]))
        return total

    def _candidate_slices(self,bounds):
        cover,visited=_octree_cover(bounds,self.root_size,self.depth)
        slices=[]; candidates=0
        for d,prefix in cover:
            shift=3*(self.depth-d)
            lo=np.uint64(prefix)<<np.uint64(shift)
            hi=np.uint64(prefix+1)<<np.uint64(shift)
            a=int(np.searchsorted(self.morton,lo,"left")); b=int(np.searchsorted(self.morton,hi,"left"))
            if b>a: slices.append((a,b)); candidates += b-a
        return slices,visited,candidates,len(cover)

    def region_codes(self,region,t):
        bounds=metric_box_to_bounds(self.spec,region)
        slices,visited,candidates,nranges=self._candidate_slices(bounds)
        parts=[]
        bit=np.uint64(1)<<np.uint64(t)
        for a,b in slices:
            keep=(self.masks[a:b]&bit)!=0
            if np.any(keep): parts.append(self.linear[a:b][keep])
        out=np.sort(np.concatenate(parts)) if parts else np.empty(0,dtype=np.uint64)
        self.last_query_stats={"nodes_visited":visited,"prefix_ranges":nranges,"candidate_leaves":candidates,"result_voxels":len(out)}
        return out

    def changes_codes(self,region,ta,tb):
        if tb<ta: ta,tb=tb,ta
        bounds=metric_box_to_bounds(self.spec,region)
        slices,visited,candidates,nranges=self._candidate_slices(bounds)
        add=[]; rem=[]; ba=np.uint64(1)<<np.uint64(ta); bb=np.uint64(1)<<np.uint64(tb)
        for a,b in slices:
            m=self.masks[a:b]; A=(m&ba)!=0; B=(m&bb)!=0
            ka=(~A)&B; kr=A&(~B)
            if np.any(ka): add.append(self.linear[a:b][ka])
            if np.any(kr): rem.append(self.linear[a:b][kr])
        added=np.sort(np.concatenate(add)) if add else np.empty(0,dtype=np.uint64)
        removed=np.sort(np.concatenate(rem)) if rem else np.empty(0,dtype=np.uint64)
        self.last_change_stats={"nodes_visited":visited,"prefix_ranges":nranges,"candidate_leaves":candidates,"result_voxels":len(added)+len(removed)}
        return added,removed

    def compact_bytes(self):
        # Same logical accounting as Step-6 OctreeShared: 9 bytes/node + 4 bytes/leaf signature id.
        return 12 + 9*self.node_count + 4*self.leaf_count + self.sig_bytes

    def stats(self):
        return {"nodes":self.node_count,"leaves":self.leaf_count,"unique_temporal_signatures":len(self.unique_masks),"temporal_prefixes":self.temporal_prefixes,"octree_depth":self.depth}


# ---------- Vectorized exact adaptive dyadic blocks ----------

def _ceil_log2(n:int): return max(1,math.ceil(math.log2(max(1,n))))


def _block_key(arr:np.ndarray,bits):
    xb,yb,zb,lb=bits
    # arr x,y,z,sx,sy,sz. Sizes are dyadic; encode log2 sizes.
    lx=np.log2(arr[:,3]).astype(np.uint64); ly=np.log2(arr[:,4]).astype(np.uint64); lz=np.log2(arr[:,5]).astype(np.uint64)
    key=arr[:,0].astype(np.uint64)
    sh=xb; key |= arr[:,1].astype(np.uint64)<<np.uint64(sh)
    sh+=yb; key |= arr[:,2].astype(np.uint64)<<np.uint64(sh)
    sh+=zb; key |= lx<<np.uint64(sh)
    sh+=lb; key |= ly<<np.uint64(sh)
    sh+=lb; key |= lz<<np.uint64(sh)
    return key


def _merge_axis_fast(arr:np.ndarray,axis:int,limits,bits):
    if len(arr)<2: return arr,0
    keys=_block_key(arr,bits)
    order=np.argsort(keys,kind="stable"); sk=keys[order]
    origin=arr[:,axis]
    size=arr[:,axis+3]
    lower=(origin%(2*size)==0) & (origin+2*size<=limits[axis])
    lower_idx=np.flatnonzero(lower)
    if not len(lower_idx): return arr,0
    shifts=(0,bits[0],bits[0]+bits[1])
    sibkey=keys[lower_idx] + (size[lower_idx].astype(np.uint64)<<np.uint64(shifts[axis]))
    pos=np.searchsorted(sk,sibkey)
    valid=pos<len(sk)
    found=np.zeros(len(pos),dtype=bool)
    found[valid]=sk[pos[valid]]==sibkey[valid]
    lower_idx=lower_idx[found]; pos=pos[found]
    if not len(lower_idx): return arr,0
    sib_idx=order[pos]
    # Alignment makes pairs disjoint, but assert cheaply in tests via unique count.
    consumed=np.zeros(len(arr),dtype=bool); consumed[lower_idx]=True; consumed[sib_idx]=True
    parents=arr[lower_idx].copy(); parents[:,axis+3] *= 2
    return np.concatenate([arr[~consumed],parents],axis=0),len(lower_idx)


def merge_coords_fast(coords:np.ndarray,limits,progress_label=None):
    n=len(coords)
    arr=np.empty((n,6),dtype=np.int32); arr[:,:3]=coords.astype(np.int32,copy=False); arr[:,3:]=1
    bits=(_ceil_log2(limits[0]),_ceil_log2(limits[1]),_ceil_log2(limits[2]),4)
    cycle=0
    while True:
        cycle+=1; merged=0
        before=len(arr)
        for axis in range(3):
            arr,c=_merge_axis_fast(arr,axis,limits,bits); merged+=c
        if progress_label:
            print(f"    [{progress_label}] merge cycle {cycle}: {before:,}->{len(arr):,} blocks ({merged:,} merges)",flush=True)
        if merged==0: break
    return arr


class PackedAdaptiveBucket:
    def __init__(self,prepared,union_codes,masks,bucket_voxels=8,progress=True):
        self.prepared=prepared; self.spec=prepared.spec; self.n_sessions=len(prepared.session_codes); self.bucket_voxels=bucket_voxels
        self.union_codes=union_codes; self.history_masks=masks
        self.unique_masks=np.unique(masks)
        self.sig_bytes,self.temporal_prefixes=temporal_signature_bytes(self.unique_masks,self.n_sessions)
        self.sid_to_mask=self.unique_masks.astype(np.uint64)
        blocks=[]
        order=np.argsort(masks,kind="stable"); sm=masks[order]; su=union_codes[order]
        starts=np.r_[0,1+np.flatnonzero(sm[1:]!=sm[:-1])]; ends=np.r_[starts[1:],len(sm)]
        limits=(self.spec.nx,self.spec.ny,self.spec.nz)
        for gi,(a,b) in enumerate(zip(starts,ends)):
            mask=int(sm[a]); sid=int(np.searchsorted(self.unique_masks,np.uint64(mask)))
            codes=su[a:b]; x,y,z=linear_decode(codes,self.spec); coords=np.column_stack((x,y,z)).astype(np.int32,copy=False)
            label=f"sig {gi+1}/{len(starts)} vox={len(coords):,}" if progress else None
            merged=merge_coords_fast(coords,limits,label)
            sidcol=np.full((len(merged),1),sid,dtype=np.int32)
            blocks.append(np.concatenate([merged,sidcol],axis=1))
        self.blocks=np.concatenate(blocks,axis=0) if blocks else np.empty((0,7),dtype=np.int32)
        self._build_buckets(progress)
        self.last_query_stats={}; self.last_change_stats={}

    def _build_buckets(self,progress):
        b=self.bucket_voxels; B=self.blocks; spec=self.spec
        nbx=(spec.nx+b-1)//b; nby=(spec.ny+b-1)//b; nbz=(spec.nz+b-1)//b
        self._bucket_dims=(nbx,nby,nbz)
        refs_k=[]; refs_i=[]
        bx0=B[:,0]//b; by0=B[:,1]//b; bz0=B[:,2]//b
        bx1=(B[:,0]+B[:,3]-1)//b; by1=(B[:,1]+B[:,4]-1)//b; bz1=(B[:,2]+B[:,5]-1)//b
        one=(bx0==bx1)&(by0==by1)&(bz0==bz1)
        ids=np.flatnonzero(one).astype(np.int32)
        keys=((bx0[one].astype(np.uint64)*np.uint64(nby)+by0[one].astype(np.uint64))*np.uint64(nbz)+bz0[one].astype(np.uint64))
        refs_k.append(keys); refs_i.append(ids)
        multi=np.flatnonzero(~one)
        for rid in multi:
            xs=np.arange(bx0[rid],bx1[rid]+1,dtype=np.uint64); ys=np.arange(by0[rid],by1[rid]+1,dtype=np.uint64); zs=np.arange(bz0[rid],bz1[rid]+1,dtype=np.uint64)
            kk=((xs[:,None,None]*np.uint64(nby)+ys[None,:,None])*np.uint64(nbz)+zs[None,None,:]).ravel()
            refs_k.append(kk); refs_i.append(np.full(len(kk),rid,dtype=np.int32))
        allk=np.concatenate(refs_k) if refs_k else np.empty(0,dtype=np.uint64)
        alli=np.concatenate(refs_i) if refs_i else np.empty(0,dtype=np.int32)
        o=np.argsort(allk,kind="stable"); self.ref_keys=allk[o]; self.ref_ids=alli[o]
        self.bucket_keys,self.bucket_starts,self.bucket_counts=np.unique(self.ref_keys,return_index=True,return_counts=True)
        self.bucket_references=len(self.ref_ids)
        if progress:
            print(f"[adaptive] bucket index: {len(self.bucket_keys):,} buckets, {self.bucket_references:,} refs",flush=True)

    def _candidates(self,bounds):
        x0,x1,y0,y1,z0,z1=bounds; b=self.bucket_voxels; nbx,nby,nbz=self._bucket_dims
        xs=np.arange(x0//b,x1//b+1,dtype=np.uint64); ys=np.arange(y0//b,y1//b+1,dtype=np.uint64); zs=np.arange(z0//b,z1//b+1,dtype=np.uint64)
        q=((xs[:,None,None]*np.uint64(nby)+ys[None,:,None])*np.uint64(nbz)+zs[None,None,:]).ravel()
        p=np.searchsorted(self.bucket_keys,q); valid=p<len(self.bucket_keys); p=p[valid]; q=q[valid]; hit=p[self.bucket_keys[p]==q]
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
        changed=A!=C
        direction=np.where(C,1,-1).astype(np.int8)
        return ids[changed],direction[changed],refs,scanned

    def _materialize(self,ids,bounds):
        if not len(ids): return np.empty(0,dtype=np.uint64)
        x0,x1,y0,y1,z0,z1=bounds; out=[]; spec=self.spec
        for rid in ids:
            b=self.blocks[rid]
            ax0=max(x0,int(b[0])); ax1=min(x1,int(b[0]+b[3]-1)); ay0=max(y0,int(b[1])); ay1=min(y1,int(b[1]+b[4]-1)); az0=max(z0,int(b[2])); az1=min(z1,int(b[2]+b[5]-1))
            xs=np.arange(ax0,ax1+1,dtype=np.uint64); ys=np.arange(ay0,ay1+1,dtype=np.uint64); zs=np.arange(az0,az1+1,dtype=np.uint64)
            codes=(((xs[:,None,None]*np.uint64(spec.ny)+ys[None,:,None])*np.uint64(spec.nz))+zs[None,None,:]).ravel()
            out.append(codes)
        return np.sort(np.concatenate(out)) if out else np.empty(0,dtype=np.uint64)

    def region_codes(self,region,t):
        bounds=metric_box_to_bounds(self.spec,region); ids,_,refs,scanned=self._select(bounds,t=t); out=self._materialize(ids,bounds)
        self.last_query_stats={"blocks_scanned":scanned,"active_blocks":len(ids),"bucket_refs":refs,"result_voxels":len(out)}
        return out

    def changes_codes(self,region,ta,tb):
        if tb<ta: ta,tb=tb,ta
        bounds=metric_box_to_bounds(self.spec,region); ids,direction,refs,scanned=self._select(bounds,ta=ta,tb=tb)
        added=self._materialize(ids[direction>0],bounds); removed=self._materialize(ids[direction<0],bounds)
        self.last_change_stats={"blocks_scanned":scanned,"changed_blocks":len(ids),"bucket_refs":refs,"result_voxels":len(added)+len(removed)}
        return added,removed

    def _spatial_prefix_bytes(self):
        total=np.zeros(len(self.blocks),dtype=np.int64)
        for dim,sizecol in ((self.spec.nx,3),(self.spec.ny,4),(self.spec.nz,5)):
            width=_ceil_log2(dim); ex=np.log2(self.blocks[:,sizecol]).astype(np.int64); plen=np.maximum(0,width-ex)
            total += 1 + ((plen+7)//8)
        return int(total.sum())

    def bst_compact_bytes(self):
        directory=12*len(self.bucket_keys)+4*self.bucket_references
        return 24+self._spatial_prefix_bytes()+4*len(self.blocks)+directory+self.sig_bytes

    def explicit_bytes(self):
        directory=12*len(self.bucket_keys)+4*self.bucket_references
        return 24+16*len(self.blocks)+directory+self.sig_bytes

    def stats(self):
        return {"blocks":len(self.blocks),"unmerged_voxels":len(self.union_codes),"spatial_compression_ratio":len(self.blocks)/len(self.union_codes) if len(self.union_codes) else 0.0,"unique_temporal_signatures":len(self.unique_masks),"temporal_prefixes":self.temporal_prefixes,"bucket_references":self.bucket_references,"bucket_count":len(self.bucket_keys),"bucket_voxels":self.bucket_voxels}
