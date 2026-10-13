# Consolidated localization module (moved without re-tuning from step912.py).
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from localization.surfels import NativeBlockSurfelMap, _merge_cov6_pair, finalize_normals
from localization.geometry_gate import GeometryGateProfile, parse_geometry_profiles
from realdata.nclt_packed import linear_decode
from representations.packed_scale import _block_key, _ceil_log2, metric_box_to_bounds
from representations.step83.time_local_core import _geometry_keys


def _cov6_eigen_metrics(cov6: np.ndarray, batch_size: int = 200_000):
    """Return smallest-eigenvalue sigma and relative planarity for cov6 rows."""
    cv = np.asarray(cov6, dtype=np.float32)
    sigma = np.full(len(cv), np.inf, dtype=np.float32)
    rel = np.full(len(cv), np.inf, dtype=np.float32)
    for a in range(0, len(cv), int(batch_size)):
        c6 = cv[a:a+int(batch_size)].astype(np.float64)
        C = np.empty((len(c6), 3, 3), dtype=np.float64)
        C[:,0,0]=c6[:,0]; C[:,0,1]=c6[:,1]; C[:,0,2]=c6[:,2]
        C[:,1,0]=c6[:,1]; C[:,1,1]=c6[:,3]; C[:,1,2]=c6[:,4]
        C[:,2,0]=c6[:,2]; C[:,2,1]=c6[:,4]; C[:,2,2]=c6[:,5]
        ev = np.linalg.eigvalsh(C)
        lam = np.maximum(ev[:,0], 0.0)
        tr = np.sum(ev, axis=1)
        sigma[a:a+len(c6)] = np.sqrt(lam).astype(np.float32)
        rel[a:a+len(c6)] = np.divide(lam, tr, out=np.full(len(c6), np.inf), where=tr>1e-12).astype(np.float32)
    return sigma, rel


def _get_node_rows(node_ids: np.ndarray, leaf_count: int):
    node_ids = np.asarray(node_ids, dtype=np.int64)
    is_leaf = node_ids < int(leaf_count)
    return is_leaf, node_ids[is_leaf], node_ids[~is_leaf] - int(leaf_count)


@dataclass
class NativeSurfelHierarchy:
    """Compact binary surfel hierarchy beneath the final Step-8.4 native roots.

    Leaf node IDs [0, leaf_count) refer directly to the frozen 0.5m voxel-surfel
    table. Internal IDs [leaf_count, leaf_count+M) refer to the arrays below.
    Only internal-node payload is duplicated, keeping memory below a full second
    copy of the voxel table.
    """
    aligned_map: object
    block_map: NativeBlockSurfelMap
    root_node_ids: np.ndarray
    child_a: np.ndarray
    child_b: np.ndarray
    counts: np.ndarray
    centroids: np.ndarray
    cov6: np.ndarray
    normals: np.ndarray
    curvature: np.ndarray
    normal_valid: np.ndarray
    plane_sigma_m: np.ndarray
    relative_planarity: np.ndarray
    normal_coherence: np.ndarray
    max_extent_m: np.ndarray
    voxel_volume: np.ndarray
    build_ms: float

    @property
    def leaf_count(self):
        return len(self.aligned_map.linear_codes)

    @property
    def internal_count(self):
        return len(self.counts)

    def node_has_children(self, node_ids):
        return np.asarray(node_ids, dtype=np.int64) >= self.leaf_count

    def node_centroids(self, node_ids):
        ids=np.asarray(node_ids,dtype=np.int64); leaf,l,i=_get_node_rows(ids,self.leaf_count)
        out=np.empty((len(ids),3),dtype=np.float32)
        out[leaf]=np.asarray(self.aligned_map.centroids)[l]
        out[~leaf]=self.centroids[i]
        return out

    def node_counts(self,node_ids):
        ids=np.asarray(node_ids,dtype=np.int64); leaf,l,i=_get_node_rows(ids,self.leaf_count)
        out=np.empty(len(ids),dtype=np.uint64)
        out[leaf]=np.asarray(self.aligned_map.counts)[l]
        out[~leaf]=self.counts[i]
        return out

    def node_cov6(self,node_ids):
        ids=np.asarray(node_ids,dtype=np.int64); leaf,l,i=_get_node_rows(ids,self.leaf_count)
        out=np.empty((len(ids),6),dtype=np.float32)
        out[leaf]=np.asarray(self.aligned_map.cov6)[l]
        out[~leaf]=self.cov6[i]
        return out

    def node_normals(self,node_ids):
        ids=np.asarray(node_ids,dtype=np.int64); leaf,l,i=_get_node_rows(ids,self.leaf_count)
        out=np.empty((len(ids),3),dtype=np.float32)
        out[leaf]=np.asarray(self.aligned_map.normals)[l]
        out[~leaf]=self.normals[i]
        return out

    def node_valid(self,node_ids):
        ids=np.asarray(node_ids,dtype=np.int64); leaf,l,i=_get_node_rows(ids,self.leaf_count)
        out=np.empty(len(ids),dtype=np.uint8)
        out[leaf]=np.asarray(self.aligned_map.normal_valid)[l]
        out[~leaf]=self.normal_valid[i]
        return out

    def node_volume(self,node_ids):
        ids=np.asarray(node_ids,dtype=np.int64); leaf,l,i=_get_node_rows(ids,self.leaf_count)
        out=np.ones(len(ids),dtype=np.int32)
        out[~leaf]=self.voxel_volume[i]
        return out

    def children(self,node_ids):
        ids=np.asarray(node_ids,dtype=np.int64)
        if np.any(ids < self.leaf_count):
            raise ValueError('leaf node has no children')
        i=ids-self.leaf_count
        return self.child_a[i].astype(np.int64),self.child_b[i].astype(np.int64)

    def gate_good(self,node_ids,profile:GeometryGateProfile):
        ids=np.asarray(node_ids,dtype=np.int64)
        leaf=ids<self.leaf_count
        out=np.ones(len(ids),dtype=bool)
        if np.any(~leaf):
            ii=ids[~leaf]-self.leaf_count
            out[~leaf]=(
                np.isfinite(self.plane_sigma_m[ii])
                & np.isfinite(self.relative_planarity[ii])
                & (self.plane_sigma_m[ii] <= float(profile.max_plane_sigma_m))
                & (self.relative_planarity[ii] <= float(profile.max_relative_planarity))
                & (self.normal_coherence[ii] >= float(profile.min_normal_coherence))
                & (self.max_extent_m[ii] <= float(profile.max_extent_m))
            )
        return out

    def internal_payload_nbytes(self):
        return int(sum(a.nbytes for a in (
            self.child_a,self.child_b,self.counts,self.centroids,self.cov6,self.normals,
            self.curvature,self.normal_valid,self.plane_sigma_m,self.relative_planarity,
            self.normal_coherence,self.max_extent_m,self.voxel_volume,self.root_node_ids,
        )))


def _merge_axis_tree(active, axis, limits, bits, leaf_count, storage, next_internal):
    blocks,cnt,mu,cv,node_ids=active
    if len(blocks)<2:
        return active,next_internal,0
    keys=_block_key(blocks,bits)
    order=np.argsort(keys,kind='stable'); sk=keys[order]
    origin=blocks[:,axis]; size=blocks[:,axis+3]
    lower=(origin%(2*size)==0)&(origin+2*size<=limits[axis])
    lower_idx=np.flatnonzero(lower)
    if not len(lower_idx): return active,next_internal,0
    shifts=(0,bits[0],bits[0]+bits[1])
    sibkey=keys[lower_idx]+(size[lower_idx].astype(np.uint64)<<np.uint64(shifts[axis]))
    pos=np.searchsorted(sk,sibkey); valid=pos<len(sk)
    found=np.zeros(len(pos),dtype=bool); found[valid]=sk[pos[valid]]==sibkey[valid]
    lower_idx=lower_idx[found]; pos=pos[found]
    if not len(lower_idx): return active,next_internal,0
    sib_idx=order[pos]
    k=len(lower_idx)
    internal_idx=np.arange(next_internal,next_internal+k,dtype=np.int32)
    node_parent=(leaf_count+internal_idx).astype(np.int32)

    pc,pm,pv=_merge_cov6_pair(
        cnt[lower_idx],mu[lower_idx],cv[lower_idx],
        cnt[sib_idx],mu[sib_idx],cv[sib_idx],
    )
    parents=blocks[lower_idx].copy(); parents[:,axis+3]*=2

    sl=slice(next_internal,next_internal+k)
    storage['child_a'][sl]=node_ids[lower_idx]
    storage['child_b'][sl]=node_ids[sib_idx]
    storage['counts'][sl]=pc.astype(np.uint64,copy=False)
    storage['centroids'][sl]=pm.astype(np.float32,copy=False)
    storage['cov6'][sl]=pv.astype(np.float32,copy=False)
    storage['max_extent_m'][sl]=(np.max(parents[:,3:6],axis=1)*storage['voxel_size_m']).astype(np.float32)
    storage['voxel_volume'][sl]=np.prod(parents[:,3:6].astype(np.int64),axis=1).astype(np.int32)

    consumed=np.zeros(len(blocks),dtype=bool); consumed[lower_idx]=True; consumed[sib_idx]=True
    return (
        np.concatenate((blocks[~consumed],parents),axis=0),
        np.concatenate((cnt[~consumed],pc),axis=0),
        np.concatenate((mu[~consumed],pm),axis=0),
        np.concatenate((cv[~consumed],pv),axis=0),
        np.concatenate((node_ids[~consumed],node_parent),axis=0),
    ),next_internal+k,k


def build_native_surfel_hierarchy(
    aligned_surfel_map, prepared, session_blocks, core, bst_map, session_index=0,
    normal_min_points=6, normal_max_curvature=0.20,
    normal_min_horizontal_component=0.20, progress=True,
):
    """Reproduce the native map while retaining a compact binary merge forest."""
    t0=time.perf_counter_ns(); t=int(session_index)
    codes=np.asarray(aligned_surfel_map.linear_codes,dtype=np.uint64)
    x,y,z=linear_decode(codes,prepared.spec)
    coords=np.column_stack((x,y,z)).astype(np.int32,copy=False)
    n=len(coords); expected_roots=len(session_blocks[t]); max_internal=max(0,n-expected_roots)
    limits=(prepared.spec.nx,prepared.spec.ny,prepared.spec.nz)
    bits=(_ceil_log2(limits[0]),_ceil_log2(limits[1]),_ceil_log2(limits[2]),4)

    storage={
        'child_a':np.empty(max_internal,dtype=np.int32),
        'child_b':np.empty(max_internal,dtype=np.int32),
        'counts':np.empty(max_internal,dtype=np.uint64),
        'centroids':np.empty((max_internal,3),dtype=np.float32),
        'cov6':np.empty((max_internal,6),dtype=np.float32),
        'max_extent_m':np.empty(max_internal,dtype=np.float32),
        'voxel_volume':np.empty(max_internal,dtype=np.int32),
        'voxel_size_m':float(prepared.spec.voxel_size_m),
    }
    blocks=np.empty((n,6),dtype=np.int32); blocks[:,:3]=coords; blocks[:,3:]=1
    active=(blocks,np.asarray(aligned_surfel_map.counts,dtype=np.uint64).copy(),
            np.asarray(aligned_surfel_map.centroids,dtype=np.float32).copy(),
            np.asarray(aligned_surfel_map.cov6,dtype=np.float32).copy(),
            np.arange(n,dtype=np.int32))
    next_internal=0; cycle=0
    while True:
        cycle+=1; before=len(active[0]); merged=0
        for axis in range(3):
            active,next_internal,c=_merge_axis_tree(active,axis,limits,bits,n,storage,next_internal)
            merged+=c
        if progress:
            print(f'    [9.1.2 hierarchy] merge cycle {cycle}: {before:,}->{len(active[0]):,} blocks ({merged:,} merges)',flush=True)
        if merged==0: break
    if next_internal!=max_internal:
        raise AssertionError(f'internal-node count mismatch: {next_internal} vs expected {max_internal}')

    blocks,cnt,mu,cv,root_nodes=active
    expected=np.asarray(session_blocks[t],dtype=np.int32)
    k1=np.sort(_geometry_keys(blocks,limits)); k2=np.sort(_geometry_keys(expected,limits))
    if not np.array_equal(k1,k2):
        raise AssertionError('hierarchy merge did not reproduce exact Step-8.4 map-session tiling')

    normals,curv,valid=finalize_normals(
        storage['cov6'],storage['counts'],normal_min_points,normal_max_curvature,
        normal_min_horizontal_component,
    )
    sigma,rel=_cov6_eigen_metrics(storage['cov6'])

    # Sign-invariant immediate-child normal coherence. Children are always older
    # nodes, so their normals are already available in either the leaf or internal table.
    def get_normals(ids):
        ids=np.asarray(ids,dtype=np.int64); leaf,l,i=_get_node_rows(ids,n)
        out=np.zeros((len(ids),3),dtype=np.float32); vv=np.zeros(len(ids),dtype=bool)
        if np.any(leaf):
            out[leaf]=np.asarray(aligned_surfel_map.normals)[l]
            vv[leaf]=np.asarray(aligned_surfel_map.normal_valid)[l].astype(bool)
        if np.any(~leaf):
            out[~leaf]=normals[i]
            vv[~leaf]=valid[i].astype(bool)
        return out,vv
    na,va=get_normals(storage['child_a']); nb,vb=get_normals(storage['child_b'])
    coherence=np.ones(max_internal,dtype=np.float32)
    both=va&vb
    coherence[both]=np.abs(np.sum(na[both]*nb[both],axis=1)).astype(np.float32)

    global_keys=_geometry_keys(core.blocks[:,:6],limits)
    local_keys=_geometry_keys(blocks,limits)
    gids=np.searchsorted(global_keys,local_keys)
    if len(gids) and (np.any(gids>=len(global_keys)) or not np.array_equal(global_keys[gids],local_keys)):
        raise AssertionError('map-session hierarchy roots not found in finalized time-local map')
    order=np.argsort(gids,kind='stable')
    root_normals,root_curv,root_valid=finalize_normals(
        cv,cnt,normal_min_points,normal_max_curvature,normal_min_horizontal_component
    )
    block_map=NativeBlockSurfelMap(
        spec=prepared.spec,origin_local_m=prepared.origin_local_m,
        global_block_ids=gids[order].astype(np.int32),blocks=blocks[order],counts=cnt[order].astype(np.uint64),
        centroids=mu[order].astype(np.float32),cov6=cv[order].astype(np.float32),
        normals=root_normals[order],curvature=root_curv[order],normal_valid=root_valid[order],
        bst_map=bst_map,session_index=t,build_ms=(time.perf_counter_ns()-t0)/1e6,
    )
    # Avoid keeping oversized active arrays alive after constructing the final map.
    hierarchy=NativeSurfelHierarchy(
        aligned_map=aligned_surfel_map,block_map=block_map,root_node_ids=root_nodes[order].astype(np.int64),
        child_a=storage['child_a'],child_b=storage['child_b'],counts=storage['counts'],centroids=storage['centroids'],cov6=storage['cov6'],
        normals=normals,curvature=curv,normal_valid=valid,plane_sigma_m=sigma,relative_planarity=rel,
        normal_coherence=coherence,max_extent_m=storage['max_extent_m'],voxel_volume=storage['voxel_volume'],
        build_ms=(time.perf_counter_ns()-t0)/1e6,
    )
    # Root payload must equal the block payload exactly (within float precision).
    rc=hierarchy.node_centroids(hierarchy.root_node_ids)
    if not np.allclose(rc,block_map.centroids,atol=1e-6,rtol=1e-6):
        raise AssertionError('hierarchy root centroids differ from native block surfels')
    return hierarchy


@dataclass
class HierarchicalLODView:
    hierarchy: NativeSurfelHierarchy
    profile: GeometryGateProfile
    max_primitives_per_root: int
    selected_node_ids: np.ndarray
    root_starts: np.ndarray
    root_counts: np.ndarray
    decision_ms: float

    @property
    def spec(self): return self.hierarchy.aligned_map.spec
    @property
    def origin_local_m(self): return self.hierarchy.aligned_map.origin_local_m
    @property
    def map_elements(self): return len(self.selected_node_ids)

    def selected_payload_nbytes(self,minimal=False):
        per_full=8+12+24+12+4+1
        per_min=8+12+12+4+1
        return int(self.map_elements*(per_min if minimal else per_full))

    def directory_nbytes(self):
        return int(self.root_starts.nbytes+self.root_counts.nbytes)

    def mean_voxels_per_primitive(self):
        total=int(self.hierarchy.leaf_count)
        return float(total/self.map_elements) if self.map_elements else 0.0


def build_hierarchical_lod_view(hierarchy:NativeSurfelHierarchy,profile:GeometryGateProfile,max_primitives_per_root:int):
    """Precompute a capped recursive LOD view; localization performs no recursion."""
    t0=time.perf_counter_ns(); K=int(max_primitives_per_root)
    if K<1 or (K&(K-1))!=0:
        raise ValueError('max_primitives_per_root must be a power of two >=1')
    roots=np.asarray(hierarchy.root_node_ids,dtype=np.int64)
    root_labels=np.arange(len(roots),dtype=np.int32)
    current_nodes=roots; current_roots=root_labels
    final_nodes=[]; final_roots=[]
    max_depth=int(round(math.log2(K)))
    for depth in range(max_depth):
        good=hierarchy.gate_good(current_nodes,profile)
        can_split=hierarchy.node_has_children(current_nodes)
        split=(~good)&can_split
        if np.any(~split):
            final_nodes.append(current_nodes[~split]); final_roots.append(current_roots[~split])
        if not np.any(split):
            current_nodes=np.empty(0,dtype=np.int64); current_roots=np.empty(0,dtype=np.int32); break
        a,b=hierarchy.children(current_nodes[split])
        rr=current_roots[split]
        current_nodes=np.column_stack((a,b)).reshape(-1)
        current_roots=np.repeat(rr,2)
    if len(current_nodes):
        final_nodes.append(current_nodes); final_roots.append(current_roots)
    nodes=np.concatenate(final_nodes) if final_nodes else np.empty(0,dtype=np.int64)
    labels=np.concatenate(final_roots) if final_roots else np.empty(0,dtype=np.int32)
    order=np.argsort(labels,kind='stable'); nodes=nodes[order]; labels=labels[order]
    counts=np.bincount(labels,minlength=len(roots)).astype(np.uint8 if K<=255 else np.uint16)
    starts=np.empty(len(roots),dtype=np.uint32 if len(nodes)<2**32 else np.uint64)
    if len(roots):
        starts[0]=0
        if len(roots)>1: starts[1:]=np.cumsum(counts[:-1],dtype=starts.dtype)
    if len(counts) and int(counts.max())>K:
        raise AssertionError('LOD root exceeded K cap')
    coverage=int(hierarchy.node_volume(nodes).sum(dtype=np.int64))
    if coverage!=hierarchy.leaf_count:
        raise AssertionError(f'LOD coverage mismatch: {coverage} vs {hierarchy.leaf_count}')
    return HierarchicalLODView(hierarchy,profile,K,nodes,starts,counts,(time.perf_counter_ns()-t0)/1e6)


class HierarchicalLODBSTIndex:
    """Step-8.5 time-index roots -> precomputed hierarchical LOD surfels."""
    def __init__(self,view:HierarchicalLODView,bst_map,session_index:int):
        self.map=view; self.bst_map=bst_map; self.session_index=int(session_index); self.last_stats={}

    def query_radius_ids(self,cx:float,cy:float,radius_m:float)->np.ndarray:
        core=self.bst_map.core; ox,oy,_=self.map.origin_local_m
        reg=(cx-radius_m-ox,cx+radius_m-ox,cy-radius_m-oy,cy+radius_m-oy,0.0,core.spec.size_z_m)
        bounds=metric_box_to_bounds(core.spec,reg)
        gids,_,refs,scanned=core._select_mode(bounds,self.session_index,index_mode='time')
        if not len(gids):
            self.last_stats={'blocks_scanned':int(scanned),'bucket_refs':int(refs),'active_blocks':0,'returned':0}
            return np.empty(0,dtype=np.int64)
        roots=self.map.hierarchy.block_map.global_block_ids
        pos=np.searchsorted(roots,gids)
        if np.any(pos>=len(roots)) or not np.array_equal(roots[pos],gids):
            raise AssertionError('active BST root missing from hierarchy root table')
        total=int(np.sum(self.map.root_counts[pos],dtype=np.int64)); ids=np.empty(total,dtype=np.int64); q=0
        for j in pos:
            a=int(self.map.root_starts[j]); c=int(self.map.root_counts[j]); ids[q:q+c]=self.map.selected_node_ids[a:a+c]; q+=c
        if len(ids):
            c=self.map.hierarchy.node_centroids(ids)
            ids=ids[(c[:,0]-cx)**2+(c[:,1]-cy)**2<=radius_m*radius_m]
        self.last_stats={'blocks_scanned':int(scanned),'bucket_refs':int(refs),'active_blocks':int(len(gids)),'returned':int(len(ids))}
        return ids

    def prepare_pyramid(self,ids,levels,cfg):
        from types import SimpleNamespace
        from localization.surfel_icp import prepare_surfel_pyramid
        ids=np.asarray(ids,dtype=np.int64); h=self.map.hierarchy
        local=SimpleNamespace(centroids=h.node_centroids(ids),cov6=h.node_cov6(ids),counts=h.node_counts(ids))
        return prepare_surfel_pyramid(local,np.arange(len(ids),dtype=np.int64),levels,cfg)

    def nbytes(self): return self.map.directory_nbytes()


__all__=[
    'NativeSurfelHierarchy','HierarchicalLODView','HierarchicalLODBSTIndex',
    'build_native_surfel_hierarchy','build_hierarchical_lod_view','parse_geometry_profiles','GeometryGateProfile',
]
