from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import SimpleNamespace

import numpy as np

from realdata.nclt_packed import linear_decode, linear_encode
from representations.packed_scale import _block_key, _ceil_log2, metric_box_to_bounds
from representations.step83.time_local_core import _geometry_keys
from localization.step903b import prepare_surfel_pyramid


def _cov6_outer(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.column_stack((x*x, x*y, x*z, y*y, y*z, z*z))


def _merge_cov6_pair(n1, m1, c1, n2, m2, c2):
    n1f = n1.astype(np.float64)
    n2f = n2.astype(np.float64)
    nt = n1f + n2f
    mean = (m1 * n1f[:, None] + m2 * n2f[:, None]) / nt[:, None]
    d1 = m1 - mean
    d2 = m2 - mean
    cov = (
        c1 * n1f[:, None]
        + c2 * n2f[:, None]
        + _cov6_outer(d1) * n1f[:, None]
        + _cov6_outer(d2) * n2f[:, None]
    ) / nt[:, None]
    return nt.astype(np.uint64), mean.astype(np.float32), cov.astype(np.float32)


def _merge_axis_payload(blocks, counts, centroids, cov6, axis, limits, bits):
    if len(blocks) < 2:
        return blocks, counts, centroids, cov6, 0
    keys = _block_key(blocks, bits)
    order = np.argsort(keys, kind='stable')
    sk = keys[order]
    origin = blocks[:, axis]
    size = blocks[:, axis + 3]
    lower = (origin % (2 * size) == 0) & (origin + 2 * size <= limits[axis])
    lower_idx = np.flatnonzero(lower)
    if not len(lower_idx):
        return blocks, counts, centroids, cov6, 0
    shifts = (0, bits[0], bits[0] + bits[1])
    sibkey = keys[lower_idx] + (size[lower_idx].astype(np.uint64) << np.uint64(shifts[axis]))
    pos = np.searchsorted(sk, sibkey)
    valid = pos < len(sk)
    found = np.zeros(len(pos), dtype=bool)
    found[valid] = sk[pos[valid]] == sibkey[valid]
    lower_idx = lower_idx[found]
    pos = pos[found]
    if not len(lower_idx):
        return blocks, counts, centroids, cov6, 0
    sib_idx = order[pos]

    consumed = np.zeros(len(blocks), dtype=bool)
    consumed[lower_idx] = True
    consumed[sib_idx] = True
    parents = blocks[lower_idx].copy()
    parents[:, axis + 3] *= 2
    pc, pm, pv = _merge_cov6_pair(
        counts[lower_idx], centroids[lower_idx], cov6[lower_idx],
        counts[sib_idx], centroids[sib_idx], cov6[sib_idx],
    )
    return (
        np.concatenate((blocks[~consumed], parents), axis=0),
        np.concatenate((counts[~consumed], pc), axis=0),
        np.concatenate((centroids[~consumed], pm), axis=0),
        np.concatenate((cov6[~consumed], pv), axis=0),
        len(lower_idx),
    )


def merge_voxel_surfels_to_blocks(coords, counts, centroids, cov6, limits, progress_label=None):
    """Exact Step-8.4 dyadic merging while propagating surfel sufficient statistics."""
    coords = np.asarray(coords, dtype=np.int32)
    n = len(coords)
    blocks = np.empty((n, 6), dtype=np.int32)
    blocks[:, :3] = coords
    blocks[:, 3:] = 1
    cnt = np.asarray(counts, dtype=np.uint64).copy()
    mu = np.asarray(centroids, dtype=np.float32).copy()
    cv = np.asarray(cov6, dtype=np.float32).copy()
    bits = (_ceil_log2(limits[0]), _ceil_log2(limits[1]), _ceil_log2(limits[2]), 4)
    cycle = 0
    while True:
        cycle += 1
        before = len(blocks)
        merged = 0
        for axis in range(3):
            blocks, cnt, mu, cv, c = _merge_axis_payload(blocks, cnt, mu, cv, axis, limits, bits)
            merged += c
        if progress_label:
            print(f'    [{progress_label}] surfel merge cycle {cycle}: {before:,}->{len(blocks):,} blocks ({merged:,} merges)', flush=True)
        if merged == 0:
            break
    return blocks, cnt, mu, cv


def finalize_normals(cov6, counts, min_points=6, max_curvature=0.20, min_horizontal_component=0.20, batch_size=200_000):
    cv = np.asarray(cov6, dtype=np.float32)
    counts = np.asarray(counts)
    n = len(cv)
    normals = np.zeros((n, 3), dtype=np.float32)
    curvature = np.full(n, np.inf, dtype=np.float32)
    valid = np.zeros(n, dtype=np.uint8)
    idx = np.flatnonzero(counts >= int(min_points))
    for a in range(0, len(idx), int(batch_size)):
        ii = idx[a:a+int(batch_size)]
        c6 = cv[ii].astype(np.float64)
        C = np.empty((len(ii), 3, 3), dtype=np.float64)
        C[:,0,0]=c6[:,0]; C[:,0,1]=c6[:,1]; C[:,0,2]=c6[:,2]
        C[:,1,0]=c6[:,1]; C[:,1,1]=c6[:,3]; C[:,1,2]=c6[:,4]
        C[:,2,0]=c6[:,2]; C[:,2,1]=c6[:,4]; C[:,2,2]=c6[:,5]
        ev, evec = np.linalg.eigh(C)
        total = np.sum(ev, axis=1)
        curv = np.divide(np.maximum(ev[:,0], 0.0), total, out=np.full(len(ii), np.inf), where=total > 1e-12)
        nn = evec[:,:,0]
        h = np.hypot(nn[:,0], nn[:,1])
        good = np.isfinite(curv) & (curv <= float(max_curvature)) & (h >= float(min_horizontal_component))
        normals[ii[good]] = nn[good].astype(np.float32)
        curvature[ii] = curv.astype(np.float32)
        valid[ii[good]] = 1
    return normals, curvature, valid


def align_voxel_surfels_to_prepared(surfel_map, prepared, session_index: int):
    """Re-express one-session surfels on the common Step-8 prepared lattice."""
    if abs(surfel_map.spec.voxel_size_m - prepared.spec.voxel_size_m) > 1e-9:
        raise ValueError('surfel and prepared voxel sizes differ')
    vs = prepared.spec.voxel_size_m
    x, y, z = linear_decode(surfel_map.linear_codes, surfel_map.spec)
    delta = np.rint((np.asarray(surfel_map.origin_local_m) - np.asarray(prepared.origin_local_m)) / vs).astype(np.int64)
    gx = x.astype(np.int64) + delta[0]
    gy = y.astype(np.int64) + delta[1]
    gz = z.astype(np.int64) + delta[2]
    if np.any(gx < 0) or np.any(gx >= prepared.spec.nx) or np.any(gy < 0) or np.any(gy >= prepared.spec.ny) or np.any(gz < 0) or np.any(gz >= prepared.spec.nz):
        raise AssertionError('aligned surfel voxel lies outside common prepared grid')
    codes = linear_encode(gx, gy, gz, prepared.spec)
    order = np.argsort(codes, kind='stable')
    codes = codes[order]
    expected = np.asarray(prepared.session_codes[int(session_index)], dtype=np.uint64)
    if not np.array_equal(codes, expected):
        raise AssertionError(f'surfel occupancy does not exactly match prepared session {session_index}: {len(codes)} vs {len(expected)}')
    return SimpleNamespace(
        spec=prepared.spec,
        origin_local_m=prepared.origin_local_m,
        linear_codes=codes,
        counts=np.asarray(surfel_map.counts)[order],
        centroids=np.asarray(surfel_map.centroids)[order],
        cov6=np.asarray(surfel_map.cov6)[order],
        normals=np.asarray(surfel_map.normals)[order],
        curvature=np.asarray(surfel_map.curvature)[order],
        normal_valid=np.asarray(surfel_map.normal_valid)[order],
    )


@dataclass
class NativeBlockSurfelMap:
    spec: object
    origin_local_m: tuple[float,float,float]
    global_block_ids: np.ndarray
    blocks: np.ndarray
    counts: np.ndarray
    centroids: np.ndarray
    cov6: np.ndarray
    normals: np.ndarray
    curvature: np.ndarray
    normal_valid: np.ndarray
    bst_map: object
    session_index: int
    build_ms: float

    def payload_nbytes(self):
        return int(sum(a.nbytes for a in (self.counts,self.centroids,self.cov6,self.normals,self.curvature,self.normal_valid)))

    def minimal_localization_payload_nbytes(self):
        return int(sum(a.nbytes for a in (self.counts,self.centroids,self.normals,self.curvature,self.normal_valid)))


class NativeBlockSurfelIndex:
    """Query Step-8.5 active block IDs, then expose one surfel per native block."""
    def __init__(self, block_map: NativeBlockSurfelMap):
        self.map = block_map
        self.last_stats = {}

    def query_radius_ids(self, cx: float, cy: float, radius_m: float) -> np.ndarray:
        core = self.map.bst_map.core
        ox, oy, _ = self.map.origin_local_m
        reg = (
            cx-radius_m-ox, cx+radius_m-ox,
            cy-radius_m-oy, cy+radius_m-oy,
            0.0, core.spec.size_z_m,
        )
        bounds = metric_box_to_bounds(core.spec, reg)
        ids, _, refs, scanned = core._select_mode(bounds, self.map.session_index, index_mode='time')
        if not len(ids):
            self.last_stats={'blocks_scanned':int(scanned),'bucket_refs':int(refs),'active_blocks':0,'returned':0}
            return np.empty(0,dtype=np.int64)
        pos = np.searchsorted(self.map.global_block_ids, ids)
        good = (pos < len(self.map.global_block_ids)) & (self.map.global_block_ids[np.minimum(pos,len(self.map.global_block_ids)-1)] == ids)
        if not np.all(good):
            raise AssertionError('time-index returned active map-session block without a surfel payload')
        rows = pos.astype(np.int64, copy=False)
        c = self.map.centroids[rows]
        keep = (c[:,0]-cx)**2 + (c[:,1]-cy)**2 <= radius_m*radius_m
        rows = rows[keep]
        self.last_stats={'blocks_scanned':int(scanned),'bucket_refs':int(refs),'active_blocks':int(len(ids)),'returned':int(len(rows))}
        return rows

    def nbytes(self):
        return int(self.map.global_block_ids.nbytes)


class BSTDecompressedSurfelIndex:
    """Control: BST query -> exact voxel decompression -> same voxel surfel payload."""
    def __init__(self, bst_map, aligned_surfel_map, session_index: int):
        self.bst_map = bst_map
        self.map = aligned_surfel_map
        self.session_index = int(session_index)
        self.codes = np.asarray(aligned_surfel_map.linear_codes,dtype=np.uint64)
        self.last_stats={}

    def query_radius_ids(self,cx:float,cy:float,radius_m:float)->np.ndarray:
        ox,oy,_=self.map.origin_local_m
        core=self.bst_map.core
        reg=(cx-radius_m-ox,cx+radius_m-ox,cy-radius_m-oy,cy+radius_m-oy,0.0,core.spec.size_z_m)
        codes=self.bst_map.region_query(reg,self.session_index,output='voxels')
        if not len(codes):
            self.last_stats=dict(core.last_query_stats)
            return np.empty(0,dtype=np.int64)
        pos=np.searchsorted(self.codes,codes)
        good=(pos<len(self.codes)) & (self.codes[np.minimum(pos,len(self.codes)-1)]==codes)
        if not np.all(good):
            raise AssertionError('decompressed BST voxel not present in map-session surfel table')
        rows=pos.astype(np.int64,copy=False)
        c=self.map.centroids[rows]
        rows=rows[(c[:,0]-cx)**2+(c[:,1]-cy)**2<=radius_m*radius_m]
        self.last_stats=dict(core.last_query_stats)
        self.last_stats['returned']=int(len(rows))
        return rows


def build_native_block_surfel_map(aligned_surfel_map, prepared, session_blocks, core, bst_map, session_index=0, normal_min_points=6, normal_max_curvature=0.20, normal_min_horizontal_component=0.20, progress=True):
    t0=time.perf_counter_ns()
    t=int(session_index)
    codes=np.asarray(aligned_surfel_map.linear_codes,dtype=np.uint64)
    x,y,z=linear_decode(codes,prepared.spec)
    coords=np.column_stack((x,y,z)).astype(np.int32,copy=False)
    label=f'map session {t} vox={len(coords):,}' if progress else None
    blocks,cnt,mu,cv=merge_voxel_surfels_to_blocks(
        coords,aligned_surfel_map.counts,aligned_surfel_map.centroids,aligned_surfel_map.cov6,
        (prepared.spec.nx,prepared.spec.ny,prepared.spec.nz),label,
    )
    expected=np.asarray(session_blocks[t],dtype=np.int32)
    k1=np.sort(_geometry_keys(blocks,(prepared.spec.nx,prepared.spec.ny,prepared.spec.nz)))
    k2=np.sort(_geometry_keys(expected,(prepared.spec.nx,prepared.spec.ny,prepared.spec.nz)))
    if not np.array_equal(k1,k2):
        raise AssertionError('surfel-propagating merge did not reproduce the exact Step-8.4 session tiling')

    voxel_equivalent = int(np.prod(blocks[:,3:6].astype(np.int64),axis=1).sum()) if len(blocks) else 0
    if voxel_equivalent != len(codes):
        raise AssertionError(f'native block tiling voxel-equivalent mismatch: {voxel_equivalent} vs {len(codes)}')
    normals,curvature,valid=finalize_normals(cv,cnt,normal_min_points,normal_max_curvature,normal_min_horizontal_component)
    global_keys=_geometry_keys(core.blocks[:,:6],(prepared.spec.nx,prepared.spec.ny,prepared.spec.nz))
    local_keys=_geometry_keys(blocks,(prepared.spec.nx,prepared.spec.ny,prepared.spec.nz))
    gids=np.searchsorted(global_keys,local_keys)
    if len(gids) and (np.any(gids>=len(global_keys)) or not np.array_equal(global_keys[gids],local_keys)):
        raise AssertionError('map-session block geometry not found in the finalized time-local map')
    order=np.argsort(gids,kind='stable')
    return NativeBlockSurfelMap(
        spec=prepared.spec,origin_local_m=prepared.origin_local_m,
        global_block_ids=gids[order].astype(np.int32),blocks=blocks[order],counts=cnt[order].astype(np.uint64),
        centroids=mu[order],cov6=cv[order],normals=normals[order],curvature=curvature[order],normal_valid=valid[order],
        bst_map=bst_map,session_index=t,build_ms=(time.perf_counter_ns()-t0)/1e6,
    )


__all__=[
    'merge_voxel_surfels_to_blocks','align_voxel_surfels_to_prepared','finalize_normals',
    'NativeBlockSurfelMap','NativeBlockSurfelIndex','BSTDecompressedSurfelIndex','build_native_block_surfel_map',
]
