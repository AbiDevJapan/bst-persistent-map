from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from localization.step91 import (
    NativeBlockSurfelMap,
    _merge_cov6_pair,
    _cov6_outer,
    finalize_normals,
)
from realdata.nclt_packed import linear_decode
from representations.packed_scale import _block_key, _ceil_log2, metric_box_to_bounds
from representations.step83.time_local_core import _geometry_keys


def _cov6_to_matrix(c6: np.ndarray) -> np.ndarray:
    a = np.asarray(c6, dtype=np.float64)
    C = np.empty((len(a), 3, 3), dtype=np.float64)
    C[:, 0, 0] = a[:, 0]; C[:, 0, 1] = a[:, 1]; C[:, 0, 2] = a[:, 2]
    C[:, 1, 0] = a[:, 1]; C[:, 1, 1] = a[:, 3]; C[:, 1, 2] = a[:, 4]
    C[:, 2, 0] = a[:, 2]; C[:, 2, 1] = a[:, 4]; C[:, 2, 2] = a[:, 5]
    return C


def _normal_outer6(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.column_stack((x*x, x*y, x*z, y*y, y*z, z*z))


@dataclass(frozen=True)
class GeometryGateProfile:
    name: str
    max_plane_sigma_m: float
    max_relative_planarity: float
    min_normal_coherence: float
    max_extent_m: float


DEFAULT_GEOMETRY_PROFILES = (
    GeometryGateProfile('strict',       0.025, 0.010, 0.985, 1.0),
    GeometryGateProfile('balanced',     0.050, 0.025, 0.950, 2.0),
    GeometryGateProfile('moderate',     0.080, 0.050, 0.900, 4.0),
    GeometryGateProfile('permissive',   0.120, 0.080, 0.800, 8.0),
)


def parse_geometry_profiles(text: str | None):
    """Parse name:sigma_m:relative:min_coherence:max_extent_m separated by ';'."""
    if text is None or not str(text).strip():
        return list(DEFAULT_GEOMETRY_PROFILES)
    out = []
    for raw in str(text).split(';'):
        raw = raw.strip()
        if not raw:
            continue
        p = raw.split(':')
        if len(p) != 5:
            raise ValueError(
                'geometry profile must be name:max_plane_sigma_m:max_relative_planarity:'
                'min_normal_coherence:max_extent_m'
            )
        name = p[0].strip()
        if not name:
            raise ValueError('geometry profile name is empty')
        prof = GeometryGateProfile(name, *(float(x) for x in p[1:]))
        if prof.max_plane_sigma_m <= 0 or prof.max_relative_planarity <= 0:
            raise ValueError('plane thresholds must be positive')
        if not (0 <= prof.min_normal_coherence <= 1):
            raise ValueError('normal coherence must be in [0,1]')
        if prof.max_extent_m <= 0:
            raise ValueError('max extent must be positive')
        out.append(prof)
    if not out:
        raise ValueError('no geometry profiles parsed')
    names = [x.name for x in out]
    if len(set(names)) != len(names):
        raise ValueError('geometry profile names must be unique')
    return out


def _merge_axis_with_nodes(blocks, counts, centroids, cov6, node_ids, parent, next_node, axis, limits, bits):
    """Exact occupancy merge plus a compact leaf->final-root merge forest."""
    if len(blocks) < 2:
        return blocks, counts, centroids, cov6, node_ids, next_node, 0
    keys = _block_key(blocks, bits)
    order = np.argsort(keys, kind='stable')
    sk = keys[order]
    origin = blocks[:, axis]
    size = blocks[:, axis + 3]
    lower = (origin % (2 * size) == 0) & (origin + 2 * size <= limits[axis])
    lower_idx = np.flatnonzero(lower)
    if not len(lower_idx):
        return blocks, counts, centroids, cov6, node_ids, next_node, 0
    shifts = (0, bits[0], bits[0] + bits[1])
    sibkey = keys[lower_idx] + (size[lower_idx].astype(np.uint64) << np.uint64(shifts[axis]))
    pos = np.searchsorted(sk, sibkey)
    valid = pos < len(sk)
    found = np.zeros(len(pos), dtype=bool)
    found[valid] = sk[pos[valid]] == sibkey[valid]
    lower_idx = lower_idx[found]
    pos = pos[found]
    if not len(lower_idx):
        return blocks, counts, centroids, cov6, node_ids, next_node, 0
    sib_idx = order[pos]

    k = len(lower_idx)
    parent_ids = np.arange(next_node, next_node + k, dtype=np.int32)
    parent[node_ids[lower_idx]] = parent_ids
    parent[node_ids[sib_idx]] = parent_ids

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
        np.concatenate((node_ids[~consumed], parent_ids), axis=0),
        next_node + k,
        k,
    )


def build_native_block_surfel_map_with_leaf_owners(
    aligned_surfel_map, prepared, session_blocks, core, bst_map, session_index=0,
    normal_min_points=6, normal_max_curvature=0.20,
    normal_min_horizontal_component=0.20, progress=True,
):
    """Reproduce Step-8.4 map-session blocks and map every base voxel to its native parent.

    The extra leaf->parent mapping is built once offline. It lets 9.1.1 switch a
    native block between one coarse surfel and its already-computed 0.5 m voxel
    surfels without decompression inside the localization loop.
    """
    t0 = time.perf_counter_ns()
    t = int(session_index)
    codes = np.asarray(aligned_surfel_map.linear_codes, dtype=np.uint64)
    x, y, z = linear_decode(codes, prepared.spec)
    coords = np.column_stack((x, y, z)).astype(np.int32, copy=False)
    n = len(coords)
    limits = (prepared.spec.nx, prepared.spec.ny, prepared.spec.nz)
    bits = (_ceil_log2(limits[0]), _ceil_log2(limits[1]), _ceil_log2(limits[2]), 4)

    blocks = np.empty((n, 6), dtype=np.int32)
    blocks[:, :3] = coords
    blocks[:, 3:] = 1
    cnt = np.asarray(aligned_surfel_map.counts, dtype=np.uint64).copy()
    mu = np.asarray(aligned_surfel_map.centroids, dtype=np.float32).copy()
    cv = np.asarray(aligned_surfel_map.cov6, dtype=np.float32).copy()
    node_ids = np.arange(n, dtype=np.int32)
    # A binary merge forest over n leaves has fewer than 2*n nodes.
    parent = np.full(max(1, 2*n), -1, dtype=np.int32)
    next_node = n

    cycle = 0
    while True:
        cycle += 1
        before = len(blocks)
        merged = 0
        for axis in range(3):
            blocks, cnt, mu, cv, node_ids, next_node, c = _merge_axis_with_nodes(
                blocks, cnt, mu, cv, node_ids, parent, next_node,
                axis, limits, bits,
            )
            merged += c
        if progress:
            print(f'    [9.1.1 owners] merge cycle {cycle}: {before:,}->{len(blocks):,} blocks ({merged:,} merges)', flush=True)
        if merged == 0:
            break

    expected = np.asarray(session_blocks[t], dtype=np.int32)
    k1 = np.sort(_geometry_keys(blocks, limits))
    k2 = np.sort(_geometry_keys(expected, limits))
    if not np.array_equal(k1, k2):
        raise AssertionError('leaf-owner merge did not reproduce the exact Step-8.4 map-session tiling')

    normals, curvature, valid = finalize_normals(
        cv, cnt, normal_min_points, normal_max_curvature,
        normal_min_horizontal_component,
    )
    global_keys = _geometry_keys(core.blocks[:, :6], limits)
    local_keys = _geometry_keys(blocks, limits)
    gids = np.searchsorted(global_keys, local_keys)
    if len(gids) and (np.any(gids >= len(global_keys)) or not np.array_equal(global_keys[gids], local_keys)):
        raise AssertionError('map-session block geometry not found in finalized time-local map')

    # Resolve every original voxel leaf to its final native block root.
    roots = np.arange(n, dtype=np.int32)
    for _ in range(64):
        p = parent[roots]
        m = p >= 0
        if not np.any(m):
            break
        roots[m] = p[m]
    else:  # pragma: no cover - dyadic grids here are far shallower than this.
        raise AssertionError('unexpectedly deep merge forest')

    node_order = np.argsort(node_ids, kind='stable')
    snodes = node_ids[node_order]
    pos = np.searchsorted(snodes, roots)
    if np.any(pos >= len(snodes)) or not np.array_equal(snodes[pos], roots):
        raise AssertionError('could not resolve leaf to final native block root')
    leaf_parent_gids = gids[node_order[pos]].astype(np.int32, copy=False)

    order = np.argsort(gids, kind='stable')
    block_map = NativeBlockSurfelMap(
        spec=prepared.spec,
        origin_local_m=prepared.origin_local_m,
        global_block_ids=gids[order].astype(np.int32),
        blocks=blocks[order],
        counts=cnt[order].astype(np.uint64),
        centroids=mu[order],
        cov6=cv[order],
        normals=normals[order],
        curvature=curvature[order],
        normal_valid=valid[order],
        bst_map=bst_map,
        session_index=t,
        build_ms=(time.perf_counter_ns() - t0) / 1e6,
    )
    del parent
    return block_map, leaf_parent_gids


@dataclass
class NativeGeometryDescriptors:
    plane_sigma_m: np.ndarray
    relative_planarity: np.ndarray
    normal_coherence: np.ndarray
    normal_votes: np.ndarray
    max_extent_m: np.ndarray
    voxel_volume: np.ndarray
    leaf_order: np.ndarray
    parent_keys: np.ndarray
    parent_starts: np.ndarray
    parent_counts: np.ndarray
    build_ms: float

    def index_nbytes(self):
        return int(sum(a.nbytes for a in (self.leaf_order, self.parent_keys, self.parent_starts, self.parent_counts)))


def build_native_geometry_descriptors(aligned_surfel_map, block_map: NativeBlockSurfelMap, leaf_parent_gids: np.ndarray, batch_size=200_000):
    """Precompute all geometry-gate signals once; profile sweeps are then O(blocks)."""
    t0 = time.perf_counter_ns()
    n = len(block_map.blocks)
    sigma = np.full(n, np.inf, dtype=np.float32)
    rel = np.full(n, np.inf, dtype=np.float32)
    for a in range(0, n, int(batch_size)):
        b = min(n, a + int(batch_size))
        C = _cov6_to_matrix(block_map.cov6[a:b])
        ev = np.linalg.eigvalsh(C)
        lam0 = np.maximum(ev[:, 0], 0.0)
        tr = np.sum(ev, axis=1)
        sigma[a:b] = np.sqrt(lam0).astype(np.float32)
        rel[a:b] = np.divide(lam0, tr, out=np.full(len(ev), np.inf), where=tr > 1e-12).astype(np.float32)

    # Group base voxel rows by native parent once. This is also the CSR used by
    # every two-level refinement profile.
    pg = np.asarray(leaf_parent_gids, dtype=np.int32)
    leaf_order = np.argsort(pg, kind='stable').astype(np.uint32 if len(pg) < 2**32 else np.uint64)
    spg = pg[leaf_order]
    parent_keys, starts, counts = np.unique(spg, return_index=True, return_counts=True)
    starts = starts.astype(np.uint32 if len(pg) < 2**32 else np.uint64)
    counts = counts.astype(np.uint32)
    if not np.array_equal(parent_keys, block_map.global_block_ids):
        raise AssertionError('leaf-parent directory does not cover exactly the map-session native block IDs')

    normals = np.asarray(aligned_surfel_map.normals, dtype=np.float32)[leaf_order]
    valid = np.asarray(aligned_surfel_map.normal_valid, dtype=np.uint8)[leaf_order].astype(bool)
    outer = _normal_outer6(normals)
    outer[~valid] = 0.0
    votes = np.add.reduceat(valid.astype(np.uint32), starts.astype(np.int64)).astype(np.uint32)
    sums = np.add.reduceat(outer, starts.astype(np.int64), axis=0)
    coherence = np.ones(n, dtype=np.float32)
    idx = np.flatnonzero(votes >= 2)
    for a in range(0, len(idx), int(batch_size)):
        ii = idx[a:a+int(batch_size)]
        C = _cov6_to_matrix(sums[ii])
        ev = np.linalg.eigvalsh(C)
        coherence[ii] = (ev[:, -1] / votes[ii].astype(np.float64)).astype(np.float32)

    ext = np.max(block_map.blocks[:, 3:6], axis=1).astype(np.float32) * float(block_map.spec.voxel_size_m)
    volume = np.prod(block_map.blocks[:, 3:6].astype(np.int64), axis=1).astype(np.int32)
    return NativeGeometryDescriptors(
        plane_sigma_m=sigma,
        relative_planarity=rel,
        normal_coherence=coherence,
        normal_votes=votes,
        max_extent_m=ext,
        voxel_volume=volume,
        leaf_order=leaf_order,
        parent_keys=parent_keys.astype(np.int32),
        parent_starts=starts,
        parent_counts=counts,
        build_ms=(time.perf_counter_ns() - t0) / 1e6,
    )


def profile_keep_native_mask(profile: GeometryGateProfile, desc: NativeGeometryDescriptors) -> np.ndarray:
    """Return true when one native block surfel is retained; false => 0.5 m fallback."""
    # Single voxels cannot be refined further and are always kept.
    leaf = desc.voxel_volume <= 1
    coherence_ok = (desc.normal_votes < 2) | (desc.normal_coherence >= float(profile.min_normal_coherence))
    good = (
        np.isfinite(desc.plane_sigma_m)
        & np.isfinite(desc.relative_planarity)
        & (desc.plane_sigma_m <= float(profile.max_plane_sigma_m))
        & (desc.relative_planarity <= float(profile.max_relative_planarity))
        & coherence_ok
        & (desc.max_extent_m <= float(profile.max_extent_m))
    )
    return leaf | good


@dataclass
class GeometryAwareSurfelMap:
    spec: object
    origin_local_m: tuple[float, float, float]
    parent_global_ids: np.ndarray
    counts: np.ndarray
    centroids: np.ndarray
    cov6: np.ndarray
    normals: np.ndarray
    curvature: np.ndarray
    normal_valid: np.ndarray
    profile: GeometryGateProfile
    kept_native_blocks: int
    fallback_parent_blocks: int
    fallback_voxels: int
    build_ms: float

    def payload_nbytes(self):
        return int(sum(a.nbytes for a in (
            self.parent_global_ids, self.counts, self.centroids, self.cov6,
            self.normals, self.curvature, self.normal_valid,
        )))

    def minimal_localization_payload_nbytes(self):
        return int(sum(a.nbytes for a in (
            self.parent_global_ids, self.counts, self.centroids,
            self.normals, self.curvature, self.normal_valid,
        )))


class GeometryAwareBSTSurfelIndex:
    """Step-8.5 time index -> precomputed coarse-or-voxel localization primitives."""
    def __init__(self, surfel_map: GeometryAwareSurfelMap, bst_map, session_index: int):
        self.map = surfel_map
        self.bst_map = bst_map
        self.session_index = int(session_index)
        p = np.asarray(surfel_map.parent_global_ids, dtype=np.int32)
        if len(p) and np.any(p[1:] < p[:-1]):
            raise ValueError('geometry-aware parent ids must be sorted')
        self.parent_keys, starts, counts = np.unique(p, return_index=True, return_counts=True)
        self.parent_starts = starts.astype(np.uint32 if len(p) < 2**32 else np.uint64)
        self.parent_counts = counts.astype(np.uint32)
        self.last_stats = {}

    def nbytes(self):
        return int(sum(a.nbytes for a in (self.parent_keys, self.parent_starts, self.parent_counts)))

    def query_radius_ids(self, cx: float, cy: float, radius_m: float) -> np.ndarray:
        core = self.bst_map.core
        ox, oy, _ = self.map.origin_local_m
        reg = (
            cx-radius_m-ox, cx+radius_m-ox,
            cy-radius_m-oy, cy+radius_m-oy,
            0.0, core.spec.size_z_m,
        )
        bounds = metric_box_to_bounds(core.spec, reg)
        ids, _, refs, scanned = core._select_mode(bounds, self.session_index, index_mode='time')
        if not len(ids):
            self.last_stats = {'blocks_scanned': int(scanned), 'bucket_refs': int(refs), 'active_blocks': 0, 'returned': 0}
            return np.empty(0, dtype=np.int64)
        pos = np.searchsorted(self.parent_keys, ids)
        good = (pos < len(self.parent_keys)) & (self.parent_keys[np.minimum(pos, len(self.parent_keys)-1)] == ids)
        if not np.all(good):
            raise AssertionError('time-index returned active block missing from geometry-aware localization overlay')
        pos = pos.astype(np.int64, copy=False)
        total = int(np.sum(self.parent_counts[pos], dtype=np.int64))
        rows = np.empty(total, dtype=np.int64)
        q = 0
        for j in pos:
            a = int(self.parent_starts[j]); c = int(self.parent_counts[j])
            rows[q:q+c] = np.arange(a, a+c, dtype=np.int64)
            q += c
        c = self.map.centroids[rows]
        rows = rows[(c[:, 0]-cx)**2 + (c[:, 1]-cy)**2 <= radius_m*radius_m]
        self.last_stats = {
            'blocks_scanned': int(scanned), 'bucket_refs': int(refs),
            'active_blocks': int(len(ids)), 'returned': int(len(rows)),
        }
        return rows


def build_geometry_aware_surfel_map(
    aligned_surfel_map,
    block_map: NativeBlockSurfelMap,
    leaf_parent_gids: np.ndarray,
    desc: NativeGeometryDescriptors,
    profile: GeometryGateProfile,
):
    """Build one Pareto point without changing the underlying lifelong BST map.

    A native block either remains one surfel or falls back to the exact base
    voxel surfels belonging to that same native parent. There is no online
    geometry decision and no online block decompression.
    """
    t0 = time.perf_counter_ns()
    keep = profile_keep_native_mask(profile, desc)
    if len(keep) != len(block_map.blocks):
        raise AssertionError('profile mask size mismatch')

    coarse_rows = np.flatnonzero(keep)
    reject_parent_gids = block_map.global_block_ids[~keep]
    # leaf_parent_gids are aligned with the original voxel-surfel row order.
    # searchsorted is valid because block global IDs are sorted in block_map.
    parent_pos = np.searchsorted(block_map.global_block_ids, leaf_parent_gids)
    if np.any(parent_pos >= len(block_map.global_block_ids)) or not np.array_equal(block_map.global_block_ids[parent_pos], leaf_parent_gids):
        raise AssertionError('leaf parent global IDs are inconsistent with native block table')
    leaf_keep = ~keep[parent_pos]
    leaf_rows = np.flatnonzero(leaf_keep)

    parent_ids = np.concatenate((block_map.global_block_ids[coarse_rows], leaf_parent_gids[leaf_rows])).astype(np.int32, copy=False)
    counts = np.concatenate((block_map.counts[coarse_rows], np.asarray(aligned_surfel_map.counts)[leaf_rows])).astype(np.uint64, copy=False)
    centroids = np.concatenate((block_map.centroids[coarse_rows], np.asarray(aligned_surfel_map.centroids)[leaf_rows]), axis=0).astype(np.float32, copy=False)
    cov6 = np.concatenate((block_map.cov6[coarse_rows], np.asarray(aligned_surfel_map.cov6)[leaf_rows]), axis=0).astype(np.float32, copy=False)
    normals = np.concatenate((block_map.normals[coarse_rows], np.asarray(aligned_surfel_map.normals)[leaf_rows]), axis=0).astype(np.float32, copy=False)
    curvature = np.concatenate((block_map.curvature[coarse_rows], np.asarray(aligned_surfel_map.curvature)[leaf_rows])).astype(np.float32, copy=False)
    normal_valid = np.concatenate((block_map.normal_valid[coarse_rows], np.asarray(aligned_surfel_map.normal_valid)[leaf_rows])).astype(np.uint8, copy=False)

    order = np.argsort(parent_ids, kind='stable')
    parent_ids = parent_ids[order]
    counts = counts[order]; centroids = centroids[order]; cov6 = cov6[order]
    normals = normals[order]; curvature = curvature[order]; normal_valid = normal_valid[order]

    # Exact coverage check: each kept native block contributes its volume; each
    # rejected block contributes one primitive per constituent voxel.
    represented = int(desc.voxel_volume[keep].sum(dtype=np.int64) + np.count_nonzero(leaf_keep))
    voxel_total = int(desc.voxel_volume.sum(dtype=np.int64))
    if represented != voxel_total:
        raise AssertionError(f'geometry-aware coverage mismatch: {represented} vs {voxel_total}')

    return GeometryAwareSurfelMap(
        spec=aligned_surfel_map.spec,
        origin_local_m=aligned_surfel_map.origin_local_m,
        parent_global_ids=parent_ids,
        counts=counts,
        centroids=centroids,
        cov6=cov6,
        normals=normals,
        curvature=curvature,
        normal_valid=normal_valid,
        profile=profile,
        kept_native_blocks=int(np.count_nonzero(keep)),
        fallback_parent_blocks=int(np.count_nonzero(~keep)),
        fallback_voxels=int(len(leaf_rows)),
        build_ms=(time.perf_counter_ns() - t0) / 1e6,
    ), keep


__all__ = [
    'GeometryGateProfile', 'DEFAULT_GEOMETRY_PROFILES', 'parse_geometry_profiles',
    'build_native_block_surfel_map_with_leaf_owners',
    'NativeGeometryDescriptors', 'build_native_geometry_descriptors',
    'profile_keep_native_mask', 'GeometryAwareSurfelMap',
    'GeometryAwareBSTSurfelIndex', 'build_geometry_aware_surfel_map',
]

@dataclass
class GeometryAwareSurfelView:
    """Zero-copy two-level localization view used for profile sweeps.

    The benchmark keeps the full voxel and native surfel tables resident once,
    then each profile is only a boolean native/fallback decision mask. Reported
    standalone payload bytes are computed as if the selected primitives alone
    were serialized for deployment.
    """
    aligned_map: object
    block_map: NativeBlockSurfelMap
    desc: NativeGeometryDescriptors
    profile: GeometryGateProfile
    keep_native: np.ndarray

    @property
    def spec(self):
        return self.aligned_map.spec

    @property
    def origin_local_m(self):
        return self.aligned_map.origin_local_m

    @property
    def map_elements(self):
        keep = self.keep_native
        fallback_voxels = int(self.desc.voxel_volume[~keep].sum(dtype=np.int64))
        return int(np.count_nonzero(keep) + fallback_voxels)

    @property
    def fallback_parent_blocks(self):
        return int(np.count_nonzero(~self.keep_native))

    @property
    def kept_native_blocks(self):
        return int(np.count_nonzero(self.keep_native))

    @property
    def fallback_voxels(self):
        return int(self.desc.voxel_volume[~self.keep_native].sum(dtype=np.int64))

    def selected_payload_nbytes(self, minimal=False):
        # Per selected primitive payload mirrors the Step-9.1 surfel arrays.
        # count:uint64 + centroid:3*f32 + cov6:6*f32 + normal:3*f32 + curvature:f32 + valid:u8
        per_full = 8 + 12 + 24 + 12 + 4 + 1
        per_min = 8 + 12 + 12 + 4 + 1
        return int(self.map_elements * (per_min if minimal else per_full))

    def standalone_refinement_directory_nbytes(self):
        # If selected primitives are serialized grouped by coarse BST parent,
        # only fallback parents need a (gid,start,count) directory entry.
        return int(self.fallback_parent_blocks * (4 + 4 + 4) + len(self.keep_native))


class GeometryAwareBSTVirtualIndex:
    """Same Step-8.5 temporal query, then choose one native surfel or base voxels.

    IDs returned to the matcher are virtual: [0,B) are native block rows and
    [B,B+V) are original 0.5 m voxel-surfel rows. This avoids duplicating a
    several-hundred-MiB payload for every threshold profile during the sweep.
    """
    def __init__(self, view: GeometryAwareSurfelView, bst_map, session_index: int):
        self.map = view
        self.bst_map = bst_map
        self.session_index = int(session_index)
        self.B = len(view.block_map.blocks)
        self.last_stats = {}

    def query_radius_ids(self, cx: float, cy: float, radius_m: float) -> np.ndarray:
        core = self.bst_map.core
        ox, oy, _ = self.map.origin_local_m
        reg = (
            cx-radius_m-ox, cx+radius_m-ox,
            cy-radius_m-oy, cy+radius_m-oy,
            0.0, core.spec.size_z_m,
        )
        bounds = metric_box_to_bounds(core.spec, reg)
        gids, _, refs, scanned = core._select_mode(bounds, self.session_index, index_mode='time')
        if not len(gids):
            self.last_stats = {'blocks_scanned': int(scanned), 'bucket_refs': int(refs), 'active_blocks': 0, 'returned': 0}
            return np.empty(0, dtype=np.int64)
        bpos = np.searchsorted(self.map.block_map.global_block_ids, gids)
        if np.any(bpos >= self.B) or not np.array_equal(self.map.block_map.global_block_ids[bpos], gids):
            raise AssertionError('active BST block missing from geometry-aware map-session block table')
        bpos = bpos.astype(np.int64, copy=False)
        native = bpos[self.map.keep_native[bpos]]
        fallback_pos = bpos[~self.map.keep_native[bpos]]

        parts = []
        if len(native):
            parts.append(native.astype(np.int64, copy=False))
        for j in fallback_pos:
            a = int(self.map.desc.parent_starts[j]); c = int(self.map.desc.parent_counts[j])
            leaf_rows = self.map.desc.leaf_order[a:a+c].astype(np.int64, copy=False)
            parts.append(self.B + leaf_rows)
        if not parts:
            ids = np.empty(0, dtype=np.int64)
        elif len(parts) == 1:
            ids = parts[0]
        else:
            ids = np.concatenate(parts)

        if len(ids):
            c = self._centroids(ids)
            ids = ids[(c[:, 0]-cx)**2 + (c[:, 1]-cy)**2 <= radius_m*radius_m]
        self.last_stats = {
            'blocks_scanned': int(scanned), 'bucket_refs': int(refs),
            'active_blocks': int(len(gids)), 'native_parent_blocks': int(len(native)),
            'fallback_parent_blocks': int(len(fallback_pos)), 'returned': int(len(ids)),
        }
        return ids

    def _decode(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        is_block = ids < self.B
        return is_block, ids[is_block], ids[~is_block] - self.B

    def _centroids(self, ids):
        is_block, b, v = self._decode(ids)
        out = np.empty((len(ids), 3), dtype=np.float32)
        out[is_block] = self.map.block_map.centroids[b]
        out[~is_block] = np.asarray(self.map.aligned_map.centroids)[v]
        return out

    def prepare_pyramid(self, ids, levels, cfg):
        from types import SimpleNamespace
        from localization.step903b import prepare_surfel_pyramid
        ids = np.asarray(ids, dtype=np.int64)
        is_block, b, v = self._decode(ids)
        c = np.empty((len(ids), 3), dtype=np.float32)
        cv = np.empty((len(ids), 6), dtype=np.float32)
        w = np.empty(len(ids), dtype=np.uint64)
        c[is_block] = self.map.block_map.centroids[b]
        cv[is_block] = self.map.block_map.cov6[b]
        w[is_block] = self.map.block_map.counts[b]
        c[~is_block] = np.asarray(self.map.aligned_map.centroids)[v]
        cv[~is_block] = np.asarray(self.map.aligned_map.cov6)[v]
        w[~is_block] = np.asarray(self.map.aligned_map.counts)[v]
        local = SimpleNamespace(centroids=c, cov6=cv, counts=w)
        return prepare_surfel_pyramid(local, np.arange(len(ids), dtype=np.int64), levels, cfg)

    def nbytes(self):
        return int(self.map.keep_native.nbytes)


def build_geometry_aware_virtual_view(aligned_surfel_map, block_map, desc, profile):
    t0 = time.perf_counter_ns()
    keep = profile_keep_native_mask(profile, desc)
    view = GeometryAwareSurfelView(aligned_surfel_map, block_map, desc, profile, keep)
    return view, (time.perf_counter_ns() - t0) / 1e6


__all__ += [
    'GeometryAwareSurfelView', 'GeometryAwareBSTVirtualIndex',
    'build_geometry_aware_virtual_view',
]
