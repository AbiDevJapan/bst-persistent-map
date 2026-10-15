from __future__ import annotations

import math
import time
import numpy as np

from representations.packed_scale import (
    temporal_signature_bytes, merge_coords_fast, metric_box_to_bounds,
)
from realdata.nclt_packed import linear_decode
from .adaptive_core import AdaptiveBlockCore


def _ceil_log2(n: int) -> int:
    return max(1, math.ceil(math.log2(max(1, n))))


def _geometry_keys(blocks: np.ndarray, limits) -> np.ndarray:
    """Pack aligned dyadic [x,y,z,sx,sy,sz] geometry into a uint64 key."""
    if not len(blocks):
        return np.empty(0, dtype=np.uint64)
    xb, yb, zb = (_ceil_log2(int(x)) for x in limits)
    lb = 4  # enough for dyadic exponents up through 2^15 per axis
    lx = np.log2(blocks[:, 3]).astype(np.uint64)
    ly = np.log2(blocks[:, 4]).astype(np.uint64)
    lz = np.log2(blocks[:, 5]).astype(np.uint64)
    if xb + yb + zb + 3 * lb > 64:
        raise ValueError('grid is too large for the Step-8.4 packed geometry key')
    key = blocks[:, 0].astype(np.uint64)
    sh = xb
    key |= blocks[:, 1].astype(np.uint64) << np.uint64(sh)
    sh += yb
    key |= blocks[:, 2].astype(np.uint64) << np.uint64(sh)
    sh += zb
    key |= lx << np.uint64(sh)
    sh += lb
    key |= ly << np.uint64(sh)
    sh += lb
    key |= lz << np.uint64(sh)
    return key


def build_session_spatial_blocks(prepared, max_sessions=None, progress=True):
    """Build an exact adaptive spatial tiling independently for each session.

    This is the key Step-8.4 change.  Spatial merging no longer requires two
    neighboring voxels to share an entire lifetime history.  Each session is
    tiled independently, so spatial resolution may change with time.

    Returns ``(blocks_by_session, build_ms_by_session)`` where each block array
    is int32 [x0,y0,z0,sx,sy,sz].
    """
    n = len(prepared.session_codes) if max_sessions is None else min(int(max_sessions), len(prepared.session_codes))
    limits = (prepared.spec.nx, prepared.spec.ny, prepared.spec.nz)
    out = []
    times = []
    for t, codes in enumerate(prepared.session_codes[:n]):
        t0 = time.perf_counter_ns()
        x, y, z = linear_decode(codes, prepared.spec)
        coords = np.column_stack((x, y, z)).astype(np.int32, copy=False)
        label = f'session {t+1}/{n} vox={len(coords):,}' if progress else None
        merged = merge_coords_fast(coords, limits, label)
        ms = (time.perf_counter_ns() - t0) / 1e6
        out.append(merged)
        times.append(ms)
        if progress:
            ratio = len(merged) / len(coords) if len(coords) else 0.0
            print(f'[time-local] session {t+1}/{n}: {len(coords):,} vox -> {len(merged):,} blocks ({ratio:.3f}), {ms/1000:.2f}s', flush=True)
    return out, times


class TimeLocalAdaptiveCore(AdaptiveBlockCore):
    """BST-agnostic time-local adaptive block map.

    Construction differs from :class:`AdaptiveBlockCore`:

      Step 8.3: group voxels by complete lifetime mask, then spatially merge.
      Step 8.4: spatially merge each session independently, then deduplicate
                identical block geometries across sessions and attach a shared
                temporal validity signature to each geometry.

    Consequently, adjacent voxels with different lifetime histories can still
    share one spatial block during the sessions in which the local tiling makes
    that merge possible.  Spatial resolution is therefore allowed to change
    over time while reconstruction remains exact.

    The core contains no BST-ID encoding logic.  The same core can be wrapped
    by an explicit-coordinate map or by a replaceable BST codec.
    """

    construction = 'time_local_session_tiling'

    def __init__(self, prepared, session_blocks, union_codes=None, bucket_voxels=8, progress=True):
        self.prepared = prepared
        self.spec = prepared.spec
        self.n_sessions = len(prepared.session_codes)
        if self.n_sessions > 64:
            raise ValueError('Step 8.4 uint64 temporal masks support at most 64 sessions')
        if len(session_blocks) < self.n_sessions:
            raise ValueError('session_blocks does not cover all prepared sessions')
        self.bucket_voxels = bucket_voxels
        self.union_codes = union_codes if union_codes is not None else np.empty(0, dtype=np.uint64)
        self.history_masks = None

        finalize_t0 = time.perf_counter_ns()
        limits = (self.spec.nx, self.spec.ny, self.spec.nz)
        blocks_parts = []
        key_parts = []
        bit_parts = []
        self.session_block_instances = 0
        for t, B in enumerate(session_blocks[:self.n_sessions]):
            B = np.asarray(B, dtype=np.int32)
            self.session_block_instances += len(B)
            if not len(B):
                continue
            blocks_parts.append(B)
            key_parts.append(_geometry_keys(B, limits))
            bit_parts.append(np.full(len(B), np.uint64(1) << np.uint64(t), dtype=np.uint64))

        if blocks_parts:
            all_blocks = np.concatenate(blocks_parts, axis=0)
            keys = np.concatenate(key_parts)
            bits = np.concatenate(bit_parts)
            order = np.argsort(keys, kind='stable')
            sk = keys[order]
            starts = np.r_[0, 1 + np.flatnonzero(sk[1:] != sk[:-1])]
            sbits = bits[order]
            geom_masks = np.bitwise_or.reduceat(sbits, starts)
            unique_geom = all_blocks[order[starts]]
            unique_geom_keys = sk[starts].copy()
            # Drop the large concatenated/sort-key temporary arrays as early as possible.
            del all_blocks, keys, bits, order, sk, sbits
        else:
            unique_geom = np.empty((0, 6), dtype=np.int32)
            unique_geom_keys = np.empty(0, dtype=np.uint64)
            geom_masks = np.empty(0, dtype=np.uint64)

        self.unique_masks = np.unique(geom_masks)
        self.sig_bytes, self.temporal_prefixes = temporal_signature_bytes(self.unique_masks, self.n_sessions)
        self.sid_to_mask = self.unique_masks.astype(np.uint64)
        sids = np.searchsorted(self.unique_masks, geom_masks).astype(np.int32)
        self.blocks = np.concatenate([unique_geom, sids[:, None]], axis=1) if len(unique_geom) else np.empty((0, 7), dtype=np.int32)
        self.geometry_finalize_ms = (time.perf_counter_ns() - finalize_t0) / 1e6

        t0 = time.perf_counter_ns()
        self._build_buckets(progress)
        self.union_index_build_ms = (time.perf_counter_ns() - t0) / 1e6

        t0 = time.perf_counter_ns()
        self._build_time_buckets(session_blocks[:self.n_sessions], unique_geom_keys, progress)
        self.time_index_build_ms = (time.perf_counter_ns() - t0) / 1e6
        del geom_masks, unique_geom_keys
        self.last_query_stats = {}
        self.last_change_stats = {}

        if progress:
            reuse = self.session_block_instances / len(self.blocks) if len(self.blocks) else 0.0
            print(f'[time-local] {self.session_block_instances:,} session block instances -> {len(self.blocks):,} unique geometries (reuse {reuse:.3f}x)', flush=True)

    def _build_time_buckets(self, session_blocks, unique_geom_keys, progress):
        """Build a counted (session, spatial-bucket) posting index.

        Unlike the Step-8.4 union bucket index, every posting here is active for
        the session encoded in its key.  Queries therefore avoid retrieving
        geometries that overlap the region spatially but are inactive at t.

        This is an auxiliary runtime index and its bytes are counted separately
        by ``common_storage_bytes(index_mode="time")``.
        """
        b = self.bucket_voxels
        _, nby, nbz = self._bucket_dims
        refs_k = []
        refs_i = []
        n_sessions_u = np.uint64(self.n_sessions)

        for t, B0 in enumerate(session_blocks):
            B = np.asarray(B0, dtype=np.int32)
            if not len(B):
                continue
            gk = _geometry_keys(B, (self.spec.nx, self.spec.ny, self.spec.nz))
            gids = np.searchsorted(unique_geom_keys, gk).astype(np.int32)
            if len(gids) and not np.array_equal(unique_geom_keys[gids], gk):
                raise AssertionError('time-local geometry lookup failed while building temporal bucket index')

            bx0 = B[:, 0] // b; by0 = B[:, 1] // b; bz0 = B[:, 2] // b
            bx1 = (B[:, 0] + B[:, 3] - 1) // b
            by1 = (B[:, 1] + B[:, 4] - 1) // b
            bz1 = (B[:, 2] + B[:, 5] - 1) // b
            one = (bx0 == bx1) & (by0 == by1) & (bz0 == bz1)
            ids = gids[one]
            spatial = ((bx0[one].astype(np.uint64) * np.uint64(nby) + by0[one].astype(np.uint64)) * np.uint64(nbz) + bz0[one].astype(np.uint64))
            refs_k.append(spatial * n_sessions_u + np.uint64(t))
            refs_i.append(ids)

            multi = np.flatnonzero(~one)
            for j in multi:
                xs = np.arange(bx0[j], bx1[j] + 1, dtype=np.uint64)
                ys = np.arange(by0[j], by1[j] + 1, dtype=np.uint64)
                zs = np.arange(bz0[j], bz1[j] + 1, dtype=np.uint64)
                spatial = ((xs[:, None, None] * np.uint64(nby) + ys[None, :, None]) * np.uint64(nbz) + zs[None, None, :]).ravel()
                refs_k.append(spatial * n_sessions_u + np.uint64(t))
                refs_i.append(np.full(len(spatial), gids[j], dtype=np.int32))

        allk = np.concatenate(refs_k) if refs_k else np.empty(0, dtype=np.uint64)
        alli = np.concatenate(refs_i) if refs_i else np.empty(0, dtype=np.int32)
        if len(allk):
            o = np.argsort(allk, kind='stable')
            self.time_ref_keys = allk[o]
            self.time_ref_ids = alli[o]
            self.time_bucket_keys, self.time_bucket_starts, self.time_bucket_counts = np.unique(
                self.time_ref_keys, return_index=True, return_counts=True
            )
        else:
            self.time_ref_keys = np.empty(0, dtype=np.uint64)
            self.time_ref_ids = np.empty(0, dtype=np.int32)
            self.time_bucket_keys = np.empty(0, dtype=np.uint64)
            self.time_bucket_starts = np.empty(0, dtype=np.int64)
            self.time_bucket_counts = np.empty(0, dtype=np.int64)
        self.time_bucket_references = len(self.time_ref_ids)
        if progress:
            print(
                f'[time-local] temporal bucket index: {len(self.time_bucket_keys):,} (bucket,session) keys, '
                f'{self.time_bucket_references:,} refs', flush=True
            )

    def _candidates_time(self, bounds, t):
        x0, x1, y0, y1, z0, z1 = bounds
        b = self.bucket_voxels
        _, nby, nbz = self._bucket_dims
        xs = np.arange(x0 // b, x1 // b + 1, dtype=np.uint64)
        ys = np.arange(y0 // b, y1 // b + 1, dtype=np.uint64)
        zs = np.arange(z0 // b, z1 // b + 1, dtype=np.uint64)
        spatial = ((xs[:, None, None] * np.uint64(nby) + ys[None, :, None]) * np.uint64(nbz) + zs[None, None, :]).ravel()
        q = spatial * np.uint64(self.n_sessions) + np.uint64(t)
        p = np.searchsorted(self.time_bucket_keys, q)
        valid = p < len(self.time_bucket_keys)
        p = p[valid]; q = q[valid]
        hit = p[self.time_bucket_keys[p] == q]
        if not len(hit):
            return np.empty(0, dtype=np.int32), 0
        parts = [self.time_ref_ids[s:s+c] for s, c in zip(self.time_bucket_starts[hit], self.time_bucket_counts[hit])]
        refs = sum(len(x) for x in parts)
        ids = np.unique(np.concatenate(parts)) if len(parts) > 1 else np.unique(parts[0])
        return ids, refs

    def _select_mode(self, bounds, t, index_mode='union'):
        if index_mode == 'union':
            return self._select(bounds, t=t)
        if index_mode != 'time':
            raise ValueError(f'unknown index_mode={index_mode!r}')
        ids, refs = self._candidates_time(bounds, t)
        if not len(ids):
            return ids, np.empty(0, dtype=np.int8), refs, 0
        B = self.blocks[ids]
        x0, x1, y0, y1, z0, z1 = bounds
        inter = (
            (B[:, 0] <= x1) & (B[:, 0] + B[:, 3] - 1 >= x0) &
            (B[:, 1] <= y1) & (B[:, 1] + B[:, 4] - 1 >= y0) &
            (B[:, 2] <= z1) & (B[:, 2] + B[:, 5] - 1 >= z0)
        )
        ids = ids[inter]
        # Every posting in this index is already active at t; no temporal mask
        # filtering is required after the spatial intersection check.
        return ids, np.ones(len(ids), dtype=np.int8), refs, len(ids)

    def region_native(self, region, t, index_mode='union'):
        bounds = metric_box_to_bounds(self.spec, region)
        ids, _, refs, scanned = self._select_mode(bounds, t, index_mode=index_mode)
        out = self._clip_blocks(ids, bounds)
        self.last_query_stats = {
            'blocks_scanned': scanned, 'active_blocks': len(ids), 'bucket_refs': refs,
            'native_primitives': len(out), 'native_kind': 'blocks', 'index_mode': index_mode,
        }
        return out

    def region_voxels(self, region, t, index_mode='union'):
        bounds = metric_box_to_bounds(self.spec, region)
        ids, _, refs, scanned = self._select_mode(bounds, t, index_mode=index_mode)
        out = self._materialize(ids, bounds)
        self.last_query_stats = {
            'blocks_scanned': scanned, 'active_blocks': len(ids), 'bucket_refs': refs,
            'result_voxels': len(out), 'native_kind': 'blocks', 'index_mode': index_mode,
        }
        return out

    def changes_native(self, region, ta, tb):
        raise NotImplementedError(
            'Time-local tilings may use different block boundaries at ta and tb. '
            'There is no exact signed native-block difference without an additional '
            'spatial Boolean decomposition; use output="voxels" for exact changes.'
        )

    def changes_voxels(self, region, ta, tb, index_mode='union'):
        """Exact endpoint difference for time-varying block tilings."""
        if tb < ta:
            ta, tb = tb, ta
        a = self.region_voxels(region, ta, index_mode=index_mode)
        sa = dict(self.last_query_stats)
        b = self.region_voxels(region, tb, index_mode=index_mode)
        sb = dict(self.last_query_stats)
        added = np.setdiff1d(b, a, assume_unique=True)
        removed = np.setdiff1d(a, b, assume_unique=True)
        self.last_change_stats = {
            'blocks_scanned': sa.get('blocks_scanned', 0) + sb.get('blocks_scanned', 0),
            'active_blocks_endpoints': sa.get('active_blocks', 0) + sb.get('active_blocks', 0),
            'bucket_refs': sa.get('bucket_refs', 0) + sb.get('bucket_refs', 0),
            'result_voxels': len(added) + len(removed),
            'native_kind': 'blocks',
            'change_strategy': 'exact_endpoint_materialize_then_setdiff',
        }
        return added, removed

    def common_storage_bytes(self, index_mode='union'):
        if index_mode == 'union':
            directory = 12 * len(self.bucket_keys) + 4 * self.bucket_references
        elif index_mode == 'time':
            directory = 12 * len(self.time_bucket_keys) + 4 * self.time_bucket_references
        else:
            raise ValueError(f'unknown index_mode={index_mode!r}')
        return 24 + directory + self.sig_bytes

    def stats(self, index_mode='union'):
        nblocks = len(self.blocks)
        if index_mode == 'union':
            refs = self.bucket_references; count = len(self.bucket_keys)
        elif index_mode == 'time':
            refs = self.time_bucket_references; count = len(self.time_bucket_keys)
        else:
            raise ValueError(f'unknown index_mode={index_mode!r}')
        union_bytes = 12 * len(self.bucket_keys) + 4 * self.bucket_references
        time_bytes = 12 * len(self.time_bucket_keys) + 4 * self.time_bucket_references
        return {
            'blocks': nblocks,
            'session_block_instances': self.session_block_instances,
            'geometry_reuse_factor': self.session_block_instances / nblocks if nblocks else 0.0,
            'block_records_per_union_voxel': nblocks / len(self.union_codes) if len(self.union_codes) else float('nan'),
            'unique_temporal_signatures': len(self.unique_masks),
            'temporal_prefixes': self.temporal_prefixes,
            'bucket_references': refs,
            'bucket_count': count,
            'bucket_voxels': self.bucket_voxels,
            'index_mode': index_mode,
            'union_bucket_references': self.bucket_references,
            'union_bucket_count': len(self.bucket_keys),
            'time_bucket_references': self.time_bucket_references,
            'time_bucket_count': len(self.time_bucket_keys),
            'union_index_bytes': union_bytes,
            'time_index_bytes': time_bytes,
            'time_index_extra_bytes': time_bytes - union_bytes,
            'geometry_finalize_ms': self.geometry_finalize_ms,
            'union_index_build_ms': self.union_index_build_ms,
            'time_index_build_ms': self.time_index_build_ms,
            'construction': self.construction,
        }
