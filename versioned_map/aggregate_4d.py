"""Step 10.5A: exact *tile-coverage* aggregation across X/Y/Z/T.

This is a local integer-tile-domain adapter to the unmodified upstream BST-ID
region algebra, NOT the author's geodetic lon/lat/altitude coordinate encoder.
The canonical tile grid (nonnegative integer x,y,z, <=2**21) is mapped
injectively to binary x/y/f prefixes at zoom 21. Time is UTC UNIX seconds.
A four-dimensional cell means: every tile in its spatial box has at least one
observed endpoint throughout its time interval. It does NOT say their geometry
is identical. Payloads remain exact per-tile versioned blob references.

The builder coalesces (1) contiguous presence runs across time even if the
payload changes, and (2) aligned dyadic sibling cells along any of x/y/f/t.
Only full siblings are merged: no phantom occupied tile and no false negative.
The index performs direct 4D cell-box intersections. A per-tile version map
resolves each candidate to the exact historical payload; no delta replay.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import time
import zlib

import numpy as np

from .author import author_module
from .interval_store import IntervalMap, _dyadic_time_cover, _timestamp, MAX_TIME
from .region_index import _parse_xyz_key, _tile_range
from .store import _from_bytes, _region_filter, canonical

SPATIAL_BITS = 21
INDEX_VERSION = 1


def _raw_presence_runs(records, sessions):
    """Combine adjacent nonempty tile states even when their geometry changed."""
    by_tile = defaultdict(list)
    for r in records:
        if int(r['count']) > 0:
            by_tile[r['key']].append((int(r['start']), int(r['stop'])))
    for key, vals in by_tile.items():
        vals.sort()
        lo, hi = vals[0]
        for a, b in vals[1:]:
            if a < hi:
                raise ValueError('overlapping tile validity intervals')
            if a == hi:
                hi = b
            else:
                yield key, lo, hi
                lo, hi = a, b
        yield key, lo, hi


def _leaf_cells(records, sessions):
    seen = set()
    presence = 0
    for key, lo, hi in _raw_presence_runs(records, sessions):
        presence += 1
        xyz = _parse_xyz_key(key)
        if any(x >= (1 << SPATIAL_BITS) for x in xyz):
            raise ValueError('tile index exceeds 21-bit canonical grid')
        start = _timestamp(sessions[lo])
        stop = _timestamp(sessions[hi]) if hi < len(sessions) else MAX_TIME
        for z, p in _dyadic_time_cover(start, stop):
            cell = (SPATIAL_BITS, xyz[0], SPATIAL_BITS, xyz[1],
                    SPATIAL_BITS, xyz[2], z, p)
            seen.add(cell)
    return seen, presence


def _merge_exact_siblings(cells):
    """Bottom-up set-union coarsening in all four axes; O(axes * cells * rounds).

    A space merge is legal only if two spatial siblings have identical
    descriptors on the other three axes; a time merge uses the same rule.
    The union of covered (tile, timestamp) points is exactly preserved.
    """
    out = set(cells)
    merged = [0, 0, 0, 0]
    iterations = 0
    while True:
        changed = 0
        iterations += 1
        for axis in (3, 0, 1, 2):  # time first, then space
            zpos = 2*axis; vpos = zpos+1
            pairs = defaultdict(dict)
            for c in out:
                z = c[zpos]; value = c[vpos]
                if z <= 1:
                    continue
                parent = list(c)
                parent[zpos] = z-1
                parent[vpos] = value >> 1
                pairs[tuple(parent)][value & 1] = c
            removed = set(); inserted = set()
            for parent, bits in pairs.items():
                if 0 in bits and 1 in bits:
                    removed.add(bits[0]);removed.add(bits[1]);inserted.add(parent)
            if removed:
                out.difference_update(removed)
                out.update(inserted)
                merged[axis] += len(inserted)
                changed += len(inserted)
        if not changed:
            break
    return out, {'merged_pairs_by_axis': dict(zip(('x','y','z','t'), merged)),
                 'merge_rounds': iterations}


def _ranges(cells):
    if not len(cells):
        return np.zeros((0,8), dtype=np.int64)
    z = cells[:, 0::2].astype(np.int64)
    v = cells[:, 1::2].astype(np.int64)
    bits = np.array([SPATIAL_BITS, SPATIAL_BITS, SPATIAL_BITS, 32], dtype=np.int64)
    shift = bits - z
    low = v << shift
    high = (v+1) << shift
    return np.column_stack([low[:,0],high[:,0],low[:,1],high[:,1],
                            low[:,2],high[:,2],low[:,3],high[:,3]])


class Merged4DIndex:
    def __init__(self, interval, *, coalesced=True):
        self.coalesced = bool(coalesced)
        self.map = interval if isinstance(interval, IntervalMap) else IntervalMap(interval)
        if self.map.meta['mode'] != 'bst':
            raise ValueError('merged BST index requires the official BST interval store')
        self.root = self.map.root
        self._cells = None
        self._boxes = None
        self._tiles = None
        self._build_info = None

    @property
    def path(self):
        return self.root/('aggregate_105a_merged.npz' if self.coalesced else 'aggregate_105a_raw.npz')

    def build(self, *, merge=None):
        """Rebuild from immutable interval records without rewriting geometry."""
        start = time.perf_counter()
        if merge is None:merge = self.coalesced
        if bool(merge) != self.coalesced:
            raise ValueError('build merge setting must match instance coalesced setting')
        raw, presence_runs = _leaf_cells(self.map.records, self.map.meta['sessions'])
        nodes, m = _merge_exact_siblings(raw) if merge else (raw, {
            'merged_pairs_by_axis': dict.fromkeys(('x','y','z','t'),0),
            'merge_rounds':0})
        array = np.asarray(sorted(nodes),dtype=np.uint32).reshape(-1,8)
        Cell = author_module('region_algebra').Cell
        # True author Cell.to_id / from_id, for a clearly labeled LOCAL tile
        # address frame. Store opaque 17-byte framed words losslessly (V17
        # avoids numpy S17's trailing-NUL truncation).
        ids = np.empty(len(array), dtype='V17')
        for i, c in enumerate(array):
            cell = Cell.from_parts(x=(int(c[0]),int(c[1])),
                                   y=(int(c[2]),int(c[3])),
                                   f=(int(c[4]),int(c[5])),
                                   t=(int(c[6]),int(c[7])))
            encoded = cell.to_id()
            if Cell.from_id(encoded) != cell:
                raise AssertionError('official Cell ID round-trip failed')
            ids[i] = np.void(encoded.to_bytes(17,'big'))
        digest = sha256((self.root/'records.json').read_bytes()).hexdigest()
        f = self.path.with_name(self.path.stem+'.tmp.npz')
        np.savez_compressed(f, cells=array, bst_ids=ids,
                            digest=np.asarray(digest),
                            merged=np.asarray(int(merge),dtype=np.int8),
                            version=np.asarray(INDEX_VERSION,dtype=np.int16))
        f.replace(self.path)
        self._cells=self._boxes=self._tiles=None
        info = {'raw_four_d_cells':len(raw),'merged_four_d_cells':len(nodes),
                'presence_runs':presence_runs,**m,
                'index_bytes':self.path.stat().st_size,
                'build_ms':(time.perf_counter()-start)*1000,
                'frame':'local 21-bit integer tile grid + UTC Unix seconds; not geodetic'}
        self._build_info=info
        return info

    def load(self):
        if self._cells is not None:
            return
        with np.load(self.path, allow_pickle=False) as z:
            digest=sha256((self.root/'records.json').read_bytes()).hexdigest()
            if str(z['digest']) != digest or int(z['version'])!=INDEX_VERSION or bool(int(z['merged']))!=self.coalesced:
                raise IOError('stale or incompatible merged index: rebuild')
            cells = z['cells'].astype(np.uint32, copy=True)
            ids = z['bst_ids'].copy()
        if cells.ndim!=2 or cells.shape[1]!=8 or len(ids)!=len(cells):
            raise IOError('invalid merged index shape')
        Cell=author_module('region_algebra').Cell
        for i, c in enumerate(cells):
            # Full framed verification is done at load, outside hot timing.
            cell=Cell.from_parts(x=(int(c[0]),int(c[1])),
                                 y=(int(c[2]),int(c[3])),
                                 f=(int(c[4]),int(c[5])),
                                 t=(int(c[6]),int(c[7])))
            if cell.to_id().to_bytes(17,'big') != bytes(ids[i]):
                raise IOError('corrupt merged BST-ID word')
        rows=defaultdict(list)
        for i,r in enumerate(self.map.records):
            rows[r['key']].append(i)
        for key, rr in rows.items():
            rr.sort(key=lambda i:self.map.records[i]['start'])
        self._tiles = {key:(indices,[self.map.records[i]['start'] for i in indices])
                       for key,indices in rows.items()}
        self._cells=cells
        self._boxes=_ranges(cells)

    def _candidates(self,bounds,*,time_a,time_b):
        self.load()
        m=self.map.meta['source_meta']
        extent = _tile_range(bounds,m['dims'],m['tile_side'])
        if extent is None:
            return set(),{'cells_total':len(self._cells),'cells_hit':0,
                          'tile_candidates':0,'temporal_candidates':0}
        boxes=self._boxes
        mask = (boxes[:,0] < extent[0][1]+1) & (boxes[:,1] > extent[0][0])
        mask &= (boxes[:,2] < extent[1][1]+1) & (boxes[:,3] > extent[1][0])
        mask &= (boxes[:,4] < extent[2][1]+1) & (boxes[:,5] > extent[2][0])
        mask &= (boxes[:,6] < time_b) & (boxes[:,7] > time_a)
        indices=np.flatnonzero(mask)
        tiles=set()
        for i in indices:
            b=boxes[i]
            for x in range(max(int(b[0]),extent[0][0]),min(int(b[1])-1,extent[0][1])+1):
                for y in range(max(int(b[2]),extent[1][0]),min(int(b[3])-1,extent[1][1])+1):
                    for z in range(max(int(b[4]),extent[2][0]),min(int(b[5])-1,extent[2][1])+1):
                        tiles.add(f'{x},{y},{z}')
        return tiles,{'cells_total':len(self._cells),'cells_hit':len(indices),
                      'tile_candidates':len(tiles),'temporal_candidates':len(indices)}

    def select(self,session,bounds,*,to_session=None):
        sessions=self.map.meta['sessions']
        if session not in sessions:raise KeyError(session)
        version=sessions.index(session)
        if to_session is None:
            a,b=_timestamp(session),_timestamp(session)+1
        else:
            if to_session not in sessions or sessions.index(to_session)<version:
                raise ValueError('invalid history range')
            j=sessions.index(to_session)
            a=_timestamp(session)
            b=_timestamp(sessions[j+1]) if j+1<len(sessions) else MAX_TIME
        tile_keys, stats=self._candidates(bounds,time_a=a,time_b=b)
        result=[];examined=0
        for key in tile_keys:
            pair=self._tiles.get(key)
            if pair is None:continue
            indices,starts=pair
            pos=max(0,bisect_right(starts,version)-1)
            for row in indices[pos:]:
                r=self.map.records[row]
                examined+=1
                if to_session is None:
                    if r['start'] <= version < r['stop']:
                        result.append(row)
                    if r['start'] > version:break
                else:
                    if r['start'] > j:break
                    if r['start'] <= j and r['stop'] > version:
                        result.append(row)
        return sorted(set(result)),{**stats,'records_examined':examined}

    def query(self,session,bounds,*,stats=False):
        t0=time.perf_counter()
        rows,info=self.select(session,bounds)
        index_ms=(time.perf_counter()-t0)*1000
        spec=self.map._world_spec();bits=0;pieces=[]
        for row in rows:
            r=self.map.records[row]
            blob=(self.root/'blobs'/r['hash']).read_bytes();bits+=len(blob)
            if sha256(blob).hexdigest()!=r['hash']:
                raise IOError('immutable tile checksum mismatch')
            codes=_from_bytes(zlib.decompress(blob))
            if len(codes)!=r['count']:
                raise IOError('incorrect tile element count')
            pieces.append(_region_filter(codes,spec,bounds))
        result=canonical(np.concatenate(pieces) if pieces else np.empty(0,dtype=np.uint64))
        metrics={**info,'index_ms':index_ms,'tiles_read':len(rows),
                 'compressed_bytes_read':bits,'latency_ms':(time.perf_counter()-t0)*1000,
                 'replayed_transitions':0, 'mode':'bst_merged_4d'}
        return (result,metrics) if stats else result

    def history_records(self,start,stop,bounds):
        rows,info=self.select(start,bounds,to_session=stop)
        return [dict(self.map.records[r]) for r in rows],info

    def disk_stats(self):
        self.load()
        return {'merged_index_bytes':self.path.stat().st_size,
                'cell_count':len(self._cells),
                'array_RAM_lower_bound':self._cells.nbytes+self._boxes.nbytes,
                'underlying_interval_bytes':self.map.disk_stats()['total_bytes']}
