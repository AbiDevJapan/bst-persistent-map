"""Step 10.5 -- directly queryable four-dimensional version-valid tile database.

Every record is one *complete* immutable tile payload valid over consecutive
observed-map versions. No data deltas are replayed. UTC time is version-validity
and is NOT a claim of environmental persistence between observations.

The BST mode builds an actual 4-axis author-Cell index:
  - x/y/f are full-zoom official tile-ANCHOR coordinates;
  - t is a minimal dyadic cover of the validity interval;
  - every posting is the lossless official BST-ID of that (x,y,f,t) Cell.
Indexed candidate selection uses these components, followed by exact integer
spatial bounds and per-voxel filtering. Because tile anchors are NOT tile-box
encodings, the coarse spatial candidate range expands to every intersecting
canonical tile. This prevents false negatives for boundary-crossing ROIs.

The Morton control has *identical intervals and payloads*, but a Morton tile
hash and a binary search over per-tile interval start versions.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math
import shutil
import time
import zlib
import numpy as np

from .store import VersionedStore, _from_bytes, _region_filter, canonical
from .region_index import (_author_payload, _morton3, _parse_xyz_key,
                           _tile_axis_payload, _tile_range, ZOOMS, PREFIX_ZOOMS,
                           manifest_hash)
from .author import author_module

SCHEMA = 1
MAX_TIME = 1 << 32


def _timestamp(session):
    t = int(datetime.fromisoformat(session).replace(tzinfo=timezone.utc).timestamp())
    if not 0 <= t < MAX_TIME:
        raise ValueError('official BST-ID time is a 32-bit UNIX timestamp')
    return t


def _put_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, separators=(',', ':'), sort_keys=True))
    temporary.replace(path)


def _dyadic_time_cover(start, stop):
    """Exact minimal cover of [start,stop) in unsigned 32-bit UNIX seconds.

    The official Cell algebra requires zoom >=1. An exact all-time cover is
    represented by the two zoom-1 children if necessary.
    """
    if not 0 <= int(start) < int(stop) <= MAX_TIME:
        raise ValueError('invalid 32-bit half-open time interval')
    out = []
    cursor = int(start)
    while cursor < stop:
        alignment = (cursor & -cursor) if cursor else MAX_TIME
        largest = 1 << ((stop - cursor).bit_length() - 1)
        length = min(alignment, largest, 1 << 31)
        zoom = 32 - (length.bit_length() - 1)
        out.append((zoom, cursor >> (32 - zoom)))
        cursor += length
    return tuple(out)


def _validity_times(record, sessions):
    return _timestamp(sessions[record['start']]), (_timestamp(sessions[record['stop']])
             if record['stop'] < len(sessions) else MAX_TIME)


class IntervalMap:
    def __init__(self, root):
        self.root = Path(root)
        self._meta = None
        self._records = None
        self._index = None

    @property
    def meta(self):
        if self._meta is None:
            self._meta = json.loads((self.root / 'metadata.json').read_text())
        return self._meta

    @property
    def records(self):
        if self._records is None:
            self._records = json.loads((self.root / 'records.json').read_text())
        return self._records

    @classmethod
    def build(cls, source: VersionedStore, root, mode='bst', *, through_session=None):
        if mode not in ('bst', 'morton'):
            raise ValueError('mode must be bst or morton')
        src = source.meta()
        if mode == 'bst' and src['codec'] != 'official':
            raise ValueError('BST interval index requires an official-codec source')
        dest = cls(root)
        if dest.root.exists():
            raise FileExistsError('choose a new interval database directory')
        versions = source.all_versions()
        sessions = list(versions)
        if not sessions:
            raise ValueError('source has no sessions')
        if through_session is not None:
            if through_session not in sessions:
                raise KeyError(through_session)
            sessions = sessions[:sessions.index(through_session)+1]
        stamps = [_timestamp(s) for s in sessions]
        if any(a >= b for a,b in zip(stamps, stamps[1:])):
            raise ValueError('sessions must have increasing UTC timestamps')
        start = time.perf_counter()
        dest.root.mkdir(parents=True)
        (dest.root / 'blobs').mkdir()
        try:
            # Build one run for each tile while retaining geometry identical to
            # the source snapshots; unchanged consecutive hashes share a run.
            history = defaultdict(list)
            for i, session in enumerate(sessions):
                for key, entry in versions[session].items():
                    _parse_xyz_key(key)
                    history[key].append((i, entry['hash'], int(entry['count'])))
            records = []
            for key, events in sorted(history.items()):
                for idx, digest, count in events:
                    if records and records[-1]['key'] == key and records[-1]['stop'] == idx and records[-1]['hash'] == digest:
                        records[-1]['stop'] = idx + 1
                    else:
                        records.append({'key':key, 'start':idx, 'stop':idx+1,
                                        'hash':digest, 'count':count})
            # Every distinct referenced compressed tile is written ONCE.
            for digest in sorted({r['hash'] for r in records}):
                src_file = source.root / 'blobs' / digest
                b = src_file.read_bytes()
                if sha256(b).hexdigest() != digest:
                    raise IOError('source snapshot blob checksum mismatch: ' + digest)
                (dest.root/'blobs'/digest).write_bytes(b)
            _put_json(dest.root/'records.json', records)
            _put_json(dest.root/'metadata.json', {
                'schema':SCHEMA, 'mode':mode, 'source_meta':src, 'sessions':sessions,
                'source_manifest_hashes':{s:manifest_hash(versions[s]) for s in sessions},
                'geometry':'exact canonical observed-endpoint uint64 voxel codes',
                'semantics':'session-version validity; absence is not observed free space',
                'time_domain':'UTC Unix seconds, unsigned 32 bits; newest valid until next appended version',
                'index_role':('official four-axis BST-ID cells and x/y/f/t prefix postings'
                              if mode == 'bst' else 'Morton tile hash and sorted version intervals'),
                'build_s':time.perf_counter()-start,
            })
            dest.rebuild_index()
            dest.validate_structure()
            return dest
        except BaseException:
            shutil.rmtree(dest.root, ignore_errors=True)
            raise

    def rebuild_index(self):
        """Persist a real BST-4D posting index or a conventional Morton index.

        Snapshot payloads are neither rewritten nor decompressed. The four-axis
        BST cell's official integer is serialized as fixed 17-byte big-endian,
        with arrays documenting each cell's time zoom and record pointer.
        """
        start = time.perf_counter()
        mode=self.meta['mode']; rec=self.records; src=self.meta['source_meta']
        xyz=np.asarray([_parse_xyz_key(r['key']) for r in rec],dtype=np.int32).reshape(-1,3)
        arrays={'coordinates':xyz,'mode':np.asarray(mode),
                'record_digest':np.asarray(sha256((self.root/'records.json').read_bytes()).hexdigest())}
        if mode == 'bst':
            Cell=author_module('region_algebra').Cell
            # Actual official quantized anchors, not independently chosen Morton bits.
            original=[tuple(int(_tile_axis_payload(i,int(pos[i])*src['tile_side'],src)) for i in range(3))
                      for pos in xyz]
            orig=np.asarray(original,dtype=np.uint64).reshape(-1,3)
            arrays['official_axes']=orig
            postings=[]; all_ids=[]
            for i,r in enumerate(rec):
                lo,hi=_validity_times(r,self.meta['sessions'])
                for zoom,prefix in _dyadic_time_cover(lo,hi):
                    cell=Cell.from_parts(x=(28,int(orig[i,0])),y=(28,int(orig[i,1])),
                                         f=(24,int(orig[i,2])),t=(int(zoom),int(prefix)))
                    encoded=cell.to_id()
                    if Cell.from_id(encoded)!=cell:
                        raise AssertionError('official four-dimensional BST cell roundtrip failed')
                    # Zooms are at most 28+28+24+32+24 = 136 bits = 17 bytes.
                    all_ids.append(encoded.to_bytes(17,'big'))
                    postings.append((i,zoom,prefix))
            arr=np.asarray(postings,dtype=np.uint64).reshape(-1,3)
            arrays['post_record']=arr[:,0].astype(np.int32)
            arrays['post_zoom']=arr[:,1].astype(np.uint8)
            arrays['post_prefix']=arr[:,2].astype(np.uint32)
            arrays['bst4d_ids']=np.asarray(all_ids,dtype='S17')
        else:
            arrays['morton']=np.asarray([_morton3(*p) for p in xyz],dtype=np.uint64)
        path=self.root/'index.npz';tmp=self.root/'index.tmp.npz'
        np.savez_compressed(tmp,**arrays)
        tmp.replace(path)
        self._index=None
        return {'mode':mode,'index_file_bytes':path.stat().st_size,
                'index_build_ms':(time.perf_counter()-start)*1000,
                'records':len(rec), 'four_d_cells':len(arrays.get('post_record',[]))}

    def _load_index(self):
        if self._index is not None:
            return self._index
        rec=self.records; mode=self.meta['mode']
        with np.load(self.root/'index.npz',allow_pickle=False) as z:
            expected=sha256((self.root/'records.json').read_bytes()).hexdigest()
            if str(z['record_digest'])!=expected or str(z['mode'])!=mode:
                raise IOError('stale or inconsistent four-dimensional index; rebuild index')
            xyz=z['coordinates'].astype(np.int32,copy=True)
            if len(xyz)!=len(rec):
                raise IOError('index row count mismatch')
            if mode=='morton':
                # Per-tile sorted interval starts for O(log history length) lookup.
                lookup=defaultdict(list)
                for i,r in enumerate(rec):
                    tilecode=_morton3(*map(int,xyz[i]));lookup[tilecode].append(i)
                for code,rows in lookup.items():
                    rows.sort(key=lambda i:rec[i]['start'])
                self._index={'xyz':xyz,'tiles':dict(lookup),
                             'starts':{key:[rec[i]['start'] for i in rows] for key,rows in lookup.items()}}
            else:
                axes=z['official_axes'].astype(np.uint64,copy=True)
                rid=z['post_record'].astype(np.int32,copy=True)
                zoom=z['post_zoom'].astype(np.uint8,copy=True)
                prefix=z['post_prefix'].astype(np.uint32,copy=True)
                ids=z['bst4d_ids'].copy()
                if len(zoom)!=len(prefix) or len(zoom)!=len(rid) or len(ids)!=len(rid):
                    raise IOError('malformed four-dimensional index')
                # Parse and validate the persisted author's actual four-axis cell.
                Cell=author_module('region_algebra').Cell
                time_lookup=defaultdict(set); fragment_for_record=defaultdict(list)
                for i, (row,d,v) in enumerate(zip(rid,zoom,prefix)):
                    row=int(row);d=int(d);v=int(v)
                    if row<0 or row>=len(rec):raise IOError('bad posting record ID')
                    # np.bytes_ trims trailing NUL only; IDs are big-endian and
                    # have nonzero final bytes in general, but a fixed-width
                    # string cannot preserve trailing NUL bytes! Reconstruct
                    # from the four stored components and compare prefix bytes.
                    cell=Cell.from_parts(x=(28,int(axes[row,0])),y=(28,int(axes[row,1])),
                                         f=(24,int(axes[row,2])),t=(d,v))
                    raw=cell.to_id().to_bytes(17,'big')
                    if raw!=bytes(ids[i]).ljust(17,b'\x00'):
                        raise IOError('BST-ID cell payload mismatch')
                    time_lookup[(d,v)].add(row)
                    fragment_for_record[row].append((d,v))
                spatial=[];rows=[];prefix_by_row=[]
                for axis in range(3):
                    shift=ZOOMS[axis]-PREFIX_ZOOMS[axis]
                    values=(axes[:,axis] >> shift).astype(np.uint32)
                    order=np.argsort(values,kind='stable').astype(np.int32)
                    spatial.append(values[order]);rows.append(order);prefix_by_row.append(values)
                self._index={'xyz':xyz,'axes':axes,'spatial_values':spatial,
                             'spatial_rows':rows,'prefix_by_row':prefix_by_row,
                             'time_lookup':dict(time_lookup),'fragment_for_record':fragment_for_record}
        return self._index

    def _spatial_range(self,bounds):
        m=self.meta['source_meta']
        return _tile_range(bounds,m['dims'],m['tile_side'])

    def _temporal_rows(self,t):
        index=self._load_index();matches=set()
        for zoom in range(1,33):
            matches.update(index['time_lookup'].get((zoom,int(t)>>(32-zoom)),()))
        return matches

    def _select(self,when,bounds,*,time_start=None,time_end=None):
        """Select 4D records; when is stored session name for point queries.
        time_start/end is optional half-open UTC range for history queries.
        """
        sessions=self.meta['sessions']; t=_timestamp(when)
        if when not in sessions:raise KeyError('unknown stored version: '+when)
        version=sessions.index(when)
        idx=self._load_index(); rec=self.records
        ranges=self._spatial_range(bounds)
        if ranges is None:
            return [],{'spatial_candidates':0,'temporal_candidates':0,'records_examined':0,'index_probes':0}
        if self.meta['mode']=='morton':
            ids=[];nprobe=0
            for x in range(ranges[0][0],ranges[0][1]+1):
                for y in range(ranges[1][0],ranges[1][1]+1):
                    for z in range(ranges[2][0],ranges[2][1]+1):
                        nprobe+=1;key=_morton3(x,y,z)
                        arr=idx['tiles'].get(key)
                        if not arr:continue
                        if time_start is None:
                            pos=bisect_right(idx['starts'][key],version)-1
                            if pos>=0:
                                row=arr[pos]
                                if rec[row]['start']<=version<rec[row]['stop']:
                                    ids.append(row)
                        else:
                            # There is one nonoverlapping sorted interval chain
                            # per tile. Avoid a global historical manifest scan.
                            pos=max(0,bisect_right(idx['starts'][key],sessions.index(time_start))-1)
                            for row in arr[pos:]:
                                lo,hi=_validity_times(rec[row],sessions)
                                if lo>=time_end:break
                                if hi>_timestamp(time_start):ids.append(row)
            return sorted(set(ids)),{'spatial_candidates':nprobe,'temporal_candidates':len(ids),
                                     'records_examined':len(ids),'index_probes':nprobe}
        # Four-dimensional official BST-ID. Filter by indexed time prefix first,
        # then by stored spatial prefixes, and finally canonical tile bounds.
        if time_start is None:
            temporal=self._temporal_rows(t)
        else:
            temporal=set()
            for qz,qv in _dyadic_time_cover(_timestamp(time_start),time_end):
                # Query Cell temporal cover: two prefixes overlap iff one is
                # an ancestor of the other. For arbitrary time ranges, a
                # prebuilt list of all time posting keys is still small and
                # is filtered using this exact author prefix rule.
                for (pz,pv),rows in idx['time_lookup'].items():
                    depth=min(qz,pz)
                    if (qv>>(qz-depth))==(pv>>(pz-depth)):
                        temporal.update(rows)
        low=[]; high=[]; candidates=[]
        for axis,(a,b) in enumerate(ranges):
            size=self.meta['source_meta']['tile_side']
            val0=_tile_axis_payload(axis,a*size,self.meta['source_meta'])
            val1=_tile_axis_payload(axis,b*size,self.meta['source_meta'])
            shift=ZOOMS[axis]-PREFIX_ZOOMS[axis]
            lo,hi=sorted((val0>>shift,val1>>shift));low.append(lo);high.append(hi)
            col=idx['spatial_values'][axis]
            L=int(np.searchsorted(col,lo,'left'));R=int(np.searchsorted(col,hi,'right'))
            candidates.append((R-L,axis,L,R))
        nprobe,axis,L,R=min(candidates)
        source=idx['spatial_rows'][axis][L:R]
        if len(source):
            keep=np.ones(len(source),dtype=bool)
            for ax in range(3):
                keep &= (idx['prefix_by_row'][ax][source]>=low[ax]) & (idx['prefix_by_row'][ax][source]<=high[ax])
            source=source[keep]
        spatial_count=len(source)
        result=[];Cell=author_module('region_algebra').Cell
        for row in source:
            row=int(row)
            if row not in temporal:continue
            xyz=idx['xyz'][row]
            if any(int(xyz[j])<ranges[j][0] or int(xyz[j])>ranges[j][1] for j in range(3)):
                continue
            r=rec[row]
            start,stop=_validity_times(r,sessions)
            if time_start is None and not (start<=t<stop):continue
            if time_start is not None and not (start<time_end and stop>_timestamp(time_start)):continue
            # Actual upstream 4D region compatibility check, rather than a
            # bare integer timestamp comparison. The spatial dimensions are
            # the exact same author-produced full-zoom tile anchor cells.
            def cell_at(time_zoom,time_prefix):
                ax=idx['axes'][row]
                return Cell.from_parts(x=(28,int(ax[0])),y=(28,int(ax[1])),
                                       f=(24,int(ax[2])),t=(time_zoom,time_prefix))
            fragments=idx['fragment_for_record'][row]
            if time_start is None:
                query=cell_at(32,t)
                if not any(author_module('region_algebra').compatible(query,cell_at(z,v))
                           for z,v in fragments):
                    raise IOError('BST 4D index returned incompatible valid record')
            result.append(row)
        return sorted(set(result)),{'spatial_candidates':spatial_count,
                                    'temporal_candidates':len(temporal),
                                    'records_examined':len(source), 'index_probes':int(nprobe)}

    def query(self,session,bounds,*,stats=False,validation='full'):
        if validation not in ('full','hash'):
            raise ValueError('validation must be full or hash')
        start=time.perf_counter()
        t0=time.perf_counter();rows,sinfo=self._select(session,bounds)
        index_ms=(time.perf_counter()-t0)*1000
        spec=self._world_spec();pieces=[];bytes_read=0
        for row in rows:
            r=self.records[row]
            blob=(self.root/'blobs'/r['hash']).read_bytes();bytes_read+=len(blob)
            if sha256(blob).hexdigest()!=r['hash']:
                raise IOError('immutable tile blob checksum mismatch')
            codes=_from_bytes(zlib.decompress(blob))
            if len(codes)!=r['count']:
                raise IOError('corrupted immutable tile payload length')
            if validation=='full' and not np.array_equal(codes,canonical(codes)):
                raise IOError('corrupted immutable tile payload canonical form')
            pieces.append(_region_filter(codes,spec,bounds))
        result=canonical(np.concatenate(pieces) if pieces else np.empty(0,dtype=np.uint64))
        info={'mode':self.meta['mode'],**sinfo,'index_ms':index_ms,
              'tiles_read':len(rows),'compressed_bytes_read':bytes_read,
              'replayed_transitions':0,'latency_ms':(time.perf_counter()-start)*1000}
        return (result,info) if stats else result

    def _world_spec(self):
        from synthetic.world import WorldSpec
        m=self.meta['source_meta'];v=float(m['voxel_size_m']);d=m['dims']
        return WorldSpec(*[int(x)*v for x in d],v)

    def history_records(self,from_session,to_session,bounds):
        sessions=self.meta['sessions']
        a=sessions.index(from_session);b=sessions.index(to_session)
        if a>b:raise ValueError('history range must increase')
        end=_timestamp(sessions[b+1]) if b+1<len(sessions) else MAX_TIME
        ids,info=self._select(from_session,bounds,time_start=from_session,time_end=end)
        return [dict(self.records[i]) for i in ids],info

    def append_from_store(self,source,target_session):
        """Append the next map version. Immutable geometry blobs remain immutable;
        only the compact validity manifest and derived 4D index are updated.
        """
        source_versions=source.all_versions();sessions=self.meta['sessions']
        if target_session not in source_versions:raise KeyError(target_session)
        all_names=list(source_versions)
        previous=sessions[-1]
        if all_names.index(target_session)!=all_names.index(previous)+1:
            raise ValueError('append requires next chronological source version')
        if any(manifest_hash(source_versions[s])!=h for s,h in self.meta['source_manifest_hashes'].items()):
            raise ValueError('source historical manifests changed since build')
        if self.meta['source_meta']!=source.meta():
            raise ValueError('source metadata incompatible with interval database')
        _timestamp(target_session)
        if _timestamp(target_session)<=_timestamp(previous):raise ValueError('nonmonotonic UTC session time')
        current=len(sessions);new=source_versions[target_session];old=source_versions[previous]
        existing=self.records;tail={r['key']:r for r in existing if r['stop']==current}
        new_records=[dict(r) for r in existing]
        new_tail={r['key']:r for r in new_records if r['stop']==current}
        added_blobs=0;new_runs=0;extended=0;closed=0
        for key in sorted(set(old)|set(new)):
            before=old.get(key);after=new.get(key)
            if before and before['hash']==(after or {}).get('hash'):
                new_tail[key]['stop']+=1;extended+=1
            else:
                if before:closed+=1
                if after:
                    digest=after['hash'];dst=self.root/'blobs'/digest
                    if not dst.exists():
                        srcdata=(source.root/'blobs'/digest).read_bytes()
                        if sha256(srcdata).hexdigest()!=digest:raise IOError('source corruption')
                        dst.write_bytes(srcdata);added_blobs+=len(srcdata)
                    new_records.append({'key':key,'start':current,'stop':current+1,
                                        'hash':digest,'count':int(after['count'])})
                    new_runs+=1
        new_records.sort(key=lambda r:(r['key'],r['start']))
        old_meta=self.meta
        new_meta=dict(old_meta)
        new_meta['sessions']=sessions+[target_session]
        new_meta['source_manifest_hashes']={**old_meta['source_manifest_hashes'],target_session:manifest_hash(new)}
        # Make file replacement atomic per file; old blob contents never change.
        _put_json(self.root/'records.json',new_records)
        _put_json(self.root/'metadata.json',new_meta)
        self._records=new_records;self._meta=new_meta;self._index=None
        idx_result=self.rebuild_index()
        # Keep optional Step 10.5A four-dimensional aggregate sidecars coherent
        # when incremental sessions are appended. The original payload hashes
        # and histories remain untouched.
        for coalesced,suffix in ((True,'merged'),(False,'raw')):
            if (self.root/f'aggregate_105a_{suffix}.npz').exists():
                from .aggregate_4d import Merged4DIndex
                Merged4DIndex(self,coalesced=coalesced).build()
        return {'session':target_session,'new_interval_runs':new_runs,
                'extended_unchanged_runs':extended,'closed_runs':closed,
                'new_blob_bytes':added_blobs,**idx_result}

    def validate_structure(self):
        m=self.meta; rec=self.records; sessions=m['sessions'];n=len(sessions)
        if m['schema']!=SCHEMA or m['mode'] not in ('bst','morton'):
            raise ValueError('invalid interval database format')
        if not sessions or any(_timestamp(a)>=_timestamp(b) for a,b in zip(sessions,sessions[1:])):
            raise ValueError('invalid historical session order')
        last=defaultdict(list)
        for r in rec:
            if not 0<=r['start']<r['stop']<=n:raise ValueError('invalid record validity')
            last[r['key']].append((r['start'],r['stop']))
            _parse_xyz_key(r['key'])
        for key,intervals in last.items():
            intervals.sort()
            if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):
                raise ValueError('overlapping states for same tile '+key)
        self._load_index()
        return True

    def disk_stats(self):
        blobs=list((self.root/'blobs').glob('*'))
        totals={
            'blob_bytes':sum(p.stat().st_size for p in blobs),
            'blob_count':len(blobs),'record_bytes':(self.root/'records.json').stat().st_size,
            'metadata_bytes':(self.root/'metadata.json').stat().st_size,
            'index_bytes':(self.root/'index.npz').stat().st_size,
            'interval_records':len(self.records),
        }
        totals['total_bytes']=sum(totals[k] for k in ('blob_bytes','record_bytes','metadata_bytes','index_bytes'))
        return totals
