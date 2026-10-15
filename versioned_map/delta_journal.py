"""Step 10.4: exact, self-contained per-tile observed-endpoint delta journal.

One immutable compressed base snapshot plus transition journals of sorted, exact
uint64 voxel-code additions and removals.  No missing observation is inferred
as free space. The two address modes use IDENTICAL geometry/delta payloads:
Morton+time versus official BST-ID+time labels. IDs are *not* used to accelerate
search in this step, so differences are attributable only to address storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import struct
import time
import zlib
import numpy as np

from .store import VersionedStore, canonical, _bytes, _from_bytes, _region_filter
from .region_index import _morton3, manifest_hash
from .codec import Codec, Frame

FORMAT = 1
HEADER = struct.Struct('<II')


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, sort_keys=True, separators=(',', ':')))
    tmp.replace(path)


def _read_and_check_blob(path, digest):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != digest:
        raise IOError('delta journal blob hash mismatch: ' + digest)
    return raw


def _encode_delta(added, removed):
    a = canonical(added)
    r = canonical(removed)
    if len(a) >= (1 << 32) or len(r) >= (1 << 32):
        raise ValueError('too many changes in one tile')
    return zlib.compress(HEADER.pack(len(a), len(r)) + _bytes(a) + _bytes(r), level=6)


def _decode_delta(blob):
    raw = zlib.decompress(blob)
    if len(raw) < HEADER.size:
        raise IOError('truncated delta header')
    na, nr = HEADER.unpack_from(raw)
    if len(raw) != HEADER.size + 8 * (na + nr):
        raise IOError('invalid delta array lengths')
    added = _from_bytes(raw[HEADER.size:HEADER.size + na * 8])
    removed = _from_bytes(raw[HEADER.size + na * 8:])
    if len(added) != len(np.unique(added)) or len(removed) != len(np.unique(removed)):
        raise IOError('duplicate voxel in delta')
    if not np.array_equal(added, np.sort(added)) or not np.array_equal(removed, np.sort(removed)):
        raise IOError('noncanonical delta arrays')
    return added, removed


def _apply(base, added, removed):
    base = canonical(base)
    if len(np.intersect1d(added, removed, assume_unique=True)):
        raise IOError('delta adds and removes the same voxel')
    if len(removed) and not np.array_equal(np.intersect1d(base, removed, assume_unique=True), removed):
        raise IOError('delta removes voxel absent from prior map')
    if len(added) and len(np.intersect1d(base, added, assume_unique=True)):
        raise IOError('delta adds voxel already present')
    return np.union1d(np.setdiff1d(base, removed, assume_unique=True), added)


def _tile_overlap(key, bounds, side):
    x, y, z = map(int, key.split(','))
    x0, x1, y0, y1, z0, z1 = bounds
    return (x * side < x1 and (x + 1) * side > x0
            and y * side < y1 and (y + 1) * side > y0
            and z * side < z1 and (z + 1) * side > z0)


def _address(key, session, mode, codec, side):
    coords = tuple(map(int, key.split(',')))
    stamp = int(datetime.fromisoformat(session).replace(tzinfo=timezone.utc).timestamp())
    if mode == 'morton':
        return f'{_morton3(*coords):016x}:{stamp}'
    return codec.label(*(x * side for x in coords), stamp)


class DeltaJournal:
    def __init__(self, root):
        self.root = Path(root)
        self._metadata = None
        self._base = None
        self._transitions = None
        self._candidate_keys = None
        self._decode_cache = {}
        self.cache_decoded = False  # default: fair decompression-vs-snapshot benchmark

    @classmethod
    def build(cls, store: VersionedStore, root, address_mode='morton', *, verify=True, through_session=None):
        """Write an independent, complete archive. Existing destination is never overwritten."""
        if address_mode not in ('morton', 'bst'):
            raise ValueError('address_mode must be morton or bst')
        source = store.meta()
        if address_mode == 'bst' and source['codec'] != 'official':
            raise ValueError('BST mode requires the official-codec source database')
        out = cls(root)
        if out.root.exists():
            raise FileExistsError(f'choose an unused delta journal directory: {out.root}')
        out.root.mkdir(parents=True)
        (out.root / 'blobs').mkdir()
        clock = time.perf_counter()
        try:
            versions = store.all_versions()
            names = list(versions)
            if through_session is not None:
                if through_session not in versions:
                    raise KeyError(f'unknown through_session: {through_session}')
                names = names[:names.index(through_session) + 1]
            if not names:
                raise ValueError('cannot journal empty database')
            side = int(source['tile_side'])
            label_codec = Codec('official', Frame(**source['frame'])) if address_mode == 'bst' else None
            base_session = names[0]
            base = {}
            for key, ent in sorted(versions[base_session].items()):
                digest = ent['hash']
                content = _read_and_check_blob(store.root / 'blobs' / digest, digest)
                dest = out.root / 'blobs' / digest
                if not dest.exists():
                    dest.write_bytes(content)
                base[key] = {'hash': digest, 'count': int(ent['count']),
                             'address': _address(key, base_session, address_mode, label_codec, side)}
            _write_json(out.root / 'base.json', base)
            transition_info = []
            for before_name, after_name in zip(names, names[1:]):
                start = time.perf_counter()
                before = versions[before_name]
                after = versions[after_name]
                records = {}
                added_voxels = removed_voxels = touched = delta_bytes = new_blob_bytes = 0
                # Unchanged content-hash tiles are truly reused, with zero delta bytes.
                for key in sorted(set(before) | set(after)):
                    old = before.get(key)
                    new = after.get(key)
                    if old is not None and new is not None and old['hash'] == new['hash']:
                        continue
                    a = (_from_bytes(zlib.decompress(_read_and_check_blob(
                        store.root / 'blobs' / old['hash'], old['hash'])))
                         if old is not None else np.empty(0, dtype=np.uint64))
                    b = (_from_bytes(zlib.decompress(_read_and_check_blob(
                        store.root / 'blobs' / new['hash'], new['hash'])))
                         if new is not None else np.empty(0, dtype=np.uint64))
                    add = np.setdiff1d(b, a, assume_unique=True)
                    remove = np.setdiff1d(a, b, assume_unique=True)
                    if not len(add) and not len(remove):
                        continue
                    if verify and not np.array_equal(_apply(a, add, remove), b):
                        raise AssertionError('delta does not reconstruct target: ' + key)
                    blob = _encode_delta(add, remove)
                    digest = sha256(blob).hexdigest()
                    path = out.root / 'blobs' / digest
                    if not path.exists():
                        path.write_bytes(blob)
                        new_blob_bytes += len(blob)
                    records[key] = {'hash': digest, 'add': int(len(add)), 'remove': int(len(remove)),
                                    'address': _address(key, after_name, address_mode, label_codec, side)}
                    added_voxels += len(add)
                    removed_voxels += len(remove)
                    touched += 1
                    delta_bytes += len(blob)
                transition_path = out.root / 'transitions' / f'{before_name}__{after_name}.json'
                _write_json(transition_path, records)
                transition_info.append({'from': before_name, 'to': after_name, 'changed_tiles': touched,
                                        'added_observations': added_voxels, 'removed_observations': removed_voxels,
                                        'compressed_transition_payload_bytes_repeated': delta_bytes,
                                        'unique_delta_blob_bytes_added': new_blob_bytes,
                                        'transition_manifest_bytes':transition_path.stat().st_size,
                                        'incremental_write_bytes':new_blob_bytes+transition_path.stat().st_size,
                                        'build_ms': (time.perf_counter() - start) * 1000})
            _write_json(out.root / 'metadata.json', {
                'schema': FORMAT, 'source_meta': source, 'sessions': names,
                'address_mode': address_mode, 'base_session': base_session,
                'source_manifest_hashes': {s: manifest_hash(versions[s]) for s in names},
                'semantics': 'observed endpoint differences, NOT confirmed changes in occupancy',
                'transitions': transition_info,
                'build_ms': (time.perf_counter() - clock) * 1000,
                'journal_encoding': 'zlib6 of u32 additions/u32 removals and canonical u64 arrays',
                'address_role': 'journal labels only; spatial lookup uses canonical tile coordinates'
            })
            result = cls(out.root)
            if verify:
                result.validate_structure()
            return result
        except Exception:
            shutil.rmtree(out.root, ignore_errors=True)
            raise

    def append_from_store(self, store: VersionedStore, target_session, *, verify=True):
        """Ingest exactly one new chronological session into an existing journal.

        The source snapshot store is used only as the observation input. This
        delta archive remains self-contained after the append completes.
        Append cannot skip or rewrite a historical session.
        """
        start = time.perf_counter()
        oldmeta = dict(self.meta)
        versions = store.all_versions()
        names = list(versions)
        existing = list(oldmeta['sessions'])
        if existing != names[:len(existing)]:
            raise ValueError('source historical prefix differs from journal')
        if len(names) <= len(existing) or target_session != names[len(existing)]:
            raise ValueError('append requires exactly the next chronological session')
        if oldmeta['source_meta'] != store.meta():
            raise ValueError('source grid, frame or codec changed')
        from .region_index import manifest_hash
        for session in existing:
            if manifest_hash(versions[session]) != oldmeta['source_manifest_hashes'][session]:
                raise ValueError('historical source manifest changed')
        before_name = existing[-1]
        before, after = versions[before_name], versions[target_session]
        mode = oldmeta['address_mode']
        side = int(oldmeta['source_meta']['tile_side'])
        codec = Codec('official', Frame(**oldmeta['source_meta']['frame'])) if mode=='bst' else None
        records={}
        added_voxels=removed_voxels=touched=delta_bytes=new_blob_bytes=0
        for key in sorted(set(before)|set(after)):
            old, new = before.get(key),after.get(key)
            if old and new and old['hash']==new['hash']:
                continue
            a = (_from_bytes(zlib.decompress(_read_and_check_blob(
                store.root/'blobs'/old['hash'],old['hash'])))
                if old else np.empty(0,np.uint64))
            b = (_from_bytes(zlib.decompress(_read_and_check_blob(
                store.root/'blobs'/new['hash'],new['hash'])))
                if new else np.empty(0,np.uint64))
            add, remove = np.setdiff1d(b,a,assume_unique=True),np.setdiff1d(a,b,assume_unique=True)
            if not len(add) and not len(remove):continue
            if verify and not np.array_equal(_apply(a,add,remove),b):
                raise AssertionError('delta fails exact reconstruction')
            blob = _encode_delta(add,remove)
            digest=sha256(blob).hexdigest()
            path=self.root/'blobs'/digest
            if not path.exists():
                path.write_bytes(blob)
                new_blob_bytes+=len(blob)
            records[key]={'hash':digest,'add':int(len(add)),'remove':int(len(remove)),
                          'address':_address(key,target_session,mode,codec,side)}
            added_voxels+=len(add);removed_voxels+=len(remove);touched+=1;delta_bytes+=len(blob)
        transition_file=self.root/'transitions'/f'{before_name}__{target_session}.json'
        if transition_file.exists():
            raise FileExistsError('transition file already exists; inspect interrupted append')
        _write_json(transition_file,records)
        elapsed_ms=(time.perf_counter()-start)*1000
        update={'from':before_name,'to':target_session,'changed_tiles':touched,
                'added_observations':added_voxels,'removed_observations':removed_voxels,
                'compressed_transition_payload_bytes_repeated':delta_bytes,
                'unique_delta_blob_bytes_added':new_blob_bytes,
                'transition_manifest_bytes':transition_file.stat().st_size,
                'incremental_write_bytes':new_blob_bytes+transition_file.stat().st_size,
                'build_ms':elapsed_ms}
        newmeta=dict(oldmeta)
        newmeta['sessions'] = existing+[target_session]
        newmeta['source_manifest_hashes']={**oldmeta['source_manifest_hashes'],
                                            target_session:manifest_hash(after)}
        newmeta['transitions']=list(oldmeta['transitions'])+[update]
        newmeta['append_total_ms']=float(oldmeta.get('append_total_ms',0.))+elapsed_ms
        _write_json(self.root/'metadata.json',newmeta)
        self._metadata=None;self._transitions=None;self._candidate_keys=None;self._decode_cache.clear()
        if verify:self.validate_structure()
        return update

    @property
    def meta(self):
        if self._metadata is None:
            self._metadata = json.loads((self.root / 'metadata.json').read_text())
            if self._metadata['schema'] != FORMAT:
                raise ValueError('journal schema mismatch')
        return self._metadata

    @property
    def base(self):
        if self._base is None:
            self._base = json.loads((self.root / 'base.json').read_text())
        return self._base

    @property
    def transitions(self):
        if self._transitions is None:
            names = self.meta['sessions']
            self._transitions = [json.loads((self.root / 'transitions' / f'{a}__{b}.json').read_text())
                                 for a, b in zip(names, names[1:])]
        return self._transitions

    def validate_structure(self):
        meta = self.meta
        if meta['address_mode'] not in ('bst', 'morton'):
            raise ValueError('unknown address mode')
        if meta['base_session'] != meta['sessions'][0] or len(self.transitions) != len(meta['sessions']) - 1:
            raise ValueError('invalid journal sequence')
        side = int(meta['source_meta']['tile_side'])
        codec = Codec('official', Frame(**meta['source_meta']['frame'])) if meta['address_mode'] == 'bst' else None
        for sess, group in [(meta['base_session'], self.base)] + [(meta['sessions'][i+1], t)
                                                        for i, t in enumerate(self.transitions)]:
            for key, entry in group.items():
                if entry['address'] != _address(key, sess, meta['address_mode'], codec, side):
                    raise IOError('journal tile identifier mismatch')
                if not (self.root / 'blobs' / entry['hash']).is_file():
                    raise IOError('missing journal blob')
        return True

    def _candidate_tile_keys(self):
        if self._candidate_keys is None:
            all_keys = set(self.base)
            for step in self.transitions:
                all_keys.update(step)
            self._candidate_keys = sorted(all_keys)
        return self._candidate_keys

    def _decode_base(self, entry):
        digest = entry['hash']
        if digest not in self._decode_cache:
            raw = _read_and_check_blob(self.root / 'blobs' / digest, digest)
            v = _from_bytes(zlib.decompress(raw))
            if len(v) != entry['count'] or not np.array_equal(v, canonical(v)):
                raise IOError('base snapshot content mismatch')
            if self.cache_decoded: self._decode_cache[digest] = v
            return v
        return self._decode_cache[digest]

    def _decode_step(self, entry):
        digest = entry['hash']
        if digest not in self._decode_cache:
            raw = _read_and_check_blob(self.root / 'blobs' / digest, digest)
            value = _decode_delta(raw)
            if self.cache_decoded: self._decode_cache[digest] = value
        else:
            value = self._decode_cache[digest]
        if len(value[0]) != entry['add'] or len(value[1]) != entry['remove']:
            raise IOError('delta count does not match journal')
        return value

    def _tile_at(self, key, index):
        arr = self._decode_base(self.base[key]).copy() if key in self.base else np.empty(0, dtype=np.uint64)
        for record in self.transitions[:index]:
            if key in record:
                add, remove = self._decode_step(record[key])
                arr = _apply(arr, add, remove)
        return arr

    def query(self, session, bounds, *, stats=False):
        start = time.perf_counter()
        sessions = self.meta['sessions']
        if session not in sessions:
            raise KeyError(session)
        index = sessions.index(session)
        if len(bounds) != 6 or any(bounds[j] >= bounds[j+1] for j in (0,2,4)):
            raise ValueError('invalid half-open bounds')
        side = self.meta['source_meta']['tile_side']
        selection_start = time.perf_counter()
        touched = [k for k in self._candidate_tile_keys() if _tile_overlap(k, bounds, side)]
        selection_ms = (time.perf_counter() - selection_start) * 1000
        from synthetic.world import WorldSpec
        nx, ny, nz = self.meta['source_meta']['dims']
        v = self.meta['source_meta']['voxel_size_m']
        spec = WorldSpec(nx*v, ny*v, nz*v, v)
        parts=[]; read_bytes=0; transition_records=0
        for k in touched:
            if k in self.base:read_bytes += (self.root / 'blobs' / self.base[k]['hash']).stat().st_size
            for tr in self.transitions[:index]:
                if k in tr:
                    read_bytes += (self.root / 'blobs' / tr[k]['hash']).stat().st_size
                    transition_records += 1
            arr = self._tile_at(k, index)
            if len(arr):parts.append(_region_filter(arr, spec, bounds))
        result = canonical(np.concatenate(parts) if parts else np.empty(0, np.uint64))
        info = {'address_mode': self.meta['address_mode'], 'candidate_tiles': len(touched),
                'transition_records_applied': transition_records, 'compressed_bytes_accessed': read_bytes,
                'selection_ms': selection_ms, 'latency_ms': (time.perf_counter() - start) * 1000}
        return (result,info) if stats else result

    def reconstruct(self, session):
        nx, ny, nz = self.meta['source_meta']['dims']
        return self.query(session, (0, nx, 0, ny, 0, nz))

    def disk_stats(self):
        base = (self.root / 'base.json').stat().st_size
        manifest_files = list((self.root / 'transitions').glob('*.json'))
        transitions = sum(p.stat().st_size for p in manifest_files)
        payloads = list((self.root / 'blobs').iterdir())
        blobs = sum(p.stat().st_size for p in payloads)
        meta = (self.root / 'metadata.json').stat().st_size
        return {'base_manifest_bytes': base, 'transition_manifest_bytes': transitions,
                'metadata_bytes': meta, 'unique_blob_bytes': blobs, 'unique_blob_count': len(payloads),
                'total_bytes': base+transitions+blobs+meta}


def source_storage_controls(store: VersionedStore):
    """Measure independent snapshot (no dedup) and content-addressed store bytes.

    Both controls are the original store's exact zlib6 tile payloads; independent
    snapshots repeat blob bytes per version. No attempt to conflate with full
    occupancy mapping, or to include OS filesystem block/slack overhead.
    """
    versions = store.all_versions()
    per_hash = {p.name: p.stat().st_size for p in (store.root / 'blobs').iterdir()}
    total = sum(per_hash[item['hash']] for group in versions.values() for item in group.values())
    independent_manifest = len(json.dumps(versions, sort_keys=True, separators=(',', ':')).encode())
    current = store.disk_stats()
    return {'independent_repeated_compressed_tile_bytes': int(total),
            'independent_snapshot_manifest_bytes_comparable': independent_manifest,
            'independent_total_bytes_estimate': total+independent_manifest+store.meta_file.stat().st_size,
            'shared_snapshot_store_bytes': current['base_database_bytes'],
            'shared_snapshot_with_indexes_bytes': current['total_bytes'],
            'note': 'Independent snapshot byte total sums actual stored zlib tile blob sizes per session; this is a byte model, not a separately materialized filesystem.'}


def cumulative_storage_rows(store, journals):
    """Report per-version cumulative byte curves, including stored address overhead.

    Snapshot manifest costs are measured as per-version compact JSON length;
    journal metadata size is fixed at final size in each row (conservative,
    not a historical on-disk measurement at each ingest instant).
    """
    versions = store.all_versions();names = list(versions)
    blob_sizes={p.name:p.stat().st_size for p in (store.root/'blobs').iterdir()}
    distinct=set(); independent=shared_blobs=independent_manifest=shared_manifest=0
    journal_blob_seen={mode:set() for mode in journals}
    result=[]
    for i,session in enumerate(names):
        manifest=versions[session]
        blob_refs={x['hash'] for x in manifest.values()}
        independent += sum(blob_sizes[x['hash']] for x in manifest.values())
        independent_manifest += len(json.dumps(manifest,sort_keys=True,separators=(',', ':')).encode())
        new=blob_refs-distinct
        shared_blobs += sum(blob_sizes[h] for h in new)
        distinct.update(blob_refs)
        shared_manifest += len(json.dumps(manifest,sort_keys=True,separators=(',', ':')).encode())
        for mode, journal in journals.items():
            if session not in journal.meta['sessions']:continue
            ids=journal.base if i==0 else journal.transitions[i-1]
            for item in ids.values():journal_blob_seen[mode].add(item['hash'])
            disk_bytes=sum((journal.root/'blobs'/h).stat().st_size for h in journal_blob_seen[mode])
            manifest_bytes=(journal.root/'base.json').stat().st_size
            manifest_bytes+=sum((journal.root/'transitions'/f'{names[k]}__{names[k+1]}.json').stat().st_size
                                  for k in range(i))
            result.append({'session_index':i+1,'session':session,'method':mode+'_delta',
                           'cumulative_bytes':disk_bytes+manifest_bytes+(journal.root/'metadata.json').stat().st_size,
                           'payload_bytes':disk_bytes,'manifest_bytes':manifest_bytes,
                           'metadata_note':'final-size metadata included at all stages'})
        for method,bytes_total,payload,manifest_b in (
            ('independent_snapshots',independent+independent_manifest+store.meta_file.stat().st_size,independent,independent_manifest),
            ('content_addressed_snapshots',shared_blobs+shared_manifest+store.meta_file.stat().st_size,shared_blobs,shared_manifest)):
            result.append({'session_index':i+1,'session':session,'method':method,
                           'cumulative_bytes':bytes_total,'payload_bytes':payload,
                           'manifest_bytes':manifest_b,'metadata_note':'per-session modeled manifest overhead'})
    return result
