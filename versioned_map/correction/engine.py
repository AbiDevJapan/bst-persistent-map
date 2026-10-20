"""Correction-aware historical map engine.

Key invariants:
- Local source measurements are immutable; poses are revisioned.
- Each source contributes a SET of fine global voxels (not one count per point).
- Tile counters account for overlapping sources; removing one never erases another.
- Each commit touches old/new contribution tiles only; unaffected tile_versions persist.
- Exact source and tile state updates occur in a single SQLite transaction.
- Immutable compressed pages are content-addressed and reused across revisions.
- A committed revision is queryable without replaying correction history.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .voxel import (
    GRID_SIDE, Pose, canonical_blob, count_pack, count_unpack, decode_indices,
    encode_indices, filter_roi, full_mask, grid_roi_bounds, indices_to_grid,
    load_blob, ordered_grid, roi_tile_range, tile_keyed_indices,
)

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS revisions(
  revision INTEGER PRIMARY KEY, committed_utc TEXT NOT NULL, operation TEXT NOT NULL,
  sources_changed INTEGER NOT NULL, tiles_touched INTEGER NOT NULL,
  tiles_modified INTEGER NOT NULL, note TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(
  source_id TEXT PRIMARY KEY, robot_id TEXT NOT NULL,
  observed_at INTEGER NOT NULL, admitted_revision INTEGER NOT NULL,
  local_blob BLOB NOT NULL, local_sha256 TEXT NOT NULL, point_count INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS idx_sources_time ON sources(observed_at, admitted_revision);
CREATE TABLE IF NOT EXISTS pose_versions(
  source_id TEXT NOT NULL REFERENCES sources(source_id), revision INTEGER NOT NULL,
  matrix BLOB NOT NULL, PRIMARY KEY(source_id,revision));
CREATE TABLE IF NOT EXISTS source_tiles(
  source_id TEXT NOT NULL REFERENCES sources(source_id),
  tx INTEGER NOT NULL,ty INTEGER NOT NULL,tz INTEGER NOT NULL,
  indices BLOB NOT NULL,
  PRIMARY KEY(source_id,tx,ty,tz));
CREATE TABLE IF NOT EXISTS tile_counts(
  tx INTEGER NOT NULL,ty INTEGER NOT NULL,tz INTEGER NOT NULL,
  counters BLOB NOT NULL, PRIMARY KEY(tx,ty,tz));
CREATE TABLE IF NOT EXISTS pages(
  page_hash TEXT PRIMARY KEY,encoding TEXT NOT NULL,
  payload BLOB NOT NULL,voxel_count INTEGER NOT NULL, raw_bit_count INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS tile_versions(
  tx INTEGER NOT NULL,ty INTEGER NOT NULL,tz INTEGER NOT NULL,
  from_revision INTEGER NOT NULL,to_revision INTEGER,
  page_hash TEXT REFERENCES pages(page_hash),
  PRIMARY KEY(tx,ty,tz,from_revision));
CREATE INDEX IF NOT EXISTS idx_tile_versions_roi ON tile_versions(tx,ty,tz,from_revision,to_revision);
CREATE INDEX IF NOT EXISTS idx_tile_versions_current ON tile_versions(to_revision);
"""

@dataclass(frozen=True)
class UpdateResult:
    revision: int
    operation: str
    sources_changed: int
    tiles_touched: int
    tiles_modified: int
    local_update_ms: float

    def as_dict(self): return vars(self)


class CorrectionMap:
    """Persistent signed-grid occupancy history with batched pose corrections.

    All times are integer UTC seconds. `observed_at` means source acquisition time;
    `revision` means when the service learned/committed the source or correction.
    """

    def __init__(self, filename: str | Path, *, voxel_size: float = .5,
                 origin: tuple[float,float,float] = (0,0,0)):
        self.filename=Path(filename)
        self.filename.parent.mkdir(parents=True,exist_ok=True)
        self.con=sqlite3.connect(str(self.filename),timeout=30)
        self.con.execute('PRAGMA journal_mode=WAL')
        self.con.execute('PRAGMA synchronous=FULL')
        self.con.executescript(SCHEMA)
        row=self.con.execute('SELECT value FROM settings WHERE key=?',('map_config',)).fetchone()
        config={'voxel_size':float(voxel_size),'origin':list(map(float,origin)), 'tile_side':GRID_SIDE,
                'format':'correction-map-1'}
        if row is None:
            if voxel_size <= 0 or not np.isfinite(voxel_size) or not np.isfinite(config['origin']).all():
                raise ValueError('invalid global grid configuration')
            with self.con:
                self.con.execute('INSERT INTO settings VALUES(?,?)',('map_config',json.dumps(config,sort_keys=True)))
                self.con.execute('INSERT OR IGNORE INTO revisions VALUES(?,?,?,?,?,?,?)',
                                 (0,'1970-01-01T00:00:00Z','empty',0,0,0,''))
            self.config=config
        else:
            saved=json.loads(row[0]); self.config=saved
            # Reopening a DB must never silently modify the coordinate frame.
            # Default constructor is intended to read the stored frame.
            if (float(voxel_size),tuple(map(float,origin))) != (.5,(0.,0.,0.)) and (
                float(voxel_size) != saved['voxel_size'] or list(map(float,origin)) != saved['origin']):
                raise ValueError('the requested grid differs from the stored global frame')
        self.voxel_size=float(self.config['voxel_size'])
        self.origin=tuple(self.config['origin'])

    def close(self): self.con.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
    @property
    def revision(self) -> int:
        return int(self.con.execute('SELECT MAX(revision) FROM revisions').fetchone()[0])

    def _validate_rev(self, revision: int | None) -> int:
        r=self.revision if revision is None else int(revision)
        if r < 0 or r > self.revision: raise ValueError('revision not committed')
        return r

    def _current_counts(self, tile) -> dict[int,int]:
        row=self.con.execute('SELECT counters FROM tile_counts WHERE tx=? AND ty=? AND tz=?',tile).fetchone()
        return count_unpack(row[0] if row else None)

    def _current_page_hash(self, tile):
        row=self.con.execute('SELECT page_hash FROM tile_versions WHERE tx=? AND ty=? AND tz=? AND to_revision IS NULL',tile).fetchone()
        return row[0] if row else None

    def _commit_changed_tiles(self, updates: dict[tuple[int,int,int],tuple[list[set[int]],list[set[int]]]], r: int) -> tuple[int,int]:
        touched=modified=0
        for tile,(remove,add) in sorted(updates.items()):
            touched += 1
            counts=self._current_counts(tile)
            prior=frozenset(counts)
            for contribution in remove:
                for ix in contribution:
                    if ix not in counts: raise RuntimeError(f'missing contribution in {tile}, index {ix}')
                    counts[ix]-=1
                    if counts[ix] == 0: del counts[ix]
            for contribution in add:
                for ix in contribution:
                    counts[ix]=counts.get(ix,0)+1
            if counts:
                self.con.execute('INSERT OR REPLACE INTO tile_counts VALUES(?,?,?,?)',
                                 (*tile,count_pack(counts)))
            else:
                self.con.execute('DELETE FROM tile_counts WHERE tx=? AND ty=? AND tz=?',tile)
            next_occupied=frozenset(counts)
            if prior == next_occupied: continue  # correction did not change union occupancy
            modified += 1
            page_hash=None
            if next_occupied:
                encoding,payload,canonical_hash,n=encode_indices(np.fromiter(next_occupied,dtype=np.uint16))
                page_hash=canonical_hash.hex()
                self.con.execute('INSERT OR IGNORE INTO pages VALUES(?,?,?,?,?)',
                                 (page_hash,encoding,payload,n,32768))
            self.con.execute('UPDATE tile_versions SET to_revision=? '
                             'WHERE tx=? AND ty=? AND tz=? AND to_revision IS NULL', (r,*tile))
            self.con.execute('INSERT INTO tile_versions VALUES(?,?,?,?,?,?)',(*tile,r,None,page_hash))
        return touched,modified

    def _new_revision(self,operation,sources_changed,updates,note,t0):
        r=self.revision+1
        touched,modified=self._commit_changed_tiles(updates,r)
        self.con.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?,?)',
                         (r,datetime.datetime.now(datetime.timezone.utc).isoformat(),operation,
                          sources_changed,touched,modified,str(note)))
        return UpdateResult(r,operation,sources_changed,touched,modified,(time.perf_counter()-t0)*1000.)

    def add_sources(self, records: Iterable[Mapping], note: str='') -> UpdateResult:
        """Insert several immutable observations as one atomic historical revision.

        Each record: source_id, robot_id, observed_at (UTC seconds), points Nx3
        in source frame, and pose (Pose or 4x4 source->global transform).
        """
        entries=list(records)
        if not entries: raise ValueError('no sources to add')
        ids=[str(rec['source_id']) for rec in entries]
        if len(set(ids)) != len(ids): raise ValueError('duplicate source IDs in batch')
        t0=time.perf_counter()
        with self.con:
            self.con.execute('BEGIN IMMEDIATE')
            r=self.revision+1
            for rec in entries:
                sid=str(rec['source_id']); robot=str(rec['robot_id']); observed=int(rec['observed_at'])
                if not sid or not robot: raise ValueError('source_id and robot_id required')
                from .voxel import _points
                pts=_points(rec['points'])
                pose=rec['pose'] if isinstance(rec['pose'],Pose) else Pose(rec['pose'])
                blob=canonical_blob(pts)
                self.con.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?)',
                                 (sid,robot,observed,r,blob,hashlib.sha256(blob).hexdigest(),len(pts)))
                self.con.execute('INSERT INTO pose_versions VALUES(?,?,?)',(sid,r,pose.matrix.astype('<f8').tobytes()))
                contrib=tile_keyed_indices(pts,pose,self.voxel_size,self.origin)
                for tile,ii in contrib.items():
                    self.con.execute('INSERT INTO source_tiles VALUES(?,?,?,?,?)',
                                     (sid,*tile,ii.tobytes()))
            # Gather per-source contribution multiplicities rather than set union.
            # Sequential counter application inside this transaction preserves overlaps.
            updates=self._gather_new_additions(ids)
            result=self._new_revision('ingest',len(entries),updates,note,t0)
        return replace(result,local_update_ms=(time.perf_counter()-t0)*1000.)

    def _gather_new_additions(self,source_ids):
        updates={}
        for sid in source_ids:
            rows=self.con.execute('SELECT tx,ty,tz,indices FROM source_tiles WHERE source_id=?',(sid,)).fetchall()
            for tx,ty,tz,blob in rows:
                tile=(int(tx),int(ty),int(tz))
                updates.setdefault(tile,([],[]))[1].append(set(map(int,np.frombuffer(blob,dtype='<u2'))))
        # convert to list of additions; _commit_changed_tiles handles multiplicity
        return updates

    def correct_poses(self, changes: Mapping[str,Pose|np.ndarray], note: str='') -> UpdateResult:
        if not changes: raise ValueError('empty corrections')
        t0=time.perf_counter()
        with self.con:
            self.con.execute('BEGIN IMMEDIATE')
            r=self.revision+1
            updates: dict[tuple[int,int,int],tuple[list[set[int]],list[set[int]]]]={}
            for sid,new_pose in sorted(changes.items()):
                new_pose=new_pose if isinstance(new_pose,Pose) else Pose(new_pose)
                row=self.con.execute('SELECT local_blob FROM sources WHERE source_id=?',(sid,)).fetchone()
                if row is None: raise KeyError(f'unknown source: {sid}')
                pts=load_blob(row[0])
                old_rows=self.con.execute('SELECT tx,ty,tz,indices FROM source_tiles WHERE source_id=?',(sid,)).fetchall()
                old={(int(x),int(y),int(z)):set(map(int,np.frombuffer(v,dtype='<u2'))) for x,y,z,v in old_rows}
                new=tile_keyed_indices(pts,new_pose,self.voxel_size,self.origin)
                for tile in old.keys() | new.keys():
                    pair=updates.setdefault(tile,([],[]))
                    if tile in old: pair[0].append(old[tile])
                    if tile in new: pair[1].append(set(map(int,new[tile])))
                self.con.execute('INSERT INTO pose_versions VALUES(?,?,?)',
                                 (sid,r,new_pose.matrix.astype('<f8').tobytes()))
                self.con.execute('DELETE FROM source_tiles WHERE source_id=?',(sid,))
                for tile,ii in new.items():
                    self.con.execute('INSERT INTO source_tiles VALUES(?,?,?,?,?)',(sid,*tile,ii.tobytes()))
            result=self._new_revision('pose_correction',len(changes),updates,note,t0)
        return replace(result,local_update_ms=(time.perf_counter()-t0)*1000.)

    def _query_tiles(self,revision,bounds):
        (xl,xh),(yl,yh),(zl,zh)=roi_tile_range(bounds)
        return self.con.execute('''SELECT v.tx,v.ty,v.tz,p.encoding,p.payload,p.voxel_count,v.page_hash
            FROM tile_versions v JOIN pages p ON p.page_hash=v.page_hash
            WHERE v.tx BETWEEN ? AND ? AND v.ty BETWEEN ? AND ? AND v.tz BETWEEN ? AND ?
              AND v.from_revision <= ? AND (v.to_revision IS NULL OR v.to_revision > ?)
            ORDER BY v.tx,v.ty,v.tz''', (xl,xh,yl,yh,zl,zh,revision,revision))

    def query(self,bounds,revision:int|None=None, verify:bool=False) -> np.ndarray:
        r=self._validate_rev(revision); b=grid_roi_bounds(bounds)
        pieces=[]
        for tx,ty,tz,enc,payload,n,digest in self._query_tiles(r,b):
            idx=decode_indices(enc,payload)
            if verify:
                if len(idx)!=n or encode_indices(idx)[2].hex()!=digest:
                    raise IOError('corrupt content-addressed map page')
            vox=filter_roi(indices_to_grid(idx,(tx,ty,tz)),b)
            if len(vox): pieces.append(vox)
        if not pieces: return np.empty((0,3),dtype=np.int64)
        # Pages cover disjoint regions and each page contains unique voxels.
        # A lexicographic sort is sufficient; global np.unique is unnecessary.
        coords=np.concatenate(pieces,axis=0)
        return coords[np.lexsort((coords[:,2],coords[:,1],coords[:,0]))]

    def _source_pose_asof(self,sid,revision):
        row=self.con.execute('SELECT matrix FROM pose_versions WHERE source_id=? AND revision<=? '
                             'ORDER BY revision DESC LIMIT 1',(sid,revision)).fetchone()
        if row is None: raise RuntimeError('missing pose history')
        return Pose(np.frombuffer(row[0],dtype='<f8').reshape(4,4).copy())

    def full_rebuild(self,bounds=None,revision:int|None=None,observed_until:int|None=None,
                     pose_overrides:Mapping[str,Pose|np.ndarray]|None=None) -> np.ndarray:
        """Independent correctness oracle, including retrospective observation-time queries."""
        r=self._validate_rev(revision)
        if observed_until is None:
            rows=self.con.execute('SELECT source_id,local_blob FROM sources WHERE admitted_revision<=?',(r,))
        else:
            rows=self.con.execute('SELECT source_id,local_blob FROM sources WHERE admitted_revision<=? AND observed_at<=?',
                                  (r,int(observed_until)))
        seen:set[tuple[int,int,int]]=set()
        from .voxel import voxelize
        for sid,raw in rows:
            pose=self._source_pose_asof(sid,r)
            if pose_overrides and sid in pose_overrides:
                override=pose_overrides[sid]
                pose=override if isinstance(override,Pose) else Pose(override)
            v=voxelize(load_blob(raw),pose,self.voxel_size,self.origin)
            if bounds is not None: v=filter_roi(v,bounds)
            seen.update(map(tuple,v.tolist()))
        if not seen: return np.empty((0,3),dtype=np.int64)
        return ordered_grid(np.asarray(sorted(seen),dtype=np.int64))

    def query_observed_history(self,bounds,observed_until:int,as_of_revision:int|None=None)->np.ndarray:
        """Bitemporal: observations through time T, with poses known at revision R.

        Initial implementation intentionally uses the full rebuild oracle; this
        is a precise functionality baseline, NOT an optimized bitemporal index.
        """
        return self.full_rebuild(bounds,as_of_revision,observed_until)

    def diff(self,bounds,left_revision:int,right_revision:int)->dict:
        """Exact changed-voxel counts using XOR/AND on packed occupancy tiles.

        Computes counts without materializing global XYZ voxel arrays.
        """
        from .voxel import N_BYTES
        b=grid_roi_bounds(bounds)
        left=self._validate_rev(left_revision);right=self._validate_rev(right_revision)
        (xl,xh),(yl,yh),(zl,zh)=roi_tile_range(b)
        # Get tile unions at two versions; the occupancy is already direct-read.
        def page_map(rev):
            return {(tx,ty,tz):full_mask(enc,payload) for tx,ty,tz,enc,payload,_,_ in self._query_tiles(rev,b)}
        a=page_map(left);c=page_map(right)
        pop=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(axis=1).astype(np.uint8)
        added=removed=0
        for key in a.keys() | c.keys():
            prev=a.get(key,np.zeros(N_BYTES,dtype=np.uint8)); nxt=c.get(key,np.zeros(N_BYTES,dtype=np.uint8))
            # Apply a bit mask for partial boundary tiles. Full tiles need none.
            mins=np.asarray(key,dtype=np.int64)*GRID_SIDE
            if (b[0]<=mins[0] and b[1]>=mins[0]+GRID_SIDE and
                b[2]<=mins[1] and b[3]>=mins[1]+GRID_SIDE and
                b[4]<=mins[2] and b[5]>=mins[2]+GRID_SIDE):
                roi_mask=None
            else:
                roi_mask=np.zeros(N_BYTES,dtype=np.uint8)
                lx0=max(0,b[0]-mins[0]); lx1=min(GRID_SIDE,b[1]-mins[0]);
                ly0=max(0,b[2]-mins[1]); ly1=min(GRID_SIDE,b[3]-mins[1]);
                lz0=max(0,b[4]-mins[2]); lz1=min(GRID_SIDE,b[5]-mins[2]);
                iz,iy,ix=np.mgrid[lz0:lz1,ly0:ly1,lx0:lx1]
                ids=(ix+32*iy+1024*iz).ravel().astype(np.int64)
                np.bitwise_or.at(roi_mask,ids//8,(1 << (ids%8)).astype(np.uint8))
            add=np.bitwise_and(nxt,np.bitwise_not(prev))
            rem=np.bitwise_and(prev,np.bitwise_not(nxt))
            if roi_mask is not None: add &= roi_mask; rem &= roi_mask
            added+=int(pop[add].sum());removed+=int(pop[rem].sum())
        return {'added_observations':added,'no_longer_observed':removed,'symmetric_difference':added+removed,
                'left_revision':left,'right_revision':right}

    def stats(self)->dict:
        con=self.con
        def val(sql):return int(con.execute(sql).fetchone()[0] or 0)
        physical_bytes=sum(p.stat().st_size for p in (self.filename,
                             Path(str(self.filename)+'-wal'),Path(str(self.filename)+'-shm')) if p.exists())
        return {'revision':self.revision,'sources':val('SELECT COUNT(*) FROM sources'),
                'revisions':val('SELECT COUNT(*) FROM revisions')-1,
                'historical_tile_records':val('SELECT COUNT(*) FROM tile_versions'),
                'current_nonempty_tiles':val('SELECT COUNT(*) FROM tile_versions WHERE to_revision IS NULL AND page_hash IS NOT NULL'),
                'unique_compressed_pages':val('SELECT COUNT(*) FROM pages'),
                'compressed_geometry_bytes':val('SELECT SUM(LENGTH(payload)) FROM pages'),
                'immutable_observation_bytes':val('SELECT SUM(LENGTH(local_blob)) FROM sources'),
                'physical_sqlite_bytes_including_wal':physical_bytes,
                'main_file_bytes':self.filename.stat().st_size if self.filename.exists() else 0}

    def revision_log(self):
        return [dict(zip(('revision','committed_utc','operation','sources_changed','tiles_touched',
                          'tiles_modified','note'),row)) for row in self.con.execute('SELECT * FROM revisions ORDER BY revision')]

    def verify(self,revision:int|None=None,bounds=None):
        r=self._validate_rev(revision)
        if bounds is None:
            # Exact grid union bounds from all historical records; signed grid.
            rows=self.con.execute('SELECT MIN(tx),MAX(tx),MIN(ty),MAX(ty),MIN(tz),MAX(tz) FROM tile_versions').fetchone()
            if rows[0] is None: return True
            bounds=(rows[0]*32,(rows[1]+1)*32,rows[2]*32,(rows[3]+1)*32,rows[4]*32,(rows[5]+1)*32)
        actual=self.query(bounds,r,verify=True)
        expected=self.full_rebuild(bounds,r)
        if not np.array_equal(actual,expected):
            raise AssertionError(f'map mismatch at revision {r}: {len(actual)} vs {len(expected)}')
        return True
