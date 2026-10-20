"""Step 11.1: reproducible NCLT scan provenance and controlled pose-error experiment.

Raw synchronized NCLT LiDAR points are in the mobile platform body frame.
Ground truth is the NCLT local metric global frame (Z down).  This adapter
reuses the existing project's tested decoder and nearest-pose matching rules.

IMPORTANT: the prepared points are filtered local XYZ, not raw byte-exact LiDAR
and not original intensity/laser-id.  Original file checksums are recorded.
The perturbation is a world-frame SE(3) left multiplication: T_bad=D@T_gt.
Corrections restore T_gt. This simulates known pose error; it does not run SLAM.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np

from realdata.nclt import (
    discover_nclt_session, load_ground_truth, _NearestPoseMatcher,
    _list_scan_files, _timestamp_from_scan, read_velodyne_sync_bin, _rot_zyx,
)
from .engine import CorrectionMap
from .voxel import Pose


def _hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path,obj) -> None:
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def _delta_matrix(translation, yaw_deg: float) -> np.ndarray:
    if len(translation)!=3 or not np.all(np.isfinite(translation)) or not math.isfinite(yaw_deg):
        raise ValueError('translation and yaw must be finite')
    theta=math.radians(yaw_deg)
    co,si=math.cos(theta),math.sin(theta)
    D=np.eye(4,dtype=np.float64)
    D[:3,:3]=[[co,-si,0],[si,co,0],[0,0,1]]
    D[:3,3]=translation
    return D


@dataclass(frozen=True)
class NCLTSourceConfig:
    scan_stride: int = 20
    max_scans_per_session: int = 64
    min_range_m: float = 2.0
    max_range_m: float = 80.0
    max_pose_gap_us: int = 100_000
    max_points_per_scan: int = 0  # 0 retains all filtered local points
    corrupt_from: int = 0
    corrupt_count: int = 8
    corrupt_translation_m: tuple[float,float,float] = (1.,0.,0.)
    corrupt_yaw_deg: float = 0.
    robot_prefix: str = 'nclt'

    def __post_init__(self):
        if self.scan_stride<=0 or self.max_scans_per_session<=0 or self.max_pose_gap_us<0:
            raise ValueError('invalid scan selection or pose-gap thresholds')
        if not (0<=self.min_range_m<self.max_range_m) or self.max_points_per_scan<0:
            raise ValueError('invalid point/range settings')
        if self.corrupt_from<0 or self.corrupt_count<=0:
            raise ValueError('correction segment needs nonnegative start and positive length')
        if not self.robot_prefix or '/' in self.robot_prefix:
            raise ValueError('robot prefix cannot be empty or contain /')
        _delta_matrix(self.corrupt_translation_m,self.corrupt_yaw_deg)
        if np.array_equal(_delta_matrix(self.corrupt_translation_m,self.corrupt_yaw_deg),np.eye(4)):
            raise ValueError('pose perturbation must be nonidentity')


def prepare_nclt(data_root: str|Path, sessions: list[str], output: str|Path, config: NCLTSourceConfig,
                 *, overwrite:bool=False) -> dict:
    """Prepare per-scan local XYZ, ground-truth and corrupted pose manifests.

    This generates an externally inspectable chain of provenance; does NOT
    change the original NCLT data and does NOT invent source dependencies from
    Step 10 globally merged maps. No writes into the reference SQLite engine.
    """
    if not sessions or len(set(sessions))!=len(sessions):
        raise ValueError('provide distinct NCLT sessions')
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f'{output} exists and is nonempty; choose a new output or --overwrite')
    # Preflight every GT file before creating output files. A single nonfinite
    # GT sample must not invalidate an otherwise usable trajectory; the shared
    # loader discards bad rows but NEVER interpolates over missing intervals.
    paths = [discover_nclt_session(data_root, s) for s in sessions]
    prepared_ground_truth = []
    for source_paths in paths:
        gt, gt_stats = load_ground_truth(source_paths.gt_csv, return_stats=True)
        prepared_ground_truth.append((source_paths, gt, gt_stats))

    output.mkdir(parents=True,exist_ok=True)
    scans_dir=output/'scans';scans_dir.mkdir(exist_ok=True)
    delta=_delta_matrix(config.corrupt_translation_m,config.corrupt_yaw_deg)
    rows=[];truth_rows=[];corrections={};quality=[];count=0
    skipped_gap=skipped_empty=0
    gt_quality=[]
    for paths_for_session, gt, gt_stats in prepared_ground_truth:
        gt_quality.append(dict(session=paths_for_session.session_name, **gt_stats))
        matcher=_NearestPoseMatcher(gt)
        scans=_list_scan_files(paths_for_session.scan_dir)[::config.scan_stride]
        scans=scans[:config.max_scans_per_session]
        selected=0
        for scan in scans:
            ts_us=_timestamp_from_scan(scan)
            near,gap=matcher.nearest(ts_us)
            if gap>config.max_pose_gap_us:
                skipped_gap+=1;continue
            xyz=read_velodyne_sync_bin(scan)
            if len(xyz):
                lengths=np.linalg.norm(xyz,axis=1)
                xyz=xyz[(lengths>=config.min_range_m)&(lengths<=config.max_range_m)]
            if not len(xyz):
                skipped_empty+=1;continue
            if config.max_points_per_scan and len(xyz)>config.max_points_per_scan:
                # Deterministic evenly spaced subsample, identical before and after correction.
                ids=(np.arange(config.max_points_per_scan,dtype=np.int64)*len(xyz)//config.max_points_per_scan)
                xyz=xyz[ids]
            gt_mat=np.eye(4,dtype=np.float64)
            gt_mat[:3,:3]=_rot_zyx(near[4],near[5],near[6])
            gt_mat[:3,3]=near[1:4]
            Pose(gt_mat)  # Reject invalid rotation before writing a source.
            corrupted=(config.corrupt_from<=count<config.corrupt_from+config.corrupt_count)
            initial=(delta@gt_mat if corrupted else gt_mat)
            Pose(initial)
            sid=f'nclt/{paths_for_session.session_name}/{ts_us}'
            filename=f'{paths_for_session.session_name}_{ts_us}.npz'
            target=scans_dir/filename
            np.savez_compressed(target,points=np.ascontiguousarray(xyz,dtype=np.float64))
            relative=(Path('scans')/filename).as_posix()
            base={'source_id':sid,'robot_id':f'{config.robot_prefix}-{paths_for_session.session_name}',
                  'observed_at':ts_us//1_000_000,'observed_at_us':ts_us,
                  'npz':relative,'scan_file':str(scan.resolve()),
                  'raw_scan_sha256':_hash_file(scan),'prepared_sha256':_hash_file(target),
                  'pose_gap_us':int(gap),'points_kept':int(len(xyz)),
                  'groundtruth_file':str(paths_for_session.gt_csv.resolve())}
            rows.append(dict(base,pose=initial.tolist()))
            truth_rows.append(dict(base,pose=gt_mat.tolist()))
            quality.append(dict(source_id=sid,session=paths_for_session.session_name,scan_time_us=ts_us,
                                gt_time_us=int(near[0]),gap_us=int(gap),points=int(len(xyz)),
                                perturbed=int(corrupted)))
            if corrupted: corrections[sid]=gt_mat.tolist()
            selected+=1;count+=1
    if not rows:
        raise RuntimeError(
            'No scans prepared. '
            f'Skipped {skipped_gap} selected scans for ground-truth gaps above '
            f'{config.max_pose_gap_us} us and {skipped_empty} empty scans. '
            'Finite ground-truth rows by session: ' +
            '; '.join(f"{info['session']}={info['rows_valid']}/{info['rows_total']}" for info in gt_quality) +
            '. Check scan filename timestamps against GT microseconds. '
            'Do not relax --max-pose-gap-us without evaluating trajectory gaps.'
        )
    if not corrections:raise RuntimeError('no scans in corruption segment; adjust --corrupt-from/--corrupt-count')
    _write_json(output/'manifest_initial.json',rows)
    _write_json(output/'manifest_corrected.json',truth_rows)
    _write_json(output/'corrections.json',corrections)
    with (output/'scan_provenance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(quality[0]));w.writeheader();w.writerows(quality)
    meta={'step':'11.1','data_root':str(Path(data_root).resolve()),'sessions':sessions,
          'config':asdict(config),'sources':len(rows),'correction_sources':len(corrections),
          'skipped_pose_gap':skipped_gap,'skipped_empty':skipped_empty,
          'groundtruth_quality':gt_quality,
          'measured_pose_gap_us_p50':float(np.median([r['gap_us'] for r in quality])),
          'measured_pose_gap_us_p95':float(np.percentile([r['gap_us'] for r in quality],95)),
          'timestamps':'scan filenames are microseconds; correction engine observed_at is UTC seconds; full microseconds retained in provenance',
          'frame':'native NCLT body XYZ to NCLT local metric world (Z down), nearest ground truth, no interpolation',
          'perturbation':'controlled world-frame D @ T_gt, corrected to T_gt; not an estimated loop closure',
          'geometry':'filtered motion-compensated local XYZ; original raw LiDAR files identified by SHA256',
          'output_files':['manifest_initial.json','manifest_corrected.json','corrections.json','scan_provenance.csv']}
    _write_json(output/'metadata.json',meta)
    return meta


def _get_records(manifest:Path)->Iterator[dict]:
    items=json.loads(manifest.read_text())
    if not isinstance(items,list) or not items:raise ValueError('manifest must be a nonempty list')
    seen=set()
    for item in items:
        sid=str(item['source_id'])
        if sid in seen:raise ValueError('duplicate source ID')
        seen.add(sid)
        npz=Path(item['npz'])
        if not npz.is_absolute():npz=manifest.parent/npz
        if 'prepared_sha256' in item and _hash_file(npz)!=item['prepared_sha256']:
            raise IOError(f'prepared file SHA256 mismatch: {npz}')
        with np.load(npz,allow_pickle=False) as z:
            pts=np.array(z['points'],copy=True)
        yield {'source_id':sid,'robot_id':item['robot_id'],'observed_at':int(item['observed_at']),
               'points':pts,'pose':Pose(np.array(item['pose'],dtype=np.float64))}


def _ingest(db:CorrectionMap,manifest:Path,batch_size:int)->dict:
    n=batches=0;tile_touched=0;ingest_ms=0.
    batch=[]
    for rec in _get_records(manifest):
        batch.append(rec)
        if len(batch)>=batch_size:
            result=db.add_sources(batch,note='Step 11.1 NCLT provenance ingest')
            n+=len(batch);batches+=1;tile_touched+=result.tiles_touched;ingest_ms+=result.local_update_ms
            batch=[]
    if batch:
        result=db.add_sources(batch,note='Step 11.1 NCLT provenance ingest')
        n+=len(batch);batches+=1;tile_touched+=result.tiles_touched;ingest_ms+=result.local_update_ms
    return {'sources':n,'batches':batches,'touched_tiles_during_ingest':tile_touched,'db_batch_time_ms':ingest_ms}


def _attach_provenance(db:CorrectionMap,prepared_root:Path)->int:
    """Persist source-level NCLT provenance alongside immutable source points.

    This auxiliary ledger is keyed to source IDs and never changes map
    geometry or revision semantics. Ground truth remains experiment reference.
    """
    initial=json.loads((prepared_root/'manifest_initial.json').read_text())
    corrected=json.loads((prepared_root/'manifest_corrected.json').read_text())
    gt={r['source_id']:r for r in corrected}
    if set(gt)!={r['source_id'] for r in initial}:
        raise ValueError('initial/corrected provenance manifests have mismatched source IDs')
    db.con.execute("""CREATE TABLE IF NOT EXISTS nclt_source_provenance(
       source_id TEXT PRIMARY KEY REFERENCES sources(source_id),
       observed_at_us INTEGER NOT NULL,
       source_scan_path TEXT NOT NULL,
       source_scan_sha256 TEXT NOT NULL,
       prepared_sha256 TEXT NOT NULL,
       groundtruth_file TEXT NOT NULL,
       nearest_pose_gap_us INTEGER NOT NULL,
       reference_pose BLOB NOT NULL)""")
    with db.con:
        for entry in initial:
            sid=entry['source_id']
            if db.con.execute('SELECT 1 FROM sources WHERE source_id=?',(sid,)).fetchone() is None:
                raise ValueError(f'source absent when attaching provenance: {sid}')
            reference=Pose(np.asarray(gt[sid]['pose'],dtype=np.float64))
            db.con.execute('INSERT INTO nclt_source_provenance VALUES(?,?,?,?,?,?,?,?)',
                (sid,int(entry['observed_at_us']),entry['scan_file'],
                 entry['raw_scan_sha256'],entry['prepared_sha256'],entry['groundtruth_file'],
                 int(entry['pose_gap_us']),reference.matrix.astype('<f8').tobytes()))
    return len(initial)

def _all_bounds(arrays:list[np.ndarray])->tuple[int,int,int,int,int,int]:
    valid=[a for a in arrays if len(a)]
    if not valid:return (0,1,0,1,0,1)
    lo=np.min([a.min(axis=0) for a in valid],axis=0)
    hi=np.max([a.max(axis=0) for a in valid],axis=0)+1
    return (int(lo[0]),int(hi[0]),int(lo[1]),int(hi[1]),int(lo[2]),int(hi[2]))


def _hash_voxels(a:np.ndarray)->str:
    return hashlib.sha256(np.ascontiguousarray(a,dtype='<i8').tobytes()).hexdigest()


def benchmark_nclt(prepared_root:str|Path,output:str|Path,*,voxel_size:float=.5,
                   origin=(0.,0.,0.),batch_size:int=16,repeats:int=3,
                   full_disk_rebuild:bool=True)->dict:
    """Controlled NCLT correction trial with all three correctness paths.

    Base DB is ingested once and cloned *outside* update timings. Per trial:
      A. immutable-source full rebuild in memory using corrected poses
      B. incremental correction+SQLite commit on a byte-identical DB clone
      C. optional separate full corrected database materialization from the
         exact same input scans, with complete SQLite commits.
    Comparing A against B isn't an equivalent timed boundary; report separately.
    """
    if repeats<=0 or batch_size<=0:raise ValueError('repeats and batch size must be positive')
    prepared_root=Path(prepared_root).resolve();output=Path(output).resolve()
    initial=prepared_root/'manifest_initial.json'
    corrected=prepared_root/'manifest_corrected.json'
    changes=json.loads((prepared_root/'corrections.json').read_text())
    if not changes:raise ValueError('no controlled corrections in this dataset')
    updated={sid:Pose(m) for sid,m in changes.items()}
    output.mkdir(parents=True,exist_ok=True)
    base_path=output/'initial_map.sqlite'
    if base_path.exists():raise FileExistsError(f'{base_path} already exists; choose a new output')
    t=time.perf_counter()
    with CorrectionMap(base_path,voxel_size=voxel_size,origin=tuple(origin)) as db:
        ingest_stats=_ingest(db,initial,batch_size)
        provenance_rows=_attach_provenance(db,prepared_root)
        base_revision=db.revision
        base_stats=db.stats()
        # Validate correction poses with actual source records *before* the benchmark.
        missing=set(changes)-{r[0] for r in db.con.execute('SELECT source_id FROM sources')}
        if missing:raise ValueError(f'correction source missing from ingested DB: {missing}')
    initial_build_ms=(time.perf_counter()-t)*1000
    rows=[];consistent_hash=None;bounds=None
    for repeat in range(repeats):
        with CorrectionMap(base_path) as base:
            t=time.perf_counter()
            reference=base.full_rebuild(pose_overrides=updated)
            reference_ms=(time.perf_counter()-t)*1000
            expected_hash=_hash_voxels(reference)
            # Independent history oracle from pre-correction committed map.
            before=base.full_rebuild()
            bounds=_all_bounds([before,reference])
            before_hash=_hash_voxels(before)
        with tempfile.TemporaryDirectory(prefix='nclt_correction_trial_') as tmp:
            trial_path=Path(tmp)/'incremental.sqlite'
            # SQLite online backup includes transactionally consistent committed data.
            with sqlite3.connect(str(base_path)) as source, sqlite3.connect(str(trial_path)) as dest:
                source.backup(dest)
            with CorrectionMap(trial_path) as trial:
                rev_before=trial.revision
                t=time.perf_counter()
                result=trial.correct_poses(updated,note='controlled NCLT reference correction')
                incremental_ms=(time.perf_counter()-t)*1000
                t=time.perf_counter()
                got=trial.query(bounds)
                retrieval_ms=(time.perf_counter()-t)*1000
                if not np.array_equal(reference,got):
                    raise AssertionError(f'incremental != rebuild (repeat {repeat}): {len(got)} vs {len(reference)}')
                old=trial.query(bounds,rev_before)
                if not np.array_equal(before,old):
                    raise AssertionError('pre-correction historical revision not preserved')
                if not trial.verify():raise AssertionError('independent full rebuild mismatch')
                correct_stats=trial.stats()
                snapshot_revision=result.revision
            whole_disk_ms=None;whole_disk_bytes=None
            if full_disk_rebuild:
                # This is a *cold*, full corrected DB build including reading all
                # prepared sources, ingesting them and fully committing pages.
                rebuild_path=Path(tmp)/'rebuilt.sqlite'
                t=time.perf_counter()
                with CorrectionMap(rebuild_path,voxel_size=voxel_size,origin=tuple(origin)) as complete:
                    build_stats=_ingest(complete,corrected,batch_size)
                    all_corrected=complete.query(bounds)
                    whole_disk_bytes=complete.stats()['physical_sqlite_bytes_including_wal']
                whole_disk_ms=(time.perf_counter()-t)*1000
                if not np.array_equal(got,all_corrected):
                    raise AssertionError('freshly materialized corrected DB differs from incremental DB')
            rows.append({'repeat':repeat,'source_count':base_stats['sources'],
                         'corrected_source_count':len(changes),'incremental_commit_ms':incremental_ms,
                         'incremental_retrieval_ms':retrieval_ms,'full_union_rebuild_ms':reference_ms,
                         'full_disk_materialization_ms':whole_disk_ms,
                         'affected_tiles':result.tiles_touched,'modified_tiles':result.tiles_modified,
                         'before_voxels':len(before),'after_voxels':len(got),
                         'pre_correction_sha256':before_hash,'corrected_sha256':expected_hash,
                         'initial_revision':rev_before,'corrected_revision':snapshot_revision,
                         'incremental_file_bytes':correct_stats['physical_sqlite_bytes_including_wal'],
                         'full_disk_bytes':whole_disk_bytes,'exact':True,'historical_revision_exact':True})
            consistent_hash=expected_hash
    # Preserve a corrected, revision-queryable database after timing.
    final_path=output/'corrected_map.sqlite'
    if final_path.exists():raise FileExistsError(f'{final_path} already exists')
    with sqlite3.connect(str(base_path)) as source, sqlite3.connect(str(final_path)) as dest:
        source.backup(dest)
    with CorrectionMap(final_path) as persistent:
        prior_revision=persistent.revision
        persistent.correct_poses(updated,note='Step 11.1 persisted corrected version')
        if not persistent.verify():raise AssertionError('saved corrected database failed exact verification')
        if _hash_voxels(persistent.query(bounds))!=consistent_hash:
            raise AssertionError('saved corrected DB differs from all benchmark trials')
        final_revision=persistent.revision
    path=output/'trials.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'step':'11.1','input':str(prepared_root),'sources':base_stats['sources'],
             'changed_sources':len(changes),'base_revision':base_revision,
             'initial_ingest_wall_ms':initial_build_ms,'initial_ingest':ingest_stats,
             'auditable_provenance_rows':provenance_rows,
             'persisted_original_database':'initial_map.sqlite',
             'persisted_corrected_database':'corrected_map.sqlite',
             'persisted_original_revision':prior_revision,
             'persisted_corrected_revision':final_revision,
             'n_trials':repeats,'voxel_size_m':voxel_size,'origin_m':list(origin),
             'base_database_stats':base_stats,
             'corrected_database_stats':correct_stats,
             'touched_tile_fraction_of_initial_active':(rows[0]['affected_tiles']/base_stats['current_nonempty_tiles']
                  if base_stats['current_nonempty_tiles'] else None),
             'corrected_voxel_sha256':consistent_hash,
             'corrected_voxel_count':rows[0]['after_voxels'],
             'median_incremental_commit_ms':float(np.median([r['incremental_commit_ms'] for r in rows])),
             'p95_incremental_commit_ms':float(np.percentile([r['incremental_commit_ms'] for r in rows],95)),
             'median_full_union_rebuild_ms':float(np.median([r['full_union_rebuild_ms'] for r in rows])),
             'median_full_disk_materialization_ms':float(np.median([r['full_disk_materialization_ms'] for r in rows])) if full_disk_rebuild else None,
             'median_affected_tiles':float(np.median([r['affected_tiles'] for r in rows])),
             'all_exact':True,'all_previous_revisions_preserved':True,
             'method':'NCLT nearest-groundtruth scans with controlled SE3 perturbation followed by correction to original reference poses',
             'timing_note':('full_union_rebuild_ms is in-memory source transforms+set union, excludes disk; '
                            'incremental_commit_ms is SQLite transaction+page updates; '
                            'full_disk_materialization_ms is new DB from prepared sources including input read/SQLite ingest; '
                            'cloning SQLite DB is excluded from incremental correction timing. '
                            'Cache warmup/order not fully controlled; do not treat speed ratios as rigorous yet.'),
             'limitation':'Controlled GT pose perturbations, not SLAM loop closure; nearest poses; filtered body XYZ only; endpoint occupancy only'}
    _write_json(output/'summary.json',summary)
    _write_json(output/'metadata.json',{'prepared_metadata':json.loads((prepared_root/'metadata.json').read_text()),
                                       'benchmark':{'batch_size':batch_size,'repeats':repeats,'full_disk_rebuild':full_disk_rebuild,
                                                    'axis_order':'X,Y,Z','voxel_grid':'floor((T @ body_point - origin)/voxel_size)',
                                                    'comparison':'exact sorted int64 voxel coordinates'}})
    return summary
