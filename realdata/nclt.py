from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import json
import math
import os

import numpy as np

from synthetic.generator import GeneratedDataset
from synthetic.world import Session, Change, WorldSpec, Voxel


@dataclass(frozen=True)
class NCLTIngestConfig:
    voxel_size_m: float = 0.5
    scan_stride: int = 10
    min_range_m: float = 2.0
    max_range_m: float = 80.0
    min_points_per_voxel: int = 1
    max_pose_gap_us: int = 100_000
    max_scans_per_session: int | None = None
    # ROI is in the native NCLT local frame AFTER transforming the body-frame
    # synchronized scans: (xmin,xmax,ymin,ymax,zmin,zmax). NCLT local z points down.
    roi_local_m: tuple[float,float,float,float,float,float] | None = None


@dataclass(frozen=True)
class NCLTSessionPaths:
    session_name: str
    scan_dir: Path
    gt_csv: Path


@dataclass
class NCLTSessionStats:
    session_name: str
    scan_files_total: int = 0
    scans_selected: int = 0
    scans_used: int = 0
    scans_skipped_pose_gap: int = 0
    raw_points: int = 0
    filtered_points: int = 0
    occupied_world_voxels: int = 0
    pose_gap_us_median: float = 0.0
    pose_gap_us_p95: float = 0.0


@dataclass
class NCLTPreparedDataset:
    dataset: GeneratedDataset
    session_names: list[str]
    # world-grid origin expressed in native NCLT local coordinates.
    origin_local_m: tuple[float,float,float]
    session_stats: list[NCLTSessionStats]
    config: NCLTIngestConfig

    def metadata(self) -> dict:
        return {
            "session_names": self.session_names,
            "origin_local_m": list(self.origin_local_m),
            "config": asdict(self.config),
            "world": {
                "size_x_m": self.dataset.spec.size_x_m,
                "size_y_m": self.dataset.spec.size_y_m,
                "size_z_m": self.dataset.spec.size_z_m,
                "voxel_size_m": self.dataset.spec.voxel_size_m,
                "nx": self.dataset.spec.nx,
                "ny": self.dataset.spec.ny,
                "nz": self.dataset.spec.nz,
            },
            "session_stats": [asdict(x) for x in self.session_stats],
            "semantics": {
                "occupied": "voxel received >= min_points_per_voxel transformed LiDAR returns in that session",
                "absent": "not occupied in that session map; NOT interpreted as observed free space",
                "changes": "endpoint differences between session occupancy maps; observation-history differences, not semantic ground-truth environmental changes",
            },
        }


def _candidate_scan_dirs(root: Path, session: str) -> list[Path]:
    return [
        root / session / "velodyne_sync",
        root / f"{session}_vel" / "velodyne_sync",
        root / "velodyne_data" / f"{session}_vel" / "velodyne_sync",
        root / "velodyne_data" / session / "velodyne_sync",
        root / "sensor_data" / session / "velodyne_sync",
    ]


def _candidate_gt_files(root: Path, session: str) -> list[Path]:
    return [
        root / session / f"groundtruth_{session}.csv",
        root / session / "groundtruth.csv",
        root / "ground_truth" / f"groundtruth_{session}.csv",
        root / "ground_truth" / session / f"groundtruth_{session}.csv",
        root / f"groundtruth_{session}.csv",
        root / f"{session}_groundtruth.csv",
    ]


def discover_nclt_session(data_root: str | Path, session: str) -> NCLTSessionPaths:
    root = Path(data_root)
    scan = next((p for p in _candidate_scan_dirs(root, session) if p.is_dir()), None)
    gt = next((p for p in _candidate_gt_files(root, session) if p.is_file()), None)
    if scan is None:
        tried = "\n  ".join(str(p) for p in _candidate_scan_dirs(root, session))
        raise FileNotFoundError(f"Could not find velodyne_sync for {session}. Tried:\n  {tried}")
    if gt is None:
        tried = "\n  ".join(str(p) for p in _candidate_gt_files(root, session))
        raise FileNotFoundError(f"Could not find ground-truth CSV for {session}. Tried:\n  {tried}")
    return NCLTSessionPaths(session, scan, gt)


def read_velodyne_sync_bin(path: str | Path) -> np.ndarray:
    """
    NCLT synchronized scan format:
      uint16 x, uint16 y, uint16 z, uint8 intensity, uint8 laser_id
    repeated per point (8 bytes/point), little-endian.

    Coordinates decode as meters:
      coord_m = uint16 * 0.005 - 100.0

    Synchronized scans are already motion-compensated and expressed in the
    Segway body frame by the NCLT dataset.
    """
    path = Path(path)
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 8:
        raise ValueError(f"{path} size {raw.size} is not divisible by 8 bytes/point")
    if raw.size == 0:
        return np.empty((0,3), dtype=np.float64)

    # Structured dtype avoids Python loops and preserves little-endian decoding.
    dt = np.dtype([
        ("x", "<u2"), ("y", "<u2"), ("z", "<u2"),
        ("intensity", "u1"), ("laser", "u1"),
    ])
    pts = np.fromfile(path, dtype=dt)
    xyz = np.empty((len(pts),3), dtype=np.float64)
    xyz[:,0] = pts["x"].astype(np.float64) * 0.005 - 100.0
    xyz[:,1] = pts["y"].astype(np.float64) * 0.005 - 100.0
    xyz[:,2] = pts["z"].astype(np.float64) * 0.005 - 100.0

    # NCLT uses (0,0,0) sensor-frame hits to encode invalid/out-of-range values.
    valid = ~np.all(np.isclose(xyz, 0.0), axis=1)
    return xyz[valid]


def load_ground_truth(path: str | Path) -> np.ndarray:
    """
    Returns Nx7 float64:
      utime_us, x, y, z, roll, pitch, yaw
    """
    gt = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if gt.ndim == 1:
        gt = gt.reshape(1,-1)
    if gt.shape[1] < 7:
        raise ValueError(f"Expected >=7 columns in {path}, got {gt.shape[1]}")
    gt = gt[:, :7]
    order = np.argsort(gt[:,0])
    return gt[order]


def _rot_zyx(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    Standard active body->local ZYX matrix Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Vectorized over angles; returns (...,3,3).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    out = np.empty(np.broadcast_shapes(np.shape(roll), np.shape(pitch), np.shape(yaw)) + (3,3))
    out[...,0,0] = cy*cp
    out[...,0,1] = cy*sp*sr - sy*cr
    out[...,0,2] = cy*sp*cr + sy*sr
    out[...,1,0] = sy*cp
    out[...,1,1] = sy*sp*sr + cy*cr
    out[...,1,2] = sy*sp*cr - cy*sr
    out[...,2,0] = -sp
    out[...,2,1] = cp*sr
    out[...,2,2] = cp*cr
    return out


class _NearestPoseMatcher:
    def __init__(self, gt: np.ndarray):
        self.gt = gt
        self.t = gt[:,0].astype(np.int64)

    def nearest(self, timestamp_us: int):
        t = int(timestamp_us)
        i = int(np.searchsorted(self.t, t))
        candidates=[]
        if i < len(self.t): candidates.append(i)
        if i > 0: candidates.append(i-1)
        j = min(candidates, key=lambda k: abs(int(self.t[k])-t))
        row = self.gt[j]
        gap = abs(int(self.t[j])-t)
        return row, gap


def _timestamp_from_scan(path: Path) -> int:
    # Supports "123.bin" and common "123.bin.csv"-derived layouts by taking
    # the first dot-separated token.
    return int(path.name.split(".")[0])


def _list_scan_files(scan_dir: Path) -> list[Path]:
    files = [p for p in scan_dir.iterdir() if p.is_file() and ".bin" in p.name]
    return sorted(files, key=_timestamp_from_scan)


def _filter_body_points(points: np.ndarray, cfg: NCLTIngestConfig) -> np.ndarray:
    if len(points) == 0:
        return points
    r = np.linalg.norm(points, axis=1)
    keep = (r >= cfg.min_range_m) & (r <= cfg.max_range_m)
    return points[keep]


def _apply_roi(points_local: np.ndarray, roi):
    if roi is None or len(points_local) == 0:
        return points_local
    xmin,xmax,ymin,ymax,zmin,zmax = roi
    m = (
        (points_local[:,0] >= xmin) & (points_local[:,0] < xmax) &
        (points_local[:,1] >= ymin) & (points_local[:,1] < ymax) &
        (points_local[:,2] >= zmin) & (points_local[:,2] < zmax)
    )
    return points_local[m]


def _points_to_absolute_voxels(points_local: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points_local) == 0:
        return np.empty((0,3), dtype=np.int64)
    return np.floor(points_local / voxel_size).astype(np.int64)


def _session_world_voxels(
    paths: NCLTSessionPaths,
    cfg: NCLTIngestConfig,
) -> tuple[set[tuple[int,int,int]], NCLTSessionStats]:
    gt = load_ground_truth(paths.gt_csv)
    matcher = _NearestPoseMatcher(gt)
    scans = _list_scan_files(paths.scan_dir)
    selected = scans[::cfg.scan_stride]
    if cfg.max_scans_per_session is not None:
        selected = selected[:cfg.max_scans_per_session]

    st = NCLTSessionStats(
        session_name=paths.session_name,
        scan_files_total=len(scans),
        scans_selected=len(selected),
    )
    gaps=[]
    voxel_counts: dict[tuple[int,int,int], int] = {}

    for scan in selected:
        ts = _timestamp_from_scan(scan)
        pose, gap = matcher.nearest(ts)
        gaps.append(gap)
        if gap > cfg.max_pose_gap_us:
            st.scans_skipped_pose_gap += 1
            continue

        body = read_velodyne_sync_bin(scan)
        st.raw_points += len(body)
        body = _filter_body_points(body, cfg)
        if len(body) == 0:
            continue

        _, x,y,z,roll,pitch,yaw = pose
        R = _rot_zyx(np.array(roll), np.array(pitch), np.array(yaw))
        local = body @ R.T
        local[:,0] += x
        local[:,1] += y
        local[:,2] += z
        local = _apply_roi(local, cfg.roi_local_m)
        st.filtered_points += len(local)
        st.scans_used += 1
        if len(local) == 0:
            continue

        vox = _points_to_absolute_voxels(local, cfg.voxel_size_m)
        # Count each point, then threshold at session level.
        uniq, cnt = np.unique(vox, axis=0, return_counts=True)
        for v,c in zip(uniq, cnt):
            key=(int(v[0]),int(v[1]),int(v[2]))
            voxel_counts[key] = voxel_counts.get(key,0) + int(c)

    occupied = {v for v,c in voxel_counts.items() if c >= cfg.min_points_per_voxel}
    st.occupied_world_voxels = len(occupied)
    if gaps:
        st.pose_gap_us_median = float(np.median(gaps))
        st.pose_gap_us_p95 = float(np.percentile(gaps,95))
    return occupied, st


def _build_generated_dataset(
    absolute_sessions: list[set[tuple[int,int,int]]],
    voxel_size: float,
) -> tuple[GeneratedDataset, tuple[float,float,float]]:
    if not absolute_sessions:
        raise ValueError("No sessions")
    union=set().union(*absolute_sessions)
    if not union:
        raise ValueError("No occupied voxels after filtering. Relax ROI/filter thresholds.")

    arr=np.asarray(list(union), dtype=np.int64)
    vmin=arr.min(axis=0)
    vmax=arr.max(axis=0)

    shifted=[]
    for vox in absolute_sessions:
        shifted.append(frozenset(
            (x-int(vmin[0]), y-int(vmin[1]), z-int(vmin[2]))
            for x,y,z in vox
        ))

    dims=(vmax-vmin+1).astype(int)
    spec=WorldSpec(
        size_x_m=float(dims[0]*voxel_size),
        size_y_m=float(dims[1]*voxel_size),
        size_z_m=float(dims[2]*voxel_size),
        voxel_size_m=float(voxel_size),
    )
    sessions=[Session(i,occ) for i,occ in enumerate(shifted)]
    changes=[]
    for i in range(1,len(sessions)):
        prev=set(sessions[i-1].occupied); cur=set(sessions[i].occupied)
        changes.append(Change(i, frozenset(cur-prev), frozenset(prev-cur)))

    ds=GeneratedDataset(
        spec=spec,
        sessions=sessions,
        changes=changes,
        objects_by_session=[],
    )
    ds.validate()

    origin=tuple(float(v*voxel_size) for v in vmin)
    return ds, origin


def prepare_nclt_dataset(
    data_root: str | Path,
    session_names: Iterable[str],
    config: NCLTIngestConfig | None = None,
) -> NCLTPreparedDataset:
    cfg=config or NCLTIngestConfig()
    names=list(session_names)
    if len(names) < 2:
        raise ValueError("Use at least two NCLT sessions for a long-term experiment.")
    if cfg.scan_stride < 1:
        raise ValueError("scan_stride must be >=1")
    if cfg.voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be >0")
    if cfg.min_points_per_voxel < 1:
        raise ValueError("min_points_per_voxel must be >=1")

    absolute=[]
    stats=[]
    for name in names:
        paths=discover_nclt_session(data_root,name)
        vox,st=_session_world_voxels(paths,cfg)
        absolute.append(vox);stats.append(st)

    ds,origin=_build_generated_dataset(absolute,cfg.voxel_size_m)
    return NCLTPreparedDataset(ds,names,origin,stats,cfg)


def save_prepared(prepared: NCLTPreparedDataset, output_dir: str | Path):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"metadata.json").write_text(json.dumps(prepared.metadata(),indent=2))
    prepared.dataset.save(out/"dataset")
