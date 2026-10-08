from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha1
from pathlib import Path
from typing import Iterable
import json
import math
import time

import numpy as np

from synthetic.world import WorldSpec
from .nclt import (
    NCLTIngestConfig,
    NCLTSessionPaths,
    NCLTSessionStats,
    discover_nclt_session,
    load_ground_truth,
    read_velodyne_sync_bin,
    _list_scan_files,
    _timestamp_from_scan,
    _filter_body_points,
    _apply_roi,
    _rot_zyx,
)

# Signed voxel coordinates are packed into 63 bits: 21 bits per axis.
_SIGNED_BITS = 21
_SIGNED_BIAS = 1 << (_SIGNED_BITS - 1)
_SIGNED_MASK = (1 << _SIGNED_BITS) - 1


def pack_signed_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    z = np.asarray(z, dtype=np.int64)
    lo = -_SIGNED_BIAS
    hi = _SIGNED_BIAS - 1
    if (x.size and (x.min() < lo or x.max() > hi)) or \
       (y.size and (y.min() < lo or y.max() > hi)) or \
       (z.size and (z.min() < lo or z.max() > hi)):
        raise ValueError("Absolute voxel coordinates exceed 21-bit signed packing range")
    ux = (x + _SIGNED_BIAS).astype(np.uint64)
    uy = (y + _SIGNED_BIAS).astype(np.uint64)
    uz = (z + _SIGNED_BIAS).astype(np.uint64)
    return ux | (uy << _SIGNED_BITS) | (uz << (2 * _SIGNED_BITS))


def unpack_signed_xyz(code: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(code, dtype=np.uint64)
    x = (c & _SIGNED_MASK).astype(np.int64) - _SIGNED_BIAS
    y = ((c >> _SIGNED_BITS) & _SIGNED_MASK).astype(np.int64) - _SIGNED_BIAS
    z = ((c >> (2 * _SIGNED_BITS)) & _SIGNED_MASK).astype(np.int64) - _SIGNED_BIAS
    return x, y, z


def linear_encode(x: np.ndarray, y: np.ndarray, z: np.ndarray, spec: WorldSpec) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    y = np.asarray(y, dtype=np.uint64)
    z = np.asarray(z, dtype=np.uint64)
    return ((x * np.uint64(spec.ny) + y) * np.uint64(spec.nz) + z).astype(np.uint64)


def linear_decode(code: np.ndarray, spec: WorldSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(code, dtype=np.uint64)
    z = (c % np.uint64(spec.nz)).astype(np.int32)
    q = c // np.uint64(spec.nz)
    y = (q % np.uint64(spec.ny)).astype(np.int32)
    x = (q // np.uint64(spec.ny)).astype(np.int32)
    return x, y, z


class _SparseCountAccumulator:
    """Chunked NumPy sparse counter; avoids millions of Python dict entries."""

    def __init__(self, threshold: int):
        self.threshold = int(threshold)
        self.codes = np.empty(0, dtype=np.uint64)
        self.counts = None if threshold == 1 else np.empty(0, dtype=np.uint32)
        self.pending: list[np.ndarray] = []

    def add(self, codes: np.ndarray):
        if len(codes):
            self.pending.append(np.asarray(codes, dtype=np.uint64))

    def flush(self):
        if not self.pending:
            return
        batch = np.concatenate(self.pending)
        self.pending.clear()
        if self.threshold == 1:
            batch = np.unique(batch)
            self.codes = batch if not len(self.codes) else np.union1d(self.codes, batch)
            return

        bcodes, bcounts = np.unique(batch, return_counts=True)
        bcounts = bcounts.astype(np.uint32)
        if not len(self.codes):
            self.codes, self.counts = bcodes, bcounts
            return
        union = np.union1d(self.codes, bcodes)
        counts = np.zeros(len(union), dtype=np.uint32)
        counts[np.searchsorted(union, self.codes)] += self.counts
        counts[np.searchsorted(union, bcodes)] += bcounts
        self.codes, self.counts = union, counts

    def finish(self) -> np.ndarray:
        self.flush()
        if self.threshold == 1:
            return self.codes
        return self.codes[self.counts >= self.threshold]


@dataclass
class PackedNCLTPrepared:
    spec: WorldSpec
    session_names: list[str]
    session_codes: list[np.ndarray]  # sorted uint64 linear voxel ids
    origin_local_m: tuple[float, float, float]
    session_stats: list[NCLTSessionStats]
    config: NCLTIngestConfig

    def metadata(self) -> dict:
        return {
            "session_names": self.session_names,
            "origin_local_m": list(self.origin_local_m),
            "config": asdict(self.config),
            "world": {
                "size_x_m": self.spec.size_x_m,
                "size_y_m": self.spec.size_y_m,
                "size_z_m": self.spec.size_z_m,
                "voxel_size_m": self.spec.voxel_size_m,
                "nx": self.spec.nx, "ny": self.spec.ny, "nz": self.spec.nz,
            },
            "session_stats": [asdict(x) for x in self.session_stats],
            "semantics": {
                "occupied": "voxel received >= min_points_per_voxel transformed LiDAR returns in that session",
                "absent": "not occupied in that session map; NOT interpreted as observed free space",
                "changes": "endpoint observation-history differences, not semantic ground-truth environmental changes",
            },
            "step81": {
                "packed_voxels": True,
                "python_tuple_voxel_sets": False,
                "session_cache": True,
            },
        }


def _config_fingerprint(cfg: NCLTIngestConfig) -> str:
    raw = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), default=list)
    return sha1(raw.encode()).hexdigest()[:12]


def _cache_paths(cache_dir: Path, session: str, cfg: NCLTIngestConfig):
    tag = _config_fingerprint(cfg)
    stem = cache_dir / f"{session}_{tag}"
    return stem.with_suffix(".npy"), stem.with_suffix(".json")


def _nearest_pose_rows(gt: np.ndarray, timestamps: np.ndarray):
    gt_t = gt[:, 0].astype(np.int64)
    ts = np.asarray(timestamps, dtype=np.int64)
    right = np.searchsorted(gt_t, ts)
    right_clip = np.minimum(right, len(gt_t) - 1)
    left_clip = np.maximum(right - 1, 0)
    rgap = np.abs(gt_t[right_clip] - ts)
    lgap = np.abs(gt_t[left_clip] - ts)
    choose_left = (right == len(gt_t)) | ((right > 0) & (lgap <= rgap))
    idx = np.where(choose_left, left_clip, right_clip)
    return gt[idx], np.abs(gt_t[idx] - ts)


def _session_absolute_codes(
    paths: NCLTSessionPaths,
    cfg: NCLTIngestConfig,
    cache_dir: Path | None,
    flush_scans: int = 100,
    progress_every: int = 250,
) -> tuple[np.ndarray, NCLTSessionStats]:
    cache_npy = cache_json = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_npy, cache_json = _cache_paths(cache_dir, paths.session_name, cfg)
        if cache_npy.exists() and cache_json.exists():
            codes = np.load(cache_npy, mmap_mode=None)
            st = NCLTSessionStats(**json.loads(cache_json.read_text()))
            print(f"  [{paths.session_name}] cache hit: {len(codes):,} voxels", flush=True)
            return np.asarray(codes, dtype=np.uint64), st

    gt = load_ground_truth(paths.gt_csv)
    scans = _list_scan_files(paths.scan_dir)
    selected = scans[::cfg.scan_stride]
    if cfg.max_scans_per_session is not None:
        selected = selected[:cfg.max_scans_per_session]

    timestamps = np.asarray([_timestamp_from_scan(p) for p in selected], dtype=np.int64)
    pose_rows, gaps = _nearest_pose_rows(gt, timestamps)

    st = NCLTSessionStats(
        session_name=paths.session_name,
        scan_files_total=len(scans),
        scans_selected=len(selected),
    )
    acc = _SparseCountAccumulator(cfg.min_points_per_voxel)
    t0 = time.perf_counter()

    for i, (scan, pose, gap) in enumerate(zip(selected, pose_rows, gaps), start=1):
        if int(gap) > cfg.max_pose_gap_us:
            st.scans_skipped_pose_gap += 1
            continue
        body = read_velodyne_sync_bin(scan)
        st.raw_points += len(body)
        body = _filter_body_points(body, cfg)
        if not len(body):
            continue

        _, px, py, pz, roll, pitch, yaw = pose
        R = _rot_zyx(np.array(roll), np.array(pitch), np.array(yaw))
        local = body @ R.T
        local[:, 0] += px; local[:, 1] += py; local[:, 2] += pz
        local = _apply_roi(local, cfg.roi_local_m)
        st.filtered_points += len(local)
        st.scans_used += 1
        if len(local):
            vox = np.floor(local / cfg.voxel_size_m).astype(np.int64)
            acc.add(pack_signed_xyz(vox[:, 0], vox[:, 1], vox[:, 2]))

        if i % flush_scans == 0:
            acc.flush()
        if i % progress_every == 0 or i == len(selected):
            # Do not flush just to print; this is intentionally approximate until final flush.
            elapsed = time.perf_counter() - t0
            print(
                f"  [{paths.session_name}] scans {i:,}/{len(selected):,}; "
                f"unique>=partial {len(acc.codes):,}; {elapsed:.1f}s",
                flush=True,
            )

    occupied = acc.finish()
    st.occupied_world_voxels = len(occupied)
    if len(gaps):
        st.pose_gap_us_median = float(np.median(gaps))
        st.pose_gap_us_p95 = float(np.percentile(gaps, 95))

    if cache_npy is not None:
        np.save(cache_npy, occupied, allow_pickle=False)
        cache_json.write_text(json.dumps(asdict(st), indent=2))
        print(f"  [{paths.session_name}] cached -> {cache_npy}", flush=True)
    return occupied, st


def prepare_nclt_packed(
    data_root: str | Path,
    session_names: Iterable[str],
    config: NCLTIngestConfig | None = None,
    cache_dir: str | Path | None = ".cache/step8_1",
    flush_scans: int = 100,
    progress_every: int = 250,
) -> PackedNCLTPrepared:
    cfg = config or NCLTIngestConfig()
    names = list(session_names)
    if len(names) < 2:
        raise ValueError("Use at least two sessions")
    if len(names) > 64:
        raise ValueError("Step 8.1 packed temporal masks support at most 64 sessions")
    cache = None if cache_dir is None else Path(cache_dir)

    absolute: list[np.ndarray] = []
    stats: list[NCLTSessionStats] = []
    for name in names:
        print(f"[ingest] {name}", flush=True)
        p = discover_nclt_session(data_root, name)
        codes, st = _session_absolute_codes(p, cfg, cache, flush_scans, progress_every)
        absolute.append(codes); stats.append(st)

    # Establish one common grid without constructing a giant Python union set.
    mins = np.array([np.iinfo(np.int64).max]*3, dtype=np.int64)
    maxs = np.array([np.iinfo(np.int64).min]*3, dtype=np.int64)
    for codes in absolute:
        x, y, z = unpack_signed_xyz(codes)
        mins = np.minimum(mins, [x.min(), y.min(), z.min()])
        maxs = np.maximum(maxs, [x.max(), y.max(), z.max()])
    dims = (maxs - mins + 1).astype(int)
    spec = WorldSpec(
        size_x_m=float(dims[0] * cfg.voxel_size_m),
        size_y_m=float(dims[1] * cfg.voxel_size_m),
        size_z_m=float(dims[2] * cfg.voxel_size_m),
        voxel_size_m=float(cfg.voxel_size_m),
    )

    session_codes: list[np.ndarray] = []
    for name, codes in zip(names, absolute):
        x, y, z = unpack_signed_xyz(codes)
        lin = linear_encode(x - mins[0], y - mins[1], z - mins[2], spec)
        lin = np.unique(lin)
        session_codes.append(lin)
        print(f"[grid] {name}: {len(lin):,} packed voxels", flush=True)

    origin = tuple(float(v * cfg.voxel_size_m) for v in mins)
    return PackedNCLTPrepared(spec, names, session_codes, origin, stats, cfg)
