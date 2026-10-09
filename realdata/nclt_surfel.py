from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from synthetic.world import WorldSpec
from .nclt import (
    NCLTIngestConfig,
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
from .nclt_packed import (
    _session_absolute_codes,
    _nearest_pose_rows,
    pack_signed_xyz,
    unpack_signed_xyz,
    linear_encode,
)


@dataclass
class NCLTSurfelMap:
    """0.5m-style voxel index with sub-voxel geometric payload.

    Arrays are aligned by occupied base voxel. `cov6` packs xx,xy,xz,yy,yz,zz.
    Coordinates/centroids are in the NCLT local metric frame used by Steps 8-9.
    """
    spec: WorldSpec
    origin_local_m: tuple[float, float, float]
    linear_codes: np.ndarray
    counts: np.ndarray
    centroids: np.ndarray
    cov6: np.ndarray
    normals: np.ndarray
    curvature: np.ndarray
    normal_valid: np.ndarray
    session_stats: NCLTSessionStats
    config: NCLTIngestConfig
    occupancy_cache_hit: bool
    occupancy_load_ms: float
    stats_build_ms: float

    def payload_nbytes(self) -> int:
        return int(sum(a.nbytes for a in (
            self.linear_codes, self.counts, self.centroids, self.cov6,
            self.normals, self.curvature, self.normal_valid,
        )))

    def minimal_localization_payload_nbytes(self) -> int:
        # Full covariance is useful for later probabilistic/GICP work but the
        # Step-9.0.3B point-to-plane matcher only requires mu,n,N,quality.
        return int(sum(a.nbytes for a in (
            self.linear_codes, self.counts, self.centroids,
            self.normals, self.curvature, self.normal_valid,
        )))


def _cache_exists(cache_dir: Path | None, session: str, cfg: NCLTIngestConfig) -> bool:
    if cache_dir is None:
        return False
    from .nclt_packed import _cache_paths
    npy, js = _cache_paths(cache_dir, session, cfg)
    return npy.exists() and js.exists()


def _reduce_scan_into_stats(
    local: np.ndarray,
    voxel_size_m: float,
    occupied_abs_codes: np.ndarray,
    counts: np.ndarray,
    sum1: np.ndarray,
    sum2: np.ndarray,
) -> None:
    if len(local) == 0:
        return
    # Historical NCLT ground truth contains a small number of non-finite rows
    # in some sessions.  A non-finite transformed return must never reach the
    # integer voxel cast or moment accumulator: one NaN can poison a voxel's
    # covariance and later make np.linalg.eigh fail for the whole batch.
    local = np.asarray(local, dtype=np.float64)
    finite = np.all(np.isfinite(local), axis=1)
    if not np.all(finite):
        local = local[finite]
    if len(local) == 0:
        return
    vox = np.floor(local / float(voxel_size_m)).astype(np.int64)
    codes = pack_signed_xyz(vox[:, 0], vox[:, 1], vox[:, 2])
    pos = np.searchsorted(occupied_abs_codes, codes)
    safe = np.minimum(pos, max(0, len(occupied_abs_codes) - 1))
    keep = (pos < len(occupied_abs_codes)) & (occupied_abs_codes[safe] == codes)
    if not np.any(keep):
        return
    pos = pos[keep].astype(np.int64, copy=False)
    pts = np.asarray(local[keep], dtype=np.float64)
    vv = vox[keep]
    off = pts - vv.astype(np.float64) * float(voxel_size_m)

    order = np.argsort(pos, kind="stable")
    ids = pos[order]
    off = off[order]
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    uid = ids[starts]
    gc = np.diff(np.r_[starts, len(ids)]).astype(np.uint32)
    s1 = np.add.reduceat(off, starts, axis=0)
    x, y, z = off[:, 0], off[:, 1], off[:, 2]
    q = np.column_stack((x*x, x*y, x*z, y*y, y*z, z*z))
    s2 = np.add.reduceat(q, starts, axis=0)

    counts[uid] += gc
    sum1[uid] += s1.astype(np.float32)
    sum2[uid] += s2.astype(np.float32)


def _finalize_geometry(
    abs_codes: np.ndarray,
    voxel_size_m: float,
    counts: np.ndarray,
    sum1: np.ndarray,
    sum2: np.ndarray,
    normal_min_points: int,
    normal_max_curvature: float,
    normal_min_horizontal_component: float,
    batch_size: int = 200_000,
):
    n = len(abs_codes)
    mean_off = np.zeros((n, 3), dtype=np.float32)
    nz = counts > 0
    mean_off[nz] = sum1[nz] / counts[nz, None].astype(np.float32)
    ax, ay, az = unpack_signed_xyz(abs_codes)
    origins = np.column_stack((ax, ay, az)).astype(np.float64) * float(voxel_size_m)
    centroids = (origins + mean_off.astype(np.float64)).astype(np.float32)

    cov6 = np.zeros((n, 6), dtype=np.float32)
    # Only finalize voxels whose accumulated sufficient statistics are finite.
    # This is a second line of defense against a corrupt scan/pose and also
    # makes cached historical-map construction robust to legacy bad inputs.
    finite_stats = np.all(np.isfinite(sum1), axis=1) & np.all(np.isfinite(sum2), axis=1)
    valid_count = (counts >= int(normal_min_points)) & finite_stats
    idx = np.flatnonzero(valid_count)
    if len(idx):
        c = counts[idx].astype(np.float64)
        m = mean_off[idx].astype(np.float64)
        e2 = sum2[idx].astype(np.float64) / c[:, None]
        cov6[idx, 0] = (e2[:, 0] - m[:, 0]*m[:, 0]).astype(np.float32)
        cov6[idx, 1] = (e2[:, 1] - m[:, 0]*m[:, 1]).astype(np.float32)
        cov6[idx, 2] = (e2[:, 2] - m[:, 0]*m[:, 2]).astype(np.float32)
        cov6[idx, 3] = (e2[:, 3] - m[:, 1]*m[:, 1]).astype(np.float32)
        cov6[idx, 4] = (e2[:, 4] - m[:, 1]*m[:, 2]).astype(np.float32)
        cov6[idx, 5] = (e2[:, 5] - m[:, 2]*m[:, 2]).astype(np.float32)

    normals = np.zeros((n, 3), dtype=np.float32)
    curvature = np.full(n, np.inf, dtype=np.float32)
    normal_valid = np.zeros(n, dtype=np.uint8)
    for a in range(0, len(idx), int(batch_size)):
        ii = idx[a:a+int(batch_size)]
        c6 = cov6[ii].astype(np.float64)
        C = np.empty((len(ii), 3, 3), dtype=np.float64)
        C[:, 0, 0] = c6[:, 0]; C[:, 0, 1] = c6[:, 1]; C[:, 0, 2] = c6[:, 2]
        C[:, 1, 0] = c6[:, 1]; C[:, 1, 1] = c6[:, 3]; C[:, 1, 2] = c6[:, 4]
        C[:, 2, 0] = c6[:, 2]; C[:, 2, 1] = c6[:, 4]; C[:, 2, 2] = c6[:, 5]
        finite_c = np.all(np.isfinite(C), axis=(1, 2))
        if not np.any(finite_c):
            continue
        jj = ii[finite_c]
        Cf = C[finite_c]
        try:
            ev, evec = np.linalg.eigh(Cf)
            solved = np.ones(len(jj), dtype=bool)
        except np.linalg.LinAlgError:
            # Extremely rare fallback: isolate the bad 3x3 matrix instead of
            # aborting construction of an entire historical session map.
            ev = np.full((len(jj), 3), np.nan, dtype=np.float64)
            evec = np.full((len(jj), 3, 3), np.nan, dtype=np.float64)
            solved = np.zeros(len(jj), dtype=bool)
            for k, M in enumerate(Cf):
                try:
                    ek, vk = np.linalg.eigh(M)
                except np.linalg.LinAlgError:
                    continue
                ev[k] = ek; evec[k] = vk; solved[k] = True
        total = np.sum(ev, axis=1)
        curv = np.divide(np.maximum(ev[:, 0], 0.0), total, out=np.full(len(jj), np.inf), where=total > 1e-12)
        nn = evec[:, :, 0]
        h = np.hypot(nn[:, 0], nn[:, 1])
        good = solved & np.isfinite(curv) & np.all(np.isfinite(nn), axis=1) & (curv <= float(normal_max_curvature)) & (h >= float(normal_min_horizontal_component))
        normals[jj[good]] = nn[good].astype(np.float32)
        curvature[jj] = curv.astype(np.float32)
        normal_valid[jj[good]] = 1
    return centroids, cov6, normals, curvature, normal_valid


def build_nclt_surfel_map(
    data_root: str | Path,
    session: str,
    config: NCLTIngestConfig,
    cache_dir: str | Path | None = ".cache/step8_1",
    normal_min_points: int = 6,
    normal_max_curvature: float = 0.20,
    normal_min_horizontal_component: float = 0.20,
    progress_every: int = 250,
) -> NCLTSurfelMap:
    """Build sub-voxel surfel payload over the existing occupancy lattice.

    Occupancy codes may be loaded from the Step-8 cache. Surfel statistics are
    rebuilt in memory from the raw map-session returns and are not persisted,
    avoiding additional disk use by default.
    """
    paths = discover_nclt_session(data_root, session)
    cache_path = Path(cache_dir) if cache_dir is not None else None
    hit = _cache_exists(cache_path, session, config)
    tload = time.perf_counter_ns()
    abs_codes, st = _session_absolute_codes(paths, config, cache_path)
    occupancy_load_ms = (time.perf_counter_ns() - tload) / 1e6
    if not len(abs_codes):
        raise RuntimeError(f"No occupied voxels for {session}")

    counts = np.zeros(len(abs_codes), dtype=np.uint32)
    sum1 = np.zeros((len(abs_codes), 3), dtype=np.float32)
    sum2 = np.zeros((len(abs_codes), 6), dtype=np.float32)

    gt = load_ground_truth(paths.gt_csv)
    scans = _list_scan_files(paths.scan_dir)
    selected = scans[::config.scan_stride]
    if config.max_scans_per_session is not None:
        selected = selected[:config.max_scans_per_session]
    ts = np.asarray([_timestamp_from_scan(p) for p in selected], dtype=np.int64)
    pose_rows, gaps = _nearest_pose_rows(gt, ts)
    t0 = time.perf_counter()
    used = 0
    skipped_nonfinite_pose = 0
    dropped_nonfinite_points = 0
    for i, (scan, pose, gap) in enumerate(zip(selected, pose_rows, gaps), start=1):
        if int(gap) > int(config.max_pose_gap_us):
            continue
        if not np.all(np.isfinite(np.asarray(pose[1:7], dtype=np.float64))):
            skipped_nonfinite_pose += 1
            continue
        body = read_velodyne_sync_bin(scan)
        body = _filter_body_points(body, config)
        if not len(body):
            continue
        _, px, py, pz, roll, pitch, yaw = pose
        R = _rot_zyx(np.array(roll), np.array(pitch), np.array(yaw))
        local = body @ R.T
        local[:, 0] += px; local[:, 1] += py; local[:, 2] += pz
        finite_local = np.all(np.isfinite(local), axis=1)
        if not np.all(finite_local):
            dropped_nonfinite_points += int(np.count_nonzero(~finite_local))
            local = local[finite_local]
        local = _apply_roi(local, config.roi_local_m)
        _reduce_scan_into_stats(local, config.voxel_size_m, abs_codes, counts, sum1, sum2)
        used += 1
        if i % int(progress_every) == 0 or i == len(selected):
            print(f"  [{session} surfel] scans {i:,}/{len(selected):,}; used={used:,}; {time.perf_counter()-t0:.1f}s", flush=True)

    if skipped_nonfinite_pose or dropped_nonfinite_points:
        print(
            f"  [{session} surfel] non-finite guard: skipped_pose_scans={skipped_nonfinite_pose:,}; "
            f"dropped_points={dropped_nonfinite_points:,}", flush=True,
        )

    centroids, cov6, normals, curvature, normal_valid = _finalize_geometry(
        abs_codes, config.voxel_size_m, counts, sum1, sum2,
        normal_min_points, normal_max_curvature, normal_min_horizontal_component,
    )
    del sum1, sum2
    ax, ay, az = unpack_signed_xyz(abs_codes)
    mins = np.asarray([ax.min(), ay.min(), az.min()], dtype=np.int64)
    maxs = np.asarray([ax.max(), ay.max(), az.max()], dtype=np.int64)
    dims = (maxs - mins + 1).astype(int)
    spec = WorldSpec(
        float(dims[0]*config.voxel_size_m), float(dims[1]*config.voxel_size_m),
        float(dims[2]*config.voxel_size_m), float(config.voxel_size_m),
    )
    linear_codes = linear_encode(ax-mins[0], ay-mins[1], az-mins[2], spec)
    origin = tuple(float(v*config.voxel_size_m) for v in mins)
    stats_build_ms = (time.perf_counter() - t0) * 1000.0
    return NCLTSurfelMap(
        spec=spec, origin_local_m=origin, linear_codes=np.asarray(linear_codes, dtype=np.uint64),
        counts=counts, centroids=centroids, cov6=cov6, normals=normals,
        curvature=curvature, normal_valid=normal_valid, session_stats=st, config=config,
        occupancy_cache_hit=hit, occupancy_load_ms=float(occupancy_load_ms), stats_build_ms=float(stats_build_ms),
    )


__all__ = ["NCLTSurfelMap", "build_nclt_surfel_map"]
