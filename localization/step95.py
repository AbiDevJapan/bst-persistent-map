from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import math
import multiprocessing as mp
import time
from typing import Iterable

import numpy as np

from localization.step90 import Pose3DOF, wrap_angle
from realdata.nclt import _rot_zyx


@dataclass(frozen=True)
class ScanContextConfig:
    num_rings: int = 20
    num_sectors: int = 60
    max_radius_m: float = 80.0
    lidar_height_m: float = 2.0

    def validate(self):
        if self.num_rings <= 0 or self.num_sectors <= 0:
            raise ValueError('Scan Context ring/sector counts must be positive')
        if self.max_radius_m <= 0:
            raise ValueError('Scan Context max radius must be positive')
        return self


@dataclass(frozen=True)
class ScanContextMatch:
    reference_index: int
    distance: float
    ring_distance: float
    sector_shift: int
    relative_yaw_rad: float


@dataclass
class RobustRegistrationResult:
    T_target_source: np.ndarray
    converged: bool
    inliers: int
    iterations: int
    error: float
    runtime_ms: float
    backend: str
    rotation_inliers: int = 0
    failure_reason: str = ''


@dataclass(frozen=True)
class GlobalHypothesis:
    reference_index: int
    sc_distance: float
    sc_ring_distance: float
    sc_sector_shift: int
    sc_relative_yaw_rad: float
    T_world_query: np.ndarray
    kiss_inliers: int
    kiss_rotation_inliers: int
    scan_gicp_inliers: int
    scan_gicp_converged: bool
    frontend_runtime_ms: float
    used_kiss: bool
    failure_reason: str = ''


def make_scan_context(points: np.ndarray, config: ScanContextConfig | None = None) -> np.ndarray:
    """Compute a standard max-height polar Scan Context descriptor.

    Input points are expected in the LiDAR/body frame. Empty cells remain zero;
    occupied cells store max(z + lidar_height), clipped at zero so empty remains
    distinguishable.  This implementation is intentionally NumPy-only so the
    place-recognition stage has no additional binary dependency.
    """
    cfg = (config or ScanContextConfig()).validate()
    pts = np.asarray(points, dtype=np.float64)
    out = np.zeros((cfg.num_rings, cfg.num_sectors), dtype=np.float32)
    if pts.ndim != 2 or (len(pts) and pts.shape[1] != 3):
        raise ValueError('points must be Nx3')
    if len(pts) == 0:
        return out
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return out
    r = np.hypot(pts[:, 0], pts[:, 1])
    keep = (r > 1e-6) & (r <= float(cfg.max_radius_m))
    if not np.any(keep):
        return out
    pts = pts[keep]
    r = r[keep]
    theta = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2.0 * np.pi)
    ring = np.floor(r / float(cfg.max_radius_m) * cfg.num_rings).astype(np.int64)
    sector = np.floor(theta / (2.0 * np.pi) * cfg.num_sectors).astype(np.int64)
    ring = np.clip(ring, 0, cfg.num_rings - 1)
    sector = np.clip(sector, 0, cfg.num_sectors - 1)
    height = np.maximum(pts[:, 2] + float(cfg.lidar_height_m), 1e-3).astype(np.float32)
    np.maximum.at(out, (ring, sector), height)
    return out


def scan_context_ring_key(desc: np.ndarray) -> np.ndarray:
    d = np.asarray(desc, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError('Scan Context descriptor must be 2-D')
    return np.mean(d, axis=1, dtype=np.float32)


def _column_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Scan Context compares sector columns and ignores columns empty in either scan.
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    valid = (na > 1e-12) & (nb > 1e-12)
    if not np.any(valid):
        return 1.0
    dot = np.sum(a[:, valid] * b[:, valid], axis=0)
    cos = np.clip(dot / (na[valid] * nb[valid]), -1.0, 1.0)
    return float(1.0 - np.mean(cos))


def scan_context_align(query_desc: np.ndarray, reference_desc: np.ndarray) -> tuple[float, int, float]:
    """Return (distance, best circular sector shift, relative query yaw).

    We roll the reference descriptor to align it to the query. If the reference
    needs a +shift roll, the query sensor yaw relative to the reference is the
    negative angular shift under the usual body-frame convention.
    """
    q = np.asarray(query_desc, dtype=np.float32)
    r = np.asarray(reference_desc, dtype=np.float32)
    if q.shape != r.shape or q.ndim != 2:
        raise ValueError('Scan Context descriptors must have identical 2-D shapes')
    best_d = float('inf')
    best_s = 0
    for s in range(q.shape[1]):
        d = _column_cosine_distance(q, np.roll(r, s, axis=1))
        if d < best_d:
            best_d = d
            best_s = s
    # Convert circular shift to shortest signed shift.
    signed = best_s if best_s <= q.shape[1] // 2 else best_s - q.shape[1]
    rel_yaw = signed * (2.0 * math.pi / q.shape[1])
    return float(best_d), int(signed), wrap_angle(rel_yaw)


class ScanContextDatabase:
    def __init__(self, descriptors: np.ndarray, ring_keys: np.ndarray | None = None):
        d = np.asarray(descriptors, dtype=np.float32)
        if d.ndim != 3:
            raise ValueError('descriptors must be NxRxS')
        self.descriptors = d
        if ring_keys is None:
            ring_keys = np.mean(d, axis=2, dtype=np.float32)
        k = np.asarray(ring_keys, dtype=np.float32)
        if k.shape != d.shape[:2]:
            raise ValueError('ring_keys shape mismatch')
        self.ring_keys = k

    def query_descriptor(self, query_desc: np.ndarray, *, top_k: int = 5, candidate_pool: int = 30) -> list[ScanContextMatch]:
        if len(self.descriptors) == 0:
            return []
        q = np.asarray(query_desc, dtype=np.float32)
        if q.shape != self.descriptors.shape[1:]:
            raise ValueError('query descriptor shape mismatch')
        qk = scan_context_ring_key(q)
        diff = self.ring_keys - qk[None, :]
        ring_dist = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float64))
        pool = min(max(int(candidate_pool), int(top_k)), len(ring_dist))
        if pool <= 0:
            return []
        cand = np.argpartition(ring_dist, pool - 1)[:pool]
        matches = []
        for i in cand:
            d, shift, yaw = scan_context_align(q, self.descriptors[int(i)])
            matches.append(ScanContextMatch(int(i), float(d), float(ring_dist[int(i)]), int(shift), float(yaw)))
        matches.sort(key=lambda m: (m.distance, m.ring_distance, m.reference_index))
        return matches[:max(1, int(top_k))]

    def query_points(self, points: np.ndarray, config: ScanContextConfig, *, top_k: int = 5, candidate_pool: int = 30):
        return self.query_descriptor(make_scan_context(points, config), top_k=top_k, candidate_pool=candidate_pool)


def pose_row_to_matrix(pose_row) -> np.ndarray:
    row = np.asarray(pose_row, dtype=np.float64).reshape(-1)
    if len(row) < 7:
        raise ValueError('NCLT pose row must contain utime,x,y,z,roll,pitch,yaw')
    _, x, y, z, roll, pitch, yaw = row[:7]
    vals = np.asarray([x, y, z, roll, pitch, yaw], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError('non-finite pose row')
    R = _rot_zyx(np.array(float(roll)), np.array(float(pitch)), np.array(float(yaw)))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def matrix_to_pose3d(T: np.ndarray) -> Pose3DOF:
    A = np.asarray(T, dtype=np.float64)
    if A.shape != (4, 4) or not np.all(np.isfinite(A)):
        raise ValueError('T must be finite 4x4')
    yaw = math.atan2(A[1, 0], A[0, 0])
    return Pose3DOF(float(A[0, 3]), float(A[1, 3]), wrap_angle(yaw))


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R = np.asarray(R_gt, dtype=np.float64).T @ np.asarray(R_est, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def pose_errors_6d(T_est: np.ndarray, T_gt: np.ndarray) -> dict[str, float]:
    A = np.asarray(T_est, dtype=np.float64)
    G = np.asarray(T_gt, dtype=np.float64)
    dt = A[:3, 3] - G[:3, 3]
    xy = math.hypot(float(dt[0]), float(dt[1]))
    xyz = float(np.linalg.norm(dt))
    yaw_est = math.atan2(A[1, 0], A[0, 0])
    yaw_gt = math.atan2(G[1, 0], G[0, 0])
    return {
        'translation_xy_m': float(xy),
        'translation_3d_m': xyz,
        'yaw_deg': abs(math.degrees(wrap_angle(yaw_est - yaw_gt))),
        'rotation_deg': rotation_error_deg(A[:3, :3], G[:3, :3]),
    }


def scan_context_pose_guess(reference_pose_row, relative_yaw_rad: float) -> np.ndarray:
    T = pose_row_to_matrix(reference_pose_row)
    # Keep reference translation/roll/pitch and change yaw by the SC estimate.
    row = np.asarray(reference_pose_row, dtype=np.float64).copy()
    row[6] = wrap_angle(float(row[6]) + float(relative_yaw_rad))
    return pose_row_to_matrix(row)


def require_frontend_dependencies(require_kiss: bool = True, require_small_gicp: bool = True) -> dict[str, object]:
    """Resolve native frontend dependencies without loading both OpenMP runtimes.

    On macOS arm64 the PyPI wheels for KISS-Matcher and small_gicp can each
    carry/link libomp. Importing both into one interpreter may abort with OMP
    Error #15.  KISS is therefore presence-checked here and loaded only inside
    an isolated worker process; small_gicp remains in the parent process.
    """
    missing = []
    mods = {}
    if require_kiss:
        try:
            if importlib.util.find_spec('kiss_matcher') is None:
                raise ModuleNotFoundError('kiss_matcher')
            mods['kiss_matcher'] = None  # loaded in IsolatedKISSMatcherBackend worker only
        except Exception as exc:
            missing.append(('kiss_matcher', exc))
    if require_small_gicp:
        try:
            mods['small_gicp'] = importlib.import_module('small_gicp')
        except Exception as exc:  # pragma: no cover - depends on environment
            missing.append(('small_gicp', exc))
    if missing:
        names = ', '.join(x[0] for x in missing)
        raise RuntimeError(
            'Step 9.5 global-to-local frontend is missing binary dependencies: '
            f'{names}. Install with: pip install "kiss-matcher>=1.0.1,<1.1" "small-gicp>=1.0.1,<1.1"'
        ) from missing[0][1]
    return mods


class KISSMatcherBackend:
    def __init__(self, voxel_size_m: float = 0.30, module=None):
        self.voxel_size_m = float(voxel_size_m)
        if self.voxel_size_m <= 0:
            raise ValueError('KISS voxel size must be positive')
        self.km = module if module is not None else importlib.import_module('kiss_matcher')

    def estimate(self, source_points: np.ndarray, target_points: np.ndarray) -> RobustRegistrationResult:
        src = np.asarray(source_points, dtype=np.float32)
        tgt = np.asarray(target_points, dtype=np.float32)
        if len(src) < 20 or len(tgt) < 20:
            return RobustRegistrationResult(np.eye(4), False, 0, 0, float('inf'), 0.0, 'kiss_matcher', failure_reason='too_few_points')
        t0 = time.perf_counter_ns()
        try:
            cfg = self.km.KISSMatcherConfig(self.voxel_size_m)
            matcher = self.km.KISSMatcher(cfg)
            result = matcher.estimate(src, tgt)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = np.asarray(result.rotation, dtype=np.float64)
            T[:3, 3] = np.asarray(result.translation, dtype=np.float64).reshape(3)
            rin = int(matcher.get_num_rotation_inliers())
            fin = int(matcher.get_num_final_inliers())
            ok = bool(np.all(np.isfinite(T)) and fin >= 3)
            reason = '' if ok else 'insufficient_inliers_or_nonfinite'
            return RobustRegistrationResult(T, ok, fin, 0, float('nan'), (time.perf_counter_ns()-t0)/1e6,
                                             'kiss_matcher', rotation_inliers=rin, failure_reason=reason)
        except Exception as exc:  # pragma: no cover - backend-dependent
            return RobustRegistrationResult(np.eye(4), False, 0, 0, float('inf'), (time.perf_counter_ns()-t0)/1e6,
                                             'kiss_matcher', failure_reason=f'{type(exc).__name__}: {exc}')


def _kiss_worker_main(conn, voxel_size_m: float):
    """Worker target that is the only process importing kiss_matcher."""
    try:
        km = importlib.import_module('kiss_matcher')
        backend = KISSMatcherBackend(voxel_size_m, module=km)
        conn.send(('ready', None))
        while True:
            msg = conn.recv()
            if msg is None or msg[0] == 'close':
                break
            op, src, tgt = msg
            if op != 'estimate':
                conn.send(('error', f'unknown operation: {op}'))
                continue
            conn.send(('result', backend.estimate(src, tgt)))
    except BaseException as exc:  # process boundary: propagate readable failure
        try:
            conn.send(('fatal', f'{type(exc).__name__}: {exc}'))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


class IsolatedKISSMatcherBackend:
    """Persistent spawn-process wrapper around KISS-Matcher.

    This avoids loading KISS-Matcher's OpenMP runtime into the same macOS
    process as small_gicp.  The worker is persistent, so startup cost is paid
    once per benchmark rather than once per registration.
    """
    def __init__(self, voxel_size_m: float = 0.30, start_timeout_s: float = 30.0):
        self.voxel_size_m = float(voxel_size_m)
        if self.voxel_size_m <= 0:
            raise ValueError('KISS voxel size must be positive')
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe(duplex=True)
        self._conn = parent
        self._proc = ctx.Process(target=_kiss_worker_main, args=(child, self.voxel_size_m),
                                 name='step95-kiss-worker', daemon=True)
        self._proc.start()
        child.close()
        if not self._conn.poll(float(start_timeout_s)):
            self.close(force=True)
            raise RuntimeError('KISS worker did not start within timeout')
        kind, payload = self._conn.recv()
        if kind != 'ready':
            self.close(force=True)
            raise RuntimeError(f'KISS worker failed to initialize: {payload}')

    def estimate(self, source_points: np.ndarray, target_points: np.ndarray) -> RobustRegistrationResult:
        if self._proc is None or not self._proc.is_alive():
            return RobustRegistrationResult(np.eye(4), False, 0, 0, float('inf'), 0.0,
                                             'kiss_matcher_isolated', failure_reason='worker_not_alive')
        src = np.asarray(source_points, dtype=np.float32)
        tgt = np.asarray(target_points, dtype=np.float32)
        t0 = time.perf_counter_ns()
        try:
            self._conn.send(('estimate', src, tgt))
            kind, payload = self._conn.recv()
            if kind == 'result':
                result = payload
                # Include IPC in the measured backend runtime, not just native compute.
                result.runtime_ms = (time.perf_counter_ns() - t0) / 1e6
                result.backend = 'kiss_matcher_isolated'
                return result
            return RobustRegistrationResult(np.eye(4), False, 0, 0, float('inf'),
                                             (time.perf_counter_ns()-t0)/1e6, 'kiss_matcher_isolated',
                                             failure_reason=f'worker_{kind}: {payload}')
        except (EOFError, BrokenPipeError, OSError) as exc:
            return RobustRegistrationResult(np.eye(4), False, 0, 0, float('inf'),
                                             (time.perf_counter_ns()-t0)/1e6, 'kiss_matcher_isolated',
                                             failure_reason=f'worker_io_error: {exc}')

    def close(self, force: bool = False):
        proc = getattr(self, '_proc', None)
        conn = getattr(self, '_conn', None)
        if proc is None:
            return
        if proc.is_alive() and not force:
            try:
                conn.send(('close',))
                proc.join(timeout=3.0)
            except Exception:
                pass
        if proc.is_alive():
            proc.terminate(); proc.join(timeout=3.0)
        try:
            conn.close()
        except Exception:
            pass
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class SmallGICPBackend:
    def __init__(self, *, downsampling_resolution_m: float = 0.40,
                 max_correspondence_distance_m: float = 2.0, num_threads: int = 4,
                 registration_type: str = 'GICP', module=None):
        self.downsampling_resolution_m = float(downsampling_resolution_m)
        self.max_correspondence_distance_m = float(max_correspondence_distance_m)
        self.num_threads = int(num_threads)
        self.registration_type = str(registration_type)
        self.sg = module if module is not None else importlib.import_module('small_gicp')

    def align(self, target_points: np.ndarray, source_points: np.ndarray, init_T_target_source: np.ndarray) -> RobustRegistrationResult:
        tgt = np.asarray(target_points, dtype=np.float64)
        src = np.asarray(source_points, dtype=np.float64)
        init = np.asarray(init_T_target_source, dtype=np.float64)
        if len(src) < 20 or len(tgt) < 20:
            return RobustRegistrationResult(init.copy(), False, 0, 0, float('inf'), 0.0, 'small_gicp', failure_reason='too_few_points')
        t0 = time.perf_counter_ns()
        try:
            result = self.sg.align(
                tgt, src,
                init_T_target_source=init,
                registration_type=self.registration_type,
                downsampling_resolution=self.downsampling_resolution_m,
                max_correspondence_distance=self.max_correspondence_distance_m,
                num_threads=self.num_threads,
            )
            T = np.asarray(result.T_target_source, dtype=np.float64)
            ok = bool(getattr(result, 'converged', True) and T.shape == (4,4) and np.all(np.isfinite(T)))
            return RobustRegistrationResult(
                T if T.shape == (4,4) else init.copy(), ok,
                int(getattr(result, 'num_inliers', 0)), int(getattr(result, 'iterations', 0)),
                float(getattr(result, 'e', float('nan'))), (time.perf_counter_ns()-t0)/1e6,
                'small_gicp', failure_reason='' if ok else 'not_converged_or_nonfinite')
        except Exception as exc:  # pragma: no cover - backend-dependent
            return RobustRegistrationResult(init.copy(), False, 0, 0, float('inf'), (time.perf_counter_ns()-t0)/1e6,
                                             'small_gicp', failure_reason=f'{type(exc).__name__}: {exc}')


def compose_world_query(reference_pose_row, T_reference_query: np.ndarray) -> np.ndarray:
    return pose_row_to_matrix(reference_pose_row) @ np.asarray(T_reference_query, dtype=np.float64)


__all__ = [
    'ScanContextConfig', 'ScanContextMatch', 'ScanContextDatabase', 'RobustRegistrationResult', 'GlobalHypothesis',
    'make_scan_context', 'scan_context_ring_key', 'scan_context_align', 'pose_row_to_matrix', 'matrix_to_pose3d',
    'rotation_error_deg', 'pose_errors_6d', 'scan_context_pose_guess', 'require_frontend_dependencies',
    'KISSMatcherBackend', 'IsolatedKISSMatcherBackend', 'SmallGICPBackend', 'compose_world_query',
]
