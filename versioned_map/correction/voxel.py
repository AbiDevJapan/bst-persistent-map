"""Deterministic signed voxel coordinates and adaptive bitplane/sparse tile pages."""
from __future__ import annotations

import hashlib
import math
import zlib
from dataclasses import dataclass

import numpy as np

GRID_SIDE = 32
N_BITS = GRID_SIDE ** 3
N_BYTES = N_BITS // 8
_DTYPE_U16 = np.dtype('<u2')
_DTYPE_U32 = np.dtype('<u4')
# SQLite storage format is independent of numpy's native byte order.
COUNT_DTYPE = np.dtype([('index', '<u2'), ('count', '<u4')])


def _points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError('points must be a finite Nx3 floating array')
    return np.ascontiguousarray(pts)


@dataclass(frozen=True)
class Pose:
    matrix: np.ndarray

    def __post_init__(self):
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (4, 4) or not np.isfinite(m).all():
            raise ValueError('pose must be a finite 4x4 matrix')
        if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
            raise ValueError('invalid homogeneous pose last row')
        r = m[:3, :3]
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(r), 1.0, atol=1e-6):
            raise ValueError('rotation must be orthonormal and proper')
        object.__setattr__(self, 'matrix', np.ascontiguousarray(m))

    @staticmethod
    def identity() -> 'Pose':
        return Pose(np.eye(4, dtype=np.float64))

    @staticmethod
    def translation(x: float, y: float, z: float) -> 'Pose':
        m = np.eye(4, dtype=np.float64)
        m[:3, 3] = x, y, z
        return Pose(m)

    def transform(self, local_xyz: np.ndarray) -> np.ndarray:
        pts = _points(local_xyz)
        return pts @ self.matrix[:3, :3].T + self.matrix[:3, 3]


def voxelize(points: np.ndarray, pose: Pose, voxel_size: float, origin: tuple[float, float, float]) -> np.ndarray:
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('voxel_size must be positive and finite')
    world = pose.transform(points)
    scaled = (world - np.asarray(origin, dtype=np.float64)) / float(voxel_size)
    if np.any(np.abs(scaled) > 2**51):
        raise OverflowError('map coordinates exceed safe float-to-integer grid range')
    return np.floor(scaled).astype(np.int64)


def tile_keyed_indices(points: np.ndarray, pose: Pose, voxel_size: float = .5,
                       origin: tuple[float,float,float]=(0.0,0.0,0.0)) -> dict[tuple[int,int,int], np.ndarray]:
    v = voxelize(points, pose, voxel_size, origin)
    tiles = np.floor_divide(v, GRID_SIDE)
    local = v - tiles * GRID_SIDE
    indices = (local[:,0] + GRID_SIDE*local[:,1] + (GRID_SIDE**2)*local[:,2]).astype(_DTYPE_U16)
    if len(indices)==0:
        return {}
    # Group in vectorized NumPy rather than creating a Python object per point.
    # This matters for 100k+ point submaps and preserves signed tile coordinates.
    unique_tiles,inverse=np.unique(tiles,axis=0,return_inverse=True)
    order=np.lexsort((indices,inverse))
    labels=inverse[order]; sorted_ix=indices[order]
    splits=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(order)]
    return {tuple(map(int,unique_tiles[labels[splits[j]]])):
            np.unique(sorted_ix[splits[j]:splits[j+1]])
            for j in range(len(splits)-1)}


def encode_indices(indices: np.ndarray) -> tuple[str, bytes, bytes, int]:
    """Return (encoding, zlib payload, canonical SHA256, number of occupied voxels)."""
    ii = np.unique(np.asarray(indices, dtype=_DTYPE_U16))
    if ii.size and int(ii.max()) >= N_BITS:
        raise ValueError('tile index out of range')
    # Packing a boolean lattice is faster than scatter OR into the same bytes.
    # It is still LSB-first and retains the Step 11.0 canonical SHA-256 format.
    active = np.zeros(N_BITS, dtype=np.uint8)
    active[ii] = 1
    canonical = np.packbits(active, bitorder='little').tobytes()
    digest = hashlib.sha256(canonical).digest()
    d = zlib.compress(canonical, level=6)
    sparse = zlib.compress(ii.tobytes(), level=6)
    if len(sparse) < len(d):
        return 'sparse_u16', sparse, digest, int(ii.size)
    return 'bitplane', d, digest, int(ii.size)


def decode_indices(encoding: str, data: bytes) -> np.ndarray:
    raw = zlib.decompress(data)
    if encoding == 'bitplane':
        if len(raw) != N_BYTES:
            raise ValueError('corrupt bitplane length')
        return np.flatnonzero(np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little')).astype(_DTYPE_U16)
    if encoding == 'sparse_u16':
        if len(raw) % 2 != 0:
            raise ValueError('corrupt sparse payload length')
        ids = np.frombuffer(raw, dtype=_DTYPE_U16).copy()
        if len(ids) and (int(ids[-1]) >= N_BITS or np.any(ids[1:] <= ids[:-1])):
            raise ValueError('corrupt or unsorted sparse payload')
        return ids
    raise ValueError(f'unknown encoding {encoding}')


def full_mask(encoding: str, data: bytes) -> np.ndarray:
    if encoding == 'bitplane':
        arr = np.frombuffer(zlib.decompress(data), dtype=np.uint8)
        if arr.size != N_BYTES:
            raise ValueError('corrupt bitplane length')
        return arr.copy()
    ii = decode_indices(encoding, data)
    active = np.zeros(N_BITS, dtype=np.uint8)
    active[ii] = 1
    return np.packbits(active, bitorder='little')


def indices_to_grid(indices: np.ndarray, tile: tuple[int,int,int]) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    xyz = np.column_stack((indices % GRID_SIDE,
                           indices // GRID_SIDE % GRID_SIDE,
                           indices // (GRID_SIDE*GRID_SIDE)))
    return xyz + GRID_SIDE*np.asarray(tile,dtype=np.int64)


def ordered_grid(points: np.ndarray) -> np.ndarray:
    v = np.asarray(points,dtype=np.int64).reshape((-1,3))
    if not len(v): return np.empty((0,3), dtype=np.int64)
    return np.unique(v, axis=0)


def count_pack(counts: dict[int,int]) -> bytes:
    arr = np.empty(len(counts), dtype=COUNT_DTYPE)
    for i,(ix,cnt) in enumerate(sorted(counts.items())):
        if cnt < 1 or cnt > 2**32-1 or ix < 0 or ix >= N_BITS:
            raise OverflowError('invalid source counter')
        arr[i] = ix, cnt
    return zlib.compress(arr.tobytes(), level=6)


def count_unpack(data: bytes | None) -> dict[int,int]:
    if data is None: return {}
    raw=zlib.decompress(data)
    if len(raw)%COUNT_DTYPE.itemsize: raise ValueError('corrupt count table')
    arr=np.frombuffer(raw,dtype=COUNT_DTYPE)
    return {int(ix):int(cnt) for ix,cnt in arr.tolist()}


def grid_roi_bounds(bounds):
    if len(bounds) != 6: raise ValueError('bounds: xmin xmax ymin ymax zmin zmax')
    b = [int(v) for v in bounds]
    if any(b[k] >= b[k+1] for k in (0,2,4)):
        raise ValueError('bounds must be half-open with a positive extent')
    return tuple(b)


def roi_tile_range(bounds):
    b=grid_roi_bounds(bounds)
    return tuple((b[i]//GRID_SIDE, (b[i+1]-1)//GRID_SIDE) for i in (0,2,4))


def filter_roi(coords: np.ndarray, bounds):
    b=grid_roi_bounds(bounds)
    return coords[(coords[:,0]>=b[0]) & (coords[:,0]<b[1]) &
                  (coords[:,1]>=b[2]) & (coords[:,1]<b[3]) &
                  (coords[:,2]>=b[4]) & (coords[:,2]<b[5])]


def canonical_blob(points: np.ndarray) -> bytes:
    pts = _points(points)
    return zlib.compress(pts.astype('<f8',copy=False).tobytes(), 6)


def load_blob(data: bytes) -> np.ndarray:
    raw=zlib.decompress(data)
    if len(raw)%24: raise ValueError('corrupt local measurement blob')
    return np.frombuffer(raw,dtype='<f8').reshape(-1,3).copy()
