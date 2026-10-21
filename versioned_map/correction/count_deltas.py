"""Step 11.3: sparse signed contribution deltas for versioned voxel pages.

This module is intentionally independent of BST-ID and spatial indexing.
It applies all removals and insertions together, keeping per-source overlap
multiplicities correct without allocating Python sets for each voxel.

All indices are local to a 32^3 tile and stored as sorted uint16 values.
Counters are unsigned 32-bit and stored in the existing on-disk format.
"""
from __future__ import annotations

import zlib
from typing import Iterable

import numpy as np

from .voxel import COUNT_DTYPE, N_BITS


def unpack_counts(raw: bytes | None) -> np.ndarray:
    """Decode existing counter blobs into a validated, writable structured array."""
    if raw is None:
        return np.empty(0, dtype=COUNT_DTYPE)
    unpacked = zlib.decompress(raw)
    if len(unpacked) % COUNT_DTYPE.itemsize:
        raise ValueError('corrupt count table length')
    counts = np.frombuffer(unpacked, dtype=COUNT_DTYPE).copy()
    if (counts.size and (int(counts['index'][-1]) >= N_BITS or
                         np.any(counts['index'][1:] <= counts['index'][:-1]) or
                         np.any(counts['count'] == 0))):
        raise ValueError('corrupt count table indices/counts')
    return counts


def pack_counts(counts: np.ndarray) -> bytes:
    arr = np.asarray(counts, dtype=COUNT_DTYPE)
    if (arr.size and (int(arr['index'][-1]) >= N_BITS or
                      np.any(arr['index'][1:] <= arr['index'][:-1]) or
                      np.any(arr['count'] == 0))):
        raise ValueError('invalid count array')
    return zlib.compress(arr.tobytes(), level=6)


def signed_deltas(removals: Iterable[np.ndarray], additions: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return unique sorted indices and signed net change per index.

    A source's indices are deduplicated during voxelization. Source multiplicity
    across many scans is preserved by summing signed contributions, including
    removals and additions at the same voxel within a single correction batch.
    """
    parts: list[np.ndarray] = []
    signs: list[np.ndarray] = []
    for arrays, sign in ((removals, -1), (additions, 1)):
        for values in arrays:
            ids = np.asarray(values, dtype='<u2').reshape(-1)
            if ids.size:
                if int(ids.max()) >= N_BITS:
                    raise ValueError('tile voxel index out of range')
                parts.append(ids)
                signs.append(np.full(ids.size, sign, dtype=np.int64))
    if not parts:
        return np.empty(0, dtype='<u2'), np.empty(0, dtype=np.int64)
    all_ids = np.concatenate(parts)
    all_signs = np.concatenate(signs)
    unique, inv = np.unique(all_ids, return_inverse=True)
    net = np.bincount(inv, weights=all_signs, minlength=len(unique)).astype(np.int64)
    nonzero = net != 0
    return unique[nonzero], net[nonzero]


def apply_deltas(previous: np.ndarray, indices: np.ndarray, changes: np.ndarray) -> np.ndarray:
    """Merge sparse signed changes with persisted counter array (no 32^3 scan)."""
    previous = np.asarray(previous, dtype=COUNT_DTYPE)
    indices = np.asarray(indices, dtype='<u2').reshape(-1)
    changes = np.asarray(changes, dtype=np.int64).reshape(-1)
    if indices.size != changes.size:
        raise ValueError('delta sizes differ')
    if indices.size == 0:
        return previous.copy()
    if np.any(indices[1:] <= indices[:-1]):
        raise ValueError('deltas not sorted and unique')
    keys = np.union1d(previous['index'], indices)
    result_counts = np.zeros(len(keys), dtype=np.int64)
    if len(previous):
        result_counts[np.searchsorted(keys, previous['index'])] = previous['count']
    result_counts[np.searchsorted(keys, indices)] += changes
    if np.any(result_counts < 0):
        raise RuntimeError('removing source contribution absent from a tile')
    if np.any(result_counts > np.iinfo(np.uint32).max):
        raise OverflowError('voxel source counter overflow')
    positive = result_counts > 0
    out = np.empty(int(positive.sum()), dtype=COUNT_DTYPE)
    out['index'] = keys[positive]
    out['count'] = result_counts[positive].astype('<u4')
    return out
