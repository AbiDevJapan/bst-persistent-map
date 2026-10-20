"""Correction-aware, identifier-independent lifelong occupancy map research core.

The existing versioned_map Step 10 and localization implementations remain available.
This package uses immutable local measurements, precise pose revisions, SQLite MVCC
and compressed exact occupancy pages. It does not require the BST-ID codec.
"""
from .engine import CorrectionMap, UpdateResult
from .voxel import GRID_SIDE, Pose, tile_keyed_indices

__all__ = ['CorrectionMap', 'UpdateResult', 'Pose', 'GRID_SIDE', 'tile_keyed_indices']
