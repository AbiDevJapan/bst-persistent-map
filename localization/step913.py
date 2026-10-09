from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import time

import numpy as np

from localization.step903b import prepare_surfel_pyramid
from localization.step912 import HierarchicalLODView, HierarchicalLODBSTIndex
from representations.packed_scale import metric_box_to_bounds


@dataclass
class MaterializedLODMap:
    """Standalone deployed LOD payload detached from the construction hierarchy.

    The map stores only the statistics used by the Step-9 surfel matcher plus a
    compact root->LOD ragged directory.  No leaf table, merge forest, node IDs,
    or unselected hierarchy nodes are required at runtime.

    Arrays are intentionally kept at the same precision as Step 9.1.2 so the
    materialized path is numerically equivalent to the hierarchy-backed K view.
    """

    spec: object
    origin_local_m: tuple[float, float, float]
    global_block_ids: np.ndarray
    root_starts: np.ndarray
    root_counts: np.ndarray
    counts: np.ndarray
    centroids: np.ndarray
    cov6: np.ndarray
    max_primitives_per_root: int
    materialize_ms: float

    @property
    def map_elements(self) -> int:
        return len(self.counts)

    def geometry_payload_nbytes(self) -> int:
        return int(self.counts.nbytes + self.centroids.nbytes + self.cov6.nbytes)

    def directory_nbytes(self) -> int:
        return int(self.global_block_ids.nbytes + self.root_starts.nbytes + self.root_counts.nbytes)

    def runtime_nbytes(self) -> int:
        return int(self.geometry_payload_nbytes() + self.directory_nbytes())

    def mean_voxels_per_primitive(self, voxel_equivalent: int) -> float:
        return float(voxel_equivalent / self.map_elements) if self.map_elements else 0.0


@dataclass
class MaterializationVerification:
    payload_exact: bool
    directory_exact: bool
    query_exact: bool
    pyramid_exact: bool
    checks: int

    @property
    def passed(self) -> bool:
        return bool(self.payload_exact and self.directory_exact and self.query_exact and self.pyramid_exact)


def materialize_lod_view(view: HierarchicalLODView) -> MaterializedLODMap:
    """Copy one frozen LOD view into a hierarchy-free runtime representation."""
    t0 = time.perf_counter_ns()
    h = view.hierarchy
    ids = np.asarray(view.selected_node_ids, dtype=np.int64)

    counts64 = h.node_counts(ids)
    if len(counts64) and int(np.max(counts64)) > np.iinfo(np.uint32).max:
        raise OverflowError('selected LOD point count exceeds uint32; exact compact materialization unavailable')

    # Use the actual runtime inputs consumed by prepare_surfel_pyramid.
    # Keeping float32 here preserves the Step-9.1.2 statistics exactly.
    counts = counts64.astype(np.uint32, copy=True)
    centroids = h.node_centroids(ids).astype(np.float32, copy=True)
    cov6 = h.node_cov6(ids).astype(np.float32, copy=True)

    gids = np.asarray(h.block_map.global_block_ids, dtype=np.int32).copy()
    starts = np.asarray(view.root_starts).copy()
    root_counts = np.asarray(view.root_counts).copy()

    if len(gids) != len(starts) or len(gids) != len(root_counts):
        raise AssertionError('materialized root directory length mismatch')
    if len(root_counts) and int(root_counts.max()) > int(view.max_primitives_per_root):
        raise AssertionError('materialized root directory exceeds K cap')
    if len(root_counts) and int(np.sum(root_counts, dtype=np.int64)) != len(counts):
        raise AssertionError('materialized root directory does not cover selected LOD payload')
    if len(gids) > 1 and np.any(gids[1:] <= gids[:-1]):
        raise AssertionError('global block IDs must be strictly increasing for searchsorted lookup')

    return MaterializedLODMap(
        spec=h.aligned_map.spec,
        origin_local_m=h.aligned_map.origin_local_m,
        global_block_ids=gids,
        root_starts=starts,
        root_counts=root_counts,
        counts=counts,
        centroids=centroids,
        cov6=cov6,
        max_primitives_per_root=int(view.max_primitives_per_root),
        materialize_ms=(time.perf_counter_ns() - t0) / 1e6,
    )


class MaterializedLODBSTIndex:
    """Step-8.5 time-index roots -> hierarchy-free materialized LOD surfels."""

    def __init__(self, lod_map: MaterializedLODMap, bst_map, session_index: int):
        self.map = lod_map
        self.bst_map = bst_map
        self.session_index = int(session_index)
        self.last_stats = {}

    def query_radius_ids(self, cx: float, cy: float, radius_m: float) -> np.ndarray:
        core = self.bst_map.core
        ox, oy, _ = self.map.origin_local_m
        reg = (
            cx - radius_m - ox, cx + radius_m - ox,
            cy - radius_m - oy, cy + radius_m - oy,
            0.0, core.spec.size_z_m,
        )
        bounds = metric_box_to_bounds(core.spec, reg)
        gids, _, refs, scanned = core._select_mode(bounds, self.session_index, index_mode='time')
        if not len(gids):
            self.last_stats = {
                'blocks_scanned': int(scanned), 'bucket_refs': int(refs),
                'active_blocks': 0, 'returned': 0,
            }
            return np.empty(0, dtype=np.int64)

        roots = self.map.global_block_ids
        pos = np.searchsorted(roots, gids)
        if np.any(pos >= len(roots)) or not np.array_equal(roots[pos], gids):
            raise AssertionError('active BST root missing from materialized LOD directory')

        cnt = self.map.root_counts[pos].astype(np.int64, copy=False)
        total = int(np.sum(cnt, dtype=np.int64))
        if total == 0:
            self.last_stats = {
                'blocks_scanned': int(scanned), 'bucket_refs': int(refs),
                'active_blocks': int(len(gids)), 'returned': 0,
            }
            return np.empty(0, dtype=np.int64)

        # Expand the compact ragged directory without retaining any hierarchy IDs.
        rows = np.empty(total, dtype=np.int64)
        q = 0
        for j, c in zip(pos, cnt):
            a = int(self.map.root_starts[j]); c = int(c)
            rows[q:q+c] = np.arange(a, a+c, dtype=np.int64)
            q += c

        c = self.map.centroids[rows]
        keep = (c[:, 0] - cx) ** 2 + (c[:, 1] - cy) ** 2 <= radius_m * radius_m
        rows = rows[keep]
        self.last_stats = {
            'blocks_scanned': int(scanned), 'bucket_refs': int(refs),
            'active_blocks': int(len(gids)), 'returned': int(len(rows)),
        }
        return rows

    def prepare_pyramid(self, ids, levels, cfg):
        ids = np.asarray(ids, dtype=np.int64)
        local = SimpleNamespace(
            centroids=self.map.centroids,
            cov6=self.map.cov6,
            counts=self.map.counts,
        )
        return prepare_surfel_pyramid(local, ids, levels, cfg)

    def nbytes(self) -> int:
        return self.map.directory_nbytes()


def verify_materialization(
    view: HierarchicalLODView,
    materialized: MaterializedLODMap,
    bst_map,
    session_index: int,
    query_centers: list[tuple[float, float]],
    radius_m: float,
    levels,
    cfg,
    max_pyramid_checks: int = 3,
) -> MaterializationVerification:
    """Assert the deployed table is equivalent to the Step-9.1.2 K view.

    Payload and directory are checked exactly.  Query outputs are compared after
    mapping hierarchy node IDs to their selected-row positions.  A few local map
    pyramids are also compared exactly/allclose before the construction hierarchy
    can be discarded.
    """
    h = view.hierarchy
    src = np.asarray(view.selected_node_ids, dtype=np.int64)
    payload_exact = (
        np.array_equal(materialized.counts, h.node_counts(src).astype(np.uint32))
        and np.array_equal(materialized.centroids, h.node_centroids(src).astype(np.float32))
        and np.array_equal(materialized.cov6, h.node_cov6(src).astype(np.float32))
    )
    directory_exact = (
        np.array_equal(materialized.global_block_ids, h.block_map.global_block_ids.astype(np.int32))
        and np.array_equal(materialized.root_starts, view.root_starts)
        and np.array_equal(materialized.root_counts, view.root_counts)
    )

    href = HierarchicalLODBSTIndex(view, bst_map, session_index)
    hnew = MaterializedLODBSTIndex(materialized, bst_map, session_index)
    # Selected node IDs are unique because the LOD forms a disjoint cover.
    order = np.argsort(src)
    src_sorted = src[order]
    query_exact = True
    pyramid_exact = True
    checks = 0
    pyramid_checks = 0

    for cx, cy in query_centers:
        cx = float(cx); cy = float(cy)
        if not (np.isfinite(cx) and np.isfinite(cy)):
            raise ValueError(
                f'non-finite materialization verification center: cx={cx!r}, cy={cy!r}; '
                'query selection must reject non-finite ground-truth poses'
            )
        old_ids = href.query_radius_ids(cx, cy, float(radius_m))
        new_rows = hnew.query_radius_ids(cx, cy, float(radius_m))
        pos = np.searchsorted(src_sorted, old_ids)
        if len(old_ids):
            if np.any(pos >= len(src_sorted)) or not np.array_equal(src_sorted[pos], old_ids):
                raise AssertionError('hierarchy query returned node not present in selected LOD table')
            old_rows = order[pos].astype(np.int64, copy=False)
        else:
            old_rows = np.empty(0, dtype=np.int64)
        if not np.array_equal(old_rows, new_rows):
            query_exact = False
            break
        checks += 1

        if pyramid_checks < int(max_pyramid_checks):
            p_old = href.prepare_pyramid(old_ids, levels, cfg)
            p_new = hnew.prepare_pyramid(new_rows, levels, cfg)
            if len(p_old) != len(p_new):
                pyramid_exact = False
                break
            for a, b in zip(p_old, p_new):
                if not (
                    np.array_equal(a.counts, b.counts)
                    and np.array_equal(a.valid, b.valid)
                    and np.allclose(a.centroids, b.centroids, atol=0.0, rtol=0.0)
                    and np.allclose(a.normals, b.normals, atol=1e-12, rtol=1e-12)
                ):
                    pyramid_exact = False
                    break
            if not pyramid_exact:
                break
            pyramid_checks += 1

    return MaterializationVerification(
        payload_exact=bool(payload_exact),
        directory_exact=bool(directory_exact),
        query_exact=bool(query_exact),
        pyramid_exact=bool(pyramid_exact),
        checks=int(checks),
    )


__all__ = [
    'MaterializedLODMap', 'MaterializedLODBSTIndex', 'MaterializationVerification',
    'materialize_lod_view', 'verify_materialization',
]
