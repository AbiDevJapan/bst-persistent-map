from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
import math
import time

import numpy as np

from localization.step90 import Pose3DOF, level_body_points, transform_leveled, wrap_angle
from localization.step901 import PointToPlaneICP3DOFConfig, PointToPlaneICP3DOFResult, _limit_increment
from localization.step902 import voxel_downsample_first
from localization.step903b import (
    PreparedSurfelLevel, CoarseToFineSurfelResult, aggregate_surfels,
    scan_to_map_coarse_to_fine_surfel_3dof,
)
from localization.step913 import MaterializedLODBSTIndex, MaterializedLODMap

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass
class PreparedPersistentSurfelLevel:
    config: object
    centroids: np.ndarray
    normals: np.ndarray
    valid: np.ndarray
    counts: np.ndarray
    tree: object
    persistence: np.ndarray


@dataclass
class CausalPersistenceBuild:
    """Quantized per-LOD persistence scores for each causal query session.

    ``scores_q8[t]`` contains one uint8 score per selected K=4 surfel for
    localizing session index ``t``.  Score 255 means the spatial support was
    occupied in every session available strictly before t (including the map
    session).  The query session itself and future sessions are never used.
    """
    scores_q8: dict[int, np.ndarray]
    stats: list[dict]
    build_ms: float
    expansion_ms: float
    membership_ms: float

    def score_bytes_per_active_session(self) -> int:
        if not self.scores_q8:
            return 0
        return int(next(iter(self.scores_q8.values())).nbytes)


def _sorted_membership(reference_codes: np.ndarray, target_codes: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_codes, dtype=np.uint64)
    tgt = np.asarray(target_codes, dtype=np.uint64)
    if len(tgt) > 1 and np.any(tgt[1:] < tgt[:-1]):
        tgt = np.sort(tgt)
    pos = np.searchsorted(tgt, ref)
    out = np.zeros(len(ref), dtype=bool)
    good = pos < len(tgt)
    if np.any(good):
        out[good] = tgt[pos[good]] == ref[good]
    return out


def expand_selected_nodes_to_leaves(view):
    """Return (leaf_row, selected_row) for the disjoint selected K view.

    Expansion is breadth-wise and visits each canonical map voxel exactly once.
    It is construction-only; neither mapping is retained in deployed runtime.
    """
    h = view.hierarchy
    nodes = np.asarray(view.selected_node_ids, dtype=np.int64)
    labels = np.arange(len(nodes), dtype=np.int32 if len(nodes) < 2**31 else np.int64)
    leaf_parts = []
    label_parts = []
    while len(nodes):
        leaf = nodes < h.leaf_count
        if np.any(leaf):
            leaf_parts.append(nodes[leaf].astype(np.int64, copy=False))
            label_parts.append(labels[leaf])
        internal = ~leaf
        if not np.any(internal):
            break
        a, b = h.children(nodes[internal])
        lab = labels[internal]
        nodes = np.column_stack((a, b)).reshape(-1)
        labels = np.repeat(lab, 2)
    leaves = np.concatenate(leaf_parts) if leaf_parts else np.empty(0, dtype=np.int64)
    labs = np.concatenate(label_parts) if label_parts else np.empty(0, dtype=np.int32)
    if len(leaves) != int(h.leaf_count):
        raise AssertionError(f'causal persistence leaf coverage mismatch: {len(leaves)} != {h.leaf_count}')
    if len(leaves) and (int(leaves.min()) < 0 or int(leaves.max()) >= int(h.leaf_count)):
        raise AssertionError('causal persistence expansion produced invalid leaf row')
    return leaves, labs


def build_causal_persistence(view, prepared, map_session_index: int, query_session_indices) -> CausalPersistenceBuild:
    """Build causal repeated-occupancy confidence for the frozen selected LOD.

    For query time t, each Jan/map-session canonical voxel receives

        p_leaf(t) = (# occupied sessions in [map_t, t)) / (t-map_t).

    Selected hierarchical surfels receive the voxel-volume mean of their leaf
    scores.  Scores are quantized to uint8 for deployment; 255 == 1.0.

    This is deliberately a *soft observation-persistence proxy*: absent voxels
    are not interpreted as observed free space.  No geometry is removed.
    """
    t_all = time.perf_counter_ns()
    map_t = int(map_session_index)
    qidx = sorted(set(int(x) for x in query_session_indices))
    if any(q <= map_t for q in qidx):
        raise ValueError('causal persistence query sessions must occur after the map session')
    if any(q >= len(prepared.session_codes) for q in qidx):
        raise ValueError('query session index exceeds prepared history')

    t0 = time.perf_counter_ns()
    leaf_rows, selected_labels = expand_selected_nodes_to_leaves(view)
    expansion_ms = (time.perf_counter_ns() - t0) / 1e6
    nsel = len(view.selected_node_ids)
    support = np.bincount(selected_labels, minlength=nsel).astype(np.float64)
    if not np.array_equal(support.astype(np.int64), view.hierarchy.node_volume(view.selected_node_ids).astype(np.int64)):
        raise AssertionError('selected LOD support counts do not match hierarchy node volume')

    t0 = time.perf_counter_ns()
    map_codes = np.asarray(view.hierarchy.aligned_map.linear_codes, dtype=np.uint64)
    # One bit per history session; current experiments use five sessions.
    if len(prepared.session_codes) > 8:
        raise ValueError('Step 9.3 uint8 causal presence mask supports at most 8 sessions')
    presence = np.zeros(len(map_codes), dtype=np.uint8)
    for s in range(map_t, max(qidx)):
        found = _sorted_membership(map_codes, prepared.session_codes[s])
        presence[found] |= np.uint8(1 << s)
    if not np.all((presence & np.uint8(1 << map_t)) != 0):
        raise AssertionError('map-session leaves must all be occupied in the map session')
    membership_ms = (time.perf_counter_ns() - t0) / 1e6

    bitcount = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    scores = {}
    stats = []
    for q in qidx:
        denom = q - map_t
        mask = 0
        for s in range(map_t, q):
            mask |= 1 << s
        leaf_seen = bitcount[np.bitwise_and(presence, np.uint8(mask))]
        # selected_labels partitions all map leaves, so bincount gives the exact
        # sum of causal occupied-session counts over each selected surfel support.
        obs = np.bincount(selected_labels, weights=leaf_seen[leaf_rows].astype(np.float64), minlength=nsel)
        p = obs / (support * float(denom))
        if np.any(~np.isfinite(p)) or np.any(p <= 0) or np.any(p > 1.0 + 1e-12):
            raise AssertionError('invalid causal persistence score')
        q8 = np.rint(np.clip(p, 0.0, 1.0) * 255.0).astype(np.uint8)
        scores[q] = q8
        pf = q8.astype(np.float32) / 255.0
        stats.append({
            'query_session_index': q,
            'prior_session_count': denom,
            'map_elements': nsel,
            'mean_persistence': float(np.mean(pf)),
            'median_persistence': float(np.median(pf)),
            'p10_persistence': float(np.quantile(pf, 0.10)),
            'p90_persistence': float(np.quantile(pf, 0.90)),
            'fraction_full_persistence': float(np.mean(q8 == 255)),
            'fraction_half_or_lower': float(np.mean(pf <= 0.5)),
            'score_payload_mib': q8.nbytes / (1024**2),
        })

    # With only the map session available, every map voxel must have p=1. This
    # makes Jan->Feb an exact no-extra-history control.
    first = map_t + 1
    if first in scores and not np.all(scores[first] == 255):
        raise AssertionError('first causal query session must have all-one persistence weights')

    return CausalPersistenceBuild(
        scores_q8=scores,
        stats=stats,
        build_ms=(time.perf_counter_ns() - t_all) / 1e6,
        expansion_ms=expansion_ms,
        membership_ms=membership_ms,
    )


def _aggregate_persistence(centroids, counts, persistence, voxel_m):
    mu = np.asarray(centroids, dtype=np.float64)
    w = np.asarray(counts, dtype=np.float64)
    p = np.asarray(persistence, dtype=np.float64)
    if not len(mu):
        return np.empty(0, dtype=np.float32)
    keys = np.floor(mu / float(voxel_m)).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]; w = w[order]; p = p[order]
    starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
    W = np.add.reduceat(w, starts)
    wp = np.add.reduceat(w * p, starts)
    return np.divide(wp, W, out=np.ones(len(W), dtype=np.float64), where=W > 0).astype(np.float32)


def prepare_persistent_surfel_pyramid(materialized: MaterializedLODMap, ids, persistence_q8, levels, cfg):
    if cKDTree is None:
        raise RuntimeError('Step 9.3 requires scipy.spatial.cKDTree') from _SCIPY_IMPORT_ERROR
    ids = np.asarray(ids, dtype=np.int64)
    p_all = np.asarray(persistence_q8, dtype=np.uint8)
    if len(p_all) != materialized.map_elements:
        raise ValueError('persistence score length must equal materialized LOD rows')
    c = materialized.centroids[ids]
    cv = materialized.cov6[ids]
    w = materialized.counts[ids]
    p = p_all[ids].astype(np.float32) / 255.0
    out = []
    for lev in levels:
        mu, n, valid, cnt, _ = aggregate_surfels(
            c, cv, w, lev.map_voxel_m,
            cfg.normal_min_neighbors, cfg.normal_max_curvature,
            cfg.normal_min_horizontal_component,
        )
        pp = _aggregate_persistence(c, w, p, lev.map_voxel_m)
        if len(pp) != len(mu):
            raise AssertionError('persistence pyramid grouping differs from geometry grouping')
        tree = cKDTree(mu) if len(mu) else None
        out.append(PreparedPersistentSurfelLevel(lev, mu, n, valid, cnt, tree, pp))
    return out


class PersistenceWeightedLODBSTIndex:
    """Frozen compact K=4 geometry plus one causal uint8 persistence vector."""
    def __init__(self, materialized: MaterializedLODMap, bst_map, session_index: int, persistence_q8, history_session_count: int):
        self.base = MaterializedLODBSTIndex(materialized, bst_map, session_index)
        self.map = materialized
        self.persistence_q8 = np.asarray(persistence_q8, dtype=np.uint8)
        if len(self.persistence_q8) != materialized.map_elements:
            raise ValueError('persistence_q8 length mismatch')
        self.history_session_count = int(history_session_count)
        self.last_stats = {}

    def query_radius_ids(self, cx, cy, radius_m):
        ids = self.base.query_radius_ids(cx, cy, radius_m)
        self.last_stats = dict(self.base.last_stats)
        if len(ids):
            p = self.persistence_q8[ids].astype(np.float32) / 255.0
            self.last_stats.update({
                'mean_persistence': float(np.mean(p)),
                'median_persistence': float(np.median(p)),
                'full_persistence_fraction': float(np.mean(self.persistence_q8[ids] == 255)),
                'history_sessions': self.history_session_count,
            })
        else:
            self.last_stats.update({
                'mean_persistence': float('nan'), 'median_persistence': float('nan'),
                'full_persistence_fraction': float('nan'), 'history_sessions': self.history_session_count,
            })
        return ids

    def prepare_pyramid(self, ids, levels, cfg):
        return prepare_persistent_surfel_pyramid(self.map, ids, self.persistence_q8, levels, cfg)

    def nbytes(self):
        return int(self.base.nbytes() + self.persistence_q8.nbytes)


def scan_to_persistent_point_to_plane_icp_3dof(
    points_body, map_points, normals, normal_valid, persistence,
    initial_pose, fixed_z, fixed_roll, fixed_pitch, config=None, map_tree=None,
):
    """Point-to-plane ICP with causal persistence as a soft normal-equation weight.

    Correspondence search, validity checks, trimming, convergence thresholds and
    reported residuals are unchanged.  Only A and b use persistence weights.
    """
    if cKDTree is None:
        raise RuntimeError('Step 9.3 requires scipy.spatial.cKDTree') from _SCIPY_IMPORT_ERROR
    cfg = config or PointToPlaneICP3DOFConfig()
    pts = np.asarray(points_body, dtype=np.float64)
    mp = np.asarray(map_points, dtype=np.float64)
    ns_all = np.asarray(normals, dtype=np.float64)
    valid_all = np.asarray(normal_valid, dtype=bool)
    pers_all = np.asarray(persistence, dtype=np.float64)
    if len(pers_all) != len(mp):
        raise ValueError('persistence/map length mismatch')
    if len(pts) == 0:
        return PointToPlaneICP3DOFResult(initial_pose, False, 0, 0, float('inf'), 0.0, 'empty_scan')
    if len(mp) < cfg.min_correspondences:
        return PointToPlaneICP3DOFResult(initial_pose, False, 0, 0, float('inf'), 0.0, 'local_map_too_small')
    tree = map_tree if map_tree is not None else cKDTree(mp)
    q = level_body_points(pts, fixed_roll, fixed_pitch)
    pose = initial_pose; conv = False; reason = 'max_iterations'; lc = ln = 0; lr = la = float('inf')
    t0 = time.perf_counter_ns()
    for it in range(1, cfg.max_iterations + 1):
        world = transform_leveled(q, pose, fixed_z)
        dist, idx = tree.query(world, k=1, workers=1)
        keep = np.isfinite(dist) & (dist <= cfg.max_correspondence_m) & valid_all[idx]
        ids = np.flatnonzero(keep)
        if len(ids) < cfg.min_correspondences:
            reason = 'too_few_normal_correspondences'
            lc = int(np.count_nonzero(np.isfinite(dist) & (dist <= cfg.max_correspondence_m)))
            ln = int(len(ids)); break
        tidx = idx[ids]; targets = mp[tidx]; ns = ns_all[tidx]
        residual = np.einsum('ij,ij->i', ns, world[ids] - targets)
        if 0 < cfg.trim_fraction < 1 and len(ids) > cfg.min_correspondences:
            take = max(cfg.min_correspondences, int(math.ceil(len(ids) * cfg.trim_fraction)))
            sel = np.argpartition(np.abs(residual), take - 1)[:take]
            ids = ids[sel]; tidx = tidx[sel]; targets = targets[sel]; ns = ns[sel]; residual = residual[sel]
        ww = np.clip(pers_all[tidx], 1e-6, 1.0)
        qq = q[ids]; c0, s0 = math.cos(pose.yaw), math.sin(pose.yaw)
        dpx = -s0 * qq[:, 0] - c0 * qq[:, 1]
        dpy = c0 * qq[:, 0] - s0 * qq[:, 1]
        J = np.empty((len(ids), 3), dtype=np.float64)
        J[:, 0] = ns[:, 0]; J[:, 1] = ns[:, 1]; J[:, 2] = ns[:, 0] * dpx + ns[:, 1] * dpy
        A = J.T @ (ww[:, None] * J)
        b = -(J.T @ (ww * residual))
        A.flat[::4] += float(cfg.damping)
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(A, b, rcond=None)
        if not np.all(np.isfinite(delta)):
            reason = 'nonfinite_update'; break
        delta = _limit_increment(delta, cfg)
        pose = Pose3DOF(pose.x + float(delta[0]), pose.y + float(delta[1]), wrap_angle(pose.yaw + float(delta[2])))
        lc = int(np.count_nonzero(np.isfinite(dist) & (dist <= cfg.max_correspondence_m)))
        ln = int(len(ids))
        # Keep the same unweighted residual metrics as the frozen baseline.
        lr = float(math.sqrt(np.mean(residual * residual)))
        la = float(np.mean(np.abs(residual)))
        if math.hypot(float(delta[0]), float(delta[1])) <= cfg.translation_tolerance_m and abs(math.degrees(float(delta[2]))) <= cfg.yaw_tolerance_deg:
            conv = True; reason = ''; break
    return PointToPlaneICP3DOFResult(
        pose, conv, it if 'it' in locals() else 0, lc, lr,
        (time.perf_counter_ns() - t0) / 1e6, reason, ln, la,
    )


def scan_to_map_coarse_to_fine_persistent_3dof(points_body, prepared_levels, initial_pose, fixed_z, fixed_roll, fixed_pitch, base_config=None):
    cfg0 = base_config or PointToPlaneICP3DOFConfig()
    # Exact null-control path: if causal history provides no discrimination,
    # call the frozen matcher directly rather than relying on algebraic equality.
    if prepared_levels and all(np.all(p.persistence == 1.0) for p in prepared_levels):
        base = [PreparedSurfelLevel(p.config, p.centroids, p.normals, p.valid, p.counts, p.tree) for p in prepared_levels]
        return scan_to_map_coarse_to_fine_surfel_3dof(points_body, base, initial_pose, fixed_z, fixed_roll, fixed_pitch, cfg0)

    pose = initial_pose; results = []; total_it = 0; total_ms = 0.0; final = None
    pts = np.asarray(points_body, dtype=np.float64)
    for p in prepared_levels:
        lev = p.config
        scan = voxel_downsample_first(pts, lev.scan_voxel_m)
        cfg = replace(
            cfg0, max_iterations=int(lev.max_iterations),
            max_correspondence_m=float(lev.max_correspondence_m),
            trim_fraction=float(lev.trim_fraction),
            max_step_translation_m=float(lev.max_step_translation_m),
            max_step_yaw_deg=float(lev.max_step_yaw_deg),
        )
        res = scan_to_persistent_point_to_plane_icp_3dof(
            scan, p.centroids, p.normals, p.valid, p.persistence,
            pose, fixed_z, fixed_roll, fixed_pitch, cfg, p.tree,
        )
        results.append((lev.name, res)); total_it += res.iterations; total_ms += res.match_ms; pose = res.pose; final = res
        if res.failure_reason in {'empty_scan', 'local_map_too_small', 'too_few_correspondences', 'too_few_normal_correspondences', 'nonfinite_update'}:
            break
    if final is None:
        final = PointToPlaneICP3DOFResult(initial_pose, False, 0, 0, float('inf'), 0.0, 'no_levels')
    return CoarseToFineSurfelResult(
        pose, bool(final.converged and len(results) == len(prepared_levels)), total_it,
        final.correspondences, final.normal_correspondences, final.rmse_m,
        final.mean_abs_plane_residual_m, total_ms, final.failure_reason, results,
    )


__all__ = [
    'CausalPersistenceBuild', 'PersistenceWeightedLODBSTIndex',
    'build_causal_persistence', 'expand_selected_nodes_to_leaves',
    'prepare_persistent_surfel_pyramid', 'scan_to_persistent_point_to_plane_icp_3dof',
    'scan_to_map_coarse_to_fine_persistent_3dof',
]
