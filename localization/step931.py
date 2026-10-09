from __future__ import annotations

from dataclasses import dataclass, replace
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
from localization.step93 import _sorted_membership, expand_selected_nodes_to_leaves, _aggregate_persistence

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass
class PreparedHistoryRobustSurfelLevel:
    config: object
    centroids: np.ndarray
    normals: np.ndarray
    valid: np.ndarray
    counts: np.ndarray
    tree: object
    instability: np.ndarray


@dataclass
class CausalHistoryPatternBuild:
    """Per-selected-surfel causal transition-instability scores.

    ``instability_q8[t]`` contains one uint8 score per frozen K=4 selected
    surfel for localizing session index ``t``.  The score is the voxel-support
    mean of the fraction of adjacent *prior-session* occupancy states that
    changed.  The query session and all future sessions are excluded.

    Examples for prior histories (map session is always occupied):
      1111 -> 0 transitions / 3 = 0.0
      1110 -> 1 / 3
      1011 -> 2 / 3
      1010 -> 3 / 3 = 1.0

    This is an intermittency descriptor, not an observed-free probability.
    """
    instability_q8: dict[int, np.ndarray]
    stats: list[dict]
    build_ms: float
    expansion_ms: float
    membership_ms: float

    def score_bytes_per_active_session(self) -> int:
        if not self.instability_q8:
            return 0
        return int(next(iter(self.instability_q8.values())).nbytes)


def build_causal_history_patterns(view, prepared, map_session_index: int, query_session_indices) -> CausalHistoryPatternBuild:
    t_all = time.perf_counter_ns()
    map_t = int(map_session_index)
    qidx = sorted(set(int(x) for x in query_session_indices))
    if any(q <= map_t for q in qidx):
        raise ValueError('causal history-pattern query sessions must occur after the map session')
    if any(q >= len(prepared.session_codes) for q in qidx):
        raise ValueError('query session index exceeds prepared history')
    if len(prepared.session_codes) > 8:
        raise ValueError('Step 9.3.1 uint8 causal presence mask supports at most 8 sessions')

    t0 = time.perf_counter_ns()
    leaf_rows, selected_labels = expand_selected_nodes_to_leaves(view)
    expansion_ms = (time.perf_counter_ns() - t0) / 1e6
    nsel = len(view.selected_node_ids)
    support = np.bincount(selected_labels, minlength=nsel).astype(np.float64)
    if not np.array_equal(support.astype(np.int64), view.hierarchy.node_volume(view.selected_node_ids).astype(np.int64)):
        raise AssertionError('history-pattern selected support does not match hierarchy node volume')

    t0 = time.perf_counter_ns()
    map_codes = np.asarray(view.hierarchy.aligned_map.linear_codes, dtype=np.uint64)
    presence = np.zeros(len(map_codes), dtype=np.uint8)
    for s in range(map_t, max(qidx)):
        found = _sorted_membership(map_codes, prepared.session_codes[s])
        presence[found] |= np.uint8(1 << s)
    if not np.all((presence & np.uint8(1 << map_t)) != 0):
        raise AssertionError('map-session leaves must all be occupied in the map session')
    membership_ms = (time.perf_counter_ns() - t0) / 1e6

    scores = {}
    stats = []
    for q in qidx:
        prior_count = q - map_t
        if prior_count <= 1:
            leaf_inst = np.zeros(len(map_codes), dtype=np.float64)
            transition_den = 0
        else:
            transition_den = prior_count - 1
            transitions = np.zeros(len(map_codes), dtype=np.uint8)
            for s in range(map_t, q - 1):
                a = ((presence >> np.uint8(s)) & np.uint8(1)).astype(np.uint8)
                b = ((presence >> np.uint8(s + 1)) & np.uint8(1)).astype(np.uint8)
                transitions += (a != b).astype(np.uint8)
            leaf_inst = transitions.astype(np.float64) / float(transition_den)

        agg = np.bincount(
            selected_labels,
            weights=leaf_inst[leaf_rows],
            minlength=nsel,
        ) / support
        if np.any(~np.isfinite(agg)) or np.any(agg < -1e-12) or np.any(agg > 1.0 + 1e-12):
            raise AssertionError('invalid causal history instability score')
        q8 = np.rint(np.clip(agg, 0.0, 1.0) * 255.0).astype(np.uint8)
        scores[q] = q8
        sf = q8.astype(np.float32) / 255.0
        stats.append({
            'query_session_index': q,
            'prior_session_count': prior_count,
            'transition_intervals': transition_den,
            'map_elements': nsel,
            'mean_instability': float(np.mean(sf)),
            'median_instability': float(np.median(sf)),
            'p90_instability': float(np.quantile(sf, 0.90)),
            'fraction_stable_zero_transition': float(np.mean(q8 == 0)),
            'fraction_high_instability_ge_half': float(np.mean(sf >= 0.5)),
            'score_payload_mib': q8.nbytes / (1024**2),
        })

    first = map_t + 1
    if first in scores and not np.all(scores[first] == 0):
        raise AssertionError('first causal query session must have zero transition instability')

    return CausalHistoryPatternBuild(
        instability_q8=scores,
        stats=stats,
        build_ms=(time.perf_counter_ns() - t_all) / 1e6,
        expansion_ms=expansion_ms,
        membership_ms=membership_ms,
    )


def prepare_history_robust_surfel_pyramid(materialized: MaterializedLODMap, ids, instability_q8, levels, cfg):
    if cKDTree is None:
        raise RuntimeError('Step 9.3.1 requires scipy.spatial.cKDTree') from _SCIPY_IMPORT_ERROR
    ids = np.asarray(ids, dtype=np.int64)
    h_all = np.asarray(instability_q8, dtype=np.uint8)
    if len(h_all) != materialized.map_elements:
        raise ValueError('history instability score length must equal materialized LOD rows')
    c = materialized.centroids[ids]
    cv = materialized.cov6[ids]
    w = materialized.counts[ids]
    h = h_all[ids].astype(np.float32) / 255.0
    out = []
    for lev in levels:
        mu, n, valid, cnt, _ = aggregate_surfels(
            c, cv, w, lev.map_voxel_m,
            cfg.normal_min_neighbors, cfg.normal_max_curvature,
            cfg.normal_min_horizontal_component,
        )
        hh = _aggregate_persistence(c, w, h, lev.map_voxel_m)
        if len(hh) != len(mu):
            raise AssertionError('history-pattern pyramid grouping differs from geometry grouping')
        tree = cKDTree(mu) if len(mu) else None
        out.append(PreparedHistoryRobustSurfelLevel(lev, mu, n, valid, cnt, tree, hh))
    return out


class HistoryRobustLODBSTIndex:
    """Frozen compact K=4 geometry plus one causal uint8 transition-instability vector."""
    def __init__(self, materialized: MaterializedLODMap, bst_map, session_index: int, instability_q8, history_session_count: int):
        self.base = MaterializedLODBSTIndex(materialized, bst_map, session_index)
        self.map = materialized
        self.instability_q8 = np.asarray(instability_q8, dtype=np.uint8)
        if len(self.instability_q8) != materialized.map_elements:
            raise ValueError('instability_q8 length mismatch')
        self.history_session_count = int(history_session_count)
        self.last_stats = {}

    def query_radius_ids(self, cx, cy, radius_m):
        ids = self.base.query_radius_ids(cx, cy, radius_m)
        self.last_stats = dict(self.base.last_stats)
        if len(ids):
            h = self.instability_q8[ids].astype(np.float32) / 255.0
            self.last_stats.update({
                'mean_history_instability': float(np.mean(h)),
                'median_history_instability': float(np.median(h)),
                'stable_history_fraction': float(np.mean(self.instability_q8[ids] == 0)),
                'history_sessions': self.history_session_count,
            })
        else:
            self.last_stats.update({
                'mean_history_instability': float('nan'),
                'median_history_instability': float('nan'),
                'stable_history_fraction': float('nan'),
                'history_sessions': self.history_session_count,
            })
        return ids

    def prepare_pyramid(self, ids, levels, cfg):
        return prepare_history_robust_surfel_pyramid(self.map, ids, self.instability_q8, levels, cfg)

    def nbytes(self):
        return int(self.base.nbytes() + self.instability_q8.nbytes)


def history_robust_weights(residual, instability):
    """Parameter-free residual/history interaction used by Step 9.3.1.

    Stable history (instability=0) always has weight 1.  For unstable history,
    residuals at or below the current correspondence median also retain weight
    1.  Only above-median residuals are reduced:

        q = max(|r| / median(|r|) - 1, 0)
        w = 1 / (1 + instability * q^2)

    Thus history never suppresses a currently good geometric correspondence.
    """
    r = np.abs(np.asarray(residual, dtype=np.float64))
    h = np.clip(np.asarray(instability, dtype=np.float64), 0.0, 1.0)
    if len(r) != len(h):
        raise ValueError('residual/history length mismatch')
    if not len(r):
        return np.empty(0, dtype=np.float64)
    scale = float(np.median(r))
    # Numerical guard only.  If all residuals are exactly zero, no temporal
    # weighting is needed and this preserves an exact unit-weight update.
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.ones(len(r), dtype=np.float64)
    q = np.maximum(r / scale - 1.0, 0.0)
    return 1.0 / (1.0 + h * q * q)


def scan_to_history_robust_point_to_plane_icp_3dof(
    points_body, map_points, normals, normal_valid, instability,
    initial_pose, fixed_z, fixed_roll, fixed_pitch, config=None, map_tree=None,
):
    """Point-to-plane ICP with residual-gated causal history-pattern weighting."""
    if cKDTree is None:
        raise RuntimeError('Step 9.3.1 requires scipy.spatial.cKDTree') from _SCIPY_IMPORT_ERROR
    cfg = config or PointToPlaneICP3DOFConfig()
    pts = np.asarray(points_body, dtype=np.float64)
    mp = np.asarray(map_points, dtype=np.float64)
    ns_all = np.asarray(normals, dtype=np.float64)
    valid_all = np.asarray(normal_valid, dtype=bool)
    hist_all = np.asarray(instability, dtype=np.float64)
    if len(hist_all) != len(mp):
        raise ValueError('history/map length mismatch')
    if len(pts) == 0:
        return PointToPlaneICP3DOFResult(initial_pose, False, 0, 0, float('inf'), 0.0, 'empty_scan')
    if len(mp) < cfg.min_correspondences:
        return PointToPlaneICP3DOFResult(initial_pose, False, 0, 0, float('inf'), 0.0, 'local_map_too_small')
    tree = map_tree if map_tree is not None else cKDTree(mp)
    qbody = level_body_points(pts, fixed_roll, fixed_pitch)
    pose = initial_pose; conv = False; reason = 'max_iterations'; lc = ln = 0; lr = la = float('inf')
    t0 = time.perf_counter_ns()
    for it in range(1, cfg.max_iterations + 1):
        world = transform_leveled(qbody, pose, fixed_z)
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
        ww = history_robust_weights(residual, hist_all[tidx])
        qq = qbody[ids]; c0, s0 = math.cos(pose.yaw), math.sin(pose.yaw)
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
        lr = float(math.sqrt(np.mean(residual * residual)))
        la = float(np.mean(np.abs(residual)))
        if math.hypot(float(delta[0]), float(delta[1])) <= cfg.translation_tolerance_m and abs(math.degrees(float(delta[2]))) <= cfg.yaw_tolerance_deg:
            conv = True; reason = ''; break
    return PointToPlaneICP3DOFResult(
        pose, conv, it if 'it' in locals() else 0, lc, lr,
        (time.perf_counter_ns() - t0) / 1e6, reason, ln, la,
    )


def scan_to_map_coarse_to_fine_history_robust_3dof(points_body, prepared_levels, initial_pose, fixed_z, fixed_roll, fixed_pitch, base_config=None):
    cfg0 = base_config or PointToPlaneICP3DOFConfig()
    # Exact first-session/same-session null path.
    if prepared_levels and all(np.all(p.instability == 0.0) for p in prepared_levels):
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
        res = scan_to_history_robust_point_to_plane_icp_3dof(
            scan, p.centroids, p.normals, p.valid, p.instability,
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
    'CausalHistoryPatternBuild', 'HistoryRobustLODBSTIndex',
    'build_causal_history_patterns', 'prepare_history_robust_surfel_pyramid',
    'history_robust_weights', 'scan_to_history_robust_point_to_plane_icp_3dof',
    'scan_to_map_coarse_to_fine_history_robust_3dof',
]
