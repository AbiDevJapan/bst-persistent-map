from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HistoricalSelection:
    query_session: str
    query_index: int
    latest_prior_session: str
    latest_prior_index: int


def causal_latest_prior_plan(history_sessions, map_session: str, query_sessions):
    """Return the immediately preceding available session for each query session.

    The plan is causal by construction: a query at history index q can use only
    q-1 or earlier.  Step 9.4 deliberately chooses q-1 (the most recent
    available prior session), not the globally best historical session.
    """
    names = list(history_sessions)
    if map_session not in names:
        raise ValueError('map_session must appear in history_sessions')
    map_i = names.index(map_session)
    out = []
    last_q = -1
    for s in query_sessions:
        if s not in names:
            raise ValueError(f'query session not in history: {s}')
        q = names.index(s)
        if q <= map_i:
            raise ValueError('Step 9.4 query sessions must occur after map_session')
        if q <= last_q:
            raise ValueError('query_sessions must be chronological and unique')
        prior = q - 1
        if prior < map_i:
            raise AssertionError('latest prior session predates the base map')
        out.append(HistoricalSelection(s, q, names[prior], prior))
        last_q = q
    if not out:
        raise ValueError('at least one query session is required')
    return out


def historical_catalog_totals(snapshot_bytes_by_session, lod_bytes_by_session, bst_bytes: int):
    """Count deployed historical-map catalog storage without double-counting BST.

    `snapshot_bytes_by_session` contains one independent voxel-surfel localization
    map per causal prior session.  `lod_bytes_by_session` contains one compact
    K=4 localization overlay per prior session.  The BST history representation
    is counted exactly once.
    """
    snap = int(sum(int(v) for v in snapshot_bytes_by_session.values()))
    overlays = int(sum(int(v) for v in lod_bytes_by_session.values()))
    bst = int(bst_bytes)
    total = bst + overlays
    reduction = 1.0 - (total / snap) if snap > 0 else float('nan')
    return {
        'independent_snapshot_catalog_bytes': snap,
        'bst_history_bytes': bst,
        'compact_lod_overlay_catalog_bytes': overlays,
        'bst_historical_catalog_bytes': total,
        'catalog_storage_reduction_vs_snapshots': float(reduction),
    }


__all__ = ['HistoricalSelection', 'causal_latest_prior_plan', 'historical_catalog_totals']
