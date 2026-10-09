from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from benchmark.step92_benchmark import MAIN_PERTURBATION
from benchmark.step913_benchmark import FROZEN_PROFILE, FROZEN_K
from localization.step901 import PointToPlaneICP3DOFConfig
from localization.step902 import DEFAULT_PYRAMID


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'missing Step 9.2 output: {path}')
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def _num(v):
    if v is None or v == '':
        return None
    s = str(v)
    if s.lower() in {'true','false'}:
        return s.lower() == 'true'
    try:
        if any(c in s.lower() for c in ('.','e','nan','inf')):
            return float(s)
        return int(s)
    except ValueError:
        return v


def _typed_rows(rows: list[dict]) -> list[dict]:
    return [{k: _num(v) for k, v in r.items()} for r in rows]


def recover_step92_metadata(
    output_dir,
    *,
    history_sessions=('2012-01-15','2012-02-02','2012-03-17','2012-05-11','2012-06-15'),
    map_session='2012-01-15',
    query_sessions=('2012-02-02','2012-03-17','2012-05-11','2012-06-15'),
    query_count_per_session=50,
    same_session_query_count=50,
    excluded_tuning_query_scans_csv='results/step9_1_2_hierarchical_lod/query_scans.csv',
    perturbations='0.5:1',
    perturb_seed=42090,
    localization_max_range_m=30.0,
    scan_downsample_m=0.75,
    max_scan_points=8000,
    local_map_radius_m=40.0,
    success_translation_m=0.5,
    success_yaw_deg=1.0,
    icp_config=None,
):
    """Recover metadata after the original Step-9.2 late reporting crash.

    The failed 9.2 runner writes all scientific CSV outputs before metadata.json.
    This function intentionally does not invent unavailable construction timings.
    """
    out = Path(output_dir)
    session_summary = _typed_rows(_read_csv(out/'session_summary.csv'))
    pooled = _typed_rows(_read_csv(out/'pooled_summary.csv'))
    paired = _typed_rows(_read_csv(out/'paired_outcomes.csv'))
    gates = _typed_rows(_read_csv(out/'validation_gates.csv'))
    memory = _typed_rows(_read_csv(out/'deployment_memory.csv'))
    verification_rows = _typed_rows(_read_csv(out/'materialization_verification.csv'))
    query_rows = _typed_rows(_read_csv(out/'query_scans.csv'))

    base_cfg = icp_config or PointToPlaneICP3DOFConfig()
    cross = [r for r in query_rows if r.get('scenario') == 'cross_session']
    same = [r for r in query_rows if r.get('scenario') == 'same_session_control']
    excluded_count = sum(bool(r.get('excluded_by_tuning_csv', False)) for r in cross)

    metadata = {
        'step': '9.2',
        'purpose': 'Frozen multi-session held-out localization validation of compact balanced K=4 BST LOD versus 0.5m voxel surfels.',
        'recovered_after_late_reporting_crash': True,
        'recovery_note': 'All scientific CSVs were already written before the original metadata serialization crash. Construction timings that existed only in process memory are left unavailable rather than guessed.',
        'history_sessions': list(history_sessions),
        'map_session': map_session,
        'query_sessions': list(query_sessions),
        'query_count_per_session': int(query_count_per_session),
        'total_cross_session_scans': len(cross),
        'same_session_query_count': len(same),
        'excluded_tuning_query_scans_csv': excluded_tuning_query_scans_csv,
        'excluded_cross_session_scan_names_observed': int(excluded_count),
        'frozen_geometry_profile': asdict(FROZEN_PROFILE),
        'frozen_k': int(FROZEN_K),
        'main_validation_perturbation': MAIN_PERTURBATION,
        'perturbations_cli': perturbations,
        'perturb_seed': int(perturb_seed),
        'localization_max_range_m': float(localization_max_range_m),
        'scan_downsample_m': float(scan_downsample_m),
        'max_scan_points': int(max_scan_points),
        'local_map_radius_m': float(local_map_radius_m),
        'success_thresholds': {'translation_m': float(success_translation_m), 'yaw_deg': float(success_yaw_deg)},
        'base_icp_config': asdict(base_cfg),
        'pyramid': [asdict(x) for x in DEFAULT_PYRAMID],
        'methods': ['voxel_surfel_0.5m', 'bst_lod_k4_compact'],
        'map_built_once_for_all_query_sessions': True,
        'construction_only': {
            'session_block_build_ms': None,
            'session_block_build_ms_per_session': None,
            'adaptive_core_build_ms': None,
            'voxel_index_build_ms': None,
            'hierarchy_build_ms': None,
            'lod_decision_ms': None,
            'materialize_ms': next((r.get('materialize_ms') for r in memory if r.get('map_geometry') == 'bst_lod_k4_compact'), None),
            'timings_unavailable_due_to_original_late_reporting_crash': True,
        },
        'materialization_verification': verification_rows[0] if verification_rows else {},
        'deployment_memory': memory,
        'validation_gates': gates,
        'paired_outcomes': paired,
        'session_summary': session_summary,
        'pooled_summary': pooled,
        'runtime_recovery': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine()},
        'scientific_caveat': 'Recovered metadata only; scientific CSV results are original run outputs. Frozen balanced gate and K=4 remain unchanged. PrototypeBSTSpatialCodec remains a local research codec, not verified upstream BST-ID.',
    }
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=True))
    return metadata


__all__ = ['recover_step92_metadata']
