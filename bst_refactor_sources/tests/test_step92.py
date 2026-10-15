from __future__ import annotations

import math
import numpy as np

from benchmark.step913_benchmark import _finite_pose_mask
from benchmark.step92_benchmark import (
    _exact_mcnemar_p,
    _paired_outcomes,
    _pooled_summary,
    _session_summary,
    _validation_gates,
    _session_block_timing_metadata,
)


def _row(geom, session, success, primitives, total, *, scan=0, perturb='xy0.5_yaw1'):
    return {
        'map_geometry': geom,
        'scenario': 'cross_session',
        'query_session': session,
        'perturbation': perturb,
        'success': success,
        'converged': True,
        'original_scan_index': scan,
        'final_translation_error_m': 0.2 if success else 0.7,
        'final_yaw_error_deg': 0.8 if success else 1.2,
        'retrieval_ms': 5.0,
        'map_prep_ms': total-15.0,
        'match_ms': 10.0,
        'total_ms': total,
        'query_points': 3000,
        'local_map_primitives': primitives,
        'correspondences': 2800,
        'normal_correspondences': 1200,
        'final_plane_rmse_m': 0.1,
        'mean_abs_plane_residual_m': 0.08,
        'iterations': 15,
    }


def test_exact_mcnemar_balanced_and_extreme():
    assert _exact_mcnemar_p(0, 0) == 1.0
    assert _exact_mcnemar_p(3, 3) == 1.0
    assert 0.0 < _exact_mcnemar_p(6, 0) < 0.05


def test_multisession_paired_and_gate():
    rows=[]
    # Four sessions x five paired trials. Voxel succeeds 12/20, BST 11/20.
    sessions=['s1','s2','s3','s4']
    vpat=[1,1,1,0,0]
    bpat=[1,1,0,1,0]
    for s in sessions:
        for i,(v,b) in enumerate(zip(vpat,bpat)):
            rows.append(_row('voxel_surfel_0.5m',s,bool(v),100000,400,scan=i))
            rows.append(_row('bst_lod_k4_compact',s,bool(b),50000,280,scan=i))
    ss=_session_summary(rows)
    assert len(ss)==8
    pooled=_pooled_summary(rows)
    by={r['map_geometry']:r for r in pooled}
    assert by['voxel_surfel_0.5m']['trials']==20
    assert math.isclose(by['bst_lod_k4_compact']['mean_local_map_primitives'],50000)
    paired=_paired_outcomes(rows)
    pall=[r for r in paired if r['query_session']=='ALL'][0]
    assert pall['paired_trials']==20
    assert pall['voxel_only_success']==4
    assert pall['bst_only_success']==4
    gate=_validation_gates(pooled)[0]
    assert gate['meets_success_retention_0.90']
    assert gate['meets_primitive_reduction_0.40']
    assert gate['meets_speedup_1.20']
    assert gate['overall_gate_pass']


def test_gate_can_fail_without_tuning_side_effect():
    rows=[]
    for i in range(10):
        rows.append(_row('voxel_surfel_0.5m','s',True,100000,400,scan=i))
        rows.append(_row('bst_lod_k4_compact','s',i<7,70000,380,scan=i))
    gate=_validation_gates(_pooled_summary(rows))[0]
    assert not gate['meets_success_retention_0.90']
    assert not gate['meets_primitive_reduction_0.40']
    assert not gate['meets_speedup_1.20']
    assert not gate['overall_gate_pass']


def test_nonfinite_ground_truth_rows_are_rejected_by_query_selection_mask():
    poses=np.array([
        [1,0,0,0,0,0,0],
        [2,np.nan,0,0,0,0,0],
        [3,0,0,0,0,np.inf,0],
    ],dtype=float)
    assert _finite_pose_mask(poses).tolist()==[True,False,False]


def test_session_block_timing_list_serializes_as_total_and_per_session():
    m=_session_block_timing_metadata([8.67,6.29,5.81,6.53,4.69])
    assert len(m['session_block_build_ms_per_session'])==5
    assert math.isclose(m['session_block_build_ms'], sum(m['session_block_build_ms_per_session']))
    assert all(isinstance(x,float) for x in m['session_block_build_ms_per_session'])


def test_recover_existing_step92_outputs_without_rebuild(tmp_path):
    import csv, json
    from benchmark.step92_recover import recover_step92_metadata

    def write(name, fields, rows):
        with (tmp_path/name).open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

    write('session_summary.csv',['map_geometry','scenario','query_session','perturbation'],[
        {'map_geometry':'voxel_surfel_0.5m','scenario':'cross_session','query_session':'s1','perturbation':'xy0.5_yaw1'}])
    write('pooled_summary.csv',['map_geometry','scenario','perturbation','trials','success_rate'],[
        {'map_geometry':'voxel_surfel_0.5m','scenario':'cross_session_pooled','perturbation':'xy0.5_yaw1','trials':1,'success_rate':1.0}])
    write('paired_outcomes.csv',['query_session','perturbation','paired_trials'],[
        {'query_session':'ALL','perturbation':'xy0.5_yaw1','paired_trials':1}])
    write('validation_gates.csv',['perturbation','overall_gate_pass'],[
        {'perturbation':'xy0.5_yaw1','overall_gate_pass':True}])
    write('deployment_memory.csv',['map_geometry','materialize_ms'],[
        {'map_geometry':'bst_lod_k4_compact','materialize_ms':442.0}])
    write('materialization_verification.csv',['payload_exact','query_exact'],[
        {'payload_exact':True,'query_exact':True}])
    write('query_scans.csv',['scenario','excluded_by_tuning_csv'],[
        {'scenario':'cross_session','excluded_by_tuning_csv':False},
        {'scenario':'same_session_control','excluded_by_tuning_csv':False},
    ])
    meta=recover_step92_metadata(tmp_path, history_sessions=['m','s1'], map_session='m', query_sessions=['s1'], query_count_per_session=1, same_session_query_count=1)
    assert meta['recovered_after_late_reporting_crash'] is True
    assert meta['construction_only']['session_block_build_ms'] is None
    assert meta['construction_only']['materialize_ms'] == 442.0
    assert (tmp_path/'metadata.json').exists()
    loaded=json.loads((tmp_path/'metadata.json').read_text())
    assert loaded['frozen_k']==4
