#!/usr/bin/env python3
"""Compare refactored Step 9.6 benchmark output against the frozen pre-refactor run.

No third-party dependencies. Runtime metrics are diagnostic rather than equality gates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

IMMUTABLE_METADATA = (
    'session_names', 'origin_local_m', 'config', 'world',
    'history_sessions', 'query_sessions', 'causal_latest_prior_plan',
    'query_count_per_session', 'no_pose_initialization', 'frozen_k',
    'frozen_geometry_profile', 'scan_context_fallback_config',
    'bevplace_retrieval_top_k', 'reference_keyframes', 'global_registration',
    'local_refinement', 'success_thresholds', 'retriever_provenance',
)
KEY = ('query_session', 'scan_file', 'original_scan_index')
METRICS = ('success', 'strict_6d_success', 'global_2m5deg_success')


def read_csv(folder, filename):
    with (folder / filename).open(newline='') as f:
        return list(csv.DictReader(f))


def classify(method):
    method = method.lower()
    if 'bst' in method: return 'bst_k4'
    if 'voxel' in method: return 'voxel'
    raise ValueError('Unknown map_geometry class: ' + method)


def indexed(rows, key):
    d = {}
    for r in rows:
        k = tuple(r[c] for c in key)
        if k in d: raise ValueError(f'Duplicate key: {k}')
        d[k] = r
    return d


def close(a, b, atol=1e-5):
    try: return abs(float(a) - float(b)) <= atol
    except (TypeError, ValueError): return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path(__file__).resolve().parent / 'regression' / 'golden_step96')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--max-prediction-differences', type=int, default=0,
                        help='Explicit tolerance for nondeterministic registration outcomes')
    parser.add_argument('--max-candidate-differences', type=int, default=0,
                        help='Explicit tolerance for chosen reference identities')
    a = parser.parse_args()
    base, new = a.baseline, a.candidate
    failures = []
    observations = []
    def assert_check(ok, message):
        if not ok: failures.append(message)

    meta_base, meta_new = (json.loads((d/'metadata.json').read_text()) for d in (base,new))
    for key in IMMUTABLE_METADATA:
        assert_check(meta_base.get(key) == meta_new.get(key), f'Metadata mismatch: {key}')

    bq = indexed(read_csv(base,'query_scans.csv'), KEY)
    nq = indexed(read_csv(new,'query_scans.csv'), KEY)
    assert_check(bq.keys() == nq.keys(), 'Query set/scan filenames differ')
    for k in bq.keys() & nq.keys():
        for field in ('historical_map_session','timestamp_us','pose_gap_us'):
            assert_check(bq[k].get(field) == nq[k].get(field), f'Query {k} field {field} changed')

    bm = indexed(read_csv(base,'map_geometry_stats.csv'),('historical_map_session',))
    nm = indexed(read_csv(new,'map_geometry_stats.csv'),('historical_map_session',))
    assert_check(bm.keys() == nm.keys(), 'Map geometry session set differs')
    for k in bm.keys() & nm.keys():
        for field in ('voxel_elements', 'bst_k4_elements','voxel_geometry_mib',
                      'bst_k4_geometry_mib','bst_k4_directory_mib'):
            assert_check(close(bm[k][field],nm[k][field]),f'Map geometry changed: {k} {field}')

    br = indexed(read_csv(base,'reference_database_stats.csv'),('historical_map_session',))
    nr = indexed(read_csv(new,'reference_database_stats.csv'),('historical_map_session',))
    assert_check(br.keys()==nr.keys(), 'Reference database session set differs')
    for k in br.keys() & nr.keys():
        for field in ('keyframes','descriptor_bytes','bevplace_descriptor_bytes'):
            assert_check(br[k][field]==nr[k][field], f'Reference DB changed: {k} {field}')

    bf = indexed(read_csv(base,'frontend_trials.csv'), KEY)
    nf = indexed(read_csv(new,'frontend_trials.csv'), KEY)
    assert_check(bf.keys()==nf.keys(), 'Frontend query identity differs')
    candidate_changes = []
    for k in bf.keys() & nf.keys():
        for field in ('bevplace_topk','bevplace_topk_best_position_error_m',
                      'bevplace_top1_position_error_m'):
            assert_check(close(bf[k][field],nf[k][field]),f'Retrieval changed: {k} {field}')
        if bf[k]['selected_reference_scan_index']!=nf[k]['selected_reference_scan_index']:
            candidate_changes.append(k)
    assert_check(len(candidate_changes)<=a.max_candidate_differences,
                 f'Selected reference changed for {len(candidate_changes)} queries; examples {candidate_changes[:5]}')
    observations.append(f'Selected-reference changes: {len(candidate_changes)}/200')

    def trial_key(row): return tuple(row[c] for c in KEY)+(classify(row['map_geometry']),)
    bt = {trial_key(r):r for r in read_csv(base,'localization_trials.csv')}
    nt = {trial_key(r):r for r in read_csv(new,'localization_trials.csv')}
    assert_check(bt.keys()==nt.keys(), 'Map/query trial set differs')
    pred_changes = []
    for k in bt.keys() & nt.keys():
        if any(bt[k][field]!=nt[k][field] for field in METRICS): pred_changes.append(k)
        assert_check(bt[k]['local_map_primitives']==nt[k]['local_map_primitives'],
                     f'Primitive count changed: {k}')
    assert_check(len(pred_changes)<=a.max_prediction_differences,
                 f'Localization prediction changed for {len(pred_changes)} trials; examples {pred_changes[:5]}')
    observations.append(f'Prediction changes: {len(pred_changes)}/400')

    bsum = {classify(r['map_geometry']):r for r in read_csv(base,'localization_summary.csv')}
    nsum = {classify(r['map_geometry']):r for r in read_csv(new,'localization_summary.csv')}
    assert_check(bsum.keys()==nsum.keys(), 'Summary map classes differ')
    for method in bsum.keys() & nsum.keys():
        for field in ('fine_success_rate','strict_6d_success_rate','global_2m5deg_success_rate',
                      'mean_local_map_primitives'):
            observations.append(f'{method} {field}: baseline={bsum[method][field]}, candidate={nsum[method][field]}')
        for field in ('median_frontend_ms','median_local_gicp_ms','median_map_retrieval_ms','median_total_ms'):
            observations.append(f'{method} {field}: baseline={bsum[method][field]}, candidate={nsum[method][field]} (timing not an equality gate)')

    print('\n'.join(observations))
    if failures:
        print(f'\nFAILED: {len(failures)} issues:')
        for message in failures[:25]:print(' - '+message)
        if len(failures)>25:print(f' ... and {len(failures)-25} more')
        return 1
    print('\nPASS: query/reference/map invariants and frozen trial predictions match.')
    print('NOTE: timing and offline neural preprocessing are NOT included in equality gates.')
    return 0

if __name__=='__main__': raise SystemExit(main())
