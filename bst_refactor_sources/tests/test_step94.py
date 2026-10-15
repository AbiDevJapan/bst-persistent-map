import math

from localization.step94 import causal_latest_prior_plan, historical_catalog_totals
from benchmark.step94_benchmark import (
    STATIC, LATEST_VOXEL, BST_LATEST, MAIN_PERTURBATION,
    _paired_compare, _historical_gates,
)


def test_causal_latest_prior_plan_exact():
    h=['2012-01-15','2012-02-02','2012-03-17','2012-05-11','2012-06-15']
    q=h[1:]
    p=causal_latest_prior_plan(h,h[0],q)
    assert [x.query_session for x in p] == q
    assert [x.latest_prior_session for x in p] == h[:-1]
    assert [x.latest_prior_index for x in p] == [0,1,2,3]
    assert all(x.latest_prior_index < x.query_index for x in p)


def test_causal_latest_prior_rejects_nonchronological_or_past():
    h=['a','b','c']
    try:
        causal_latest_prior_plan(h,'a',['c','b'])
        assert False
    except ValueError:
        pass
    try:
        causal_latest_prior_plan(h,'b',['a'])
        assert False
    except ValueError:
        pass


def test_historical_catalog_counts_bst_once():
    s={'jan':100,'feb':90,'mar':80,'may':70}
    l={'jan':50,'feb':45,'mar':40,'may':35}
    x=historical_catalog_totals(s,l,60)
    assert x['independent_snapshot_catalog_bytes']==340
    assert x['compact_lod_overlay_catalog_bytes']==170
    assert x['bst_history_bytes']==60
    assert x['bst_historical_catalog_bytes']==230
    assert math.isclose(x['catalog_storage_reduction_vs_snapshots'],1-230/340)


def _row(method, session, scan, success, total, prim, perturb=MAIN_PERTURBATION):
    return {
        'map_geometry':method,'scenario':'cross_session','query_session':session,
        'original_scan_index':scan,'perturbation':perturb,'success':success,
        'total_ms':float(total),'local_map_primitives':float(prim),
    }


def test_paired_compare_counts_discordant_trials():
    rows=[]
    for method, vals in [(STATIC,[1,1,0,0]),(LATEST_VOXEL,[1,0,1,0])]:
        for i,v in enumerate(vals): rows.append(_row(method,'mar',i,bool(v),100,100))
    out=_paired_compare(rows,STATIC,LATEST_VOXEL,label_a='static',label_b='latest_prior',first_query_session='feb')
    r=next(x for x in out if x['query_session']=='mar')
    assert r['both_success']==1
    assert r['static_only_success']==1
    assert r['latest_prior_only_success']==1
    assert r['neither_success']==1
    assert r['net_latest_prior_success_difference']==0
    assert r['exact_mcnemar_p']==1.0


def test_historical_gate_separates_temporal_gain_and_bst_representation():
    rows=[]
    sessions=['feb','mar','may','jun']
    # first session is excluded from the later-session gate. Later static 50%, latest 60%, BST 55%.
    for s in sessions:
        for i in range(20):
            if s=='feb':
                ss=lv=bst=(i<10)
            else:
                ss=i<10; lv=i<12; bst=i<11
            rows += [
                _row(STATIC,s,i,ss,400,100),
                _row(LATEST_VOXEL,s,i,lv,400,100),
                _row(BST_LATEST,s,i,bst,250,50),
            ]
    storage={'catalog_storage_reduction_vs_snapshots':0.30}
    g=_historical_gates(rows,[],storage,'feb')[0]
    assert math.isclose(g['static_success_rate'],0.5)
    assert math.isclose(g['latest_prior_voxel_success_rate'],0.6)
    assert g['strict_temporal_selection_gain_observed']
    assert g['bst_success_ratio_to_latest_prior_voxel'] > 0.90
    assert math.isclose(g['bst_primitive_reduction_vs_latest_prior_voxel'],0.5)
    assert math.isclose(g['bst_median_total_speedup_vs_latest_prior_voxel'],1.6)
    assert g['bst_representation_gate_pass']
    assert g['historical_storage_gate_pass']


def test_checkpoint_csv_typed_roundtrip(tmp_path):
    from benchmark.step94_benchmark import _read_checkpoint_csv
    p=tmp_path/'cp.csv'
    p.write_text('success,original_scan_index,total_ms,map_geometry,empty\nTrue,17,123.5,static_jan_voxel,\nFalse,18,nan,bst_latest_prior_k4,\n')
    rows=_read_checkpoint_csv(p)
    assert rows[0]['success'] is True
    assert rows[1]['success'] is False
    assert rows[0]['original_scan_index'] == 17
    assert isinstance(rows[0]['original_scan_index'], int)
    assert rows[0]['total_ms'] == 123.5
    assert math.isnan(rows[1]['total_ms'])
    assert rows[0]['map_geometry'] == 'static_jan_voxel'
    assert rows[0]['empty'] == ''
