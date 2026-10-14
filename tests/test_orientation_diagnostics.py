"""Step 9.7A must be observational and preserve the golden Step 9.6 scores."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pytest

from benchmark.orientation_diagnostics import (
    analyze, make_pose_audit, signed_yaw_error_deg, yaw_update_deg, yaw_landscape
)


def T(yaw_deg=0, xy=(0,0)):
    a=math.radians(yaw_deg)
    M=np.eye(4)
    M[:2,:2]=[[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]
    M[:2,3]=xy
    return M


def test_signed_yaw_wraparound():
    assert signed_yaw_error_deg(T(-179),T(179))==pytest.approx(2)
    assert signed_yaw_error_deg(T(179),T(-179))==pytest.approx(-2)
    assert yaw_update_deg(T(179),T(-179))==pytest.approx(2)
    assert yaw_update_deg(T(-179),T(179))==pytest.approx(-2)


def test_audit_does_not_accept_unconverged_result():
    gicp=SimpleNamespace(converged=False,error=float('nan'),iterations=5,inliers=3,failure_reason='test')
    A=T(2, (1,2))
    r=make_pose_audit(map_geometry='bst',query_session='feb',scan_file='abc.bin',original_scan_index=33,
                      T_gt=T(1),T_front=A,T_final=A,gicp_result=gicp,
                      target_points=np.zeros((0,3)),source_points=np.zeros((0,3)))
    assert r['signed_frontend_yaw_error_deg']==pytest.approx(1)
    assert r['signed_final_yaw_error_deg']==pytest.approx(1)
    assert r['signed_map_yaw_update_deg']==0
    assert r['map_translation_update_3d_m']==0
    assert r['nn_overlap_fraction'] is None


def test_yaw_cost_proxy_center_is_local_minimum():
    rng=np.random.default_rng(7)
    p=rng.uniform(-15,15,size=(400,3))
    q=yaw_landscape(T(0),p,p,delta_deg=.5)
    assert q['nn_overlap_fraction']==pytest.approx(1)
    assert q['nn_rmse_inliers_m']==pytest.approx(0, abs=1e-12)
    assert q['yaw_proxy_curvature_m2_per_rad2']>0
    assert q['yaw_proxy_preferred_offset_deg']==0


def test_analyze_frozen_golden(tmp_path):
    base=Path(__file__).resolve().parents[1]/'regression/golden_step96'
    out=tmp_path/'out'
    result=analyze(base,out)
    bst=next(x for x in result if x['map_geometry'].endswith('_bst_k4') and x['query_session']=='ALL')
    vox=next(x for x in result if x['map_geometry'].endswith('_voxel') and x['query_session']=='ALL')
    assert (bst['queries'],bst['frontend_fine'],bst['final_fine'],bst['final_xy'],bst['final_yaw'])==(200,96,95,158,101)
    assert (bst['yaw_only_failures'],bst['front_fine_lost'],bst['front_not_fine_recovered'])==(63,12,11)
    assert (vox['final_fine'],vox['final_xy'],vox['final_yaw'])==(103,162,104)
    assert json.loads((out/'orientation_metadata.json').read_text())['has_signed_yaw_audit'] is False
    assert len(list(csv.DictReader((out/'orientation_trials.csv').open())))==400


def test_analyze_bad_flag_is_rejected(tmp_path):
    base=Path(__file__).resolve().parents[1]/'regression/golden_step96'
    import shutil
    src=tmp_path/'src';src.mkdir()
    shutil.copy(base/'frontend_trials.csv',src/'frontend_trials.csv')
    loc=list(csv.DictReader((base/'localization_trials.csv').open()))
    loc[0]['success']='False' if loc[0]['success']=='True' else 'True'
    with (src/'localization_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(loc[0]));w.writeheader();w.writerows(loc)
    with pytest.raises(ValueError,match='inconsistent'):
        analyze(src,tmp_path/'out')


def test_source_has_no_tuning():
    src=(Path(__file__).resolve().parents[1]/"benchmark/localization_benchmark.py").read_text()
    assert 'orientation_diagnostics=False' in src
    assert 'T_final=lres.T_target_source if lres.converged else T_front' in src
    assert 'if orientation_diagnostics:' in src


def test_analyze_with_signed_audit(tmp_path):
    # Synthetic instrumentation reproduces old absolute errors while respecting
    # the frozen non-converged frontend fallback identity. This tests CSV joins.
    base=Path(__file__).resolve().parents[1]/'regression/golden_step96'
    import shutil
    src=tmp_path/'src';src.mkdir()
    for filename in ('frontend_trials.csv','localization_trials.csv'):
        shutil.copy(base/filename,src/filename)
    loc=list(csv.DictReader((src/'localization_trials.csv').open()))
    audits=[]
    for row in loc:
        a=float(row['frontend_yaw_error_deg'])
        b=float(row['final_yaw_error_deg'])
        audits.append(dict(
            map_geometry=row['map_geometry'],query_session=row['query_session'],
            scan_file=row['scan_file'],original_scan_index=row['original_scan_index'],
            signed_frontend_yaw_error_deg=a,signed_final_yaw_error_deg=b,
            signed_map_yaw_update_deg=b-a,
            map_translation_update_3d_m=0 if row['local_gicp_converged']=='False' else 0.1,
            map_translation_update_xy_m=0 if row['local_gicp_converged']=='False' else 0.1,
            nn_overlap_fraction=0.7,yaw_proxy_curvature_m2_per_rad2=2.0))
    with (src/'orientation_pose_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audits[0]));w.writeheader();w.writerows(audits)
    summary=analyze(src,tmp_path/'out')
    assert len(summary)==10
    meta=json.loads((tmp_path/'out/orientation_metadata.json').read_text())
    assert meta['has_signed_yaw_audit'] is True
    assert (tmp_path/'out/orientation_quality.csv').exists()
    assert len(list(csv.DictReader((tmp_path/'out/orientation_trials.csv').open())))==400
