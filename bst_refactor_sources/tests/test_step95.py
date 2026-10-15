import math
from types import SimpleNamespace

import numpy as np

from localization.step95 import (
    ScanContextConfig, ScanContextDatabase, make_scan_context, scan_context_align,
    pose_row_to_matrix, pose_errors_6d, scan_context_pose_guess,
    KISSMatcherBackend, SmallGICPBackend, compose_world_query,
)


def _rotz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=float)


def test_scan_context_recovers_yaw_shift():
    cfg=ScanContextConfig(num_rings=8,num_sectors=36,max_radius_m=40,lidar_height_m=2)
    # Deliberately non-symmetric angular/radial/height pattern.
    ref=np.array([
        [10,0,1.0],[15,3,2.5],[20,8,0.2],[-7,22,3.0],[-25,-4,1.5],[5,-31,4.0],[28,-14,0.8]
    ],dtype=float)
    delta=math.radians(30)
    query=ref @ _rotz(-delta).T
    q=make_scan_context(query,cfg); r=make_scan_context(ref,cfg)
    dist,shift,rel=scan_context_align(q,r)
    assert dist < 1e-6
    assert shift == 3
    assert abs(math.degrees(rel)-30.0) < 1e-9


def test_scan_context_database_ranks_matching_place():
    cfg=ScanContextConfig(8,36,40,2)
    a=np.array([[10,0,1],[15,5,2],[-20,2,3],[3,-30,4]],float)
    b=np.array([[5,10,3],[-8,12,1],[25,-2,4],[-5,-28,2]],float)
    da=make_scan_context(a,cfg); db=make_scan_context(b,cfg)
    D=np.stack([da,db])
    database=ScanContextDatabase(D)
    q=a @ _rotz(math.radians(-20)).T
    m=database.query_points(q,cfg,top_k=1,candidate_pool=2)
    assert len(m)==1 and m[0].reference_index==0
    assert m[0].distance < 1e-6


def test_pose_composition_and_errors():
    ref=np.array([0,10,20,3,0,0,math.radians(40)],float)
    T_ref_query=np.eye(4); T_ref_query[:3,:3]=_rotz(math.radians(10)); T_ref_query[:3,3]=[1,2,.5]
    T=compose_world_query(ref,T_ref_query)
    expected=pose_row_to_matrix(ref) @ T_ref_query
    assert np.allclose(T,expected)
    e=pose_errors_6d(T,T)
    assert e['translation_3d_m'] < 1e-12 and e['rotation_deg'] < 1e-5


def test_scan_context_pose_guess_adds_relative_yaw():
    ref=np.array([0,10,20,3,0.1,-0.05,math.radians(40)],float)
    T=scan_context_pose_guess(ref,math.radians(30))
    yaw=math.degrees(math.atan2(T[1,0],T[0,0]))
    assert abs(yaw-70.0)<1e-9
    assert np.allclose(T[:3,3],[10,20,3])


class _FakeKISSConfig:
    def __init__(self,v): self.v=v
class _FakeKISSMatcher:
    def __init__(self,cfg): self.cfg=cfg
    def estimate(self,src,tgt):
        return SimpleNamespace(rotation=np.eye(3),translation=np.array([1.,2.,3.]))
    def get_num_rotation_inliers(self): return 17
    def get_num_final_inliers(self): return 13
class _FakeKISSModule:
    KISSMatcherConfig=_FakeKISSConfig
    KISSMatcher=_FakeKISSMatcher


def test_kiss_adapter_contract():
    b=KISSMatcherBackend(.3,module=_FakeKISSModule)
    pts=np.arange(90,dtype=float).reshape(30,3)
    r=b.estimate(pts,pts)
    assert r.converged and r.inliers==13 and r.rotation_inliers==17
    assert np.allclose(r.T_target_source[:3,3],[1,2,3])


class _FakeSG:
    @staticmethod
    def align(target,source,**kwargs):
        T=np.asarray(kwargs['init_T_target_source']).copy(); T[0,3]+=0.25
        return SimpleNamespace(T_target_source=T,converged=True,iterations=5,num_inliers=123,e=.4)


def test_small_gicp_adapter_contract():
    b=SmallGICPBackend(module=_FakeSG)
    pts=np.arange(90,dtype=float).reshape(30,3)
    r=b.align(pts,pts,np.eye(4))
    assert r.converged and r.inliers==123 and r.iterations==5
    assert abs(r.T_target_source[0,3]-.25)<1e-12


def test_dependency_resolver_does_not_import_kiss_in_parent(monkeypatch):
    import localization.step95 as s95
    sentinel=object()
    monkeypatch.setattr(s95.importlib.util,'find_spec',lambda name: sentinel if name=='kiss_matcher' else None)
    calls=[]
    def fake_import(name):
        calls.append(name)
        if name=='kiss_matcher':
            raise AssertionError('kiss_matcher must not be imported in parent process')
        if name=='small_gicp':
            return sentinel
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(s95.importlib,'import_module',fake_import)
    deps=s95.require_frontend_dependencies()
    assert deps['kiss_matcher'] is None
    assert deps['small_gicp'] is sentinel
    assert calls == ['small_gicp']
