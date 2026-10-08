import math
from types import SimpleNamespace
import numpy as np

from realdata.nclt_packed import pack_signed_xyz, linear_encode
from realdata.nclt_surfel import _reduce_scan_into_stats, _finalize_geometry
from localization.step903b import aggregate_surfels, SurfelLocalMapIndex, scan_to_surfel_point_to_plane_icp_3dof
from localization.step90 import Pose3DOF
from localization.step901 import PointToPlaneICP3DOFConfig
from synthetic.world import WorldSpec


def test_subvoxel_statistics_recover_plane_centroid_and_normal():
    vs=0.5
    # Points lie in one voxel on plane x=0.10 with variation in y/z.
    y=np.linspace(0.05,0.45,8); z=np.linspace(0.04,0.44,8)
    pts=np.array([[0.10, yy, zz] for yy in y for zz in z],dtype=float)
    code=pack_signed_xyz(np.array([0]),np.array([0]),np.array([0]))
    counts=np.zeros(1,dtype=np.uint32); s1=np.zeros((1,3),np.float32); s2=np.zeros((1,6),np.float32)
    _reduce_scan_into_stats(pts,vs,code,counts,s1,s2)
    cen,cov,norm,curv,valid=_finalize_geometry(code,vs,counts,s1,s2,6,0.2,0.2)
    assert counts[0]==len(pts)
    assert abs(float(cen[0,0])-0.10)<1e-4
    assert valid[0]==1
    assert abs(float(norm[0,0]))>0.95
    assert float(curv[0])<1e-4


def test_weighted_surfel_aggregation_preserves_plane():
    mu=np.array([[0.10,0.1,0.1],[0.10,0.4,0.4],[0.10,0.7,0.7]],float)
    cov=np.tile(np.array([1e-4,0,0,0.02,0,0.02]),(3,1))
    cnt=np.array([20,30,40],np.uint32)
    M,n,v,w,c6=aggregate_surfels(mu,cov,cnt,1.0,6,0.2,0.2)
    assert len(M)==1 and w[0]==90 and v[0]
    assert abs(M[0,0]-0.10)<1e-9
    assert abs(n[0,0])>0.95


def test_surfel_local_index_returns_nearby_rows():
    spec=WorldSpec(10,10,2,0.5)
    xyz=np.array([[2,2,0],[3,3,0],[15,15,0]],int)
    codes=linear_encode(xyz[:,0],xyz[:,1],xyz[:,2],spec)
    fake=SimpleNamespace(
        linear_codes=codes,spec=spec,origin_local_m=(0.,0.,0.),
        centroids=(xyz.astype(float)+0.5)*0.5,
    )
    idx=SurfelLocalMapIndex(fake,2.0)
    got=idx.query_radius_ids(1.5,1.5,1.0)
    assert set(got.tolist())=={0,1}


def test_direct_surfel_point_to_plane_recovers_planar_xy_shift():
    # Two perpendicular vertical planes give x/y/yaw constraints.
    vals=np.linspace(-4,4,17)
    map_pts=[]; normals=[]
    for y in vals:
        for z in (-1.,0.,1.):
            map_pts.append([0.,y,z]); normals.append([1.,0.,0.])
    for x in vals:
        for z in (-1.,0.,1.):
            map_pts.append([x,0.,z]); normals.append([0.,1.,0.])
    mp=np.asarray(map_pts); ns=np.asarray(normals); valid=np.ones(len(mp),bool)
    # Body points equal world points under GT pose (0,0,0).
    body=mp.copy()
    init=Pose3DOF(0.6,-0.5,math.radians(3.0))
    cfg=PointToPlaneICP3DOFConfig(max_iterations=30,max_correspondence_m=2.0,trim_fraction=0.9,min_correspondences=20,normal_min_horizontal_component=0.0,max_step_translation_m=1.0,max_step_yaw_deg=5.0)
    res=scan_to_surfel_point_to_plane_icp_3dof(body,mp,ns,valid,init,0.,0.,0.,cfg)
    assert math.hypot(res.pose.x,res.pose.y)<0.12
    assert abs(math.degrees(res.pose.yaw))<0.5
