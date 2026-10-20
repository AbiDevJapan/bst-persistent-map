from __future__ import annotations
import json,sqlite3
from types import SimpleNamespace
import numpy as np
import pytest
from collab_feasibility import s3e
from collab_feasibility.benchmark import benchmark


def test_s3e_gt_pose_interpolator(tmp_path):
    p=tmp_path/'gt.txt'
    p.write_text('# gt\n1000 10 20 2 0 0 0 1\n1001 12 20 2 0 0 0 1\n1002 14 20 2 0 0 0 1\n')
    a=s3e.gt_file(p)
    xyz,R=s3e.PoseInterpolator(a).at(1000.5)
    np.testing.assert_allclose(xyz,[11,20,2]);np.testing.assert_allclose(R,np.eye(3))
    assert s3e.PoseInterpolator(a).at(1003) is None


def test_pointcloud2_padding_endianness_and_nonfinite():
    # two rows, one point each, extra 4 byte padding per row
    for endian in ('<','>'):
        raw=b''.join(np.array([x,2,3],dtype=endian+'f4').tobytes()+b'pad!' for x in (1,float('nan')))
        f=[SimpleNamespace(name=n,datatype=7,offset=4*j) for j,n in enumerate('xyz')]
        msg=SimpleNamespace(fields=f,data=raw,width=1,height=2,point_step=12,row_step=16,is_bigendian=(endian=='>'))
        np.testing.assert_allclose(s3e.pointcloud_xyz(msg),[[1,2,3]])


def test_topic_selection_ambiguous_requires_override():
    topics=[dict(id=i+1,name=f'/{r}/velodyne_points',type='sensor_msgs/msg/PointCloud2',serialization_format='cdr')
            for i,r in enumerate(('alpha','bob','carol'))]
    result=s3e.select_lidar_topics(topics)
    assert set(result)==set(s3e.ROBOTS)
    with pytest.raises(ValueError,match='alpha'):
        s3e.select_lidar_topics(topics+[dict(id=4,name='/alpha/other',type='sensor_msgs/msg/PointCloud2',serialization_format='cdr')])


def test_extrinsic_required_and_explicit_identity():
    with pytest.raises(ValueError,match='Provide calibrated'):
        s3e.load_extrinsics()
    assert all(np.array_equal(v,np.eye(4)) for v in s3e.load_extrinsics(assume_identity=True).values())


def test_replay_small_real_format_mock(tmp_path,monkeypatch):
    # Simulate the shape of a real ROS2 S3E dataset, not its actual contents.
    # Tests the full importer + three VersionedStores + Geometry4D + protocol.
    db=tmp_path/'S3E_Playground_2.db3'; gtroot=tmp_path/'gt';gtroot.mkdir()
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE topics(id INTEGER, name TEXT, type TEXT, serialization_format TEXT)')
        for i,r in enumerate(s3e.ROBOTS):
            con.execute('INSERT INTO topics VALUES(?,?,?,?)',(i+1,f'/{r}/velodyne_points','sensor_msgs/msg/PointCloud2','cdr'))
    t0=1660988157
    for i,r in enumerate(s3e.ROBOTS):
        lines=[f'{t0+j} {744073+i*2+j*.5:.3f} 2553112.0 0.0 0 0 0 1' for j in range(31)]
        (gtroot/f'{r}_gt.txt').write_text('\n'.join(lines))
    monkeypatch.setattr(s3e,'_make_typestore',lambda _topics:None)
    def fake_scans(db,selected,typestore,*,stride,max_scans):
        for j in range(30):
            for i,r in enumerate(s3e.ROBOTS):
                # A vertical wall with 200 points in each scan, shifted with robot.
                world=np.array([[8.0,k*.5,z] for k in range(10) for z in range(2,9)],float)
                pts=world-np.array([i*2+j*.5,0,0])
                yield r,t0+j+.25,pts
    monkeypatch.setattr(s3e,'decode_scans',fake_scans)
    sources,meta=s3e.build_s3e(db,gtroot,tmp_path/'out',epochs=3,voxel_size=1.,scan_stride=1,
                       max_range=30.,assume_identity_extrinsics=True)
    assert len(sources)==3
    assert all(meta['scan_stats'][r]['accepted']==30 for r in s3e.ROBOTS)
    assert all(len(s[0].all_versions())==3 for s in sources)
    tasks=s3e.s3e_request_tasks(sources,meta,9,2.,8.,1.,10)
    assert {t['source_robot'] for t in tasks}
    summary,info=benchmark(sources,tmp_path/'results',requests=9,gps_sigma_m=2.,
                            alignment_sigma_m=1.,local_radius_m=8.,seed=10,tasks=tasks,dataset_meta=meta)
    assert all(s['all_correct'] for s in summary)
    assert (tmp_path/'results'/'report.md').exists()
    assert info['source_robots']==3
    assert info['nonempty_target_region_fraction']>0, 'fixture must exercise real shared geometry'
