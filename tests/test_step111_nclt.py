"""Real NCLT format fixture, provenance, controlled correction, and exact rebuild."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest

from versioned_map.correction.nclt_ingest import NCLTSourceConfig, prepare_nclt, benchmark_nclt
from versioned_map.correction import CorrectionMap


def fixture_nclt(root,session='2012-03-17',count=6):
    scan_dir=root/session/'velodyne_sync'
    scan_dir.mkdir(parents=True)
    ts0=1_331_910_000_000_000
    rows=[]
    dt=np.dtype([('x','<u2'),('y','<u2'),('z','<u2'),('intensity','u1'),('laser','u1')])
    for i in range(count):
        us=ts0+i*100_000
        points=np.zeros(6,dtype=dt)
        points['x']=[20_600+i*25,20_800+i*25,21_000+i*25,21_200+i*25,21_400+i*25,0]
        points['y']=[20_200,20_400,20_600,20_800,21_000,0]
        points['z']=[20_050,20_060,20_100,20_120,20_150,0]
        points.tofile(scan_dir/f'{us}.bin')
        rows.append([us,float(i)*.5,0,0,0,0,0.03*i])
    np.savetxt(root/session/f'groundtruth_{session}.csv',np.asarray(rows),delimiter=',',fmt='%.9f')
    return session


def test_nclt_prepare_exact_scan_provenance(tmp_path):
    session=fixture_nclt(tmp_path/'data')
    cfg=NCLTSourceConfig(scan_stride=1,max_scans_per_session=10,min_range_m=0,max_range_m=100,
                         corrupt_from=1,corrupt_count=2,corrupt_translation_m=(1,0,0),corrupt_yaw_deg=3)
    result=prepare_nclt(tmp_path/'data',[session],tmp_path/'out',cfg)
    assert result['sources']==6 and result['correction_sources']==2
    initial=json.loads((tmp_path/'out'/'manifest_initial.json').read_text())
    truth=json.loads((tmp_path/'out'/'manifest_corrected.json').read_text())
    corrections=json.loads((tmp_path/'out'/'corrections.json').read_text())
    assert len(corrections)==2
    assert initial[1]['source_id'] in corrections
    assert initial[0]['pose']==truth[0]['pose']
    assert initial[1]['pose']!=truth[1]['pose']
    assert initial[1]['observed_at_us']>1_000_000_000_000_000
    raw=Path(initial[1]['scan_file'])
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==initial[1]['raw_scan_sha256']
    with np.load(tmp_path/'out'/initial[1]['npz'],allow_pickle=False) as arr:
        assert arr['points'].shape==(5,3)
        assert arr['points'].dtype==np.float64
    with pytest.raises(FileExistsError):prepare_nclt(tmp_path/'data',[session],tmp_path/'out',cfg)


def test_nclt_pose_gap_skip(tmp_path):
    session=fixture_nclt(tmp_path/'data',count=3)
    gt=tmp_path/'data'/session/f'groundtruth_{session}.csv'
    rows=np.loadtxt(gt,delimiter=',')
    rows[1,0]+=200_000
    np.savetxt(gt,rows,delimiter=',',fmt='%.9f')
    cfg=NCLTSourceConfig(scan_stride=1,max_scans_per_session=3,min_range_m=0,
                         max_pose_gap_us=20_000,corrupt_from=0,corrupt_count=1)
    result=prepare_nclt(tmp_path/'data',[session],tmp_path/'out',cfg)
    assert result['sources']<3 and result['skipped_pose_gap']>0


def test_controlled_nclt_full_rebuild(tmp_path):
    session=fixture_nclt(tmp_path/'data',count=9)
    cfg=NCLTSourceConfig(scan_stride=1,max_scans_per_session=9,min_range_m=0,
                         corrupt_from=3,corrupt_count=3,corrupt_translation_m=(2,0.5,0),corrupt_yaw_deg=5)
    prepare_nclt(tmp_path/'data',[session],tmp_path/'prepared',cfg)
    out=benchmark_nclt(tmp_path/'prepared',tmp_path/'experiment',batch_size=4,repeats=2)
    assert out['all_exact'] and out['all_previous_revisions_preserved']
    assert out['sources']==9 and out['changed_sources']==3
    assert out['median_full_disk_materialization_ms']>0
    assert out['corrected_voxel_count']>0
    with CorrectionMap(tmp_path/'experiment'/'initial_map.sqlite') as db:
        assert db.stats()['sources']==9
        assert db.verify()
    csv_lines=(tmp_path/'experiment'/'trials.csv').read_text().splitlines()
    assert len(csv_lines)==3


def test_prepare_rejects_missing_gt_or_invalid_config(tmp_path):
    with pytest.raises(ValueError): NCLTSourceConfig(scan_stride=0)
    with pytest.raises(ValueError): NCLTSourceConfig(corrupt_translation_m=(0,0,0),corrupt_yaw_deg=0)
    with pytest.raises(FileNotFoundError): prepare_nclt(tmp_path,['2012-03-17'],tmp_path/'out',NCLTSourceConfig())


def test_nclt_decoder_rejects_truncated_packet(tmp_path):
    from realdata.nclt import read_velodyne_sync_bin
    broken=tmp_path/'bad.bin'
    broken.write_bytes(b'123456789')
    with pytest.raises(ValueError,match='not divisible'):
        read_velodyne_sync_bin(broken)


def test_nclt_traceable_sqlite_ledger(tmp_path):
    session=fixture_nclt(tmp_path/'data',count=4)
    prepare_nclt(tmp_path/'data',[session],tmp_path/'prepared',
                 NCLTSourceConfig(scan_stride=1,max_scans_per_session=4,min_range_m=0,
                                  corrupt_from=1,corrupt_count=2))
    benchmark_nclt(tmp_path/'prepared',tmp_path/'results',repeats=1,full_disk_rebuild=False)
    with CorrectionMap(tmp_path/'results'/'corrected_map.sqlite') as db:
        n=db.con.execute('SELECT COUNT(*) FROM nclt_source_provenance').fetchone()[0]
        assert n==4
        src=db.con.execute('SELECT source_id, observed_at_us, reference_pose FROM nclt_source_provenance LIMIT 1').fetchone()
        assert src[0].startswith('nclt/') and src[1]>1_000_000_000_000_000
        assert len(src[2])==128
        assert db.revision>1 and db.verify()
        assert db.verify(db.revision-1)


def test_load_ground_truth_filters_nonfinite_rows_without_fabricating_poses(tmp_path):
    from realdata.nclt import load_ground_truth, _NearestPoseMatcher
    p = tmp_path / 'gt.csv'
    rows = np.asarray([
        [100_000., 1., 2., 3., 0., 0., 0.],
        [200_000., np.nan, 2., 3., 0., 0., 0.],
        [np.inf, 1., 2., 3., 0., 0., 0.],
        [300_000., 2., 3., 4., 0., 0., .2],
        [400_000., 2., 3., 4., 0., np.inf, .2],
    ])
    np.savetxt(p, rows, delimiter=',')
    gt, info = load_ground_truth(p, return_stats=True)
    assert gt.shape == (2, 7)
    assert np.isfinite(gt).all()
    assert list(gt[:, 0]) == [100_000, 300_000]
    assert info['rows_total'] == 5
    assert info['rows_discarded'] == 3
    assert info['rows_nonfinite_pose'] == 2
    assert info['rows_nonfinite_timestamp'] == 1
    assert np.array_equal(gt, load_ground_truth(p))
    _, gap = _NearestPoseMatcher(gt).nearest(200_000)
    assert gap == 100_000  # missing pose must NOT be interpolated


def test_prepare_accepts_partially_nonfinite_nclt_groundtruth(tmp_path):
    session = fixture_nclt(tmp_path / 'data', count=7)
    path = tmp_path / 'data' / session / f'groundtruth_{session}.csv'
    rows = np.loadtxt(path, delimiter=',')
    rows[1, 1] = np.nan
    rows[3, 5] = np.inf
    np.savetxt(path, rows, delimiter=',', fmt='%.9f')
    config = NCLTSourceConfig(scan_stride=1, max_scans_per_session=7,
                              min_range_m=0, max_pose_gap_us=20_000,
                              corrupt_from=2, corrupt_count=1)
    result = prepare_nclt(tmp_path / 'data', [session], tmp_path / 'prepared', config)
    assert result['sources'] == 5
    assert result['skipped_pose_gap'] == 2
    assert result['groundtruth_quality'][0]['rows_valid'] == 5
    assert result['groundtruth_quality'][0]['rows_discarded'] == 2
    assert result['correction_sources'] == 1


def test_load_ground_truth_rejects_all_nonfinite_rows(tmp_path):
    from realdata.nclt import load_ground_truth
    p = tmp_path / 'bad.csv'
    np.savetxt(p, np.array([[100., np.nan, 0., 0., 0., 0., 0.],
                             [200., 0., 0., np.inf, 0., 0., 0.]]), delimiter=',')
    with pytest.raises(ValueError, match='No finite timestamp'):
        load_ground_truth(p)
