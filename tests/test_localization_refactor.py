"""Fast dependency/behavior guards for the unified localization pipeline."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from benchmark.localization_benchmark import method_names, VOXEL_METHOD, BST_METHOD
from benchmark.localization_helpers import FROZEN_PROFILE, FROZEN_K, _cap_points, _runtime_voxel_geometry_bytes, _union_excluded
from localization.global_frontend import ScanContextConfig, ScanContextDatabase, make_scan_context
from localization.retrieval import HistoricalRetriever


def test_frozen_geometry_and_names():
    assert FROZEN_K == 4
    assert (FROZEN_PROFILE.max_plane_sigma_m, FROZEN_PROFILE.max_relative_planarity,
            FROZEN_PROFILE.min_normal_coherence, FROZEN_PROFILE.max_extent_m) == (0.05, 0.025, 0.95, 2.0)
    assert method_names('bevplace') == (VOXEL_METHOD, BST_METHOD)
    assert method_names('scan_context') == ('sc_kiss_gicp_latest_prior_voxel', 'sc_kiss_gicp_latest_prior_bst_k4')
    with pytest.raises(ValueError): method_names('invalid')


def test_cap_points_preserves_order():
    p = np.arange(60, dtype=float).reshape(20, 3)
    assert np.array_equal(_cap_points(p, 7), p[np.linspace(0, 19, 7, dtype=np.int64)])
    assert np.array_equal(_cap_points(p, 20), p)


def test_voxel_storage_definition():
    m=SimpleNamespace(counts=np.zeros(3,dtype=np.int32),centroids=np.zeros((3,3),dtype=np.float64),cov6=np.zeros((3,6),dtype=np.float32))
    assert _runtime_voxel_geometry_bytes(m) == m.counts.nbytes+m.centroids.nbytes+m.cov6.nbytes


def test_bevplace_exact_replay(tmp_path):
    shared=dict(checkpoint_sha256=np.array('sha'),preprocess_id=np.array('p'),model_id=np.array('REIN'))
    np.savez(tmp_path/'reference_jan.npz',descriptors=np.array([[1.,0],[0.,1]],dtype=np.float32),
             scan_files=np.array(['a.bin','b.bin']),original_scan_indices=np.array([10,20]), **shared)
    np.savez(tmp_path/'query_feb.npz',descriptors=np.array([[.9,.1]],dtype=np.float32),
             scan_files=np.array(['q.bin']),original_scan_indices=np.array([33]), **shared)
    provider=HistoricalRetriever('bevplace',cache_dir=tmp_path,top_k=5)
    ref={'scan_indices':np.array([10,20]),'scans':[SimpleNamespace(name=f'{i}.bin') for i in range(21)]}
    ref['scans'][10].name='a.bin'; ref['scans'][20].name='b.bin'
    assert provider.load_reference('jan',['feb'],ref) == 16
    assert [m.reference_index for m in provider.query('feb','q.bin',33,np.empty((0,3)))] == [0,1]
    assert provider.provenance['checkpoint_sha256']=='sha'
    with pytest.raises(ValueError,match='index mismatch'): provider.query('feb','q.bin',32,np.empty((0,3)))
    ref['scan_indices']=np.array([20,10])
    with pytest.raises(ValueError,match='EXACT'): provider.load_reference('jan',['feb'],ref)


def test_sc_ranking_equivalent():
    cfg=ScanContextConfig(20,60,80,2)
    rng=np.random.default_rng(77); pts=rng.normal(size=(120,3)); pts[:,:2]*=12
    desc=make_scan_context(pts,cfg)
    ref={'descriptors':np.stack([desc,desc]), 'ring_keys':np.stack([desc.mean(axis=1),desc.mean(axis=1)])}
    provider=HistoricalRetriever('scan_context',sc_config=cfg,top_k=2,candidate_pool=2,descriptor_voxel_m=0)
    provider.load_reference('jan',['feb'],ref)
    m=provider.query('feb','q.bin',33,pts)
    baseline=ScanContextDatabase(ref['descriptors'],ref['ring_keys']).query_descriptor(desc,top_k=2,candidate_pool=2)
    assert [x.reference_index for x in m] == [x.reference_index for x in baseline]
    assert [x.distance for x in m] == [x.distance for x in baseline]


def test_scenario_exclusions(tmp_path):
    p=tmp_path/'previous.csv'; p.write_text('scenario,scan_file\ncross_session,a.bin\nsame_session,b.bin\n')
    assert _union_excluded([p]) == {'a.bin'}


def test_no_legacy_benchmark_or_localization_step_imports():
    root=Path(__file__).resolve().parents[1]
    files=list((root/'benchmark').glob('localization_*.py'))
    files+= [root/'localization'/f'{name}.py' for name in ('pose','point_to_plane','pyramid','surfel_icp','surfels','geometry_gate','lod','compact_map','history','global_frontend','bevplace','retrieval')]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith(('localization.step','benchmark.step')), (path,node.module)
