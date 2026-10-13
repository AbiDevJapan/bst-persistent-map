"""Frozen Step-9.5-compatible BEVPlace++ retrieval from official pretrained descriptors.

No modification of the BST/voxel map, registration, or localization representation.
Does not perform neural inference itself: use run_step9_6_extract.py to produce NPZs.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class BEVPlaceMatch:
    reference_index: int    # exact row of Step-9.5 reference database
    distance: float         # unit-normalized Euclidean distance, ascending
    # Compatibility with ScanContextMatch, not used as geometric yaw evidence:
    ring_distance: float = float('nan')
    sector_shift: int = 0
    relative_yaw_rad: float = 0.0


def unit_descriptors(desc: np.ndarray) -> np.ndarray:
    d = np.asarray(desc, dtype=np.float32)
    if d.ndim != 2 or not d.shape[0] or not d.shape[1]:
        raise ValueError('descriptors must be nonempty N x D')
    if not np.all(np.isfinite(d)):
        raise ValueError('nonfinite descriptor')
    norms = np.linalg.norm(d.astype(np.float64), axis=1)
    if np.any(norms < 1e-12):
        raise ValueError('zero-norm descriptor')
    return np.ascontiguousarray(d / norms[:, None], dtype=np.float32)


class BEVPlaceDatabase:
    def __init__(self, descriptors: np.ndarray):
        self.descriptors = unit_descriptors(descriptors)

    def query_descriptor(self, descriptor: np.ndarray, *, top_k: int = 5):
        q = unit_descriptors(np.asarray(descriptor, dtype=np.float32).reshape(1, -1))[0]
        if q.shape[0] != self.descriptors.shape[1]:
            raise ValueError('BEVPlace++ descriptor dimension mismatch')
        if top_k < 1:
            raise ValueError('top_k must be positive')
        # Already normalized: squared L2 = 2 - 2 * dot; clamp for float noise.
        squared = np.maximum(0.0, 2.0 - 2.0 * (self.descriptors @ q))
        # argsort stable: tied scores go to smaller historical DB index.
        indices = np.argsort(squared, kind='stable')[:int(top_k)]
        return [BEVPlaceMatch(int(i), float(np.sqrt(squared[i]))) for i in indices]


def _scalar(z, key: str) -> str:
    if key not in z: raise ValueError(f'missing descriptor metadata: {key}')
    a = np.asarray(z[key])
    if a.ndim != 0: raise ValueError(f'invalid scalar metadata: {key}')
    return str(a.item())


class BEVPlaceDescriptorCache:
    """Loads independent reference/query descriptors with matching provenance.

    Files are required to contain exporter-produced scan_file and original scan
    index mappings; filename-only or row-order-only matches are not trusted.
    """
    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)

    def _load(self, kind: str, session: str) -> dict:
        path = self.root / f'{kind}_{session}.npz'
        if not path.is_file(): raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            required = ('descriptors','scan_files','original_scan_indices','checkpoint_sha256','preprocess_id','model_id')
            for key in required:
                if key not in z: raise ValueError(f'{path}: missing {key}')
            files = np.asarray(z['scan_files']).astype(str)
            indices = np.asarray(z['original_scan_indices'], dtype=np.int64)
            desc = unit_descriptors(z['descriptors'])
            meta = {k:_scalar(z,k) for k in ('checkpoint_sha256','preprocess_id','model_id')}
            poses = np.asarray(z['pose_rows'],dtype=np.float64) if 'pose_rows' in z else None
        if len(set(files.tolist())) != len(files) or len(set(indices.tolist())) != len(indices):
            raise ValueError(f'{path}: duplicated reference/query scans')
        if not (len(files) == len(indices) == len(desc)):
            raise ValueError(f'{path}: descriptor rows and identifiers mismatch')
        if poses is not None and len(poses) != len(desc):
            raise ValueError(f'{path}: reference pose count mismatch')
        return dict(descriptors=desc,scan_files=files,original_scan_indices=indices,meta=meta,poses=poses,path=path)

    def load_pair(self, reference_session: str, query_session: str, reference_scan_indices: np.ndarray,
                  reference_scan_files: list[str] | np.ndarray) -> tuple[BEVPlaceDatabase, dict]:
        ref = self._load('reference', reference_session)
        q = self._load('query', query_session)
        if ref['meta'] != q['meta']:
            raise ValueError('BEVPlace++ checkpoint/model/preprocess metadata mismatch')
        if ref['descriptors'].shape[1] != q['descriptors'].shape[1]:
            raise ValueError('query/reference feature dimensions disagree')
        idx = np.asarray(reference_scan_indices, dtype=np.int64)
        names = np.asarray([Path(str(p)).name for p in reference_scan_files]).astype(str)
        if not np.array_equal(ref['original_scan_indices'], idx) or not np.array_equal(ref['scan_files'], names):
            raise ValueError('reference descriptors do not match EXACT Step 9.5 database rows/order')
        q_by_name = {str(name):(int(oi), d) for name, oi, d in zip(q['scan_files'],q['original_scan_indices'],q['descriptors'])}
        return BEVPlaceDatabase(ref['descriptors']), dict(query_by_name=q_by_name,provenance=ref['meta'],reference_path=str(ref['path']),query_path=str(q['path']))

    @staticmethod
    def query_descriptor(query_by_name: dict, scan_name: str, original_scan_index: int) -> np.ndarray:
        name = Path(scan_name).name
        if name not in query_by_name: raise KeyError(f'no BEVPlace++ descriptor for query {name}')
        oi, desc = query_by_name[name]
        if int(oi) != int(original_scan_index):
            raise ValueError(f'query index mismatch for {name}: {oi} != {original_scan_index}')
        return desc
