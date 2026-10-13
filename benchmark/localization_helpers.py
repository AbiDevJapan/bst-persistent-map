"""Frozen experiment constants and small I/O/data helpers.

Copied from historical steps with unchanged point-capping and map byte counting.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from localization.geometry_gate import GeometryGateProfile
from realdata.nclt import discover_nclt_session, _list_scan_files, _timestamp_from_scan, load_ground_truth
from realdata.nclt_packed import _nearest_pose_rows

FROZEN_PROFILE = GeometryGateProfile('balanced', 0.050, 0.025, 0.950, 2.0)
FROZEN_K = 4

def _write_csv(path: Path, rows: list[dict]):
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def _pctl(vals, q):
    if not vals:
        return float("nan")
    s = np.sort(np.asarray(vals, dtype=float))
    return float(np.quantile(s, q))

def _cap_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    # Deterministic uniform subsampling across file order.
    idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[idx]

def _read_excluded_scan_names(csv_path: str | Path | None, scenario: str = 'cross_session') -> set[str]:
    if csv_path is None:
        return set()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = set()
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            if 'scenario' in r and r['scenario'] and r['scenario'] != scenario:
                continue
            if r.get('scan_file'):
                out.add(r['scan_file'])
    return out

def _finite_pose_mask(pose_rows: np.ndarray) -> np.ndarray:
    """True for GT rows whose x/y/z/roll/pitch/yaw fields are all finite."""
    a = np.asarray(pose_rows, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError('pose_rows must be Nx7 or wider')
    return np.all(np.isfinite(a[:, 1:7]), axis=1)

def _select_query_scans_excluding(
    data_root: str | Path,
    session: str,
    count: int,
    max_pose_gap_us: int,
    excluded_names: set[str],
    exclude_ingest_stride: int | None = None,
):
    paths = discover_nclt_session(data_root, session)
    scans = _list_scan_files(paths.scan_dir)
    gt = load_ground_truth(paths.gt_csv)
    ts = np.asarray([_timestamp_from_scan(p) for p in scans], dtype=np.int64)
    pose_rows, gaps = _nearest_pose_rows(gt, ts)
    # A small number of NCLT ground-truth rows can contain non-finite pose
    # fields even though their timestamps are valid.  Timestamp-gap filtering
    # alone therefore is not sufficient for localization query selection.
    # Reject those rows here, before expensive map/hierarchy construction.
    finite_pose = _finite_pose_mask(pose_rows)
    valid = np.flatnonzero((gaps <= int(max_pose_gap_us)) & finite_pose)
    if exclude_ingest_stride is not None and int(exclude_ingest_stride) > 1:
        stride = int(exclude_ingest_stride)
        valid = valid[(valid % stride) != 0]
    if excluded_names:
        keep = np.asarray([scans[int(i)].name not in excluded_names for i in valid], dtype=bool)
        valid = valid[keep]
    if not len(valid):
        raise RuntimeError(f'No held-out query scans in {session} pass selection filters')
    n = min(int(count), len(valid))
    take = valid[np.linspace(0, len(valid) - 1, n, dtype=np.int64)]
    return [(scans[i], pose_rows[i], int(gaps[i]), int(i)) for i in take]

def _runtime_voxel_geometry_bytes(aligned) -> int:
    # Exact arrays consumed by prepare_surfel_pyramid at runtime.
    return int(aligned.counts.nbytes + aligned.centroids.nbytes + aligned.cov6.nbytes)


def _union_excluded(paths, scenario='cross_session'):
    """Exclude scans from prior benchmark CSVs; preserves the scenario filter.

    Historical code omitted the required scenario argument on this rarely-used
    path, causing a TypeError; supply the intended default here.
    """
    out = set()
    for path in paths or []:
        if path:
            out |= _read_excluded_scan_names(path, scenario)
    return out
