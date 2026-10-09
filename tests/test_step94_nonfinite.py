import numpy as np

from realdata.nclt_packed import pack_signed_xyz
from realdata.nclt_surfel import _reduce_scan_into_stats, _finalize_geometry


def test_surfel_reduce_drops_nonfinite_points_before_voxel_cast():
    codes = np.sort(pack_signed_xyz(
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
    ))
    counts = np.zeros(1, dtype=np.uint32)
    sum1 = np.zeros((1, 3), dtype=np.float32)
    sum2 = np.zeros((1, 6), dtype=np.float32)
    pts = np.asarray([
        [0.10, 0.10, 0.10],
        [np.nan, 0.20, 0.20],
        [0.20, 0.20, 0.20],
        [0.10, np.inf, 0.10],
    ], dtype=np.float64)

    _reduce_scan_into_stats(pts, 0.5, codes, counts, sum1, sum2)

    assert counts.tolist() == [2]
    assert np.all(np.isfinite(sum1))
    assert np.all(np.isfinite(sum2))


def test_surfel_finalize_skips_poisoned_voxel_statistics():
    codes = np.sort(pack_signed_xyz(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
    ))
    counts = np.asarray([6, 6], dtype=np.uint32)
    sum1 = np.zeros((2, 3), dtype=np.float32)
    sum2 = np.zeros((2, 6), dtype=np.float32)
    # Simulate a legacy accumulator poisoned by one bad transformed return.
    sum2[0, 0] = np.nan
    # A finite anisotropic covariance for the second voxel.
    sum2[1] = np.asarray([0.12, 0.0, 0.0, 0.06, 0.0, 0.006], dtype=np.float32)

    centroids, cov6, normals, curvature, normal_valid = _finalize_geometry(
        codes, 0.5, counts, sum1, sum2,
        normal_min_points=6,
        normal_max_curvature=0.20,
        normal_min_horizontal_component=0.0,
    )

    assert np.all(np.isfinite(centroids))
    assert np.all(np.isfinite(cov6))
    assert normal_valid[0] == 0
    assert np.isinf(curvature[0])
    assert np.all(np.isfinite(normals))
