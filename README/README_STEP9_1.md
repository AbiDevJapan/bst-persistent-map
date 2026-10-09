# Step 9.1 — Native BST block-surface localization feasibility

## Goal

Step 9.1 is the first localization experiment that consumes the **native Step-8.5 time-local BST representation** rather than forcing it to return voxels.

The question is deliberately narrow:

> Within the local ICP basin established in Steps 9.0–9.0.3B, can one geometric surfel per native adaptive block preserve useful scan-to-map localization accuracy while reducing the number of map primitives supplied downstream?

This is **not** a global relocalization experiment.

## Frozen map representation

Step 9.1 uses the five already-downloaded NCLT sessions:

- 2012-01-15
- 2012-02-02
- 2012-03-17
- 2012-05-11
- 2012-06-15

The map representation is the frozen Step-8.5 system:

- 0.5 m canonical occupancy lattice;
- Step-8.4 time-local adaptive spatial tilings;
- Step-8.5 counted `(spatial bucket, session)` query index;
- `BSTTimeLocalTimeIndexedMap`;
- isolated `SpatialBlockCodec` boundary.

No new BST encoding rule is introduced in Step 9.1.

## Block surfel construction

For the map session (`2012-01-15` by default), Step 9.1 rebuilds the validated Step-9.0.3B per-voxel sufficient statistics from raw returns:

- point count `N`;
- centroid `mu`;
- covariance `Sigma`.

Those statistics are then propagated through **the exact same dyadic merge operations** used to form the Step-8.4 session tiling. When two sibling blocks merge, their sufficient statistics are merged analytically. The implementation asserts that the resulting spatial blocks are exactly the same geometry as the existing Step-8.4 map-session tiling.

Each resulting native block therefore has:

- its existing BST/adaptive bounds and temporal validity in the map representation;
- point count;
- sub-voxel centroid;
- covariance;
- derived plane normal;
- curvature / normal-validity flag.

There is no block-to-voxel decompression in the native path.

## Three localization interfaces

Step 9.1 benchmarks the same point-to-plane matcher with three map interfaces:

1. `voxel_surfel_0.5m`
   - direct Step-9.0.3B voxel-surfel baseline.

2. `bst_decompressed_voxel_surfel`
   - Step-8.5 BST query;
   - exact voxel decompression;
   - lookup of the **same voxel surfels** as baseline.
   - This is the compatibility/control path.

3. `bst_native_block_surfel`
   - Step-8.5 time-aware native block query;
   - one aggregated surfel per adaptive block;
   - no voxel decompression.

Before timing, Step 9.1 verifies that direct voxel-surfel retrieval and BST-decompressed voxel-surfel retrieval return exactly the same surfel rows for every benchmark initialization.

## Recommended run

Reuse the scans from Step 9.0.2 and the five existing Step-8 caches:

```bash
python3 run_step9_1.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --query-scans-csv results/step9_0_2_controls/query_scans.csv \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --query-count 10 \
  --same-session-query-count 10 \
  --perturbations 0.5:1 \
  --localization-max-range 30 \
  --scan-downsample 0.75 \
  --max-scan-points 8000 \
  --local-map-radius 40 \
  --output results/step9_1_native_blocks
```

No new NCLT session download is required.

## Outputs

- `localization_trials.csv`
- `localization_summary.csv`
- `query_scans.csv`
- `map_geometry_stats.csv`
- `metadata.json`

Important fields include:

- success rate;
- translation / yaw error;
- retrieval time;
- map preparation time;
- matching time;
- total time;
- local map primitive count;
- block candidates scanned / bucket references for the native BST path;
- block geometry payload size;
- full Step-8.5 BST representation size;
- mean voxel-equivalent cells represented per native primitive.

## Interpretation

The decisive comparison is:

`voxel_surfel_0.5m` vs `bst_native_block_surfel`.

A strong Step-9.1 result would show that native blocks retain approximately the voxel-surfel localization quality while returning substantially fewer local map primitives and reducing retrieval / downstream map handling.

The decompressed control should have essentially the same localization result as the direct voxel-surfel baseline. If it does not, treat that as an implementation error before interpreting native-block accuracy.

Suggested feasibility gate:

- zero-control accuracy close to the voxel-surfel baseline;
- `0.5 m / 1 deg` success rate at least ~90% of the voxel-surfel baseline success rate;
- materially fewer native primitives (ideally around the ~3x reduction already observed in Step 8.5 native map queries).

Do not require Step 9.1 to solve the `1 m / 3 deg` global/capture-range problem. That is a registration/front-end problem, not the native-map representation question isolated here.

## Scientific caveat

The spatial codec is still `PrototypeBSTSpatialCodec`, not verified upstream BST-ID. Step 9.1 tests whether the **native adaptive block representation** can carry useful localization geometry. The codec remains isolated and replaceable.
