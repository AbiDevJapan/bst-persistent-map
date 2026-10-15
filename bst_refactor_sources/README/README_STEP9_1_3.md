# Step 9.1.3 — Compact materialized K=4 runtime

## Goal

Step 9.1.2 established a useful development operating point: the frozen **balanced** geometry gate with **K=4** hierarchical LOD. However, that benchmark kept the full merge forest resident so selected LOD entries could refer back to hierarchy node IDs. The hierarchy itself was construction machinery, not part of the intended deployed representation.

Step 9.1.3 asks a narrower engineering question:

> Can the selected K=4 LOD be materialized into a standalone runtime table, discard the construction hierarchy, preserve exactly the same localization result, and count the actual arrays required by the unchanged matcher?

There is **no new localization algorithm and no threshold/K tuning** in this step.

## Frozen operating point

Step 9.1.3 hard-freezes the Step-9.1.2 development choice:

- base occupancy/index lattice: `0.5 m`;
- geometry gate: `balanced`;
- absolute plane sigma: `0.050 m`;
- relative planarity: `0.025`;
- child-normal coherence: `0.950`;
- maximum node extent: `2.0 m`;
- maximum LOD primitives per native root: **K=4**.

These values are not swept in this package.

## Construction vs deployed runtime

Offline construction still temporarily creates the binary merge hierarchy:

`0.5 m voxel surfels -> native merge forest -> balanced K=4 selection`

Then Step 9.1.3 materializes only the selected K=4 payload:

- `uint32` point count;
- `float32` centroid `(x,y,z)`;
- `float32` covariance `cov6`;
- global Step-8.5 block ID for every native root;
- root start offset;
- root selected-count (`1..4`).

After verification/reference localization, the construction hierarchy and selected hierarchy node IDs are released. The deployed query path is:

`Step-8.5 (bucket,session) BST root query -> compact root directory -> materialized LOD rows -> radius filter -> unchanged coarse-to-fine surfel matcher`

No merge-tree traversal, recursive splitting, PCA gating, voxel decompression decision, or selected hierarchy node ID is needed online.

## Corrected runtime-memory accounting

Earlier Steps 9.0.3B–9.1.2 reported a "minimal localization payload" based on centroid/normal-style fields. The current matcher actually rebuilds coarse/medium/fine map levels from sufficient statistics, so its exact runtime input is:

`count + centroid + covariance`

Step 9.1.3 therefore counts those exact arrays. It also counts the global root block IDs that are required to map Step-8.5 query results into the root directory.

The CSV distinguishes:

- construction-only hierarchy bytes;
- deployed LOD geometry bytes;
- deployed root-directory bytes;
- existing BST/history bytes;
- total deployed bytes for the method.

This is the memory accounting to use going forward for this matcher.

## Equivalence checks

Before the hierarchy can be discarded, Step 9.1.3 asserts:

1. materialized counts equal the selected hierarchy-node counts;
2. materialized centroids are exactly equal;
3. materialized covariances are exactly equal;
4. root starts/counts and global block IDs are equal;
5. radius queries return exactly the same selected primitives;
6. several prepared coarse/medium/fine local-map pyramids are numerically identical.

With the reference path enabled, the benchmark also runs both:

- `bst_lod_k4_reference` — Step-9.1.2 hierarchy-backed K=4;
- `bst_lod_k4_compact` — materialized hierarchy-free K=4;

and asserts equality of deterministic localization outputs trial-by-trial (pose, success, iterations, correspondences, residuals, failure state). Timing fields are intentionally excluded from the equality assertion.

## Recommended equivalence run

Use the same Step-9.1.2 development scans first:

```bash
python3 run_step9_1_3.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --query-scans-csv results/step9_1_2_hierarchical_lod/query_scans.csv \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --query-count 20 \
  --same-session-query-count 20 \
  --perturbations 0.5:1 \
  --localization-max-range 30 \
  --scan-downsample 0.75 \
  --max-scan-points 8000 \
  --local-map-radius 40 \
  --output results/step9_1_3_compact_k4
```

This run is for implementation equivalence, not a new performance claim.

## Optional held-out run

After equivalence passes, select new scans while excluding the tuning CSV:

```bash
python3 run_step9_1_3.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --exclude-query-scans-csv results/step9_1_2_hierarchical_lod/query_scans.csv \
  --query-count 50 \
  --same-session-query-count 50 \
  --perturbations 0.5:1 \
  --no-reference \
  --output results/step9_1_3_heldout_jan_feb
```

`--exclude-query-scans-csv` excludes all scan filenames listed for the matching scenario and then deterministically spreads the requested held-out scans across the remaining valid traversal.

For final paper-level validation, repeat the frozen runtime on additional cross-session pairs rather than retuning K or the geometry gate.

## Benchmarked methods

The default equivalence run includes:

- `voxel_surfel_0.5m` — conventional runtime baseline;
- `bst_lod_k4_reference` — hierarchy-backed Step-9.1.2 reference;
- `bst_lod_k4_compact` — proposed deployed runtime.

Use `--no-reference` for larger held-out runs after equivalence is established.

## Outputs

- `localization_trials.csv`
- `localization_summary.csv`
- `query_scans.csv`
- `deployment_memory.csv`
- `materialization_verification.csv`
- `metadata.json`

### `deployment_memory.csv`

Reports per method:

- map elements;
- exact runtime geometry bytes;
- directory/index bytes;
- existing BST representation bytes;
- deployed runtime total;
- construction hierarchy bytes;
- whether construction hierarchy is retained at runtime;
- materialization time;
- mean base voxels represented per primitive.

### `materialization_verification.csv`

Reports structural/query/pyramid equivalence and, when the reference path is enabled, trial-output equivalence.

## Interpretation

A successful result is straightforward:

- compact K=4 gives the same deterministic localization output as the Step-9.1.2 K=4 path;
- compact runtime does not retain the merge hierarchy;
- actual deployed memory is counted from the arrays the matcher really consumes;
- runtime remains comparable to or better than the Step-9.1.2 K=4 result.

If these hold, future localization experiments should use only `bst_lod_k4_compact` and treat the hierarchy as an offline construction artifact.

## Scientific caveats

- The balanced gate and K=4 were selected on development data in Steps 9.1.1/9.1.2 and are frozen here.
- Replaying those 20 scans demonstrates implementation equivalence only.
- Held-out scans and additional session pairs are required before final localization claims.
- The underlying Step-8.5 occupancy/history semantics and time-aware index are unchanged.
- `PrototypeBSTSpatialCodec` remains a local research codec, not verified upstream BST-ID.
