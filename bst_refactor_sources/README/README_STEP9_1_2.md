# Step 9.1.2 — Hierarchical multi-resolution block-surfels

## Goal

Step 9.1.1 showed that geometry-aware gating is useful, but its two-level fallback is too expensive: a rejected native parent expands directly to all of its 0.5 m voxel surfels, so the map rapidly approaches voxel-map size.

Step 9.1.2 tests the next question:

> Can a rejected native block descend only as far as needed through a dyadic surfel hierarchy, preserving localization quality without giving up most of the native primitive-count/runtime advantage?

The underlying lifelong map is still the unchanged Step-8.5 time-local BST map. Step 9.1.2 adds only a localization LOD overlay.

## Core design

While reproducing the exact Step-8.4 map-session merges, Step 9.1.2 retains a compact binary merge forest. The final roots are exactly the same native adaptive blocks used by Step 9.1 / 9.1.1.

Each internal node stores sufficient geometry for localization:

- point count;
- centroid;
- covariance;
- derived normal / curvature / validity;
- absolute plane thickness;
- relative planarity;
- immediate-child normal coherence;
- physical extent;
- two child node IDs.

Leaf node IDs refer directly to the frozen 0.5 m voxel-surfel table, so leaf geometry is not duplicated.

## Offline LOD selection

For each native root:

1. test the geometry gate;
2. if it passes, keep that node as one surfel;
3. if it fails and the node has children, descend to the two children;
4. repeat until a child passes or the `K` depth cap is reached.

The default comparison uses:

- `K=1` — pure Step-9.1 native block surfel;
- `K<=2`;
- `K<=4`;
- `K<=8`.

`K` is an **upper bound**, not a forced count. A planar root can remain one primitive even in the K=8 view.

Because the hierarchy is binary, the maximum recursion depth is only `log2(K)`. All subdivision decisions and root-to-selected-node tables are precomputed before localization.

Online localization performs only:

`BST time-aware root query -> precomputed root LOD lookup -> radius filter -> existing surfel matcher`

There is no PCA, threshold search, recursive split decision, or voxel decompression decision in the online loop.

## Frozen geometry gate

Step 9.1.2 intentionally freezes one gate while sweeping K, so hierarchy depth is the only new variable. The default is the Step-9.1.1 **balanced** setting:

- absolute plane sigma <= 0.050 m;
- relative planarity <= 0.025;
- child-normal coherence >= 0.950;
- maximum node extent <= 2.0 m.

The child-normal term is hierarchical here: when both child normals are valid it uses `abs(dot(n_a, n_b))`, so opposite normal sign is treated as equivalent.

You can override the gate with:

`--geometry-profile name:sigma:relative:coherence:extent`

but the recommended first run keeps the balanced profile fixed.

## Exactness checks

The implementation asserts that:

- final hierarchy roots reproduce the exact Step-8.4 map-session tiling;
- hierarchy root surfel centroids match the native block surfels;
- every LOD view covers exactly the same number of base voxels as the frozen map;
- no root produces more than K selected surfels.

Thus K changes only localization geometry resolution, not map occupancy/history content.

## Recommended run

```bash
python3 run_step9_1_2.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --query-scans-csv results/step9_0_2_controls/query_scans.csv \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --query-count 20 \
  --same-session-query-count 20 \
  --perturbations 0.5:1 \
  --geometry-profile balanced:0.050:0.025:0.950:2.0 \
  --k-values 2,4,8 \
  --localization-max-range 30 \
  --scan-downsample 0.75 \
  --max-scan-points 8000 \
  --local-map-radius 40 \
  --output results/step9_1_2_hierarchical_lod
```

No new NCLT sessions are required.

## Benchmarked methods

- `voxel_surfel_0.5m` — conventional voxel-surfel baseline;
- `bst_native_block_surfel` — pure native K=1 baseline;
- `bst_lod_k2`;
- `bst_lod_k4`;
- `bst_lod_k8`.

All methods use the same coarse-to-fine point-to-plane matcher and the same replayed initial poses.

## Outputs

- `localization_trials.csv`
- `localization_summary.csv`
- `query_scans.csv`
- `lod_profile_stats.csv`
- `pareto_summary.csv`
- `metadata.json`

`lod_profile_stats.csv` reports:

- global selected primitive count;
- valid-normal fraction;
- geometry payload estimate;
- LOD directory bytes;
- unchanged BST representation bytes;
- roots refined;
- mean / p95 / max selected surfels per root;
- selected leaf-vs-internal-node fraction;
- mean voxel-equivalent cells per selected primitive;
- hierarchy build time and LOD-decision time.

`pareto_summary.csv` focuses on the cross-session `0.5 m / 1 deg` development case and reports success, primitive reduction, total-time speedup, pose error, storage, and Pareto flags.

## Development-set warning

The same 20 cross-session scans remain a development/tuning set. The balanced gate was selected using Step 9.1.1 results, and K is selected here. Do **not** use these 20 scans as the final unbiased performance claim.

After choosing K, validate the frozen operating point on a larger held-out scan set and additional session pairs.

## What counts as a useful result

The target is not necessarily K=8. A useful operating point should preserve approximately voxel-surfel localization quality while retaining a large share of the K=1 primitive/runtime advantage.

A strong outcome would be roughly:

- baseline-level success / pose error;
- 40–60% fewer local primitives than voxel surfels;
- 1.5–2x lower total localization time;
- substantially smaller geometry payload than full voxel fallback.

## Scientific caveats

- Step 9.1.2 is a localization-specific LOD overlay, not a new lifelong-map representation.
- The underlying Step-8.5 occupancy/history representation and time-aware BST query index are unchanged.
- The matcher remains a local refinement method, not a global relocalizer.
- `PrototypeBSTSpatialCodec` remains a local research codec, not verified upstream BST-ID.
