# Step 8.4 — Time-local adaptive spatial tiling

## Why this step exists

Step 8.3 exposed a real limitation of the original adaptive representation. A voxel could only merge spatially with a neighbor when both voxels had the **same complete lifetime occupancy history**. As the number of sessions grows, histories diversify and this condition becomes increasingly restrictive.

Step 8.4 tests an exact alternative without downloading any more NCLT sessions:

1. Build an adaptive dyadic spatial tiling **independently for each session**.
2. Deduplicate identical block geometries across sessions.
3. Attach a shared temporal validity signature to each deduplicated geometry.
4. Query only geometries whose signature is active at the requested time.

This allows the spatial resolution to change over time. Two neighboring voxels may share one block during sessions where they can be merged even if their complete lifetime histories differ.

This is a research optimization, not a claim of globally minimum 4-D rectangular cover.

## Methods

Step 8.4 retains every Step 8.3 method and adds two methods built from the exact same new BST-agnostic core:

- `time_local_explicit` — explicit block coordinates + temporal signature ID.
- `bst_time_local` — the same blocks/signatures/query code, with spatial block addressing delegated to `PrototypeBSTSpatialCodec`.

The existing methods remain:

- `snapshot`
- `delta_version`
- `voxel_history`
- `octree_shared_sparse`
- `adaptive_explicit` (Step 8.3 full-history-conditioned blocks)
- `bst_compact_bucket` (Step 8.3 compact wrapper)

The comparison `time_local_explicit` vs `bst_time_local` isolates compact BST-style spatial addressing. The comparison `bst_compact_bucket` vs `bst_time_local` isolates the **construction optimization**.

## BST isolation

The new `TimeLocalAdaptiveCore` contains no BST import or BST encoding logic. It owns only:

- per-session adaptive spatial merging;
- geometry deduplication across sessions;
- shared temporal signatures;
- bucket indexing;
- native local-map retrieval;
- optional voxel decompression.

`BSTTimeLocalMap` is a thin wrapper. The replaceable codec boundary remains `bst_id/codec.py`, so integrating a verified upstream BST-ID codec later does not require rewriting this optimization.

## Query semantics

`output="native"` for `bst_time_local` returns clipped active blocks for `M(R,t)` without decompression.

`output="voxels"` expands those blocks and is exact-array checked against Snapshot for every generated historical query.

Because block boundaries may differ between two times, a signed native-block change set is not uniquely defined without an additional spatial Boolean decomposition. Therefore:

- historical/local-map queries: benchmark both native and voxel modes;
- change queries: benchmark exact voxel mode for `bst_time_local`;
- native change timing for the time-local methods is deliberately omitted rather than reporting misleading unequal semantics.

## Run on the five sessions you already have

The defaults match the completed Step 8.3 session set:

```bash
python3 run_step8_4.py \
  --data-root data \
  --sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --scales 2,3,5 \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --min-range 2 \
  --max-range 80 \
  --queries 20 \
  --warmups 2 \
  --repetitions 5 \
  --output-modes both \
  --output results/step8_4_time_local
```

The default cache directory remains `.cache/step8_1`. If Step 8.3 was run from the same project directory, no additional NCLT sessions should need to be downloaded or re-preprocessed.

Step 8.4 does **not** persist the large per-session block tilings to disk. They are computed in memory and reused across N=2,3,5 during that run; output is only the normal CSV/JSON result set.

## What to inspect first

At N=5 compare:

1. `stat_blocks` and `compact_mib`:
   - `bst_compact_bucket` vs `bst_time_local`
   - `voxel_history` vs `bst_time_local`

2. Native `region_at_time`:
   - `mean_result_primitives`
   - `median_ms`, `p95_ms`
   - especially `voxel_history`, `octree_shared_sparse`, `bst_compact_bucket`, `bst_time_local`

3. Canonical voxel `region_at_time` and `changes` to ensure the optimized representation remains usable when a downstream consumer explicitly needs voxels.

A successful result does **not** require every metric to improve. The optimization is useful if it materially improves the storage/native-query Pareto position while retaining exact reconstruction. If it reduces native primitives but increases storage, that is also a clear result and tells us that session-local tiling is better viewed as a localization-oriented representation than as the final archival encoding.
