# Step 8.5 — Counted time-aware bucket indexing

## Why this step exists

Step 8.4 showed that session-local spatial tiling roughly halved the number of native blocks returned for a local map, but the union bucket index still retrieved geometries that were spatially relevant yet inactive at the requested session. At N=5 this made the native query inspect substantially more records than it returned.

Step 8.5 changes **only the auxiliary query index**. The Step-8.4 block table, temporal signatures, optional decompression, and BST codec remain unchanged.

## The index

Step 8.4 union index:

```text
spatial bucket -> every block geometry that ever overlaps the bucket
                 -> temporal state is checked after retrieval
```

Step 8.5 time-aware index:

```text
(spatial bucket, session) -> only block geometries active in that session
```

A region query at time `t` therefore avoids pulling inactive geometries into the candidate set.

The index is not free. `compact_bytes()` includes the selected index directory and all block references. The benchmark retains the Step-8.4 union-index methods in the same run so the speed/storage tradeoff is measured directly.

## New methods

Step 8.5 retains every previous method and adds:

- `time_local_tindex_explicit`
- `bst_time_local_tindex`

These use the **exact same `TimeLocalAdaptiveCore.blocks`** as:

- `time_local_explicit`
- `bst_time_local`

The only difference is the auxiliary index policy.

Therefore:

- `bst_time_local` vs `bst_time_local_tindex` isolates the time-aware index.
- `time_local_tindex_explicit` vs `bst_time_local_tindex` isolates the compact BST-style spatial encoding under the new index.
- `bst_compact_bucket` vs `bst_time_local_tindex` changes both construction and index and is not the clean index-only ablation.

## BST isolation remains unchanged

`TimeLocalAdaptiveCore` is BST-agnostic. `BSTTimeLocalTimeIndexedMap` is a thin wrapper that selects `index_mode="time"` and delegates spatial encoding to `SpatialBlockCodec`.

A future verified BST-ID codec still replaces only `bst_id/codec.py`.

## Query semantics

Historical/local-map queries support both:

```python
output="native"
output="voxels"
```

For the time-local representations, native output is the same clipped block set under both the union and time-aware indexes.

Exact voxel change queries also use the time-aware index at each endpoint before set difference.

Native signed change blocks remain deliberately unsupported because time-varying tilings can use different block boundaries at the two endpoints.

## Storage accounting

For the union index:

```text
12 bytes × number of spatial bucket directory entries
+ 4 bytes × number of block references
```

For the time-aware index:

```text
12 bytes × number of (bucket,session) directory entries
+ 4 bytes × number of active block references
```

Only the index selected by a method is included in that method's `compact_bytes()` value. The in-memory benchmark core constructs both indexes once to avoid duplicating the very large block table, but the reported standalone storage and build accounting do not charge either method for the other index.

The build CSV includes diagnostic fields such as:

- `stat_union_bucket_references`
- `stat_time_bucket_references`
- `stat_union_index_bytes`
- `stat_time_index_bytes`
- `stat_time_index_extra_bytes`
- `stat_union_index_build_ms`
- `stat_time_index_build_ms`

## Run using the same five cached sessions

No new NCLT download is needed.

```bash
python3 run_step8_5.py \
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
  --output results/step8_5_time_index
```

The default cache remains `.cache/step8_1`.

## What to inspect first

At N=5, compare `bst_time_local` and `bst_time_local_tindex` for historical native queries:

1. `median_ms` and `p95_ms`
2. `mean_inspected`
3. `mean_result_primitives` — this should be identical because the block table/output is unchanged
4. `compact_mib`
5. `stat_time_index_extra_bytes`

A successful result is a Pareto tradeoff, not necessarily a free speedup. The main question is how many MiB of extra index storage are required to move `mean_inspected` toward the number of actually active/returned blocks.

If the latency improvement is substantial, Step 8.5 should be the last representation-side optimization before the localization feasibility experiment.
