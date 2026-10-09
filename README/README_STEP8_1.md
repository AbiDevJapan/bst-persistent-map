# Step 8.1 — Scale-safe real NCLT benchmark

Step 8.1 is the optimized replacement for the original Step 8 benchmark after full-route NCLT produced 4–5+ million occupied voxels per session and exposed Python-object scaling limits.

## What changed

- **No millions of `(x,y,z)` Python tuples.** Session maps are sorted packed `uint64` NumPy arrays.
- **Disk cache.** Raw NCLT preprocessing is cached per session + ingest configuration under `.cache/step8_1` by default.
- **Chunked vectorized voxel accumulation.** The old giant Python dictionary is gone.
- **Scale-safe Octree baseline.** `octree_shared_flat` has the same sparse-Octree leaf/history semantics and logical node accounting, but uses flat Morton-prefix ranges instead of millions of Python node objects.
- **Exact adaptive algorithm, vectorized.** The Step-5/6 independent dyadic x/y/z merge is preserved, but sibling matching uses packed NumPy keys.
- **No full-map materialization by default.** 5 m and 20 m history/change queries are the default. Full reconstruction of millions of return voxels is opt-in with `--include-full`.
- **Exact local correctness gate.** Every generated timed local query is compared exactly, as sorted packed arrays, against Snapshot once before timing.
- **Useful real-data queries.** 75% of history queries are occupied-centered; 75% of change queries are change-centered when possible. This avoids benchmarking mostly-empty random boxes in NCLT's large route bounding box.
- **Correct half-open region semantics.** Step 8.1 treats metric regions as `[min,max)` and uses `ceil(max/r)-1` for inclusive voxel upper bounds.

## Recommended first run for your two downloaded sessions

```bash
python3 run_step8_1.py \
  --data-root data \
  --sessions 2012-01-15,2012-02-02 \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --min-range 2 \
  --max-range 80 \
  --min-points-per-voxel 1 \
  --queries 10 \
  --warmups 1 \
  --repetitions 3 \
  --output results/step8_1_nclt2
```

For the first optimized run, use **10 queries / 1 warmup / 3 repeats**. After it completes and the output looks sensible, rerun the same cached sessions with `--queries 20 --warmups 2 --repetitions 5`. Raw NCLT ingestion will be skipped because the cache is reusable.

## Cache behavior

The first run prints progress while reading each session and writes packed session caches to:

```text
.cache/step8_1/
```

The cache key includes voxel size, scan stride, ranges, point threshold, pose-gap limit, ROI, and max-scan setting. Changing one of those parameters creates a separate cache automatically.

To disable caching:

```bash
--no-cache
```

To populate the cache without building any representation:

```bash
python3 run_step8_1.py ... --preprocess-only
```

## Why full queries are disabled by default

A full NCLT session can contain several million occupied voxels. Asking every method to return the entire map 20 times × warmups × repetitions primarily measures allocation and sorting of huge output arrays rather than index quality.

The paper-relevant default therefore measures localized historical/change queries at 5 m and 20 m. Full-map reconstruction remains available as a separate stress test:

```bash
--include-full
```

Do not use it in the first run.

## Output

```text
real_build_storage.csv
real_queries.csv
session_stats.csv
history_complexity.json
metadata.json
```

## Main comparison

Use:

```text
bst_compact_bucket vs octree_shared_flat
```

`adaptive_explicit` is also reported. It aliases the exact same adaptive runtime structure but uses explicit rectangular-bound logical storage, preserving the Step-7 BST-encoding ablation without duplicating a multi-million-block index in RAM.

## Important scientific caveats

- `octree_shared_flat` is a scalable flat Morton implementation of the same sparse Octree leaf/history semantics; it is actually a stronger systems baseline than the old Python-pointer Octree for large data.
- BST compact encoding remains the local Step-7 prototype and is **not** yet verified upstream BST-ID.
- Session differences remain **observation-history differences**, not ground-truth environmental change; missing occupancy is not treated as observed free space.
- Step 8.1 evaluates representation, not SLAM, because NCLT reference poses are used.
