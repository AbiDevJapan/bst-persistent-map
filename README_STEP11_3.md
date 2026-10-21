# Step 11.3 — Batched Contribution Deltas and Sparse Copy-on-Write Map Updates

**Aim:** Make pose-correction commits cheaper without replacing the already-correct Step 11.2 engine, its database schema, or its voxel semantics. This is a research **engineering milestone**, not a claimed new algorithm or a paper-grade baseline.

## Research role

The longer-term objective is exact historical map correction under delayed loop closures, with immutable source provenance and fast revision queries. Step 11.1 demonstrated correctness on NCLT; Step 11.2 checked update locality from 64 to 256 scans. Step 11.3 improves the *inner correction loop* so future experiments can compare optimized persistent occupancy against Roaring/OpenVDB-style structures and state-of-the-art mapping systems. It leaves SLAM accuracy, 3D surfel reconstruction, real loop-closure optimization, networking, and compression format learning for later milestones.

## Optimizations implemented

1. **Batched signed sparse deltas:** For each affected 32³ voxel tile, combine all corrected sources' old and new `uint16` local indices into net signed multiplicity changes. Coalesce identical indices before touching the persisted tile counter array. This avoids Python per-voxel `set` and `dict` loops in the update hot path.
2. **Vectorized reference counters:** Read legacy `(<u2 index,<u4 source_count)` zlib arrays directly with NumPy, apply delta changes with sorted union/search operations, and retain exact source overlap counts. No schema migration.
3. **Changed-only source dependencies:** Update `source_tiles` rows only if a source actually changes its occupied voxel set within a tile. An unchanged pose skips decoding entirely; a subvoxel pose correction still appends a pose-history event even when there is no occupancy change.
4. **Copy-on-write occupancy history:** Only write a new immutable tile page when the occupied voxel **union** changes. Changes to source multiplicities must still update the current counter table; they do not create unnecessary historical geometry pages. Pages retain canonical SHA-256 addresses.
5. **Packed bitmap encoding:** Construct canonical bitplanes with `np.packbits` rather than scatter-OR in Python/NumPy. Canonical byte order and SHA-256 digest are **unchanged** from Step 11.0. The page codec still chooses the smaller of zlib-compressed dense bitplanes and sparse local indices. This is **not** a full Roaring or compressed-domain update engine.
6. **Atomicity:** All revised source dependencies, current counters, optional geometry pages, interval changes, pose versions and the new revision log record remain in a single `BEGIN IMMEDIATE` SQLite transaction. The original tables and Step 11.0 API are unchanged.

### Corrections by source and tile

```
Correction batch: source A and B, old/new poses
        |
  for each source: reproject local observations
        |
   compare old vs new per-source tile indices
        |
  only changed (tile,source) pairs -> signed old(-1)/new(+1)
        |
   coalesce duplicate index updates per tile
        |
     apply to sorted source-count arrays
        |
   if occupied union changed: new immutable bitmap/sparse page
        |
  commit counters + changed source deps + version history atomically
```

### Semantics that must not change

- A voxel is *occupied* when at least one source observed an endpoint there. A missing endpoint does **not** imply free space.
- Source-to-voxel contributors are sets, not raw point multiplicity; two sources at the same voxel create source count 2.
- A new revision is created for a pose correction even if it leaves the rasterized voxel set unchanged.
- Previous committed revision queries must be byte-for-byte equivalent in coordinates, as verified against full reconstruction.
- Original local observations remain immutable, as do their SHA-256 identifiers.
- Observation time is distinct from revision/pose-correction time.

## Apply over Step 11.2

```bash
unzip -o Step11_3_Batched_Correction_Patch.zip
python -m pytest -q tests/test_step113_deltas.py tests/test_step11_correction.py tests/test_step111_nclt.py tests/test_step112_light.py
```

For existing Step 11.1 persisted data, no re-ingestion or database migration is required.

## Minimal real-NCLT data check

Use your saved Step 11.1 corrected database, **not** the prepared scan directory:

```bash
python run_step113_smoke.py \
  --database results/step11_1_experiment/corrected_map.sqlite \
  --output results/step11_3_smoke.json
```

The script clones the database using SQLite's backup API **outside timing**. It reverts the latest set of pose corrections and restores them, verifying each new revision against independent reconstruction, previous revision preservation, and immutable source data. The user's database is not modified. One run, two commits. No large benchmark suite.

## Observed development evidence (not paper results)

- Focused tests: see run output; clean-room extraction should run the same files.
- On a **single local trial** using the supplied 64-scan NCLT `corrected_map.sqlite`, reverting eight corrected poses with the previous Step 11.2 engine took ~336 ms versus ~227 ms with the updated engine. Restoring took ~262 ms versus ~225 ms. The trial uses the **same previously persisted database and voxel codec**, but run order, disk caches, and Python environment are not controlled, so don't cite these numbers as a stable speedup.
- Step 11.2 measured ~158, ~163, ~175 ms incremental commits with 64, 128, 256 scans on the user's Mac. Our tests here run on a different machine and correction direction; do not directly compare absolute times.

## What this DOES NOT implement

- No new four-dimensional BST-ID hierarchy and no BST-ID-specific speed claims.
- No full compressed bitmap **delta update**: the occupied tile's 4 KB canonical bitmap is recompressed when its union changes. We *avoid unchanged tiles*, not recompression of changed ones.
- No Roaring, RLE source-count compression, or optimal index choice; those remain candidates for later ablations.
- No full pose graph, loop closure, surfel/TSDF reconstruction, ground-truth georeferencing or multi-robot synchronization.
- No improvement claims over Voxgraph, maplab 2.0, Kimera-Multi, LT-Mapper or OpenVDB. These have capabilities our current endpoint-only database does not replicate.

## Next small milestone

Integrate the preserved Step 9 localization interface with geometrically adequate historical source statistics, then prototype page-level incremental union maintenance (e.g. sparse removals/additions without decoding entire source sets) only if profiling shows a bottleneck. Benchmark mature baselines **after** the architecture is feature-complete, with just lightweight correctness/smoke checks now.
