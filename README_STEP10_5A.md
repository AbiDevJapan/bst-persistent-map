# Step 10.5A — Fair 4D Historical Queries + Lossless Spatiotemporal Coalescing

**Patch target:** `Step10_5_Direct_4D_Historical_Map.zip` (Step 10.5 checkout).
Apply with `unzip -o Step10_5A_BST_4D_Coalescing.zip` from the project's root.
The original localization code and the author's `third_party/bstid_author` package are **not modified**.

## A critical distinction: what was and was not merged previously

- **Step 10.5:** merged *consecutive versions of an identical complete-tile payload* into one `[start, stop)` interval. It did not merge spatially neighboring cells. It did not coalesce 4D cells as your earlier space-and-time BST mapping prototypes did.
- **Step 10.5A tile-cover index:** constructs nonempty **tile-presence** cells in **four dimensions**, normalizes them with exact bottom-up sibling merging along X, Y, Z and T, and queries the resulting 4D boxes directly. A spatial merge is legal even when the child tile payloads differ, because the covering cell means only that every child tile has *at least one observed endpoint* over the covered time interval. **Geometry is still fetched from the per-tile immutable payloads. This coalesces index cells but does not compress the geometry itself.**
- **Step 10.5A optional voxel-cell representation:** builds exact *occupied-endpoint voxel* presence over a bounded ROI, and merges true voxel cells in X/Y/Z/T only when both siblings are present for the whole region/time. This **does compress occupied-cell geometry**, but can have low spatial merging on sparse LiDAR maps. It is region-bounded and experimental; it does not replace the scalable per-tile blob database yet.

**Semantics:** "not observed" does not mean "free space". A temporal merge asserts the specified endpoint/tile was present in each stored observation state, not that the physical world remained unchanged continuously.

## BST-ID implementation changes (kept out of the authors' package)

1. **4D build-time sibling coalescing:** `versioned_map/aggregate_4d.py` builds dyadic time covers of per-tile presence runs and merges exact siblings along any axis until a fixed point. Every step preserves the exact set of present (tile, timestamp) pairs. `voxel_4d.py` applies the same operation to exact endpoint voxels for small ROIs.
2. **Actual single-stage 4D candidate filtering:** tests intersection between each coalesced cell's X/Y/Z/T bounds and the query's spatial bounding box and time interval in one vectorized operation. Removes Step 10.5's time-first query behavior that admitted an entire session before spatial filtering.
3. **Local-grid index frame made explicit:** this index embeds nonnegative *integer tile* or *integer voxel* positions at 21-bit X/Y/F zooms, and **actual UTC UNIX seconds** on the T axis. This is a different coordinate frame than the authors' *geodetic* longitude/latitude/elevation encoder. We use the author's `Cell.from_parts`, `to_id` and `from_id` unmodified for prefix-correct 4D cells. **Do not geodecode these local-frame BST IDs or combine them with geodetic-ID indexes.** The former Step 10.5 stored geodetic tile-anchor IDs; these new IDs are explicitly local-domain indices. Metadata labels the frame.
4. **Fixed-width binary-ID correctness:** opaque 17-byte integer words are stored as NumPy `V17`, not `S17`, avoiding silent stripping of trailing null bytes.
5. **Aligned integrity-validation work:** `IntervalMap.query(..., validation='hash')` can skip expensive canonical-array resorting in timed comparisons, while all methods verify SHA-256 checksums and returned voxel-code equality. The old `validation='full'` remains the default for compatibility.
6. **Automatic rebuilding on append:** any existing merged and unmerged aggregate sidecars are regenerated after `interval-append`, preventing use of stale time intervals. `interval-build --aggregate-cells` optionally builds merged index immediately during database construction.

The original official package in `third_party/bstid_author/` remains byte-for-byte unmodified. "BST-ID implementation improvements" refer to the **project's indexing adapter and storage/query subsystem**, not changes to the upstream standard or codec.

## Install and tests

```bash
unzip -o Step10_5A_BST_4D_Coalescing.zip
python -m pytest -q tests/test_step105a_aggregate.py \
  tests/test_step105_interval.py tests/test_step104_deltas.py \
  tests/test_step103_index.py tests/test_step10_versioning.py
```

## Build on the existing NCLT Step 10.5 interval databases

```bash
python run_versioned_map.py aggregate-build \
  --interval-root results/step10_5_intervals
```

This creates **two** index sidecars beside `bst/records.json`:

- `bst/aggregate_105a_merged.npz`: coalesced 4D BST cells
- `bst/aggregate_105a_raw.npz`: same cell-cover method **without** sibling merging (control)

It does *not* rewrite geometry blobs or overwrite Step 10.5's `index.npz`. The index reports initial cells, final cells, per-axis merges, build milliseconds and on-disk bytes. If merging cannot occur for sparse data, the report will say so.

For a new database build, use:

```bash
python run_versioned_map.py interval-build \
  --database results/step10_official \
  --interval-root results/step10_5_intervals_new \
  --address-mode both --aggregate-cells
```

## Queries

One historical map (same integer half-open ROI convention):

```bash
python run_versioned_map.py aggregate-query \
  --interval results/step10_5_intervals/bst \
  --session 2012-03-17 \
  --bounds 722 786 965 1029 76 140
```

Actual historical **4D range** returning all payload validity records overlapping *both* a spatial box and the February–June period:

```bash
python run_versioned_map.py aggregate-history \
  --interval results/step10_5_intervals/bst \
  --from-session 2012-02-02 --to-session 2012-06-15 \
  --bounds 722 786 965 1029 76 140
```

Optional unmerged control: add `--unmerged` to these commands.

## Exactness and fair benchmarking

```bash
python run_versioned_map.py aggregate-verify \
  --database results/step10_official \
  --interval-root results/step10_5_intervals \
  --queries 200 --seed 10506 --full-maps

python run_versioned_map.py aggregate-benchmark \
  --database results/step10_official \
  --interval-root results/step10_5_intervals \
  --output results/step10_5a_official \
  --queries 200 --repeats 3 --seed 10506
```

Benchmarked point-in-time paths: indexed snapshots, original BST interval, Morton+time interval, **unmerged BST 4D** and **merged BST 4D**. Real 4D range queries compare all four interval/index methods on **complete version-valid record sets**, not just matching tile keys. Methods rotate ordering and index initialization is outside timing. Interval reads and snapshot reads all check SHA-256 and result equivalence; the former no longer performs expensive redundant per-tile canonical sorts in the timed path. Outputs: `trials.csv`, `summary.csv`, `storage.json`, `metadata.json`.

**Storage interpretation:** `aggregate_105a_merged.npz` is a *sidecar* to the existing interval database. The baseline `interval_records`, geometry blobs and Step 10.5's index remain present and should be counted or explicitly retired before making total-system storage claims. The unmerged and merged sidecars should not both be counted in a production BST system; keep both for ablation.

**Memory interpretation:** reported RAM estimates count NumPy arrays only, not Python dictionaries, object overhead, peak build memory or process RSS. Full cold-start/build memory profiling is future work.

## Optional: actually merge occupied endpoint voxels over space AND time

For an ROI with manageable endpoint count:

```bash
python run_versioned_map.py voxel4d-build \
  --database results/step10_official \
  --output results/step10_5a_voxel_roi.npz \
  --bounds 722 754 965 997 76 108 \
  --max-atoms 500000

python run_versioned_map.py voxel4d-query \
  --index results/step10_5a_voxel_roi.npz \
  --session 2012-03-17 \
  --bounds 722 754 965 997 76 108
```

This separate experiment encodes the exact observed voxel set directly as coalesced 4D BST cells and reconstructs the requested region by expanding them. It will refuse huge intermediate cell counts rather than silently allocate unbounded RAM. Compare `raw_4d_cells`, `final_4d_cells` and file size to the source's compressed ROI observations. This is **not yet a competitive full-NCLT substitute for the compressed tile blobs**; evaluating scalable streaming build, true lossless occupancy compression and distortion-controlled aggregation is further research.

## Core correctness guarantees and limits

- Four-axis merging requires *both exact binary siblings*, with identical other-axis descriptors. It cannot fill holes or assume unobserved free space.
- The 4D tile-presence cover must never be used as final map geometry. Each candidate is resolved through the same exact immutable tile payload and validity interval, preserving all source points.
- Temporal version validity is based on stored session timestamps, not arbitrary intra-session time. The last version is valid through the author's 32-bit UNIX-time horizon.
- Cell spatial prefix values in this experiment are **local-grid IDs**, and **not geodetic cell positions**. This is an explicit domain-specific use of the official bit/prefix algebra.
- Built with a handful of synthetic dense/sparse controls and regression tests. **No full NCLT Step 10.5A benchmark is claimed in this patch.**
