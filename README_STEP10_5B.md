# Step 10.5B: Build-time adaptive **geometry** cells + historical time aggregation

This is an **additive patch** to the Step 10.5A project. Your previous
`results/step10_official` voxel snapshot DB remains unmodified. The legacy
localization pipeline is unaffected.

## What is (and is not) merged

1. **Spatial geometry:** within each 32³ tile and each session, every occupied
   0.5 m endpoint voxel starts as a fully occupied dyadic cell. Adjacent,
   aligned, fully occupied sibling cells are recursively merged across X/Y/Z
   while the total number of fine voxels within a block does not exceed K
   (`--k 4` by default). **The geometry itself** is now stored as larger
   spatial blocks, rather than as a list of unmerged voxel IDs.
2. **Temporal geometry:** the same spatial block across different historical
   versions is stored **once**, with a version-membership bitmask (up to 32
   versions). The builder makes an additional sibling-merge pass only for
   neighboring blocks with *identical* version masks. Thus history influences
   spatial merging in the final representation.
3. **4D BST-ID identifiers:** real upstream `region_algebra.Cell` IDs encode
   X/Y/F/T tile time covers in `bst_index.npz`, and true **merged geometry cell**
   4D IDs can be exported with `Geometry4D.four_d_cells(...)`. All coordinate
   prefix math is on a LOCAL nonnegative 21-bit voxel grid and UNIX seconds,
   not geodetic BST-ID coordinates. Upstream author source remains unchanged.

**This is a lossless endpoint-occupancy representation.** A merged cell of
shape (2,2,1) implies *all four* constituent fine voxels were observed. A
sparse region does not get falsely filled. Original world-frame scan points,
per-point covariance and normals are **not** present in the Step 10 database;
this is **not** an exact reimplementation of Step 9's planarity/normal-coherent
K=4 materialized surfel gate. The `geometry-describe` command derives the
centroid and second moments of the *occupied voxel centers* analytically;
it does not recover raw LiDAR covariance or normals. A subsequent lossy
surfel experiment would need raw sufficient statistics and a distortion gate.

Time membership denotes a stored *version observation*, not continuously
observed physical occupancy or free-space evidence. A block valid across two
sessions is not proof that it stayed occupied throughout elapsed time.

## Installation

From the Step 10.5A working directory:

```sh
unzip -o Step10_5B_Adaptive_Geometry_Versioning.zip
python -m pytest -q tests/test_step105b_geometry.py \
  tests/test_step105a_aggregate.py tests/test_step105_interval.py \
  tests/test_step104_deltas.py tests/test_step103_index.py \
  tests/test_step10_versioning.py
```

## Build from existing NCLT snapshot store

No re-ingestion of raw NCLT scans is necessary:

```sh
python run_versioned_map.py geometry-build \
  --database results/step10_official \
  --geometry-root results/step10_5b_k4 \
  --k 4
```

For an **identical-information unmerged geometry** ablation:

```sh
python run_versioned_map.py geometry-build \
  --database results/step10_official \
  --geometry-root results/step10_5b_k1 \
  --k 1
```

Do not reuse an existing `--geometry-root`; builds deliberately refuse
clobbering output. The builder currently supports tile sizes up to 256 and
up to 32 chronological sessions (32-bit masks), so NCLT's 27 sessions fit.

### Query exact historical endpoint voxels

```sh
python run_versioned_map.py geometry-query \
  --geometry results/step10_5b_k4 \
  --session 2012-03-17 \
  --bounds 722 786 965 1029 76 140 --mode bst

python run_versioned_map.py geometry-query \
  --geometry results/step10_5b_k4 \
  --session 2012-03-17 \
  --bounds 722 786 965 1029 76 140 --mode morton
```

Both must return **exactly** the same sorted `uint64` IDs as the original
snapshot reference. Spatial cell expansion happens only at query time.

### Use the compressed geometry *without* expanding it

```sh
python run_versioned_map.py geometry-primitives \
  --geometry results/step10_5b_k4 \
  --session 2012-03-17 \
  --bounds 722 786 965 1029 76 140 --mode bst \
  --save-npy results/step10_5b_blocks.npy

python run_versioned_map.py geometry-describe \
  --geometry results/step10_5b_k4 \
  --session 2012-03-17 \
  --bounds 722 786 965 1029 76 140 --limit 10
```

`geometry-primitives` saves `N × 6` integer blocks
`[x,y,z,width_x,width_y,width_z]` in voxel grid units. They are **real
geometry cells** with all fine voxels occupied, not 16 m tile labels.
`geometry-describe` reports center positions and covariance of the supported
voxel-center lattice (not LiDAR samples).

## Incremental build and append

```sh
python run_versioned_map.py geometry-build \
  --database results/step10_official \
  --geometry-root results/step10_5b_streaming \
  --k 4 --through-session 2012-01-15

for s in 2012-02-02 2012-03-17 2012-05-11 2012-06-15; do
  python run_versioned_map.py geometry-append \
    --database results/step10_official \
    --geometry results/step10_5b_streaming --session "$s"
done
```

Append processes the **new session's tile blobs** and extends existing
spatial-block time masks. It rebuilds both index sidecars. Old immutable
content-addressed blobs can remain unreferenced after append; optional
`geometry-append --gc` removes these **only if no concurrent reader is
using an older manifest**. Updating source snapshots from raw LiDAR directly
is not part of this patch.

## Verify and benchmark

```sh
python run_versioned_map.py geometry-verify \
  --database results/step10_official \
  --geometry results/step10_5b_k4 \
  --queries 200 --seed 10507 --full-maps

python run_versioned_map.py geometry-benchmark \
  --database results/step10_official \
  --geometry results/step10_5b_k4 \
  --unmerged-geometry results/step10_5b_k1 \
  --output results/step10_5b_official \
  --queries 200 --repeats 3 --seed 10507
```

The benchmark **builds any missing indexed Morton snapshot controls before
starting timing**, so the snapshot baseline is not disadvantaged by an
unindexed manifest scan.

Outputs:
- `trials.csv` and `summary.csv`: paired **exact** historical voxel queries;
  BST tile-time selector vs Morton on identical adaptive cells, indexed
  snapshots, and optional `K=1` geometry control.
- `primitive_trials.csv` and `primitive_summary.csv`: directly retrieve the
  geometric block representation without expanding back to fine voxels.
- `storage.json`: bytes for shared geometric payloads, BST and Morton indexes,
  matched K=1 geometry baseline, and the original snapshot database.
- `metadata.json`: exactness and benchmark conditions.

Query workload: 50% known occupied-tile centered and 50% random regions,
with sides 16/32/64/128 fine voxels, across historical sessions.

## Interpretation and limitations

- **Geometry effect**: compare `--k 4` to `--k 1` using the *same Morton index*.
  Report primitive-count reduction **and full bytes including indexes**;
  zlib voxel arrays can already be very compact, so reduced primitives do not
  guarantee smaller files.
- **BST-ID-specific index effect**: compare BST to Morton within the same
  `K=4` database. Geometry, temporal bitmasks and compressed payloads are
  identical; only candidate selection and indexing differ.
- **Temporal effect**: compare `spatial_blocks_across_sessions` to
  `distinct_spatial_blocks_over_time` in `geometry-stats`. Unlike the earlier
  tile-hash interval method, unchanged blocks can be reused **inside a tile
  even if other voxels in the tile change**.
- `bst_index.npz` uses true author 4D tile-time cells as a candidate cover;
  the per-geometry-cell temporal masks are checked after tile selection.
  `four_d_cells(...)` exports true 4D IDs for the actual merged blocks, but a
  fully ordered, sub-tile, hierarchical 4D trie is not implemented. The
  author BST-ID library was **not modified**.
- The spatial merger merges **only fully occupied dyadic siblings**, with a
  hard K cap. It does not merge sparse partially occupied neighboring cells
  into approximate surfels. That would be a separate *lossy* design requiring
  source scan geometry and error evaluation.
- A `K=4` block can cover at most 4 original voxels, so don't infer
  8-to-1 3D cubic cell merges are possible at that K.
- Latency is local disk + CPU, no network transfer; no NCLT Step 10.5B
  measurements are bundled. Always inspect total storage and correctness
  before claiming a BST-specific improvement.
