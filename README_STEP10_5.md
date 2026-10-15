# Step 10.5 — Direct 4D BST-ID historical map database

This **patch** applies on top of the existing Step 10.4 checkout (which already includes Step 10.3). It does not replace or alter `store.py`, `delta_journal.py`, or localization. Install with `unzip -o ...` in the project root.

## What is implemented

- **IntervalMap** is a self-contained content-addressed database containing exact original zlib-compressed tile payloads and a compact `records.json` of successive unchanged-payload validity intervals. Every record refers to one *complete tile*, so point-in-time historical ROI queries **do not replay any deltas**. Records are merged only for *consecutive stored sessions with identical payload hashes*, and missing observations are never called free space.
- `bst` mode builds a persisted **four-axis index** using the **author's `region_algebra.Cell`**. Each record's [UTC start, UTC end) validity is covered exactly by dyadic t-prefix fragments. Each fragment has full-zoom author x/y/f tile-anchor codes and a variable-zoom t prefix; the serialized 17-byte values are genuine author `Cell.to_id()` encodings, checked against reconstructed cells. Lookup intersects indexed time ancestors with independent spatial prefix ranges, then exact integer tile boundaries. Point queries additionally verify actual 4D `Cell.compatible`.
- `morton` mode is a **strong conventional baseline**: the same interval records, the exact same byte-identical tile geometry, a Morton tile hash table, and per-tile binary search by version number. It does not inherit the slow unindexed manifest scan.
- `history_records` supports a query across stored historical versions, via BST temporal prefix overlap (or a per-tile Morton interval scan). It returns unique records whose validity intersects the requested *range of stored versions*, not observations recorded continuously over all seconds.
- `append_from_store` ingests **the next snapshot session**, closes changed tile validity intervals, extends unchanged intervals and adds new immutable blobs; it updates the compact manifest and derived index. It **does not yet ingest raw LiDAR directly** and does not have multi-writer transactions. Past geometry blobs are never modified.
- A benchmark compares **BST direct query**, **Morton direct query**, and **both Step 10.3 Morton-indexed and manifest-scan snapshot queries** on exactly identical cached observations and fixed repeated ROIs. It checks byte-for-byte voxel-code equality on every trial.

## On-disk layout

```
results/step10_5_intervals/
  bst/
    metadata.json       # spatial frame and time/codec/validity semantics
    records.json        # key, start-version, stop-version, payload hash, count
    index.npz           # official x/y/f payloads + author 4D cells/time-prefix postings
    blobs/<SHA256>      # compressed exact tile payloads, once per unique blob
  morton/
    ...                 # same geometry/validity/blob payloads; Morton+time index
```

An interval is of the form `(tile_xyz, [version_start,version_stop), hash)`. UTC version timestamps are converted to BST-ID `t` prefixes; the last stored version ends at 2^32 (2106-02-07) in the persisted index **only as a version-validity convention**. On append the open intervals are closed/extended as needed. The database only supports direct point queries for versions in `metadata.json['sessions']`.

**Spatial caveat:** upstream BST-ID encodes longitude/latitude/altitude and time. The NCLT local grid is mapped into the explicitly specified synthetic geospatial frame from Step 10. Canonical integer voxels are authoritative. The BST spatial cell addresses *tile anchors*, not exact full tile volumes. The query expands the ROI to all intersecting integer tiles before quantized prefix search, avoiding false negatives, and applies exact integer voxel filtering after retrieval. This is a genuine 4D indexed record lookup, not yet a hierarchy of arbitrary-sized geometric regions, nor a general continuous-space 4D database.

## Install and test

```bash
unzip -o Step10_5_Direct_4D_Historical_Map.zip
python -m pytest -q tests/test_step105_interval.py tests/test_step104_deltas.py tests/test_step103_index.py tests/test_step10_versioning.py
```

## Build from the existing five-session snapshot database

No raw NCLT processing is required if `results/step10_official` exists.

```bash
python run_versioned_map.py interval-build \
  --database results/step10_official \
  --interval-root results/step10_5_intervals \
  --address-mode both
```

Building both outputs makes independent, comparable self-contained databases. The input `results/step10_official` must have been constructed with the official codec.

Query March's map with BST or Morton:

```bash
python run_versioned_map.py interval-query \
  --interval results/step10_5_intervals/bst \
  --session 2012-03-17 \
  --bounds 722 786 965 1029 76 140

python run_versioned_map.py interval-query \
  --interval results/step10_5_intervals/morton \
  --session 2012-03-17 \
  --bounds 722 786 965 1029 76 140
```

The bounds are **half-open integer voxel indices**, not metric meters. To inspect validity intervals overlapping multiple stored versions:

```bash
python run_versioned_map.py interval-history \
  --interval results/step10_5_intervals/bst \
  --from-session 2012-02-02 \
  --to-session 2012-05-11 \
  --bounds 722 786 965 1029 76 140
```

## Append incrementally

If you want to evaluate the *actual per-session update path*, first build the base only in a fresh directory:

```bash
python run_versioned_map.py interval-build \
  --database results/step10_official \
  --interval-root results/step10_5_incremental \
  --address-mode both \
  --through-session 2012-01-15

for s in 2012-02-02 2012-03-17 2012-05-11 2012-06-15; do
  python run_versioned_map.py interval-append \
    --database results/step10_official \
    --interval-root results/step10_5_incremental \
    --address-mode both --session "$s"
done
```

Existing `interval-root` paths are *never overwritten by interval-build*.

## Verify and benchmark

```bash
python run_versioned_map.py interval-verify \
  --database results/step10_official \
  --interval-root results/step10_5_intervals \
  --queries 200 --seed 10505 --full-maps

python run_versioned_map.py interval-benchmark \
  --database results/step10_official \
  --interval-root results/step10_5_intervals \
  --output results/step10_5_official \
  --queries 200 --repeats 3 --seed 10505
```

Benchmark files:

- `trials.csv`: paired exact query outcomes with total/index latency, tiles, bytes, indexed candidates, ROI size and no-delta-replay proof.
- `summary.csv`: P50/P95/mean and size-specific timing per method.
- `storage.json`: separate *full measured on-disk bytes* for snapshot, BST interval DB and Morton interval DB, including index and metadata. Two DBs store their own copies of immutable blobs for independence.
- `metadata.json`: query generation, number of repeats, hardware, correctness contract and timing caveats.

**Benchmark controls:** repeated warm-process execution; order rotated per query/repeat; decoded tile blobs are *not cached*. Both `snapshot_scan` and `snapshot_indexed` (Step 10.3 Morton-hash index, built if absent) are measured. Index loading/build/initialization are **not part of query timing**, and can be substantial (especially the BST 4D postings). Be sure to compare `compressed_bytes_read` and true disk/RAM index overhead, not latency alone.

## What is — and is not — evidence for BST-ID

This step eliminates sequential replay and implements joint spatial and temporal *author BST-ID* indexing. It establishes **functional feasibility**. It does **not guarantee superiority** over Morton+time. If the latter wins, the BST-ID-specific contribution remains unproven. The validity run sharing may reduce storage, but that gain exists **identically in both modes** and is not a BST-specific result.

Follow-up work: per-tile/version-local update without a full compact index rebuild, larger adaptive spatial regions, native geometric cell covers (instead of anchors), cold-start memory profiling, compressed sync packets, and larger-session NCLT scalability.
