# Step 11: Correction-Aware Lifelong Mapping — Research Starting Point

**Status:** runnable, tested correction-aware reference engine, not a validated publication-grade NCLT system.

**The roadmap supplied in `docs/USER_RESEARCH_ROADMAP.md`** prioritizes global map pose corrections, provenance, bitemporal historical reads, and preservation of localization quality. The preceding project experiments also showed that the BST-ID codec alone was not winning fairly against Morton/hash baselines and that bitmap-oriented storage was promising.

This refactor resolves those competing objectives by **decoupling map truth and correction semantics from the index format**. A new, deliberately modest `versioned_map/correction/` backend uses a signed global voxel frame, compressed bitmap/sparse pages and SQLite MVCC. The **complete historical BST-ID/Morton and localization code is preserved** as an experimental baseline; there is no claim that the new backend is a BST-ID algorithm or that it outperforms the previous snapshot query latency. The next experiment can implement a dynamic BST-ID index and compare it against exactly this source-corrected map state.

## What is actually implemented

| Milestone | Status | Evidence/qualification |
| --- | --- | --- |
| 11.0 reproducible baseline | Implemented in package | Frozen source from uploaded Step 10→10.5B archive sequence; previous tests retained |
| 11.1 immutable observations, provenance, pose history | **Implemented** | Multi-robot source IDs, source-frame point blobs, revisions and source→tile contributions |
| 11.2 independent full map reconstruction | **Implemented** | Rebuild every source's points at its pose known at revision R |
| 11.3 correction engine | **Implemented — synthetic tests** | Batched corrections, counter-safe source removal and reinsertion, localized tile updates, atomically published revision |
| 11.4 committed historical revisions | **Implemented** | Copy-on-write immutable pages + revision validity intervals; previous committed maps directly queryable |
| 11.4 retrospective observation-time queries | **Correct oracle only** | Rebuilds relevant source observations using poses known at requested revision, not indexed/optimized |
| 11.5 dynamic BST-ID | **Not implemented** | Author-supplied implementation and Step 10 experiments retained; no native BST correction updates yet |
| 11.6 fair Morton/BST comparisons | **Not performed** | New core provides codec-neutral occupancy outputs and reference timings |
| 11.7 localization validation | **Not integrated** | Legacy Step 9 code preserved; surfel statistics are not reconstructed from endpoint occupancy |

The new code intentionally does **not** claim paper-ready novelty, multi-robot consensus, loop-closure computation, original LiDAR covariance, physical occupancy/free-space, or NCLT pose-correction accuracy.

## Layout

```text
run_correction_map.py               # New primary correction-aware CLI
versioned_map/
  correction/
    __init__.py
    voxel.py                      # signed voxelization, reversible uint16 tile IDs, bitmap codec
    engine.py                     # immutable observations, contribution dependencies,
                                  # copy-on-write pages, pose revisions, bitemporal reads
    benchmark.py                  # paired full rebuild vs incremental correction
scripts/
  make_synthetic_correction_data.py
configs/
  step11_experiment.json

tests/test_step11_correction.py     # new correctness suite

run_versioned_map.py                # Existing Steps 10–10.5B, unchanged
versioned_map/geometry_105b.py     # Existing BST/Morton adaptive geometry experiment
third_party/bstid_author/          # Author BST-ID source, unmodified
run_localization.py, localization/ # Existing Step 9 pipeline, unchanged
```

This is a **compatibility-first reorganization**, not a destructive rename of research modules. Old command lines still work; new experiments use the standalone correction CLI.

## Architecture

```text
 Local source scan (immutable XYZ points, robot ID, observation time)
       + 4x4 source-to-global pose at each committed revision
                             |
                             v
                 Signed global 0.5 m voxel grid
                             |
                             v
           Source -> {32³ tile -> sorted local voxel IDs}
                             |
                    Correction transaction
     remove only selected sources' OLD contributions
      add only those sources' NEW pose contributions
        counts retain overlapping other-robot sources
                             |
                             v
          Affected tile occupancy: direct set of bits
                   sparse_u16 or bitplane
                             |
                             v
           Immutable SHA256 content-addressed pages
                             |
                             v
         SQLite tile_versions (revision validity)
                             |
        +--------------------+-------------------+
        |                    |                   |
  direct revision ROI  temporal XOR count  correction-aware history
```

For each voxel `v`, the mutable *current* source count is

`count(v) = number of distinct admitted sources whose local measurement set transforms into v`.

For any source correction, update **only its old and new tiles**. A voxel is present iff its count is nonzero; a correction cannot erase another robot's overlapping observation. Each distinct occupied tile state is compressed into either a 4096-byte-uncompressed bitmap (`32³/8`) or an array of packed 16-bit local indices, whichever compresses smaller with zlib-6. Immutable pages are shared by content SHA256. Empty tiles are represented by tombstones in the revision intervals.

A committed map revision R is selected by the interval `[from_revision, to_revision)`; querying a previous revision does **not** replay intervening changes. Reinterpreting observations collected up to time T **as known at revision R** is exposed separately through `observed-history`: the initial implementation rebuilds from source observations for correctness.

**Times:** `observed_at` is Unix time (integer UTC seconds); revision is monotonic **transaction time**, not a measurement timestamp. No inference of environmental stability is made between scans.

**Coordinate frame:** local point coordinates are projected by an SE(3) 4×4 transform into a fixed metric global frame; `floor((world-origin)/voxel_size)` yields signed grid coordinates. The caller must align all robots' frames ahead of time and supply any pose corrections; this engine does not estimate SLAM poses.

## Install

```bash
unzip Step11_Correction_Aware_Lifelong_Map_Starter.zip -d correction-aware-map
cd correction-aware-map
python -m pip install -r requirements_step11.txt
python -m pytest -q tests/test_step11_correction.py
```

### 1. Generate and ingest a synthetic 4-robot dataset

```bash
python scripts/make_synthetic_correction_data.py --output examples/generated --sources 64 --points 250
python run_correction_map.py init --database results/correction_map.sqlite --voxel-size 0.5
python run_correction_map.py ingest-manifest \
  --database results/correction_map.sqlite \
  --manifest examples/generated/manifest.json --batch-size 16
python run_correction_map.py stats --database results/correction_map.sqlite
python run_correction_map.py verify --database results/correction_map.sqlite
```

The JSON source manifest contains **local-frame** NPZ paths, source and robot IDs, timestamps, and optionally an initial 4×4 global pose (embedded NPZ `pose` is also accepted). This generator creates synthetic scans ONLY; it is not NCLT processing.

### 2. Correct an old robot pose without rebuilding unrelated tiles

```bash
python run_correction_map.py correct \
  --database results/correction_map.sqlite \
  --source-id scan_0000 --translation 0.5 0 0
```

For a multi-scan loop closure, provide one atomic correction batch:

```bash
python run_correction_map.py correct-batch \
  --database results/correction_map.sqlite \
  --corrections-json examples/generated/example_corrections.json
```

`--translation` is a delta to the last committed source pose's translation. `correct-batch` accepts **absolute** 4×4 source-to-global matrices.

### 3. Query historical beliefs or a retrospectively corrected map

```bash
python run_correction_map.py query --database results/correction_map.sqlite \
  --revision 1 --bounds -100 100 -100 100 -20 20

python run_correction_map.py observed-history --database results/correction_map.sqlite \
  --observed-until 1550000010 --revision 5 \
  --bounds -100 100 -100 100 -20 20

python run_correction_map.py diff --database results/correction_map.sqlite \
  --left 1 --right 5 --bounds -100 100 -100 100 -20 20
```

`query` reads stored version-valid occupancy pages; `observed-history` rebuilds original sources observed before timestamp T **using all pose updates known at revision R**. `diff` counts newly observed and no-longer-observed voxels using XOR/AND on packed bitmap pages without expanding them into point coordinates.

### 4. Run the paired correction benchmark

```bash
python run_correction_map.py benchmark --database results/correction_map.sqlite \
  --source-id scan_0000 --repeats 5 --output results/step11_benchmark
```

This writes `trials.csv` and `summary.json`. Each trial clones the DB outside timed operations, compares a corrected-source incremental commit with an independently rebuilt complete union, and checks **exact equality**. `incremental_ms` includes transaction commit/durability. `full_rebuild_ms` currently measures in-memory rebuilding rather than persistence, so raw timing is **not** a fully fair end-to-end rebuild comparison. Keep this distinction explicit.

## Bring real NCLT data into the new core

**The existing Step 10 snapshot database is insufficient for a real pose-correction study.** It stores global occupied endpoint voxel IDs but not their original source-scan poses and dependencies. You cannot reconstruct those from compressed occupancy alone.

Convert real NCLT scans into per-scan or per-submap `scan.npz` files containing `points` (N×3, local source frame) and `pose` (4×4 transform into a common global metric frame). Create a JSON manifest:

```json
[
  {
    "source_id": "robot-A/2012-03-17/scan-001",
    "robot_id": "robot-A",
    "observed_at": 1331940000,
    "npz": "scans/scan-001.npz"
  }
]
```

Timestamps above are illustrative; use the actual acquisition timestamps. A correction to a local SLAM trajectory must be provided as absolute revised transforms. Subsequent experiments can measure localized correction cost for one scan, many scans, entire submaps, and trajectory segments. Save both the old and corrected poses for reproducibility.

Do **not** import a Step 10 world map as if it were a set of real local scans with true provenance: that would fabricate the dependencies we are trying to evaluate.

## Verified baseline vs planned research

- Tested: exact voxel occupancy at committed revisions, multiple overlapping robots, negative signed coordinates, proper rigid transforms, atomic rollback, batch correction, repeated randomized revisions, source deduplication, bitmap/sparse encoding, bitemporal reference semantics, history after process reopen.
- Benchmarked only on synthetic source scans: performance depends on source footprint, overlap, affected tile count, map size, and durability settings.
- Historical source measurements are stored as float64 local XYZ. Raw LiDAR per-point intensity, timestamps, normals and covariances are **not** retained. Integrating the existing FAST-LIVO2/Step 9 localization backend requires a separate full surfel-statistics layer.
- SQLite supports one serialized writer and consistent reader snapshots. It is **not** a decentralized multi-robot conflict-resolution protocol.
- Direct committed-revision reads are spatially indexed; retrospective observation-time queries are currently full-rebuild oracles.
- Disk bytes include source provenance, SQLite index pages and WAL; compressed occupancy payload bytes alone are not a whole-system storage result.

### Next experiments

1. **M1 on real sources:** NCLT scan/submap ingestion, long-trajectory pose corrections, full rebuild equality, affected-region scaling.
2. **M2 algorithms:** Dynamic BST-ID, hierarchical Morton, and spatial hash backends consuming the **same** source delta events and occupancy bitpages. Measure localized index maintenance; do not ascribe generic bitmap compression to the index.
3. **M3 head-to-head:** Storage including provenance, revision query latency, affected-only correction update latency, index memory, log replay/batch behavior, synchronized transfers. Include map-size scaling at constant correction radius.
4. **M4 localization:** Build genuine per-source surfel sums/cross-products with correct removal and reinsertion. Compare unchanged Step 9 localization scans against full rebuild and incrementally corrected maps.

**Decision gate:** do not make BST-ID the default if it does not beat a well-implemented Morton/spatial hash system on a predefined meaningful workload.
