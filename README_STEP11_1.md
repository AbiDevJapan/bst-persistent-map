# Step 11.1 — NCLT Source Provenance & Controlled Pose-Correction Benchmark

## Why this step exists

Step 11.0 provided a **synthetic** correction-aware historical map with immutable local XYZ sources, pose revisions, bitmap pages, source-overlap counters, and a full-rebuild correctness oracle. **Step 11.1 tests those assumptions using genuine LiDAR scan footprints and poses.** It supplies the reference experiment for future dynamic BST-ID, Morton, spatial-hash, and compressed-bitmap indexing designs. This is research **Milestone M1** (correct incremental correction) from `docs/USER_RESEARCH_ROADMAP.md`.

Step 11.1 is **not** a new SLAM/loop-closure estimator. It perturbs selected *ground-truth* NCLT poses and measures the consequences of restoring them. The design isolates **map data structure update cost** from pose-estimation quality, allowing a deterministic correctness oracle.

## What is new

- `run_nclt_correction.py`: `prepare` and `experiment` subcommands.
- `versioned_map/correction/nclt_ingest.py`: timestamped NCLT source conversion, explicit world-frame correction, microsecond-accurate provenance manifest, and paired real-format experiment.
- `tests/test_step111_nclt.py`: NCLT binary-format fixtures, GT timestamp matching, filtering, pose correction, historical version preservation, and fresh corrected database equivalence.
- `realdata/nclt.py`: one-pass synchronized scan decoding rather than two separate reads of the same file. **No changes to decoded XYZ or coordinate conventions.**
- Optional `nclt_source_provenance` auxiliary SQLite ledger with original scan SHA256, prepared scan SHA256, microsecond acquisition timestamp, nearest GT pose gap and reference pose.

**Unchanged:** canonical `floor((world-origin)/voxel_size)` voxel semantics, bitmap/sparse page codec, immutable source ownership, source overlap counts, existing Step 10 baseline methods, upstream BST-ID, and localization.

## NCLT expected layout

Any folder structure supported by `realdata/nclt.py`, e.g.:

```text
data/
  2012-03-17/
    velodyne_sync/
      133...bin
    groundtruth_2012-03-17.csv
  2012-05-11/
    velodyne_sync/
    groundtruth_2012-05-11.csv
```

Or separate `velodyne_data/YYYY-MM-DD_vel/velodyne_sync/` and `ground_truth/groundtruth_YYYY-MM-DD.csv` trees.

`velodyne_sync` binary uses packed `[uint16 x, uint16 y, uint16 z, uint8 intensity, uint8 laser]` (8 B/point), `xyz_m = uint16 * .005 - 100`. The project decoder returns synchronized **body-frame XYZ**. NCLT GT CSV rows are `utime_us, x, y, z, roll, pitch, yaw`; the pose is applied as active `Rz(yaw) @ Ry(pitch) @ Rx(roll)` plus translation in the **NCLT metric frame (Z down)**. Nearest ground-truth timestamps with gaps above 100,000 µs are excluded by default. There is **no GT interpolation**, no GNSS georeferencing and no original scan intensity retention.

## 1. Prepare immutable source scans

```bash
python -m pip install -r requirements_step11.txt

python run_nclt_correction.py prepare \
  --data-root data \
  --sessions 2012-03-17 \
  --output results/step11_1_prepared \
  --scan-stride 20 \
  --max-scans-per-session 64 \
  --corrupt-from 16 \
  --corrupt-count 8 \
  --translation 1.0 0 0 \
  --yaw-deg 2.0
```

For a second session, specify `--sessions 2012-03-17 2012-05-11` and assign a distinct `robot_id` to each session. That simulates multiple sources contributing to the same global map; it is **not** validation of concurrent multi-robot SLAM.

Selection is sorted chronologically per session and subsampled by `scan_stride`. The corruption window indexes **accepted scans across the provided sessions**, starting at zero. A source retains its full microsecond timestamp and original SHA256. In the correction engine, `observed_at` is integer UTC seconds; the exact microseconds remain in the provenance ledger and manifest.

### Prepared files

```text
results/step11_1_prepared/
  scans/*.npz                 # filtered immutable body-frame float64 XYZ
  manifest_initial.json       # initially perturbed source-to-global SE(3)
  manifest_corrected.json     # unperturbed nearest-NCLT-GT SE(3)
  corrections.json            # source ID -> reference 4x4 pose (selected sources)
  scan_provenance.csv         # per-source timestamps, GT gaps, point counts
  metadata.json               # frame, preprocessing, exact config and selection
```

For larger scans, `--max-points-per-scan 10000` performs **deterministic thinning of filtered source points**. Set zero (default) to retain *all* filtered points. This is an evaluation preprocessing choice; it must be held fixed between the corrected and uncorrected map.

Range filtering defaults to 2–80 m in the body frame, preserving the earlier NCLT pipeline's conventions. Keep the configuration identical for the full-rebuild and incremental methods.

**Caution:** The prepared XYZ files plus core SQLite sources duplicate local-point storage intentionally for reproducibility. Measure and report their bytes separately from the map's immutable pages. The source is preserved as filtered XYZ, not a complete original NCLT packet; the original scan file/hash is tracked for traceability. Do not claim lossless retention of raw LiDAR intensity, laser IDs or original packet bytes.

## 2. Run the controlled correction experiment

```bash
python run_nclt_correction.py experiment \
  --prepared results/step11_1_prepared \
  --output results/step11_1_experiment \
  --voxel-size 0.5 \
  --batch-size 16 \
  --repeats 3
```

This command:

1. Builds `initial_map.sqlite` from prepared scans with the **deliberately incorrect** poses.
2. Records each scan's immutable original point data and provenance in the database.
3. Computes the exact full corrected endpoint-voxel union from all original source points and reference poses.
4. Clones the original database **outside the measured correction operation** and corrects only selected sources within one atomic transaction; overlapping sources are counted rather than accidentally removed.
5. Compares the exact sorted `int64` global voxel coordinates with the independent full rebuild; verifies the historical original revision is still accessible.
6. Optionally creates a **fresh, fully committed corrected SQLite database** from the exact same source points, verifies matching geometry, and times complete materialization.
7. Saves `corrected_map.sqlite` with the correction committed as a new revision, plus trials/summary/metadata.

Generated output:

```text
results/step11_1_experiment/
  initial_map.sqlite
  corrected_map.sqlite
  trials.csv
  summary.json
  metadata.json
```

**Important timing boundaries:**

- `incremental_commit_ms`: one atomic correction transaction plus bitmap-page changes in SQLite; clone/setup excluded.
- `full_union_rebuild_ms`: independently transform **all** sources and calculate occupied voxel union in memory, excluding disk materialization.
- `full_disk_materialization_ms`: from scratch, read the prepared source NPZ data and create/commit a new corrected SQLite store. This includes ingestion and disk work and is therefore a different cost boundary from a warm increment.
- `incremental_retrieval_ms`: query the exact corrected occupancy after commit.

The two full rebuild numbers complement each other, but neither by itself is a perfectly normalized end-to-end speedup against an online map system. Future benchmarks should include a warm in-place full-recompute-and-commit baseline on the same storage engine, matched cache state, and varied changesets.

On a large run, `--skip-full-disk-rebuild` skips the expensive fresh-DB materialization (but still checks exact in-memory full reconstruction). Use distinct output paths because commands refuse to overwrite existing experiment databases.

## 3. Inspect original versus corrected history

The `summary.json` file reports `persisted_original_revision` and `persisted_corrected_revision`; use those values:

```bash
python run_correction_map.py stats --database results/step11_1_experiment/corrected_map.sqlite

python run_correction_map.py log --database results/step11_1_experiment/corrected_map.sqlite

python run_correction_map.py query \
  --database results/step11_1_experiment/corrected_map.sqlite \
  --revision 4 --bounds -300 300 -300 300 -100 100

python run_correction_map.py verify --database results/step11_1_experiment/corrected_map.sqlite
```

`--bounds` are signed **integer voxel coordinates**, not meters. Revision 4 above is illustrative; use `summary.json` rather than assuming a fixed revision number.

## 4. Test correctness and reproducibility

```bash
python -m pytest -q \
  tests/test_step111_nclt.py \
  tests/test_step11_correction.py \
  tests/test_step105b_geometry.py
```

For a full regression, run the Step 10.0–10.5B and Step 11.0–11.1 test files together.

Test gates: source count and identity, pose-match timing, global transform correctness, no fabricated provenance, preserved historical map revision, exact bit-level occupied-set equivalence, no lost overlapping scan contributions, and atomic correction semantics.

## 5. Research role and experimental roadmap

**Where Step 11.1 sits:**

`M0: frozen synthetics -> M1: *real scan correction and exact full-rebuild equivalence* [THIS STEP] -> M2: dynamic indexing/efficient local corrections -> M3: controlled BST-ID/Morton/hash/bitmap comparison -> M4: same localization performance -> paper`

### Questions this step answers

- Can we preserve original observations and their pose dependencies for each real scan?
- Can a late correction move only its own contributions while retaining overlapping scans?
- Do post-correction committed pages reconstruct the exact same occupied map as a full correction-aware rebuild?
- Can we retrieve the historically *incorrect* previous revision and the corrected current version without revision contamination?
- How many tiles must be touched per correction relative to the total map, and how does this change with source count and correction size?

### Questions it does NOT answer yet

- Is BST-ID faster than Morton/bitmap/hashing? **Not tested.** BST-ID is intentionally out of the correction critical path at Step 11.1.
- Are corrected map poses obtained from a real loop closure? **No.** Corrections reverse controlled perturbations against NCLT reference poses.
- Does localization remain accurate? **Not tested.** Current endpoint-only pages do not reproduce raw-point surfel normals/covariances.
- Are observations from physically distinct robots in a shared georeferenced frame? **Not proven.** Sessions simulate multi-source overlap; NCLT local GT frame is assumed consistent.
- Is `observed-history` optimized for bitemporal queries? **No.** Reference full-rebuild implementation remains.

### Next scientific experiments (Step 11.2)

1. Fix a correction subset (e.g., 8 scans) while growing ingested sources: 32, 64, 128, 256, and more.
2. Sweep correction spatial extent, shift/yaw magnitude, correction batch sizes, and overlapping-source density.
3. Add an **in-place full rematerialization** control with the same SQLite durability and data pages, rather than comparing unlike timing boundaries.
4. Compare a spatial hash, Morton hierarchy, packed bitmap index, and dynamic BST-ID over identical source-dependency events.
5. Report total DB and source-provenance bytes, P50/P95 commit latency, touched/modified tile fractions, exactness, and cold/warm query latency.
6. Reintroduce source-level sufficient statistics if localization accuracy must be evaluated; voxel-center covariance is not a replacement for original surfel moments.

**Research outcome:** Step 11.1 validates the *correctness and measurement apparatus* for correction-aware lifelong mapping. It is the essential prerequirement for a credible data-structure speedup claim, not itself a proof of novelty.
