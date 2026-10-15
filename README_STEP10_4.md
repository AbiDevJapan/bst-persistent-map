# Step 10.4 — Exact incremental per-tile delta versioning

**Apply on top of Step 10.3.** This patch adds *self-contained*, appendable historical delta archives without changing the original versioned snapshot database, author BST-ID implementation, Step 10.3 query engines, or Step 9 localization code.

## Installation

From the root of the working Step 10.3 checkout:

```bash
unzip -o Step10_4_Incremental_Delta_Versioning.zip
python -m pytest -q tests/test_step104_deltas.py tests/test_step103_index.py tests/test_step10_versioning.py
```

The patch replaces `run_versioned_map.py` and adds two `versioned_map/` modules, tests and this README. Dependencies remain Python, NumPy, and the existing installed Step 10 dependencies. The system uses zlib and SHA256 from the standard library. No retraining or relocalization is involved.

## 1. Build Morton and official BST-ID delta archives

Reuses the *existing* complete five-session snapshot store. No need to rebuild NCLT scans or overwrite the snapshot store.

```bash
python run_versioned_map.py delta-build \
  --database results/step10_official \
  --journal-root results/step10_4_journals \
  --address-mode both
```

Results:

- `results/step10_4_journals/morton/`
- `results/step10_4_journals/bst/`

Each is an independent archive containing `base.json`, `metadata.json`, immutable compressed `blobs/`, and chronological `transitions/<from>__<to>.json`. The base contains the first map's exact tile payloads. Each subsequent version stores only per-tile observed-endpoint `added` and `removed` uint64 voxel codes. Bytes are zlib6-compressed and content-addressed by the SHA256 of the compressed blob. Unchanged tiles store **no transition entry**. Empty-after-update tiles are represented correctly by removals.

A source snapshot store is required to BUILD deltas, but not to QUERY already-built delta journals.

## 2. True incremental append: January -> February -> March -> May -> June

Optionally create a journal containing only the first version, despite the source snapshot store already containing all sessions:

```bash
python run_versioned_map.py delta-build \
  --database results/step10_official \
  --journal-root results/step10_4_incremental \
  --address-mode both \
  --through-session 2012-01-15
```

Then append new sessions in chronological order:

```bash
for s in 2012-02-02 2012-03-17 2012-05-11 2012-06-15; do
  python run_versioned_map.py delta-append \
    --database results/step10_official \
    --journal-root results/step10_4_incremental \
    --address-mode both --session "$s"
done
```

`delta-append` enforces consecutive chronology, source manifest integrity, and identical map-frame metadata. It updates the new transition and metadata without rewriting any prior transition. The journal is self-contained after each successful append. Build and append produce per-transition changed tiles, newly/no-longer observed endpoints, unique compressed bytes added, actual transition JSON bytes, write bytes, and build time. The journal directory must be new; files are not silently overwritten.

**Note:** This Step 10.4 path ingests *existing* NCLT session snapshots as its input. Direct online LiDAR scan-to-delta ingestion will be a separate system integration. Transitions are calculated on the canonical 0.5 m observed-endpoint grid, not the frozen lossy K=4 localization surfels.

## 3. Historical reconstruction

```bash
python run_versioned_map.py delta-query \
  --journal results/step10_4_journals/bst \
  --session 2012-05-11 \
  --bounds 722 786 965 1029 76 140 \
  --save-npy results/step10_4_region.npy
```

All bounds are six half-open **integer voxel** coordinates: X0 X1 Y0 Y1 Z0 Z1. The query loads only spatially overlapping base and transition tiles, replays the changes through the requested version, and filters by exact integer voxel coordinates. Internal spatial tile lookup is currently a canonical tile directory scan; the BST-ID address is verified and stored but **not yet** the runtime 4D prefix traversal.

Validate exact reconstruction on seeded regional queries:

```bash
python run_versioned_map.py delta-verify \
  --database results/step10_official \
  --journal-root results/step10_4_journals \
  --queries 200 --seed 10404
```

For stronger validation, include entire 5-session maps (may use substantial RAM and time):

```bash
python run_versioned_map.py delta-verify \
  --database results/step10_official \
  --journal-root results/step10_4_journals \
  --queries 200 --seed 10404 --full-maps
```

All reconstruction uses exact sorted uint64 code equivalence. A corrupt compressed blob triggers an error rather than returning inaccurate results.

## 4. Paired benchmark

```bash
python run_versioned_map.py delta-benchmark \
  --database results/step10_official \
  --journal-root results/step10_4_journals \
  --output results/step10_4_official \
  --queries 200 --repeats 3 --seed 10404
```

**Three matched paths for every query:**

1. **`snapshot`** — the current exact content-addressed full-tile snapshot database.
2. **`morton_delta`** — exact base + additions/removals, with conventional Morton+UTC time address strings.
3. **`bst_delta`** — exact same base + changes, with actual official BST-ID x/y/elevation/time labels.

Any disagreement in output voxel codes raises an error. Each query measures latency, compressed data accessed, candidate tiles, and transition records applied. Rotated timing order reduces method-order bias. No decoded blobs are cached inside the delta query implementation (for memory fairness; OS page caches are not disabled).

Outputs:

- `trials.csv`: query-level paired timings and exactness fields.
- `summary.csv`: P50/P95/mean timings and accesses.
- `growth.csv`: cumulative recorded bytes after each historical session, for four approaches.
- `storage.json`: actual measured existing store bytes, modeled independent snapshot bytes (summing real compressed blobs per version), actual delta journal bytes, and per-transition update costs.
- `metadata.json`: protocol, machine and attribution caveats.

Inspect current archived storage without rerunning timing:

```bash
python run_versioned_map.py delta-stats \
  --database results/step10_official \
  --journal-root results/step10_4_journals
```

## 5. Important experimental interpretation

**What Step 10.4 demonstrates:** an exact, self-contained, appendable *version-control architecture*, with measured per-transition delta costs and paired regional query correctness. It explicitly handles both additions and disappearances from *observed endpoints*, without labeling absent endpoints as physically free space.

**What it does NOT demonstrate:** faster 4D BST temporal range traversal, binary on-wire serialization, physical change detection, synchronized receiver timing, or inherent compression superiority of BST-ID. Both delta modes have **identical** compressed voxel payloads, and differ only in JSON address strings. Any measured byte differences between those two journal variants therefore measure address serialization/metadata overhead, **not** a BST-ID intrinsic delta-compression benefit. Even if Morton is smaller or faster, this is a valid outcome.

The independent-snapshots baseline is a byte model based on the exact stored zlib6 blobs summed once per version. The content-addressed-snapshots and delta-journal totals are actual file sizes, excluding filesystem allocation overhead. `growth.csv` uses final metadata size at each intermediate point, slightly overcounting early versions.

This is the foundation for Step 10.5: **compact binary regional synchronization packets**, including receiver application/correctness, bytes transferred, and a fair Morton/Hilbert or standard binary delta baseline.
