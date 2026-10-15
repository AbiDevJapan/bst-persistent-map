# Step 10.3 — BST-ID prefix-indexed historical map queries

**What this patch adds:** a persistent, author-BST-ID-based spatial prefix index for historical tile selection, a competitive Morton-hash control, exact randomized ROI equivalence checks, and a paired latency/storage benchmark. This is an overlay **on top of the uploaded Step 10 refactor**. The unchanged localization code is not modified.

**Scope limitation:** The BST-ID labels are currently geospatial XYHT point IDs of **tile anchors**, not native metric volumetric cuboids. To avoid missed tiles, the ROI is first expanded to the anchors of **all potentially intersecting integer tiles**, then passed through the author's actual geospatial quantization and BST-ID per-axis prefixes. The index returns candidates, which are always post-filtered by the canonical integer tile bounding boxes. The original voxel-level half-open ROI filter remains authoritative. Temporal restriction is handled through immutable historical session manifests; BST-ID's 32-bit temporal payload is verified during index construction but temporal *range* traversal is not yet a native 4D BST-ID prefix query. Do NOT claim complete BST-ID 4D region algebra or compression advantage from these results.

## Install

From your existing Step 10 project root:

```bash
unzip -o Step10_3_BST_Prefix_Region_Queries.zip
python -m pytest -q tests/test_step103_index.py tests/test_step10_versioning.py
```

This patch uses the **unchanged** author's package already included at `third_party/bstid_author/bst_id`. Its loader is improved to avoid conflicts with the old project package named `bst_id`. It has no additional dependencies beyond the Step 10 project (`numpy`, `pytest`).

## 1. Build the persisted index for your existing five-session database

First build the Step 10 database as documented previously (the following does **not** rebuild NCLT or change existing map blobs):

```bash
python run_versioned_map.py index --database results/step10_official --index-mode both
```

This builds *separate*, measurable `indexes/<session>_bst_prefix.npz` and `indexes/<session>_morton_hash.npz` files for each stored session. They are loaded once per benchmark process, and are checked against a canonical SHA256 of their manifest to reject stale indexes.

Build just the Morton control for prototype- or Morton-labeled databases:

```bash
python run_versioned_map.py index --database results/step10_morton --index-mode morton_hash
```

## 2. Run identical region queries

```bash
python run_versioned_map.py query --database results/step10_official \
  --session 2012-03-17 --bounds 100 200 100 200 0 30 --index-mode scan
python run_versioned_map.py query --database results/step10_official \
  --session 2012-03-17 --bounds 100 200 100 200 0 30 --index-mode bst_prefix
python run_versioned_map.py query --database results/step10_official \
  --session 2012-03-17 --bounds 100 200 100 200 0 30 --index-mode morton_hash
```

All three must return identical `voxels`, `tiles_read`, and `compressed_bytes_read`. The returned stats additionally show `index_ms`, `keys_examined`, and `index_probes`.

**Historical change and regional sync** also accept `--index-mode bst_prefix` or `--index-mode morton_hash`, applying the same indexed selection to source and target snapshots. `history` is a new multi-session query:

```bash
python run_versioned_map.py history --database results/step10_official \
  --from-session 2012-02-02 --to-session 2012-06-15 \
  --bounds 100 200 100 200 0 30 --index-mode bst_prefix
```

The results are independent historical voxel observations; absence of an endpoint is **not** evidence of free space or physical object removal.

## 3. Paired benchmark

```bash
python run_versioned_map.py benchmark --database results/step10_official \
  --output results/step10_3_official --queries 200 --repeats 3 --seed 10303
```

If indexes have not been built, append `--build-indexes` to create them first; the associated construction time is tracked separately. Reuse existing indexes for timing-only runs. The benchmark builds 200 seeded ROIs over stored tiles plus no-hit locations, with sizes 32, 64 and 128 **voxels**. These are not meters. For a 0.5 m lattice they represent nominal 16, 32 and 64 m cube side lengths.

Each query/repeat runs three methods (`scan`, `bst_prefix`, `morton_hash`) in rotated order, after a uniform warm-up query, then asserts exact equality of BOTH selected tile keys and sorted uint64 voxel codes. On any mismatch the benchmark stops with an error. Results:

- `trials.csv` — paired per-query per-mode total and index selection times, number of candidate/touched tiles, bytes read and equivalence.
- `summary.csv` — median and P95 times and counts.
- `metadata.json` — dataset/codecs, dimensions, query protocol, build-time figures (if built), map+index bytes, machine description.

**Benchmark caveats:** Warm OS cache, non-isolated CPU and local file reads; no network bandwidth or cold-cache measurements. No full NCLT results have been run in the assistant environment. Timings on your machine must be interpreted in the context of map size. Store `disk_stats()['total_bytes']` now includes persisted index files. `array_bytes_estimate` is a lower bound on in-memory index cost: it omits Python dictionary/hash-table overhead and temporary per-row prefix caches. Prefer measured RSS when comparing systems.

## 4. What is actually new about the BST-ID fast path?

- Parses the *actual author-produced 128-bit-ish BST-ID label* (four axes with zooms 28/28/24/32) into x/y/altitude/time dyadic tiles using the author's **region algebra**.
- Builds sorted postings by **coarser coordinate prefixes** (zooms x=22, y=22, altitude=16) for each chronological manifest; the temporal label is checked against the historical session.
- Converts requested integer-voxel ROI to its exact overlapping tile-anchor range and the author's XYH quantizers (including Web Mercator's reversed Y axis), then probes prefix postings and applies exact integer tile bounds.
- Uses an independent Morton *integer tile* hash lookup as a fair high-performing alternative, rather than only benchmarking a full manifest scan.

BST-ID is **not** assumed faster a priori: the Morton hash is O(number of ROI tiles) and may be superior, especially for small regions. Quantized geospatial IDs can collide; the index stores postings of original canonical tile rows and never treats the label as a unique voxel address.

## 5. Next steps after a successful 10.3 benchmark

- 10.4: immutable per-tile content delta/update journal and exact historical reconstruction, with matched prefix and Morton controls.
- 10.5: compact binary regional synchronization, including receiver cost, payload bytes and wire checksums.
- 10.6: controlled physical changes with visibility/free-space evidence separated from endpoint-history differences.
- 10.7: full NCLT and optional 27-session comparisons to spatial snapshot/delta baselines, with independent validation sessions and complete process RSS.

**Research gate:** BST-ID must show a measurable win in total memory/bytes or indexed region/history retrieval relative to the Morton-hash **and** versioned-delta baselines. Beating only full manifest scanning is not sufficient.
