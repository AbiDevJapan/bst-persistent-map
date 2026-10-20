# BST-ID distributed mapping feasibility v0.1

## What this *actually* tests

A coarse GPS prior defines map regions requested by one simulated robot from another. Each sender uses the **uploaded source code's** `VersionedStore` and `Geometry4D` (Step 10.5B), and the receiver reconstructs its selected remote voxel endpoint region. Every returned map is compared against an untimed exact reference. Regions with unknown occupancy are not treated as free.

Methods are:

1. `voxel_tiles`: conventional immutable zlib-compressed voxel tiles, content-addressed **across requests and versions** (strong tiled baseline).
2. `voxel_roi`: exact u64 voxel codes clipped to ROI and compressed, cached by ROI content hash.
3. `k4_morton`: exact K4 occupied blocks selected using existing Morton mode, compressed and cached by ROI content hash.
4. `k4_bst`: **identical** K4 occupied blocks and same wire format, with the existing BST 4D index tile selector.
5. `k4_tile_morton` and `k4_tile_bst`: full K4 block **tiles** with content-hash reuse across region requests and versions; isolates ROI clipping and cache behavior from geometry compression. These two modes differ only in tile selector.

The *same* geometric representation is used for both K4 modes. If K4 transmits fewer bytes, that is a geometry representation finding, **not** by itself evidence that BST-ID indexing is superior. The tile caches intentionally benefit from partial overlap between successive ROI requests. The two K4 selector alternatives share byte-for-byte identical packets at equal receiver state. Tile and blob manifests are binary-packed rather than expanded as verbose JSON (to avoid handicapping tile-based baselines).

## Quick start (no external data)

```bash
pip install -r requirements.txt
python run_collab_feasibility.py demo --output results/collab_demo \
  --robots 3 --epochs 3 --requests 36 --gps-sigma 3 --local-radius 6
pytest -q tests/test_collab_feasibility.py
```

The demo generates 3 independent robot-owned, partially overlapping endpoint maps, each with changing coverage across three chronological versions. It uses a shared *perfect* global frame for the underlying geometry but noisy GPS for region selection. It does **not** produce real sensor scans or run SLAM. Map build time is not included in request latency (see Step 10 benchmarks for build accounting).

## Use your existing Step 10 NCLT maps

```bash
python run_collab_feasibility.py existing \
 --database /path/to/versioned_store \
 --geometry /path/to/step10_5b_K4_geometry \
 --output results/collab_nclt --requests 200 --gps-sigma 3
```

To create these databases from NCLT data using the existing script:

```bash
python run_versioned_map.py build --database results/nclt_store \
  --data-root /path/to/NCLT --codec morton
python run_versioned_map.py geometry-build --database results/nclt_store \
  --geometry-root results/nclt_k4 --k 4
```

The Step 10 `geometry-build` command builds the K4 data from a Step 10 `build` source if necessary. Both source and geometry must have **all matching session versions** and K=4. This NCLT mode is a **peer-replay proxy**; it is not simultaneously recorded multi-robot footage. Requesters are simulated, and no NCLT files are bundled.

A prepackaged three-robot test (3 versions, 36 requests, GPS σ=3 m, local radius 6 m) yielded 42.5% fewer response bytes for K4 BST ROI than content-addressed voxel tiles, **while K4 BST and K4 Morton sent identical bytes**. This is a **synthetic microbenchmark** and does not establish a publishable improvement yet. The measured first-stage wire savings depend on the ROI size, map structure and cache behavior.

The benchmark produces `trials.csv`, `summary.csv`, `bandwidth_sweep.csv`, `metadata.json`, and `report.md`. It logs real local retrieval/encode/decode CPU timings and actual serialized wire bytes. `modeled_link_ms` uses an ideal 2 Mbps serial transmission formula and is **not a measured network transfer**.

## Limitations and decision boundary

- No real LiDAR scan-to-map localization, so **no ATE or localization success claim**. Exact voxel consistency is a prerequisite, not a substitute for localization accuracy.
- No physical GPS data, map-frame drift, inter-robot registration, pose-graph optimization, or real networking.
- Receiver caches have no eviction and do not measure their own RSS; metadata holds complete ROI boxes and hashes. This can bias long-lived behavior.
- ROI packets are whole-ROI snapshots, not sub-block incremental updates. The tile baseline has content-hash reuse at tile granularity; BST ROI does not yet have equal reuse at block granularity.
- Current geometric cells are exact endpoint voxel boxes, not plane surfels. The official BST-ID Cell algebra indexes a local voxel frame for this Step 10 experiment; no claim of geographic coordinate accuracy is made.
- Geometry4D is prebuilt from full session history. This is a storage/indexing replay rather than online causal training/update timing.

A positive first-screen result requires >=30% reduced total wire bytes against content-addressed `voxel_tiles` with 100% exact region reconstruction and reasonable CPU. A stronger paper-grade claim additionally requires downstream real LiDAR localization accuracy retention (proposed >=95%), actual transport tests, multiple independently recorded robots, and full storage/rebuild and update costs.
