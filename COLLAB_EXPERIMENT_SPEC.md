# Collaborative BST-ID mapping: feasibility screen / frozen plan

## Scope
Coarse GNSS/GPS position is available to every robot, with user-selectable horizontal 1σ uncertainty and a separate shared-frame alignment uncertainty. Each robot owns an independent, chronologically versioned endpoint-voxel map. A receiver requests a geometric ROI and maintains a private compressed-blob cache. Inter-robot transforms are **assumed known for this experiment**; no map-frame optimization, scan matching, or actual ROS 2 transport is performed.

## Frozen first-stage comparison

- Voxel ROI payload, zlib+uint64, content-addressed response.
- Strong content-addressed versioned voxel tile baseline.
- K4 exact occupied block ROI, compared with identical Morton and BST selectors.
- Strong content-addressed K4 block tile baseline, compared with identical Morton and BST selectors.

All methods must reconstruct byte-for-byte-identical sorted canonical voxel IDs in each requested region. Both K4 selector variants must serialize bit-identical packets given equal cache state. There is no geometric approximation in this Stage 1 experiment, but original laser-point geometry has **already been voxelized**.

## Primary endpoints

- Total **response** bytes per full replay (packet, metadata, cache misses, compressed payloads).
- Median and 95th percentile local sender+receiver CPU time.
- Simple modeled serial-link total latency under 0.25, 0.5, 1, 2, 5, 10, 25 Mbps; no physical network claim.
- Receiver voxel correctness for all requests; GPS ROI geometric enclosure rate and requested-versus-true endpoint coverage.
- Per-representation stored bytes and index bytes.

## Conditional first-stage pass

1. Exact map content reconstructed on every trial, for every method.
2. >=30% lower cumulative response bytes compared with cached voxel tiles.
3. No major regression in modeled end-to-end latency at intended network bandwidth.
4. K4 BST-vs-Morton comparison reported separately. If they send identical payloads and differ only in uncertain CPU timings, do **not** claim a BST-specific improvement.

## Paper-grade gate still required

Run on multiple independently recorded robot datasets, with local GPS perturbation, uncertain map-frame registration, real network emulation (loss, latency, disconnects), map-version/loop-correction updates, and scan-to-map localization using actual held-out LiDAR scans. Compare to a competent octree/voxel-tile implementation at matched map information quality. Require >=95% of baseline localization success (provisional gate); quantify mapping completeness and synchronization correctness. Compare CPU, receiver RAM, on-disk build cost, and steady-state map update costs. This stage is **not** complete.

## Leakage warning

The Step 10.5B `Geometry4D` in the upload builds a shared multi-version geometry catalog from the supplied complete history. It selects the correct historical version during queries but its geometry construction is not causally online. To claim online collaborative mapping, run incremental append and update timing without inspecting future versions; then test inter-robot frame-transform corrections.
