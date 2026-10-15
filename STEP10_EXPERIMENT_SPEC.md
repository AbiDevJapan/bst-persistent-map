# Step 10.0 frozen evaluation specification

**Hypothesis:** explicit spatial/temporal identifiers can reduce map versioning and regional synchronization cost beyond well-engineered versioned voxel/Morton baselines.

## Fixed data
NCLT chronological sessions: 2012-01-15, 2012-02-02, 2012-03-17, 2012-05-11, 2012-06-15. Use the same registered NCLT local frame, 0.5m endpoints, point/range filtering, scan stride and ROI for all methods. No pose estimation inside the map-store benchmark. No held-out scan may be inserted into its own historical version for downstream localization tests.

## Primary correctness
Full-session exact canonical uint64 voxel reconstruction; randomized half-open integer grid region queries; exact set differences for observation changes; byte-for-byte-equivalent receiver-region reconstructions after packet application; corruption rejection; cross-version monotonic chronology. **Absence is unknown**, not evidence of free space.

## Baselines
1. Independent snapshots (same uint64 voxel sets); 2. conventional content-addressed versioned tile store; 3. Morton-index labels + same tile/delta store; 4. author's BST-ID index labels + same store; 5. original prototype labels + same store. Adaptive K=4 compressed surfels constitute a *separate lossy geometry ablation* with error metrics, not a fair exact equivalence comparison.

## Metrics
Compressed on-disk blob bytes, manifest/index bytes, RSS/peak RSS, ingest CPU and wall time, incremental new bytes, query P50/P95, number of tiles examined, bytes read, region delta packet bytes, sender encoding and receiver decode/apply times, exact set equivalence, repeat three runs on same hardware. Distinguish cold disk and warm page cache and quantify index-build time. Reserve additional NCLT sessions as validation after thresholds are fixed.

## Important limitations of current Step 10 foundation
The supplied author's codec encodes **longitude / latitude / elevation / Unix time**, not native metric NCLT blocks. This patch uses a documented local planar coordinate frame and uses IDs as **tile metadata labels**. It does NOT yet accelerate spatial queries via upstream region algebra or make these IDs directly replace the legacy dyadic 3D prefix codec; exact ROI filtering is still integer-grid-based. Thus *do not attribute any storage, query, or synchronization advantage to BST-ID yet*. Further Step 10.1 must implement correct region cover/prefix filtering, zoom selection, and evaluate information loss/collisions under NCLT resolution. Current snapshot/delta storage is a benchmark foundation; full comparative index experiments are not yet implemented.
