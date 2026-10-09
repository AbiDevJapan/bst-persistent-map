# Step 9.1.1 — Geometry-aware dynamic-resolution localization overlay

## Goal

Step 9.1 showed that one surfel per native Step-8.5 block reduces local primitives and total localization time substantially, but some localization quality is lost because occupancy/time-compatible blocks can still contain geometry that is not well represented by one plane.

Step 9.1.1 asks a narrower question:

> Can we recover voxel-surfel localization quality by retaining native block surfels only where one plane is a good geometric approximation, while falling back to the frozen 0.5 m voxel surfels only inside complex native blocks?

This is a **localization overlay** over the unchanged lifelong BST map. It does not change the Step-8.5 temporal representation, histories, compact codec, or `(bucket, session)` query index.

## Why the implementation is two-level

A fully recursive geometry-aware dyadic split is possible, but it adds more build logic and repeated geometry tests. Step 9.1.1 deliberately uses the cheaper first experiment:

- keep an existing native adaptive block as **one block surfel**, or
- fall back to the exact **0.5 m voxel surfels contained by that same native parent**.

This isolates whether geometry-aware dynamic resolution is useful before adding intermediate split levels.

The decision is made **offline once**. Localization never runs PCA, threshold search, recursive splitting, or block decompression decisions.

## Geometry gate

For every native map-session block, four signals are precomputed once:

1. **Absolute plane thickness**

   `sigma_perp = sqrt(lambda_min(Sigma))`

   This prevents a physically thick / poorly fitted surface from appearing planar merely because the block is large.

2. **Relative planarity**

   `E_plane = lambda_min(Sigma) / trace(Sigma)`

3. **Sign-invariant normal coherence**

   For valid base-voxel normals `n_i`, Step 9.1.1 forms

   `C_n = mean(n_i n_i^T)`

   and uses its largest eigenvalue. It approaches 1 when the voxel normals agree up to sign and drops when normals are dispersed. This catches corners / mixed surfaces without an online child-normal comparison.

4. **Maximum physical block extent**

   Very large blocks can be limited even if their covariance is globally planar.

A block is kept native only when it passes all profile thresholds. A one-voxel native block is always retained because it cannot be refined further.

## Default Pareto sweep

The defaults are **development settings**, not final tuned constants:

| profile | sigma max | relative max | normal coherence min | max extent |
|---|---:|---:|---:|---:|
| strict | 0.025 m | 0.010 | 0.985 | 1.0 m |
| balanced | 0.050 m | 0.025 | 0.950 | 2.0 m |
| moderate | 0.080 m | 0.050 | 0.900 | 4.0 m |
| permissive | 0.120 m | 0.080 | 0.800 | 8.0 m |

Do **not** interpret the names as claims that one setting is optimal. The purpose is to produce a quality/compression/runtime curve.

Profiles can be replaced from the CLI with:

`--geometry-profiles 'name:sigma:relative:coherence:extent;...'`

## Query path

Every geometry-aware profile still starts with the same Step-8.5 time-aware query:

`(spatial bucket, session) -> active native BST block IDs`

Then each active native parent contributes either:

- one native block surfel, or
- its pre-associated base voxel surfels.

The voxel-to-native-parent relation is precomputed offline once. There is no block-to-voxel materialization inside the localization loop.

## Why 20 scans now

Step 9.0.3A/B and Step 9.1 used 10 scans as feasibility tests. Step 9.1.1 increases the development set to **20 cross-session scans** by default (plus 20 same-session zero controls).

These 20 scans are still a **threshold-tuning set**, not the final statistical evaluation. After selecting an operating point, the chosen profile must be evaluated on a larger **held-out** scan set and additional session pairs. Do not report the 20-scan threshold-selection result as unbiased final performance.

## Recommended run

Reuse the Step 9.0.2 replay list and the five existing Step-8 caches:

```bash
python3 run_step9_1_1.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --query-scans-csv results/step9_0_2_controls/query_scans.csv \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --query-count 20 \
  --same-session-query-count 20 \
  --perturbations 0.5:1 \
  --localization-max-range 30 \
  --scan-downsample 0.75 \
  --max-scan-points 8000 \
  --local-map-radius 40 \
  --output results/step9_1_1_geometry_aware
```

No additional NCLT download is required.

## Benchmarked methods

- `voxel_surfel_0.5m` — conventional voxel-surfel reference.
- `bst_native_block_surfel` — unchanged Step-9.1 native-block reference.
- `bst_geom_strict`
- `bst_geom_balanced`
- `bst_geom_moderate`
- `bst_geom_permissive`

The Step-9.1 decompressed compatibility control is not rerun by default because Step 9.1 already established trial-for-trial equivalence. This keeps the larger 20-scan sweep focused on the new variable.

## Outputs

- `localization_trials.csv`
- `localization_summary.csv`
- `query_scans.csv`
- `geometry_profile_stats.csv`
- `pareto_summary.csv`
- `metadata.json`

`geometry_profile_stats.csv` reports, per profile:

- selected localization primitives;
- fraction of native parents kept coarse;
- number of fallback parents and fallback voxels;
- selected geometry payload estimate;
- standalone refinement-directory estimate;
- unchanged BST representation bytes;
- mean voxel-equivalent cells per selected primitive;
- all gate thresholds.

`pareto_summary.csv` focuses on the cross-session `0.5 m / 1 deg` development case and reports:

- success rate and ratio to voxel baseline;
- whether the profile reaches the Step-9.1 `>= 90% of voxel success` gate;
- primitive reduction;
- total-time speedup;
- median pose errors;
- simple Pareto flags for success-vs-primitives and success-vs-total-time.

## Memory accounting note

For threshold sweeps the implementation keeps the full voxel-surfel table and native-block surfel table resident once, then represents each profile as a boolean mask. This avoids physically duplicating several hundred MiB for every candidate profile.

The reported per-profile payload is the **standalone selected payload** that would be needed if that operating point were serialized for deployment. The metadata separately records the shared sweep-only voxel-to-parent mapping memory.

## Interpretation

The desired result is not necessarily the most aggressive profile. Look for a Pareto point that:

1. recovers approximately the voxel-surfel success / pose accuracy;
2. retains a large fraction of the Step-9.1 primitive-count and total-time savings;
3. does not require a large fallback payload.

If a two-level profile recovers localization quality efficiently, recursive intermediate spatial levels are unnecessary complexity. If all useful operating points fall close to full voxel fallback, then Step 9.1.2 should test true multi-level geometry-aware splitting.

## Scientific caveats

- This step does **not** alter the lifelong representation; it is a localization-specific dynamic-resolution view over the frozen Step-8.5 map.
- The 20 replayed cross-session scans are used for threshold development. The selected profile requires held-out validation before a paper claim.
- The matcher remains a local refinement method; global relocalization is out of scope here.
- `PrototypeBSTSpatialCodec` remains a local research codec, not verified upstream BST-ID.
