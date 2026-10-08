# Step 9.0.3B — 0.5 m voxel-surfel localization

## Goal

Step 9.0.3A showed that reducing occupancy-center resolution from 0.5 m to 0.25 m sharply reduced the zero-initialization localization floor, but increased occupied map elements and runtime memory substantially.

Step 9.0.3B tests the more useful alternative:

> keep the **0.5 m indexing lattice**, but replace each occupied voxel center with sub-voxel geometry estimated from the actual LiDAR returns that fell in that voxel.

This is still a conventional voxel-map localization baseline. **No native BST blocks are consumed yet.**

## Map payload

For each occupied 0.5 m map voxel, the builder accumulates

- `N`: raw return count,
- `mu`: mean transformed LiDAR position,
- `Sigma`: 3x3 within-voxel covariance,
- `n`: smallest-eigenvalue covariance eigenvector,
- curvature / normal-validity quality.

The occupancy codes from `.cache/step8_1` define which voxels exist. Geometry is rebuilt from the raw Jan-15 returns in memory. No additional surfel cache is written to disk.

The three localization pyramid levels are created by statistically merging these sufficient statistics. Coarser levels therefore use weighted LiDAR centroids/covariances; they never fall back to voxel centers or neighborhood-PCA normals.

## Clean comparison

The default run evaluates both:

1. `occupancy_center_0.5m` — frozen Step-9.0.2 geometry;
2. `voxel_surfel_0.5m` — same 0.5 m indexing lattice with sub-voxel geometry.

They use identical query scans, initial perturbations, map radius, ICP pyramid and success threshold.

The prior Step-9.0.3A `0.25m` result remains the fine-resolution reference.

## Recommended run

```bash
python3 run_step9_0_3b.py \
  --data-root data \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --query-scans-csv results/step9_0_2_controls/query_scans.csv \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --query-count 10 \
  --same-session-query-count 10 \
  --perturbations 0.5:1,1:3 \
  --localization-max-range 30 \
  --scan-downsample 0.75 \
  --max-scan-points 8000 \
  --local-map-radius 40 \
  --output results/step9_0_3b_surfel
```

If your replay CSV is elsewhere, point `--query-scans-csv` to it. A Step-9.0.2 mixed-scenario CSV is automatically filtered to the `cross_session` rows. Same-session off-stride controls are selected again from Jan 15.

## Outputs

- `localization_trials.csv`
- `localization_summary.csv`
- `query_scans.csv`
- `surfel_map_stats.csv`
- `metadata.json`

`surfel_map_stats.csv` reports full surfel payload memory, a minimal localization payload estimate, index memory, valid-normal fraction, occupancy-cache load time, surfel geometry build time and index-build time.

## Decision rule

The first result to inspect is `zero_control`.

A successful 9.0.3B result should move the 0.5 m surfel zero-control error substantially toward the already measured 0.25 m occupancy-center result, while retaining approximately the 0.5 m number of spatial elements.

Then inspect `xy0.5_yaw1` and `xy1_yaw3` to see whether the improved objective also improves capture/recovery, rather than only reducing the final geometry bias.

Do not change the Step-8 representation results to surfels retroactively. Step 9.0.3B is a localization-payload experiment layered on the frozen 0.5 m spatial index.
