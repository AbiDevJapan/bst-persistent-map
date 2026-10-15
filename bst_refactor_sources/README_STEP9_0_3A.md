# Step 9.0.3A — Voxel-size localization diagnostic

## Purpose

Step 9.0.2 showed that the conventional occupancy-center localization objective has a non-trivial pose floor even from exact ground-truth initialization. Step 9.0.3A tests whether that floor is materially caused by the chosen occupancy voxel size.

This is **not** a new BST experiment and does not change the Step-8 representation results. It changes only the conventional localization map resolution.

## Controlled variable

Test occupancy maps at multiple base resolutions while keeping the following fixed in physical units:

- query scan identities;
- query scan downsampling;
- point-to-plane optimizer;
- correspondence radii;
- normal-estimation radii;
- local-map radius;
- physical coarse-bucket width;
- perturbation RNG and success thresholds.

The default resolutions are executed in this order:

```text
0.5 m, 1.0 m, 0.25 m
```

The 0.5 m baseline is run first because it should reuse the existing Step-8.1 cache; the potentially largest 0.25 m map is run last. Results are checkpointed after each completed resolution.

## Important pyramid detail

The Step-9.0.2 fine map level was 0.5 m. Leaving it fixed would downsample a 0.25 m occupancy map back to 0.5 m and invalidate the ablation.

Step 9.0.3A therefore scales only the **map-side** pyramid with the tested base voxel size:

| Base map | coarse map | medium map | fine map |
|---:|---:|---:|---:|
| 0.25 m | 0.50 m | 0.375 m | 0.25 m |
| 0.50 m | 1.00 m | 0.75 m | 0.50 m |
| 1.00 m | 2.00 m | 1.50 m | 1.00 m |

The scan-side pyramid remains 1.5 / 1.0 / 0.75 m, and the physical correspondence and normal radii remain unchanged.

## Low-disk cache policy

The default is:

```text
--cache-policy reuse-only
```

This reuses an exact existing ingest cache if present (especially the 0.5 m Step-8.1 cache) but **does not write new caches** for 0.25 m or 1.0 m. Each resolution-specific map is built in memory, benchmarked, checkpointed to CSV/JSON, and released before the next map is built.

Use `--cache-policy read-write` only if you intentionally want to keep the new voxel maps on disk.

## Workload

Part A intentionally focuses on the diagnostic cases:

- cross-session zero perturbation — primary resolution-floor measurement;
- cross-session 0.5 m / 1°;
- cross-session 1.0 m / 3°;
- same-session off-stride zero perturbation.

The default uses 10 cross-session and 10 same-session scans to avoid rerunning the full expensive perturbation matrix during this diagnostic.

## Recommended command

```bash
python3 run_step9_0_3a.py \
  --data-root data \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --query-scans-csv results/step9_0_2_controls/query_scans.csv \
  --resolutions 0.5,1.0,0.25 \
  --scan-stride 10 \
  --query-count 10 \
  --same-session-query-count 10 \
  --perturbations 0.5:1,1:3 \
  --cache-policy reuse-only \
  --output results/step9_0_3a_voxel_size
```

A mixed Step-9.0.2 `query_scans.csv` is accepted; Step 9.0.3A automatically filters it to `cross_session` rows for replay.

## Outputs

```text
localization_trials.csv
localization_summary.csv
map_resolution_stats.csv
query_scans.csv
metadata.json
```

`map_resolution_stats.csv` reports voxel count, map-array size, index-array size, cache hit status, build/load time, physical bucket width, and map-side pyramid resolutions for each voxel size.

## Interpretation

The main comparison is the zero-control floor:

```text
voxel size  -> median final translation/yaw error from exact GT initialization
```

If 0.25 m substantially lowers the 0.5 m floor, occupancy quantization is a major limitation. If the floor remains similar, simply shrinking voxels is unlikely to solve localization and Step 9.0.3B should move to sub-voxel surfel geometry.

The 1.0 m case is a coarse negative/control condition: if its error floor increases, the experiment is sensitive to voxel quantization in the expected direction.

## Scope

Step 9.0.3A still uses occupied voxel centers and PCA normals. It does not add surfels, change BST/adaptive representation logic, or consume native BST blocks.
