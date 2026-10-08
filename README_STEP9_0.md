# Step 9.0 — Conventional voxel localization harness

Step 9.0 starts the localization phase **without using native BST blocks yet**. Its purpose is to validate the scan-to-map benchmark, pose perturbation protocol, map retrieval, optimizer, timing decomposition, and output format before Step 9.1 changes the map geometry interface.

## What is deliberately frozen

- Map: conventional 0.5 m voxel occupancy from one NCLT session.
- Query scans: synchronized Velodyne scans from a different NCLT session.
- Initialization: ground-truth pose plus deterministic x/y/yaw perturbation.
- Local map retrieval is centered on the **perturbed initial pose**, not GT.
- Optimized state: x, y, yaw only.
- z, roll, pitch are held at their reference values in 9.0.
- Matching: trimmed point-to-point ICP with 3-D nearest-neighbor correspondences and planar SE(2) updates.
- This is a harness-validation localizer, not the final localization contribution.

## Why 3-DOF first?

NCLT is a ground-vehicle dataset. Holding z/roll/pitch fixed lets Step 9.0 isolate whether the conventional local-map retrieval and registration loop can recover known planar perturbations. Step 9.1/9.2 can then test native blocks without simultaneously debugging a full 6-DOF backend.

## No new dataset download is needed

The default sessions are the first two already used in Steps 8.1–8.5:

- map: `2012-01-15`
- query: `2012-02-02`

The map preprocessing reuses `.cache/step8_1` when the ingestion configuration matches.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Step 9.0 adds SciPy for `scipy.spatial.cKDTree`.

## Recommended run

```bash
python3 run_step9_0.py \
  --data-root data \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --query-count 20 \
  --perturbations 0.5:1,1:3,2:5,5:10 \
  --localization-max-range 30 \
  --scan-downsample 0.75 \
  --max-scan-points 8000 \
  --local-map-radius 40 \
  --icp-max-correspondence 2.0 \
  --output results/step9_0_localization
```

The perturbation format is `translation_magnitude:yaw_deg`. For every scan/level, translation direction and yaw sign are deterministic from `--perturb-seed`, but vary across trials.

## Outputs

`localization_trials.csv`
: One row per scan × perturbation. Includes initial/final pose error, convergence, correspondence count, RMSE, local map size, and timing.

`localization_summary.csv`
: Aggregated success/error/timing by perturbation level.

`query_scans.csv`
: Exact scan filenames/timestamps and GT rows chosen for the benchmark. Keep this file: later Step 9.x experiments should reuse the same scans.

`metadata.json`
: Complete configuration, map/query sessions, success thresholds, optimizer settings and scientific caveats.

## Timings

Each trial separates:

1. `retrieval_ms` — retrieve conventional local voxel centers around the initial pose.
2. `map_prep_ms` — build the local `cKDTree`.
3. `match_ms` — iterative scan-to-map matching only.
4. `total_ms` — retrieval + preparation + matching/overhead.

This separation is intentional. Step 9.2 will compare the same stages for voxel and native-block map interfaces.

## Success criterion

Default success is:

- final planar translation error <= 0.5 m, and
- final yaw error <= 1.0 degree.

For Step 9.0, the most important go/no-go levels are `0.5 m / 1°` and `1 m / 3°`. `2 m / 5°` and especially `5 m / 10°` are basin-of-convergence stress tests, not mandatory for advancing to Step 9.1.

Suggested go/no-go:

- >= 80% success for `0.5:1` and `1:3`, and
- median final error <= 0.5 m / 1° for those levels.

If that fails, fix the localization harness before introducing BST/native-block geometry.

## Scientific limitations

- This is not full SLAM.
- It is not yet full 6-DOF localization.
- GT/reference z, roll and pitch are used and held fixed.
- The query session differs from the map session, but NCLT GT is used to create the controlled initialization and evaluate final error.
- The map is conventional voxel occupancy. No BST/native-block localization claim is made in Step 9.0.
- Occupancy retains the same Step 8 semantics: absence is not interpreted as observed free space.
