# Step 8 — Real repeated-traversal validation on NCLT

Step 8 stops changing the representation. It asks whether the Step-7 winner
survives real repeated LiDAR observations.

## Main method carried forward

```text
adaptive dyadic rectangular blocks
+ shared temporal validity signatures
+ compact BST-style block encoding
+ coarse spatial bucket index
```

The slower Step-7 `bst_prefix_index` is intentionally not part of the main real-data benchmark.

## NCLT assumptions used

- Use `velodyne_sync/UTIME.bin`, not raw `velodyne_hits.bin`.
- Each synchronized file is one motion-compensated revolution recorded in the
  Segway body frame.
- Match each scan timestamp to the provided NCLT ground-truth pose.
- Transform body-frame scan points into the native NCLT local frame.
- Voxelize session point clouds into a shared integer grid.

No SLAM algorithm is being evaluated in Step 8.

## Important semantics

A voxel absent from one session map is **not** automatically claimed to be
observed free space.

Therefore:

```text
session A occupied - session B occupied
```

is called an **observation-history difference**, not a semantic environmental
change.

True change detection / visibility reasoning is a later stage.

## Recommended pilot sessions

Default CLI pilot:

```text
2012-01-15
2012-02-02
2012-04-29
2012-05-26
2012-08-04
```

These dates have been used together in prior NCLT long-term stability work
because of their useful overlap. You can replace them with any extracted NCLT sessions.

## Expected NCLT layouts

The loader auto-detects common layouts, including:

```text
DATA/
  2012-01-15/
    velodyne_sync/*.bin
    groundtruth_2012-01-15.csv
```

or the official-style separated layout:

```text
DATA/
  velodyne_data/
    2012-01-15_vel/
      velodyne_sync/*.bin
  ground_truth/
    groundtruth_2012-01-15.csv
```

## Smoke run

Start cheap:

```bash
python run_step8.py \
  --data-root /path/to/NCLT \
  --sessions 2012-01-15,2012-02-02 \
  --voxel-size 0.5 \
  --scan-stride 20 \
  --max-scans-per-session 300 \
  --queries 3 \
  --warmups 1 \
  --repetitions 2 \
  --output results/step8_smoke
```

## Pilot experiment

```bash
python run_step8.py \
  --data-root /path/to/NCLT \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --min-range 2 \
  --max-range 80 \
  --min-points-per-voxel 1 \
  --queries 20 \
  --warmups 2 \
  --repetitions 5 \
  --output results/step8_nclt5
```

## Full-density sensitivity run

Do not jump to this first.

```bash
python run_step8.py \
  --data-root /path/to/NCLT \
  --sessions 2012-01-15,2012-02-02,2012-04-29,2012-05-26,2012-08-04 \
  --voxel-size 0.5 \
  --scan-stride 1 \
  --queries 20 \
  --warmups 2 \
  --repetitions 5 \
  --output results/step8_nclt5_fullscan
```

## Outputs

```text
real_build_storage.csv
real_queries.csv
session_stats.csv
history_complexity.json
metadata.json
```

Optional `--save-prepared` also serializes the voxelized sessions.

## Step-8 success criteria

The central comparison is:

```text
bst_compact_bucket vs octree_shared
```

A strong result is:

```text
BST storage <= 0.70 * OctreeShared storage
BST 5m query <= 1.25 * OctreeShared 5m query
BST 20m/change queries < OctreeShared
```

An excellent result would be `BST storage <= 0.50 * OctreeShared`.

## Before publication

Step 8 does not solve:
- visibility/free-space reasoning,
- semantic change detection,
- online incremental updates,
- localization,
- verified upstream BST-ID encoding.

It is specifically the real-data representation validation.
