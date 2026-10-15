# Step 5 — Adaptive space-time validity representation

Step 4 showed that one full x/y/z/t key per transition does not scale well.
Step 5 changes the *representation*, not just the code speed.

## Hypothesis

A lifelong map should represent an occupied state as a validity region:

```text
spatial region × temporal interval
```

instead of only as a sequence of point events.

The proposed representation uses:

1. exact occupied intervals `[t_start, t_end)`,
2. exact dyadic temporal-prefix covers,
3. independent dyadic x/y/z merging when neighboring voxels have identical
   temporal histories,
4. direct one-pass endpoint change queries.

This tests heterogeneous resolution in both space and time.

## New ablations

### `bst_events` (optional)
The Step-4 implementation: one fixed-resolution x/y/z/t key per state event. The full Step-4 results already contain this baseline, so Step 5 omits it by default to keep 500–1000-session runs practical. Add `--include-events` when you want the ablation rerun.

### `bst_interval`
One spatial voxel record per voxel, but occupancy history is stored as temporal
validity prefixes. This isolates the effect of replacing predecessor-event
queries with direct time-prefix membership.

### `bst_adaptive`
Adds independent x/y/z merging for adjacent voxels with identical temporal
histories. A thin wall can become coarse along x and z while remaining one
voxel thick along y. This is closer to BST-ID's independent-dimension precision
idea than a conventional octree cube.

## Run

First:

```bash
python run_step5.py --sessions 10,25,50,100 --queries 8 --warmups 1 --repetitions 3

# Optional: include the old Step-4 event baseline
python run_step5.py --sessions 10,25,50,100 --include-events
```

Then, if correct:

```bash
python run_step5.py \
  --sessions 10,25,50,100,250,500,1000 \
  --queries 20 --warmups 2 --repetitions 5 \
  --output results/step5_full
```

## What to inspect

The strongest evidence would be:

- `bst_interval` beating `bst_events` strongly as N grows: temporal intervals
  solved the predecessor-scan problem.
- `bst_adaptive` storing materially fewer spatial blocks than unmerged voxels.
- `bst_adaptive` approaching/beating Octree on 5 m and 20 m queries while
  retaining a storage advantage.
- direct `changes()` queries scaling with local candidate blocks rather than
  total historical event count.

## Stop condition

If at N=500–1000 `bst_adaptive` is still substantially larger *and* >2× slower
than Octree+time for 5 m historical/change queries, stop pursuing BST-ID for
this lifelong-history application.

## Scientific caveat

These remain local-metric BST-style research prototypes. Before publication,
the exact upstream `nagasawayuki/bst-id` encoding and semantics must be
integrated/cross-validated, and strong optimized baselines should be used.
