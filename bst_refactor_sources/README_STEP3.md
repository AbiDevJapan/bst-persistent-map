# Step 3 — Competing Temporal Map Representations

Step 3 adds three map implementations with the same logical API:

1. `SnapshotMap`
2. `OctreeTemporalMap`
3. `BSTTemporalMapPrototype`

The goal of this step is **correctness equivalence**, not publication-grade
performance benchmarking.

## Run

```bash
pip install -r requirements.txt
python run_step3.py --sessions 30 --random-queries 25
pytest -q
```

## Common operations

Each representation supports:

```python
snapshot(t)
region_at_time(region, t)
changes(region, t_a, t_b)
```

For every tested query, outputs must satisfy:

```text
Snapshot == Octree+time == BST prototype
```

before timing results are considered meaningful.

## Baseline A — SnapshotMap

Stores every session independently.

This is easy to query but storage scales approximately with:

```text
(number of sessions) × (map size)
```

## Baseline B — OctreeTemporalMap

Uses a true 3-D Octree. Each voxel leaf stores only occupancy **state
transitions**:

```text
voxel -> [(t0, occupied), (t7, free), (t13, occupied), ...]
```

This is the strong conventional baseline.

## Method C — BSTTemporalMapPrototype

Stores each transition as an event addressed by a hierarchical binary key over:

```text
x, y, z, t
```

Leading key prefixes correspond to progressively smaller 4-D regions. For Step
3, events are grouped into fixed-length leading-prefix buckets. A query rejects
buckets whose spatiotemporal prefix bounds cannot intersect the query, then
checks exact events.

### Important scientific caveat

`LocalMetricBSTEncoder` is a **Step-3 robotics prototype** that interleaves
significant x/y/z/t bits. It preserves the hierarchical/prefix property needed
to develop and test the map architecture, but this project does **not** claim it
is byte-for-byte identical to the upstream `nagasawayuki/bst-id` encoder.

The encoder is intentionally isolated in:

```text
representations/bst_encoder.py
```

so the exact upstream encoder can later replace it without changing the
experimental API.

Do not publish timing or storage claims as "BST-ID vs Octree" until the exact
BST-ID implementation/specification has been integrated and cross-validated.

## Step-3 completion criterion

Step 3 is complete when:

- all three maps reconstruct identical full snapshots,
- all three answer identical local historical queries,
- all three return identical endpoint change queries,
- randomized equivalence tests pass,
- BST encode/decode and prefix bounds tests pass.

Only then move to Step 4: controlled benchmark instrumentation for time,
storage, inspected records, and scaling.
