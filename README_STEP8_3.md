# Step 8.3 — Real-session scaling, non-Octree baselines, and isolated BST codec

Step 8.3 stops changing the map algorithm and asks a broader representation question:

> As repeated real NCLT sessions accumulate, where does the adaptive map sit on the storage/query Pareto frontier compared with snapshots, sparse Octrees, voxel histories, and base+delta versioning?

It also makes decompression explicit and isolates all BST-specific spatial encoding behind a replaceable codec boundary.

## Methods

The default comparison set is:

```text
snapshot
DeltaVersionMap
PackedVoxelHistoryMap
Sparse OctreeShared
AdaptiveExplicit
BSTCompact
```

### Snapshot

Stores every session independently. It is the fast-query / high-redundancy extreme.

### DeltaVersionMap

Stores the **latest session as the base** and reverse positive/negative voxel deltas:

```text
M_latest
+ reverse delta latest -> latest-1
+ reverse delta latest-1 -> latest-2
+ ...
```

This is a neutral base+delta/version-control baseline. It does **not** claim to reproduce a specific published lifelong mapper.

### PackedVoxelHistoryMap

Stores one packed spatial voxel key per ever-observed voxel plus a shared temporal-signature ID. It is the strong baseline for:

```text
voxel -> temporal history
```

It is intentionally implemented with packed sorted arrays rather than Python hash objects so the comparison is not dominated by language overhead.

### Sparse OctreeShared

The hardened Step-8.2 sparse Morton Octree baseline.

### AdaptiveExplicit

The proposed adaptive blocks with ordinary explicit rectangular bounds.

### BSTCompact

The **same exact AdaptiveBlockCore** as AdaptiveExplicit, with only spatial block encoding replaced by a BST-style codec.

## BST isolation architecture

```text
real NCLT observations
        |
        v
AdaptiveBlockCore            <-- NO BST imports
  - temporal signatures
  - adaptive x/y/z merging
  - spatial bucket index
  - native block queries
  - optional voxel materialization
        |
        +------------------------+
        |                        |
        v                        v
AdaptiveExplicitMap        BSTCompactMap
 explicit bounds             thin wrapper
                                  |
                                  v
                         SpatialBlockCodec
                                  |
                                  v
                     PrototypeBSTSpatialCodec
                         (bst_id/codec.py)
```

The future verified BST-ID integration point is only:

```text
bst_id/codec.py
```

A replacement codec implements:

```python
encode_block(block, spec)
spatial_table_bytes(blocks, spec)
```

The adaptive algorithm and benchmark do not need to change.

**Important:** `PrototypeBSTSpatialCodec` is still the local dyadic-prefix prototype, not a verified upstream BST-ID implementation.

## Optional decompression

Every Step-8.3 representation uses a common query interface:

```python
map.region_query(region, t, output="voxels")
map.region_query(region, t, output="native")
```

and:

```python
map.changes_query(region, ta, tb, output="voxels")
map.changes_query(region, ta, tb, output="native")
```

### `output="voxels"`

Canonical equal-output comparison. Every method must return the exact same sorted voxel IDs as Snapshot.

Use this for strict representation-to-representation fairness.

### `output="native"`

Representation-native retrieval.

For AdaptiveExplicit / BSTCompact:

```text
[x0, y0, z0, sx, sy, sz, temporal_signature_id]
```

blocks are returned without decompression.

For the sparse Octree the natural finest leaves are voxels, so native output remains voxel leaves.

Native timing is intentionally marked `equal_output=False` in the CSV. Do not present it as an apples-to-apples dense reconstruction benchmark. It measures the local-map primitive each representation would naturally hand to a downstream system.

## Scaling experiment

The default NCLT sessions are:

```text
2012-01-15
2012-02-02
2012-04-29
2012-05-26
2012-08-04
```

and the default nested prefixes are:

```text
N = 2, 3, 5
```

All prefixes use one common prepared coordinate grid. This keeps spatial coordinates fixed while history length grows.

## Run

```bash
python3 run_step8_3.py \
  --data-root data \
  --sessions 2012-01-15,2012-02-02,2012-04-29,2012-05-26,2012-08-04 \
  --scales 2,3,5 \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --min-range 2 \
  --max-range 80 \
  --min-points-per-voxel 1 \
  --queries 20 \
  --warmups 2 \
  --repetitions 5 \
  --output-modes both \
  --output results/step8_3_nclt_scaling
```

The Step-8.1 cache namespace is reused by default:

```text
.cache/step8_1
```

so already-cached sessions should not be reprocessed.

## Outputs

```text
scaling_build_storage.csv
scaling_queries.csv
history_complexity_scaling.csv
session_stats.csv
metadata.json
```

### `scaling_build_storage.csv`

One row per method and session count, including:

```text
method_build_ms
shared_history_ms
standalone_total_build_ms
compact_mib
```

### `scaling_queries.csv`

Contains both `output_mode=voxels` and `output_mode=native` rows.

Important fields:

```text
sessions
method
query_kind
region
span
output_mode
equal_output
median_ms
p95_ms
mean_result_primitives
mean_voxel_equivalent
```

For native adaptive queries, `mean_result_primitives` is the number of blocks while `mean_voxel_equivalent` is the number of voxels those blocks represent.

## Main plots to make

### 1. Storage scaling

```text
compact MiB vs number of sessions
```

for all six methods.

### 2. Canonical query scaling

For `output_mode=voxels`:

```text
5 m historical latency vs N
20 m historical latency vs N
5 m change latency vs N
20 m change latency vs N
```

### 3. Native local-map retrieval

For `output_mode=native` compare at least:

```text
Sparse OctreeShared
AdaptiveExplicit
BSTCompact
```

Report both latency and primitive count. Do not call different primitive types equal-output.

### 4. Temporal-history complexity

Plot:

```text
unique histories vs N
fraction zero-transition vs N
mean transitions/voxel vs N
```

## Decisions intentionally frozen in Step 8.3

1. **Delta anchor:** latest session, with reverse deltas for history.
2. **Voxel-history baseline:** packed sorted voxel keys + shared temporal signatures.
3. **Spatial query sizes:** 5 m and 20 m; full materialization remains opt-in.
4. **Targeting:** 75% occupied/change-centered queries, 25% uniform random.
5. **Scaling:** nested chronological prefixes on one fixed coordinate grid.
6. **BST role:** spatial encoding only; adaptive mapping/query logic is generic.

These can be ablated later, but should not be changed while interpreting the first N=2/3/5 scaling result.

## Scientific decision after this step

The key question is not whether one method wins every metric. It is whether BSTCompact/adaptive mapping occupies a useful Pareto region compared with:

```text
Snapshot          -- fastest explicit versions
DeltaVersion      -- compact sequential versioning
VoxelHistory      -- direct per-voxel time history
Sparse Octree     -- strong spatial hierarchy
AdaptiveExplicit  -- isolates the adaptive-map contribution
BSTCompact        -- isolates compact hierarchical encoding
```

If the adaptive/BST storage curve grows substantially more slowly while native local-map retrieval remains fast, proceed to the localization feasibility module.
