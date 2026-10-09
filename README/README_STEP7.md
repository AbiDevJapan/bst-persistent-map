# Step 7 — Isolate the BST-specific contribution

Step 6 showed that the adaptive representation strongly beats strengthened
Octree baselines. Step 7 asks a stricter question:

> If the adaptive spatial blocks and temporal histories are held exactly
> constant, what does BST-style prefix representation/indexing itself add?

## Critical discovery

The Step-5/6 `BSTAdaptiveIntervalMapPrototype` does **not** use a true BST-ID
prefix index for runtime spatial lookup. It stores compact prefix-like block
metadata, but queries use a coarse fixed-size Python bucket directory.

Therefore the large Step-6 query advantage should be attributed to:

- adaptive dyadic rectangular blocks,
- temporal validity signatures,
- reduced candidate count,
- coarse spatial pruning,

not yet to BST-ID lookup itself.

Step 7 makes that distinction explicit.

## Methods

### `adaptive_explicit`

Ordinary integer block IDs and explicit block bounds:

```text
uint16 x0,y0,z0,sx,sy,sz
uint32 temporal_signature_id
```

It uses the **same adaptive blocks** and the **same coarse bucket index** as
Step-6 `bst_adaptive`.

### `bst_compact_bucket`

This is the unchanged Step-6 `BSTAdaptiveIntervalMapPrototype`.

Same blocks.
Same temporal histories.
Same coarse bucket query index.

The difference from `adaptive_explicit` is primarily compact hierarchical
spatial-prefix representation in the logical storage model.

Therefore:

```text
adaptive_explicit vs bst_compact_bucket
```

is the storage/encoding isolation test.

Their query timings should be approximately equal. If they are, that is
expected and scientifically important.

### `bst_prefix_index`

Uses exactly the same blocks and temporal histories again, but replaces coarse
buckets with a hierarchical dyadic prefix-posting index independently over
x/y/z.

A spatial query is decomposed into dyadic query prefixes. Candidate lookup finds
stored prefixes that are ancestors or descendants of those query prefixes.

This tests:

```text
coarse ordinary bucket lookup
vs
hierarchical prefix lookup
```

without changing the map itself.

This is still a local research prototype and must NOT be called the verified
upstream BST-ID implementation.

## Hard correctness control

For every scenario/seed:

```python
adaptive_explicit.blocks
==
bst_compact_bucket.blocks
==
bst_prefix_index.blocks
```

is asserted before query benchmarking.

All query outputs must also match Snapshot exactly.

## Reproducibility

The full suite uses 5 seeds:

```text
1, 7, 21, 42, 84
```

and six important environments:

```text
baseline
static95
static50
change_high
fragmented
voxel_025m
```

This avoids treating multiple labels for identical Step-6 parameter settings as
independent evidence.

## Quick run

```bash
python run_step7.py \
  --suite quick \
  --sessions 100 \
  --queries 4 \
  --warmups 1 \
  --repetitions 3 \
  --output results/step7_quick
```

## Full run

```bash
python run_step7.py \
  --suite full \
  --sessions 250 \
  --queries 20 \
  --warmups 2 \
  --repetitions 5 \
  --output results/step7_full
```

Outputs:

```text
isolation_build.csv
isolation_queries.csv
isolation_summary.csv
metadata.json
```

## Decision rules

### Outcome A — strong BST-specific result

If `bst_prefix_index` beats `adaptive_explicit` by roughly 20%+ on important
localized/history/change queries without destroying the storage advantage, then
hierarchical prefix indexing contributes measurable value beyond the adaptive
map representation.

### Outcome B — encoding-only benefit

If:

```text
adaptive_explicit query time ~= bst_compact_bucket query time
```

but compact storage is materially smaller, then the BST-style representation
contributes compact addressing but not runtime speed in the current system.

This is still useful, but the paper claim should be narrower.

### Outcome C — adaptive blocks explain nearly everything

If `bst_prefix_index` is not better than the ordinary baseline and the only
large gain is shared by all identical-block methods, then the publishable
contribution is the adaptive spatiotemporal block map, not BST-ID itself.

Do not hide this outcome. Reframe the work.

## Upstream BST-ID integration

The published work is:

Yuki Nagasawa and Yuichi Yaguchi,
"BST-ID: A Variable-Length Binary Spatiotemporal ID Scheme for Hierarchical and
Bandwidth-Constrained Systems," IEEE SMC 2025, pp. 279–284,
DOI 10.1109/SMC58881.2025.11343277.

The Step-7 code deliberately labels its prefix index as a prototype. Replace
the codec/index layer with the authors' verified BST-ID implementation/spec
before making a paper claim specifically about BST-ID.
