# Step 4 — Controlled scaling benchmark

This step asks the first serious research question: does the BST-style x/y/z/t representation provide a materially better storage/query tradeoff than Octree + temporal history?

For each history length N, measure build time, compact logical serialized bytes, historical-region median/p95 latency, change-query median/p95 latency, and records/nodes/events inspected.

Spatial query scales: 5 m x 5 m, 20 m x 20 m, and full map. Change spans: short=1 session, medium≈10% of history, long≈50% of history.

Quick run:

```bash
python run_step4.py --sessions 10,25,50,100 --queries 8 --warmups 1 --repetitions 3
```

Full scaling run:

```bash
python run_step4.py --sessions 10,25,50,100,250,500,1000 --queries 20 --warmups 2 --repetitions 5 --output results/step4_full
```

Outputs are `build_storage.csv`, `queries.csv`, and `metadata.json`.

`compact_bytes` is a deterministic compact binary-layout estimate, not Python heap memory. Representative queries are checked for exact equality before timings count.

Continue only if BST shows a substantial advantage against Octree+time: e.g. <=0.5x Octree latency for 5 m historical queries at comparable storage, materially lower storage at comparable latency, or a clearly superior storage/query Pareto frontier. A 5–10% difference is not enough.

Scientific caveat: the current method is still `BSTTemporalMapPrototype`, not yet the exact upstream `nagasawayuki/bst-id` encoder.
