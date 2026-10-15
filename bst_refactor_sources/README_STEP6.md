# Step 6 — Fairness and adversarial evaluation

Step 5 showed a promising Pareto advantage, but it also gave BST more than a
new ID: validity intervals, temporal-history sharing, and anisotropic spatial
merging. Step 6 tests whether the result survives when the Octree receives the
same temporal advantages and when the synthetic world is deliberately changed
to favor or hurt the proposed method.

## Methods

1. `snapshot` — full-session reference.
2. `octree_time` — Step-4/5 Octree with transition history.
3. `octree_interval` — **new:** Octree leaves use exact dyadic temporal validity prefixes.
4. `octree_shared` — **new strongest fair baseline:** Octree + validity prefixes + globally shared temporal signatures.
5. `bst_interval` — Step-5 temporal-prefix ablation.
6. `bst_adaptive` — Step-5 proposed adaptive x/y/z/time representation, unchanged.

`octree_shared` is the baseline that matters most. If BST only beats
`octree_time` but not `octree_shared`, the Step-5 advantage is mostly temporal
compression rather than BST-style spatial/temporal hierarchy.

## Adversarial sweeps

The full suite changes one major property at a time:

- static fraction: target 95%, 80%, 50%;
- dynamic change probability: 0.03, 0.15, 0.60 per session;
- spatial coherence: coherent large cuboids, mixed objects, fragmented small objects;
- voxel resolution: 1.0 m, 0.5 m, 0.25 m;
- world scale: 50 m, 100 m, 200 m square worlds.

The generator reports the *observed* static fraction and ever-changed fraction,
so conclusions should use measured values rather than assuming the target was
reached exactly.

## Quick run

```bash
python run_step6.py \
  --suite quick \
  --sessions 100 \
  --queries 4 \
  --warmups 1 \
  --repetitions 3 \
  --output results/step6_quick
```

The quick suite covers six difficult corners: mostly static, dynamic-heavy,
low/high change rate, coherent geometry, and fragmented geometry.

## Full reviewer-grade synthetic run

```bash
python run_step6.py \
  --suite full \
  --sessions 250 \
  --queries 20 \
  --warmups 2 \
  --repetitions 5 \
  --output results/step6_full
```

The 0.25 m and 200 m scenarios can be expensive in Python. If necessary, run
those two separately at 100 sessions first, then scale once correctness is
confirmed.

## Outputs

```text
results/step6_full/
├── scenario_build.csv
├── scenario_queries.csv
└── metadata.json
```

Every row includes scenario parameters and observed dataset statistics, so you
can plot performance against actual static fraction, changed-voxel fraction,
resolution, and scale.

## Primary comparisons

Focus on:

```text
octree_shared  vs  bst_adaptive
```

For each scenario record:

- compact storage ratio;
- build-time ratio;
- 5 m and 20 m historical-query median/p95;
- 5 m and 20 m short/medium/long change-query median/p95;
- nodes/blocks inspected.

## Decision rules

A strong result is not “BST wins one scenario.” Continue toward real datasets
only if most of the following hold:

1. `bst_adaptive` remains at least ~25–30% smaller than `octree_shared` in the
   baseline and mostly-static cases.
2. Its 5 m query median is within ~1.25× of `octree_shared` even in fragmented
   geometry.
3. It remains faster on medium/full regional change queries in several change
   rates, not only one seed.
4. The advantage degrades smoothly as coherence is destroyed; it should not
   collapse unpredictably.
5. Repeating with multiple seeds preserves the qualitative ranking.

If `octree_shared` closes nearly all storage and query gaps, then the Step-5
win came mainly from interval compression/history sharing rather than BST-ID.
That is a useful negative result and should stop this particular paper framing.

## Scientific caveat

As in Step 5, the BST variants use the local-metric research prototype rather
than a verified byte-for-byte upstream `nagasawayuki/bst-id` implementation.
Do not make final BST-ID claims until the upstream encoding is integrated and
cross-validated.
