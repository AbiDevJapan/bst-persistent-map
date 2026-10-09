# Step 9.2 — Frozen Multi-Session Localization Validation

Step 9.2 is a validation step, not another optimization step.

It freezes the localization representation selected in Steps 9.1.1–9.1.3:

- canonical occupancy/index lattice: **0.5 m**;
- geometry gate: `balanced`;
- absolute plane sigma: **0.05 m**;
- relative planarity: **0.025**;
- child-normal coherence: **0.95**;
- maximum node extent: **2.0 m**;
- hierarchical LOD cap: **K=4**;
- deployed representation: hierarchy-free compact materialized K=4 table;
- matcher and ICP pyramid unchanged from Step 9.1.3.

No threshold is selected, optimized, or changed by this benchmark.

## Main experiment

The default map session is `2012-01-15`. The four later downloaded NCLT sessions are independent query sessions:

- `2012-01-15 -> 2012-02-02`
- `2012-01-15 -> 2012-03-17`
- `2012-01-15 -> 2012-05-11`
- `2012-01-15 -> 2012-06-15`

The map/history representation is built **once** and reused for all four query sessions. This avoids rebuilding the five-session BST and Jan-15 surfel geometry four times.

By default, 50 scans are selected from each query session, giving 200 cross-session scans. The Jan->Feb selection excludes the Step-9.1.2 tuning scans when the exclusion CSV is available. The Mar/May/Jun session pairs were never used for parameter tuning and are therefore session-level held-out tests.

The primary localization perturbation remains:

- `0.5 m / 1 degree`.

Every selected scan also receives a zero-initialization control. `1 m / 3 degree` can be added as a stress test with `--perturbations 0.5:1,1:3`, but it is not the frozen pass/fail gate.

## Compared methods

Only two deployed methods are compared:

1. `voxel_surfel_0.5m`
2. `bst_lod_k4_compact`

The full construction hierarchy is used only to materialize the compact K=4 payload, verified, then released before validation localization begins.

## Frozen validation gates

For the pooled `0.5 m / 1 degree` cross-session result:

- BST success retention relative to voxel surfels >= **0.90**;
- local primitive reduction >= **0.40**;
- median end-to-end speedup >= **1.20x**.

The benchmark reports the gates; it never changes parameters in response to them.

## Outputs

- `localization_trials.csv` — every trial with `map_session`, `query_session`, and `session_pair`.
- `session_summary.csv` — per-session-pair results.
- `pooled_summary.csv` — pooled cross-session results across all query sessions.
- `paired_outcomes.csv` — both-success / voxel-only / BST-only / neither and exact discordant-pair p-value.
- `validation_gates.csv` — frozen pooled gate evaluation.
- `query_scans.csv` — all selected scans and held-out-selection metadata.
- `deployment_memory.csv` — deployed resident-memory accounting.
- `materialization_verification.csv` — exact compact-payload/query/pyramid verification.
- `metadata.json` — complete frozen protocol and build information.

## Run

```bash
python3 run_step9_2.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-sessions 2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --query-count-per-session 50 \
  --same-session-query-count 50 \
  --exclude-query-scans-csv results/step9_1_2_hierarchical_lod/query_scans.csv \
  --perturbations 0.5:1 \
  --output results/step9_2_multisession
```

No additional NCLT sessions are required.

## Interpretation

Step 9.2 tests external validity across the four available cross-session pairs. It should not be used to choose a new geometry threshold or K. If one session fails, report that session explicitly rather than modifying the representation and rerunning the same benchmark.

The `PrototypeBSTSpatialCodec` remains the local research codec and is not a verified upstream BST-ID implementation.

### NCLT non-finite GT guard
Step 9.2 filters held-out query candidates whose matched ground-truth pose has any non-finite x/y/z/roll/pitch/yaw value. This check occurs before expensive map construction. Materialization verification also rejects non-finite centers with an explicit diagnostic instead of failing inside grid-bound conversion.

## Recovery from the original late metadata serialization bug

An early Step 9.2 package serialized `session_block_ms` as a scalar even though
`build_session_spatial_blocks` returns one timing per session. The failure occurred
only after all scientific CSV outputs had already been written.

If a run ended with:

```text
TypeError: float() argument must be a string or a real number, not 'list'
```

**do not rerun the map/localization benchmark.** Finalize the existing output directory:

```bash
python3 finalize_step9_2.py --output results/step9_2_multisession
```

This reconstructs `metadata.json` from the already-written CSV outputs. Construction
build timings that existed only in process memory are marked unavailable rather than
guessed. New full runs store both the total session-block build time and the per-session
build-time list and do not have this failure.
