# Step 9.3 — Causal persistence-aware localization

Step 9.3 freezes the Step-9.2 spatial representation and matcher geometry:

- canonical lattice: 0.5 m;
- geometry gate: balanced (`sigma <= 0.05 m`, relative planarity `<= 0.025`, child-normal coherence `>= 0.95`, extent `<= 2 m`);
- hierarchical localization LOD: `K=4`;
- compact hierarchy-free materialized runtime;
- same coarse-to-fine point-to-plane local-registration benchmark.

The only new variable is **causal temporal persistence weighting**.

## Persistence definition

For a query session with history index `t`, a canonical map voxel gets

`p = (# sessions from the map session through t-1 in which that voxel was occupied) / (# such prior sessions)`.

The query session itself and future sessions are excluded. A selected K=4 surfel receives the voxel-support mean of its constituent canonical voxels. Scores are quantized to one `uint8` per deployed surfel and interpreted as `score/255`.

Absence is **not** treated as observed free space. This is a repeated-occupancy observation proxy, not semantic permanence.

ICP keeps every geometry primitive and every correspondence candidate. After the frozen residual trimming step, persistence enters only as a soft normal-equation weight:

`A = J^T W J`, `b = -J^T W r`, with `W_ii = p_i`.

There is no hard persistence threshold and no tuned exponent/floor.

## Causal controls

With Jan-15 as the map:

- Jan→Feb has only Jan history available, therefore all persistence scores are exactly 1 and weighted/unweighted localization must be exactly identical. This is asserted trial-by-trial.
- Jan→Mar uses Jan+Feb only.
- Jan→May uses Jan+Feb+Mar only.
- Jan→Jun uses Jan+Feb+Mar+May only.

This avoids hindsight leakage from the session being localized.

## Recommended new held-out run

Exclude both the Step-9.1.2 tuning scans and the Step-9.2 held-out scans:

```bash
python3 run_step9_3.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-sessions 2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --query-count-per-session 50 \
  --same-session-query-count 50 \
  --exclude-query-scans-csv results/step9_1_2_hierarchical_lod/query_scans.csv \
  --exclude-query-scans-csv results/step9_2_multisession/query_scans.csv \
  --perturbations 0.5:1 \
  --output results/step9_3_persistence
```

Default methods are only the clean temporal ablation:

- `bst_lod_k4_compact` — frozen unweighted K=4 runtime;
- `bst_lod_k4_persistence` — identical retrieval/geometry with causal persistence weighting.

Add `--include-voxel-baseline` only if another external voxel-surfel comparison is desired. Step 9.2 already established the frozen BST-vs-voxel relationship, so omitting it roughly reduces this new experiment's localization work.

## Outputs

- `localization_trials.csv`
- `session_summary.csv`
- `pooled_summary.csv`
- `paired_history_outcomes.csv`
- `persistence_stats.csv`
- `persistence_gates.csv`
- `null_control_verification.csv`
- `deployment_memory.csv`
- `materialization_verification.csv`
- `query_scans.csv`
- `metadata.json`

The primary later-session diagnostic excludes Feb's no-extra-history null control. It reports paired success changes over Mar/May/Jun, success retention, runtime overhead, and exact primitive-count preservation. The package does not retune any spatial or temporal threshold if persistence fails.
