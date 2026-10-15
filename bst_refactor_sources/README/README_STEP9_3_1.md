# Step 9.3.1 — History-pattern-aware robust localization

Step 9.3.1 keeps the complete Step-9.2/9.3 spatial pipeline frozen:

- canonical lattice: 0.5 m;
- balanced geometry gate (`sigma <= 0.05 m`, relative planarity `<= 0.025`, child-normal coherence `>= 0.95`, extent `<= 2 m`);
- hierarchical localization LOD: `K=4`;
- compact hierarchy-free materialized runtime;
- same coarse-to-fine point-to-plane matcher, trimming, convergence thresholds and success gate.

It compares three BST methods on exactly the same retrieved geometry:

1. `bst_lod_k4_compact` — frozen unweighted control;
2. `bst_lod_k4_persistence` — Step-9.3 linear repeated-occupancy weighting, retained as the negative temporal control;
3. `bst_lod_k4_history_robust` — new causal history-pattern-aware robust weighting.

No temporal threshold is tuned in this step.

## Why this differs from Step 9.3

Step 9.3 continuously multiplied every ICP constraint by repeated-occupancy frequency. That is problematic because an absent voxel is **not** known free; it can be missing because of viewpoint, occlusion, range or sampling.

Step 9.3.1 instead describes **intermittency**. For each canonical Jan/map voxel and a query session `t`, it uses only sessions strictly before `t` and counts occupancy-state transitions between adjacent prior sessions.

Examples:

- `1111` -> zero transitions -> stable;
- `1110` -> one transition;
- `1011` -> two transitions;
- `1010` -> three transitions -> highly intermittent.

The selected K=4 surfel receives the voxel-support mean transition rate. The score is quantized to one `uint8` per selected surfel. It is an observation-pattern descriptor, not semantic permanence and not observed-free probability.

## Residual-gated temporal weighting

Temporal history is used only when the current geometric evidence is suspicious.

After the frozen correspondence search and residual trimming, let

`m = median(|r|)`

and for each retained correspondence

`q = max(|r| / m - 1, 0)`.

With causal transition-instability `u in [0,1]`, the normal-equation weight is

`w = 1 / (1 + u q^2)`.

Consequences:

- stable history (`u=0`) always gives `w=1`;
- any residual at or below the current median also gives `w=1`;
- only a correspondence that is both historically intermittent **and** currently above-median in residual is downweighted;
- no geometry is removed and there is no hard temporal threshold.

The reported plane residual metrics remain unweighted, so they stay comparable to prior steps.

## Causality / null control

With Jan-15 as the map:

- Jan→Feb has only one prior session, so transition instability is exactly zero; all three BST methods must be exactly identical. This is asserted trial-by-trial.
- Jan→Mar uses Jan+Feb history only.
- Jan→May uses Jan+Feb+Mar only.
- Jan→Jun uses Jan+Feb+Mar+May only.

The query session itself and future sessions never contribute to temporal descriptors.

## Recommended fresh held-out run

Use a new scan set by excluding the prior tuning/validation experiments, including Step 9.3:

```bash
python3 run_step9_3_1.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-sessions 2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --query-count-per-session 50 \
  --same-session-query-count 50 \
  --exclude-query-scans-csv results/step9_1_2_hierarchical_lod/query_scans.csv \
  --exclude-query-scans-csv results/step9_2_multisession/query_scans.csv \
  --exclude-query-scans-csv results/step9_3_persistence/query_scans.csv \
  --perturbations 0.5:1 \
  --output results/step9_3_1_history_robust
```

Do not change the weighting rule after seeing these results. If this method fails, retain the negative result rather than threshold-shopping.

## Outputs

- `localization_trials.csv`
- `session_summary.csv`
- `pooled_summary.csv`
- `paired_persistence_outcomes.csv`
- `paired_history_robust_outcomes.csv`
- `persistence_stats.csv`
- `history_pattern_stats.csv`
- `persistence_gates.csv`
- `history_robust_gates.csv`
- `null_control_verification.csv`
- `deployment_memory.csv`
- `materialization_verification.csv`
- `query_scans.csv`
- `metadata.json`

The primary 9.3.1 gate is evaluated on Mar/May/Jun, excluding Feb's exact no-history control. It requires >=95% success retention versus unweighted K=4, <=10% median runtime overhead, and identical primitive count. Positive paired success gain is reported separately rather than forced as a pass condition.
