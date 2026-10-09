# Step 9.4 — Causal historical map selection

Step 9.4 stops modifying ICP weights. It asks a more direct lifelong-localization question:

> For a query session, does using the most recent map that existed **before** that session improve local registration, and can the BST history representation preserve that capability with substantially fewer localization primitives/storage than independent voxel-surfel snapshots?

The Step-9.2 spatial/localization configuration stays frozen: 0.5 m canonical lattice, balanced geometry gate, K=4 hierarchical LOD, compact materialized runtime, and the same coarse-to-fine point-to-plane matcher.

## Causal map rule

With the five available NCLT sessions in chronological order:

- Feb query -> Jan map
- Mar query -> Feb map
- May query -> Mar map
- Jun query -> May map

The query session and all future sessions are excluded. There is no "best historical map" search after seeing the query result.

## Three-way decomposition

1. `static_jan_voxel`: always localize against the original Jan voxel-surfel map.
2. `latest_prior_voxel`: localize against the immediately preceding session's voxel-surfel map. This isolates the **temporal map-selection effect** without any BST compression.
3. `bst_latest_prior_k4`: use the same latest-prior session date, but retrieve its active roots through the Step-8.5 time-aware BST and localize with the frozen compact K=4 LOD. This isolates the **BST historical representation effect**.

The Feb case is an exact temporal null control because the latest prior session is Jan. `static_jan_voxel` and `latest_prior_voxel` must therefore be identical trial-by-trial; Step 9.4 asserts this.

## Storage accounting

Two historical-map catalogs are counted:

- independent voxel-surfel snapshots for Jan, Feb, Mar and May;
- one five-session BST history representation plus one compact K=4 localization overlay for each causal prior session.

The BST history bytes are counted once. Construction merge forests are not retained in deployment.

This is not claiming that latest-prior accuracy itself is caused by BST. A latest-prior improvement is a lifelong map-update effect. The BST-specific claim is tested by success retention, primitive/runtime reduction, and historical catalog storage versus independent snapshots.

## Recommended fresh held-out run

Exclude every previous query set so Step 9.4 is not evaluated on the scans used by 9.1.2, 9.2, 9.3, or 9.3.1:

```bash
python3 run_step9_4.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-sessions 2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --query-count-per-session 50 \
  --exclude-query-scans-csv results/step9_1_2_hierarchical_lod/query_scans.csv \
  --exclude-query-scans-csv results/step9_2_multisession/query_scans.csv \
  --exclude-query-scans-csv results/step9_3_persistence/query_scans.csv \
  --exclude-query-scans-csv results/step9_3_1_history_robust/query_scans.csv \
  --perturbations 0.5:1 \
  --output results/step9_4_historical_selection
```

The benchmark builds the five-session BST once. Localization geometry for Jan/Feb/Mar/May is then built **sequentially**, evaluated, counted, and released so the experiment does not require all construction hierarchies or raw surfel maps in memory simultaneously.

## Outputs

- `localization_trials.csv`
- `session_summary.csv`
- `pooled_summary.csv`
- `temporal_selection_outcomes.csv`
- `bst_historical_outcomes.csv`
- `historical_map_stats.csv`
- `historical_storage.csv`
- `historical_gates.csv`
- `materialization_verification.csv`
- `null_control_verification.csv`
- `query_scans.csv`
- `metadata.json`

The main BST representation gate requires >=90% success retention versus latest-prior voxel surfels, >=40% local primitive reduction, and >=1.20x median total speedup. Historical catalog storage has a separate >=20% reduction gate. A strict positive latest-prior-vs-static success gain is reported separately rather than assumed.
