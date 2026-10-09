# Step 9.0.2 — controlled localization diagnostics + coarse-to-fine point-to-plane ICP

Step 9.0.2 is a **diagnostic refinement of the conventional voxel localization harness**. It still does not consume native BST blocks.

Step 9.0.1 showed that point-to-plane ICP does move the pose on real cross-session NCLT data, but the fixed success gate was not met consistently. Step 9.0.2 separates three possible causes before any native-block localization is attempted:

1. the cross-session map/ground-truth consistency floor;
2. a matcher implementation/capture-range problem;
3. actual session-to-session geometry differences.

## What is new

### A. Cross-session zero-perturbation control

For every frozen cross-session query scan, Step 9.0.2 also runs:

```text
initial pose = NCLT reference pose
```

and records how far the point-to-plane objective moves the pose away from that reference.

This gives an empirical registration floor for this map/session pair. If zero-initialized scans systematically settle 0.3–0.5 m away from reference pose, a 0.5 m success gate is close to the objective/data floor rather than purely an optimizer test.

The zero control is appended **without consuming random numbers**, so the four nonzero cross-session perturbations keep the exact Step-9.0/9.0.1 RNG sequence when `--query-scans-csv` and perturbation settings are unchanged.

### B. Same-session off-stride control

A second scenario uses the map session itself for query scans. To avoid the trivial case where a query scan is one of the scans directly aggregated into the stride-sampled map, control queries satisfy:

```text
scan_index % map_scan_stride != 0
```

For the normal `--scan-stride 10` map, none of these query scans is directly one of the every-10th scans used to construct the map.

This same-session experiment is an **optimistic diagnostic control**, not a final localization benchmark. Strong same-session recovery together with weaker cross-session recovery points to real temporal/map inconsistency rather than a broken optimizer.

### C. Three-level coarse-to-fine point-to-plane ICP

Default pyramid:

| level | scan sampling | map sampling | max correspondence | iterations |
|---|---:|---:|---:|---:|
| coarse | 1.50 m | 1.00 m | 4.0 m | 15 |
| medium | 1.00 m | 0.75 m | 2.0 m | 15 |
| fine | 0.75 m | 0.50 m | 1.0 m | 20 |

Each level uses the final pose from the previous level. PCA map normals are still estimated lazily, now within each pyramid level. The output CSV records level-specific correspondence counts, normal counts, residuals, iterations, timings and failure reasons.

The pyramid can be overridden with:

```text
--pyramid name:scan_voxel:map_voxel:max_corr:iterations,...
```

## Frozen main cross-session protocol

The main comparison remains:

- map: `2012-01-15`;
- cross-session queries: `2012-02-02`;
- 20 query scans;
- perturbations: `0.5:1,1:3,2:5,5:10`;
- perturbation seed: `42090`;
- optimized DoF: x, y, yaw;
- z, roll and pitch fixed from NCLT reference pose;
- local map retrieved around each trial's initial pose;
- nominal success gate: <= 0.5 m translation and <= 1 degree yaw.

## Recommended run

Replay the exact previous cross-session scan list:

```bash
python3 run_step9_0_2.py \
  --data-root data \
  --map-session 2012-01-15 \
  --query-session 2012-02-02 \
  --query-scans-csv results/step9_0_localization/query_scans.csv \
  --voxel-size 0.5 \
  --scan-stride 10 \
  --query-count 20 \
  --same-session-query-count 20 \
  --perturbations 0.5:1,1:3,2:5,5:10 \
  --localization-max-range 30 \
  --scan-downsample 0.75 \
  --max-scan-points 8000 \
  --local-map-radius 40 \
  --output results/step9_0_2_controls
```

The existing `.cache/step8_1` map cache is reused by default. No new NCLT session is required.

If the old scan-list file lives elsewhere, pass its path to `--query-scans-csv`.

## Outputs

- `localization_trials.csv`
- `localization_summary.csv`
- `query_scans.csv`
- `metadata.json`

New important fields include:

- `scenario`: `cross_session` or `same_session_control`;
- `pose_shift_from_initial_m` / `yaw_shift_from_initial_deg`;
- per-level `level_coarse_*`, `level_medium_*`, `level_fine_*` diagnostics;
- `original_scan_index`;
- `directly_ingested_scan` in `query_scans.csv`.

For `same_session_control`, `directly_ingested_scan` should always be false.

## How to interpret the result

### Case 1 — zero control is already near the success boundary

If cross-session `zero_control` moves away from GT by, for example, 0.3–0.5 m, then the current nominal 0.5 m gate is close to the registration/data floor. Evaluate perturbation recovery relative to this floor before declaring the matcher broken.

### Case 2 — same-session succeeds, cross-session does not

This supports the interpretation that the matcher is operational but cross-session geometry / voxelization / reference-pose inconsistency limits absolute GT error.

### Case 3 — same-session also fails badly

Do not move to native BST localization. The conventional matcher still needs work.

### Case 4 — coarse stages improve large perturbations but fine stage stalls near the zero floor

The capture-range problem is solved; remaining error is likely dominated by the cross-session objective/data floor rather than initialization.

## Gate before Step 9.1

Do **not** apply the old 90%/80% success gate blindly before looking at zero-control error.

A practical Step-9.1 go decision should require both:

1. same-session control demonstrates that the conventional coarse-to-fine matcher can recover the small perturbations reliably; and
2. cross-session nonzero trials converge to errors comparable to the observed cross-session zero-control floor.

This is a diagnostic step. It does not change any BST representation, storage/index result, or Step-8.x conclusion.
