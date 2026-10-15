# Step 9.5 — Global-to-local lifelong relocalization

Step 9.5 replaces the deliberately weak local-ICP initialization experiment with a modern coarse-to-fine relocalization stack while keeping the lifelong map representation frozen.

## Pipeline

```
query LiDAR scan
    ↓
Scan Context place retrieval over the causal latest-prior mapping session
    ↓ top-K prior-session reference scans
KISS-Matcher scan-to-scan global registration
    ↓
small_gicp GICP scan-to-scan refinement
    ↓ world-pose hypothesis from the stored reference-scan map pose
latest-prior local-map retrieval
    ├── full 0.5 m voxel-surfel map
    └── frozen compact BST K=4 map
    ↓
small_gicp 6-DoF GICP map refinement
```

The query is **not given a pose perturbation or local initial pose**.  Scan Context determines place candidates; KISS-Matcher solves the hard rigid scan-to-scan alignment; small_gicp performs 6-DoF refinement.  The exact same frontend hypothesis is passed to the full voxel and compact BST maps so the map-representation comparison remains controlled.

KISS-Matcher is intentionally run scan-to-scan rather than scan-to-fused-map.  This keeps source/target sampling domains similar and lets the historical map be used where it is strongest: final local refinement.

## External dependencies

Step 9.5 adds two research dependencies:

```bash
pip install "kiss-matcher>=1.0.1,<1.1" "small-gicp>=1.0.1,<1.1"
```

Both projects provide Python packages; Step 9.5 imports them lazily and fails **before expensive BST/map construction** if either is unavailable.

## Recommended paired run

Reuse the exact 200 scans from the Step-9.4 capture-basin run.  This makes the comparison against the old local-only matcher paired rather than changing the test scans.

```bash
python3 run_step9_5.py \
  --data-root data \
  --history-sessions 2012-01-15,2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --map-session 2012-01-15 \
  --query-sessions 2012-02-02,2012-03-17,2012-05-11,2012-06-15 \
  --query-count-per-session 50 \
  --query-scans-csv results/step9_4_capture_basin/query_scans.csv \
  --output results/step9_5_global_to_local
```

There is deliberately no `--perturbations` option.  This is a global relocalization experiment.

## Default frontend configuration

- Scan Context: 20 rings × 60 sectors, 80 m radius.
- Reference keyframes: candidate scans every 5 raw scans; retain a keyframe after 5 m translation, 15° yaw, or 5 s.
- Retrieve top 5 Scan Context candidates from a 30-candidate ring-key pool.
- KISS-Matcher resolution: 0.30 m.
- Scan-to-scan GICP: 0.30 m downsampling, 2.0 m max correspondence.
- Final map GICP: 0.40 m downsampling, 2.0 m max correspondence, 50 m local-map radius.
- Frozen BST localization representation: balanced geometry gate, K=4.

These are fixed defaults for the first validation.  Do not tune them separately for individual NCLT sessions.

## Metrics

Step 9.5 reports three success definitions:

1. `success`: current paper-continuity criterion, XY error ≤0.5 m and yaw error ≤1°.
2. `global_2m5deg_success`: 3-D translation ≤2 m and full SO(3) rotation error ≤5°.
3. `strict_6d_success`: 3-D translation ≤0.5 m and full SO(3) rotation error ≤2°.

It separately reports:

- Scan Context top-1/top-K place recall at 5/10/20 m;
- KISS + scan-GICP pose error before map refinement;
- final voxel-map and BST-map pose errors;
- local primitive count;
- frontend, retrieval, local-GICP and total latency.

## Main outputs

- `frontend_trials.csv`
- `place_recognition_summary.csv`
- `localization_trials.csv`
- `localization_summary.csv`
- `session_summary.csv`
- `reference_database_stats.csv`
- `map_geometry_stats.csv`
- `representation_gates.csv`
- `query_scans.csv`
- `metadata.json`

## Interpretation

A localization gain from Scan Context/KISS/GICP is a **frontend/system contribution**, not a BST-ID contribution.  The BST claim remains the controlled comparison against the full latest-prior voxel map using the same global hypothesis: success retention, local primitive reduction, runtime, and historical storage.

The current `PrototypeBSTSpatialCodec` is still the local research codec.  Replacing it with the author-approved BST-ID artifact remains a separate integration/revalidation step.

## macOS OpenMP isolation (9.5.1)

On macOS arm64, the PyPI wheels for KISS-Matcher and small_gicp may load distinct
copies of `libomp.dylib`. Importing both in one Python interpreter can abort with
`OMP: Error #15`. Do **not** use `KMP_DUPLICATE_LIB_OK=TRUE` for benchmark runs.

Step 9.5.1 keeps `small_gicp` in the main benchmark process and runs KISS-Matcher
inside one persistent `spawn` worker process. The KISS worker is reused for all
candidate registrations, so startup is paid once. IPC time is included in the
reported KISS runtime. This isolates the two OpenMP runtimes without changing the
registration algorithms or scientific protocol.

Run `python3 check_step9_5_deps.py` before NCLT. A successful check ends with:
`Step 9.5 dependency/API smoke test PASSED (KISS isolated from small_gicp)`.
