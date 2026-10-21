## Step 11.3 — Batched correction deltas (current)

Start at [README_STEP11_3.md](README_STEP11_3.md) for the lossless, schema-compatible correction update optimization, focused tests, and minimal real-NCLT smoke check.

## Step 11.2 — Light scalability diagnostic

See [README_STEP11_2_LIGHT.md](README_STEP11_2_LIGHT.md) for the earlier 64/128/256 NCLT locality smoke experiment.

## Step 11.1 — Real NCLT correction baseline

See [README_STEP11_1.md](README_STEP11_1.md) for timestamped NCLT source ingestion, provenance, and exact pose-correction experiments. All earlier Step 10 and Step 9 modules remain available.

# Lifelong 3D Mapping — Correction-Aware Research Branch

**Architecture overview: [README_STEP11.md](README_STEP11.md)**

Primary CLI: `run_correction_map.py`; real NCLT: `run_nclt_correction.py`; scaling: `run_correction_scaling.py`; current tiny smoke test: `run_step113_smoke.py`. Source local measurements are immutable and source poses are revisioned. The codec-neutral map core uses exact source contribution counts and immutable compressed occupancy pages; BST-ID is retained as an *independent comparison baseline*, not coupled to the correction algorithm.

Existing Steps 9.x and 10.x preserved for reproducibility. No claim is made of end-to-end SLAM, localization quality, or state-of-the-art indexing superiority.
