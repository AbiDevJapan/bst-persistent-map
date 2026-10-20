## Step 11.1 — Real NCLT correction baseline

See [README_STEP11_1.md](README_STEP11_1.md) for timestamped NCLT source ingestion, provenance and exact pose-correction experiments. The existing Step 11 and Step 10 workflows remain available.

# Lifelong 3D Mapping — Correction-Aware Research Branch

**New research starting point: [README_STEP11.md](README_STEP11.md)**

The new primary CLI is `run_correction_map.py`. It adds immutable local-source measurements, robot and source provenance, pose revision history, correction-aware incremental updates, direct historical map revisions, and compressed bitmap/sparse occupancy pages. The mapping/storage index is deliberately codec-neutral so BST-ID can be compared fairly with Morton and spatial hashing.

**Preserved, runnable baselines:**

- Step 10.0–10.5B historical snapshots, deltas, interval 4D, and adaptive geometry: `run_versioned_map.py` and `README_STEP10.md` through `README_STEP10_5B.md`.
- Step 9 localization: `run_localization.py`, `localization/`, and `README_STEP9_7J.md`.
- Original author-supplied BST-ID Python code under `third_party/bstid_author/`, unmodified.

The new correction core is a correctness-first baseline, **not yet a dynamic BST-ID correction algorithm or a proven NCLT speedup**.
