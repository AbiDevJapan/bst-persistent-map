# BST Lifelong Mapping Prototype — through Step 9.2

This package contains the cumulative prototype through **Step 9.2: frozen multi-session localization validation**.

The representation and localization operating point are frozen from Step 9.1.3. Step 9.2 does not tune the map. It builds the five-session Jan-15 reference map once and validates the deployed compact K=4 localization view against conventional 0.5 m voxel surfels on held-out scans from Feb, Mar, May, and Jun 2012.

Run the new experiment with:

```bash
python3 run_step9_2.py \
  --data-root data \
  --query-count-per-session 50 \
  --exclude-query-scans-csv results/step9_1_2_hierarchical_lod/query_scans.csv \
  --output results/step9_2_multisession
```

See `README_STEP9_2.md` for the frozen protocol, gates, and output definitions. Earlier step READMEs and runners are retained for reproducibility.

Important scientific caveat: `PrototypeBSTSpatialCodec` remains a local research codec, not a verified upstream BST-ID codec.
