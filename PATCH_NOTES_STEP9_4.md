# Step 9.4 non-finite historical-surfel patch

This patch fixes a failure observed while building the 2012-03-17 latest-prior surfel map:
`RuntimeWarning: invalid value encountered in cast` followed by
`numpy.linalg.LinAlgError: Eigenvalues did not converge`.

The fix is deliberately input-safety only; it does not change the frozen K=4 profile,
ICP parameters, causal latest-prior plan, query set, or evaluation gates.

Changes:
- reject a scan when its matched 6-DoF ground-truth pose contains a non-finite value;
- remove non-finite transformed returns before voxel indexing and moment accumulation;
- exclude non-finite sufficient statistics from surfel normal/covariance eigendecomposition;
- if a batched eigendecomposition still fails, isolate the offending 3x3 covariance instead of aborting the whole historical map;
- apply the same finite-input contract to fresh packed-occupancy builds and the legacy tuple ingest path;
- add regression tests for non-finite point ingestion and poisoned covariance statistics.

Existing Step-8 occupancy caches are still accepted; the patch does not force a cache rebuild.

## Resume support

The patched CLI adds `--resume-checkpoint`.  It reloads completed trials and historical-map
statistics from the existing output directory and skips prior sessions whose checkpoint was
written.  A checkpoint implies materialization verification passed because the original 9.4
code writes the checkpoint only after that assertion and after the associated localization
trials.  For recovered prior sessions, `materialization_verification.csv` records
`verification_source=checkpoint_implies_pass` and `query_checks=-1` because the old checkpoint
did not store the exact number of verification centers.
