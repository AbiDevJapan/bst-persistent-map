# Step 9.5.1 OpenMP-safe patch

- Fixes macOS `OMP: Error #15` when KISS-Matcher and small_gicp load separate OpenMP runtimes.
- KISS-Matcher now runs in a persistent spawned worker process.
- small_gicp remains in the parent process for low-overhead scan/map refinement.
- `require_frontend_dependencies()` presence-checks KISS without importing it in the parent.
- `check_step9_5_deps.py` now validates the isolated architecture.
- `KMP_DUPLICATE_LIB_OK=TRUE` is intentionally not used.
- KISS IPC overhead is included in `runtime_ms`.
