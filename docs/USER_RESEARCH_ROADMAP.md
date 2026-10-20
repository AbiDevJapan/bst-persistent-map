# BST-ID: Project Understanding and Research Roadmap

Research direction selected

Project: Correction-Aware Historical Global Mapping Using BST-ID Starting codebase: [bst-persistent-map — test-distributed-system](https://github.com/AbiDevJapan/bst-persistent-map/tree/test-distributed-system?utm_source=chatgpt.com) Publication goal: ICORE B-rank conference or higher Primary contribution: Efficient global map correction and historical consistency, with localization accuracy preserved.

I've now examined the actual repository, including its Step 10 versioned-map implementation, BST-ID indexing, localization frontend, and associated tests.

My understanding has changed in one important way: we already have much of the historical map infrastructure. The central research task is to make that infrastructure correction-aware, not to rebuild it from scratch.

## 1. What I understand the project to be

We want to create a globally referenced map shared by multiple robots. Every robot can contribute observations, localize against the resulting map, and eventually receive corrections to poses estimated in the past.

When a historical robot pose changes, the map must be updated without discarding the history of what was previously believed.

Our proposed contribution is a BST-ID-based method for performing those operations efficiently.

Robot A

Robot B

Robot C

Observations · Global poses · Corrections · Map queries

Correction-Aware Global BST-ID Map

Spatial-temporal index

Source dependencies

Pose revision history

Adaptive map geometry

Current map

Historical map

Localization map

### The key example

Suppose Robot A collected LiDAR scans yesterday. Those measurements were integrated into the global map using its estimated poses.

Today, loop closure corrects part of Robot A's historical trajectory.

Our system should:

1. Identify exactly which original observations are affected.
2. Determine which global map regions they previously contributed to.
3. Remove their old contributions and insert the corrected ones.
4. Update only the affected map geometry and index entries.
5. Publish a new consistent map revision.
6. Preserve previous map revisions for historical queries.
7. Allow other robots to localize against a complete, internally consistent map revision.

The research question is whether BST-ID can make this more efficient than conventional hierarchical or hashed indexing while maintaining equivalent map and localization quality.

## 2. What the existing repository already provides

The existing implementation contains two major research components.

### A. Localization: Steps 9.x

The refactored localization pipeline already provides global place recognition, initial registration, adaptive BST-ID surfel maps, and final scan-to-map refinement.

Relevant files include:

- [localization/global_frontend.py](https://github.com/AbiDevJapan/bst-persistent-map/blob/test-distributed-system/localization/global_frontend.py?utm_source=chatgpt.com)
- [localization/compact_map.py](https://github.com/AbiDevJapan/bst-persistent-map/blob/test-distributed-system/localization/compact_map.py?utm_source=chatgpt.com)
- [benchmark/localization_benchmark.py](https://github.com/AbiDevJapan/bst-persistent-map/blob/test-distributed-system/benchmark/localization_benchmark.py?utm_source=chatgpt.com)

My intention is to preserve this localization pipeline initially. We should not create a new scan matcher, since that would make it difficult to determine whether a change in accuracy comes from the map representation or localization algorithm.

### B. Historical mapping: Steps 10.x

This part of the repository is our principal starting point.

| Existing module                   | Current capability                                         | What we need to add                              |
| --------------------------------- | ---------------------------------------------------------- | ------------------------------------------------ |
| `versioned_map/store.py`          | Immutable, content-addressed historical map snapshots      | Observation ownership and pose revisions         |
| `versioned_map/delta_journal.py`  | Chronological voxel additions and removals                 | Corrections to previously inserted observations  |
| `versioned_map/interval_store.py` | Historical intervals and 4D indexing                       | Observation-time versus revision-time semantics  |
| `versioned_map/aggregate_4d.py`   | Spatial-temporal tile aggregation                          | Dependency-aware updates                         |
| `versioned_map/geometry_105b.py`  | Adaptive geometry, historical masks, BST/Morton comparison | Incremental corrected geometry                   |
| `versioned_map/region_index.py`   | Spatial query candidate selection                          | Efficient queries across corrected map revisions |

The current `Geometry4D` implementation is particularly useful because it already supports adaptive geometry cells, exact endpoint-voxel reconstruction, and comparisons between BST-ID and Morton indexes.

However, several limitations matter directly to the proposed research:

- Step 10 stores observed LiDAR endpoint voxels, not the original scans and their individual pose dependencies.
- The Step 10 geometry cannot reproduce the Step 9 surfel normals and covariance directly.
- `Geometry4D` currently uses a 32-version membership bitmask.
- Its incremental append handles new chronological sessions; it does not perform retroactive pose correction.
- Appending geometry currently rebuilds index sidecars.
- The spatial domain is a local unsigned integer voxel grid, not an inherently georeferenced global coordinate frame.
- The existing synchronization functionality does not establish a complete concurrent multi-robot mapping system.

These are verified limitations of the inspected code, not evidence that the current implementation is defective. They follow from its original experimental scope.

Our new branch should build on the Step 10 architecture while introducing reversible source contributions and map revisions.

## 3. What we are actually trying to publish

I see three closely related contributions.

| Contribution                   | Research objective                                                                                             |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| C1 — Correction-aware indexing | Identify and update only map regions affected by historical pose corrections                                   |
| C2 — Historical consistency    | Retrieve both the map previously committed and a retrospectively corrected interpretation of past observations |
| C3 — Localization preservation | Demonstrate that incremental correction retains the localization performance of a complete corrected rebuild   |

C1 should be the main algorithmic contribution. C2 is the distinctive functionality that makes the system useful for lifelong mapping. C3 is an essential application-level validation.

We are not claiming that loop-closure correction, immutable versions, or historical databases are individually new. Those ideas already exist.

Our intended novelty is the specific BST-ID-based dependency, correction, and version-indexing algorithm, assuming experiments demonstrate an advantage against equally capable alternatives.

# 4. Project roadmap

I would organize the new branch into nine major milestones, numbered Step 11.0 through Step 11.8.

11.0

Baseline and research specification

Freeze the current implementation and define correction semantics

11.1

Global observations and provenance

Store original measurements, source IDs and their poses

11.2

Full-rebuild correctness reference

Generate the exact reference map from corrected source poses

11.3

Incremental pose-correction algorithm

Correct only affected map regions

11.4

Historical, correction-aware revisions

Preserve committed maps and support retrospective historical queries

11.5

Dynamic BST-ID hierarchy

Maintain adaptive geometry and index entries after corrections

11.6

Competitive performance evaluation

Benchmark against equivalent versioned Morton and spatial hash implementations

11.7

Localization and multi-robot validation

Demonstrate non-inferior localization and consistent multi-source maps

11.8

Paper preparation

Novelty verification, ablations, analysis and submission

## Phase I — Build a provably correct correction engine

Steps 11.0–11.3 · Foundational research

This is where I would invest the most attention initially.

11.0 — Freeze and characterize the starting point

Run and preserve the current Step 10 unit tests, geometry equivalence tests, BST/Morton comparisons, and existing localization regression tests.

Record the baseline commit, datasets, configurations, dependencies, reference outputs, and timing environment.

The GitHub code is accessible now, but I have not executed its tests. We need a reproducible baseline before making modifications.

11.1 — Introduce globally referenced source observations

The core new abstraction should be an immutable scan or submap, stored in its own coordinate frame.

Each source must have a stable identifier, observation timestamp, source robot, estimated global pose, and enough geometric information to reconstruct its global contributions.

Start with a metric global frame and stable signed integer spatial coordinates. Physical georeferencing can be added separately.

The design should support multiple robots without requiring network communication yet.

11.2 — Implement deterministic full reconstruction

Before optimizing anything, create a reference implementation that reconstructs the entire global map from original observations and their corrected poses.

Use the same voxelization rules, grid origin, spatial resolution, and numerical precision that incremental correction will use.

This becomes the correctness oracle.

11.3 — Implement incremental pose correction

Develop a dependency index connecting observations to their contributed global cells.

When a submap's pose changes, locate its previous contributions, find its new contributions, and recompute affected cells without touching unrelated regions.

Start with rigid corrections to complete submaps. Later support corrections to individual scans or trajectory segments.

Primary correctness experiment

Method A

Rebuild the whole map from corrected poses

Method B

Incrementally update affected contributions

Equivalent corrected map

Exact endpoint occupancy; geometrically equivalent derived payloads under defined tolerances

Milestone M1 — Correction correctness demonstrated

Exit criterion: The incremental method consistently reproduces the designated full-rebuild map across synthetic tests, overlapping sources, and controlled NCLT trajectory corrections.

This is the foundation for the entire research project.

## Phase II — Develop the novel historical BST-ID mechanism

Steps 11.4–11.5 · Main algorithmic contribution

11.4 — Introduce correction-aware historical revisions

The current repository models historical observation sessions. We need an additional revision dimension that records when the system accepted new information or a correction.

For example:

| Revision | Event                                      | Result                   |
| -------- | ------------------------------------------ | ------------------------ |
| R0       | Robot A maps an area                       | Initial map              |
| R1       | Robot B adds observations                  | Expanded map             |
| R2       | Robot A receives a loop-closure correction | Corrected interpretation |
| R3       | Robot C contributes new observations       | Further expanded map     |

At R3, a query for R1 must still return the map committed at R1.

We also need the ability to query the observations from R1 interpreted with corrections known at R3.

This is the bitemporal capability discussed earlier.

I would extend `VersionedStore`, `DeltaJournal`, and `IntervalMap` rather than replace their concepts. However, the current chronological-session and 32-version-mask designs will need changes to support arbitrary correction revisions at scale.

11.5 — Make BST-ID updates genuinely incremental

Extend the adaptive geometry architecture in `geometry_105b.py`.

The hierarchy should maintain correct spatial coverage when observations move, when regions become empty, and when adaptive cells split or merge.

We also need to avoid rebuilding the entire BST-ID index sidecar after each correction.

That is a potentially important algorithmic contribution.

We should benchmark two alternatives: updating a dynamic BST-ID representation directly, and rebuilding only selected affected hierarchy regions with structural sharing.

Milestone M2 — Correction-aware historical BST-ID operational

Exit criterion: Old map revisions remain queryable, corrected revisions are accurate, and corrections update only their affected map/dependency/index regions.

At this point, we should have our main novel algorithm implemented.

## Phase III — Establish whether the algorithm is publishable

Step 11.6 · Principal experimental evaluation

We must compare BST-ID against competitive alternatives that support the same functionality.

| Baseline                        | Purpose                                   |
| ------------------------------- | ----------------------------------------- |
| Full reconstruction             | Reference correctness and cost            |
| Versioned spatial hash          | Strong incremental-update alternative     |
| Versioned Morton index          | Strong compact spatial-index alternative  |
| BST-ID without adaptive updates | Ablation for the new correction mechanism |
| BST-ID without revision sharing | Ablation for historical efficiency        |

The main evaluation should cover:

- Correction cost: Median and P95 update latency, affected blocks, index maintenance cost.
- Historical cost: Storage overhead, retrieval latency, revision count scalability.
- Spatial efficiency: Work performed relative to the size of the corrected region.
- Memory: Total resident and serialized memory, including original observation storage and dependency indexes.
- Correctness: Incremental map equivalence and historical revision integrity.

One especially important experiment is to increase the total map size while holding the corrected region approximately constant.

If the corrected region stays small, we want BST-ID's update cost to depend mostly on that region, not the full map.

However, a strong spatial hash may also exhibit this property. We therefore need to show a BST-ID-specific benefit, such as improved historical retrieval, index storage, or combined update/query efficiency.

Milestone M3 — Research advantage established

Exit criterion: At least one substantial, reproducible advantage against a strong equivalent baseline, without an unacceptable regression in other operations.

If M3 fails, we should revisit the algorithm before committing to a full multi-robot system.

## Phase IV — Preserve localization and demonstrate multi-robot utility

Step 11.7 · Application validation

Connect corrected map revisions to the existing `localization/` framework.

The critical experimental control is to run the same localization scans with the same frontend, matcher, initialization, and geometric representation wherever possible.

Evaluate localization success, translation error, rotation error, and end-to-end matching latency against a full-reconstruction map.

Because Step 10 stores endpoint occupancy while Step 9 uses surfel statistics, we must introduce a proper surfel reconstruction/materialization layer before claiming localization equivalence. Voxel-center-derived covariance is not a substitute for original LiDAR statistics.

Next, assign observations to several simulated robots and test overlapping map contributions, delayed pose corrections, and consistent revision reads.

NCLT session replay is useful for an initial evaluation, but it does not by itself establish performance under true simultaneous multi-robot SLAM.

Milestone M4 — Localization and multi-robot validity demonstrated

Exit criterion: Localization performance satisfies a predefined non-inferiority test and the shared map remains consistent under multiple contributors and correction histories.

We do not need a fully decentralized SLAM backend for the first paper. A centralized global map service is sufficient if the central research contribution is correction-aware indexing.

## Phase V — Publication

Step 11.8 · Paper development

The final paper should make a narrow, defensible claim:

> A BST-ID-based correction-aware historical map index enables efficient localized updates under retroactive pose revisions, preserves exact historical map semantics, and maintains localization performance.

The paper needs a formal description of the index, correctness invariants, complexity analysis, strong baselines, ablations, and end-to-end localization results.

The most appropriate conference track will depend on which contribution proves strongest: spatial-temporal indexing, lifelong mapping, or collaborative robotic mapping.

# 5. Proposed code changes

I would extend the existing `versioned_map/` package instead of introducing a completely separate top-level global mapping system.

Proposed architecture

New files below are proposals, not files already in the repository.

```
versioned_map/
├── store.py                 # Extend existing
├── delta_journal.py         # Extend existing
├── interval_store.py        # Extend existing
├── geometry_105b.py         # Extend/refactor
├── region_index.py          # Extend existing
│
├── global_frame.py          # New
├── observations.py          # New
├── source_dependencies.py   # New
├── pose_revisions.py        # New
├── correction_engine.py     # New
├── revision_manager.py      # New
├── dynamic_bst.py           # New
│
└── baselines/               # New
    ├── morton_dynamic.py
    └── hash_dynamic.py

localization/
└── revision_map_adapter.py  # New

tests/
├── test_correction_correctness.py
├── test_revision_history.py
├── test_dependency_integrity.py
└── test_localization_noninferiority.py
```

The highest-risk changes are `source_dependencies.py`, `correction_engine.py`, and `dynamic_bst.py`.

The most important modules to preserve as references are the existing static `Geometry4D`, `VersionedStore`, and localization frontend.

# 6. Major milestones and research decision gates

| Milestone                           | What must be demonstrated                                      | Decision                              |
| ----------------------------------- | -------------------------------------------------------------- | ------------------------------------- |
| M0 — Reproducible baseline          | Existing tests and benchmark configurations are frozen         | Begin new development                 |
| M1 — Correct incremental correction | Incremental result agrees with full reconstruction             | Continue with revision semantics      |
| M2 — Versioned BST-ID algorithm     | Exact historical queries and localized index updates           | Begin full comparative experiments    |
| M3 — Competitive advantage          | Meaningful result versus strong Morton/hash baselines          | Decide whether paper thesis is viable |
| M4 — Localization preservation      | Statistically supported non-inferior localization              | Establish application validity        |
| M5 — Research submission            | Complete experiments, theory, ablations and novelty comparison | Submit to a suitable B+ conference    |

I consider M3 the decisive publication gate.

Successfully implementing correction-aware maps is an engineering achievement. Demonstrating that the new algorithm offers a defensible benefit over strong alternatives is what could make it a research paper.

## 7. The immediate next milestone

I would begin by freezing the baseline and implementing the first new data model.

Milestone M0–M1 checklist

0/7 completed

Create a research branch from test-distributed-system

Run and archive current Step 10 tests and localization regression results

Define stable global frame and source observation IDs

Define pose-correction and revision semantics

Implement immutable scan/submap storage

Implement deterministic full-map reconstruction

Create the first synthetic pose-correction equivalence tests

## Final assessment

I now see this as a more focused and technically grounded project than our initial proposal.

We already have:

- A functioning historical map storage and indexing research framework.
- An adaptive BST-ID representation with comparative baselines.
- A separately developed global localization pipeline.
- Unit tests and benchmark harnesses that can be preserved.

What we need to invent is the connection between immutable source measurements, retroactive pose corrections, adaptive spatial indexing, and historical consistency.

I would deliberately keep network optimization and fully decentralized operation outside the first paper's critical path.

The most important outcome of this research would be an algorithm capable of answering two questions efficiently:

1. When historical robot poses change, which parts of the global map and its index must be corrected?
2. How can we maintain and retrieve both the previously committed and newly corrected map states without rebuilding the complete history?

If we can answer those questions with a novel BST-ID mechanism, demonstrate a meaningful efficiency advantage, and preserve localization accuracy, we have a credible foundation for an ICORE B-rank research paper.