# BST-ID Lifelong Mapping — Experiment Specification v0.1

## Objective

Determine whether a BST-ID-based spatiotemporal map representation provides a better **storage–query tradeoff** than conventional spatial map structures with explicit temporal history, while preserving equivalent historical map information.

The experiment isolates **map representation and querying only**. SLAM accuracy, registration, semantic understanding, persistence estimation, and localization are excluded at this stage.

## Research Hypotheses

**Null hypothesis \(H_0\):**

\[
\text{BST-ID provides no meaningful storage/query advantage over an Octree with temporal histories.}
\]

**Research hypothesis \(H_1\):**

\[
\text{BST-ID improves the storage/query Pareto frontier for spatiotemporal map histories.}
\]

A later experiment will extend \(H_1\) using adaptive spatial and temporal resolution.

## Input Representation

Every observation is represented as

\[
o_i=(x_i,y_i,z_i,t_i,s_i)
\]

where:

- \(x_i,y_i,z_i\): metric world coordinates,
- \(t_i\): integer mapping-session identifier,
- \(s_i\in\{0,1\}\): voxel occupancy state.

Initial voxel resolution:

\[
r=0.5\text{ m}.
\]

All methods receive exactly the same aligned observations.

## Methods

### A — Snapshot Baseline

Store a complete independent occupancy map for every session:

\[
M_1,M_2,\ldots,M_N.
\]

### B — Octree + Temporal History

Use a spatial Octree:

\[
\text{Octree}(x,y,z)
\]

with each leaf containing temporal state transitions:

\[
[(t_1,s_1),(t_2,s_2),\ldots].
\]

### C — BST-ID Spatiotemporal Map

Encode observations using:

\[
ID_i=BST(x_i,y_i,z_i,t_i).
\]

Initial experiments use fixed spatial and temporal resolution.

Adaptive resolution is deliberately excluded until the basic representation has been validated.

## Required Operations

Every representation must implement identical logical operations:

\[
\texttt{insert}(x,y,z,t,state)
\]

\[
\texttt{snapshot}(t)
\]

\[
\texttt{region\_at\_time}(R,t)
\]

\[
\texttt{changes}(R,t_a,t_b).
\]

`region_at_time` reconstructs the map within spatial region \(R\) at time \(t\).

`changes` returns spatial cells whose state differs between \(t_a\) and \(t_b\).

## Correctness Requirement

For every query:

\[
M_{snapshot}(R,t)
=
M_{octree}(R,t)
=
M_{BST}(R,t)
\]

at the selected voxel resolution.

Performance measurements are valid only when returned map states are equivalent.

## Query Scales

Test spatial regions approximately equal to:

\[
5\times5\times3\text{ m}
\]

\[
20\times20\times5\text{ m}
\]

\[
100\times100\times10\text{ m}
\]

and the complete environment.

Historical queries should be evaluated at recent, middle, and old sessions.

## Primary Metrics

Measure:

\[
S=\text{stored bytes}
\]

\[
T_{insert}=\text{session insertion/update latency}
\]

\[
T_{snapshot}=\text{full historical reconstruction latency}
\]

\[
T_{local}=\text{localized historical-query latency}
\]

\[
T_{change}=\text{regional change-query latency}.
\]

Also record the number of map records/nodes inspected per query.

## Initial Go/No-Go Criterion

Continue development if BST-ID demonstrates at least one meaningful advantage against the Octree temporal baseline, such as:

\[
T_{local,BST}\le0.5T_{local,Octree}
\]

at comparable storage and identical output,

or materially lower storage at comparable query performance,

or a clearly superior storage/query Pareto frontier.

Improvements of approximately 5–10% alone will not be considered sufficient evidence.

## Initial Synthetic Environment

Environment dimensions:

\[
100\times100\times5\text{ m}.
\]

Voxel size:

\[
0.5\text{ m}.
\]

Generate 100 initial mapping sessions, later scaling to:

\[
10,\ 25,\ 50,\ 100,\ 250,\ 500,\ 1000
\]

sessions.

Different spatial regions will have deliberately different temporal change frequencies so that the effect of temporal structure can be measured.

## Scope Boundary

The experiment does **not** initially evaluate:

- SLAM accuracy,
- sensor noise,
- map registration,
- change-detection algorithms,
- semantics,
- learned models,
- ROS,
- FAST-LIO/FAST-LIVO2,
- localization accuracy.

Those are introduced only after the map representation itself demonstrates measurable value.