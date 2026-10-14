# Tiled processing

Tiled processing bounds peak geometry memory by keeping the source STL in a disk-backed spatial
index. Each adaptive octree leaf is loaded, optimized, validated, and returned to disk separately.
The final binary STL is assembled sequentially without constructing a global vertex or face array.

## Current method

Triangles belong to the octree leaf containing their centroid. For tile $i$, with $n_i$ source
triangles, the target allocation starts with a four-triangle floor $m_i=\min(4,n_i)$. The remaining
global budget is distributed by largest remainder:

$$
q_i=(T-\sum_j m_j)\frac{n_i-m_i}{\sum_j(n_j-m_j)},
\qquad
t_i=m_i+\lfloor q_i\rfloor+r_i,
$$

where $r_i\in\{0,1\}$ is assigned in descending order of the fractional part of $q_i$ until
$\sum_i t_i=T$. Tile order breaks equal-fraction ties, so the allocation is deterministic.

where:

- $t_i$ is the tile target;
- $n_i$ is the tile source count;
- $m_i$ is the minimum allocation; and
- $T$ is the requested global target.

The allocated targets sum exactly to $T$. The final output may exceed $T$ when a tile needs more
triangles to preserve its protected boundary. The production result reports this difference rather
than discarding boundary geometry to force an exact count.

For a tile edge $e$, let $c_i(e)$ be its triangle incidence count inside tile $i$. The protected
vertex set is:

$$
B_i = \left\{v : v \in e \land c_i(e)=1\right\}.
$$

GPU surface projection keeps these vertices fixed:

$$
\mathbf{x}'_v = \mathbf{x}_v \quad \text{for every } v \in B_i.
$$

Boundary-preserving QEM then treats the same open edges as constrained boundaries. This is
deliberately conservative. It prevents independently processed tiles from moving shared source
coordinates, but may retain more geometry near tile boundaries than a future shared-boundary
optimizer.

## Reference measurement

The 4,126,315-triangle reference mesh was partitioned into 50 adaptive tiles. The requested target
was 249,184 triangles. Boundary protection retained 495 additional triangles, producing 249,679.
Index reuse excluded index construction from the optimization time.

| Measurement | In-memory Fast QEM | Forced tiled Fast QEM |
|---|---:|---:|
| Optimization time | 8.58 s | 4.67 s with 8 workers |
| Sequential tiled time | N/A | 27.34 s |
| Parallel speedup | N/A | 5.85x |
| Cached validation and assembly | N/A | 0.22 s |
| Output triangles | 249,184 | 249,679 |
| RMS surface error | 0.114% | 0.121% |
| P95 surface error | 0.227% | 0.236% |
| Dimension drift | 0.068% | 0.005% |
| Surface-area change | 0.399% | 0.095% |
| Non-manifold rate | 0.020% | 0.017% |

Sequential, parallel, and cached runs produced the same 12,484,034-byte STL with SHA-256
`00455d33955b99149db58e70293484cacbc71372879ea61bcd693abd2133c713`. The assembled STL count
matched the sum of all validated tile outputs in every run, with zero triangles lost during
assembly. A connected 67,600-triangle grid fixture retained the source boundary-edge count after
cross-tile reduction, proving that the tested partition did not create open tile seams. The same
fixture verifies cancellation cleanup, checkpoint resume, corrupt-checkpoint rebuilding, and
byte-identical sequential and parallel output.

The implementation limits concurrent workers by both the configured process count and memory
budget. Index construction keeps at most 64 tile files open. Tile results and run manifests are
written atomically with checksums. Final STL assembly is streamed, cancellable, and published only
after its record count is verified.

## Publication threshold

Octrees, centroid partitioning, boundary constraints, proportional target allocation, and streamed
STL assembly are established techniques. The current evidence supports a reproducible engineering
report, not a novelty claim. A later research publication should make a novelty claim only if a
distinct interconnect method survives matched-count, adversarial seam, and independent replication
tests.

Run `python benchmarks/benchmark_tiled_processing.py --help` for the reproducible timing and
assembly-consistency benchmark. The automated contract tests are in `tests/test_out_of_core.py`.
