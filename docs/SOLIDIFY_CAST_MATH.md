# Surface completion and cast construction

This reference describes the implemented 0.4 geometry path. It does not establish novelty or
claim that an incomplete scan has a uniquely recoverable shape. See the [tool guide](SOLIDIFY_AND_CAST.md)
for controls and the [source acknowledgements](../SOURCES.md) for incorporated dependencies.

## Preserve measured facets before filling missing volume

TetGen constructs constrained tetrahedra around the input facets. Input coordinates and facet
membership are checked before accepting the volume mesh. A tetrahedron receives an inside or
outside label. A boundary is extracted where neighboring labels differ.

$$
V_i=\frac{1}{6}\left|\det(\mathbf b_i-\mathbf a_i,\mathbf c_i-\mathbf a_i,\mathbf d_i-\mathbf a_i)\right|,
\qquad A_f=\frac{1}{2}\| (\mathbf b_f-\mathbf a_f)\times(\mathbf c_f-\mathbf a_f)\|.
$$

| Symbol | Meaning |
| --- | --- |
| $V_i$ | Volume of tetrahedron $i$ |
| $A_f$ | Area of triangular face $f$ |
| $\mathbf a,\mathbf b,\mathbf c,\mathbf d$ | Coordinates of that tetrahedron or face's vertices |
| $\det$ | Signed volume determinant; absolute value removes orientation dependence |

For each measured triangle, its normal and the adjacent tetrahedron centers determine supported
inside and outside labels. These are hard constraints rather than a smoothing preference.

$$
s_{if}=(\mathbf c_i-\mathbf m_f)\cdot\mathbf n_f.
$$

| Symbol | Meaning |
| --- | --- |
| $\mathbf c_i$ | Tetrahedron center |
| $\mathbf m_f$ | Measured triangle center |
| $\mathbf n_f$ | Measured triangle's oriented normal |
| $s_{if}$ | Negative means the center is behind the face; positive means it is in front |

This requires usable surface orientation and a feasible constrained volume. Contradictory or
intersecting evidence can prevent completion; it is not evidence that a guessed surface is correct.

## Combine surface support and a compact completion boundary

The initial volume labeling minimizes a graph-cut objective:

$$
E(\ell)=\sum_i u_i(\ell_i)+\sum_{(i,j)}A_{ij}[\ell_i\ne\ell_j]
          +\sum_{f\in H}A_f\ell_{o(f)},
\qquad
\lambda=\frac{\sum_{f\in F}A_f}{\sum_i V_i}.
$$

$$
u_i(\ell_i)=\lambda V_i[\ell_i\ne q_i].
$$

| Symbol | Meaning |
| --- | --- |
| $\ell_i$ | Chosen inside label (1) or outside label (0) |
| $q_i$ | Inside/outside preference from signed surface distance |
| $u_i$ | Cost of disagreeing with that distance-based preference |
| $A_{ij}$ | Shared face area between neighboring tetrahedra |
| $F$ | Measured input facets |
| $H$ | Faces on the outer tetrahedral hull |
| $o(f)$ | Tetrahedron owning a hull face |
| $[P]$ | 1 when statement $P$ is true, otherwise 0 |
| $\lambda$ | Converts volume-weighted disagreement to the area scale of the boundary cost |

The hull term penalizes placing material against the outer hull. Measured facet-side constraints
override soft preferences. The implementation chooses a constraint penalty larger than the sum
of finite costs. PyMaxflow solves the initial labeling.

A short boundary alone does not guarantee a manifold surface. Ambiguous neighborhoods are
subsequently constrained through Python-SAT/RC2. Edge incidence and connected vertex fans are
checked, and generated contacts are corrected without freely moving the measured surface.
Final boundary and intersection validation uses PyMeshFix. A failed validation remains a failure.

## CPU and GPU roles

Closest-surface queries can run through Warp on a selected compatible GPU, using angle-weighted
pseudonormal signed distance. The CPU fallback uses VTK's implicit surface distance. Tetrahedralization,
graph construction, graph cuts, and local topology constraints remain separate stages. A fast
GPU query stage does not imply that the complete operation is GPU-bound or that GPU and CPU
classifications are interchangeable on every ambiguous input.

The implementation relates to [robust inside/outside segmentation](https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.html)
and [angle-weighted pseudonormal distance](https://doi.org/10.1109/TVCG.2005.49).
These are related methods and implementation references, not claims of inventing their algorithms.

## Clearance, cavity, and shell

Cast measures supported exterior depth and constructs a cavity on a geometry-derived sampling grid.
Missing or back-facing observations are treated as unknown instead of automatically declaring free
space. The control pipeline can first use Solidify for an open input. Gap bridging uses morphological
closing; exterior construction uses a sampled fitted hull or box and the selected wall thickness.

$$
r_g=\left\lceil\frac{g}{2h}\right\rceil,
\qquad C'=\mathrm{close}_{r_g}(C),
\qquad M=O\setminus C'.
$$

| Symbol | Meaning |
| --- | --- |
| $h$ | Sampling pitch in model coordinates |
| $g$ | Requested gap-bridging distance |
| $r_g$ | Integer grid radius used by the closing kernel |
| $C$ | Measured cavity occupancy |
| $C'$ | Cavity occupancy after gap bridging |
| $O$ | Outer-body occupancy after wall construction |
| $M$ | Material remaining around the cavity |

Grid extraction is approximate at the sampling pitch. Clearance, gap bridging, and wall thickness
serve different purposes; larger bridging can remove intentional recesses. Closed topology alone
does not prove an accurate fit or sufficient physical wall thickness for a given printer.

## Split geometry and continued reduction

$$
\pi(\mathbf x)=(\mathbf x-\mathbf o)\cdot\mathbf n.
$$

| Symbol | Meaning |
| --- | --- |
| $\mathbf o$ | Split-plane origin |
| $\mathbf n$ | Split-plane normal |
| $\pi(\mathbf x)$ | Signed side of a point relative to the split plane |

The body is clipped and capped on each side, including its cavity. Halves remain separate connected
parts. Post-generation reduction uses the selected QEM path and closed-output validation; the
reducer is not the volume-completion algorithm. See the [algorithm reference](ALGORITHMS.md).

## Export preserves coordinates separately from edit-friendly local coordinates

For COLLADA, a component-local origin avoids an insertion point far from its geometry. A local
scale keeps small faces above the downstream importer's face-creation tolerance:

$$
\mathbf p_{local}=100(\mathbf p-\mathbf o),
\qquad \mathbf p_{world}=\mathbf o+0.01\mathbf p_{local}.
$$

Here $\mathbf o$ is the bounding-box center. The two transforms preserve world coordinates
algebraically. Export/import tests check bounds, units, closure, and surface coordinates; importer
cleanup can still alter geometry, so native application checks remain distinct from file round trips.
