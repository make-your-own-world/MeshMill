# 0.3.1 development performance comparison

## Scope and conditions

The comparison used the installed Windows 0.3.0 application and the local 0.3.1 development
candidate. Each GUI measurement started a new process and used the same unoptimized STL,
selected CUDA GPU, reduction method, target, and display mode. Runs were sequential rather than
concurrent. Filesystem and CUDA kernel caches were retained; these are not first-install or
cold-storage measurements.

- Input: 3,154,411 triangles and 1,609,643 vertices.
- Algorithm: Depth + Density Adaptive QEM, Balanced quality.
- Requested target: 312,592 triangles.
- Result: 312,516 triangles and 188,773 vertices.
- Processing: full mesh, on the same machine and selected GPU.
- GUI endpoint: a verification image written after the completed preview is displayed. Both
  versions use the same verification path, including its 750 ms settling delay and capture work.
- The input is an additional unoptimized test scan, not the bundled composite sample. The
  measurements below do not describe all mesh sizes or reduction targets.

The candidate GUI ran from local source, while the baseline was the installed executable.
End-to-end differences therefore include packaging/startup effects as well as workflow changes.
The command-line comparison and isolated display measurements help distinguish those effects
from reduction speed.

## Matched GUI runs

All values are elapsed seconds from process launch to the verification endpoint.

| Display | Installed run 1 | Installed run 2 | Candidate run 1 | Candidate run 2 | Installed mean | Candidate mean | Less elapsed time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wireframe | 25.37 | 25.26 | 23.06 | 23.02 | 25.31 | 23.04 | 9.0% |
| Density | 26.18 | 26.21 | 23.00 | 22.49 | 26.20 | 22.74 | 13.2% |

Two runs per display establish a repeatable observation on this fixture, not a statistical
performance guarantee. Other workloads, storage, drivers, mesh characteristics, and cache state
can change the result. The Density comparison includes the complete workflow; its percentage
is not the speedup of the GPU density kernel alone.

## Isolated density visualization

The candidate's CPU reference and GPU display paths were run on identical coordinate and face
arrays with the same normalization and palette calculation.

| Mesh | CPU | First GPU call | Subsequent GPU calls | Warmed speedup |
| --- | ---: | ---: | ---: | ---: |
| Full input | 1.498 s | 1.246 s | 0.453 / 0.455 s | 3.3x |
| Reduced result | 0.158 s | 0.049 s | 0.048 / 0.048 s | 3.3x |

The first full-input GPU call includes process-local CUDA initialization. The reduced result
was tested afterward in the same process, so its first listed call is not a cold CUDA startup.
GPU timings include input transfer, calculation, and returning the display field to the CPU.

Full-input colors differed from the CPU reference by at most one channel level out of 255 on
24 vertices. Reduced-result colors differed by at most one channel level on one vertex.
This field is used for visualization, not reduction decisions or editable geometry.

## Command line and geometry equivalence

The installed CLI completed loading, reduction, and STL export in **18.16 s**, compared with
**17.93 s** for the candidate CLI in one matched run. The approximately 1% difference is too small
to present as an established reduction speedup.

The exported binary STL triangle payloads were byte-for-byte identical, including coordinates,
normals, triangle order, and attribute bytes. STL headers were excluded from the comparison.
Displayed result counts and quality measurements also agreed in the GUI comparison.

## Where time remains

A separate two-run candidate profile, with Density enabled and the same target, averaged:

| Stage | Seconds | Share of measured total |
| --- | ---: | ---: |
| Application startup | 3.29 | 14.1% |
| Read, measure, and prepare source display | 7.32 | 31.5% |
| Reduce, check, and display result | 12.38 | 53.2% |
| Apply and export | 0.27 | 1.2% |
| Total | 23.25 | 100.0% |

This profile used a separate instrumented GUI harness and an export endpoint. It is not the same
run set as the installed-versus-candidate verification comparison above.

Additional isolated measurements located the largest costs:

| Operation | Seconds | Notes |
| --- | ---: | --- |
| CPU QEM | 6.77 | Connectivity work and ordered collapse processing. |
| STL parsing, welding, and array conversion | 3.52 | Mean loading-function time from the GUI profile. |
| Topology and surface finishing | 2.92 | Includes 2.65 s matching surface area. |
| GPU depth/density surface analysis | 2.19 | Algorithm-reported elapsed analysis time, not kernel-only time. |
| Source display normals | 1.77 | Mean viewport-preparation time from the GUI profile. |
| Source density visualization | 1.07 | GUI-worker measurement, including first-use overhead. |
| Final quality reporting | 0.21 | Separate isolated measurement. |

These rows combine separate instrumented measurements and are not additive. In particular,
surface-area matching is included within topology/surface finishing. GPU initialization completed
during loading in the profiled GUI runs rather than adding its full duration afterward.

## Interpretation and remaining work

The measured benefit is faster preparation, analysis, and display, with more informative progress.
The ordered QEM reducer, topology repair, and surface-finishing decisions remain unchanged.

Repeated measurements of immutable source bounds during surface finishing and display-normal
preparation are candidates for further low-risk investigation. They are not additional completed
optimizations in this comparison. Changing STL welding or QEM ordering would require separate
geometry-equivalence testing.
