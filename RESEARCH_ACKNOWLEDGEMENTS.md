# Research acknowledgements

MeshMill builds on published research and open-source software. This list credits work that
informed the mathematics, implementation, experiments, and comparisons, including approaches
that were explored but did not become part of the application. Those experiments helped clarify
the problem and guide the methods that followed.

Each entry describes its contribution. Results from a particular MeshMill experiment describe
that experiment's input and implementation, rather than the value of the referenced work as a whole.
Software distributed with MeshMill retains the licenses and notices listed in
[third-party notices](THIRD_PARTY_NOTICES.md) and [corresponding source](SOURCES.md).

| Work | Contribution to MeshMill's development |
| --- | --- |
| [CuPy](https://github.com/cupy/cupy) | GPU arrays, custom kernels, reductions, sorting, and synchronization used in surface analysis and GPU simplification experiments. |
| [Decimation of Triangle Meshes](https://www.cs.columbia.edu/~allen/PHOTOPAPERS/Schroeder-etal-sg92.pdf) | Topology-oriented vertex-removal background and the comparison method exposed through VTK's `vtkDecimatePro`. |
| [External Memory Management and Simplification of Huge Meshes](https://www.cs.upc.edu/~pere/PapersWeb/SGI/Cignoni.pdf) | External-memory mesh organization, octree-based processing, and protected block boundaries considered during tiled-reduction design. |
| [Fast Quadric Mesh Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification) | The compiled QEM baseline and collapse-scheduling approach used to study reduction speed, quality, and topology. |
| [fast-simplification](https://github.com/pyvista/fast-simplification) | The Python bindings used for Fast QEM and the final ordered reduction stage of the depth-guided methods. |
| [High-performance simplification of triangular surfaces using a GPU](https://pmc.ncbi.nlm.nih.gov/articles/PMC8341488/) | Conflict sets and independent parallel collapses informed GPU mutation prototypes and their topology checks. The experiments helped establish the conflict, connectivity, and target-count tradeoffs on layered scans. |
| [MeshFix and PyMeshFix](https://github.com/pyvista/pymeshfix) | Repair and intersection-validation infrastructure. The upstream research citation is *A Lightweight Approach to Repairing Digitized Polygon Meshes* (2010). Local experiments with repair behavior also helped define how much measured surface could be changed during completion. |
| [Modern Robotics](https://modernrobotics.northwestern.edu/) | Public rigid-motion and coordinate-frame background for axial calibration. |
| [NIST roundness measurements](https://www.itl.nist.gov/div898/handbook/mpc/section3/mpc344.htm) | Measurement background for interpreting runout and axis correction from rotating surface contacts. |
| [NumPy](https://github.com/numpy/numpy) | Array operations, statistics, indexing, and linear algebra throughout the implementation and experiments. Its [least-squares API](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) is used in calibration. |
| [Out-of-Core Simplification of Large Polygonal Models](https://www.cs.princeton.edu/courses/archive/spr01/cs598b/papers/lindstrom00.pdf) | Bounded-memory simplification and quadric-clustering background considered during large-mesh and streaming design. |
| [PyMaxflow and MAXFLOW](https://github.com/pmneila/PyMaxflow) | Graph-cut infrastructure used to choose inside/outside volume labels. The upstream citation is [An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision](https://discovery.ucl.ac.uk/id/eprint/13383/). |
| [Python-SAT and RC2](https://github.com/pysathq/pysat) | SAT/MaxSAT infrastructure used to resolve local topology constraints after the initial volume labeling. |
| [Real-time Mesh Simplification Using the GPU](https://gfx.cs.princeton.edu/gfx/pubs/DeCoro_2007_RMS/real_time_simplification.pdf) | Streaming quadric clustering informed GPU clustering experiments. Comparing their speed and treatment of nearby surface sheets helped guide the decision to retain indexed geometry through surface analysis. |
| [Robust Inside-Outside Segmentation using Generalized Winding Numbers](https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.html) | Related constrained-volume architecture for reasoning about incomplete surfaces and inside/outside segmentation. MeshMill uses a different surface-distance query in its completion implementation. |
| [RXMesh](https://github.com/owensgroup/RXMesh) | Public GPU-resident dynamic-mesh reference. Its QSlim example was built and timed during comparison work, informing the investigation of conversion costs, topology mutation, and reduction scheduling. |
| [SciPy](https://github.com/scipy/scipy) | Numerical, spatial-search, sparse-connectivity, and triangulation facilities used in selected experiments and validation. Its [least-squares documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) provided a soft-L1 reference for calibration; calibration uses its own fitting path rather than that solver. |
| [Signed Distance Computation Using the Angle Weighted Pseudonormal](https://backend.orbit.dtu.dk/ws/portalfiles/portal/3977815/B_rentzen.pdf) | Surface-distance sign treatment used when classifying volume samples near triangle faces, edges, and vertices. |
| [Simplifying Surfaces with Color and Texture using Quadric Error Metrics](https://www.mgarland.org/research/quadrics.html) | Attribute-aware quadric background for density weighting and experiments with scalar-guided reduction. Those comparisons helped distinguish attribute preservation from an explicit regional triangle budget. |
| [Surface Simplification Using Quadric Error Metrics](https://publications.ri.cmu.edu/surface-simplification-using-quadric-error-metrics) | The accumulated-plane quadric objective and edge-collapse foundation underlying the QEM reduction methods. |
| [TetGen](https://wias-berlin.de/software/tetgen/) and its [Python wrapper](https://github.com/pyvista/tetgen) | Constrained tetrahedralization used to divide space around measured surface facets during solid completion. |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Public structured 3D representation and GPU-oriented geometry reference reviewed during the exploration of alternative geometry pipelines. |
| [trellis.cpp](https://github.com/pwilkin/trellis.cpp) | Public implementation reference reviewed alongside TRELLIS while comparing representation and processing choices. |
| [VTK](https://github.com/Kitware/VTK) | Mesh I/O, rendering, selection, measurements, hull and morphology operations, and comparison reducers. Relevant implementations include [Quadric Decimation](https://vtk.org/doc/nightly/html/classvtkQuadricDecimation.html), [Decimate Pro](https://vtk.org/doc/nightly/html/classvtkDecimatePro.html), [Flying Edges](https://vtk.org/doc/nightly/html/classvtkFlyingEdges3D.html), and the [Static Cell Locator](https://vtk.org/doc/nightly/html/classvtkStaticCellLocator.html). |
| [Warp](https://github.com/NVIDIA/warp) | Open-source GPU surface-query infrastructure used for closest-surface and signed-distance operations in completion and cast preparation. |

The [MeshMill reports](https://github.com/make-your-own-world/rnd) connect these references to the
formulas, experiments, and design decisions. The [algorithm comparison](benchmarks/ALGORITHM_COMPARISON.md)
and [GPU analysis discussion](docs/GPU_QEM_AND_DEPTH_ANALYSIS.md) retain the results of alternative
approaches alongside the methods used in the application. The complete application and build
dependencies are credited in [third-party notices](THIRD_PARTY_NOTICES.md).
