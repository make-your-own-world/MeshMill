# Corresponding source and third-party notices

MeshMill source and build instructions are included in its release source tree.
The following unmodified source archives identify the new surface-completion dependencies.
The companion source collection contains these archives alongside the MeshMill source tree. Each upstream archive is verified by SHA-256.

## pymeshfix 0.18.1

[Upstream source archive](https://files.pythonhosted.org/packages/fc/d8/c5124ee26c10bd2d77333eb5a8cddc89cb45ced9f1b173235e13ecfa6ebd/pymeshfix-0.18.1.tar.gz)

SHA-256: `c3386a123e3ecfcff59f229a4fc620ae0e03fa9c7fec025fbd2885349fa8fa3f`

## tetgen 0.8.4

[Upstream source archive](https://files.pythonhosted.org/packages/af/b6/bd320925cc4127c6f838c9127fdb0b2292147286500489d5ba6aac5d34e1/tetgen-0.8.4.tar.gz)

SHA-256: `b6ca979ccecf43ddc0d31fa4f5a05da3cfeb4ab0705b8128530745bbab38bb86`

## PyMaxflow 1.3.2

[Upstream source archive](https://files.pythonhosted.org/packages/f4/d8/a719512ee03084541fc8e6f1fa06be366fb97f9c39f0b3b3bba7364769e8/pymaxflow-1.3.2.tar.gz)

SHA-256: `19f1668dd718d1451a1ad18902ac674b1b380cda1150c855b5e817f1b0de3095`

## python-sat 1.9.dev17

[Upstream source archive](https://files.pythonhosted.org/packages/d8/90/dfe512facf5def6efd483c05506495860d5fffb33573f09b419fd1f8d5b8/python_sat-1.9.dev17.tar.gz)

SHA-256: `826e9ef8a8086f5139c05eb89aaa05ebfa72f8a622cd7be429d10a2b1cbb1955`

## Licence texts

The `licenses/` directory contains retained upstream texts and notices. Python package
metadata also retains dependency notices in the binary bundle. CUDA components retain
their NVIDIA terms; they are not relicensed as MeshMill code.

## Research acknowledgements

The MAXFLOW implementation requests citation of Boykov and Kolmogorov,
An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision
(IEEE TPAMI, 2004). MeshFix research use requests citation of Attene, A Lightweight Approach
to Repairing Digitized Polygon Meshes (The Visual Computer, 2010).
TetGen is credited for constrained tetrahedralization; Python-SAT for SAT/MaxSAT infrastructure.

## Qt and Python for Qt source

The distributed Qt/PySide6/Shiboken6 version is 6.11.2. Matching upstream source is available at:

- [Qt 6.11.2 source archive](https://download.qt.io/archive/qt/6.11/6.11.2/single/)
- [PySide6 and Shiboken6 6.11.2 source archive](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.2-src/)

These are the source versions corresponding to the dynamically loaded Qt libraries. Their upstream
license texts are retained in `licenses/`. Python-SAT's source archive also includes the native
solver sources and patches; their separate notices are retained under `licenses/python-sat-solvers/`.

## Research references

[Research acknowledgements](RESEARCH_ACKNOWLEDGEMENTS.md) records the public work consulted during
implementation and experimentation, including alternatives studied or tested along the way.
