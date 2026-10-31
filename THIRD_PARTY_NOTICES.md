# Third-party software

MeshMill depends on the following software. These components retain their own licenses and
copyright notices.

| Component | Version | License |
| --- | ---: | --- |
| [Python](https://www.python.org/) | 3.12 | Python Software Foundation License |
| [NumPy](https://numpy.org/) | 2.5.3 | BSD-3-Clause and bundled permissive licenses |
| [fast-simplification](https://github.com/pyvista/fast-simplification) | 0.2.0 | MIT |
| [VTK](https://vtk.org/) | 9.7.1 | BSD-3-Clause |
| [PySide6](https://doc.qt.io/qtforpython-6/) | 6.11.2 | LGPL-3.0-only, GPL-2.0-only, or GPL-3.0-only |
| [Shiboken6](https://doc.qt.io/qtforpython-6/shiboken6/) | 6.11.2 | LGPL-3.0-only, GPL-2.0-only, or GPL-3.0-only |
| [PyInstaller](https://pyinstaller.org/) | 6.22.3 | GPL-2.0-or-later with a bootloader exception |
| [CuPy](https://cupy.dev/) | 13.6.0 | MIT and bundled permissive licenses |
| [PyMeshFix](https://github.com/pyvista/pymeshfix) | 0.18.1 | GPL-3.0-or-later (shipped licence text); retain upstream notices |
| [TetGen Python wrapper](https://github.com/pyvista/tetgen) | 0.8.4 | MIT wrapper; TetGen core AGPL-3.0-or-later |
| [PyMaxflow](https://github.com/pmneila/PyMaxflow) | 1.3.2 | GPL-3.0-or-later (MAXFLOW); retain upstream notices |
| [Python-SAT](https://github.com/pysathq/pysat) | 1.9.dev17 | MIT interface and bundled solver licenses |
| [Warp](https://github.com/NVIDIA/warp) | 1.9.1 | Apache-2.0 and retained bundled third-party notices |
| [six](https://github.com/benjaminp/six) | Dependency of Python-SAT | MIT |
| [fastrlock](https://github.com/scoder/fastrlock) | 0.8.3 | MIT |
| [NVIDIA CUDA Runtime](https://developer.nvidia.com/cuda-toolkit) | 12.4.127 | NVIDIA CUDA Toolkit EULA |
| [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/) | 12.0.140 | NVIDIA CUDA Toolkit EULA |

The Windows bundle distributes PySide6 and Shiboken6 under their GPL-3.0 option, consistent
with MeshMill's GPL-3.0-or-later license. NumPy wheels contain additional permissively licensed
components; their notices are included in the installed package metadata and should be retained
when redistributing a modified binary bundle.

CuPy and `fastrlock` are used by the optional Depth Adaptive QEM computation path. Their complete
license texts are retained in the packaged Python metadata under `_internal/cuda_runtime`.

The Windows installer and portable archive include unmodified CUDA Runtime and NVRTC binary
components identified as redistributable in Attachment A of the
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/). They are stored in MeshMill's private
`_internal/cuda_runtime` directory and are accessed only by MeshMill. NVIDIA components remain
proprietary NVIDIA software, are not licensed under the GNU GPL, and are intended for use with
compatible NVIDIA hardware. Complete NVIDIA license texts are retained at:

- `_internal/cuda_runtime/nvidia_cuda_runtime_cu12-12.4.127.dist-info/License.txt`
- `_internal/cuda_runtime/nvidia_cuda_nvrtc_cu12-12.0.140.dist-info/License.txt`

MeshMill's permission to interoperate with these separately licensed components is described in
[`CUDA_EXCEPTION.md`](CUDA_EXCEPTION.md). No NVIDIA driver, system-wide CUDA Toolkit, background
service, or scheduled task is installed.

## Surface-completion components

Solidify uses PyMeshFix for surface checks, TetGen for constrained tetrahedralization,
PyMaxflow for graph cuts, and Python-SAT for topology constraints. These libraries are
incorporated dependencies, rather than research references alone. Their license texts
and notices accompany the distributed dependencies. The Python wrapper's license does
not replace the license of its bundled native library.

TetGen's core uses AGPLv3-or-later. GPLv3 section 13 permits combination with AGPLv3,
while retaining the AGPL network-interaction requirements for the combination. See
[GPLv3 section 13](https://www.gnu.org/licenses/gpl-3.0.html#section13).
MeshMill's CUDA exception covers MeshMill-owned code; it does not grant exceptions
for third-party components.

See [corresponding source and acknowledgements](SOURCES.md) and the included `licenses/` directory.
