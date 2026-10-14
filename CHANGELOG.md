# Changelog

Published releases are listed newest first. Preview builds are marked separately.

## 0.3.0 - Unreleased

MeshMill 0.3.0 improves performance with dense geometry, adds a higher-quality GPU-guided
reduction option, and refines the desktop workflow.

### Performance and large meshes

- Added tiled processing for larger meshes, with parallel work, cancellation, resume, and safe
  result assembly.
- Added processing estimates based on mesh size, available CPU, memory, GPU, and the selected
  algorithm. MeshMill shows the processing mode, worker count, estimated time, and expected tradeoff
  before a long operation begins.
- GPU-guided reduction now prepares its GPU work once, then reuses it for later optimizations.
- Long operations now use one clearer status window for preparation, analysis, reduction, validation,
  and application.

### Depth + Density Adaptive QEM

- Added **Depth + Density Adaptive QEM (GPU / Best)**. It uses GPU surface measurement and local
  density information to preserve useful detail while reducing redundant scan geometry.
- Automatic algorithm selection uses Depth + Density Adaptive QEM when its compatible GPU backend
  is available, otherwise Density balanced remains the quality-oriented fallback.
- Added GPU selection for compatible NVIDIA systems. Automatic selection prefers the compatible GPU
  with the most available video memory, and Preferences can pin MeshMill to a specific GPU.
- Added measured quality metrics, plain-language tooltips, and a consistent quality comparison row.

### Interface and workflow

- Added application themes, viewport appearance controls, and a stepped interface font-size control.
- Updated orientation, density, scale, selection, and ruler overlays to match the selected theme.
- Added in-app full-screen control and `Ctrl+Enter` shortcut, pointer-centered keyboard orbiting,
  depth-scaled panning, more reliable zooming, and persisted named orientations.
- Added configurable placement and visibility for viewport overlays, a compact scale indicator, and
  improved metrics history controls.
- Added an editable PowerShell window that can copy or save a command-line script for completed
  full-mesh optimization passes.
- Improved About, support, issue-report, update, and changelog windows.

### Documentation and testing

- Updated algorithm, GPU analysis, out-of-core, platform-testing, CLI, and tiled-processing
  documentation.
- Refreshed shaded, vertices, wireframe, and density screenshots from the v0.3 application.
- Added regression coverage for the depth-guided and tiled-processing paths.

### Related research

| Publication | Focus | Version |
| --- | --- | --- |
| [Depth + Density Adaptive QEM](https://github.com/make-your-own-world/rnd/blob/main/depth-density-adaptive-qem/README.md) | Combined depth-gauge and density-guided reduction for the strongest measured preservation results. | 1.1 |
| [Boundary-Preserved Deterministic QEM](https://github.com/make-your-own-world/rnd/blob/main/boundary-preserved-deterministic-qem/README.md) | Bounded-memory, deterministic tiled reduction and streamed assembly for large binary STL meshes. | 1.0 |

### Localization status

MeshMill 0.3.0 is currently released in English. Updated translations will follow after review.

## 0.2.0 - 2026-10-09

MeshMill 0.2.0 adds large-mesh and interface performance improvements, a higher-quality GPU-guided reduction option, measured quality comparisons, improved navigation, and a production Windows installer.

### Performance improvements

- **Parallel large-mesh analysis:** analysis of the 4,126,315-triangle reference mesh improved from 10.531 seconds with one worker to 1.838 seconds with eight workers, a 5.73x speedup. Both runs produced identical bounds, 50 adaptive tiles, and 4,126,315 assigned triangles.
- **Faster adaptive-index creation:** complete index creation improved from 8.836 seconds to 5.406 seconds with a warm filesystem cache, a 1.63x speedup. Partition publication remains single-owner so output stays deterministic and tile files do not receive competing writes.
- **Memory-aware STL analysis:** large binary STL files are inspected and partitioned in bounded blocks instead of requiring the complete source to be allocated at once.
- **Bounded overview loading:** meshes above the configured memory budget open as a manageable read-only overview rather than exhausting system memory or freezing the interface.
- **Automatic processing budgets:** MeshMill derives worker counts, overview limits, and geometry-job capacity from available CPU, RAM, and VRAM instead of relying on a fixed triangle ceiling.
- **Adaptive spatial indexing:** uneven meshes are divided according to their measured triangle distribution. The reference mesh produced 50 independently addressable tiles with a 65,075-triangle median and a 249,043-triangle maximum.
- **Reusable caches:** persistent source fingerprints, spatial indexes, GPU kernels, density data, and comparison results avoid repeating validated work when the underlying mesh and settings have not changed.
- **Responsive geometry operations:** reduction, analysis, loading, and other heavy geometry work run outside the interface, with progress reporting and cancellation where the operation can be safely interrupted.
- **Measured CPU and GPU dispatch:** small jobs stay on the CPU when GPU transfer and synchronization would cost more than the calculation. GPU analysis is used where resident data and parallel surface measurements provide a measured benefit.
- **Expanded performance telemetry:** the metrics panel separates CPU, RAM, GPU, and VRAM and reports geometry activity, processing capacity, cumulative I/O totals, active I/O time, and peak transfer samples.
- **Faster close-range navigation:** pointer-centered orbiting and depth-scaled panning keep viewport movement usable when working close to dense geometry.

These improvements were implemented before and independently of Depth Adaptive QEM. The out-of-core work in this release covers bounded inspection, overview loading, indexing, scheduling, and caching. Full streaming reduction and cross-tile assembly remain roadmap work.

### Depth Adaptive QEM

Depth Adaptive QEM is an optional GPU / Higher Quality algorithm. It uses adaptive, physically scaled surface probes on the GPU, then hands the measured surface guidance to boundary-preserving Fast QEM assembly.

Fast QEM remains the default because it starts immediately and works on every supported system. Depth Adaptive QEM is available when MeshMill detects compatible NVIDIA hardware.

On the 4,126,315-triangle reference mesh at a matched 250,000-triangle target:

| Algorithm | Time | RMS error | P95 error | Maximum drift | Area error |
|---|---:|---:|---:|---:|---:|
| Fast QEM | 9.827 s | 1.1721 mm | 2.3504 mm | 0.6987 mm | 0.3993% |
| Depth Adaptive QEM | 10.156 s | 1.0668 mm | 2.0586 mm | 0.0234 mm | 0.1241% |

This measured run improved RMS surface error by 9.0%, P95 error by 12.4%, maximum dimension drift by 96.7%, and area error by 68.9%, with a 3.3% runtime increase.

The Windows packages include the minimal CUDA runtime needed by this algorithm. It does not install a system-wide CUDA toolkit or a background service. The first run prepares and caches GPU kernels for the local machine; later runs reuse that cache.

### Quality comparisons

- Added a compact measured-quality row for RMS, P95, area, topology, drift, and reduction.
- Added plain-language tooltips describing each metric and its tradeoffs.
- Added algorithm comparison documentation, benchmark tables, formulas, and visual comparisons.
- Published the supporting paper: [Depth Adaptive QEM: GPU-Guided Surface Measurement for Practical Mesh Reduction](https://github.com/make-your-own-world/rnd/blob/main/depth-adaptive-qem/depth-adaptive-qem-v1.1.pdf).

### Viewport and workflow

- Left-drag now pans the viewport.
- Middle-drag orbits around visible geometry under the pointer.
- Panning scales with the picked geometry depth, including close zoom levels.
- Improved maximized-window layout, overlays, ruler visibility, and zoom shortcuts.
- Added per-application CUDA GPU selection by PCI address for multi-GPU systems.
- Added manual update checks and optional checks when the app opens. Automatic checking is off by default, and no update service or scheduled task is installed.
- Verified installer upgrades preserve settings and reopen the active STL when requested.

### Research and roadmap

- Added tested CPU and GPU strategies, relevant public research, formulas, benchmark methods, failures, and design decisions to the release documentation.
- Added printable solid synthesis derived from Depth Adaptive QEM, hybrid depth-and-density reduction, full out-of-core reduction and assembly, multi-mesh workspaces, control profiles, and optional distributed geometry processing to the roadmap.

### Availability

Windows is the supported release platform. Linux and macOS downloads are published separately as clearly labeled testing previews because they have not received the same real-hardware validation.

MeshMill 0.2.0 is distributed in English. Updated translations were deferred because the available translation, validation, and review work exceeded current capacity. Existing community translation files remain in the repository for future completion, but are not included in the packages because they have not passed the 0.2.0 release validation gates.

MeshMill is released under GPL-3.0. Third-party notices and the CUDA redistribution exception are included with the source and packages.

## 0.2.0-preview.2 - 2026-10-05 (Preview)

### Linux and macOS testing previews

These downloads are for early testing on Linux and macOS. Windows users should use the supported Windows release shown as Latest on the Releases page.

- `linux-x86_64.tar.gz`: 64-bit Linux systems
- `macos-arm64.zip`: Apple silicon Macs
- `macos-x86_64.zip`: Intel Macs
- Matching `SHA256SUMS` files: download verification

These packages are self-contained and unsigned. They passed automated native builds and packaged smoke tests, but have not received the same real-hardware coverage as the Windows release. See `docs/PLATFORM_TESTING.md` and report problems through the platform test issue form.

## 0.1.2 - 2026-10-05

### Windows download

Most users should download `MeshMill-0.1.2-windows-x64-setup.exe`.

- **Installer:** normal Windows installation with Start menu and optional desktop shortcuts. Upgrades preserve settings and shortcut choices; uninstall removes MeshMill preferences.
- **Portable ZIP:** no installation. Extract the complete ZIP, then run `MeshMill.exe`.
- **Sample STL:** optional large test geometry for experimenting with MeshMill.
- **SHA256SUMS.txt:** checksums for verifying downloads.

These packages are self-contained. Python, Node.js, and development dependencies are not required. Linux and macOS downloads are published separately as clearly labeled preview releases.
