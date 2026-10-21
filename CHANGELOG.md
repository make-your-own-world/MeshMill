# Changelog

Published releases are listed newest first. Preview builds are marked separately.

## 0.3.3 - 2026-10-21

### Loading and optimization preparation

- Make loaded meshes interactive without waiting for GPU initialization to finish. GPU setup
  continues in the background.
- Prepare reusable probe measurements during idle time for GPU-guided reduction. Reuse them
  across target changes and refresh them when geometry changes, retaining the existing math.
- Use the normal calculation path when background preparation is unfinished or unavailable.

### Viewport and navigation

- Establish Front from the opening camera, with the other five views derived from that fixed reference.
- Fit all six standard views closely inside the viewport with a small margin and no cropping.
- Restore the opening rotation, zoom, pan, and Front reference when reloading the original mesh.
- Preserve framing across Shaded, Density, Wireframe, and Vertices, including initial Density loading.
- Keep the mesh visible during density-color calculation.
- Improve selected view-button text and icon contrast across themes, and leave views unselected
  when no mesh is loaded.

### Metrics history

- Fill the graph at full resolution before progressively compressing older history toward the left.
- Keep recent activity detailed and retain averages, minimums, and maximums in display summaries.
- Retain raw readings within the selected history period for higher-detail viewing and match
  timeline labels to the displayed history span.

### File opening and Windows integration

- Remember the last opened folder by default. Add Smart open (the parent of the last folder)
  and a specific starting-folder option, with fallback to an existing parent for unavailable paths.
- Add a Preferences action to choose the default STL application through Windows Settings.
  Installation and upgrades advertise Open with MeshMill without replacing an existing default.

### Update preferences

- Offer Install now, Remind me next time, and Skip this version after optional startup checks.
- Add an installation action in Settings for available updates. Skipped versions remain manually installable.
- Add optional automatic installation, available only with automatic checking enabled. Both are off by default.
- Default automatic installation to hotfixes and maintenance within the current release series.
  An optional All releases scope also permits minor and major upgrades; otherwise those ask first.
- Wait for an idle app with no unsaved mesh changes before automatic installation, including
  changes made during download. Reopen the current STL after the Windows upgrade.
- Run update checking inside MeshMill, without installing a separate updater service or scheduled task.

### Preferences and interface

- Use Theme, Viewport, and Match theme consistently.
- Show custom density-gradient controls only for Custom viewport appearance, including during search.
- Place compact search beside Save, Reset, and Cancel at the bottom of Preferences.
- Retain reset confirmation, with Save applying the changes.

### Reliability

- Recover verified orphaned MeshMill workers left behind by older versions on startup.
- Check application ownership, parent-process lifetime, and process identity before recovery.
  Leave unrelated Python processes and workers with live owners running.
- Retain the worker shutdown and parent-death protection introduced in 0.3.2.

## 0.3.2 - 2026-10-21

### Worker cleanup

- Stop and join optimization workers when cancelling, closing the window, or quitting the app.
- Stop workers if their owning MeshMill process crashes or is force-closed. Windows process
  containment also covers nested workers and workers busy in native geometry or GPU code.
- Cancel tiled analysis and reduction without waiting for a busy worker to finish its current task.
- Clean up processes belonging to the selected installation before installing, upgrading, or
  uninstalling. Same-named processes from other installations and unrelated Python processes
  are left alone.

This hotfix does not change reduction algorithms, mesh quality, or navigation controls.

## 0.3.1

### Performance and progress

- Reduced loading and result-display overhead while retaining the existing reduction algorithms.
- Added optional CUDA acceleration for density visualization on larger meshes, using the selected
  GPU with a CPU fallback when acceleration is unavailable or resources are insufficient.
- Moved display-normal preparation into background work before handing geometry to the viewport.
- Reused measurements of unchanged source geometry and accelerated read-only topology checks.
- Count topology edges using packed integer keys, with optional GPU counting and an equivalent
  CPU fallback. These checks report result quality without changing the mesh.
- Process GPU density visualization in bounded batches, with resource checks, cancellation, and
  CPU fallback. Reuse completed density fields when switching back to a cached display.
- Added native QEM stage/count reporting and STL-reader progress without changing collapse decisions.
- Made progress-file updates atomic so readers do not encounter partially written status data.
- Identify processing as local and keep controls busy until the prepared viewport is ready.
- Keep connected-component selection off the UI thread, so its connectivity analysis does not
  run inside the input handler.

In a matched full-scan comparison, launch through a completed preview improved from **25.31 to
23.04 seconds in Wireframe (9%)**, and from **26.20 to 22.74 seconds in Density (13%)**. Warmed
density-color calculation improved from **1.50 to 0.45 seconds**, approximately **3.3x faster**.
Command-line execution was essentially unchanged, and exported triangle payloads matched exactly.
These are development measurements against installed 0.3.0, not a general QEM speedup. See the
[benchmark conditions and results](benchmarks/V0_3_1_PERFORMANCE.md).

### Interface and usability

- Kept crop selections editable through modifier-key presses and ruler use. Camera navigation
  clears the view-specific outline while preserving selected geometry.
  Shrinking an outline deselects geometry outside it while retaining separately added regions.
  Empty outlines remain editable, and clicking inside no longer implicitly adds the selection.
- Enabled ruler picking through selection highlights without committing or dismissing the crop.
- Added **Ctrl+double-click** to toggle a connected mesh component without changing other
  selected components. Connected-component analysis runs in a background worker.
- Simplified the selection panel to a compact row of Crop, Delete, Step back, and Clear actions.
  Optimization remains in the toolbox. Added themed action icons and directional view icons.
- Position the connected-selection panel near the click unless its position was moved manually.
- Fixed repeated arrow-key rotation progressively moving the pivot and zooming into the mesh.
  Continued rotation now preserves its pivot and camera distance.
- Smoothed keyboard rotation, pan, zoom, and mouse-wheel navigation with short, bounded steps
  and no accumulating motion backlog. Mouse dragging remains direct.
- Clear ruler measurements when the camera rotates, while preserving selected geometry.
- Increased toolbox section margins and protected button-row minimum sizes to prevent clipped
  borders and labels across the supported font sizes.
- Combined fullscreen, minimize, maximize/restore, and close controls into one themed titlebar row.
- Changed the default fullscreen shortcut to **F11**, for both entering and leaving fullscreen.
  The previous Ctrl+Enter default is migrated; other custom mappings are retained.
- Added clearable Preferences search that filters individual setting rows by their names and
  section headings. Matching labels and controls remain together; tooltips, dropdown values,
  and shortcut values do not produce matches.
- Matched metrics-graph backgrounds, grid lines, labels, and CPU/GPU traces to each application
  theme without clearing recorded history.
- Replaced the yellow General-section background in Dark Focus with a lighter dark blue.
- Added visible vertical scrollbars to both changelog views and horizontal scrolling for wide
  content. Improved link, text, code, table, and selection colors across themes.
- Prevented delayed viewport layout refreshes from rendering after the window has closed.

### Windows integration

- Added per-user STL **Open with MeshMill** registration without changing the default STL handler.
- Added installer registration and uninstall cleanup for MeshMill-owned association entries.

### Validation

- Expanded checks for unchanged geometry, CPU fallback, cancellation, selection optimization,
  apply/undo/redo/reload, window-state transitions, shortcut migration, Preferences filtering,
  theme contrast, changelog scrolling, and association install/update/removal behavior.

## 0.3.0

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
