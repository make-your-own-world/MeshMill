# MeshMill roadmap

This roadmap tracks work after 0.3.1. Checked items are completed.
Released features are documented in the [changelog](CHANGELOG.md).

## Maintenance (0.3.1)

- [x] Allow ruler measurements while a crop selection is active, preserving its edit handles.
- [x] Keep the selection box or polygon visible and editable when Ctrl, Shift, or other modifier
  keys are pressed or released.
- [x] Recalculate selection when its box or polygon changes shape. Shrinking it deselects geometry
  outside the new outline; separately added selections remain independent.
- [x] Preserve selected geometry during camera movement while clearing the view-specific outline.
  Keep the outline editable until navigation, explicit clearing, cancellation, or application.
- [x] Test transitions between ruler input, selection editing, modifier keys, and navigation
  without unintended dismissal.
- [x] Toggle connected components with Ctrl+double-click, preserving other selected components.
- [x] Simplify selection actions, add themed icons, and position the panel near the selection click.
- [x] Preserve the orbit pivot and camera distance during repeated arrow-key rotation.
- [x] Smooth keyboard and wheel navigation without an accumulating motion backlog.
- [x] Clear ruler measurements on rotation without clearing selected geometry.
- [x] Improve loading and result-display performance with background display-normal preparation,
  cached immutable measurements, accelerated reporting checks, and GPU density visualization.
- [x] Report STL-reader and native QEM progress without changing reduction decisions.
- [x] Add unified window controls, F11 fullscreen, and setting-name Preferences filtering.
- [x] Correct theme contrast, changelog scrolling, toolbox margins, and button sizing.
- [x] Add per-user STL Open with registration while preserving the default handler.
- [ ] Complete and validate current application and documentation translations, with native-speaker
  review and corrections.

## Distributed tile processing

Extend the released deterministic local tile-processing architecture to multiple explicitly trusted
MeshMill instances. One desktop hosts a job; other desktops opt in as workers. The first milestone
is LAN processing, without requiring shared-project or collaborative editing features.

- [ ] Define a versioned protocol around existing tile manifests, immutable job parameters, target
  allocations, checksums, and validation reports.
- [ ] Add in-app host and worker modes with explicit enrollment and per-job authorization.
- [ ] Report worker version, supported algorithms, CPU, RAM, GPU, available capacity, and load.
- [ ] Transfer checksummed input and result artifacts with bounded buffering and resume support.
- [ ] Schedule complete tiles according to capability and capacity, including transfer and setup
  costs in estimates.
- [ ] Add assignment leases, cancellation, reassignment, and recovery after worker or network loss.
- [ ] Keep final validation, deterministic assembly, and output publication under host ownership.
- [ ] Show participating workers, assigned regions, transfers, tile progress, ETA, and assembly
  status in the application.
- [ ] Validate local-only, single-remote-worker, and multiple-worker output consistency.
- [ ] Test incompatible versions, corrupted transfers, duplicate results, interruptions, and restart.
- [ ] Benchmark end-to-end time, transfer overhead, peak memory, and quality across independent
  meshes before recommending distributed execution.

Local processing remains the default. Automatic discovery and Internet/cloud workers are later
extensions, after explicit LAN enrollment and recovery are reliable.

## Responsive workflow visualization

Very large local and distributed jobs need a live spatial view of their work. The earlier cube
display was removed because its implementation made interaction laggy; that does not remove the
visualization requirement.

- [ ] Render queued, transferring, reading, processing, validating, completed, and failed tiles.
- [ ] Show which node owns each work region, with per-tile, per-node, and overall progress.
- [ ] Preserve a readable whole-object view and avoid obscuring or distorting the mesh.
- [ ] Run geometry work, network I/O, and progress aggregation outside the UI thread.
- [ ] Feed rendering bounded progress snapshots without blocking on workers.
- [ ] Coalesce events and cap redraw frequency; aggregate or simplify visualization when tile
  counts exceed what can be shown usefully.
- [ ] Test input latency, frame time, progress freshness, and visualization overhead under CPU,
  GPU, RAM, disk, and network load with many concurrent tiles and nodes.

## Large-mesh processing refinements

Local bounded-memory indexing, parallel tiled reduction, checkpoints, deterministic target
allocation, protected boundaries, and streamed assembly shipped in 0.3.0. Remaining work:

- [ ] Stream coarse-to-fine viewport detail rather than requiring the full-resolution mesh in RAM.
- [ ] Measure peak process-tree memory and GPU-guided tiled routes on multiple independent fixtures.
- [ ] Investigate a shared-boundary optimizer to improve quality or retained triangle count while
  preserving seam integrity and deterministic output.
- [ ] Investigate reuse of resident viewport buffers through graphics-compute interoperability on
  compatible selected adapters, with validated fallbacks.
- [ ] Pursue additional GPU reduction strategies when measurements show an end-to-end runtime or
  memory benefit while preserving the geometry and topology contract.

See [out-of-core architecture](docs/OUT_OF_CORE.md) and
[tiled-processing measurements](docs/TILED_PROCESSING.md) for the released foundation.

## Control profiles

- [ ] Add named navigation profiles for MeshMill defaults, CAD-style controls, trackpads, and
  left-handed input.
- [ ] Map mouse buttons, wheel actions, modifiers, keyboard commands, and axis controls independently.
- [ ] Configure orbit pivots, movement sensitivity, zoom, roll direction, and axis inversion.
- [ ] Detect conflicting or unreachable mappings.
- [ ] Support temporary preview, reset, duplication, import/export, and global or workspace defaults.
- [ ] Preserve profile compatibility across platforms and identify unavailable device bindings.

## Multiple meshes and surface synthesis

- [ ] Add multiple independently visible STL objects with units, transforms, bounds, provenance,
  and per-source operation history.
- [ ] Add manual alignment and automatic registration with measurable fit quality.
- [ ] Reuse adaptive spatial tiles to compare overlapping sources within bounded memory.
- [ ] Measure surface distance, normal agreement, density, variance, and observation count.
- [ ] Classify matching surfaces, conflicts, noise, gaps, and unique geometry.
- [ ] Consolidate agreeing surfaces instead of retaining duplicate layers.
- [ ] Remove enclosed, coincident, and shared geometry that contributes no exterior detail.
- [ ] Preserve unique geometry and expose ambiguous regions for review.
- [ ] Support per-source and per-region weighting.
- [ ] Preview bounds, triangle count, overlap removed, and confidence before applying synthesis.
- [ ] Validate dimensions, boundaries, normals, and topology; record reproducible source parameters.
- [ ] Extend the distributed tile contract to alignment, comparison, and synthesis after LAN
  reduction is established.

## Printable solid synthesis

Develop a separate depth-guided mode for reconstructing a closed printable exterior from an open
or incomplete surface. Ordinary reduction continues to preserve intentional open boundaries.

- [ ] Preview filled regions and estimated enclosed volume.
- [ ] Control the smallest hole to fill and largest gap to bridge.
- [ ] Protect deliberate openings through explicit regions or boundary selection.
- [ ] Remove redundant internal geometry.
- [ ] Validate watertightness, manifold edges, normals, self-intersections, and positive volume.
- [ ] Report dimensional drift and surface deviation, and identify unresolved regions before export.

## Platform support and application upgrades

- [ ] Validate Linux x86-64 previews across distributions, desktop environments, display servers,
  and GPU drivers.
- [ ] Validate Apple silicon and Intel macOS previews on real hardware.
- [ ] Add platform-native hardware metrics behind the shared metrics interface.
- [ ] Add a supported Linux AppImage and verified upgrade/relaunch tests.
- [ ] Add Developer ID signing and notarization for macOS, then verified in-app replacement.
- [ ] Preserve complete unsaved editing state through upgrades.
- [ ] Keep settings, keyboard mappings, CLI behavior, and project formats portable.

## Later project and collaboration work

These features follow distributed processing; they are not prerequisites for the initial LAN worker.

- [ ] Persist local objects, operations, checkpoints, branches, and reproducible history across sessions.
- [ ] Add synchronized mesh/region comparison and visual differences.
- [ ] Record contribution parent versions, parameters, affected regions, dependencies, and checksums.
- [ ] Merge independently validated non-overlapping contributions.
- [ ] Detect overlapping or dependency conflicts and provide visual resolution, rebase, and rerun.
- [ ] Add contribution review states and reproducible synthesis manifests.
- [ ] Add project-scoped permissions and contributor controls.
- [ ] Add bounded cloud-worker support after LAN scheduling, trust, and recovery are validated.

## Testing and scope

Use the bundled composite scan and independent purpose-built fixtures for quality, performance,
boundary, interruption, and assembly tests. Preserve benchmark inputs and report setup costs,
output counts, memory use, and quality alongside runtime.

MeshMill focuses on geometry inspection, reduction, cleanup, measurement, validation, and mesh
interchange. General modeling, sculpting, animation, materials, and scene creation remain outside
the roadmap.
