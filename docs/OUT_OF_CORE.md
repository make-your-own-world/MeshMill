# Out-of-core mesh architecture

MeshMill estimates working memory before allocating a complete mesh. Files that exceed the
configured budget can open as bounded navigation overviews or use full-resolution tiled
processing. An overview is sampled geometry, is visibly identified as such, and cannot be edited
or exported as though it were the complete source. Tiled processing uses a persistent spatial
index and streamed assembly. Zoom-dependent full-resolution viewport detail remains planned.

## Index format

The version 1 implementation gives each binary STL source a private, versioned `.meshmill-index`
cache directory containing:

- `manifest.json`, with the source size, modification time, bounded content fingerprint, bounds,
  triangle count, index version, tile target, and adaptive leaf records;
- adaptive spatial tiles addressed by octree level and Morton code;
- full-resolution triangle records in leaf tiles; and
- an atomically published manifest with adaptive leaf metadata.

Later index versions will add coarse display meshes for occupied parents plus operation-specific
boundary ownership and overlap metadata without changing existing source files.

Index creation reads the source sequentially in bounded blocks. Dense octree leaves subdivide until
they fit the configured work-unit target; sparse leaves remain coarse. This avoids forcing every
part of an irregular scan to the resolution required by its densest region. It writes temporary
tile files and atomically publishes the manifest after every required file passes validation. No
more than 64 tile files remain open at once. An
interrupted or stale index is detected from source metadata and bounded content hashes. The last
complete index remains active until its replacement is published.

The index uses source-neutral tile identities. A future multi-mesh workspace can overlay leaf sets
from several source indexes, retain source provenance per contribution, and subdivide only tiles
that require closer overlap comparison. Merge and synthesis therefore extend the same hierarchy
instead of introducing a separate whole-mesh representation.

## Viewport streaming

The viewport selects tiles using the camera frustum and screen-space error. Coarse parent tiles are
shown first. Visible child tiles replace them as the camera moves closer, while off-screen and
low-impact tiles remain coarse. RAM and VRAM have independent budgets and least-recently-used
caches. Releasing detail never releases the coarse whole-object representation.

The scheduler records these tile states: queued, reading, processing, uploading, resident, failed,
and cancelled. The viewport can color cubes by state and fill each cube in proportion to its
progress. Cancellation removes partial results and leaves the last complete representation active.

## Processing and capacity

A local work unit is one adaptive tile. Concurrency is capped by configured memory percentage,
logical processor count, and a conservative estimate derived from the largest pending tile. GPU
algorithms currently use one tile worker to avoid device contention. GPU upload and display have a
separate VRAM budget.

Operations preserve vertices on every tile-open edge. Each checkpoint records a checksum and is
validated before reuse. Assembly streams results in deterministic tile order, checks its exact
triangle count, and records the parameters and output checksum in an atomic run manifest. The same
work-unit model can later be scheduled across distributed synthesis nodes.

## CPU, GPU, and resident geometry

Storage reads, STL validation, index publication, queue ownership, topology changes, and boundary
assembly remain CPU responsibilities. Parallel per-triangle or per-vertex calculations may run on
the GPU when a measured work unit is large enough to recover dispatch and synchronization cost.

The viewport already stores visible geometry in OpenGL buffers. A GPU backend should consume those
resident buffers through a supported graphics-compute interoperability path instead of uploading a
second copy. VTK exposes mapper vertex-buffer handles, while CUDA supports registering and mapping
OpenGL buffers. Mapping is valid only while OpenGL is not using the resource, so ownership and
synchronization belong to the render thread. This path is optional and requires a native boundary;
the portable CPU path remains authoritative until equivalent results pass validation.

Adapter selection follows the active graphics context. MeshMill must query the compute device that
backs the current OpenGL context and must not enumerate or borrow other installed adapters. This
keeps system-default behavior and avoids interfering with GPUs assigned to other services.

For meshes larger than VRAM, the resident set contains a coarse whole-object representation plus
visible or active full-resolution tiles. RAM and VRAM use separate least-recently-used budgets.
Dirty tiles remain pinned until their result is applied or cancelled. Rendering and compute share
the same resident-tile catalog even when a platform backend cannot share the same physical buffer.

See [`GPU_OUT_OF_CORE.md`](GPU_OUT_OF_CORE.md) for measurements, decision thresholds, and the
validation plan.

## Safety rules

- A global sample is labeled as an overview, not full-resolution viewport detail.
- An overview cannot overwrite or export as the complete source mesh.
- Full-load requests that exceed the current budget require an explicit choice.
- Index generation, tile processing, and assembly remain cancellable and preserve the previous
  complete state.
- Capacity values are estimates and identify whether they describe the current engine or planned
  parallel tile execution.
