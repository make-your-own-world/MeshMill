# Headless workflows

This guide describes MeshMill 0.4.

The desktop script generator records committed tool operations in order. Undo and
Redo restore the corresponding workflow. Unapplied previews are excluded.

Generated PowerShell scripts have editable `InputMesh`, `OutputMesh` and `CliPath`
parameters. Output files are not overwritten. A failed operation leaves the input
unchanged and does not publish a partial final output. Temporary job files are
removed when the script exits.

```powershell
.\workflow.ps1 -InputMesh 'part.stl' -OutputMesh 'part.dae'
.\MeshMillCLI.exe part.stl --workflow workflow.json --output part.ply
```

## Operation sequence

```json
{
  "version": 1,
  "steps": [
    {"kind": "solidify", "method": "Surface repair", "quality": "balanced", "gap_mm": 2},
    {"kind": "cast", "mode": "Clamshell", "shape": "Box", "offset_mm": 0.5,
     "wall_mm": 2, "bridge_mm": 0, "split_normal": [0, 0, 1]},
    {"kind": "optimize", "algorithm": "density", "quality": "fine", "processing": "full"}
  ]
}
```

Each operation receives the preceding result. Full-mesh operations retain their
in-memory coordinates between passes. Tiled operations use the existing binary-STL
indexing and streamed assembly engine. Closed-solid optimization uses the same
closed-output validation as the application.

| Operation | Settings |
| --- | --- |
| `optimize` (default) | `algorithm`: `auto`, `fast`, `density`, `shape`, `topology`, `depth-adaptive`, `depth-density-adaptive`; `quality`: `draft`, `balanced`, `fine`; `aggressiveness`: 0–10; optional explicit `target` |
| `solidify` | `method`: `Surface repair`; `quality`: `draft`, `balanced`, `fine`; `gap_mm` |
| `cast` | `mode`: `Clamshell` or `Envelope`; `offset_mm`; `shape`: `Blob` or `Box`; `wall_mm`; `bridge_mm`; `detail` |
| `convert` | Preserve geometry and export in the output filename's format |

Shared settings: `gpu` accepts `system` or a selected CUDA PCI bus ID.
`memory_percent` bounds operation memory planning. Optimize also accepts
`processing` (`auto`, `full`, `tiled`), `workers` (0 for automatic), `auto_tiling`,
and `preserve_closed`. A requested GPU-guided algorithm reports an error if its
runtime or device is unavailable.

Cast accepts `split_normal` (three components), optional `split_origin` (three
millimeter coordinates), and `separate_halves` (translate the halves apart as in
the viewport). An omitted or null normal leaves the cast unsplit. Split halves
remain separate connected components in the combined output mesh. `prepare_solid`
enables preparation of open input using `method`, `gap_mm` and `solidify_quality`.

Output format follows the filename extension: STL, COLLADA, GLB, OBJ or PLY.
The final step can include `export_transform`, a finite, right-handed rigid 4x4 matrix.
Generated scripts record finalized axial calibration here, using the calibrated axis as Z
and its center as the export origin. The transform is applied once, after geometry operations.
`merge` on the final step controls coplanar polygon consolidation for formats that
support it. The generator uses adaptive quality presets rather than hardcoded
triangle counts, so scripts can process differently sized input meshes.

## Interactive edits

Crop, deletion, selection-only tools and manually seeded surface detection depend
on viewport input. The generator saves their resulting geometry as a separate
binary PLY checkpoint and records later operations against that checkpoint. It
retains double-precision coordinates. Earlier tool
steps already represented in the checkpoint are not run again. Keep the checkpoint
with the script, or override `InputMesh` with another prepared input.

Camera rotation, display mode, ruler measurements and preview opacity do not change
exported geometry and are not replayed. Development scripts use the current Python
interpreter and discovered repair-runtime paths. Packaged scripts use MeshMillCLI.
