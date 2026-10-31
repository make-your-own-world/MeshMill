# Solidify and Cast

Solidify and Cast are available in MeshMill 0.4.

## Shared workflow

Actions contains preview comparison, Apply, Cancel, Undo, Redo, display mode, and saving.
A preview remains provisional until applied or used to replace the working mesh. Compare it
with Original or Previous before accepting it. Undo and Redo restore committed geometry states.
Targets and statistics follow the current working mesh after each accepted operation.

## Solidify

Use Solidify to complete supported open or incomplete surfaces into closed geometry. It can
operate on the whole mesh or selected regions. Surface repair combines measured surface support,
boundary repair, and interior classification. The controls report boundary and completion measurements.

1. Open a mesh and optionally select the regions to repair.
2. Expand Solidify when completion is available and choose its settings.
3. Select Solidify and inspect the preview, especially missing or ambiguous regions.
4. Apply the result or Cancel the preview. Use Undo to return to an earlier committed state.

Completion is inferred where scan data is absent. Check thin walls, intentional openings, and
poorly observed regions against the intended part. Validation reports unresolved geometry rather
than treating an incomplete result as a successful closed solid. Already closed geometry does not
need another Solidify operation.

## Cast

Cast provides two outputs:

| Mode | Result |
| --- | --- |
| Envelope | An offset surface around the detected exterior |
| Clamshell (default) | A hollow body with a fitted cavity and optional split |

Clearance defaults to 0.5 mm. Choose a fitted Blob exterior or a Box exterior. Sampling, wall
thickness, and gap bridging are derived from the geometry and memory budget, with adjustable
controls. Manual surface hints help identify the intended exterior on ambiguous scans.
Open input can be prepared with Solidify when needed for a cavity.

Change settings, then explicitly update the preview. Overlay visibility, source visibility,
and opacity help inspect the fit without exporting another file. Display settings do not
change the output geometry.

### Split and export

Choose X, Y, Z, current view, or a custom plane. Custom controls move and rotate the split plane
in the viewport. The split passes through the body and its cavity. Halves remain separate
connected parts; displaying them apart changes their placement rather than joining them.
Save halves separately when separate part files are needed.

Replace mesh accepts the generated result as the working geometry so it can be reduced,
selected, measured, or exported. Optimize becomes the active tool afterward. Geometry changes
remain available through Undo and Redo.

Use the [format comparison](EXPORT_FORMATS.md) to choose an export. Fit and printability also
depend on the scan, selected clearance, printer, material, and downstream processing. Inspect
critical dimensions and the slicer preview before printing a fitted part.

## Repeatable processing

Generated scripts record committed tool operations. Interactive crops, selections, and manual
surface hints become an input checkpoint; later operations run against that geometry.
See [headless workflows](AUTOMATION.md) for supported parameters and limitations.

See the [surface completion and cast mathematics](SOLIDIFY_CAST_MATH.md) for the implemented objectives, constraints, and limitations.
