# Axial calibration

Use surface-contact probes to correct the center and tilt of a rotation axis. Calibration changes
the inspection reference without reshaping the mesh.

## Calibrate

1. Open a mesh and orient it for inspection.
2. Expand **Axial calibration**. The model is fitted to the viewport while retaining its orientation.
3. Choose **Vertical (turntable)**, **Left / Right**, or **In / Out**, relative to the current view.
4. Click the surface at several heights to place probes. Drag a probe to reposition it, or double-click to remove it.
5. Select **Rotate**, or press `Space`. **Reverse** changes the direction.
6. Watch angular coverage, runout, and correction values. The viewport X/Y cross stays horizontal
   and vertical. Lighter original axes show X/Y/Z degree offsets at both ends, inside the viewport edges.
   Use the rotation and panning keys for manual correction; Ctrl+Shift+Up/Down moves the alignment
   axis closer or farther through the mesh.
7. At **Stable**, fitting stops and probe overlays clear while rotation continues. Add another probe
   to refine the alignment further, or select **Finalize** to keep it and return to Optimize.

Finalize also sets the export frame: the calibrated axis becomes Z and its center becomes the
origin. STL, COLLADA, OBJ, PLY, and GLB exports receive the same rigid alignment, including cast
previews and separate halves. Distances and surface shape are preserved. The working geometry
stays in its existing coordinates; an isolated copy is transformed when saving. Generated
PowerShell workflows include that final export transform.

**Clear probes** removes contacts while retaining the current alignment. **Reset** restores
object-center rotation. Changing the viewport axis clears the probes and their fit.

During alignment rotation, framing follows the model so it stays fitted. Manual zoom takes
precedence for the rest of that alignment session. Reopening the tool enables automatic fitting
again. This behavior is limited to axial calibration.

## Measurements

- **Coverage:** how much of a full turn has valid contact for every active probe.
- **Runout:** radial variation measured as the surface passes a probe.
- **Offset:** lateral correction of the rotation center.
- **Tilt:** angular correction relative to the starting viewport orientation.
- **Stable:** coverage and validation are complete, with no further useful correction found.

The [published report](https://github.com/make-your-own-world/rnd/tree/main/axial-calibration)
contains the equations, symbol definitions, synthetic tests, implementation decisions, and attribution.
