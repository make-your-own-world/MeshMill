# Opening and exporting geometry

This guide describes MeshMill 0.4.

**Open** accepts STL, COLLADA (.dae), GLB, OBJ and PLY. Polygon faces are triangulated
for editing. Imports contain geometry only: materials, textures, animation and rigs
are not retained. COLLADA scene transforms and declared units are applied; GLB
scene transforms and meter/Y-up coordinates are converted to millimeter/Z-up
coordinates. OBJ and PLY use raw coordinates, interpreted as millimeters.

GLB import supports static, uncompressed triangle meshes. Compressed geometry,
sparse accessors, skinning and morph targets produce an explicit unsupported-format
message. Point clouds need surface reconstruction before import. Bounded-memory
import remains specific to binary STL; other formats load into memory.

**Save current mesh** and **Save preview** offer these formats. Cast exports use
the same choices, including separate files for clamshell halves.

| Format | Geometry | Units |
| --- | --- | --- |
| STL | Binary triangles, suitable for slicers | Unitless coordinates |
| COLLADA (.dae) | Indexed polygon faces, suitable for SketchUp | Millimeters recorded in the file |
| GLB | Uncompressed indexed triangles for modern interchange | Meters recorded by the format |
| OBJ | Indexed polygon faces for other modeling applications | Unitless coordinates |
| PLY | Binary indexed polygon faces | Unitless coordinates |

The Save As comparison shows strengths, tradeoffs and application compatibility,
with input face counts and available size estimates. Opening the dialog does not start geometry merging.
Selecting an application recommends a format; each format can also be selected directly.

**Use these settings next time** remembers the selected format, coplanar-merge
option and application recommendation, then switches immediately to a compact
filename, folder and format dialog with face-count and file-size estimates beside
the merge option. Counts that depend on coplanar merging are calculated during export, not before saving.
Later saves remain compact until **Compare formats…** restores the full comparison
for the current and future saves, preserving the current filename and options.
Ordinary saves and cancellations do not replace the remembered preferences.

For polygon-based SketchUp editing, choose **COLLADA** in the comparison.
[SketchUp supports COLLADA import](https://help.sketchup.com/en/sketchup/importing-and-exporting-collada-files).
For OBJ and PLY, choose millimeters when the receiving application asks for units.
The Model units dropdown changes displayed measurements, rather than rescaling exports.

COLLADA preserves polygon boundaries with separate, visually identical material
identities. This prevents SketchUp's import-time coplanar merging from erasing
shallow surface detail. Geometry is stored at a larger local size in an instanced
component, with a compensating transform restoring its original dimensions and
coordinates. This preserves tiny faces that SketchUp can discard during face
creation. Neither measure changes the working mesh or simplifies its surface.

**Merge coplanar triangles** combines adjoining triangles into larger polygon faces
when they lie on the same plane. It preserves boundary vertices, curved regions,
sharp edges and openings. Regions containing holes retain their triangles, rather
than exporting a polygon that would fill the opening. The option is unavailable
for STL and GLB, which export triangles here. GLB uses 32-bit floating-point
positions in meters; coordinate conversion can introduce rounding. It does not
apply mesh simplification or optional compression.

PLY exports limit merged faces to 255 boundary vertices for compatibility with
readers that use an 8-bit face count. Larger planar regions retain their original
triangles. The export applies this limit when creating PLY polygon faces. MeshMill's PLY importer also handles larger polygon lists without truncation.

Compatibility references:

- [SketchUp GLB import](https://help.sketchup.com/en/sketchup/working-gltf-files)
- [Blender format support](https://www.blender.org/features/pipeline/)
- [Meshy upload formats](https://help.meshy.ai/en/articles/9991884-what-3d-file-formats-does-meshy-support-full-export-list)

Coplanar merging is off by default. When selected, it runs once after Save.
Merging changes an export copy. It does not modify the working mesh, its selection,
or Undo history. Use Optimize first to reduce curved geometry as well as flat
regions. Polygon exports can reduce internal edges and repeated vertex storage;
the benefit depends on the mesh and how the receiving application imports polygons.

Exports run in the background with progress and cancellation. An existing file is
replaced after its new geometry has been written successfully. MeshMill prevents
overwriting the opened source file.
