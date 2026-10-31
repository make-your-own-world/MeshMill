"""STL-precision cleanup and topology validation for generated Cast geometry."""
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData


def clean_generated_mesh(mesh, closed=False):
    # Binary STL stores float32 coordinates. Weld at that exact precision before
    # checking topology, so validation describes the geometry SketchUp receives.
    raw = vtk_to_numpy(mesh.GetPoints().GetData())
    source_faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    if not np.isfinite(raw).all():
        raise ValueError("Generated geometry contains non-finite coordinates.")
    points, inverse = np.unique(raw.astype(np.float32), axis=0, return_inverse=True)
    faces = inverse[source_faces]
    triangles = points[faces].astype(np.float64)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1)
    keep = area > 0
    faces = faces[keep]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    duplicates = len(faces) - len(unique)
    faces = faces[np.sort(unique)]
    faces, repaired = repair_split_edges(points, faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    canonical = np.sort(edges, axis=1)
    _, edge_ids, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    balance = np.bincount(edge_ids, weights=direction, minlength=len(counts))
    report = {"boundary_edges": int(np.count_nonzero(counts == 1)),
              "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
              "inconsistent_edges": int(np.count_nonzero((counts == 2) & (balance != 0))),
              "removed_degenerate_faces": int(np.count_nonzero(~keep)),
              "removed_duplicate_faces": duplicates, "repaired_split_edges": repaired}
    if report["nonmanifold_edges"] or report["inconsistent_edges"] or (closed and report["boundary_edges"]):
        raise ValueError(f"Generated geometry failed topology validation: {report['boundary_edges']} open edges, "
                         f"{report['nonmanifold_edges']} non-manifold edges, "
                         f"{report['inconsistent_edges']} inconsistent edges.")
    used, remap = np.unique(faces, return_inverse=True)
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(points[used], deep=True))
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(0, faces.size + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(remap.ravel().astype(np.int64), deep=True))
    output = vtkPolyData()
    output.SetPoints(vertices)
    output.SetPolys(cells)
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
    normals = vtkPolyDataNormals()
    normals.SetInputData(output)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOff()
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.Update()
    result = vtkPolyData()
    result.DeepCopy(normals.GetOutput())
    from vtkmodules.vtkCommonCore import vtkIntArray
    marker = vtkIntArray()
    marker.SetName("ImprintClosedOutput")
    marker.InsertNextValue(int(closed))
    result.GetFieldData().AddArray(marker)
    return result, report


def repair_split_edges(points, faces):
    """Repair triangle T-junctions using existing collinear boundary vertices.

    Related cleanup operation: https://github.com/thomthom/cleanup
    This independent implementation changes adjacency without moving vertices.
    """
    from vtkmodules.vtkCommonCore import vtkIdList
    from vtkmodules.vtkCommonDataModel import vtkStaticPointLocator

    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(points, deep=True))
    cloud = vtkPolyData()
    cloud.SetPoints(vertices)
    locator = vtkStaticPointLocator()
    locator.SetDataSet(cloud)
    locator.BuildLocator()
    tolerance = max(float(np.ptp(points.astype(float), axis=0).max()) * 1e-10, 1e-12)
    repaired = 0
    for _ in range(3):
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        boundary = unique[counts == 1]
        if not len(boundary):
            break
        splits = {}
        candidates = vtkIdList()
        for a, b in boundary:
            start, stop = points[[a, b]].astype(float)
            direction = stop - start
            squared_length = float(direction @ direction)
            if squared_length <= tolerance * tolerance:
                continue
            locator.FindPointsWithinRadius(np.sqrt(squared_length) / 2 + tolerance,
                                           (start + stop) / 2, candidates)
            ids = np.array([candidates.GetId(i) for i in range(candidates.GetNumberOfIds())], dtype=int)
            ids = ids[(ids != a) & (ids != b)]
            if not len(ids):
                continue
            relative = points[ids].astype(float) - start
            t = relative @ direction / squared_length
            distance = np.linalg.norm(relative - t[:, None] * direction, axis=1)
            interior = (t > 0) & (t < 1) & (distance <= tolerance)
            ids, t = ids[interior], t[interior]
            if len(ids):
                splits[(int(a), int(b))] = ids[np.argsort(t)]
        if not splits:
            break
        output = []
        for face in faces:
            for offset in range(3):
                a, b, c = (int(face[(offset + j) % 3]) for j in range(3))
                key = (min(a, b), max(a, b))
                if key not in splits:
                    continue
                inserted = splits[key] if a < b else splits[key][::-1]
                chain = [a, *inserted, b]
                output.extend((chain[j], chain[j + 1], c) for j in range(len(chain) - 1))
                repaired += 1
                break
            else:
                output.append(face)
        faces = np.asarray(output, dtype=np.int64)
    return faces, repaired
