"""Local surface-sheet support analysis with explicit user seed overrides."""
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkPolyDataConnectivityFilter, vtkPolyDataNormals


def surface_sheets(mesh, seeds=()):
    tagged = vtkPolyData()
    tagged.ShallowCopy(mesh)
    ids = numpy_to_vtk(np.arange(mesh.GetNumberOfCells(), dtype=np.int64), deep=True)
    ids.SetName("imprint_source_face")
    tagged.GetCellData().AddArray(ids)
    split = vtkPolyDataNormals()
    split.SetInputData(tagged)
    split.SplittingOn()
    split.SetFeatureAngle(50)
    split.ConsistencyOff()
    split.AutoOrientNormalsOff()
    connected = vtkPolyDataConnectivityFilter()
    connected.SetInputConnection(split.GetOutputPort())
    connected.SetExtractionModeToAllRegions()
    connected.ColorRegionsOn()
    connected.Update()
    output = connected.GetOutput()
    source_ids = vtk_to_numpy(output.GetCellData().GetArray("imprint_source_face"))
    vertex_regions = vtk_to_numpy(output.GetPointData().GetArray("RegionId"))
    triangles = vtk_to_numpy(output.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    labels = np.empty(mesh.GetNumberOfCells(), dtype=np.int64)
    labels[source_ids] = vertex_regions[triangles[:, 0]]
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    vertices = points[faces]
    centers = vertices.mean(axis=1)
    normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    twice_area = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(twice_area[:, None], 1e-20)
    count = connected.GetNumberOfExtractedRegions()
    areas = np.bincount(labels, weights=twice_area / 2, minlength=count)
    lower = np.full((count, 3), np.inf)
    upper = np.full((count, 3), -np.inf)
    np.minimum.at(lower, labels, vertices.min(axis=1))
    np.maximum.at(upper, labels, vertices.max(axis=1))
    scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
    coverage = np.max(upper - lower, axis=1) / scale
    score = areas * coverage ** 2
    primary = int(np.argmax(score))
    allowed = (areas >= areas[primary] * .03) & (coverage >= .1)
    allowed[primary] = True
    # Competing, smaller sheets enclosed along at least two axes are candidates
    # for inner layers, rather than an independent end-to-end outside surface.
    margin = scale * .01
    contained = np.all(lower >= lower[primary] - margin, axis=1) & np.all(upper <= upper[primary] + margin, axis=1)
    strict_axes = ((lower > lower[primary] + margin) & (upper < upper[primary] - margin)).sum(axis=1)
    allowed[contained & (strict_axes >= 2) & (areas < areas[primary] * .98)] = False
    sign = np.ones(len(faces), dtype=np.int8)
    seed_faces = []
    seed_regions = set()
    for seed in seeds:
        index = int(np.argmin(np.sum((centers - np.asarray(seed["point"])) ** 2, axis=1)))
        region = labels[index]
        seed_regions.add(int(region))
        allowed[region] = True
        if float(normals[index] @ np.asarray(seed["normal"])) < 0:
            sign[labels == region] = -1
        seed_faces.append(index)
    report = {
        "surface_sheets": count, "supported_sheets": int(allowed.sum()),
        "primary_surface_area": float(areas[primary]), "primary_coverage": float(coverage[primary]),
        "manual_surface_picks": len(seeds),
    }
    return allowed[labels], sign, seed_faces, report, {
        "labels": labels, "areas": areas, "coverage": coverage, "lower": lower, "upper": upper,
        "seed_regions": seed_regions,
    }
