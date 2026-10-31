"""Sample opposing wall pairs and use consistent thickness to identify skin layers."""
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import mutable
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator


def analyze_wall_thickness(mesh, normals, sheet_data, supported, cancel=None, progress=lambda *args: None):
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    centers = points[faces].mean(axis=1)
    labels = sheet_data["labels"]
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    epsilon = max(scale * 1e-7, 1e-9)
    sample_ids = np.linspace(0, len(faces) - 1, min(8192, len(faces))).astype(int)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    t, sub, target = mutable(0.), mutable(0), mutable(0)
    hit, param = [0.] * 3, [0.] * 3
    pairs = {}
    progress("Measuring wall thickness", 0, len(sample_ids))
    for done, index in enumerate(sample_ids):
        if done % 128 == 0:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Wall analysis cancelled.")
            progress("Measuring wall thickness", done, len(sample_ids))
        normal = normals[index]
        if not locator.IntersectWithLine(centers[index] - epsilon * normal,
                                         centers[index] - scale * 2 * normal,
                                         epsilon * .01, t, hit, param, sub, target):
            continue
        other = int(target)
        if other == index or float(normal @ normals[other]) > -.7:
            continue
        distance = float(np.linalg.norm(np.asarray(hit) - centers[index]))
        if distance <= epsilon * 10:
            continue
        key = tuple(sorted((int(labels[index]), int(labels[other]))))
        pairs.setdefault(key, []).append((index, other, distance))
    progress("Measuring wall thickness", len(sample_ids), len(sample_ids))
    selected = supported.copy()
    thicknesses = []
    validated_pairs = 0
    excluded = set()
    for (first, second), records in pairs.items():
        if len(records) < 8:
            continue
        distances = np.array([record[2] for record in records])
        median = float(np.median(distances))
        relative_spread = float(np.median(np.abs(distances - median)) / median)
        sample_centers = centers[[record[0] for record in records]]
        extent = max(float(np.max(sheet_data["upper"][[first, second]] - sheet_data["lower"][[first, second]])), epsilon)
        coverage = float(np.ptp(sample_centers, axis=0).max() / extent)
        if relative_spread > .2 or coverage < .45:
            continue
        if first == second:
            # A single connected sheet can join inner and outer tube walls at
            # rounded rims. Recognize a cylindrical normal field before using
            # radius to separate those paired layers.
            region_ids = np.flatnonzero(labels == first)
            covariance = normals[region_ids].T @ normals[region_ids] / max(1, len(region_ids))
            values, vectors = np.linalg.eigh(covariance)
            if values[1] <= .1 or values[0] / values[1] >= .2:
                continue
            axis = vectors[:, 0]
            center = np.mean(centers[region_ids], axis=0)
            relative = centers[region_ids] - center
            radius = np.linalg.norm(relative - (relative @ axis)[:, None] * axis, axis=1)
            all_radius = np.zeros(len(faces))
            all_radius[region_ids] = radius
            outside = np.array([max(all_radius[a], all_radius[b]) for a, b, _ in records])
            inside = np.array([min(all_radius[a], all_radius[b]) for a, b, _ in records])
            if np.median(np.abs(outside - np.median(outside))) > median * .35:
                continue
            if np.median(outside - inside) < median * .65:
                continue
            validated_pairs += 1
            thicknesses.append(median)
            cutoff = float((np.median(outside) + np.median(inside)) / 2)
            selected[region_ids[radius < cutoff]] = False
            continue
        seeds = sheet_data["seed_regions"]
        skin = first if first in seeds and second not in seeds else second if second in seeds and first not in seeds else None
        if skin is None:
            for outer, inner in ((first, second), (second, first)):
                low, high = sheet_data["lower"], sheet_data["upper"]
                margin = scale * .005
                contains = np.all(low[inner] >= low[outer] - margin) and np.all(high[inner] <= high[outer] + margin)
                strict = np.count_nonzero((low[inner] > low[outer] + margin) & (high[inner] < high[outer] - margin))
                if contains and strict >= 2 and sheet_data["areas"][outer] > sheet_data["areas"][inner]:
                    skin = outer
                    break
        if skin is not None:
            validated_pairs += 1
            thicknesses.append(median)
            inner = second if skin == first else first
            if inner not in seeds:
                selected[labels == inner] = False
                selected[labels == skin] = True
                excluded.add(inner)
    return selected, {
        "wall_pairs": validated_pairs,
        "wall_thickness_mm": float(np.median(thicknesses)) if thicknesses else None,
        "thickness_excluded_sheets": len(excluded), "wall_samples": len(sample_ids),
    }
