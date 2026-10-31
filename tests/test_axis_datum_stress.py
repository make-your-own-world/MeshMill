"""Adversarial synthetic calibration shapes and independent monotonic convergence."""
import json
import time

import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

from axis_datum import fit_probe_datum, meaningful_improvement
from lathe_contact import build_contact_index
from lathe_spindle import contact_profiles, first_harmonics, pose_matrix


def shape_mesh(kind):
    angles = np.arange(96) * 2 * np.pi / 96
    points, cells = vtkPoints(), vtkCellArray()
    heights = np.linspace(-5, 5, 7)
    for height in heights:
        for angle in angles:
            radius = 1.4
            if kind == "stepped":
                radius = 1.1 if height > 0 else 1.7
            elif kind == "tapered":
                radius += .05 * height
            elif kind == "lobed":
                radius += .18 * np.cos(3 * angle)
            elif kind == "square":
                radius /= max(abs(np.cos(angle)), abs(np.sin(angle)))
            elif kind == "noisy":
                radius += .012 * np.sin(17 * angle + height) + .007 * np.cos(23 * angle)
            elif kind == "dented":
                radius -= .15 * np.exp(-((angle - .7) / .25) ** 2)
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            if kind == "oval":
                x *= 1.2
            if kind == "bent":
                x += .012 * height ** 2
            points.InsertNextPoint(x, y, height)
    for ring in range(6):
        for i in range(96):
            if kind == "missing_wedge" and i < 35:
                continue
            a, b = ring * 96 + i, ring * 96 + (i + 1) % 96
            for tri in ((a, b, a + 96), (b, b + 96, a + 96)):
                cells.InsertNextCell(3)
                for vertex in tri:
                    cells.InsertCellPoint(vertex)
    mesh = vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


@pytest.mark.parametrize("kind", ["round", "stepped", "tapered", "oval", "lobed", "square", "noisy", "dented", "bent"])
def test_broken_shapes_converge_with_real_independent_improvement(kind):
    mesh = shape_mesh(kind)
    original = np.array([mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
    locator = build_contact_index(mesh)
    wrong = pose_matrix((.23, -.17, .065, -.045))
    probes = [(-3., 1., 0., 1.4), (0., 1., 0., 1.4), (3., 1., 0., 1.4)]
    parameters = np.zeros(4)
    initial = None
    history = []
    start = time.perf_counter()
    for iteration in range(24):
        result = fit_probe_datum(locator, wrong, parameters, probes, 14.)
        if initial is None:
            initial = float(np.linalg.norm(result.before))
        if not result.accepted:
            assert result.parameters == pytest.approx(parameters)
            break
        assert meaningful_improvement(result.before, result.after, 14.)
        # A different 257-angle grid checks that acceptance isn't overfitting its audit grid.
        angles, before = contact_profiles(locator, wrong, parameters, probes, 14., count=257, phase_offset=.31)
        _, after = contact_profiles(locator, wrong, result.parameters, probes, 14., count=257, phase_offset=.31)
        old = np.linalg.norm(first_harmonics(angles, before))
        new = np.linalg.norm(first_harmonics(angles, after))
        assert new < old + 1e-6
        history.append(float(np.linalg.norm(result.after)))
        parameters = result.parameters.copy()
    else:
        pytest.fail("Did not terminate within 24 meaningful corrections")
    assert history
    assert float(np.linalg.norm(result.after)) < initial
    repeat = fit_probe_datum(locator, wrong, parameters, probes, 14.)
    assert not repeat.accepted
    assert repeat.parameters == pytest.approx(parameters)
    assert np.array_equal(original, [mesh.GetPoint(i) for i in range(mesh.GetNumberOfPoints())])
    print(json.dumps({"shape": kind, "steps": len(history), "initial": initial,
                      "final": float(np.linalg.norm(result.after)), "seconds": round(time.perf_counter()-start, 3)}))


def test_incomplete_contact_is_not_successful_calibration():
    locator = build_contact_index(shape_mesh("missing_wedge"))
    with pytest.raises(ValueError, match="contact|coverage"):
        fit_probe_datum(locator, np.eye(4), np.zeros(4), [(-3., 1., 0., 1.4), (3., 1., 0., 1.4)], 14.)


def test_numerical_noise_and_regression_are_not_improvement():
    assert not meaningful_improvement(np.array([1.]), np.array([1.-1e-10]), 14.)
    assert not meaningful_improvement(np.array([1.]), np.array([1.01]), 14.)
    assert meaningful_improvement(np.array([1.]), np.array([.99]), 14.)
