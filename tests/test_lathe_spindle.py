import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData

from lathe_contact import build_contact_index
from lathe_spindle import correct_spindle_pose, first_harmonics, pose_matrix


def stepped_shaft(oval=False):
    angles = np.arange(128) * 2 * np.pi / 128
    profiles = [(-5, 1.5), (-1, 1.5), (-1, 2), (1, 2), (1, 0.9), (5, 0.9)]
    points = vtkPoints()
    faces = vtkCellArray()
    for height, radius in profiles:
        for angle in angles:
            points.InsertNextPoint(
                radius * np.cos(angle) * (1.15 if oval else 1), radius * np.sin(angle), height
            )
    for ring in range(len(profiles) - 1):
        for index in range(128):
            a = ring * 128 + index
            b = ring * 128 + (index + 1) % 128
            c = a + 128
            d = b + 128
            for triangle in ((a, b, c), (b, d, c)):
                faces.InsertNextCell(3)
                for vertex in triangle:
                    faces.InsertCellPoint(vertex)
    mesh = vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(faces)
    return mesh


def test_fixed_spindle_two_fingers_remove_offset_and_tilt():
    locator = build_contact_index(stepped_shaft())
    wrong = pose_matrix((0.3, -0.2, 0.1, -0.08))
    probes = [(-3.0, 1.0, 0.0, 1.5), (3.0, 1.0, 0.0, 0.9)]
    parameters = np.zeros(4)
    first = None
    for _ in range(9):
        result = correct_spindle_pose(locator, wrong, parameters, probes, 12.0)
        if first is None:
            first = np.linalg.norm(result.before)
        parameters = result.parameters
    assert np.max(result.after) < 0.002
    final = pose_matrix(parameters) @ wrong
    assert np.linalg.norm(final[:2, 2]) < 0.001
    assert np.linalg.norm(final[:2, 3]) < 0.002
    assert np.linalg.norm(result.after) < first * 0.01


def test_added_finger_corrects_tilt_without_losing_final_first_alignment():
    locator = build_contact_index(stepped_shaft())
    wrong = pose_matrix((0.25, -0.15, 0.08, -0.06))
    probes = [(-3.0, 1.0, 0.0, 1.5), (3.0, 1.0, 0.0, 0.9)]
    parameters = np.zeros(4)
    for _ in range(4):
        result = correct_spindle_pose(locator, wrong, parameters, probes[:1], 12.0)
        parameters = result.parameters
    assert result.after[0] < 0.002
    for _ in range(8):
        result = correct_spindle_pose(locator, wrong, parameters, probes, 12.0)
        parameters = result.parameters
    assert np.max(result.after) < 0.002


def test_nonround_shape_retains_form_variation_after_alignment():
    locator = build_contact_index(stepped_shaft(oval=True))
    wrong = pose_matrix((0.2, -0.1, 0.05, -0.04))
    probes = [(-3.0, 1.0, 0.0, 1.5), (3.0, 1.0, 0.0, 0.9)]
    parameters = np.zeros(4)
    for _ in range(8):
        result = correct_spindle_pose(locator, wrong, parameters, probes, 12.0)
        parameters = result.parameters
    assert np.max(result.after) < 0.003
    assert np.min(result.total_runout) > 0.1


def test_profile_harmonics_separate_shape_from_centering():
    angles = np.arange(128) * 2 * np.pi / 128
    profiles = np.array(
        [3 + 0.2 * np.cos(angles) - 0.1 * np.sin(angles) + 0.4 * np.cos(2 * angles)]
    )
    assert first_harmonics(angles, profiles) == pytest.approx((0.2, -0.1))


def test_sampling_variation_does_not_hide_true_offset_on_smooth_surfaces():
    from lathe_spindle import sampling_uncertainty
    angles=np.arange(96)*2*np.pi/96
    profiles=np.array([3+.2*np.cos(angles)-.1*np.sin(angles)+.3*np.cos(2*angles)])
    assert sampling_uncertainty(angles,profiles)[0]<1e-10
