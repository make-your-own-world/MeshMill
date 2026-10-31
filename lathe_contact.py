"""Exact surface contacts for a lathe probe held in the viewing rig."""

import numpy as np
from vtkmodules.vtkCommonCore import mutable
from vtkmodules.vtkCommonDataModel import vtkGenericCell, vtkStaticCellLocator


def build_contact_index(mesh):
    locator = vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    return locator


def finger_frame(point, center, axis, forward):
    forward = np.asarray(forward, dtype=float)
    forward -= axis * float(forward @ axis)
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, axis)
    relative = np.asarray(point) - center
    height = float(relative @ axis)
    radial = relative - height * axis
    radius = max(float(np.linalg.norm(radial)), 1e-9)
    radial /= radius
    return height, float(radial @ right), float(radial @ -forward), radius


def finger_ray(frame, center, axis, forward, extent, offset=0.0):
    height, lateral, front, radius = frame
    forward = np.asarray(forward, dtype=float)
    forward -= axis * float(forward @ axis)
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, axis)
    radial = right * lateral - forward * front
    radial /= max(np.linalg.norm(radial), 1e-12)
    station = center + axis * (height + radius * offset)
    return station + radial * extent, station - radial * extent


def surface_contact(locator, start, end, matrix):
    inverse = np.linalg.inv(matrix)
    local_start = (inverse @ np.r_[start, 1.0])[:3]
    local_end = (inverse @ np.r_[end, 1.0])[:3]
    time = mutable(0.0)
    point = [0.0, 0.0, 0.0]
    coordinates = [0.0, 0.0, 0.0]
    sub_id, cell_id = mutable(0), mutable(0)
    if not locator.IntersectWithLine(
        local_start, local_end, 1e-8, time, point, coordinates, sub_id, cell_id, vtkGenericCell()
    ):
        return None
    return (matrix @ np.r_[point, 1.0])[:3]
