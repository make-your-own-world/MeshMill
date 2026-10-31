import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersSources import vtkSphereSource

from lathe_contact import build_contact_index, finger_frame, finger_ray, surface_contact


def test_finger_rides_surface_instead_of_following_a_material_point():
    source = vtkSphereSource();source.SetRadius(1);source.SetThetaResolution(128);source.SetPhiResolution(128);source.Update()
    mesh = source.GetOutput()
    points = vtk_to_numpy(mesh.GetPoints().GetData())
    points[:,0] *= 1.3
    points[:,2] *= .8
    mesh.GetPoints().Modified()
    original = points.copy()
    locator = build_contact_index(mesh)
    center = np.zeros(3);axis = np.array((0.,1.,0.))
    frame = finger_frame((0,0,.8),center,axis,(0,0,-1))
    contacts = []
    for angle in np.linspace(0,2*np.pi,80,endpoint=False):
        forward = np.array((-np.sin(angle),0.,-np.cos(angle)))
        start,end = finger_ray(frame,center,axis,forward,5.)
        hit = surface_contact(locator,start,end,np.eye(4))
        assert hit is not None
        expected = 1/np.sqrt((np.sin(angle)/1.3)**2+(np.cos(angle)/.8)**2)
        assert np.linalg.norm(hit) == pytest.approx(expected,abs=.003)
        contacts.append(hit)
    assert np.linalg.norm(contacts[0]-contacts[20]) > 1
    assert np.ptp(np.linalg.norm(contacts,axis=1)) > .49
    assert np.array_equal(points,original)


def test_finger_contact_respects_actor_transform():
    source = vtkSphereSource();source.SetRadius(1);source.Update()
    locator = build_contact_index(source.GetOutput())
    transform = np.eye(4);transform[0,0]=2;transform[0,3]=5
    hit = surface_contact(locator,(10,0,0),(0,0,0),transform)
    assert hit[0] == pytest.approx(7,abs=.06)
