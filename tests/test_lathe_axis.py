import numpy as np
import pytest

from lathe_axis import fit_lathe_axis, probe_errors, refine_lathe_axis


def test_contact_sweep_finds_eccentric_cylinder_center():
    angles = np.linspace(0, 2*np.pi, 180, endpoint=False)
    points = np.column_stack((3+5*np.cos(angles), np.zeros(len(angles)), -2+5*np.sin(angles)))
    result = fit_lathe_axis(points, (0,1,0))
    assert result.center == pytest.approx((3,0,-2), abs=1e-6)
    assert result.radius == pytest.approx(5, abs=1e-6)
    assert result.runout < 1e-6
    assert abs(result.axis @ (0,1,0)) > .999999


def test_flat_probe_sweep_does_not_claim_a_center():
    with pytest.raises(ValueError):
        fit_lathe_axis(np.column_stack((np.arange(20), np.zeros(20), np.zeros(20))), (0,1,0))


def test_tilted_multiheight_sweep_corrects_axis():
    angles = np.tile(np.linspace(0, 2*np.pi, 80, endpoint=False), 3)
    heights = np.repeat((-4,0,4),80)
    tilt = .12
    axis = np.array((np.sin(tilt),np.cos(tilt),0))
    right = np.array((np.cos(tilt),-np.sin(tilt),0))
    points = (np.outer(heights,axis) + np.outer(5*np.cos(angles),right)
              + np.outer(5*np.sin(angles),(0,0,1)) + (3,0,-2))
    result = fit_lathe_axis(points,(0,1,0))
    assert abs(result.axis @ axis) > .9999
    assert result.radius == pytest.approx(5, abs=.01)


def test_multiple_probe_heights_with_different_radii_share_one_centerline():
    angles = np.tile(np.linspace(0,2*np.pi,120,endpoint=False),3)
    groups = np.repeat(np.arange(3),120)
    heights = np.repeat((-5.,0.,5.),120)
    radii = np.repeat((2.,4.,3.),120)
    tilt = .14
    axis = np.array((np.sin(tilt),np.cos(tilt),0))
    side = np.array((np.cos(tilt),-np.sin(tilt),0))
    points = (np.outer(heights,axis) + np.outer(radii*np.cos(angles),side)
              + np.outer(radii*np.sin(angles),(0,0,1)) + (2.,0.,-3.))
    result = fit_lathe_axis(points,(0,1,0),groups)
    assert abs(result.axis @ axis) > .9999
    delta = result.center - (2.,0.,-3.)
    assert np.linalg.norm(delta - axis * (delta @ axis)) < .02
    assert result.runout < .02


def test_new_probe_refinement_preserves_previous_station():
    angles = np.tile(np.linspace(0,2*np.pi,120,endpoint=False),2)
    groups = np.repeat((0,1),120)
    tilt = .1
    axis = np.array((np.sin(tilt),np.cos(tilt),0))
    side = np.array((np.cos(tilt),-np.sin(tilt),0))
    points = (np.outer(np.repeat((0.,5.),120),axis)
              + np.outer(3*np.cos(angles),side) + np.outer(3*np.sin(angles),(0,0,1)))
    fitted = fit_lathe_axis(points,(0,1,0),groups)
    center = np.zeros(3)
    start_axis = np.array((0.,1.,0.))
    before = probe_errors(points,groups,center,start_axis)
    center, direction, protected = refine_lathe_axis(center,start_axis,fitted,points,groups,{0})
    after = probe_errors(points,groups,center,direction)
    assert after[0] <= before[0]*1.02 + fitted.radius*1e-4
    assert after[1] < before[1]
    assert protected == {0,1}
