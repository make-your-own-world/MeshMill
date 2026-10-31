import numpy as np
import pytest
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData

from surface_support_repair import fuse_surface_seams


@pytest.mark.parametrize('gap,should_join',[(.5,True),(3.,False)])
def test_fusion_closes_narrow_seam_but_keeps_wide_recess(gap,should_join):
    pitch=.25
    xyz=np.arange(-6,6+pitch,pitch)
    z,y,x=np.meshgrid(xyz,xyz,xyz,indexing='ij')
    def box(lo,hi):
        return np.maximum.reduce((lo-x,x-hi,np.abs(y)-3,np.abs(z)-3))
    field=np.minimum(box(-4,-gap/2),box(gap/2,4)).astype(np.float32)
    image=vtkImageData();image.SetDimensions(*([len(xyz)]*3));image.SetOrigin(-6,-6,-6);image.SetSpacing(pitch,pitch,pitch);image.GetPointData().SetScalars(numpy_to_vtk(field.ravel(),deep=True))
    def run(filter_,_stage):
        filter_.Update();return filter_.GetOutput()
    original=vtk_to_numpy(image.GetPointData().GetScalars()).copy()
    result=fuse_surface_seams(image,run)
    values=vtk_to_numpy(result.GetPointData().GetScalars()).reshape(field.shape)
    mid=len(xyz)//2
    assert bool(values[mid,mid,mid] < 0)==should_join
    assert values[0,0,0]>0
    assert values[mid,mid,mid+12]<0
    assert np.array_equal(original,vtk_to_numpy(image.GetPointData().GetScalars()))
