import numpy as np
import pytest
from test_clamshell_mold import inside
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkTriangleFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource

from meshmill import arrays_polydata
from solidify import solidify_mesh


def open_cube():
    cube = vtkCubeSource()
    cube.SetBounds(-5,5,-5,5,-5,5)
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(cube.GetOutputPort())
    triangles.Update()
    mesh=triangles.GetOutput()
    points=vtk_to_numpy(mesh.GetPoints().GetData())
    faces=vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1,3)
    centers=points[faces].mean(axis=1)
    return arrays_polydata(points,faces[centers[:,2]<4.9])


@pytest.mark.parametrize("method",["Boundary patches","Convex envelope"])
def test_complete_missing_face_into_solid(method):
    source=open_cube()
    before=source.GetNumberOfCells()
    output,report=solidify_mesh(source,method,2,.5,128*1024**2)
    assert inside(output,(0,0,0))
    assert not inside(output,(8,0,0))
    assert report["topology"]["boundary_edges"]==0
    assert source.GetNumberOfCells()==before


def test_voxel_solid_fills_closed_object():
    source=vtkSphereSource()
    source.SetRadius(5)
    source.Update()
    output,report=solidify_mesh(source.GetOutput(),"Voxel gap closing",1,.5,128*1024**2)
    assert inside(output,(0,0,0))
    assert report["topology"]["boundary_edges"]==0


@pytest.mark.parametrize("method",["Surface repair"])
def test_prepared_solid_drives_empty_clamshell_cavity(method):
    from test_clamshell_mold import points_inside

    from clamshell_mold import clamshell_mold
    from solidify import offset_solid
    solid,_=solidify_mesh(open_cube(),method,1,.5,128*1024**2)
    skin,_=offset_solid(solid,.5)
    halves,report=clamshell_mold(skin,wall_mm=2,bridge_mm=0,pitch_mm=.5,split_normal=(.4,.7,.2))
    assert report["prepared_solid"]
    known_empty=np.array([[x,y,z] for x in (-2,0,2) for y in (-2,0,2) for z in (-2,0,2)])
    for half in halves:
        assert not points_inside(half,known_empty).any()


def test_scope_application_preserves_unselected_faces():
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData

    from meshmill import polydata_arrays
    from solidify import combine_selected_solid
    first=open_cube()
    second=vtkCubeSource();second.SetCenter(30,0,0);second.Update()
    joined=vtkAppendPolyData();joined.AddInputData(first);joined.AddInputConnection(second.GetOutputPort());joined.Update()
    tri=vtkTriangleFilter();tri.SetInputConnection(joined.GetOutputPort());tri.Update()
    points,faces=polydata_arrays(tri.GetOutput())
    selected=points[faces].mean(axis=1)[:,0]<10
    solid,_=solidify_mesh(first,"Boundary patches",2,.5,128*1024**2)
    combined=combine_selected_solid(points,faces,selected,solid)
    result_points,result_faces=polydata_arrays(combined)
    untouched=points[faces[~selected]]
    result_untouched=result_points[result_faces][result_points[result_faces].mean(axis=1)[:,0]>10]
    assert np.array_equal(untouched,result_untouched)


def test_multiple_selected_components_solidify_independently():
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkAppendPolyData
    from vtkmodules.vtkFiltersGeneral import vtkTransformFilter

    from solidify import needs_solidification
    first=open_cube()
    move=vtkTransform();move.Translate(30,0,0)
    shifted=vtkTransformFilter();shifted.SetInputData(first);shifted.SetTransform(move);shifted.Update()
    combined=vtkAppendPolyData();combined.AddInputData(first);combined.AddInputConnection(shifted.GetOutputPort());combined.Update()
    assert needs_solidification(combined.GetOutput())
    result,report=solidify_mesh(combined.GetOutput(),"Boundary patches",2,.5,128*1024**2)
    assert inside(result,(0,0,0))
    assert inside(result,(30,0,0))
    assert not inside(result,(15,0,0))
    assert report["topology"]["boundary_edges"]==0


def test_boundary_fill_keeps_original_surface_triangles():
    from vtkmodules.util.numpy_support import vtk_to_numpy
    source=open_cube()
    source_points=vtk_to_numpy(source.GetPoints().GetData()).astype(np.float32)
    source_faces=vtk_to_numpy(source.GetPolys().GetConnectivityArray()).reshape(-1,3)
    result,_=solidify_mesh(source,"Boundary patches",2,.5,128*1024**2)
    result_points=vtk_to_numpy(result.GetPoints().GetData()).astype(np.float32)
    result_faces=vtk_to_numpy(result.GetPolys().GetConnectivityArray()).reshape(-1,3)
    def triangles(points,faces):
        return {tuple(sorted(map(tuple,points[face]))) for face in faces}
    assert triangles(source_points,source_faces).issubset(triangles(result_points,result_faces))


def test_cancelled_surface_worker_is_joined():
    import multiprocessing
    import threading

    from surface_repair import repair_surface
    before={child.pid for child in multiprocessing.active_children()}
    event=threading.Event()
    event.set()
    with pytest.raises(RuntimeError,match="cancelled"):
        repair_surface(open_cube(),cancel=event)
    assert {child.pid for child in multiprocessing.active_children()}==before
