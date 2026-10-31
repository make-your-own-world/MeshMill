import numpy as np
from test_lathe_spindle import stepped_shaft

from axis_datum import fit_probe_datum
from lathe_contact import build_contact_index
from lathe_spindle import pose_matrix


def test_probe_features_fit_shared_datum_axis_without_changing_mesh():
    locator=build_contact_index(stepped_shaft())
    wrong=pose_matrix((.3,-.2,.1,-.08))
    parameters=np.zeros(4)
    probes=[(-3.,1.,0.,1.5),(3.,1.,0.,.9)]
    for _ in range(7):
        result=fit_probe_datum(locator,wrong,parameters,probes,12.)
        parameters=result.parameters
    final=pose_matrix(parameters) @ wrong
    assert np.linalg.norm(final[:2,2])<1e-5
    assert np.linalg.norm(final[:2,3])<1e-5
    assert max(result.after)<1e-5


def test_user_reference_can_be_any_world_orientation():
    mesh=stepped_shaft()
    from vtkmodules.util.numpy_support import vtk_to_numpy
    data=vtk_to_numpy(mesh.GetPoints().GetData())
    source=data.copy()
    world=pose_matrix((4.,-3.,.6,-.4))
    error=pose_matrix((.2,-.1,.06,-.05))
    transformed=(world @ error @ np.column_stack((source,np.ones(len(source)))).T).T[:,:3]
    data[:]=transformed
    mesh.GetPoints().Modified()
    locator=build_contact_index(mesh)
    model=np.linalg.inv(world)
    parameters=np.zeros(4)
    for _ in range(6):
        result=fit_probe_datum(locator,model,parameters,[(-3.,1.,0.,1.5),(3.,1.,0.,.9)],12.)
        parameters=result.parameters
    final=pose_matrix(parameters) @ error
    assert np.linalg.norm(final[:2,2])<1e-4
    assert np.array_equal(data,transformed.astype(data.dtype))
