import numpy as np
from vtkmodules.vtkFiltersSources import vtkSphereSource

from closed_reduction import is_closed_mesh, validate_closed_reduction
from meshmill import polydata_arrays, simplify_arrays


def sphere():
    shape = vtkSphereSource()
    shape.SetThetaResolution(16)
    shape.SetPhiResolution(16)
    shape.Update()
    return polydata_arrays(shape.GetOutput())


def test_closed_detection_respects_stl_welding():
    p, f = sphere()
    assert is_closed_mesh(p, f)
    assert not is_closed_mesh(p, f[:-1])
    assert is_closed_mesh(p[f].reshape(-1, 3), np.arange(f.size).reshape(-1, 3))


def test_reduced_solid_closes_missing_face_without_resampling():
    p, f = sphere()
    op, of = validate_closed_reduction(p, f[:-1])
    assert is_closed_mesh(op, of)
    assert len(of) == len(f)
    assert np.array_equal(np.sort(op, axis=0), np.sort(p, axis=0))


def test_closed_reduction_is_validated_before_return(monkeypatch):
    import meshmill

    p, f = sphere()
    monkeypatch.setattr(
        meshmill, "_simplify_arrays_backend", lambda *args: (p.copy(), f[:-1].copy())
    )
    op, of = simplify_arrays(p, f, len(f) // 2, "Fast QEM", preserve_closed=True)
    assert is_closed_mesh(op, of)
    assert np.array_equal(p, sphere()[0])


def test_intentionally_open_input_is_not_filled(monkeypatch):
    import meshmill

    p, f = sphere()
    f = f[:-1]
    monkeypatch.setattr(meshmill, "_simplify_arrays_backend", lambda *args: (p.copy(), f.copy()))
    op, of = simplify_arrays(p, f, len(f) // 2, "Fast QEM")
    assert np.array_equal(op, p)
    assert np.array_equal(of, f)
    assert not is_closed_mesh(op, of)
