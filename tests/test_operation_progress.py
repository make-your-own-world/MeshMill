import numpy as np
import pytest
from vtkmodules.vtkFiltersSources import vtkSphereSource

import meshmill as mm
import qem_progress


def sphere():
    source = vtkSphereSource()
    source.SetThetaResolution(60)
    source.SetPhiResolution(60)
    source.Update()
    return mm.polydata_arrays(source.GetOutput())


@pytest.mark.parametrize("algorithm", ["Fast QEM", "Density balanced"])
def test_progress_does_not_change_reduction(algorithm):
    if qem_progress._meshmill_qem is None:
        pytest.skip("Build native progress adapter first")
    points, faces = sphere()
    baseline = mm.simplify_arrays(points, faces, 1000, algorithm)
    events = []
    observed = mm.simplify_arrays(points, faces, 1000, algorithm,
                                 progress=lambda *event: events.append(event))
    assert all(np.array_equal(a, b) for a, b in zip(baseline, observed))
    assert events
    assert any(stage == "Scanning QEM triangles" for stage, _, _ in events)
    assert all(completed >= 0 and total >= completed for _, completed, total in events)


def test_atomic_progress_round_trip(tmp_path):
    import json
    path = tmp_path / "progress.json"
    mm._write_process_progress(str(path), "Reading mesh", 1024, 4096)
    assert json.loads(path.read_text()) == {
        "stage": "Reading mesh", "completed": 1024, "total": 4096,
    }
    assert not path.with_suffix(".json.pending").exists()
