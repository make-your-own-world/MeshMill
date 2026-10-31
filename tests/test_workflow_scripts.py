import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest
from vtkmodules.vtkFiltersSources import vtkSphereSource

import meshmill as mm
from mesh_formats import load_geometry
from mesh_workflow import load_workflow, run_workflow
from solidify_metrics import input_metrics


def source_mesh(tmp_path):
    source = vtkSphereSource()
    source.SetRadius(5)
    source.SetThetaResolution(64)
    source.SetPhiResolution(48)
    source.Update()
    path = tmp_path / "mesh's $input.stl"
    mm.save_stl(source.GetOutput(), path)
    return path


def test_generated_powershell_runs_multiple_passes_without_gui(tmp_path):
    powershell = shutil.which('powershell') or shutil.which('pwsh')
    if not powershell:
        pytest.skip('PowerShell is unavailable')
    source = source_mesh(tmp_path)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    steps = [{'algorithm': 'fast', 'quality': 'draft', 'processing': 'full'},
             {'algorithm': 'density', 'quality': 'balanced', 'processing': 'full'}]
    script = tmp_path / "workflow's script.ps1"
    script.write_text(mm.render_optimization_script(source, steps), encoding='utf-8-sig')
    output = tmp_path / 'result.dae'
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(script),
                             '-OutputMesh', str(output)], capture_output=True, text=True, check=False, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    _, points, faces = load_geometry(output)
    assert 0 < len(faces) < 5888
    assert input_metrics(points, faces)['open_edges'] == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    # Existing outputs are rejected before invoking the application.
    before = output.read_bytes()
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(script),
                             '-OutputMesh', str(output)], capture_output=True, text=True, check=False, timeout=15)
    assert result.returncode != 0 and output.read_bytes() == before


def test_envelope_and_conversion_use_current_geometry(tmp_path):
    source = source_mesh(tmp_path)
    output = tmp_path / 'cast.glb'
    run_workflow(source, output, [{'kind': 'cast', 'mode': 'Envelope', 'offset_mm': .5},
                                  {'kind': 'convert'}], progress=lambda _: None)
    poly, points, faces = load_geometry(output)
    assert poly.GetBounds()[1] > 5.4
    assert input_metrics(points, faces)['open_edges'] == 0


def test_invalid_workflow_does_not_create_or_replace_output(tmp_path):
    source = source_mesh(tmp_path)
    output = tmp_path / 'result.stl'
    with pytest.raises(ValueError, match='Unknown workflow'):
        run_workflow(source, output, [{'kind': 'unknown'}])
    assert not output.exists()
    with pytest.raises(ValueError, match='differ'):
        run_workflow(source, source, [{'kind': 'convert'}], overwrite=True)
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps({'version': 1, 'steps': [{'kind': 'convert'}]}))
    assert load_workflow(path) == [{'kind': 'convert'}]


def test_passes_keep_full_precision_between_operations(tmp_path, monkeypatch):
    source = source_mesh(tmp_path)
    expected = []

    def reduce(points, faces, *args, **kwargs):
        if expected:
            assert np.array_equal(points, expected[-1])
        changed = points.astype(float) + 1.234567891234e-7
        expected.append(changed.copy())
        return changed, faces

    monkeypatch.setattr(mm, 'simplify_arrays', reduce)
    run_workflow(source, tmp_path / 'result.ply', [{'algorithm': 'fast'}, {'algorithm': 'fast'}],
                 progress=lambda _: None)
    assert len(expected) == 2


@pytest.mark.parametrize('shape', ['Blob', 'Box'])
def test_split_clamshell_is_closed_and_keeps_two_parts(tmp_path, monkeypatch, shape):
    from vtkmodules.vtkFiltersCore import vtkPolyDataConnectivityFilter
    source = source_mesh(tmp_path)
    monkeypatch.setattr(mm, 'available_physical_memory', lambda: 256 * 1024**2)
    output = tmp_path / 'clamshell.ply'
    run_workflow(source, output, [{'kind': 'cast', 'shape': shape, 'wall_mm': 1,
                                  'bridge_mm': 0, 'offset_mm': .5, 'detail': .5,
                                  'split_normal': [.3, .7, 1], 'split_origin': [0, 0, 0],
                                  'separate_halves': True}], progress=lambda _: None)
    poly, points, faces = load_geometry(output)
    assert input_metrics(points, faces)['open_edges'] == 0
    connected = vtkPolyDataConnectivityFilter()
    connected.SetInputData(poly)
    connected.SetExtractionModeToAllRegions()
    connected.Update()
    assert connected.GetNumberOfExtractedRegions() == 2


def test_solidify_then_optimize_uses_the_new_solid(tmp_path):
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
    from vtkmodules.vtkFiltersSources import vtkCubeSource
    cube = vtkCubeSource()
    cube.SetBounds(-5, 5, -5, 5, -5, 5)
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(cube.GetOutputPort())
    triangles.Update()
    points, faces = mm.polydata_arrays(triangles.GetOutput())
    faces = faces[points[faces].mean(axis=1)[:, 2] < 4.9]
    source = tmp_path / 'open.stl'
    mm.save_stl(mm.arrays_polydata(points, faces), source)
    output = tmp_path / 'solid.ply'
    steps = [{'kind': 'solidify', 'quality': 'draft', 'gap_mm': 2},
             {'algorithm': 'fast', 'quality': 'fine'}]
    powershell = shutil.which('powershell') or shutil.which('pwsh')
    if powershell:
        script = tmp_path / 'solid.ps1'
        script.write_text(mm.render_optimization_script(source, steps), encoding='utf-8-sig')
        result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-File', str(script),
                                 '-OutputMesh', str(output)], capture_output=True, text=True,
                                 timeout=90, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        run_workflow(source, output, steps, progress=lambda _: None)
    _, points, faces = load_geometry(output)
    assert input_metrics(points, faces)['open_edges'] == 0


def test_tiled_script_route_uses_streamed_reduction(tmp_path, monkeypatch):
    source = source_mesh(tmp_path)
    monkeypatch.setattr(mm, 'available_physical_memory', lambda: 256 * 1024**2)
    output = tmp_path / 'tiled.stl'
    run_workflow(source, output, [{'algorithm': 'fast', 'quality': 'draft',
                                  'processing': 'tiled', 'workers': 1}], progress=lambda _: None)
    _, _, faces = mm.load_stl(output)
    assert 0 < len(faces) < 5888


@pytest.mark.parametrize('algorithm', ['shape', 'topology', 'depth-adaptive', 'depth-density-adaptive'])
def test_algorithm_and_pinned_device_are_forwarded(tmp_path, monkeypatch, algorithm):
    source = source_mesh(tmp_path)
    calls = []
    monkeypatch.setattr(mm, 'gpu_backend_supported', lambda _: True)

    def reduce(points, faces, target, label, aggressiveness, gpu, **kwargs):
        calls.append((label, gpu, kwargs['preserve_closed']))
        return points, faces

    monkeypatch.setattr(mm, 'simplify_arrays', reduce)
    run_workflow(source, tmp_path / 'result.stl', [{'algorithm': algorithm, 'gpu': 'selected-device',
                                                 'preserve_closed': True}], progress=lambda _: None)
    assert calls == [(mm.CLI_ALGORITHMS[algorithm], 'selected-device', True)]


def test_standalone_cli_writes_the_requested_interchange_format(tmp_path):
    source = source_mesh(tmp_path)
    output = tmp_path / 'reduced.ply'
    assert mm.cli_main([str(source), '--algorithm', 'fast', '--quality', 'draft',
                        '--output', str(output)]) == 0
    assert output.read_bytes().startswith(b'ply\n')
    _, _, faces = load_geometry(output)
    assert 0 < len(faces) < 5888


@pytest.mark.parametrize('step', [{'quality': 'unknown'}, {'memory_percent': 0},
                                  {'kind': 'cast', 'split_normal': [0, 0, 0]},
                                  {'kind': 'cast', 'split_origin': [1, 2]},
                                  {'kind': 'solidify', 'gap_mm': -1}])
def test_bad_settings_fail_before_any_output(tmp_path, step):
    source = source_mesh(tmp_path)
    output = tmp_path / 'output.stl'
    with pytest.raises(ValueError):
        run_workflow(source, output, [step])
    assert not output.exists()

