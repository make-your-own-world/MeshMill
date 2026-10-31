"""Headless replay of committed mesh operations using the application's engines."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def run_workflow(source, output, steps, *, overwrite=False, progress=print):
    import meshmill as mm
    from closed_reduction import is_closed_mesh
    from solidify import needs_solidification, offset_solid, solidify_mesh

    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source:
        raise ValueError('Output must differ from input.')
    if output.exists() and not overwrite:
        raise ValueError('Output already exists.')
    if output.suffix.lower() not in {'.stl', '.dae', '.obj', '.ply', '.glb'}:
        raise ValueError('Choose STL, COLLADA, OBJ, PLY or GLB output.')
    if not isinstance(steps, list) or not steps:
        raise ValueError('Workflow requires at least one operation.')
    allowed = {'kind', 'algorithm', 'quality', 'aggressiveness', 'gpu', 'processing', 'workers',
               'auto_tiling', 'memory_percent', 'preserve_closed', 'method', 'gap_mm', 'offset_mm',
               'mode', 'shape', 'wall_mm', 'bridge_mm', 'detail', 'split_normal', 'split_origin',
               'prepare_solid', 'solidify_quality', 'separate_halves', 'merge', 'name', 'target', 'export_transform'}
    for step in steps:
        if not isinstance(step, dict) or set(step) - allowed:
            raise ValueError('Workflow contains unknown settings.')
        if step.get('kind', 'optimize') not in {'optimize', 'solidify', 'cast', 'convert'}:
            raise ValueError('Unknown workflow operation.')
        if 'export_transform' in step:
            from mesh_export import rigid_export_matrix
            rigid_export_matrix(step['export_transform'])
            if step is not steps[-1]:
                raise ValueError('Export calibration belongs on the final workflow step.')
        for key in ('aggressiveness', 'gap_mm', 'offset_mm', 'wall_mm', 'bridge_mm', 'detail', 'memory_percent'):
            if key in step and not np.isfinite(float(step[key])):
                raise ValueError(f'{key} must be finite.')
        if not 0 < float(step.get('memory_percent', 6)) <= 95:
            raise ValueError('Memory percent must be above zero and at most 95.')
        if step.get('method', 'Surface repair') != 'Surface repair':
            raise ValueError('Solidify method must be Surface repair.')
        for key in ('quality', 'solidify_quality'):
            if key in step and str(step[key]).lower() not in {'draft', 'balanced', 'fine'}:
                raise ValueError(f'{key} must be draft, balanced or fine.')
        for key in ('gap_mm', 'offset_mm', 'wall_mm', 'bridge_mm'):
            if key in step and float(step[key]) < 0:
                raise ValueError(f'{key} must be nonnegative.')
        if float(step.get('detail', 1)) <= 0:
            raise ValueError('Detail must be positive.')
        for key in ('split_normal', 'split_origin'):
            if step.get(key) is not None:
                vector = np.asarray(step[key], dtype=float)
                if vector.shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError(f'{key} requires three finite coordinates.')
                if key == 'split_normal' and np.linalg.norm(vector) < 1e-12:
                    raise ValueError('Split normal must be nonzero.')
    output.parent.mkdir(parents=True, exist_ok=True)
    closed = False
    cached = None
    last_progress = [None, -1]

    def report(stage, done, total):
        percent = min(100, max(0, round(100 * done / max(1, total))))
        if stage != last_progress[0] or percent >= last_progress[1] + 5 or done == total:
            progress(f'{stage}: {percent}%')
            last_progress[:] = [stage, percent]
    with tempfile.TemporaryDirectory(prefix='meshmill-workflow-') as temporary:
        current = source
        for index, step in enumerate(steps):
            kind = step.get('kind', 'optimize')
            gpu = str(step.get('gpu', 'system'))
            budget = max(1024**2, int(mm.available_physical_memory() * float(step.get('memory_percent', 6)) / 100))
            progress(f'{index + 1}/{len(steps)}: {kind}')
            if kind == 'optimize':
                algorithm_key = step.get('algorithm', 'auto')
                if algorithm_key != 'auto' and algorithm_key not in mm.CLI_ALGORITHMS:
                    raise ValueError(f'Unknown reduction algorithm: {algorithm_key}')
                algorithm = mm.default_algorithm(gpu) if algorithm_key == 'auto' else mm.CLI_ALGORITHMS[algorithm_key]
                if mm.is_gpu_guided_algorithm(algorithm) and not mm.gpu_backend_supported(gpu):
                    raise ValueError('The selected GPU-guided algorithm requires a compatible CUDA GPU and runtime.')
                route = step.get('processing', 'auto')
                if route not in {'auto', 'full', 'tiled'}:
                    raise ValueError('Processing must be auto, full or tiled.')
                workers = int(step.get('workers', 0))
                if workers < 0:
                    raise ValueError('Worker count must be zero (automatic) or positive.')
                if cached is not None and route != 'tiled':
                    poly, points, faces = cached
                    metadata = {'total_triangles': len(faces), 'overview': False}
                else:
                    poly, points, faces, metadata = mm.load_stl_smart(
                        current, route, budget, 262144, None, workers or None, report,
                        algorithm, bool(step.get('auto_tiling', True)))
                quality = str(step.get('quality', 'balanced')).title()
                if quality not in {'Draft', 'Balanced', 'Fine'}:
                    raise ValueError('Quality must be draft, balanced or fine.')
                count = metadata['total_triangles']
                # A quality preset adapts separately to each pass's current input.
                target = min(mm.smart_target(points, faces, quality), max(4, len(faces) - 1))
                if metadata.get('overview'):
                    target = round(target / len(faces) * count)
                if 'target' in step:
                    target = int(step['target'])
                    if not 4 <= target < count:
                        raise ValueError('Target must be at least four and smaller than the input.')
                aggressiveness = float(step.get('aggressiveness', 7))
                if not 0 <= aggressiveness <= 10:
                    raise ValueError('Aggressiveness must be between 0 and 10.')
                if metadata.get('overview'):
                    if closed or step.get('preserve_closed'):
                        raise ValueError('Closed-solid reduction requires Full mesh processing.')
                    from out_of_core import optimize_indexed_mesh
                    tiled_output = Path(temporary) / f'tiled-{index}.stl'
                    optimize_indexed_mesh(Path(metadata['index_directory']), tiled_output, target,
                                          algorithm=algorithm, aggressiveness=aggressiveness,
                                          gpu_preference=gpu, workers=workers or None,
                                          memory_budget_bytes=budget, progress_callback=report)
                    poly, points, faces = mm.load_stl(tiled_output)
                else:
                    points, faces = mm.simplify_arrays(points, faces, target, algorithm, aggressiveness, gpu,
                                                       progress=report, preserve_closed=closed or bool(step.get('preserve_closed')))
                    poly = mm.arrays_polydata(points, faces)
            else:
                poly, points, faces = cached if cached is not None else mm.load_stl(current)
                input_bounds = poly.GetBounds()
                if kind == 'solidify':
                    quality = str(step.get('quality', 'balanced')).title()
                    pitch = mm.MainWindow._solidify_pitch(poly, quality)
                    poly, _ = solidify_mesh(poly, step.get('method', 'Surface repair'),
                                           float(step.get('gap_mm', 0)), pitch, budget,
                                           progress=report, gpu_preference=gpu)
                    closed = True
                elif kind == 'cast':
                    mode = step.get('mode', 'Clamshell')
                    if mode not in {'Clamshell', 'Envelope'}:
                        raise ValueError('Cast mode must be Clamshell or Envelope.')
                    if needs_solidification(poly) and (mode == 'Clamshell' or step.get('prepare_solid')):
                        if not step.get('prepare_solid'):
                            raise ValueError('The scan is open. Enable Prepare with Solidify.')
                        pitch = mm.MainWindow._solidify_pitch(poly, str(step.get('solidify_quality', 'balanced')).title())
                        poly, _ = solidify_mesh(poly, step.get('method', 'Surface repair'),
                                               float(step.get('gap_mm', 0)), pitch, budget,
                                               progress=report, gpu_preference=gpu)
                    offset = float(step.get('offset_mm', .5))
                    if needs_solidification(poly):
                        from exposed_surface import exposed_surface
                        poly, _ = exposed_surface(poly, offset, progress=report, gpu_preference=gpu)
                    else:
                        poly, _ = offset_solid(poly, offset)
                    if mode == 'Clamshell':
                        from vtkmodules.vtkFiltersCore import vtkAppendPolyData

                        from clamshell_mold import (
                            clamshell_mold,
                            mold_sampling_plan,
                            separated_mold_halves,
                        )
                        bounds = np.asarray(input_bounds).reshape(3, 2)
                        bounds += np.array([-1, 1]) * offset
                        plan = mold_sampling_plan(bounds, step.get('wall_mm', 0), step.get('bridge_mm', 0),
                                                  memory_budget_bytes=budget, detail=float(step.get('detail', 1)))
                        halves, _ = clamshell_mold(poly, plan['wall_mm'], plan['bridge_mm'], plan['pitch_mm'],
                                                  step.get('split_normal'), step.get('shape', 'Blob'),
                                                  progress=report, memory_budget_bytes=budget,
                                                  split_origin=step.get('split_origin'))
                        if step.get('separate_halves'):
                            halves = separated_mold_halves(halves)
                        append = vtkAppendPolyData()
                        for half in halves:
                            append.AddInputData(half)
                        append.Update()
                        poly = append.GetOutput()
                    points, faces = mm.polydata_arrays(poly)
                    closed = is_closed_mesh(points, faces)
                if kind == 'solidify':
                    points, faces = mm.polydata_arrays(poly)
            cached = (poly, points, faces)
            if index + 1 < len(steps) and steps[index + 1].get('processing') == 'tiled':
                current = Path(temporary) / f'pass-{index}.stl'
                mm.save_stl(poly, current)
        # Publish the final file only after every operation succeeds.
        if steps[-1].get('export_transform') is not None:
            from mesh_export import calibrated_export_mesh
            poly = calibrated_export_mesh(poly, steps[-1]['export_transform'])
            points, faces = mm.polydata_arrays(poly)
        if output.suffix.lower() == '.stl':
            pending = Path(temporary) / 'result.stl'
            mm.save_stl(poly, pending)
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
                import shutil
                staging = Path(stream.name)
                try:
                    with pending.open('rb') as source_stream:
                        shutil.copyfileobj(source_stream, stream)
                except BaseException:
                    stream.close()
                    staging.unlink(missing_ok=True)
                    raise
            try:
                os.replace(staging, output)
            finally:
                staging.unlink(missing_ok=True)
        else:
            from mesh_export import write_polygon_mesh
            write_polygon_mesh(points, faces, output, merge=bool(steps[-1].get('merge', True)))
    progress(f'Saved: {output}')


def load_workflow(path):
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict) or document.get('version') != 1 or set(document) != {'version', 'steps'}:
        raise ValueError('Expected a version 1 MeshMill workflow with steps.')
    return document['steps']
