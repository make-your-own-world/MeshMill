# -*- mode: python ; coding: utf-8 -*-
import os
import subprocess
import sys
from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_dynamic_libs

# Build the observational adapter from the installed reduction core before freezing.
subprocess.run([sys.executable, 'tools/build_qem_progress.py'], check=True)

a = Analysis(
    ['meshmill.py'],
    pathex=[],
    binaries=collect_dynamic_libs('warp'),
    datas=[
        ('LICENSE', '.'),
        ('SOURCES.md', '.'),
        ('RESEARCH_ACKNOWLEDGEMENTS.md', '.'),
        ('licenses', 'licenses'),
        ('README.md', '.'),
        ('CHANGELOG.md', '.'),
        ('ROADMAP.md', '.'),
        ('THIRD_PARTY_NOTICES.md', '.'),
        ('CUDA_EXCEPTION.md', '.'),
        ('docs/SOLIDIFY_AND_CAST.md', 'docs'),
        ('docs/SOLIDIFY_CAST_MATH.md', 'docs'),
        ('docs/images/meshmill-cast-workspace.png', 'docs/images'),
        ('docs/images/meshmill-solidify-before.png', 'docs/images'),
        ('docs/images/meshmill-solidify-after.png', 'docs/images'),
        ('docs/images/meshmill-processing-progress.png', 'docs/images'),
        ('docs/images/meshmill-cast-controls.png', 'docs/images'),
        ('docs/images/meshmill-export-comparison.png', 'docs/images'),
        ('docs/images/sketchup-import.png', 'docs/images'),
        ('docs/images/cura-print-bed.png', 'docs/images'),
        ('docs/EXPORT_FORMATS.md', 'docs'),
        ('docs/AUTOMATION.md', 'docs'),
        ('docs/AXIAL_CALIBRATION.md', 'docs'),
        ('surface_distance_gpu.py', '.'),
        ('docs/OUT_OF_CORE.md', 'docs'),
        ('docs/GPU_OUT_OF_CORE.md', 'docs'),
        ('benchmarks/V0_3_1_PERFORMANCE.md', 'benchmarks'),
        ('docs/images/meshmill-shaded.png', 'docs/images'),
        ('docs/images/meshmill-vertices.png', 'docs/images'),
        ('docs/images/meshmill-wireframe.png', 'docs/images'),
        ('docs/images/meshmill-density.png', 'docs/images'),
        ('docs/images/meshmill-density-overview.png', 'docs/images'),
        ('docs/images/meshmill-density-zoom.png', 'docs/images'),
        ('docs/images/meshmill-crop-selection.png', 'docs/images'),
        ('docs/images/meshmill-connected-selection.png', 'docs/images'),
        ('assets/meshmill-mark.svg', 'assets'),
        ('assets/meshmill-logo.svg', 'assets'),
        ('assets/support-profile.webp', 'assets'),
        ('locales/en-US.json', 'locales'),
    ] + sum((copy_metadata(package) for package in ('pymeshfix', 'tetgen', 'PyMaxflow', 'python-sat', 'six', 'warp-lang')), []) + collect_data_files('warp', include_py_files=True, excludes=['tests/**', 'examples/**', 'benchmarks/**']),
    hiddenimports=[
        'unittest.mock',
        'pysat.solvers',
        'pysat.card',
        'pysat.examples.rc2',
        'vtkmodules.vtkInteractionStyle',
        'vtkmodules.vtkRenderingOpenGL2',
        'vtkmodules.vtkRenderingFreeType',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'pytest',
        'scipy',
        'cupy',
        'cupyx',
        'cupy_backends',
        'nvidia',
        'cupy_backends.cuda.libs.cudnn',
        'cupy_backends.cuda.libs.cutensor',
    ],
    noarchive=False,
    optimize=1,
)

# Qt uses the Windows ICU compatibility libraries. A developer machine may also
# expose unrelated ICU builds through PATH (for example, from PDF tooling), and
# PyInstaller can otherwise collect those DLLs while resolving Qt6Core. Bundling
# them makes Qt load the incompatible copy before the Windows implementation.
_foreign_icu_dlls = {'icuuc.dll', 'icudt78.dll'}
a.binaries = [entry for entry in a.binaries if entry[0].lower() not in _foreign_icu_dlls]

pyz = PYZ(a.pure)
gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MeshMill',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    version=os.path.join(SPECPATH, 'version_info.txt'),
    icon=os.path.join(SPECPATH, 'assets', 'meshmill.ico'),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MeshMillCLI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    version=os.path.join(SPECPATH, 'version_info.txt'),
    icon=os.path.join(SPECPATH, 'assets', 'meshmill.ico'),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    gui,
    cli,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MeshMill',
)
