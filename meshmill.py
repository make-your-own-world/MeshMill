from __future__ import annotations

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 MeshMill contributors
import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from bisect import bisect_left
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
from PySide6.QtCore import (
    QEvent,
    QKeyCombination,
    QLocale,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSettings,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QDesktopServices,
    QFontMetrics,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
    QRegion,
    QShortcut,
    QTextCharFormat,
    QTextCursor,
    QTextTable,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QBoxLayout,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTextBrowser,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticPointLocator
from vtkmodules.vtkFiltersCore import (
    vtkDecimatePro,
    vtkPolyDataConnectivityFilter,
    vtkPolyDataNormals,
    vtkQuadricDecimation,
    vtkTriangleFilter,
)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkSTLWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkCoordinate,
    vtkHardwarePicker,
    vtkPolyDataMapper,
    vtkPolyDataMapper2D,
    vtkRenderer,
    vtkWindowToImageFilter,
)

from depth_adaptive_qem import (
    cuda_devices,
    first_time_gpu_setup_needed,
    gpu_backend_supported,
    prepare_gpu_backend,
    probe_source_statistics,
    simplify_depth_adaptive_qem,
    simplify_depth_density_adaptive_qem,
)
from gpu_resident import inspect_resident_vertices
from metrics_history import history_position, history_span, summarize_history
from out_of_core import (
    build_index,
    default_index_directory,
    index_matches,
    load_manifest,
    optimize_indexed_mesh,
)
from qem_progress import simplify as simplify_with_progress
from worker_lifecycle import OwnedProcessPool, recover_orphan_workers

# Conservative allowance for source arrays, VTK objects, render data, and temporary work buffers.
ESTIMATED_BYTES_PER_TRIANGLE = 220

from localization import DEFAULT_LOCALE, locale_manager, tr
from meshmill_version import APP_VERSION

APP_NAME = "MeshMill"
REPOSITORY_URL = "https://github.com/make-your-own-world/MeshMill"
FEATURE_REQUEST_URL = f"{REPOSITORY_URL}/issues/new?labels=enhancement&title=%5BFeature%5D+"
LATEST_RELEASE_API_URL = "https://api.github.com/repos/make-your-own-world/MeshMill/releases/latest"
SUPPORT_URL = "https://buymeacoffee.com/tednv"
PRESETS = {"light": 50_000, "balanced": 200_000, "detailed": 400_000}
UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}
SMART_DIVISORS = {"Draft": 160.0, "Balanced": 500.0, "Fine": 800.0}
DEPTH_ADAPTIVE_LABEL = "Depth Adaptive QEM (GPU / Faster)"
DEPTH_DENSITY_ADAPTIVE_LABEL = "Depth + Density Adaptive QEM (GPU / Best)"
LEGACY_ALGORITHM_LABELS = {
    "Depth Adaptive QEM (GPU / Higher Quality)": DEPTH_ADAPTIVE_LABEL,
    "Depth + Density Adaptive QEM (GPU / Highest Quality)": DEPTH_DENSITY_ADAPTIVE_LABEL,
}
ALGORITHMS = [
    DEPTH_DENSITY_ADAPTIVE_LABEL,
    DEPTH_ADAPTIVE_LABEL,
    "Density balanced",
    "Fast QEM",
    "Shape preserving",
    "Preserve topology",
]
DISPLAY_MODES = ["Shaded", "Density", "Wireframe", "Vertices"]
VIEWPORT_OVERLAYS = {
    "orientation": ("Orientation", "Top left"),
    "density": ("Point density", "Top right"),
    "scale": ("Scale", "Bottom right"),
    "views": ("View controls", "Bottom left"),
}
VIEWPORT_CORNERS = ("Top left", "Top right", "Bottom left", "Bottom right")


def build_bug_report_url(title: str, details: str, diagnostics: str | None = None) -> str:
    """Build a GitHub issue draft URL from user-reviewed bug report content."""
    body = "\n\n".join(
        (
            "## What happened\n" + details.strip(),
            "## Expected result\n",
            "## Steps to reproduce\n",
        )
    )
    if diagnostics and diagnostics.strip():
        body += "\n\n## Anonymous diagnostics\n" + diagnostics.strip()
    query = urlencode({"labels": "bug", "title": title.strip() or "Bug report", "body": body})
    return f"{REPOSITORY_URL}/issues/new?{query}"


def is_gpu_guided_algorithm(algorithm: str) -> bool:
    return algorithm in (DEPTH_ADAPTIVE_LABEL, DEPTH_DENSITY_ADAPTIVE_LABEL)


def default_algorithm(preference: str = "system") -> str:
    """Choose the highest-quality available production algorithm."""
    if gpu_backend_supported(preference):
        return DEPTH_DENSITY_ADAPTIVE_LABEL
    return "Density balanced"


def resolve_startup_algorithm(
    startup_choice: str,
    last_used: str,
    gpu_preference: str = "system",
) -> str:
    """Resolve a saved startup choice without selecting an unavailable GPU method."""
    if startup_choice == "last":
        candidate = last_used
    elif startup_choice == "auto":
        candidate = default_algorithm(gpu_preference)
    else:
        candidate = startup_choice
    candidate = LEGACY_ALGORITHM_LABELS.get(candidate, candidate)
    if candidate not in ALGORITHMS:
        candidate = default_algorithm(gpu_preference)
    if is_gpu_guided_algorithm(candidate) and not gpu_backend_supported(gpu_preference):
        return "Density balanced"
    return candidate


STANDARD_VIEWS = {
    "top": ((0, 0, 1), (0, 1, 0)),
    "bottom": ((0, 0, -1), (0, 1, 0)),
    "front": ((0, 1, 0), (0, 0, 1)),
    "back": ((0, -1, 0), (0, 0, 1)),
    "left": ((1, 0, 0), (0, 0, 1)),
    "right": ((-1, 0, 0), (0, 0, 1)),
}
OPPOSITE_VIEWS = {
    "left": "right",
    "right": "left",
    "front": "back",
    "back": "front",
    "top": "bottom",
    "bottom": "top",
}
SHORTCUT_DEFINITIONS = {
    "save": ("Save", "Ctrl+S"),
    "undo": ("Undo", "Ctrl+Z"),
    "redo": ("Redo", "Ctrl+Y"),
    "redo_alt": ("Redo (alternate)", "Ctrl+Shift+Z"),
    "selection_optimize": ("Optimize selection", "Ctrl+Space"),
    "selection_crop": ("Crop selection", "Ctrl+X"),
    "selection_add": ("Add selection", "Ctrl+C"),
    "selection_delete": ("Delete selection", "Delete"),
    "selection_clear": ("Clear selection", "Esc"),
    "display_shaded": ("Display: Shaded", "F1"),
    "display_density": ("Display: Density", "F2"),
    "display_wireframe": ("Display: Wireframe", "F3"),
    "display_vertices": ("Display: Vertices", "F4"),
    "view_left": ("View: Left", "Insert"),
    "view_front": ("View: Front", "Home"),
    "view_right": ("View: Right", "PgUp"),
    "view_top": ("View: Top", "Delete"),
    "view_back": ("View: Back", "End"),
    "view_bottom": ("View: Bottom", "PgDown"),
    "set_view_left": ("Set view: Left", "Ctrl+Insert"),
    "set_view_front": ("Set view: Front", "Ctrl+Home"),
    "set_view_right": ("Set view: Right", "Ctrl+PgUp"),
    "set_view_top": ("Set view: Top", "Ctrl+Delete"),
    "set_view_back": ("Set view: Back", "Ctrl+End"),
    "set_view_bottom": ("Set view: Bottom", "Ctrl+PgDown"),
    "orbit_left": ("Orbit left", "Left"),
    "orbit_right": ("Orbit right", "Right"),
    "orbit_up": ("Orbit up", "Up"),
    "orbit_down": ("Orbit down", "Down"),
    "pan_left": ("Pan left", "Ctrl+Left"),
    "pan_right": ("Pan right", "Ctrl+Right"),
    "pan_up": ("Pan up", "Ctrl+Up"),
    "pan_down": ("Pan down", "Ctrl+Down"),
    "zoom_in": ("Zoom in", "Ctrl+Shift+Up"),
    "zoom_out": ("Zoom out", "Ctrl+Shift+Down"),
    "zoom_in_alt": ("Zoom in (alternate)", "+"),
    "zoom_out_alt": ("Zoom out (alternate)", "-"),
    "toggle_fullscreen": ("Toggle full screen", "F11"),
    "roll_counterclockwise": ("Roll counterclockwise", "Ctrl+Shift+Left"),
    "roll_clockwise": ("Roll clockwise", "Ctrl+Shift+Right"),
}
CLI_ALGORITHMS = {
    "fast": "Fast QEM",
    "density": "Density balanced",
    "shape": "Shape preserving",
    "topology": "Preserve topology",
    "depth-adaptive": DEPTH_ADAPTIVE_LABEL,
    "depth-density-adaptive": DEPTH_DENSITY_ADAPTIVE_LABEL,
}
CLI_ALGORITHM_KEYS = {label: key for key, label in CLI_ALGORITHMS.items()}
warnings.filterwarnings("ignore", message="Call to deprecated method.*", category=DeprecationWarning)


def _powershell_literal(value: str | Path) -> str:
    """Return a PowerShell single-quoted literal without interpolation."""
    return "'" + os.fspath(value).replace("'", "''") + "'"


def render_optimization_script(source_path, passes, omitted_edits=()):
    """Render an editable, headless workflow with culture-independent settings."""
    steps = [dict(step) for step in passes] or [{"kind": "convert"}]
    for step in steps:
        step.pop("target", None)
        if step.get("kind", "optimize") == "optimize":
            step["quality"] = str(step.get("quality", "balanced")).lower()
    source = Path(source_path).resolve()
    cli_name = 'MeshMillCLI.exe' if os.name == 'nt' else 'MeshMillCLI'
    cli = Path(sys.executable).with_name(cli_name)
    prefix = []
    runtime_paths = []
    if not getattr(sys, "frozen", False):
        import importlib.util
        cli = Path(sys.executable)
        if cli.stem.lower() == 'pythonw':
            cli = cli.with_name('python.exe')
        prefix = [os.fspath(Path(__file__).resolve()), "--cli"]
        for package in ('pymeshfix', 'maxflow', 'tetgen', 'pysat', 'warp'):
            dependency = importlib.util.find_spec(package)
            if dependency is not None and dependency.submodule_search_locations:
                location = os.fspath(Path(next(iter(dependency.submodule_search_locations))).parent)
                if location not in runtime_paths:
                    runtime_paths.append(location)
    document = json.dumps({"version": 1, "steps": steps}, ensure_ascii=True, allow_nan=False)
    lines = [
        "[CmdletBinding()]", "param(",
        f"    [string]$InputMesh = {_powershell_literal(source)},",
        f"    [string]$OutputMesh = {_powershell_literal(source.with_name(source.stem + '_processed.stl'))},",
        f"    [string]$CliPath = {_powershell_literal(cli)}", ")", "",
        "$ErrorActionPreference = 'Stop'",
        "$InputMesh = [IO.Path]::GetFullPath($InputMesh)",
        "$OutputMesh = [IO.Path]::GetFullPath($OutputMesh)",
        "if (-not (Test-Path -LiteralPath $InputMesh -PathType Leaf)) { throw 'Input mesh not found.' }",
        "if ([String]::Equals($InputMesh, $OutputMesh, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must differ from input.' }",
        "if (Test-Path -LiteralPath $OutputMesh) { throw 'Output already exists.' }",
        "if (-not (Test-Path -LiteralPath $CliPath -PathType Leaf)) {",
        f"    $beside = Join-Path $PSScriptRoot {_powershell_literal(cli_name)}",
        "    if (Test-Path -LiteralPath $beside -PathType Leaf) { $CliPath = $beside }",
        f"    else {{ $CliPath = (Get-Command {_powershell_literal(cli_name)} -ErrorAction Stop).Source }}", "}",
        "$prefixArguments = @(" + ', '.join(_powershell_literal(x) for x in prefix) + ")",
        "if ([IO.Path]::GetFileName($CliPath) -in @('MeshMillCLI.exe', 'MeshMillCLI')) { $prefixArguments = @() }",
        f"$workflowJson = {_powershell_literal(document)}",
        "$runtimeSearchPaths = @(" + ', '.join(_powershell_literal(x) for x in runtime_paths) + ")",
        "$previousPythonPath = $env:PYTHONPATH",
        "$jobFile = Join-Path ([IO.Path]::GetTempPath()) ('meshmill-' + [Guid]::NewGuid().ToString('N') + '.json')",
        "try {",
        "    if ($prefixArguments.Count -gt 0 -and $runtimeSearchPaths.Count -gt 0) {",
        "        $env:PYTHONPATH = (@($runtimeSearchPaths) + @($previousPythonPath) | Where-Object { $_ }) -join [IO.Path]::PathSeparator",
        "    }",
        "    [IO.File]::WriteAllText($jobFile, $workflowJson, (New-Object Text.UTF8Encoding($false)))",
        "    $cliArguments = @($InputMesh, '--workflow', $jobFile, '--output', $OutputMesh)",
        "    & $CliPath @prefixArguments @cliArguments",
        "    if ($LASTEXITCODE -ne 0) { throw ('MeshMill failed, exit code ' + $LASTEXITCODE) }",
        "    if (-not (Test-Path -LiteralPath $OutputMesh -PathType Leaf)) { throw 'MeshMill did not create the output.' }",
        "} finally {", "    $env:PYTHONPATH = $previousPythonPath",
        "    Remove-Item -LiteralPath $jobFile -ErrorAction SilentlyContinue", "}", "",
    ]
    if omitted_edits:
        lines.insert(0, '# InputMesh is an edited-mesh checkpoint. Camera and selection gestures are not replayed.')
    return "\n".join(lines)


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


def localize_widget_tree(root: QWidget) -> None:
    """Apply the active catalog to static widget labels and tooltips."""
    def refresh(widget, field, current, setter):
        if not current:
            return
        source_key = f"i18n_{field}"
        rendered_key = f"i18n_rendered_{field}"
        source = widget.property(source_key)
        if source is None or current != widget.property(rendered_key):
            source = current
        rendered = tr(str(source))
        widget.setProperty(source_key, source)
        widget.setProperty(rendered_key, rendered)
        setter(rendered)

    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        refresh(widget, "window_title", widget.windowTitle(), widget.setWindowTitle)
        refresh(widget, "tooltip", widget.toolTip(), widget.setToolTip)
        if isinstance(widget, (QLabel, QPushButton, QCheckBox)):
            refresh(widget, "text", widget.text(), widget.setText)
    direction = locale_manager.manifest.get("languages", {}).get(
        locale_manager.locale, {}
    ).get("direction", "ltr")
    root.setLayoutDirection(
        Qt.LayoutDirection.RightToLeft if direction == "rtl" else Qt.LayoutDirection.LeftToRight
    )


DEFAULT_APPEARANCE = "MeshMill Blue"
TOOLBOX_ORDER = ("Actions", "Active tool", "Stats", "Other tools", "General")


APPEARANCE_THEMES = {
    "MeshMill Blue": {
        "window": "#F4F8FC", "panel": "#FFFFFF", "surface": "#FBFDFF",
        "field": "#FFFFFF", "border": "#D5E2EF", "text": "#102A43",
        "muted": "#61748A", "accent": "#2563EB", "accent_hover": "#1D4ED8", "accent_text": "#FFFFFF",
        "file_action": "#2563EB", "file_action_hover": "#1D4ED8", "file_action_text": "#FFFFFF",
        "accent_soft": "#EAF2FF", "disabled": "#EEF2F6", "support": "#D9F99D",
        "support_hover": "#C7EF7B", "support_border": "#B7D96A",
        "overlay": "rgba(248, 252, 255, 238)", "viewport": "#0B1B34", "overlay_text": "#102A43",
        "optimization": "#2563EB", "stats": "#0F9B6C", "general": "#2563EB",
        "optimization_soft": "#EDF4FF", "stats_soft": "#ECF9F3", "general_soft": "#FFFFFF",
    },
    "Light Blue": {
        "window": "#EEF9FE", "panel": "#FFFFFF", "surface": "#F8FDFF",
        "field": "#FFFFFF", "border": "#C9E6F1", "text": "#103A58",
        "muted": "#5D7E95", "accent": "#0F9FB4", "accent_hover": "#087E92", "accent_text": "#FFFFFF",
        "file_action": "#1673C8", "file_action_hover": "#105FA7", "file_action_text": "#FFFFFF",
        "accent_soft": "#E1F8FC", "disabled": "#EDF4F6", "support": "#D9F99D",
        "support_hover": "#C7EF7B", "support_border": "#AFCF71",
        "overlay": "rgba(245, 252, 255, 238)", "viewport": "#103B5A", "overlay_text": "#103A58",
        "optimization": "#1673C8", "stats": "#0A9A78", "general": "#B57B13",
        "optimization_soft": "#EAF5FE", "stats_soft": "#E9F9F4", "general_soft": "#FFF6E2",
    },
    "Dark Focus": {
        "window": "#0D1B26", "panel": "#142633", "surface": "#192F3E",
        "field": "#10212D", "border": "#2B485A", "text": "#EAF4FB",
        "muted": "#A5BBC9", "accent": "#16C784", "accent_hover": "#0CAD6F", "accent_text": "#FFFFFF",
        "file_action": "#2188F5", "file_action_hover": "#1477E1", "file_action_text": "#FFFFFF",
        "accent_soft": "#1A3B3F", "disabled": "#243744", "support": "#C8EC8D",
        "support_hover": "#B8DD78", "support_border": "#8EB65B",
        "overlay": "rgba(13, 27, 38, 238)", "viewport": "#0B1721", "overlay_text": "#EAF4FB",
        "optimization": "#49A2FF", "stats": "#20C98A", "general": "#49A2FF",
        "optimization_soft": "#173552", "stats_soft": "#173F38", "general_soft": "#192F3E",
    },
    "Earth Tone": {
        "window": "#F5F3EC", "panel": "#FCFBF6", "surface": "#FFFDF8",
        "field": "#FFFFFF", "border": "#DDD8C8", "text": "#263D2E",
        "muted": "#687568", "accent": "#628B66", "accent_hover": "#4D7352", "accent_text": "#FFFFFF",
        "file_action": "#287CC2", "file_action_hover": "#1D659F", "file_action_text": "#FFFFFF",
        "accent_soft": "#EDF5E9", "disabled": "#F0EEE6", "support": "#DCEFA6",
        "support_hover": "#CBE38A", "support_border": "#ABC66A",
        "overlay": "rgba(252, 251, 246, 238)", "viewport": "#314A3A", "overlay_text": "#F9F7EF",
        "optimization": "#287CC2", "stats": "#3E8B68", "general": "#B47C1D",
        "optimization_soft": "#EAF3FA", "stats_soft": "#EDF6EF", "general_soft": "#FBF3E1",
    },
}


def inactive_caption_color(theme: dict[str, str]) -> str:
    base, border = QColor(theme["window"]), QColor(theme["border"])
    return QColor(*(round(.8 * a + .2 * b) for a, b in zip(
        base.getRgb()[:3], border.getRgb()[:3]
    ))).name()


def appearance_theme(name: str) -> dict[str, str]:
    return APPEARANCE_THEMES.get(name, APPEARANCE_THEMES[DEFAULT_APPEARANCE])


def selected_control_text(theme: dict[str, str]) -> str:
    color = QColor(theme["accent"])
    channels = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4
                for v in (color.redF(), color.greenF(), color.blueF())]
    luminance = sum(c * w for c, w in zip(channels, (.2126, .7152, .0722)))
    return "#FFFFFF" if luminance < .179 else "#000000"


def scrollbar_stylesheet(theme: dict[str, str]) -> str:
    return "\n".join((
        f"QScrollBar {{ background: {theme['surface']}; margin: 2px; border: 1px solid {theme['border']}; border-radius: 8px; }}",
        "QScrollBar:vertical { width: 22px; } QScrollBar:horizontal { height: 22px; }",
        f"QScrollBar::handle {{ background: {theme['muted']}; margin: 2px; border-radius: 6px; }}",
        "QScrollBar::handle:vertical { min-height: 52px; } QScrollBar::handle:horizontal { min-width: 52px; }",
        f"QScrollBar::handle:hover, QScrollBar::handle:pressed {{ background: {theme['accent']}; }}",
        f"QScrollBar::handle:disabled {{ background: {theme['border']}; }}",
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }",
        "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }",
        "QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }",
    ))


def app_stylesheet(name: str = DEFAULT_APPEARANCE, font_size: int = 12) -> str:
    theme = appearance_theme(name)
    compact_font_size = max(8, font_size - 3)
    return "\n".join(
        (
            f"QMainWindow, QWidget#mainRoot, QDialog {{ background: {theme['window']}; }}",
            f"QFrame#appHeader {{ background: {theme['window']}; border-bottom: 1px solid {theme['border']}; }}",
            f"QFrame#appHeader[windowActive='false'] {{background:{inactive_caption_color(theme)};}}",
            f"QFrame#appHeader[windowActive='false'] QLabel {{color:{theme['muted']};}}",
            f"QScrollArea#toolboxScroll, QWidget#controlsContent {{ background: {theme['window']}; }}",
            scrollbar_stylesheet(theme),
            f"QWidget {{ color: {theme['text']}; font-family: 'Segoe UI Variable Text', 'Segoe UI'; font-size: {font_size}px; }}",
            f"QMenu {{ background: {theme['panel']}; color: {theme['text']}; border: 1px solid {theme['border']}; padding: 4px; }}",
            "QMenu::item { padding: 6px 12px; border-radius: 4px; }",
            f"QMenu::item:selected {{ background: {theme['accent']}; color: {selected_control_text(theme)}; }}",
            f"QMenu::item:disabled {{ color: {theme['muted']}; }}",
            f"QMenu::separator {{ height: 1px; background: {theme['border']}; margin: 4px 6px; }}",
            f"QFrame#toolSection, QFrame#settingsSection, QFrame#metricsPanel {{ background: {theme['panel']}; border: 1px solid {theme['border']}; border-radius: 12px; }}",
            f"QFrame#splitPlaneTools {{ background: {theme['panel']}; border: 1px solid {theme['border']}; border-radius: 8px; }}",
            f"QFrame#splitPlaneTools QLabel {{ color: {theme['text']}; background: transparent; border: none; }}",
            f"QFrame#toolSection[sectionTone='optimization'] {{ border-top: 3px solid {theme['optimization']}; }}",
            f"QFrame#toolSection[sectionTone='stats'] {{ border-top: 3px solid {theme['stats']}; }}",
            f"QFrame#toolSection[sectionTone='general'] {{ background: {theme['general_soft']}; border-top: 3px solid {theme['general']}; }}",
            f"QLabel#sectionTitle {{ color: {theme['text']}; font-size: {font_size}px; font-weight: 700; padding: 2px 0 6px; }}",
            f"QToolButton#sectionExpander, QToolButton#sectionExpander:checked {{ color: {theme['text']}; border: none; background: transparent; padding: 2px; border-radius: 4px; }}",
            f"QToolButton#sectionExpander:hover {{ background: {theme['optimization_soft']}; }}",
            f"QLabel#aboutTitle {{ color: {theme['text']}; font-size: {font_size + 6}px; font-weight: 700; }}",
            f"QLabel#loadingTitle {{ color: {theme['text']}; font-size: {font_size + 6}px; font-weight: 700; }}",
            f"QLabel#sectionIcon[iconTone='optimization'] {{ color: {theme['optimization']}; }}",
            f"QLabel#sectionIcon[iconTone='stats'] {{ color: {theme['stats']}; }}",
            f"QLabel#sectionIcon[iconTone='general'] {{ color: {theme['general']}; }}",
            f"QLabel[muted='true'] {{ color: {theme['muted']}; }}",
            f"QLabel#fileCard {{ background: {theme['field']}; border: 1px solid {theme['border']}; border-left: 3px solid {theme['accent']}; border-radius: 8px; padding: 7px; color: {theme['text']}; }}",
            f"QLabel[qualityMetric='true'] {{ color: {theme['text']}; background: {theme['field']}; border: 1px solid {theme['border']}; border-radius: 6px; padding: 3px 1px; font-size: {max(10, font_size - 1)}px; font-weight: 600; }}",
            "QLabel[qualityState='good'] { color: #087F53; background: #E8F8F0; border: 2px solid #16A36A; }",
            "QLabel[qualityState='caution'] { color: #8A5900; background: #FFF5D9; border: 2px solid #D18A00; }",
            "QLabel[qualityState='warning'] { color: #A62E2E; background: #FCEBEB; border: 2px solid #D14B4B; }",
            f"QLabel[loadingMetric='true'] {{ background: {theme['accent_soft']}; color: {theme['text']}; border: 1px solid {theme['border']}; border-radius: 5px; padding: 4px 3px; font-size: {compact_font_size}px; font-weight: 700; }}",
            f"QPushButton {{ background: {theme['field']}; border: 1px solid {theme['border']}; border-radius: 8px; padding: 6px 10px; min-height: 22px; }}",
            "QPushButton::icon { padding: 0; }",
            f"QPushButton:hover {{ border-color: {theme['accent']}; background: {theme['accent_soft']}; }}",
            f"QPushButton:pressed {{ background: {theme['accent_soft']}; }}",
            f"QPushButton:disabled {{ color: {theme['muted']}; background: {theme['disabled']}; border-color: {theme['border']}; }}",
            f"QPushButton[accent='true'] {{ color: {theme['accent_text']}; background: {theme['accent']}; border-color: {theme['accent']}; font-weight: 600; }}",
            f"QPushButton[accent='true']:hover {{ background: {theme['accent_hover']}; }}",
            f"QPushButton[accent='true']:pressed {{ background: {theme['accent_hover']}; border-color: {theme['text']}; }}",
            f"QPushButton[toolMode='true']:checked {{ color: {theme['accent_text']}; background: {theme['accent']}; border-color: {theme['accent']}; }}",
            f"QPushButton[toolMode='true']:checked:hover {{ background: {theme['accent_hover']}; }}",
            f"QPushButton[toolMode='true']:disabled {{ color: {theme['muted']}; background: {theme['disabled']}; border-color: {theme['border']}; }}",
            f"QPushButton[accent='true']:disabled, QPushButton[fileAction='true']:disabled {{ color: {theme['muted']}; background: {theme['disabled']}; border-color: {theme['border']}; }}",
            f"QPushButton[fileAction='true'] {{ color: {theme['file_action_text']}; background: {theme['file_action']}; border-color: {theme['file_action']}; font-weight: 600; }}",
            f"QPushButton[fileAction='true']:hover {{ background: {theme['file_action_hover']}; border-color: {theme['file_action_hover']}; }}",
            f"QPushButton[support='true'] {{ color: #243019; background: {theme['support']}; border-color: {theme['support_border']}; font-weight: 600; }}",
            f"QPushButton[support='true']:hover {{ background: {theme['support_hover']}; border-color: {theme['support_border']}; }}",
            "QPushButton[quiet='true'] { background: transparent; text-align: center; }",
            f"QPushButton#versionLabel {{ color: {theme['muted']}; font-size: {compact_font_size}px; border: 0; background: transparent; padding: 2px; }}",
            f"QPushButton#versionLabel:hover {{ color: {theme['accent']}; text-decoration: underline; }}",
            "QPushButton[captionButton='true'] { background: transparent; border: 0; border-radius: 0; padding: 0; min-width: 38px; min-height: 28px; }",
            f"QPushButton[captionButton='true']:hover {{ background: {theme['accent_soft']}; }}",
            "QPushButton#captionClose:hover { background: #C42B1C; }",
            f"QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QTextBrowser {{ background: {theme['field']}; border: 1px solid {theme['border']}; border-radius: 7px; padding: 5px 8px; min-height: 18px; }}",
            f"QComboBox:hover, QSpinBox:hover, QLineEdit:hover {{ border-color: {theme['accent']}; }}",
            "QComboBox::drop-down { border: 0; width: 22px; }",
            f"QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {theme['border']}; }}",
            f"QSlider::sub-page:horizontal {{ background: {theme['accent']}; border-radius: 3px; }}",
            f"QSlider::handle:horizontal {{ width: 16px; margin: -5px 0; border-radius: 8px; background: {theme['field']}; border: 2px solid {theme['accent']}; }}",
            "QCheckBox::indicator { width: 16px; height: 16px; }",
            f"QCheckBox::indicator:checked {{ background: {theme['accent']}; border: 1px solid {theme['accent']}; border-radius: 4px; }}",
            f"QProgressBar {{ border: 1px solid {theme['border']}; border-radius: 6px; background: {theme['disabled']}; text-align: center; min-height: 20px; }}",
            f"QProgressBar::chunk {{ background: {theme['accent']}; border-radius: 5px; }}",
            f"QFrame#viewPad {{ background: {theme['overlay']}; border: 1px solid {theme['border']}; border-radius: 10px; }}",
            f"QFrame#viewPad QPushButton {{ background: {theme['field']}; color: {theme['text']}; border: 1px solid {theme['border']}; border-radius: 6px; min-width: 58px; min-height: 27px; padding: 2px 6px; }}",
            f"QFrame#viewPad QPushButton:hover {{ background: {theme['accent_soft']}; border-color: {theme['accent']}; }}",
            f"QFrame#viewPad QPushButton:checked, QFrame#viewPad QPushButton:pressed {{ background: {theme['accent']}; color: {selected_control_text(theme)}; border-color: {theme['accent_hover']}; }}",
            f"QFrame#loadingPanel {{ background: {theme['overlay']}; border: 1px solid {theme['border']}; border-radius: 14px; }}",
            f"QFrame#loadingPanel QLabel {{ background: transparent; color: {theme['text']}; }}",
            "QFrame#loadingPanel QProgressBar { min-height: 24px; max-height: 24px; }",
            f"QFrame#loadingPanel QProgressBar::chunk {{ background: {theme['accent']}; }}",
            f"QToolTip {{ background: {theme['panel']}; color: {theme['text']}; border: 1px solid {theme['accent']}; border-radius: 7px; padding: 7px; }}",
        )
    )


APP_STYLESHEET = app_stylesheet()


def draw_ui_icon(kind: str, color: str, size: int = 18) -> QIcon:
    """Create the small, theme-aware line icons used by the desktop controls."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color), max(1.4, size / 12))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    m = size * 0.16
    r = QRectF(m, m, size - 2 * m, size - 2 * m)
    if kind.startswith("view-"):
        vertices = [QPointF(size*x, size*y) for x,y in
                    ((.50,.28),(.70,.39),(.70,.62),(.50,.51),(.30,.39),(.30,.62),(.50,.74))]
        for a,b in ((0,1),(1,2),(2,3),(3,0),(0,4),(4,5),(5,6),(6,3),(6,2)):
            painter.drawLine(vertices[a], vertices[b])
        direction = kind[5:]
        origin, destination = {
            "left": ((.06,.50),(.39,.50)),
            "right": ((.94,.50),(.61,.50)),
            "top": ((.50,.05),(.50,.40)),
            "bottom": ((.50,.96),(.50,.63)),
            "front": ((.10,.90),(.43,.59)),
            "back": ((.90,.10),(.59,.40)),
        }[direction]
        tail = QPointF(size*origin[0],size*origin[1])
        tip = QPointF(size*destination[0],size*destination[1])
        dx,dy = destination[0]-origin[0],destination[1]-origin[1]
        length = math.hypot(dx,dy)
        dx,dy = dx/length,dy/length
        painter.drawLine(tail,tip)
        painter.drawLine(tip,tip+QPointF((-dx-dy)*size*.10,(-dy+dx)*size*.10))
        painter.drawLine(tip,tip+QPointF((-dx+dy)*size*.10,(-dy-dx)*size*.10))
    elif kind == "crop":
        painter.drawLine(QPointF(size*.30,m),QPointF(size*.30,size*.70))
        painter.drawLine(QPointF(size*.30,size*.70),QPointF(size-m,size*.70))
        painter.drawLine(QPointF(m,size*.30),QPointF(size*.70,size*.30))
        painter.drawLine(QPointF(size*.70,size*.30),QPointF(size*.70,size-m))
    elif kind == "delete":
        painter.drawLine(QPointF(m,size*.28),QPointF(size-m,size*.28))
        painter.drawRoundedRect(QRectF(size*.28,size*.32,size*.44,size*.53),1,1)
        painter.drawLine(QPointF(size*.40,size*.14),QPointF(size*.60,size*.14))
    elif kind == "invert":
        painter.drawEllipse(r)
        painter.setBrush(QColor(color))
        painter.drawPie(r, 90 * 16, 180 * 16)
    elif kind == "recent":
        painter.drawArc(r, 35 * 16, 295 * 16)
        painter.drawLine(QPointF(size*.20,size*.12), QPointF(size*.20,size*.35))
        painter.drawLine(QPointF(size*.20,size*.35), QPointF(size*.41,size*.35))
        painter.drawLine(QPointF(size*.50,size*.29), QPointF(size*.50,size*.52))
        painter.drawLine(QPointF(size*.50,size*.52), QPointF(size*.67,size*.61))
    elif kind == "undo":
        painter.drawArc(QRectF(size*.25,size*.30,size*.60,size*.52),-100*16,260*16)
        painter.drawLine(QPointF(size*.25,size*.30),QPointF(size*.25,size*.53))
        painter.drawLine(QPointF(size*.25,size*.53),QPointF(size*.48,size*.53))
    elif kind == "clear":
        painter.drawLine(QPointF(size*.27,size*.27),QPointF(size*.73,size*.73))
        painter.drawLine(QPointF(size*.73,size*.27),QPointF(size*.27,size*.73))
    elif kind == "optimize":
        painter.drawPolygon(QPolygonF((QPointF(size / 2, m), QPointF(size - m, size - m), QPointF(m, size - m))))
    elif kind == "solidify":
        points = [QPointF(size*x, size*y) for x,y in ((.5,.12),(.85,.31),(.85,.70),(.5,.90),(.15,.70),(.15,.31),(.5,.5))]
        painter.setBrush(QColor(color))
        painter.drawPolygon(QPolygonF((points[0],points[1],points[6],points[5])))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for a,b in ((1,2),(2,3),(3,4),(4,5),(5,0),(0,1),(5,6),(6,1),(6,3)):
            painter.drawLine(points[a],points[b])
    elif kind == "imprint":
        painter.drawRoundedRect(QRectF(size*.38,size*.12,size*.24,size*.30),2,2)
        painter.drawLine(QPointF(size*.5,size*.42),QPointF(size*.5,size*.56))
        painter.drawRoundedRect(QRectF(size*.21,size*.56,size*.58,size*.15),1,1)
        painter.drawLine(QPointF(size*.14,size*.86),QPointF(size*.86,size*.86))
    elif kind == "actions":
        for y in (.25,.50,.75):
            painter.drawLine(QPointF(size*.40,size*y),QPointF(size*.86,size*y))
            painter.drawLine(QPointF(size*.12,size*(y-.02)),QPointF(size*.19,size*(y+.05)))
            painter.drawLine(QPointF(size*.19,size*(y+.05)),QPointF(size*.29,size*(y-.07)))
    elif kind == "stats":
        for x, top in ((m, size * .53), (size * .43, size * .30), (size * .70, size * .12)):
            painter.drawLine(QPointF(x, size - m), QPointF(x, top))
    elif kind == "general":
        painter.drawRoundedRect(r.adjusted(1, 1, -1, -1), 2, 2)
        painter.drawLine(QPointF(size * .31, size * .43), QPointF(size * .69, size * .43))
        painter.drawLine(QPointF(size * .31, size * .62), QPointF(size * .61, size * .62))
    elif kind == "folder":
        painter.drawRoundedRect(QRectF(m, size * .35, size - 2 * m, size * .48), 2, 2)
        painter.drawLine(QPointF(m, size * .35), QPointF(size * .40, size * .35))
        painter.drawLine(QPointF(size * .40, size * .35), QPointF(size * .48, size * .25))
        painter.drawLine(QPointF(size * .48, size * .25), QPointF(size * .68, size * .25))
    elif kind == "copy":
        painter.drawRoundedRect(QRectF(size * .34, size * .22, size * .46, size * .54), 2, 2)
        painter.drawRoundedRect(QRectF(size * .18, size * .37, size * .46, size * .54), 2, 2)
    elif kind == "settings":
        painter.drawEllipse(r.adjusted(2, 2, -2, -2))
        painter.drawEllipse(QRectF(size * .43, size * .43, size * .14, size * .14))
        for angle in range(0, 360, 45):
            a = math.radians(angle)
            cx, cy = size / 2, size / 2
            painter.drawLine(QPointF(cx + math.cos(a) * size * .32, cy + math.sin(a) * size * .32), QPointF(cx + math.cos(a) * size * .42, cy + math.sin(a) * size * .42))
    elif kind == "info":
        painter.drawEllipse(r)
        painter.drawPoint(QPointF(size / 2, size * .32))
        painter.drawLine(QPointF(size / 2, size * .45), QPointF(size / 2, size * .70))
    elif kind == "github":
        painter.drawEllipse(QRectF(size * .23, size * .23, size * .54, size * .54))
        painter.drawArc(QRectF(size * .32, size * .40, size * .36, size * .30), 0, 180 * 16)
        painter.drawLine(QPointF(size * .36, size * .35), QPointF(size * .28, size * .25))
        painter.drawLine(QPointF(size * .64, size * .35), QPointF(size * .72, size * .25))
    elif kind == "open":
        painter.drawRect(QRectF(m, size * .34, size - 2 * m, size * .50))
        painter.drawLine(QPointF(size * .50, size * .13), QPointF(size * .50, size * .60))
        painter.drawLine(QPointF(size * .34, size * .30), QPointF(size * .50, size * .13))
        painter.drawLine(QPointF(size * .66, size * .30), QPointF(size * .50, size * .13))
    elif kind == "minimize":
        painter.drawLine(QPointF(size * .25, size * .5), QPointF(size * .75, size * .5))
    elif kind == "maximize":
        painter.drawRect(QRectF(size * .27, size * .27, size * .46, size * .46))
    elif kind == "restore":
        painter.drawRect(QRectF(size * .22, size * .36, size * .43, size * .43))
        painter.drawLine(QPointF(size * .36, size * .36), QPointF(size * .36, size * .22))
        painter.drawLine(QPointF(size * .36, size * .22), QPointF(size * .79, size * .22))
        painter.drawLine(QPointF(size * .79, size * .22), QPointF(size * .79, size * .65))
        painter.drawLine(QPointF(size * .79, size * .65), QPointF(size * .65, size * .65))
    elif kind == "caption-close":
        painter.drawLine(QPointF(size * .28, size * .28), QPointF(size * .72, size * .72))
        painter.drawLine(QPointF(size * .72, size * .28), QPointF(size * .28, size * .72))
    elif kind == "fullscreen":
        inset = size * .27
        extent = size * .16
        painter.drawLine(QPointF(inset, inset + extent), QPointF(inset, inset))
        painter.drawLine(QPointF(inset, inset), QPointF(inset + extent, inset))
        painter.drawLine(QPointF(size - inset - extent, inset), QPointF(size - inset, inset))
        painter.drawLine(QPointF(size - inset, inset), QPointF(size - inset, inset + extent))
        painter.drawLine(QPointF(inset, size - inset - extent), QPointF(inset, size - inset))
        painter.drawLine(QPointF(inset, size - inset), QPointF(inset + extent, size - inset))
        painter.drawLine(QPointF(size - inset - extent, size - inset), QPointF(size - inset, size - inset))
        painter.drawLine(QPointF(size - inset, size - inset - extent), QPointF(size - inset, size - inset))
    painter.end()
    return QIcon(pixmap)


class ResultBridge(QObject):
    done = Signal(int, object, object, object, object, object)


class LoadBridge(QObject):
    done = Signal(int, object, object, object, object, object, object)


class EditBridge(QObject):
    done = Signal(int, object)


class DensityBridge(QObject):
    done = Signal(int, object, object, object)


class UpdateBridge(QObject):
    done = Signal(object, object, object)
    install_done = Signal(object, object)


class OperationCancelled(Exception):
    pass


@dataclass
class MeshHistoryState:
    poly: vtkPolyData
    points: np.ndarray
    faces: np.ndarray
    actor: vtkActor | None
    modified: bool
    preview_poly: vtkPolyData | None
    preview_actor: vtkActor | None
    dimension_drift: np.ndarray | None
    showing_original: bool
    preview_dirty: bool
    committed_optimization: bool
    optimization_script_passes: tuple[dict[str, object], ...]
    optimization_script_omissions: tuple[str, ...]
    solidification_applied: bool = False
    target_value: int | None = None
    quality: str | None = None
    quality_metrics: dict | None = None
    quality_measured: bool = False
    script_base_poly: vtkPolyData | None = None
    preview_script_step: dict | None = None


class _PdhValueUnion(ctypes.Union):
    _fields_ = [
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong),
    ]


class _PdhCounterValue(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("CStatus", ctypes.c_ulong), ("value", _PdhValueUnion)]


class _PdhValueItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhCounterValue)]


class WindowsGpuCounters:
    """Read the same Windows GPU Engine counters used by Task Manager."""

    PDH_FMT_DOUBLE = 0x00000200
    PDH_MORE_DATA = 0x800007D2

    def __init__(self) -> None:
        self.available = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="windows-gpu-metrics")
        self._future = None
        self._last_request = 0.0
        self._closed = False
        self.last_sample = None
        self.adapters = windows_graphics_adapters()
        self.query = ctypes.c_void_p()
        self.engine_counter = ctypes.c_void_p()
        self.memory_counter = ctypes.c_void_p()
        if os.name != "nt":
            return
        try:
            self.pdh = ctypes.WinDLL("pdh")
            self.pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.c_void_p]
            self.pdh.PdhAddEnglishCounterW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
            ]
            self.pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            self.pdh.PdhGetFormattedCounterArrayW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.c_void_p,
            ]
            self.pdh.PdhOpenQueryW.restype = ctypes.c_ulong
            self.pdh.PdhAddEnglishCounterW.restype = ctypes.c_ulong
            self.pdh.PdhCollectQueryData.restype = ctypes.c_ulong
            self.pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_ulong
            if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
                return
            engine_status = self.pdh.PdhAddEnglishCounterW(
                self.query,
                r"\GPU Engine(*)\Utilization Percentage",
                0,
                ctypes.byref(self.engine_counter),
            )
            memory_status = self.pdh.PdhAddEnglishCounterW(
                self.query,
                r"\GPU Adapter Memory(*)\Dedicated Usage",
                0,
                ctypes.byref(self.memory_counter),
            )
            if engine_status != 0:
                return
            self.available = True
            self.has_memory_counter = memory_status == 0
            self.pdh.PdhCollectQueryData(self.query)
        except (AttributeError, OSError):
            self.available = False

    def _values(self, counter: ctypes.c_void_p) -> list[tuple[str, float]]:
        size = ctypes.c_ulong(0)
        count = ctypes.c_ulong(0)
        status = self.pdh.PdhGetFormattedCounterArrayW(
            counter, self.PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), None
        )
        if status != self.PDH_MORE_DATA or size.value == 0:
            return []
        buffer = ctypes.create_string_buffer(size.value)
        status = self.pdh.PdhGetFormattedCounterArrayW(
            counter,
            self.PDH_FMT_DOUBLE,
            ctypes.byref(size),
            ctypes.byref(count),
            buffer,
        )
        if status != 0:
            return []
        items = ctypes.cast(buffer, ctypes.POINTER(_PdhValueItem))
        return [
            (items[index].szName or "", float(items[index].FmtValue.doubleValue))
            for index in range(count.value)
            if items[index].FmtValue.CStatus in (0, 1)
        ]

    @staticmethod
    def _adapter_key(instance: str) -> str:
        normalized = instance.lower()
        luid = normalized.find("luid_")
        if luid >= 0:
            normalized = normalized[luid:]
        return normalized.split("_eng_", 1)[0]

    @staticmethod
    def aggregate(engines_raw, memory_raw, process_id, previous_adapter=None):
        engines, own_engines = {}, {}
        prefix = f"pid_{process_id}_"
        for instance, value in engines_raw:
            normalized = instance.lower()
            if not np.isfinite(value):
                continue
            luid = normalized.find("luid_")
            engine = normalized[luid:] if luid >= 0 else normalized
            engines[engine] = engines.get(engine, 0.0) + max(0.0, value)
            if normalized.startswith(prefix):
                own_engines[engine] = own_engines.get(engine, 0.0) + max(0.0, value)
        loads, graphics, compute, own = {}, {}, {}, {}
        own_graphics = {}
        for engine, value in engines.items():
            key = WindowsGpuCounters._adapter_key(engine)
            loads[key] = max(loads.get(key, 0.0), value)
            kind = engine.split("_engtype_", 1)[-1]
            if kind == "3d" or kind.startswith("graphics"):
                graphics[key] = max(graphics.get(key, 0.0), value)
            if "compute" in kind or "cuda" in kind:
                compute[key] = max(compute.get(key, 0.0), value)
        for engine, value in own_engines.items():
            key = WindowsGpuCounters._adapter_key(engine)
            own[key] = max(own.get(key, 0.0), value)
            kind = engine.split("_engtype_", 1)[-1]
            if kind == "3d" or kind.startswith("graphics"):
                own_graphics[key] = max(own_graphics.get(key, 0.0), value)
        memory = {}
        for instance, value in memory_raw:
            key = WindowsGpuCounters._adapter_key(instance)
            memory[key] = max(memory.get(key, 0.0), value)
        if not loads:
            raise OSError("Windows returned no GPU engine samples")
        # Follow MeshMill's rendering adapter, not an unrelated busy accelerator.
        active = {key: value for key, value in own_graphics.items() if value > .01}
        if active:
            adapter = max(active, key=active.get)
        elif previous_adapter in loads:
            adapter = previous_adapter
        elif own_graphics:
            adapter = max(own_graphics, key=lambda key: (own_graphics[key], own.get(key, 0)))
        elif own:
            adapter = max(own, key=own.get)
        else:
            raise OSError("MeshMill GPU adapter has not been identified yet")
        return {"adapter": adapter, "load": min(100.0, loads[adapter]),
                "graphics": min(100.0, graphics.get(adapter, 0.0)),
                "compute": min(100.0, compute.get(adapter, 0.0)),
                "process_load": min(100.0, own.get(adapter, 0.0)),
                "memory": memory.get(adapter, 0.0)}

    def _read_sample(self, previous_adapter):
        if self.pdh.PdhCollectQueryData(self.query) != 0:
            raise OSError("Windows GPU performance counters are unavailable")
        sample = self.aggregate(self._values(self.engine_counter),
                                self._values(self.memory_counter) if self.has_memory_counter else [],
                                os.getpid(), previous_adapter)
        match = re.search(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)", sample["adapter"])
        luid = (int(match[1], 16) << 32) | int(match[2], 16) if match else None
        adapter = next((a for a in self.adapters if a["luid"] == luid), {})
        sample.update(name=adapter.get("name", "Rendering adapter"),
                      index=adapter.get("index"), capacity=adapter.get("capacity", 0))
        return sample

    def sample(self) -> tuple[float, float]:
        if self._closed or not self.available:
            raise OSError("Windows GPU performance counters are unavailable")
        if self._future is not None and self._future.done():
            try:
                self.last_sample = self._future.result()
            except OSError:
                self.last_sample = None
            finally:
                self._future = None
        now = time.monotonic()
        if self._future is None and now - self._last_request >= .2:
            self._last_request = now
            adapter = self.last_sample["adapter"] if self.last_sample else None
            self._future = self._executor.submit(self._read_sample, adapter)
        if self.last_sample is None:
            raise OSError("Waiting for a Windows GPU engine sample")
        return self.last_sample["load"], self.last_sample["memory"]

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        if getattr(self, "query", None):
            try:
                self.pdh.PdhCloseQuery(self.query)
            except (AttributeError, OSError):
                pass
            self.query = ctypes.c_void_p()


class NvidiaSmiCounters:
    """Nonblocking cached per-adapter activity from NVIDIA's diagnostic utility."""

    def __init__(self) -> None:
        self._last_sample_at = 0.0
        self._samples: dict[str, tuple[float, float]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-metrics")
        self._future = None
        self._closed = False

    @staticmethod
    def _read_samples() -> dict[str, tuple[float, float]]:
        samples: dict[str, tuple[float, float]] = {}
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=pci.bus_id,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=0.4,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
            )
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    samples[parts[0].lower()] = (float(parts[1]), float(parts[2]) * 1024 * 1024)
                except ValueError:
                    continue
        except (OSError, subprocess.SubprocessError):
            pass
        return samples

    def sample(self, preference: str = "system") -> tuple[float, float] | None:
        if self._closed:
            return None
        now = time.monotonic()
        if self._future is not None and self._future.done():
            try:
                samples = self._future.result()
                if samples:
                    self._samples = samples
            finally:
                self._future = None
        if not self._closed and self._future is None and now - self._last_sample_at >= 1.0:
            self._last_sample_at = now
            self._future = self._executor.submit(self._read_samples)
        if not self._samples:
            return None
        selected = preference.lower()
        if selected != "system" and selected in self._samples:
            return self._samples[selected]
        return min(self._samples.values(), key=lambda value: value[1])

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


class ElidedPathLabel(QLabel):
    """A single-line file path label that keeps the filename visible."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self.setWordWrap(False)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text if text else "")
        self._update_elision()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_elision()

    def _update_elision(self) -> None:
        available = max(24, self.contentsRect().width() - 12)
        metrics = QFontMetrics(self.font())
        normalized = self._full_text.replace("/", "\\")
        drive, remainder = os.path.splitdrive(normalized)
        parts = [part for part in remainder.split("\\") if part]
        filename = parts[-1] if parts else normalized
        root = f"{drive}\\" if drive else ""
        candidates = [
            normalized,
            f"{root}{parts[0]}\\…\\{filename}" if len(parts) > 2 else normalized,
            f"{root}…\\{filename}" if root and len(parts) > 1 else normalized,
            f"…\\{filename}" if len(parts) > 1 else filename,
            filename,
        ]
        visible = next(
            (candidate for candidate in candidates if metrics.horizontalAdvance(candidate) <= available),
            metrics.elidedText(filename, Qt.TextElideMode.ElideRight, available),
        )
        super().setText(visible)


class ViewButton(QPushButton):
    orientationRequested = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._orientation_press = False

    def mousePressEvent(self, event) -> None:
        owner = self.window()
        control_held = bool(
            (event.modifiers() | QApplication.keyboardModifiers())
            & Qt.KeyboardModifier.ControlModifier
        ) or bool(getattr(owner, "control_key_held", False))
        if (
            event.button() == Qt.MouseButton.LeftButton
            and control_held
        ):
            self._orientation_press = True
            self.setDown(True)
            self.orientationRequested.emit()
            event.accept()
            return
        self._orientation_press = False
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._orientation_press:
            self._orientation_press = False
            self.setDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class VersionButton(QPushButton):
    rightClicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class DraggableOverlayFrame(QFrame):
    """Viewport panel that can be moved without interfering with child controls."""

    movedByUser = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._drag_global_offset: QPoint | None = None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_global_offset = event.globalPosition().toPoint() - self.pos()
            self.raise_()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_global_offset is not None and bool(event.buttons() & Qt.MouseButton.LeftButton):
            parent = self.parentWidget()
            if parent is not None:
                proposed = event.globalPosition().toPoint() - self._drag_global_offset
                x = max(8, min(proposed.x(), max(8, parent.width() - self.width() - 8)))
                y = max(8, min(proposed.y(), max(8, parent.height() - self.height() - 8)))
                self.move(x, y)
                self.movedByUser.emit()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_global_offset is not None:
            self._drag_global_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)


class OrientationIndicator(QWidget):
    """Compact world-axis view indicator for the active camera."""

    def __init__(self, parent=None) -> None:
        flags = (Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                 | Qt.WindowType.WindowTransparentForInput
                 | Qt.WindowType.WindowDoesNotAcceptFocus)
        super().__init__(parent, flags if parent is not None else Qt.WindowType.Widget)
        self._overlay_parent = parent
        self._overlay_position = QPoint(14, 14)
        self._overlay_requested_visible = True
        if parent is not None:
            parent.window().installEventFilter(self)
            parent.installEventFilter(self)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        # Keep the mesh miniature and axis labels readable in a compact overlay.
        self.setFixedSize(190, 190)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setToolTip("Current workspace mesh inside the axis cube. Solid edges face the camera; dashed edges are behind. The miniature follows camera rotation and saved-axis changes.")
        self._miniature = None
        self._world_right = np.array((1., 0., 0.))
        self._world_up = np.array((0., 1., 0.))
        self._world_forward = np.array((0., 0., -1.))
        self._right = np.array((1.0, 0.0, 0.0))
        self._up = np.array((0.0, 1.0, 0.0))
        self._forward = np.array((0.0, 0.0, -1.0))
        self._orientation = (0.0, 0.0, 0.0)
        self._theme = appearance_theme(DEFAULT_APPEARANCE)
        self._viewport_background = QColor(self._theme["viewport"])

    def move(self, *args) -> None:
        self._overlay_position = QPoint(*args) if len(args) == 2 else QPoint(args[0])
        parent = self._overlay_parent
        position = parent.mapToGlobal(self._overlay_position) if parent is not None else self._overlay_position
        super().move(position)

    def setVisible(self, visible: bool) -> None:
        self._overlay_requested_visible = visible
        parent = self._overlay_parent
        available = parent is None or (parent.window().isVisible() and not parent.window().isMinimized())
        super().setVisible(visible and available)

    def eventFilter(self, watched, event) -> bool:
        if event.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show,
                            QEvent.Type.Hide, QEvent.Type.WindowStateChange):
            self.move(self._overlay_position)
            self.setVisible(self._overlay_requested_visible)
        return super().eventFilter(watched, event)

    def set_theme(self, theme: dict[str, str], viewport_theme: dict[str, str] | None = None) -> None:
        self._theme = theme
        self._viewport_background = QColor((viewport_theme or theme)["viewport"])
        self.update()

    def set_camera(
        self,
        position,
        focal,
        view_up,
        orientation,
        reference_basis: np.ndarray | None = None,
    ) -> None:
        forward = np.asarray(focal, dtype=float) - np.asarray(position, dtype=float)
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        up = np.asarray(view_up, dtype=float)
        up -= forward * float(up @ forward)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        right = np.cross(forward, up)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        self._world_right, self._world_up, self._world_forward = right.copy(), up.copy(), forward.copy()
        if reference_basis is not None and reference_basis.shape == (3, 3):
            # Keep the mesh reference fixed while named views move the camera.
            # Recalling a view should rotate the cube, not redefine its axes.
            self._right = reference_basis.T @ right
            self._up = reference_basis.T @ up
            self._forward = reference_basis.T @ forward
        else:
            self._right = right
            self._up = up
            self._forward = forward
        self._orientation = tuple(float(value) for value in orientation)
        self.update()

    def set_miniature(self, mesh) -> None:
        self._miniature = mesh
        self.update()

    def miniature_projection(self):
        if self._miniature is None:
            return None
        points, faces = self._miniature
        screen = np.column_stack((points @ self._world_right, -points @ self._world_up))
        return screen, points @ self._world_forward, faces

    def paintEvent(self, event) -> None:
        # An owned translucent surface is composited separately from native VTK.
        # Clear the whole frame instead of changing a native child-window mask.
        artwork = QPixmap(self.size())
        artwork.fill(Qt.GlobalColor.transparent)
        painter = QPainter(artwork)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center_x = self.width() / 2.0
        center = QPointF(center_x, self.height() / 2.0)

        # Project a real cube in the same camera basis as the axes. The old static diamond
        # implied orientation but did not rotate with the live X/Y/Z arrows.
        cube_vertices = np.asarray(
            (
                (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
            ),
            dtype=float,
        )
        cube_faces = (
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        )
        cube_scale = 23.0
        projected_cube = [
            QPointF(
                center.x() + float(vertex @ self._right) * cube_scale,
                center.y() - float(vertex @ self._up) * cube_scale,
            )
            for vertex in cube_vertices
        ]
        silhouette_path = QPainterPath()
        for face in cube_faces:
            face_path = QPainterPath()
            face_path.addPolygon(QPolygonF([projected_cube[index] for index in face]))
            face_path.closeSubpath()
            silhouette_path = silhouette_path.united(face_path)
        painter.setBrush(QColor("#102033"))
        painter.setPen(QPen(QColor("#102033"), 7.0))
        painter.drawPath(silhouette_path)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#EAF3FF"), 2.5))
        painter.drawPath(silhouette_path)
        face_colors = (QColor("#FF5A5F"), QColor("#B8F35A"), QColor("#4DA3FF"))
        ordered_faces = sorted(cube_faces, key=lambda indices: -float(np.mean(cube_vertices[list(indices)] @ self._forward)))
        def draw_cube_faces(rear):
            for face in ordered_faces:
                vertices = cube_vertices[list(face)]
                axis = int(np.argmin(np.ptp(vertices, axis=0)))
                outward = np.zeros(3)
                outward[axis] = vertices[0, axis]
                behind = float(outward @ self._forward) >= 0
                if behind != rear:
                    continue
                color = QColor(face_colors[axis])
                if vertices[0, axis] < 0:
                    color = color.darker(200)
                fill = QColor(color)
                fill.setAlpha(70 if rear else 65)
                painter.setBrush(fill)
                color.setAlpha(180 if rear else 255)
                pen = QPen(color, 1.0 if rear else 1.4)
                if rear:
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawPolygon(QPolygonF([projected_cube[index] for index in face]))
        draw_cube_faces(True)
        miniature = self.miniature_projection()
        if miniature is not None:
            screen, depth, faces = miniature
            projected = screen * cube_scale + (center.x(), center.y())
            points = self._miniature[0]
            triangles = points[faces]
            normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            lighting = np.abs(normal @ (-self._world_forward * .8 + self._world_up * .4))
            painter.setPen(Qt.PenStyle.NoPen)
            base = QColor(self._theme["accent"])
            shades = [base.lighter(115 + 10 * level) for level in range(6)]
            shade_indices = np.minimum(5, (lighting * 5).astype(int))
            for index in np.argsort(-depth[faces].mean(axis=1)):
                painter.setBrush(shades[shade_indices[index]])
                painter.drawPolygon(QPolygonF([QPointF(*p) for p in projected[faces[index]]]))
        draw_cube_faces(False)

        axes = (
            ("X", np.array((1.0, 0.0, 0.0)), QColor("#FF5A5F")),
            ("Y", np.array((0.0, 1.0, 0.0)), QColor("#B8F35A")),
            ("Z", np.array((0.0, 0.0, 1.0)), QColor("#4DA3FF")),
        )
        projected = []
        for label, axis, color in axes:
            screen = np.array((float(axis @ self._right), -float(axis @ self._up)))
            depth = float(axis @ self._forward)
            projected.append((depth, label, screen, color))
        label_positions = []
        for depth, label, screen, color in sorted(projected, key=lambda item: item[1]):
            end = center + QPointF(float(screen[0] * 36.0), float(screen[1] * 36.0))
            color = color.darker(round(100 + 90 * (depth + 1) / 2))
            magnitude = float(np.linalg.norm(screen))
            anchor = {"X": (1., .2), "Y": (-.3, -1.), "Z": (.8, -.7)}[label]
            offset = screen * 67 + np.asarray(anchor) * 68 * (1 - magnitude) ** 2
            badge_center = center + QPointF(float(offset[0]), float(offset[1]))
            label_positions.append((end, badge_center, label, color))

        label_font = painter.font()
        label_font.setPointSizeF(max(9.0, label_font.pointSizeF() * .78))
        label_font.setBold(False)
        painter.setFont(label_font)
        values = dict(zip("XYZ", (round(value) for value in self._orientation)))
        badge_text = {label: f"{label}: {values[label]:+d}°" for label in "XYZ"}
        metrics = painter.fontMetrics()
        badge_width = max(metrics.horizontalAdvance(text) for text in badge_text.values()) + 10
        badge_height = metrics.height() + 6
        from orientation_layout import place_axis_badges

        cube_bounds = (
            min(point.x() for point in projected_cube),
            min(point.y() for point in projected_cube),
            max(point.x() for point in projected_cube),
            max(point.y() for point in projected_cube),
        )
        badge_points = place_axis_badges(
            [(item[1].x(), item[1].y()) for item in label_positions],
            cube_bounds, self.width(), self.height(), badge_width, badge_height,
        )
        for index, (end, _badge_center, label, color) in enumerate(label_positions):
            badge = QRect(round(badge_points[index, 0] - badge_width / 2),
                          round(badge_points[index, 1] - badge_height / 2), badge_width, badge_height)
            # Keep the connector and its halo outside the miniature's area.
            painter.save()
            outside_cube = QPainterPath()
            outside_cube.addRect(QRectF(self.rect()))
            cube_clearance = QPainterPath()
            cube_clearance.addRect(QRectF(
                cube_bounds[0] - 6, cube_bounds[1] - 6,
                cube_bounds[2] - cube_bounds[0] + 12,
                cube_bounds[3] - cube_bounds[1] + 12,
            ))
            painter.setClipPath(outside_cube.subtracted(cube_clearance))
            leader = QColor(color)
            leader.setAlpha(255)
            painter.setPen(QPen(QColor("#102033"), 5.0))
            painter.drawLine(end, QPointF(badge.center()))
            painter.setPen(QPen(leader, 2.0))
            painter.drawLine(end, QPointF(badge.center()))
            painter.restore()
            border = QColor(color)
            border.setAlpha(255)
            painter.setBrush(QColor("#102033"))
            painter.setPen(QPen(QColor("#102033"), 6.0))
            painter.drawRoundedRect(badge, 5, 5)
            painter.setPen(QPen(border, 1.5))
            painter.drawRoundedRect(badge, 5, 5)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, badge_text[label])
        painter.end()
        canvas = QPainter(self)
        canvas.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        canvas.fillRect(self.rect(), Qt.GlobalColor.transparent)
        canvas.drawPixmap(0, 0, artwork)


class ViewportScaleBar(QWidget):
    """Camera-aware scale ruler drawn over the viewport."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(150, 40)
        self._label = ""
        self._pixels = 96.0
        self._background = QColor("#0B1B34")
        self._card = QColor("#132842")
        self._border = QColor("#3C5870")
        self._foreground = QColor("#EBF2FC")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # QVTKRenderWindowInteractor is a native OpenGL surface. Transparent
        # child-widget updates can leave previous scale-bar frames behind while
        # zooming, so this small overlay clears itself with the viewport color.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_scale(self, label: str, pixels: float) -> None:
        clamped_pixels = max(24.0, min(118.0, float(pixels)))
        metrics = QFontMetrics(self.font())
        content_width = max(clamped_pixels, float(metrics.horizontalAdvance(label) + 12))
        desired_width = max(64, min(142, math.ceil(content_width + 16)))
        if (
            label == self._label
            and abs(clamped_pixels - self._pixels) < 0.25
            and desired_width == self.width()
        ):
            return
        self._label = label
        self._pixels = clamped_pixels
        self.setFixedWidth(desired_width)
        self.update()

    def set_theme(self, theme: dict[str, str], viewport_theme: dict[str, str] | None = None) -> None:
        self._background = QColor((viewport_theme or theme)["viewport"])
        self._card = QColor(theme["surface"])
        self._card.setAlpha(232)
        self._border = QColor(theme["border"])
        self._foreground = QColor(theme["text"])
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._background)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self._border, 1.0))
        painter.setBrush(self._card)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)
        foreground = QColor(self._foreground)
        foreground.setAlpha(235)
        painter.setPen(QPen(foreground, 2.0))
        right = self.width() - 8.0
        left = right - self._pixels
        y = 27.0
        painter.drawLine(QPointF(left, y), QPointF(right, y))
        painter.drawLine(QPointF(left, y - 5), QPointF(left, y + 5))
        painter.drawLine(QPointF(right, y - 5), QPointF(right, y + 5))
        foreground.setAlpha(245)
        painter.setPen(foreground)
        painter.drawText(
            QRect(4, 2, self.width() - 8, 19),
            Qt.AlignmentFlag.AlignCenter,
            self._label,
        )


class DensityLegend(QWidget):
    """Compact, rendered density scale that stays legible over the viewport."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(230, 78)
        self._theme = appearance_theme(DEFAULT_APPEARANCE)
        self._viewport_background = QColor(self._theme["viewport"])
        self._gradient_colors: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_theme(self, theme: dict[str, str], viewport_theme: dict[str, str] | None = None) -> None:
        self._theme = theme
        self._viewport_background = QColor((viewport_theme or theme)["viewport"])
        self.update()

    def set_gradient_colors(
        self, colors: tuple[tuple[int, int, int], tuple[int, int, int]] | None
    ) -> None:
        self._gradient_colors = colors
        self.update()

    # Compatibility with the previous QLabel setup. These calls are intentionally ignored
    # because this widget renders a real gradient rather than rich-text dots.
    def setText(self, _text: str) -> None:
        pass

    def setTextFormat(self, _format) -> None:
        pass

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._viewport_background)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QColor(self._theme["panel"])
        card.setAlpha(238)
        painter.setPen(QPen(QColor(self._theme["border"]), 1.0))
        painter.setBrush(card)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 9, 9)
        text_color = QColor(self._theme["text"])
        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPointSizeF(max(8.0, title_font.pointSizeF() - 1.0))
        painter.setFont(title_font)
        painter.setPen(text_color)
        painter.drawText(QRect(14, 10, self.width() - 28, 17), Qt.AlignmentFlag.AlignLeft, "POINT DENSITY")
        gradient = QLinearGradient(14, 38, self.width() - 14, 38)
        viewport = self._viewport_background
        stops = (
            density_color_stops(*self._gradient_colors)
            if self._gradient_colors is not None
            else density_color_stops((viewport.red(), viewport.green(), viewport.blue()))
        )
        for position, color in zip(np.linspace(0.0, 1.0, len(stops)), stops):
            gradient.setColorAt(float(position), QColor(*[int(component) for component in color]))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(QRectF(14, 35, self.width() - 28, 14), 6, 6)
        label_font = painter.font()
        label_font.setBold(False)
        label_font.setPointSizeF(max(8.0, label_font.pointSizeF() - 1.0))
        painter.setFont(label_font)
        painter.setPen(text_color)
        painter.drawText(QRect(14, 54, 90, 16), Qt.AlignmentFlag.AlignLeft, "Sparse")
        painter.drawText(QRect(self.width() - 104, 54, 90, 16), Qt.AlignmentFlag.AlignRight, "Dense")


class ArrowComboBox(QComboBox):
    """Combo box with a chevron that follows the active appearance palette."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        text_color = self.palette().color(QPalette.ColorRole.Text)
        if not self.isEnabled():
            text_color.setAlpha(120)
        painter.setBrush(text_color)
        center_x = self.width() - 13
        center_y = self.height() // 2 + 1
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(center_x - 4, center_y - 2),
                    QPointF(center_x + 4, center_y - 2),
                    QPointF(center_x, center_y + 3),
                ]
            )
        )


class TooltipPopup(QLabel):
    """Application-owned tooltip so Windows cannot replace the theme colors."""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.ToolTip)
        self.setObjectName("meshmillTooltip")
        self.setWordWrap(True)
        self.setMaximumWidth(520)
        self.setMargin(9)
        self.apply_appearance(DEFAULT_APPEARANCE)

    def apply_appearance(self, name: str) -> None:
        theme = appearance_theme(name)
        self.setStyleSheet(
            "QLabel#meshmillTooltip {"
            f"background-color: {theme['panel']}; color: {theme['text']}; "
            f"border: 1px solid {theme['accent']}; border-radius: 7px; "
            "font-size: 12px; font-weight: 500;"
            "}"
        )

    def show_help(self, text: str, position: QPoint) -> None:
        self.setText(text)
        self.adjustSize()
        screen = QApplication.screenAt(position) or QApplication.primaryScreen()
        target = position + QPoint(12, 18)
        if screen is not None:
            available = screen.availableGeometry()
            x = min(max(target.x(), available.left()), available.right() - self.width() + 1)
            y = min(max(target.y(), available.top()), available.bottom() - self.height() + 1)
            if y < position.y() < y + self.height():
                y = max(available.top(), position.y() - self.height() - 12)
            target = QPoint(x, y)
        self.move(target)
        self.show()
        self.raise_()


class DeferredTooltipFilter(QObject):
    """Quiet, one-second, movement-sensitive help without changing the cursor."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.widget: QWidget | None = None
        self.position = QPoint()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._show)
        self.popup = TooltipPopup()

    def apply_appearance(self, name: str) -> None:
        self.popup.apply_appearance(name)

    def eventFilter(self, watched, event) -> bool:
        if not isinstance(watched, QWidget) or not watched.toolTip():
            return False
        kind = event.type()
        if kind == QEvent.Type.ToolTip:
            return True
        if kind == QEvent.Type.Enter:
            self._arm(watched, QCursor.pos())
        elif kind == QEvent.Type.MouseMove:
            self.popup.hide()
            position = event.globalPosition().toPoint() if hasattr(event, "globalPosition") else QCursor.pos()
            self._arm(watched, position)
        elif kind in (QEvent.Type.Leave, QEvent.Type.Hide, QEvent.Type.FocusOut) and watched is self.widget:
            self.timer.stop()
            self.popup.hide()
            self.widget = None
        return False

    def _arm(self, widget: QWidget, position: QPoint) -> None:
        self.widget = widget
        self.position = position
        self.timer.start()

    def _show(self) -> None:
        if self.widget is not None and self.widget.underMouse():
            self.popup.show_help(self.widget.toolTip(), self.position)


class CropPolygonOverlay(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.points: list[QPoint] = []
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def set_points(self, points: list[QPoint]) -> None:
        self.points = [QPoint(point) for point in points]
        self.update()
        self.show()
        self.raise_()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        if len(self.points) < 3:
            return
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        polygon = QPolygonF([QPointF(point) for point in self.points])
        painter.setBrush(QColor(40, 155, 255, 18))
        pen = QPen(QColor(125, 215, 255, 245), 3)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawPolygon(polygon)
        painter.setPen(QPen(QColor(8, 35, 55, 255), 2))
        painter.setBrush(QColor(145, 225, 255, 245))
        for point in self.points:
            painter.drawEllipse(QPointF(point), 6, 6)


class GeometryActivityGraph(QWidget):
    """Recent readings in detail, with older activity summarized toward the left."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.samples: list[tuple[float, float, bool]] = []
        self.sample_times: list[float] = []
        self.refresh_seconds = 1.0
        self.retention_seconds = 900.0
        self.detail_seconds = 30.0
        self._viewport_width = 240
        self._revision = 0
        self._summary_cache = None
        self._paused_history = None
        self.max_samples = 1800
        self.pixels_per_sample = 2
        self.operation = "Idle"
        self.geometry_text = "No mesh loaded"
        self.apply_appearance(DEFAULT_APPEARANCE)
        self.setFixedHeight(128)

    def apply_appearance(self, name: str) -> None:
        theme = appearance_theme(name)
        background, foreground = QColor(theme["panel"]), QColor(theme["text"])
        dark = background.lightnessF() < 0.5
        grid = QColor(*[round(background.getRgb()[i] * 0.70 + foreground.getRgb()[i] * 0.30)
                        for i in range(3)])
        self.plot_colors = {
            "background": background, "text": foreground, "grid": grid,
            "cpu": QColor("#78B7FF" if dark else "#1D4ED8"),
            "gpu": QColor("#4EE0A7" if dark else "#16723A"),
            "legend": foreground,
            "active": QColor(120, 183, 255, 24) if dark else QColor(37, 99, 235, 18),
        }
        self.update()

    def add_sample(
        self,
        cpu: float,
        gpu: float,
        processing: bool,
        operation: str,
        points: int,
        triangles: int,
        *,
        sample_time: float | None = None,
    ) -> None:
        cpu = max(0.0, min(100.0, cpu))
        gpu = max(0.0, min(100.0, gpu))
        now = time.monotonic() if sample_time is None else sample_time
        if len(self.sample_times) != len(self.samples):
            self.sample_times = [now - (len(self.samples) - i) * self.refresh_seconds
                                 for i in range(len(self.samples))]
        self.samples.append((cpu, gpu, processing))
        self.sample_times.append(now)
        self._trim_history()
        self._revision += 1
        self.operation = operation
        self.geometry_text = (
            f"{fmt_count(points)} vertices / {fmt_count(triangles)} triangles"
            if points or triangles
            else "No mesh loaded"
        )
        self.update()

    def configure_history(
        self, refresh_ms: int, minutes: int, visible_seconds: int = 30, viewport_width: int = 0
    ) -> None:
        self.refresh_seconds = max(0.25, refresh_ms / 1000)
        self.retention_seconds = max(1, minutes * 60)
        self.detail_seconds = max(1, visible_seconds)
        self.max_samples = math.ceil(self.retention_seconds / 0.25) + 1
        self._trim_history()
        self._viewport_width = max(240, viewport_width)
        width = round(self._viewport_width * max(1.0, 30 / self.detail_seconds))
        if self.width() != width:
            self.setFixedWidth(width)
        self.update()

    def _trim_history(self) -> None:
        first = max(0, len(self.samples) - self.max_samples)
        if self.sample_times and len(self.sample_times) == len(self.samples):
            first = max(first, bisect_left(
                self.sample_times, self.sample_times[-1] - self.retention_seconds
            ))
        if first:
            self.samples = self.samples[first:]
            self.sample_times = self.sample_times[first:]
            self._revision += 1

    def set_paused(self, paused: bool) -> None:
        if paused and self._paused_history is None:
            self._paused_history = (self.samples.copy(), self.sample_times.copy())
        elif not paused:
            self._paused_history = None
        self._summary_cache = None
        self.update()

    def _history_summary(self, width: int):
        samples, times = (self._paused_history if self._paused_history is not None
                          else (self.samples, self.sample_times))
        if len(times) != len(samples):
            times = [i * self.refresh_seconds for i in range(len(samples))]
        key = (self._revision, width, self.retention_seconds, self.detail_seconds,
               self._paused_history is not None)
        if self._summary_cache is None or self._summary_cache[0] != key:
            self._summary_cache = (key, summarize_history(
                samples, times, self.retention_seconds, self.detail_seconds, width
            ))
        return self._summary_cache[1]

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colors = self.plot_colors
        painter.fillRect(self.rect(), colors["background"])
        painter.setPen(colors["text"])
        painter.drawText(
            QRect(8, 0, max(1, self.width() - 16), 20),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"Geometry activity / {self.operation} / {self.geometry_text}",
        )

        left, top, right, bottom = 8, 23, self.width() - 8, self.height() - 35
        height = max(1, bottom - top)
        painter.setPen(QPen(colors["grid"], 1))
        for fraction in (0.25, 0.5, 0.75):
            y = top + height * fraction
            painter.drawLine(left, round(y), right, round(y))

        if self.samples:
            summary = self._history_summary(max(1, right - left))
            active_color = colors["active"]
            for item in summary:
                if item.processing:
                    painter.fillRect(
                        round(left + item.x),
                        top,
                        2,
                        height,
                        active_color,
                    )

            def draw_line(channel: str, color: QColor) -> None:
                peak_color = QColor(color)
                peak_color.setAlpha(120)
                painter.setPen(QPen(peak_color, 1))
                for item in summary:
                    x = left + item.x
                    painter.drawLine(
                        QPointF(x, bottom - height * getattr(item, channel + "_min") / 100),
                        QPointF(x, bottom - height * getattr(item, channel + "_max") / 100),
                    )
                points = QPolygonF(
                    [
                        QPointF(left + item.x,
                                bottom - height * getattr(item, channel + "_mean") / 100)
                        for item in summary
                    ]
                )
                painter.setPen(QPen(QColor(color), 1.8))
                painter.drawPolyline(points)

            draw_line("cpu", colors["cpu"])
            draw_line("gpu", colors["gpu"])

        painter.setPen(colors["text"])
        occupied = []
        _, axis_times = (self._paused_history if self._paused_history is not None
                         else (self.samples, self.sample_times))
        span = history_span(axis_times, self.retention_seconds, self.detail_seconds)
        for age in sorted({0, self.detail_seconds, 120, 300, span}):
            if age > span:
                continue
            label = "Now" if age == 0 else f"{age / 60:g}m" if age >= 60 else f"{age:g}s"
            x = left + history_position(age, span, self.detail_seconds) * (right-left)
            label_width = painter.fontMetrics().horizontalAdvance(label)
            start = min(right-label_width, max(left, round(x-label_width/2)))
            if any(start < end + 8 and start + label_width > begin - 8
                   for begin, end in occupied):
                continue
            painter.drawText(start, self.height()-21, label)
            occupied.append((start, start+label_width))

        painter.fillRect(QRect(0, self.height() - 18, self.width(), 18), colors["background"])
        legend_y = self.height() - 5
        right_edge = self.width() - 8
        metrics = painter.fontMetrics()
        for text, color in (
            ("geometry processing", colors["legend"]),
            ("GPU", colors["gpu"]),
            ("CPU", colors["cpu"]),
        ):
            painter.setPen(QColor(color))
            text_width = metrics.horizontalAdvance(text)
            painter.drawText(right_edge - text_width, legend_y, text)
            right_edge -= text_width + 10


def fmt_count(value: int) -> str:
    return f"{value:,}"


def fmt_bytes(value: int) -> str:
    if value >= 1024 ** 3:
        return f"{value / 1024 ** 3:.2f} GB"
    if value >= 1024 ** 2:
        return f"{value / 1024 ** 2:.1f} MB"
    if value >= 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value} bytes"


def fmt_elapsed(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def version_key(value: str) -> tuple[int, ...]:
    numbers = tuple(int(part) for part in re.findall(r"\d+", value))
    return numbers or (0,)


def fetch_latest_release() -> dict:
    request = Request(
        LATEST_RELEASE_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"MeshMill/{APP_VERSION}",
        },
    )
    with urlopen(request, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8"))
    version = str(payload.get("tag_name", "")).strip().lstrip("vV")
    release_url = str(payload.get("html_url", "")).strip()
    if not version or not release_url:
        raise ValueError("The latest release did not include a version and download page.")
    return {
        "version": version,
        "release_url": release_url,
        "assets": [
            {
                "name": str(asset.get("name", "")),
                "url": str(asset.get("browser_download_url", "")),
                "size": int(asset.get("size", 0) or 0),
                "digest": str(asset.get("digest", "") or ""),
            }
            for asset in payload.get("assets", [])
            if isinstance(asset, dict)
        ],
    }


def update_asset_for_current_platform(release: dict) -> dict | None:
    """Return an installable asset, never a portable archive."""
    version = re.escape(str(release.get("version", "")))
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64", "x64"}:
        suffixes = ["windows-x64-setup.exe"]
    elif system == "darwin" and machine in {"arm64", "aarch64"}:
        suffixes = ["macos-arm64.pkg", "macos-universal.pkg"]
    elif system == "darwin" and machine in {"amd64", "x86_64", "x64"}:
        suffixes = ["macos-x64.pkg", "macos-universal.pkg"]
    elif system == "linux" and machine in {"amd64", "x86_64", "x64"}:
        suffixes = ["linux-x86_64.AppImage", "linux-x64.AppImage"]
    else:
        return None
    assets = release.get("assets", [])
    for suffix in suffixes:
        pattern = rf"^MeshMill-{version}-{re.escape(suffix)}$"
        for asset in assets:
            if re.match(pattern, str(asset.get("name", "")), re.IGNORECASE):
                return asset
    return None


def download_verified_update(
    asset: dict,
    destination: Path,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> Path:
    url = str(asset.get("url", ""))
    digest = str(asset.get("digest", ""))
    if not url.startswith("https://github.com/"):
        raise ValueError("The update asset is not hosted by the MeshMill GitHub release.")
    if not digest.lower().startswith("sha256:"):
        raise ValueError("The release asset does not include a SHA-256 digest.")
    expected = digest.split(":", 1)[1].strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("The release asset has an invalid SHA-256 digest.")
    request = Request(url, headers={"User-Agent": f"MeshMill/{APP_VERSION}"})
    hasher = hashlib.sha256()
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=30) as response, temporary.open("wb") as output:
            headers = getattr(response, "headers", None)
            content_length = headers.get("Content-Length", 0) if headers is not None else 0
            total = int(content_length or asset.get("size", 0) or 0)
            received = 0
            while chunk := response.read(1024 * 1024):
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled("Download cancelled")
                output.write(chunk)
                hasher.update(chunk)
                received += len(chunk)
                if progress_callback is not None:
                    progress_callback(received, total)
        if hasher.hexdigest().lower() != expected:
            raise ValueError("The downloaded update did not match its release digest.")
        temporary.replace(destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def binary_stl_size(triangle_count: int) -> int:
    return 84 + 50 * triangle_count


def bounds_size(bounds: tuple[float, ...]) -> tuple[float, float, float]:
    return bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]


def fmt_size(size: tuple[float, float, float]) -> str:
    return " x ".join(f"{v:,.3f}" for v in size)


def polydata_arrays(poly: vtkPolyData) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=np.float64)
    cells = poly.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    indices = vtk_to_numpy(cells.GetConnectivityArray())
    if np.any(np.diff(offsets) != 3):
        raise ValueError("The geometry did not resolve to a triangle-only mesh.")
    faces = np.asarray(indices.reshape(-1, 3), dtype=np.int32)
    return points, faces


def arrays_polydata(points: np.ndarray, faces: np.ndarray) -> vtkPolyData:
    vtk_points = vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64)
    connectivity = np.ascontiguousarray(faces, dtype=np.int64).ravel()
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    poly = vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)
    return poly


def load_stl(
    path: Path, progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[vtkPolyData, np.ndarray, np.ndarray]:
    if path.suffix.lower() != '.stl':
        from mesh_formats import load_geometry
        return load_geometry(path, progress_callback)
    reader = vtkSTLReader()
    reader.SetFileName(os.fspath(path))
    reader.MergingOn()
    file_bytes = path.stat().st_size
    if progress_callback is not None:
        progress_callback("Reading mesh", 0, file_bytes)
        reader.AddObserver(
            "ProgressEvent",
            lambda source, _event: progress_callback(
                "Reading mesh", round(source.GetProgress() * file_bytes), file_bytes
            ),
        )
    reader.Update()
    triangle = vtkTriangleFilter()
    triangle.SetInputConnection(reader.GetOutputPort())
    if progress_callback is not None:
        progress_callback("Preparing triangles", 0, reader.GetOutput().GetNumberOfPolys())
        triangle.AddObserver(
            "ProgressEvent",
            lambda source, _event: progress_callback(
                "Preparing triangles",
                round(source.GetProgress() * reader.GetOutput().GetNumberOfPolys()),
                reader.GetOutput().GetNumberOfPolys(),
            ),
        )
    triangle.Update()
    poly = vtkPolyData()
    poly.DeepCopy(triangle.GetOutput())
    if poly.GetNumberOfPolys() == 0:
        raise ValueError("No triangles were found in the file.")
    points, faces = polydata_arrays(poly)
    if progress_callback is not None:
        progress_callback("Preparing triangles", len(faces), len(faces))
    return poly, points, faces


def binary_stl_triangle_count(path: Path) -> int | None:
    size = path.stat().st_size
    if size < 84:
        return None
    with path.open("rb") as stream:
        header = stream.read(84)
    count = struct.unpack_from("<I", header, 80)[0]
    return count if 84 + 50 * count == size else None


def load_binary_stl_overview(
    path: Path, triangle_count: int, triangle_limit: int
) -> tuple[vtkPolyData, np.ndarray, np.ndarray]:
    sample_count = min(max(1, triangle_limit), triangle_count)
    record_dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    records = np.memmap(path, dtype=record_dtype, mode="r", offset=84, shape=(triangle_count,))
    indices = np.linspace(0, triangle_count - 1, sample_count, dtype=np.int64)
    points = np.ascontiguousarray(records[indices]["vertices"].reshape(-1, 3), dtype=np.float64)
    del records
    faces = np.arange(len(points), dtype=np.int32).reshape(-1, 3)
    return arrays_polydata(points, faces), points, faces


def tiled_runtime_break_even(
    triangle_count: int,
    algorithm: str,
    index_ready: bool,
) -> tuple[float, float, float] | None:
    """Estimate full versus tiled Fast QEM time from the reference workload.

    The estimate is intentionally limited to the CPU Fast QEM path that has a
    matched full-mesh/tiled measurement.  GPU-guided surface analysis currently
    runs independently in every tile, so its tiled result is not used for an
    automatic choice.
    """
    if triangle_count <= 0 or algorithm != "Fast QEM":
        return None

    # 4,126,315-triangle reference mesh, matched target, eight CPU workers.
    full_seconds_per_triangle = 8.58 / 4_126_315
    tiled_seconds_per_triangle = 4.67 / 4_126_315
    cold_index_seconds = 14.6
    full_seconds = triangle_count * full_seconds_per_triangle
    tiled_seconds = triangle_count * tiled_seconds_per_triangle
    setup_seconds = 0.0 if index_ready else cold_index_seconds
    return full_seconds, tiled_seconds + setup_seconds, setup_seconds


def load_stl_smart(
    path: Path,
    mode: str,
    memory_budget_bytes: int,
    overview_triangle_limit: int,
    cancel_event: threading.Event | None = None,
    analysis_workers: int | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    algorithm: str = "Fast QEM",
    allow_auto_tiling: bool = True,
) -> tuple[vtkPolyData, np.ndarray, np.ndarray, dict]:
    triangle_count = binary_stl_triangle_count(path) if path.suffix.lower() == '.stl' else None
    if path.suffix.lower() != '.stl':
        if path.stat().st_size*10 > memory_budget_bytes and mode != 'full':
            raise MemoryError('This format requires full-memory import. Increase the memory budget or convert to binary STL for bounded-memory loading.')
        poly, points, faces = load_stl(path, progress_callback)
        measurements = source_geometry_measurements(points, faces, progress_callback, cancel_event)
        return poly, points, faces, {'overview': False, 'total_triangles': len(faces),
            'estimated_memory': points.nbytes+faces.nbytes,
            'measurements': measurements, 'render_poly': prepare_viewport_poly(poly, progress_callback)}
    estimated_memory = (
        triangle_count * ESTIMATED_BYTES_PER_TRIANGLE
        if triangle_count is not None
        else path.stat().st_size * 5
    )
    index_directory = default_index_directory(path)
    try:
        cached_manifest = load_manifest(index_directory)
        index_ready = index_matches(path, cached_manifest)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        cached_manifest = None
        index_ready = False
    break_even = tiled_runtime_break_even(triangle_count or 0, algorithm, index_ready)
    use_auto_tiles = bool(
        mode == "auto"
        and allow_auto_tiling
        and estimated_memory <= memory_budget_bytes
        and break_even is not None
        and break_even[1] < break_even[0]
    )
    use_tiled = mode == "tiled" or use_auto_tiles
    use_overview = mode == "overview" or use_tiled or (
        mode == "auto" and estimated_memory > memory_budget_bytes
    )
    if use_overview:
        if triangle_count is None:
            raise MemoryError(
                "This STL is too large for the current memory budget, and bounded overview "
                "loading currently requires a binary STL. Increase the budget, choose Full mesh, "
                "or convert the file to binary STL."
            )
        poly, points, faces = load_binary_stl_overview(
            path, triangle_count, overview_triangle_limit
        )
        manifest = cached_manifest
        if manifest is None or not index_ready:
            manifest = build_index(
                path,
                index_directory,
                target_triangles_per_tile=max(262_144, overview_triangle_limit),
                cancel_event=cancel_event,
                analysis_workers=analysis_workers,
                progress_callback=progress_callback,
            )
        return poly, points, faces, {
            "overview": True,
            "total_triangles": triangle_count,
            "estimated_memory": estimated_memory,
            "index_directory": os.fspath(index_directory),
            "index_tiles": len(manifest.tiles),
            "index_max_depth": manifest.max_depth,
            "tiled_edit": use_tiled,
            "tiled_auto_selected": use_auto_tiles,
            "tiled_runtime_estimate": break_even,
        }
    poly, points, faces = load_stl(path, progress_callback=progress_callback)
    if progress_callback is not None:
        progress_callback("Measuring source geometry", 0, len(faces))
    measurements = source_geometry_measurements(points, faces, progress_callback, cancel_event)
    rendered = prepare_viewport_poly(poly, progress_callback)
    return poly, points, faces, {
        "overview": False,
        "total_triangles": len(faces),
        "estimated_memory": estimated_memory,
        "measurements": measurements,
        "render_poly": rendered,
    }


def available_physical_memory() -> int:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    else:
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            pass
    return 4 * 1024**3


def automatic_overview_triangle_limit(memory_budget_bytes: int, vram_capacity_bytes: int) -> int:
    """Choose an interactive overview size from current RAM and GPU capacity."""
    ram_allowance = max(64 * 1024**2, int(memory_budget_bytes * 0.05))
    vram_allowance = (
        max(64 * 1024**2, int(vram_capacity_bytes * 0.15))
        if vram_capacity_bytes > 0
        else ram_allowance
    )
    allowance = min(ram_allowance, vram_allowance)
    estimated_display_bytes_per_triangle = 220
    capacity = allowance // estimated_display_bytes_per_triangle
    return max(100_000, min(4_000_000, int(capacity)))


def windows_graphics_adapters() -> list[dict]:
    if os.name != "nt":
        return []

    class Guid(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class AdapterDesc1(ctypes.Structure):
        _fields_ = [
            ("Description", ctypes.c_wchar * 128),
            ("VendorId", ctypes.c_uint),
            ("DeviceId", ctypes.c_uint),
            ("SubSysId", ctypes.c_uint),
            ("Revision", ctypes.c_uint),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", ctypes.c_longlong),
            ("Flags", ctypes.c_uint),
        ]

    iid = Guid(
        0x770AAE78,
        0xF26F,
        0x4DBA,
        (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
    )
    factory = ctypes.c_void_p()

    def method(pointer: ctypes.c_void_p, index: int, prototype):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return prototype(table[index])

    try:
        create_factory = ctypes.WinDLL("dxgi").CreateDXGIFactory1
        create_factory.argtypes = [ctypes.POINTER(Guid), ctypes.POINTER(ctypes.c_void_p)]
        create_factory.restype = ctypes.c_long
        if create_factory(ctypes.byref(iid), ctypes.byref(factory)) < 0 or not factory.value:
            return []

        enum_adapter = method(
            factory,
            12,
            ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)),
        )
        adapters = []
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapter(factory, index, ctypes.byref(adapter)) < 0 or not adapter.value:
                break
            get_desc = method(
                adapter,
                10,
                ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(AdapterDesc1)),
            )
            release_adapter = method(
                adapter, 2, ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
            )
            desc = AdapterDesc1()
            if get_desc(adapter, ctypes.byref(desc)) >= 0 and not (desc.Flags & 2):
                adapters.append({"index": index, "name": desc.Description,
                                 "luid": int(desc.AdapterLuid) & 0xFFFFFFFFFFFFFFFF,
                                 "capacity": int(desc.DedicatedVideoMemory)})
            release_adapter(adapter)
            index += 1
        return adapters
    except (AttributeError, OSError, ValueError):
        return []
    finally:
        if factory.value:
            release_factory = method(
                factory, 2, ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
            )
            release_factory(factory)


def dedicated_vram_capacity() -> int:
    return max((adapter["capacity"] for adapter in windows_graphics_adapters()), default=0)


def suggested_mesh_path(source: Path, triangle_count: int) -> Path:
    """Replace generated triangle-count suffixes instead of stacking them."""
    stem = re.sub(r"(?:_\d+k)+$", "", source.stem, flags=re.IGNORECASE) or "mesh"
    output = source.with_name(f"{stem}_{triangle_count // 1000}k.stl")
    if output.name.casefold() == source.name.casefold():
        output = output.with_name(f"{output.stem}_export.stl")
    return output


def save_stl(poly: vtkPolyData, path: Path) -> None:
    marker = poly.GetFieldData().GetArray("ImprintClosedOutput")
    if marker is not None:
        from imprint_cleanup import clean_generated_mesh
        poly, _ = clean_generated_mesh(poly, closed=bool(marker.GetValue(0)))
    writer = vtkSTLWriter()
    writer.SetFileName(os.fspath(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise OSError(f"VTK could not save {path}")


class MeshInteractorStyle(vtkInteractorStyleTrackballCamera):
    """CAD-style navigation: middle-drag orbit, Shift+middle-drag pan, wheel zoom."""

    def __init__(
        self, widget: QWidget, loupe_callback, crop_callback, keyboard_nav_callback,
        measurement_callback,
    ) -> None:
        super().__init__()
        self._widget = widget
        self._loupe_callback = loupe_callback
        self._crop_callback = crop_callback
        self._keyboard_nav_callback = keyboard_nav_callback
        self._wheel_timer = QTimer(widget)
        self._wheel_timer.setInterval(16)
        self._wheel_timer.timeout.connect(self._advance_wheel)
        self._wheel_steps = 0
        self._wheel_command = None
        self._measurement_callback = measurement_callback
        self._right_down = False
        self._crop_down = False
        self._crop_edit_down = False
        self._crop_right_consumed = False
        self._consume_left_release = False
        self._left_pan_down = False
        self._pan_depth: float | None = None
        self._middle_mode: str | None = None
        self._rotation_center: np.ndarray | None = None
        self._zoom_picker = vtkHardwarePicker()
        self.invert_horizontal = False
        self.invert_vertical = False
        self.invert_zoom = False
        self.AddObserver("MiddleButtonPressEvent", self._middle_press)
        self.AddObserver("MiddleButtonReleaseEvent", self._middle_release)
        self.AddObserver("RightButtonPressEvent", self._right_press)
        self.AddObserver("RightButtonReleaseEvent", self._right_release)
        self.AddObserver("LeftButtonPressEvent", self._left_press)
        self.AddObserver("LeftButtonReleaseEvent", self._left_release)
        self.AddObserver("MouseMoveEvent", self._mouse_move)
        self.AddObserver("MouseWheelForwardEvent", self._wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self._wheel_backward)
        self.AddObserver("KeyPressEvent", self._key_press)
        self.AddObserver("KeyReleaseEvent", self._key_release)

    def _left_press(self, _obj, _event) -> None:
        lathe = getattr(self, "_lathe_callback", None)
        if lathe is not None and lathe("press", self.GetInteractor().GetEventPosition()):
            return
        self._wheel_timer.stop()
        self._wheel_steps = 0
        surface_pick = getattr(self, "_surface_pick_callback", None)
        if surface_pick is not None and surface_pick(self.GetInteractor().GetEventPosition()):
            self._consume_left_release = True
            return
        if (
            self.GetInteractor().GetShiftKey()
            and not self.GetInteractor().GetControlKey()
        ):
            self._measurement_callback("click", self.GetInteractor().GetEventPosition())
            self._consume_left_release = True
            return
        if self.GetInteractor().GetControlKey():
            self._crop_down = True
            self._widget.setCursor(Qt.CursorShape.CrossCursor)
            self._crop_callback("start", self.GetInteractor().GetEventPosition())
        elif self._crop_callback("left_press", self.GetInteractor().GetEventPosition()):
            self._crop_edit_down = True
            self._consume_left_release = True
            return
        else:
            self._crop_callback("navigation", None)
            x, y = self.GetInteractor().GetEventPosition()
            self.FindPokedRenderer(x, y)
            if self.GetCurrentRenderer() is None:
                return
            self._left_pan_down = True
            self._widget.setCursor(Qt.CursorShape.SizeAllCursor)
            self._set_pan_depth_at_cursor()
            self.StartPan()

    def _left_release(self, _obj, _event) -> None:
        lathe = getattr(self, "_lathe_callback", None)
        if lathe is not None and lathe("release", self.GetInteractor().GetEventPosition()):
            return
        if self._consume_left_release:
            self._consume_left_release = False
            if self._crop_edit_down:
                self._crop_edit_down = False
                self._crop_callback("left_release", self.GetInteractor().GetEventPosition())
            return
        if self._crop_down:
            self._crop_down = False
            self._crop_callback("finish", self.GetInteractor().GetEventPosition())
            self._widget.setCursor(Qt.CursorShape.ArrowCursor)
        elif self._left_pan_down:
            self._left_pan_down = False
            self.EndPan()
            self._pan_depth = None
            self._widget.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.OnLeftButtonUp()

    def _middle_press(self, _obj, _event) -> None:
        self._crop_callback("navigation", None)
        x, y = self.GetInteractor().GetEventPosition()
        self.FindPokedRenderer(x, y)
        if self.GetCurrentRenderer() is None:
            return
        if self.GetInteractor().GetShiftKey():
            self._middle_mode = "pan"
            self._widget.setCursor(Qt.CursorShape.SizeAllCursor)
            self._set_pan_depth_at_cursor()
            self.StartPan()
        else:
            self._middle_mode = "rotate"
            self._widget.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._set_rotation_center_at_cursor()
            self.StartRotate()

    def _middle_release(self, _obj, _event) -> None:
        if self._middle_mode == "pan":
            self.EndPan()
        elif self._middle_mode == "rotate":
            self.EndRotate()
        self._middle_mode = None
        self._rotation_center = None
        self._pan_depth = None
        self._widget.setCursor(Qt.CursorShape.ArrowCursor)

    def _right_press(self, _obj, _event) -> None:
        if self._crop_callback("right_press", self.GetInteractor().GetEventPosition()):
            self._crop_right_consumed = True
            return
        self._right_down = True
        self._widget.setCursor(Qt.CursorShape.CrossCursor)
        self._loupe_callback(True)

    def _right_release(self, _obj, _event) -> None:
        if self._crop_right_consumed:
            self._crop_right_consumed = False
            return
        self._right_down = False
        self._loupe_callback(False)
        self._widget.setCursor(Qt.CursorShape.ArrowCursor)

    def _mouse_move(self, _obj, _event) -> None:
        lathe = getattr(self, "_lathe_callback", None)
        if lathe is not None and lathe("move", self.GetInteractor().GetEventPosition()):
            return
        if self._crop_down:
            self._crop_callback("move", self.GetInteractor().GetEventPosition())
        elif self._crop_edit_down:
            self._crop_callback("edit_move", self.GetInteractor().GetEventPosition())
        elif self._right_down:
            self._loupe_callback(True)
        elif self.GetInteractor().GetShiftKey() and self._middle_mode is None:
            self._measurement_callback("move", self.GetInteractor().GetEventPosition())
        else:
            self._measurement_callback("hide", None)
            # Continue VTK's active rotate/pan/dolly interaction.
            interactor = self.GetInteractor()
            if (
                (self.invert_horizontal or self.invert_vertical)
                and (self._middle_mode in ("rotate", "pan") or self._left_pan_down)
            ):
                current_x, current_y = interactor.GetEventPosition()
                last_x, last_y = interactor.GetLastEventPosition()
                mapped_x = 2 * last_x - current_x if self.invert_horizontal else current_x
                mapped_y = 2 * last_y - current_y if self.invert_vertical else current_y
                interactor.SetEventPosition(mapped_x, mapped_y)
                if self._middle_mode == "rotate":
                    self._rotate_about_cursor_anchor()
                elif self._middle_mode == "pan" or self._left_pan_down:
                    self._pan_at_cursor_depth()
                else:
                    self.OnMouseMove()
                interactor.SetEventPosition(current_x, current_y)
            elif self._middle_mode == "rotate":
                self._rotate_about_cursor_anchor()
            elif self._middle_mode == "pan" or self._left_pan_down:
                self._pan_at_cursor_depth()
            else:
                self.OnMouseMove()
            self._measurement_callback("refresh", None)

    @staticmethod
    def _display_to_world(renderer: vtkRenderer, x: float, y: float, depth: float) -> np.ndarray:
        renderer.SetDisplayPoint(x, y, depth)
        renderer.DisplayToWorld()
        world = np.asarray(renderer.GetWorldPoint(), dtype=float)
        if abs(world[3]) > 1e-12:
            world = world / world[3]
        return world[:3]

    def _set_rotation_center_at_cursor(self) -> None:
        """Use the visible mesh beneath the pointer as the middle-drag orbit pivot."""
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        self.FindPokedRenderer(x, y)
        renderer = self.GetCurrentRenderer()
        if renderer is None:
            return
        picked = self._zoom_picker.Pick(x, y, 0.0, renderer)
        anchor = np.asarray(self._zoom_picker.GetPickPosition(), dtype=float) if picked else None
        if anchor is None or anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            bounds = np.asarray(renderer.ComputeVisiblePropBounds(), dtype=float)
            valid_bounds = (
                bounds.shape == (6,)
                and np.all(np.isfinite(bounds))
                and np.all(bounds[[1, 3, 5]] >= bounds[[0, 2, 4]])
            )
            anchor = (
                (bounds[[0, 2, 4]] + bounds[[1, 3, 5]]) * 0.5
                if valid_bounds
                else np.asarray(renderer.GetActiveCamera().GetFocalPoint(), dtype=float)
            )
        self._rotation_center = anchor
        self.SetCenterOfRotation(*anchor)

    def _set_pan_depth_at_cursor(self) -> None:
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        renderer = self.GetCurrentRenderer()
        if renderer is None:
            self._pan_depth = None
            return
        picked = self._zoom_picker.Pick(x, y, 0.0, renderer)
        anchor = (
            np.asarray(self._zoom_picker.GetPickPosition(), dtype=float)
            if picked
            else np.asarray(renderer.GetActiveCamera().GetFocalPoint(), dtype=float)
        )
        renderer.SetWorldPoint(*anchor, 1.0)
        renderer.WorldToDisplay()
        depth = float(renderer.GetDisplayPoint()[2])
        self._pan_depth = depth if np.isfinite(depth) else None

    def _pan_at_cursor_depth(self) -> None:
        interactor = self.GetInteractor()
        renderer = self.GetCurrentRenderer()
        if renderer is None:
            return
        depth = self._pan_depth
        if depth is None:
            self._set_pan_depth_at_cursor()
            depth = self._pan_depth
        if depth is None:
            return
        x, y = interactor.GetEventPosition()
        last_x, last_y = interactor.GetLastEventPosition()
        previous_world = self._display_to_world(renderer, last_x, last_y, depth)
        current_world = self._display_to_world(renderer, x, y, depth)
        motion = previous_world - current_world
        if not np.all(np.isfinite(motion)):
            return
        camera = renderer.GetActiveCamera()
        camera.SetPosition(*(np.asarray(camera.GetPosition(), dtype=float) + motion))
        camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint(), dtype=float) + motion))
        renderer.ResetCameraClippingRange()
        interactor.Render()

    @staticmethod
    def _rotate_vector(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 1e-12:
            return vector.copy()
        unit_axis = axis / axis_length
        radians = np.deg2rad(degrees)
        return (
            vector * np.cos(radians)
            + np.cross(unit_axis, vector) * np.sin(radians)
            + unit_axis * float(unit_axis @ vector) * (1.0 - np.cos(radians))
        )

    def _rotate_about_cursor_anchor(self) -> None:
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        last_x, last_y = interactor.GetLastEventPosition()
        self.FindPokedRenderer(x, y)
        renderer = self.GetCurrentRenderer()
        anchor = self._rotation_center
        if renderer is None or anchor is None:
            return
        width, height = renderer.GetRenderWindow().GetSize()
        if width <= 0 or height <= 0:
            return
        motion = self.GetMotionFactor()
        azimuth = (x - last_x) * (-20.0 / width) * motion
        elevation = (y - last_y) * (-20.0 / height) * motion
        camera = renderer.GetActiveCamera()
        position = np.asarray(camera.GetPosition(), dtype=float)
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        view_up = np.asarray(camera.GetViewUp(), dtype=float)

        position = anchor + self._rotate_vector(position - anchor, view_up, azimuth)
        focal = anchor + self._rotate_vector(focal - anchor, view_up, azimuth)
        view_up = self._rotate_vector(view_up, view_up, azimuth)

        direction = focal - position
        elevation_axis = np.cross(view_up, direction)
        position = anchor + self._rotate_vector(position - anchor, elevation_axis, elevation)
        focal = anchor + self._rotate_vector(focal - anchor, elevation_axis, elevation)
        view_up = self._rotate_vector(view_up, elevation_axis, elevation)

        if np.all(np.isfinite(position)) and np.all(np.isfinite(focal)):
            camera.SetPosition(*position)
            camera.SetFocalPoint(*focal)
            camera.SetViewUp(*view_up)
            camera.OrthogonalizeViewUp()
            renderer.ResetCameraClippingRange()
            interactor.Render()

    def _zoom_at_cursor(self, factor: float, movement_scale: float = 1.) -> None:
        callback = getattr(self, "_manual_zoom_callback", None)
        if callback is not None:
            callback()
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        self.FindPokedRenderer(x, y)
        renderer = self.GetCurrentRenderer()
        if renderer is None:
            return
        camera = renderer.GetActiveCamera()

        renderer.ResetCameraClippingRange()
        bounds = np.asarray(renderer.ComputeVisiblePropBounds(), dtype=float)
        valid_bounds = (
            bounds.shape == (6,)
            and np.all(np.isfinite(bounds))
            and np.all(bounds[[1, 3, 5]] >= bounds[[0, 2, 4]])
        )
        picked = self._zoom_picker.Pick(x, y, 0.0, renderer)
        anchor = np.asarray(self._zoom_picker.GetPickPosition(), dtype=float) if picked else None
        if anchor is None or anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            plane_point = (
                (bounds[[0, 2, 4]] + bounds[[1, 3, 5]]) * 0.5
                if valid_bounds
                else np.asarray(camera.GetFocalPoint(), dtype=float)
            )
            near_point = self._display_to_world(renderer, x, y, 0.0)
            far_point = self._display_to_world(renderer, x, y, 1.0)
            cursor_ray = far_point - near_point
            plane_normal = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
            denominator = float(cursor_ray @ plane_normal)
            if abs(denominator) > 1e-12:
                amount = float((plane_point - near_point) @ plane_normal) / denominator
                anchor = near_point + cursor_ray * amount
            else:
                anchor = plane_point

        if camera.GetParallelProjection():
            camera.SetParallelScale(camera.GetParallelScale() / factor)
            renderer.SetWorldPoint(*anchor, 1.0)
            renderer.WorldToDisplay()
            anchor_depth = renderer.GetDisplayPoint()[2]
            cursor_world = self._display_to_world(renderer, x, y, anchor_depth)
            offset = anchor - cursor_world
            focal_offset = offset
        else:
            position = np.asarray(camera.GetPosition(), dtype=float)
            ray = anchor - position
            distance = float(np.linalg.norm(ray))
            if valid_bounds:
                diagonal = float(np.linalg.norm(bounds[[1, 3, 5]] - bounds[[0, 2, 4]]))
            else:
                diagonal = 0.0
            inspection = bool(getattr(self, "_calibrated_inspection", False))
            minimum_step = max(diagonal * (1e-6 if inspection else .0015), 1e-9 if inspection else 1e-6) * movement_scale
            view_direction = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
            view_direction /= max(float(np.linalg.norm(view_direction)), 1e-12)
            forward_depth = float(ray @ view_direction)
            if distance <= 1e-12 or forward_depth <= minimum_step:
                # Once the cursor surface reaches the camera, continue through it in the current
                # viewing direction. Following a point behind the camera would reverse the axis.
                ray = view_direction
                distance = 0.0
            else:
                ray /= distance
            proportional_step = distance * abs(1.0 - 1.0 / factor)
            direction = 1.0 if factor > 1.0 else -1.0
            offset = ray * direction * max(proportional_step, minimum_step)
            # Move the focal point only by the sideways part of the cursor-ray movement. The
            # forward part changes camera distance, preventing the fixed-distance stall while the
            # lateral part keeps the selected screen position anchored under the pointer.
            focal_offset = offset - view_direction * float(offset @ view_direction)
            safe_focal_distance = max(diagonal * (1e-6 if inspection else .002), 1e-8 if inspection else 1e-5)
            remaining_focal_distance = camera.GetDistance() - float(offset @ view_direction)
            if remaining_focal_distance < safe_focal_distance:
                # Carry the focal point forward only as much as needed to preserve camera
                # orientation. This prevents VTK from flipping the view when zoom passes through
                # a surface while retaining the minimum useful movement per wheel step.
                focal_offset += view_direction * (
                    safe_focal_distance - remaining_focal_distance
                )
        if np.all(np.isfinite(offset)):
            camera.SetPosition(*(np.asarray(camera.GetPosition(), dtype=float) + offset))
            camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint(), dtype=float) + focal_offset))
        renderer.ResetCameraClippingRange()
        interactor.Render()

    def _wheel_forward(self, _obj, _event) -> None:
        self._crop_callback("navigation", None)
        if self.GetInteractor().GetControlKey():
            self._queue_wheel("roll", -5.0 if not self.invert_horizontal else 5.0)
            return
        self._queue_wheel("zoom", 1.0 / 1.20 if self.invert_zoom else 1.20)

    def _wheel_backward(self, _obj, _event) -> None:
        self._crop_callback("navigation", None)
        if self.GetInteractor().GetControlKey():
            self._queue_wheel("roll", 5.0 if not self.invert_horizontal else -5.0)
            return
        self._queue_wheel("zoom", 1.20 if self.invert_zoom else 1.0 / 1.20)

    def _queue_wheel(self, kind: str, amount: float) -> None:
        command = (kind, amount)
        self._wheel_steps = min(4, self._wheel_steps + 4) if command == self._wheel_command else 4
        self._wheel_command = command
        if not self._wheel_timer.isActive():
            self._advance_wheel()
            if self._wheel_steps:
                self._wheel_timer.start()

    def _advance_wheel(self) -> None:
        if (not self._wheel_steps or getattr(self._widget.window(), "_closing", False)
                or getattr(self._widget.window(), "busy", False)):
            self._wheel_timer.stop()
            self._wheel_steps = 0
            return
        self._wheel_steps -= 1
        kind, amount = self._wheel_command
        if kind == "roll":
            self._roll_at_view_center(amount / 4)
        else:
            self._zoom_at_cursor(amount ** .25, movement_scale=.25)
        self._measurement_callback("refresh", None)
        if not self._wheel_steps:
            self._wheel_timer.stop()

    def _roll_at_view_center(self, degrees: float) -> None:
        interactor = self.GetInteractor()
        x, y = interactor.GetEventPosition()
        self.FindPokedRenderer(x, y)
        renderer = self.GetCurrentRenderer()
        if renderer is None:
            return
        camera = renderer.GetActiveCamera()
        camera.Roll(degrees)
        camera.OrthogonalizeViewUp()
        renderer.ResetCameraClippingRange()
        interactor.Render()

    def _key_press(self, _obj, _event) -> None:
        key = self.GetInteractor().GetKeySym().lower()
        if key in ("return", "enter") and self._crop_callback("confirm", None):
            return
        if key == "escape" and self._crop_callback("cancel", None):
            return
        if self._keyboard_nav_callback(
            key,
            bool(self.GetInteractor().GetControlKey()),
            bool(self.GetInteractor().GetShiftKey()),
        ):
            return
        self.OnKeyPress()

    def _key_release(self, _obj, _event) -> None:
        if self.GetInteractor().GetKeySym().lower() in ("shift_l", "shift_r", "shift"):
            self._measurement_callback("hide", None)
        self.OnKeyRelease()


def mesh_surface_area(
    points: np.ndarray,
    faces: np.ndarray,
    progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    mass_moment: np.ndarray | None = None,
) -> float:
    """Return mesh area in bounded batches so long runs can report progress."""
    face_count = len(faces)
    if not face_count:
        return 0.0
    batch_size = 200_000
    area = 0.0
    for start in range(0, face_count, batch_size):
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled
        stop = min(start + batch_size, face_count)
        batch = faces[start:stop]
        a = points[batch[:, 0]]
        b = points[batch[:, 1]]
        c = points[batch[:, 2]]
        triangle_areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        area += float(triangle_areas.sum())
        if mass_moment is not None:
            centroids = (a.astype(np.float64)+b+c)/3.
            mass_moment += np.sum(centroids*triangle_areas[:,None],axis=0)
        if progress is not None:
            progress(stop, face_count)
    return area


def source_geometry_measurements(points, faces, progress=None, cancel_event=None):
    """Measure one immutable mesh revision using the existing CPU arithmetic."""
    bounds = points.max(axis=0) - points.min(axis=0)
    mass_moment = np.zeros(3)
    area = mesh_surface_area(
        points, faces, cancel_event=cancel_event, mass_moment=mass_moment,
        progress=(lambda done, total: progress("Measuring source geometry", done, total))
        if progress is not None else None,
    )
    center = mass_moment/area if area>0 else np.mean(points,axis=0)
    return {"bounds": bounds, "area": area, "center_of_mass": center}


def prepare_viewport_poly(poly, progress=None):
    """Prepare the existing display normals without touching editable geometry."""
    normals = vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    if progress is not None:
        total = poly.GetNumberOfPoints()
        progress("Preparing viewport normals", 0, total)
        normals.AddObserver("ProgressEvent", lambda source, _event: progress(
            "Preparing viewport normals", round(source.GetProgress() * total), total))
    normals.Update()
    prepared = vtkPolyData()
    prepared.DeepCopy(normals.GetOutput())
    if progress is not None:
        progress("Preparing viewport normals", total, total)
    return prepared


def readonly_cpu_edge_counts(faces, progress=None, cancel_event=None):
    """Count edges without changing faces, using exact integer keys when representable."""
    face_count = len(faces)
    total = max(4 * face_count, 1)
    packed = faces.dtype.kind in "iu" and (
        not faces.size or (int(faces.min()) >= 0 and int(faces.max()) <= 0xFFFFFFFF)
    )
    if packed:
        keys = np.empty(3 * face_count, dtype=np.uint64)
        for edge_index, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            for start in range(0, face_count, 262_144):
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled
                stop = min(start + 262_144, face_count)
                chunk = faces[start:stop]
                low = np.minimum(chunk[:, a], chunk[:, b]).astype(np.uint64)
                high = np.maximum(chunk[:, a], chunk[:, b]).astype(np.uint64)
                keys[edge_index * face_count + start:edge_index * face_count + stop] = (
                    (low << np.uint64(32)) | high
                )
                if progress is not None:
                    progress("Preparing topology edges", edge_index * face_count + stop, total)
        if progress is not None:
            progress("Sorting topology edges", 3 * face_count, total)
        _, counts = np.unique(keys, return_counts=True)
    else:
        # Preserve the former behavior for identifiers outside the packing range.
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges.sort(axis=1)
        if progress is not None:
            progress("Sorting topology edges", 3 * face_count, total)
        _, counts = np.unique(edges, axis=0, return_counts=True)
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled
    return counts


def mesh_quality_metrics(
    source_points: np.ndarray,
    source_faces: np.ndarray,
    output_points: np.ndarray,
    output_faces: np.ndarray,
    sample_count: int = 20_000,
    progress: Callable[[str, int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    source_measurements: dict | None = None,
    gpu_preference: str | None = None,
) -> dict[str, float]:
    """Calculate compact, deterministic quality indicators for the Stats panel."""
    if not len(source_points) or not len(output_points):
        return {}
    from geometry_workload import record
    record("CPU", source_points, source_faces)
    record("CPU", output_points, output_faces)
    rng = np.random.default_rng(20261005)
    sample_size = min(sample_count, len(source_points))
    samples = source_points[rng.choice(len(source_points), sample_size, replace=False)]
    if progress is not None:
        progress("Preparing result analysis", 0, 1)
    output_poly = arrays_polydata(output_points, output_faces)
    locator = vtkStaticPointLocator()
    locator.SetDataSet(output_poly)
    locator.BuildLocator()
    if progress is not None:
        progress("Preparing result analysis", 1, 1)
        progress("Analyzing results", 0, sample_size)
    vtk_output_points = output_poly.GetPoints()
    distances = np.empty(sample_size, dtype=np.float64)
    progress_interval = max(1, sample_size // 100)
    for index, sample in enumerate(samples):
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled
        closest = vtk_output_points.GetPoint(locator.FindClosestPoint(sample))
        distances[index] = np.linalg.norm(sample - closest)
        if progress is not None and (
            (index + 1) % progress_interval == 0 or index + 1 == sample_size
        ):
            progress("Analyzing results", index + 1, sample_size)
    area_total = len(source_faces) + len(output_faces)
    if progress is not None:
        progress("Measuring surface area", 0, max(area_total, 1))
    source_bounds = (source_measurements["bounds"] if source_measurements is not None
                     else source_points.max(axis=0) - source_points.min(axis=0))
    output_bounds = output_points.max(axis=0) - output_points.min(axis=0)
    diagonal = max(float(np.linalg.norm(source_bounds)), 1e-12)
    source_area = max(
        source_measurements["area"] if source_measurements is not None else mesh_surface_area(
            source_points,
            source_faces,
            progress=(
                lambda completed, total: progress("Measuring surface area", completed, area_total)
                if progress is not None
                else None
            ),
            cancel_event=cancel_event,
        ),
        1e-12,
    )
    output_area = mesh_surface_area(
        output_points,
        output_faces,
        progress=(
            lambda completed, total: progress(
                "Measuring surface area", len(source_faces) + completed, area_total
            )
            if progress is not None
            else None
        ),
        cancel_event=cancel_event,
    )
    topology_total = max(4 * len(output_faces), 1)
    if progress is not None:
        progress("Preparing topology edges", 0, topology_total)
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled
    edge_counts = None
    if gpu_preference is not None:
        from depth_adaptive_qem import readonly_gpu_edge_counts
        edge_counts = readonly_gpu_edge_counts(output_faces, gpu_preference)
        if edge_counts is not None:
            record("GPU", np.empty((0, 3), dtype=output_points.dtype), output_faces)
    if edge_counts is None:
        edge_counts = readonly_cpu_edge_counts(output_faces, progress, cancel_event)
    if progress is not None:
        progress("Checking mesh topology", topology_total, topology_total)
    return {
        "reduction_percent": 100.0 * (1.0 - len(output_faces) / len(source_faces)),
        "rms_percent": 100.0 * float(np.sqrt(np.mean(distances * distances))) / diagonal,
        "p95_percent": 100.0 * float(np.percentile(distances, 95)) / diagonal,
        "drift_percent": 100.0 * float(np.max(np.abs(output_bounds - source_bounds))) / diagonal,
        "area_percent": 100.0 * abs(output_area / source_area - 1.0),
        "nonmanifold_percent": 100.0 * float(np.count_nonzero(edge_counts > 2)) / len(output_faces),
    }


def mesh_quality_baseline(
    points: np.ndarray,
    faces: np.ndarray,
    estimated_triangles: int,
    algorithm: str = "Fast QEM",
    quality: str = "Balanced",
) -> dict[str, float]:
    """Return a fast projection for the requested settings without scanning the mesh."""
    if not len(points) or not len(faces):
        return {}
    reduction = max(0.0, min(1.0, 1.0 - estimated_triangles / len(faces)))
    profile = {
        "Fast QEM": (0.22, 0.55, 0.030, 0.18, 0.040),
        "Depth Adaptive QEM": (0.16, 0.40, 0.020, 0.12, 0.030),
        "Depth + Density Adaptive QEM": (0.13, 0.30, 0.001, 0.01, 0.005),
        "Density balanced": (0.28, 0.70, 0.050, 0.22, 0.050),
        "Shape preserving": (0.19, 0.48, 0.025, 0.15, 0.035),
        "Preserve topology": (0.24, 0.60, 0.030, 0.18, 0.010),
    }
    if algorithm == DEPTH_DENSITY_ADAPTIVE_LABEL:
        profile_key = "Depth + Density Adaptive QEM"
    elif algorithm.startswith("Depth Adaptive QEM"):
        profile_key = "Depth Adaptive QEM"
    else:
        profile_key = algorithm
    rms, p95, drift, area, topology = profile.get(profile_key, profile["Fast QEM"])
    quality_factor = {"Draft": 1.20, "Balanced": 1.0, "Fine": 0.82}.get(quality, 1.0)
    severity = reduction * reduction * quality_factor
    return {
        "reduction_percent": 100.0 * reduction,
        "rms_percent": rms * severity,
        "p95_percent": p95 * severity,
        "drift_percent": drift * severity,
        "area_percent": area * severity,
        "nonmanifold_percent": topology * severity,
    }


def smart_target(points: np.ndarray, faces: np.ndarray, quality: str, measurements=None) -> int:
    bounds = (measurements["bounds"] if measurements is not None
              else points.max(axis=0) - points.min(axis=0))
    diagonal = float(np.linalg.norm(bounds))
    if diagonal <= 0:
        return min(50_000, len(faces))
    desired_edge = diagonal / SMART_DIVISORS[quality]
    ideal_triangle_area = (math.sqrt(3.0) / 4.0) * desired_edge * desired_edge
    area = measurements["area"] if measurements is not None else mesh_surface_area(points, faces)
    estimate = round(area / ideal_triangle_area)
    # The surface-area estimate remains the driver, while the quality bands
    # prevent a noisy scan from making every preset nearly full density.
    quality_cap = {"Draft": 0.15, "Balanced": 0.50, "Fine": 0.75}[quality]
    upper = min(750_000, max(1_000, round(len(faces) * quality_cap)))
    return min(upper, max(10_000, estimate))


def density_color_stops(
    base_color: tuple[int, int, int] | None = None,
    dense_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Build the density ramp from the active viewport color.

    The sparse end follows the viewport hue, then rapidly returns to the
    brighter density ramp used by the original display.
    """
    base = np.asarray(base_color or (11, 27, 52), dtype=np.float64)
    if dense_color is not None:
        dense = np.asarray(dense_color, dtype=np.float64)
        sparse_qcolor = QColor(*[int(component) for component in base])
        dense_qcolor = QColor(*[int(component) for component in dense])
        sparse_hue, sparse_saturation, sparse_value, _ = sparse_qcolor.getHsvF()
        dense_hue, dense_saturation, dense_value, _ = dense_qcolor.getHsvF()
        weights = np.linspace(0.0, 1.0, 9)
        if sparse_hue >= 0.0 and dense_hue >= 0.0:
            # Preserve the visual direction chosen by the endpoints. Blue to red
            # therefore crosses cyan, green, yellow, and orange instead of fading
            # through purple as a shortest-path hue interpolation would.
            colors = []
            for weight in weights:
                color = QColor.fromHsvF(
                    sparse_hue * (1.0 - weight) + dense_hue * weight,
                    sparse_saturation * (1.0 - weight) + dense_saturation * weight,
                    sparse_value * (1.0 - weight) + dense_value * weight,
                )
                colors.append((color.red(), color.green(), color.blue()))
            return np.asarray(colors, dtype=np.float64)
        return np.asarray(
            [base * (1.0 - weight) + dense * weight for weight in weights],
            dtype=np.float64,
        )
    sparse = np.minimum(255.0, base * 1.55 + 8.0)
    sparse_mid = np.minimum(255.0, base * 2.20 + 18.0)
    return np.asarray(
        [sparse, sparse_mid, [35, 190, 226], [167, 227, 75], [250, 204, 21], [226, 55, 48]],
        dtype=np.float64,
    )


def _cpu_density_field(
    points: np.ndarray,
    faces: np.ndarray,
    cancel_event: threading.Event | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the reference display density without modifying geometry."""
    point_count = len(points)
    incident_count = np.bincount(faces.ravel(), minlength=point_count).astype(np.float32)
    incident_area = np.zeros(point_count, dtype=np.float32)
    chunk_size = 250_000
    for start in range(0, len(faces), chunk_size):
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled
        chunk = faces[start:start + chunk_size]
        a = points[chunk[:, 0]]
        b = points[chunk[:, 1]]
        c = points[chunk[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        incident_area += np.bincount(
            chunk.ravel(), weights=np.repeat(area, 3), minlength=point_count
        ).astype(np.float32, copy=False)
        if progress is not None:
            progress("Measuring triangle density", start + len(chunk), len(faces))

    valid = (incident_count > 0) & (incident_area > 0)
    log_density = np.zeros(point_count, dtype=np.float32)
    # Vertices per incident surface area is the local sampling density. Work in log space so
    # scan-density outliers remain visible without allowing them to dominate the color range.
    log_density[valid] = np.log(
        np.maximum(incident_count[valid] / incident_area[valid], np.float32(1e-30))
    )

    # One-ring smoothing makes regional density visible at normal zoom while retaining
    # narrow oversampled bands. Each face contributes its other two vertices as neighbors.
    neighbor_sum = np.zeros(point_count, dtype=np.float32)
    neighbor_count = np.zeros(point_count, dtype=np.float32)
    for start in range(0, len(faces), chunk_size):
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled
        chunk = faces[start:start + chunk_size]
        face_mean = log_density[chunk].mean(axis=1)
        destinations = chunk.ravel()
        neighbor_sum += np.bincount(
            destinations, weights=np.repeat(face_mean, 3), minlength=point_count
        ).astype(np.float32, copy=False)
        neighbor_count += np.bincount(destinations, minlength=point_count).astype(
            np.float32, copy=False
        )
        if progress is not None:
            progress("Smoothing density regions", start + len(chunk), len(faces))
    smoothed = np.divide(
        log_density + neighbor_sum,
        1.0 + neighbor_count,
        out=log_density.copy(),
        where=(1.0 + neighbor_count) > 0,
    )

    return smoothed, valid


def point_density_colors(
    points: np.ndarray,
    faces: np.ndarray,
    cancel_event: threading.Event | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    base_color: tuple[int, int, int] | None = None,
    dense_color: tuple[int, int, int] | None = None,
    gpu_preference: str | None = None,
) -> np.ndarray:
    """Map read-only local density to the active palette, with optional GPU measurement."""
    field = None
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled
    if gpu_preference is not None:
        from depth_adaptive_qem import readonly_gpu_density_field
        field = readonly_gpu_density_field(points, faces, gpu_preference, progress, cancel_event)
        if field is not None:
            from geometry_workload import record
            record("GPU", points, faces)
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled
    if field is None:
        field = _cpu_density_field(points, faces, cancel_event, progress)
    from geometry_workload import record
    record("CPU", points, faces)
    smoothed, valid = field
    normalized = np.zeros(len(points), dtype=np.float64)
    if np.any(valid):
        values = smoothed[valid]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = max(1.4826 * mad, 1e-12)
        robust_z = np.clip((smoothed - median) / robust_sigma, -8.0, 8.0)
        # 1.702*z closely approximates a normal CDF without adding a heavy statistics dependency.
        bell_curve = 1.0 / (1.0 + np.exp(-1.702 * robust_z))
        # Retain rank contrast for multimodal scans while allowing statistical outliers to flare red.
        low, high = np.percentile(values, (0.5, 99.5))
        rank_contrast = np.clip((smoothed - low) / max(high - low, 1e-12), 0.0, 1.0)
        normalized = np.maximum(bell_curve, 0.80 * rank_contrast)
        normalized[~valid] = 0.0

    if progress is not None:
        progress("Coloring density regions", 0, 1)
    stops = density_color_stops(base_color, dense_color)
    scaled = normalized * (len(stops) - 1)
    lower = np.minimum(scaled.astype(np.int32), len(stops) - 2)
    blend = (scaled - lower)[:, None]
    colors = np.ascontiguousarray(
        np.rint(stops[lower] * (1.0 - blend) + stops[lower + 1] * blend).astype(np.uint8)
    )
    if progress is not None:
        progress("Coloring density regions", 1, 1)
    return colors


def point_density_cell_colors(
    point_colors: np.ndarray,
    faces: np.ndarray,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    """Convert point density colors to stable per-triangle colors for surface rendering."""
    colors = np.empty((len(faces), 3), dtype=np.uint8)
    chunk_size = 250_000
    for start in range(0, len(faces), chunk_size):
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled
        chunk = faces[start:start + chunk_size]
        colors[start:start + len(chunk)] = np.rint(
            point_colors[chunk].astype(np.float32).mean(axis=1)
        ).astype(np.uint8)
    return colors


def density_balance_arrays(
    points: np.ndarray, faces: np.ndarray, target: int, aggressiveness: float,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use conservative QEM so redundant flat sampling collapses before geometric detail."""
    # QEM already incorporates local surface curvature in its accumulated plane error. The old
    # voxel-average prepass moved vertices before measuring that error and visibly distorted scans.
    quality_aggressiveness = min(5.0, max(0.0, aggressiveness * 0.70))
    return simplify_with_progress(
        points,
        faces,
        target_count=target,
        agg=quality_aggressiveness,
        verbose=False,
        preserve_border=True,
        progress=progress,
    )


def simplify_arrays(
    points: np.ndarray,
    faces: np.ndarray,
    target: int,
    algorithm: str,
    aggressiveness: float = 7.0,
    gpu_preference: str = "system",
    progress: Callable[[str, int, int], None] | None = None,
    probe_statistics: dict | None = None,
    preserve_closed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    from closed_reduction import validate_closed_reduction
    output_points, output_faces = _simplify_arrays_backend(
        points, faces, target, algorithm, aggressiveness, gpu_preference, progress, probe_statistics,
    )
    if preserve_closed:
        output_points, output_faces = validate_closed_reduction(output_points, output_faces, progress)
    return output_points, output_faces


def _simplify_arrays_backend(
    points: np.ndarray,
    faces: np.ndarray,
    target: int,
    algorithm: str,
    aggressiveness: float = 7.0,
    gpu_preference: str = "system",
    progress: Callable[[str, int, int], None] | None = None,
    probe_statistics: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from geometry_workload import record
    record("CPU", points, faces)
    reduction = 1.0 - target / len(faces)
    if algorithm == DEPTH_DENSITY_ADAPTIVE_LABEL:
        output_points, output_faces, _report = simplify_depth_density_adaptive_qem(
            points, faces, target, aggressiveness, gpu_preference, progress=progress,
            source_statistics=probe_statistics,
        )
        if _report.get("backend") == "CUDA":
            record("GPU", points, faces)
        return output_points, output_faces
    if algorithm == DEPTH_ADAPTIVE_LABEL or algorithm == "Depth Adaptive QEM":
        output_points, output_faces, _report = simplify_depth_adaptive_qem(
            points,
            faces,
            target,
            max(8.0, aggressiveness),
            gpu_preference,
            progress=progress,
            source_statistics=probe_statistics,
        )
        if _report.get("backend") == "CUDA":
            record("GPU", points, faces)
        return output_points, output_faces
    if algorithm == "Fast QEM":
        return simplify_with_progress(
            points, faces, target_reduction=reduction, agg=aggressiveness, verbose=False,
            progress=progress,
        )
    if algorithm == "Density balanced":
        return density_balance_arrays(points, faces, target, aggressiveness, progress=progress)
    source = arrays_polydata(points, faces)
    if algorithm == "Shape preserving":
        decimator = vtkQuadricDecimation()
        decimator.SetInputData(source)
        decimator.SetTargetReduction(reduction)
        decimator.VolumePreservationOn()
    elif algorithm == "Preserve topology":
        decimator = vtkDecimatePro()
        decimator.SetInputData(source)
        decimator.SetTargetReduction(reduction)
        decimator.PreserveTopologyOn()
        decimator.SplittingOff()
        decimator.BoundaryVertexDeletionOff()
    else:
        raise ValueError(f"Unknown optimization algorithm: {algorithm}")
    if progress is not None:
        decimator.AddObserver("ProgressEvent", lambda source, _event: progress(
            "Reducing mesh topology", round(source.GetProgress() * len(faces)), len(faces)
        ))
    decimator.Update()
    return polydata_arrays(decimator.GetOutput())


def _write_process_progress(
    progress_path: str | None, stage: str, completed: int = 0, total: int = 0
) -> None:
    if not progress_path:
        return
    try:
        destination = Path(progress_path)
        pending = destination.with_suffix(destination.suffix + ".pending")
        pending.write_text(
            json.dumps({"stage": stage, "completed": completed, "total": total}),
            encoding="utf-8",
        )
        os.replace(pending, destination)
    except OSError:
        pass


def simplify_process_job(
    points,
    faces,
    target: int,
    algorithm: str,
    gpu_preference: str = "system",
    progress_path: str | None = None,
    source_measurements: dict | None = None,
    probe_statistics: dict | None = None,
    preserve_closed: bool = False,
):
    """Run native simplification outside the UI process."""
    from geometry_workload import difference, snapshot
    workload_before = snapshot()
    try:
        report = lambda stage, completed, total: _write_process_progress(
            progress_path, stage, completed, total
        )
        p_out, f_out = simplify_arrays(
            points, faces, target, algorithm, gpu_preference=gpu_preference, progress=report,
            probe_statistics=probe_statistics, preserve_closed=preserve_closed,
        )
        metrics = mesh_quality_metrics(
            points, faces, p_out, f_out, progress=report, source_measurements=source_measurements,
            gpu_preference=gpu_preference if is_gpu_guided_algorithm(algorithm) else None,
        )
        report("Preparing optimized mesh", 1, 1)
        metrics["closed_output"] = preserve_closed
        metrics["geometry_workload"] = difference(snapshot(), workload_before)
        return p_out, f_out, metrics, None
    except Exception:  # noqa: BLE001 - worker boundary returns tracebacks to the UI process
        return None, None, None, traceback.format_exc()


def depth_adaptive_progress_text(gpu_preference: str = "system") -> str:
    if first_time_gpu_setup_needed(gpu_preference):
        return "First-time GPU setup: preparing and caching kernels (one-time setup)"
    return "Analyzing surface"


def cli_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="MeshMillCLI",
        description="Optimize mesh geometry or run a headless mesh workflow.",
        epilog=(
            "Examples:\n"
            "  MeshMillCLI input.stl --quality balanced\n"
            "  MeshMillCLI input.stl --target 150000 --output reduced.stl\n"
            "  MeshMillCLI input.stl --algorithm depth-adaptive --gpu 00000000:01:00.0\n\n"
            "Without --target, --quality, or --preset, MeshMill derives a Balanced target from "
            "the input mesh."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("input", type=Path, help="Input STL, COLLADA, GLB, OBJ or PLY mesh")
    parser.add_argument("-o", "--output", type=Path, help="Output STL, COLLADA, GLB, OBJ or PLY filename")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("-t", "--target", type=int, help="Target triangle count")
    choice.add_argument(
        "-q", "--quality", choices=("draft", "balanced", "fine"),
        help="Derive the target from mesh size and detail level (default: balanced)",
    )
    choice.add_argument(
        "-p", "--preset", choices=sorted(PRESETS),
        help="Use a fixed legacy target: light=50k, balanced=200k, detailed=400k",
    )
    parser.add_argument(
        "-a", "--aggressiveness", type=float, default=7.0,
        help="0 favors quality, 7 is balanced, 10 favors speed (default: 7)",
    )
    parser.add_argument(
        "--algorithm",
        choices=["auto", *sorted(CLI_ALGORITHMS)],
        default="auto",
        help=(
            "reduction algorithm (default: auto selects Depth + Density Adaptive QEM "
            "when GPU support is available, otherwise Density balanced)"
        ),
    )
    parser.add_argument(
        "--gpu", default="system", metavar="PCI_BUS_ID",
        help="CUDA GPU PCI bus ID for GPU-guided algorithms, or system (default: system)",
    )
    parser.add_argument("--workflow", type=Path, help="Run a version 1 JSON workflow: Optimize, Solidify, Cast and format conversion")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args(argv)

    source = args.input.expanduser().resolve()
    if not source.is_file():
        parser.error(f"input file does not exist: {source}")
    if source.suffix.lower() not in {'.stl', '.dae', '.glb', '.obj', '.ply'}:
        parser.error("input must be STL, COLLADA, GLB, OBJ or PLY")
    if not 0 <= args.aggressiveness <= 10:
        parser.error("aggressiveness must be between 0 and 10")

    if args.workflow is not None:
        if (args.target is not None or args.quality is not None or args.preset is not None
                or args.algorithm != 'auto' or args.gpu != 'system' or args.aggressiveness != 7):
            parser.error("workflow settings belong in the workflow file")
        from mesh_workflow import load_workflow, run_workflow
        try:
            run_workflow(source, args.output or source.with_name(source.stem + "_processed.stl"),
                         load_workflow(args.workflow), overwrite=args.overwrite)
        except (ValueError, OSError, RuntimeError, MemoryError, ImportError) as error:
            parser.exit(1, f"MeshMill: {error}\n")
        return 0

    print(f"Loading: {source}", flush=True)
    poly, points, faces = load_stl(source)
    original_count = len(faces)
    original_size = np.asarray(bounds_size(poly.GetBounds()))
    print(f"Original: {original_count:,} triangles", flush=True)
    print(f"Dimensions: {fmt_size(tuple(original_size))}", flush=True)
    quality = (args.quality or "balanced").title()
    target = (
        args.target
        if args.target is not None
        else PRESETS[args.preset]
        if args.preset is not None
        else min(smart_target(points, faces, quality), max(1, original_count - 1))
    )
    if target < 1_000:
        parser.error("target must be at least 1000 triangles")
    if target >= original_count:
        parser.error(f"target {target:,} is not smaller than the source ({original_count:,})")

    output = args.output
    if output is None:
        output = suggested_mesh_path(source, target)
    else:
        output = output.expanduser().resolve()
    if output == source:
        parser.error("output must differ from input")
    if output.suffix.lower() not in {'.stl', '.dae', '.glb', '.obj', '.ply'}:
        parser.error("output must be STL, COLLADA, GLB, OBJ or PLY")
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output} (use --overwrite to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)

    selected_algorithm = (
        default_algorithm(args.gpu) if args.algorithm == "auto" else CLI_ALGORITHMS[args.algorithm]
    )
    if is_gpu_guided_algorithm(selected_algorithm) and not gpu_backend_supported(args.gpu):
        parser.error(
            "the selected GPU-guided algorithm requires a compatible NVIDIA CUDA GPU and runtime"
        )
    print(
        f"Optimizing to about {target:,} triangles "
        f"(aggressiveness {args.aggressiveness:g})...",
        flush=True,
    )
    if selected_algorithm in (
        DEPTH_ADAPTIVE_LABEL,
        DEPTH_DENSITY_ADAPTIVE_LABEL,
    ):
        print(
            "First-time GPU setup: preparing and caching kernels (one-time setup)..."
            if first_time_gpu_setup_needed(args.gpu)
            else "Analyzing surface...",
            flush=True,
        )
    reduced_points, reduced_faces = simplify_arrays(
        points, faces, target, selected_algorithm, args.aggressiveness, args.gpu
    )
    reduced = arrays_polydata(reduced_points, reduced_faces)
    reduced_size = np.asarray(bounds_size(reduced.GetBounds()))
    drift = np.abs(reduced_size - original_size)
    if output.suffix.lower() == '.stl':
        save_stl(reduced, output)
    else:
        from mesh_export import write_polygon_mesh
        write_polygon_mesh(reduced_points, reduced_faces, output)
    print(f"Result: {len(reduced_faces):,} triangles", flush=True)
    print(f"Dimensions: {fmt_size(tuple(reduced_size))}", flush=True)
    print(f"Dimension drift: {fmt_size(tuple(drift))}", flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


def filter_preferences_sections(sections, query: str) -> int:
    """Filter setting rows while retaining their labels, fields, and related actions."""
    words = query.casefold().split()

    def widget_text(widgets):
        return " ".join(widget.text() for widget in widgets
                        if isinstance(widget, (QLabel, QCheckBox))).casefold()

    def layout_widgets(layout):
        result = []
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget() is not None:
                result.append(item.widget())
            elif item.layout() is not None:
                result.extend(layout_widgets(item.layout()))
        return result

    def collect(layout, groups, headers, names):
        if isinstance(layout, QFormLayout):
            for row in range(layout.rowCount()):
                widgets = []
                for role in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole):
                    item = layout.itemAt(row, role)
                    if item is not None:
                        widgets.extend([item.widget()] if item.widget() is not None else
                                       layout_widgets(item.layout()) if item.layout() is not None else [])
                if widgets:
                    groups.append(widgets)
                    label = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
                    names[id(widgets)] = widget_text([label.widget()]) if label and label.widget() else ""
        elif isinstance(layout, QGridLayout):
            grid_header, grid_rows = [], []
            for row in range(layout.rowCount()):
                widgets = []
                for index in range(layout.count()):
                    if layout.getItemPosition(index)[0] == row:
                        item = layout.itemAt(index)
                        widgets.extend([item.widget()] if item.widget() is not None else
                                       layout_widgets(item.layout()) if item.layout() is not None else [])
                if widgets:
                    if row == 0:
                        grid_header = widgets
                    else:
                        groups.append(widgets)
                        grid_rows.append(widgets)
                        # Data-only comparison cells are values, not preference names.
                        editable = any(not isinstance(widget, QLabel) for widget in widgets)
                        names[id(widgets)] = widget_text(widgets[:1]) if editable else ""
            if grid_header:
                headers.append((grid_header, grid_rows))
        elif isinstance(layout, QHBoxLayout):
            widgets = layout_widgets(layout)
            groups.append(widgets)
            names[id(widgets)] = ""
        else:
            for index in range(layout.count()):
                item = layout.itemAt(index)
                if item.layout() is not None:
                    collect(item.layout(), groups, headers, names)
                elif item.widget() is not None:
                    widgets = [item.widget()]
                    groups.append(widgets)
                    names[id(widgets)] = (widget_text(widgets) if isinstance(item.widget(), QCheckBox)
                                          else str(item.widget().property("preferenceSearchName") or "").casefold())

    visible_count = 0
    for frame in sections:
        if not hasattr(frame, "_preferences_search_rows"):
            groups, headers, names = [], [], {}
            collect(frame.layout(), groups, headers, names)
            heading = groups.pop(0)  # The section title is retained with matching rows.
            original_visibility = {widget: not widget.isHidden()
                                   for group in [heading, *groups, *[h[0] for h in headers]]
                                   for widget in group}
            frame._preferences_search_rows = heading, groups, headers, original_visibility, names
        heading, groups, headers, original_visibility, names = frame._preferences_search_rows
        context = widget_text(heading)
        any_visible = False
        for group in groups:
            searchable = context + " " + names[id(group)]
            matched = all(word in searchable for word in words)
            for widget in group:
                predicate = getattr(widget, "_preferences_search_allowed", None)
                allowed = predicate() if predicate is not None else original_visibility[widget]
                widget.setVisible(matched and allowed)
                any_visible |= matched and allowed
        for widget in heading:
            widget.setVisible(any_visible and original_visibility[widget])
        for header, rows in headers:
            show_header = any(not widget.isHidden() for row in rows for widget in row)
            for widget in header:
                widget.setVisible(show_header and original_visibility[widget])
        frame.setVisible(any_visible)
        visible_count += int(any_visible)
    return visible_count


def create_changelog_viewer(markdown: str, theme_name: str, parent=None) -> QTextBrowser:
    """Render release Markdown with explicit document and scrollbar theme colors."""
    theme = appearance_theme(theme_name)
    background, text = QColor(theme["field"]), QColor(theme["text"])
    dark = background.lightnessF() < 0.5
    link = QColor("#8CC5FF" if dark else "#1D4ED8")
    visited = QColor("#D8B4FE" if dark else "#6B21A8")
    viewer = QTextBrowser(parent)
    viewer.setObjectName("changelogViewer")
    viewer.setReadOnly(True)
    viewer.setOpenExternalLinks(True)
    viewer.document().setBaseUrl(QUrl(REPOSITORY_URL.rstrip("/") + "/blob/main/"))
    viewer.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    palette = viewer.palette()
    for role, color in ((QPalette.ColorRole.Base, background), (QPalette.ColorRole.Text, text),
                        (QPalette.ColorRole.Link, link), (QPalette.ColorRole.LinkVisited, visited),
                        (QPalette.ColorRole.Highlight, QColor("#1D4ED8")),
                        (QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))):
        palette.setColor(role, color)
    viewer.setPalette(palette)
    viewer.setStyleSheet(
        f"QTextBrowser#changelogViewer {{ background: {background.name()}; color: {text.name()}; }}"
        + scrollbar_stylesheet(theme)
    )
    viewer.document().setDefaultStyleSheet(
        f"body {{ color: {text.name()}; }} a {{ color: {link.name()}; text-decoration: underline; }}"
        f"pre, code {{ color: {text.name()}; background-color: {theme['surface']}; }}"
        f"td, th {{ border: 1px solid {theme['border']}; padding: 5px; }}"
    )
    viewer.setMarkdown(markdown)
    # Markdown imports may carry default blue anchors or explicit code backgrounds,
    # independent of the widget palette. Normalize colors while retaining formatting.
    ranges = []
    block = viewer.document().begin()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid():
                ranges.append((fragment.position(), fragment.length(), fragment.charFormat()))
            iterator += 1
        block = block.next()
    for position, length, original in ranges:
        cursor = QTextCursor(viewer.document())
        cursor.setPosition(position)
        cursor.setPosition(position + length, QTextCursor.MoveMode.KeepAnchor)
        formatting = QTextCharFormat()
        formatting.setForeground(link if original.isAnchor() else text)
        if original.isAnchor():
            formatting.setFontUnderline(True)
            href = QUrl(original.anchorHref())
            if href.isRelative():
                formatting.setAnchorHref(viewer.document().baseUrl().resolved(href).toString())
        if original.background().style() != Qt.BrushStyle.NoBrush:
            formatting.setBackground(QColor(theme["surface"]))
        cursor.mergeCharFormat(formatting)
    def theme_frames(frame):
        for child in frame.childFrames():
            if isinstance(child, QTextTable):
                formatting = child.format()
                formatting.setBorderBrush(QColor(theme["border"]))
                formatting.setBackground(background)
                child.setFormat(formatting)
            theme_frames(child)
    theme_frames(viewer.document().rootFrame())
    return viewer


class WindowTitleBar(QFrame):
    """A single caption row with OS-managed dragging and window controls."""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.window().isFullScreen():
            handle = self.window().windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            if hasattr(window, "_toggle_maximized"):
                window._toggle_maximized()
            elif window.minimumSize() != window.maximumSize():
                window.showNormal() if window.isMaximized() else window.showMaximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.settings = settings if settings is not None else QSettings("MeshMill", "MeshMill-FittedShell-Prototype")
        self.recent_files_enabled = self.settings.value("files/recent_enabled", False, type=bool)
        self.recent_files = self.settings.value("files/recent", [], type=list) if self.recent_files_enabled else []
        if not self.recent_files_enabled:
            self.settings.remove("files/recent")
        self.gpu_preference = str(
            self.settings.value("performance/gpu_preference", "system")
        )
        self.cuda_device_list = cuda_devices()
        detected_gpu_ids = {device["pci_bus_id"] for device in self.cuda_device_list}
        if self.gpu_preference != "system" and self.gpu_preference not in detected_gpu_ids:
            self.gpu_preference = "system"
        self.update_checks_enabled = self.settings.value(
            "updates/automatic", False, type=bool
        )
        self.auto_install_updates = self.update_checks_enabled and self.settings.value(
            "updates/auto_install", False, type=bool)
        self.auto_install_scope = str(self.settings.value("updates/auto_install_scope", "maintenance"))
        self._automatic_update_in_progress = False
        self._ready_auto_installer = None
        self.latest_release_version = ""
        self.latest_release_url = ""
        self.latest_release = {}
        self.update_check_status = "Not checked"
        self.update_future = None
        self.update_install_future = None
        self._update_prompt_requested = False
        self._pending_update_prompt = ""
        self._prompted_update_versions: set[str] = set()
        self.skipped_release_version = str(self.settings.value("updates/skipped_version", ""))
        self.update_prompt_timer = QTimer(self)
        self.update_prompt_timer.setInterval(500)
        self.update_prompt_timer.timeout.connect(self._offer_startup_update)
        self.locale_preference = str(self.settings.value("interface/locale", "system"))
        self.locale_code = locale_manager.select(self._resolved_locale(self.locale_preference))
        self.appearance_theme_name = str(
            self.settings.value("interface/appearance_theme", DEFAULT_APPEARANCE)
        )
        if self.appearance_theme_name not in APPEARANCE_THEMES:
            self.appearance_theme_name = DEFAULT_APPEARANCE
        saved_order = self.settings.value("interface/toolbox_order", list(TOOLBOX_ORDER))
        self.toolbox_order = (list(saved_order) if isinstance(saved_order, (list, tuple))
                              and len(saved_order) == len(TOOLBOX_ORDER)
                              and set(saved_order) == set(TOOLBOX_ORDER) else list(TOOLBOX_ORDER))
        self.font_size_position = max(
            0, min(4, self.settings.value("interface/font_size_position", 1, type=int))
        )
        metrics_default_version = self.settings.value("metrics/default_version", 0, type=int)
        self.metrics_refresh_ms = max(
            250,
            min(
                5000,
                1000 if metrics_default_version < 1 else self.settings.value(
                    "metrics/refresh_ms", 1000, type=int
                ),
            ),
        )
        self.metrics_history_minutes = max(
            1, min(120, self.settings.value("metrics/history_minutes", 15, type=int))
        )
        self.metrics_visible_seconds = max(
            10,
            min(
                300,
                self.settings.value("metrics/detail_seconds", 30, type=int),
            ),
        )
        viewport_default_version = self.settings.value(
            "interface/viewport_appearance_version", 0, type=int
        )
        self.viewport_appearance_name = str(
            self.settings.value("interface/viewport_appearance", DEFAULT_APPEARANCE)
        )
        if viewport_default_version < 1 and self.viewport_appearance_name == "match":
            self.viewport_appearance_name = DEFAULT_APPEARANCE
        if self.viewport_appearance_name not in {"match", "custom", *APPEARANCE_THEMES}:
            self.viewport_appearance_name = "match"
        self.custom_density_gradient_left = str(
            self.settings.value("interface/density_gradient_left", "#174A82")
        )
        self.custom_density_gradient_right = str(
            self.settings.value("interface/density_gradient_right", "#E23730")
        )
        self.shortcuts = {
            shortcut_id: QKeySequence(
                str(self.settings.value(f"shortcuts/{shortcut_id}", default_sequence))
            ).toString(QKeySequence.SequenceFormat.PortableText)
            for shortcut_id, (_label, default_sequence) in SHORTCUT_DEFINITIONS.items()
        }
        if self.shortcuts["toggle_fullscreen"] in {"Ctrl+Enter", "Ctrl+Return"}:
            self.shortcuts["toggle_fullscreen"] = "F11"
            self.settings.setValue("shortcuts/toggle_fullscreen", "F11")
        self.fullscreen_shortcut = QShortcut(QKeySequence(self.shortcuts["toggle_fullscreen"]), self)
        self.fullscreen_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.fullscreen_shortcut.setAutoRepeat(False)
        self.fullscreen_shortcut.activated.connect(self._toggle_full_screen)
        self.setWindowTitle(APP_NAME)
        self._apply_appearance()
        icon_path = resource_path("assets/meshmill-mark.svg")
        if icon_path.exists():
            self.setWindowIcon(QIcon(os.fspath(icon_path)))
        self.resize(1180, 780)
        self.tools_side = str(self.settings.value("interface/tools_side", "Right"))
        self.viewport_overlay_visible = {
            key: self.settings.value(f"interface/overlay_{key}_visible", True, type=bool)
            for key in VIEWPORT_OVERLAYS
        }
        self.viewport_overlay_corners = {
            key: str(
                self.settings.value(f"interface/overlay_{key}_corner", default_corner)
            )
            for key, (_label, default_corner) in VIEWPORT_OVERLAYS.items()
        }
        self._normalize_viewport_overlay_corners()
        units_default_version = self.settings.value("interface/units_default_version", 0, type=int)
        self.default_units = (
            "cm"
            if units_default_version < 2
            else str(self.settings.value("interface/default_units", "cm"))
        )
        if self.default_units not in UNIT_MM:
            self.default_units = "cm"
        self.settings.setValue("interface/default_units", self.default_units)
        self.settings.setValue("interface/units_default_version", 2)
        self.large_mesh_mode = str(self.settings.value("performance/large_mesh_mode", "auto"))
        if self.large_mesh_mode not in {"auto", "full", "overview", "tiled"}:
            self.large_mesh_mode = "auto"
        self.auto_tiling_enabled = self.settings.value(
            "performance/auto_tiling_enabled", True, type=bool
        )
        saved_memory_percent = self.settings.value(
            "performance/memory_percent", 50, type=int
        )
        self.large_mesh_memory_percent = max(
            20, min(80, round(saved_memory_percent / 5) * 5)
        )
        self.geometry_processes = max(
            0, self.settings.value("performance/geometry_processes", 0, type=int)
        )
        self.startup_algorithm = str(
            self.settings.value("optimization/startup_algorithm", "auto")
        )
        self.last_algorithm = str(
            self.settings.value(
                "optimization/last_algorithm",
                default_algorithm(self.gpu_preference),
            )
        )
        self.startup_quality = str(
            self.settings.value("optimization/startup_quality", "last")
        )
        self.last_quality = str(
            self.settings.value("optimization/last_quality", "Balanced")
        )
        if self.last_quality not in SMART_DIVISORS:
            self.last_quality = "Balanced"
        legacy_mouse = self.settings.value("navigation/invert_mouse", False, type=bool)
        legacy_keyboard = self.settings.value("navigation/invert_keyboard", False, type=bool)
        legacy_horizontal = self.settings.value(
            "navigation/invert_horizontal", False, type=bool
        )
        legacy_vertical = self.settings.value("navigation/invert_vertical", False, type=bool)
        legacy_zoom = self.settings.value("navigation/invert_zoom", False, type=bool)
        self.mouse_horizontal = self.settings.value(
            "navigation/mouse_horizontal", legacy_mouse or legacy_horizontal, type=bool
        )
        self.mouse_vertical = self.settings.value(
            "navigation/mouse_vertical", legacy_mouse or legacy_vertical, type=bool
        )
        self.mouse_zoom = self.settings.value(
            "navigation/mouse_zoom", legacy_mouse or legacy_zoom, type=bool
        )
        self.keyboard_horizontal = self.settings.value(
            "navigation/keyboard_horizontal", legacy_keyboard or legacy_horizontal, type=bool
        )
        self.keyboard_vertical = self.settings.value(
            "navigation/keyboard_vertical", legacy_keyboard or legacy_vertical, type=bool
        )
        self.keyboard_zoom = self.settings.value(
            "navigation/keyboard_zoom", legacy_keyboard or legacy_zoom, type=bool
        )
        self.source_path: Path | None = None
        self.loaded_triangle_count = 0
        self.source_poly: vtkPolyData | None = None
        self.source_points: np.ndarray | None = None
        self.source_faces: np.ndarray | None = None
        self.preview_poly: vtkPolyData | None = None
        self.preview_actor: vtkActor | None = None
        self.source_actor: vtkActor | None = None
        self.simplified_actor: vtkActor | None = None
        self.previous_view_poly: vtkPolyData | None = None
        self.previous_view_actor: vtkActor | None = None
        self.active_actor: vtkActor | None = None
        self.loupe_renderer: vtkRenderer | None = None
        self.loupe_actor: vtkActor | None = None
        self.loupe_frame: QFrame | None = None
        self.loupe_widget: QVTKRenderWindowInteractor | None = None
        self.loupe_label: QLabel | None = None
        self.showing_original = False
        self.showing_previous = False
        self.dimension_drift_model_units: np.ndarray | None = None
        self.generation = 0
        self.busy = False
        self.loading = False
        self.active_operation = ""
        self.operation_snapshot: dict | None = None
        self.operation_cancel_event: threading.Event | None = None
        self.operation_future = None
        self.operation_uses_process = False
        self.operation_progress_path: Path | None = None
        self.operation_progress_stage = ""
        self.operation_steps: list[str] = []
        self.operation_step_index = 0
        self.operation_runtime_started_at = 0.0
        self.operation_runtime_stage_started_at = 0.0
        self.operation_runtime_stage = ""
        self.operation_runtime_seconds: dict[str, float] = {}
        self.last_operation_runtime: dict[str, float] = {}
        self.last_operation_name = ""
        self.runtime_aggregate_seconds: dict[str, float] = {}
        self.runtime_aggregate_counts: dict[str, int] = {}
        self.runtime_operation_count = 0
        self.runtime_total_seconds = 0.0
        self._last_loading_metric_refresh = 0.0
        self.load_generation = 0
        self.edit_generation = 0
        self.density_generation = 0
        self.density_cache: dict[int, np.ndarray] = {}
        self.resident_vertex_buffers = {}
        self.pending_density_poly: vtkPolyData | None = None
        self.load_started_at = 0.0
        self.load_geometry_ready = False
        self.gpu_warmup_future = None
        self.geometry_activity_until = 0.0
        self.geometry_activity_label = "Idle"
        self.pending_target: int | None = None
        self.running_target: int | None = None
        self.preview_dirty = True
        self.pass_ready = False
        self.preview_is_selection = False
        self.mesh_modified = False
        self.control_key_held = False
        self.saved_standard_views: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.orientation_reference_basis: np.ndarray | None = None
        self.export_calibration_transform: np.ndarray | None = None
        self.has_committed_optimization = False
        self.optimization_script_passes: list[dict[str, object]] = []
        self.script_base_poly = None
        self.optimization_script_omissions: list[str] = []
        self.overview_mode = False
        self.tiled_mode = False
        self.tiled_index_directory: Path | None = None
        self.overview_total_triangles = 0
        self.crop_start: QPoint | None = None
        self.crop_candidate: tuple[vtkPolyData, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.crop_info_frame: QFrame | None = None
        self.crop_info_label: QLabel | None = None
        self.selection_undo_button: QPushButton | None = None
        self.crop_info_user_positioned = False
        self.crop_points: list[QPoint] = []
        self.crop_world_points: list[np.ndarray] = []
        self._crop_projection_pending = False
        self.crop_drag_index: int | None = None
        self.crop_selection_actor: vtkActor | None = None
        self.crop_overlay_actors: list[vtkActor2D] = []
        self.measure_points: list[np.ndarray] = []
        self.measure_hover_world: np.ndarray | None = None
        self.measure_pending_position: tuple[int, int] | None = None
        self.measure_overlay_actors: list[vtkActor2D] = []
        self.measure_info_frame: QFrame | None = None
        self.measure_info_label: QLabel | None = None
        self.measure_units: ArrowComboBox | None = None
        self.measure_units_overridden = False
        self.measure_picker: vtkHardwarePicker | None = None
        self.selected_face_mask: np.ndarray | None = None
        self.selection_undo_stack: list[np.ndarray | None] = []
        self.selection_base_quality: str | None = None
        self.selection_base_algorithm: str | None = None
        self.selection_base_target: int | None = None
        self.selection_target: int | None = None
        self.undo_stack: list[MeshHistoryState] = []
        self.redo_stack: list[MeshHistoryState] = []
        self.undo_bytes = 0
        self.redo_bytes = 0
        self.undo_limit_bytes = 1_000_000_000
        self.logical_cpu_count = max(1, os.cpu_count() or 1)
        selected_cuda = next(
            (
                device
                for device in self.cuda_device_list
                if device["pci_bus_id"] == self.gpu_preference
            ),
            self.cuda_device_list[0] if self.cuda_device_list else None,
        )
        self.vram_capacity_bytes = (
            int(selected_cuda["total_memory"])
            if selected_cuda is not None
            else dedicated_vram_capacity()
        )
        self._last_cpu_times: tuple[int, int, int] | None = None
        self._last_io_totals: tuple[int, int, float] | None = None
        self._io_active_seconds = 0.0
        self._peak_read_sample = 0
        self._peak_write_sample = 0
        self._capacity_available_memory = 0
        self.gpu_counters = WindowsGpuCounters()
        self.cuda_counters = NvidiaSmiCounters()
        self.verification_screenshot: Path | None = None
        self.diagnostic_log_path: Path | None = None
        self._last_diagnostic_camera_time = 0.0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh-reducer")
        self.source_measurement_cache = None
        self._probe_statistics_job = None
        self.update_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="update-check")
        self.optimization_executor = OwnedProcessPool(max_workers=1)
        self.bridge = ResultBridge()
        self.bridge.done.connect(self._finish_simplify)
        self.load_bridge = LoadBridge()
        self.load_bridge.done.connect(self._finish_load)
        self.edit_bridge = EditBridge()
        self.edit_bridge.done.connect(self._finish_mesh_edit)
        self.density_bridge = DensityBridge()
        self.density_bridge.done.connect(self._finish_density_analysis)
        self.update_bridge = UpdateBridge()
        self.update_bridge.done.connect(self._finish_update_check)
        self.update_bridge.install_done.connect(self._finish_update_download)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(650)
        self.timer.timeout.connect(self.preview)
        self.navigation_timer = QTimer(self)
        self.navigation_timer.setInterval(16)
        self.navigation_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.navigation_timer.timeout.connect(self._advance_navigation)
        self._navigation_steps = 0
        self._navigation_command = None
        self._held_navigation_key = None
        self._navigation_last_tick = None
        self._navigation_hold_started = None
        self._navigation_precision_steps = 0
        self.measure_pick_timer = QTimer(self)
        self.measure_pick_timer.setSingleShot(True)
        self.measure_pick_timer.setInterval(35)
        self.measure_pick_timer.timeout.connect(self._update_measure_hover)

        root = QWidget()
        root.setObjectName("mainRoot")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        app_header = WindowTitleBar(root)
        self.app_header = app_header
        app_header.setObjectName("appHeader")
        app_header.setMinimumHeight(34)
        app_header_layout = QHBoxLayout(app_header)
        app_header_layout.setContentsMargins(8, 0, 0, 0)
        app_header_layout.setSpacing(0)
        caption_icon = QLabel(app_header)
        caption_icon.setPixmap(QIcon(os.fspath(resource_path("assets/meshmill-mark.svg"))).pixmap(20, 20))
        caption_icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        app_header_layout.addWidget(caption_icon)
        self.caption_label = QLabel(self.windowTitle() or APP_NAME, app_header)
        self.caption_label.setContentsMargins(8, 0, 8, 0)
        self.caption_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.windowTitleChanged.connect(self.caption_label.setText)
        app_header_layout.addWidget(self.caption_label)
        app_header_layout.addStretch(1)
        self.fullscreen_button = QPushButton(app_header)
        self.fullscreen_button.setObjectName("fullscreenButton")
        self.fullscreen_button.setProperty("captionButton", True)
        self.fullscreen_button.setProperty("iconTone", "text")
        self.fullscreen_button.setAccessibleName("Full screen")
        self.fullscreen_button.setToolTip("Full screen (F11)")
        self.fullscreen_button.clicked.connect(self._toggle_full_screen)
        self._set_theme_icon(self.fullscreen_button, "fullscreen")
        app_header_layout.addWidget(self.fullscreen_button)
        for name, label, icon, callback in (
            ("minimize_button", "Minimize", "minimize", self.showMinimized),
            ("maximize_button", "Maximize", "maximize", self._toggle_maximized),
            ("close_button", "Close", "caption-close", self.close),
        ):
            button = QPushButton(app_header)
            button.setObjectName("captionClose" if name == "close_button" else name)
            button.setProperty("captionButton", True)
            button.setProperty("iconTone", "text")
            button.setAccessibleName(label)
            button.setToolTip(label)
            button.clicked.connect(callback)
            self._set_theme_icon(button, icon)
            app_header_layout.addWidget(button)
            setattr(self, name, button)
        self._sync_full_screen_button()
        root_layout.addWidget(app_header)

        content = QWidget(root)
        root_layout.addWidget(content, 1)
        outer = QHBoxLayout(content)
        self.outer_layout = outer

        controls = QScrollArea(root)
        controls.setObjectName("toolboxScroll")
        self.controls_widget = controls
        controls.setWidgetResizable(True)
        controls.setFrameShape(QFrame.Shape.NoFrame)
        controls.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls.verticalScrollBar().setSingleStep(32)
        controls.setFixedWidth(352)
        controls_content = QWidget(controls)
        self.controls_content = controls_content
        # A scroll area's content widget otherwise inherits the combined minimum width of
        # every child row. Wide paths and button groups can then extend behind the window
        # edge while the horizontal scrollbar is intentionally hidden. Let the viewport
        # determine the content width and allow each row to compress to the toolbox.
        controls_content.setMinimumWidth(0)
        controls_content.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        self._busy_control_states: dict[QWidget, bool] = {}
        controls_content.setObjectName("controlsContent")
        controls.setWidget(controls_content)
        panel = QVBoxLayout(controls_content)
        panel.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        def field_label(text: str, help_text: str) -> QLabel:
            label = QLabel(text)
            label.setToolTip(help_text)
            return label

        brand_row = QHBoxLayout()
        brand_row.addStretch(1)
        brand_logo = QLabel()
        self.brand_logo = brand_logo
        brand_logo.setFixedSize(310, 78)
        brand_logo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._refresh_brand_logo()
        brand_row.addWidget(brand_logo)
        panel.addLayout(brand_row)

        self.open_button = QPushButton("Open")
        self.envelope_button = QPushButton("Cast")
        self.envelope_button.setProperty("accent", True)
        self.envelope_button.clicked.connect(self._create_envelope)
        self.open_button.setProperty("accent", True)
        self.open_button.setProperty("fileAction", True)
        self.open_button.clicked.connect(self.open_file)
        self._set_theme_icon(self.open_button, "open")

        self.reload_button = QPushButton("Reload Original Mesh")
        self.reload_button.setProperty("quiet", True)
        self.reload_button.setEnabled(False)
        self.reload_button.clicked.connect(self._reload_original)

        self.undo_button = QPushButton("Undo")
        self.undo_button.setProperty("quiet", True)
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self._undo_working_mesh)
        self.cancel_preview_button = QPushButton("Cancel")
        self.cancel_preview_button.setProperty("quiet", True)
        self.cancel_preview_button.setEnabled(False)
        self.cancel_preview_button.setToolTip(
            "Discard the pending preview and keep the working mesh."
        )
        self.cancel_preview_button.clicked.connect(self._cancel_workflow_preview)
        self.redo_button = QPushButton("Redo")
        self.redo_button.setProperty("quiet", True)
        self.redo_button.setEnabled(False)
        self.redo_button.clicked.connect(self._redo_working_mesh)
        self.history_actions = QFrame()
        history_row = QHBoxLayout(self.history_actions)
        history_row.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        history_row.setContentsMargins(2, 2, 2, 2)
        history_row.setSpacing(6)
        for button in (self.undo_button, self.cancel_preview_button, self.redo_button):
            button.setMinimumWidth(72)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            history_row.addWidget(button)

        self._toolbox_layout = panel
        self._toolbox_sections = {}

        def tool_section(
            title_text: str, icon_name: str, tone: str
        ) -> tuple[QVBoxLayout, QHBoxLayout]:
            frame = QFrame(controls_content)
            frame.setObjectName("toolSection")
            frame.setProperty("sectionTone", tone)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(12, 11, 12, 12)
            layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
            layout.setSpacing(8)
            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(7)
            section_icon = QLabel(frame)
            section_icon.setObjectName("sectionIcon")
            section_icon.setProperty("iconTone", tone)
            section_icon.setFixedSize(18, 18)
            section_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._set_theme_icon(section_icon, icon_name)
            header.addWidget(section_icon)
            heading = QLabel(title_text, frame)
            heading.setObjectName("sectionTitle")
            header.addWidget(heading)
            header.addStretch(1)
            layout.addLayout(header)
            panel.addWidget(frame)
            self._toolbox_sections[title_text] = frame
            return layout, header

        self._make_tool_section = tool_section
        optimization_section, optimization_header = tool_section("OPTIMIZATION", "optimize", "optimization")
        self.optimization_toggle = QToolButton()
        self.optimization_toggle.setObjectName("sectionExpander")
        self.optimization_toggle.setCheckable(True)
        self.optimization_toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.optimization_toggle.setAccessibleName("Collapse Optimize")
        optimization_header.addWidget(self.optimization_toggle)
        self.optimization_body = QFrame()
        optimization_panel = QVBoxLayout(self.optimization_body)
        optimization_panel.setContentsMargins(0, 0, 0, 0)
        optimization_panel.setSpacing(8)
        optimization_section.addWidget(self.optimization_body)
        self.optimization_toggle.toggled.connect(lambda expanded: self._toggle_tool_section("Optimize", expanded))
        self.optimization_toggle.setChecked(True)
        solidify_panel, solidify_header = tool_section("SOLIDIFY", "solidify", "optimization")
        self.solidify_toggle = QToolButton()
        self.solidify_toggle.setObjectName("sectionExpander")
        self.solidify_toggle.setCheckable(True)
        self.solidify_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.solidify_toggle.setAccessibleName("Expand Solidify")
        solidify_header.addWidget(self.solidify_toggle)
        self.solidify_body = QFrame()
        solidify_form = QFormLayout(self.solidify_body)
        self.solidify_form = solidify_form
        solidify_form.setContentsMargins(0, 0, 0, 0)
        self.solidify_method = ArrowComboBox()
        from solidify import METHODS
        self.solidify_method.addItems(METHODS)
        self.solidify_method.setToolTip("Fill missing geometry while preserving measured surfaces; validate the completed solid before applying it.")
        solidify_form.addRow("Method", self.solidify_method)
        self.solidify_method.hide()
        solidify_form.labelForField(self.solidify_method).hide()
        from geometry_dimension_control import GeometryDimensionControl
        self.solidify_gap = GeometryDimensionControl("solid-gap")
        solidify_form.addRow("Gap bridging", self.solidify_gap)
        self.solidify_quality = ArrowComboBox()
        self.solidify_quality.addItems(["Draft", "Balanced", "Fine"])
        self.solidify_quality.setCurrentText("Balanced")
        solidify_form.addRow("Detail", self.solidify_quality)
        self.solidify_info_row = QWidget()
        info_layout = QHBoxLayout(self.solidify_info_row)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)
        self.solidify_info_labels = {}
        for key, title in (("open_rims", "Open rims"), ("open_edges", "Open edges"), ("rim_length", "Rim length")):
            label = QLabel(title + "\n?")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setProperty("qualityMetric", True)
            label.setMinimumWidth(0)
            info_layout.addWidget(label, 1)
            self.solidify_info_labels[key] = label
        solidify_form.addRow(self.solidify_info_row)
        self._solidify_info_key = None
        self._solidify_info_result = None
        self._solidify_info_future = None
        self.solidify_info_timer = QTimer(self)
        self.solidify_info_timer.setInterval(80)
        self.solidify_info_timer.timeout.connect(self._poll_solidify_info)
        self.solidify_preview_button = QPushButton("Solidify")
        self.solidify_preview_button.setProperty("accent", True)
        self.solidify_preview_button.clicked.connect(self._preview_solidify)
        self.solidify_preview_button.setToolTip("Create a solid preview for the selection, or for the whole mesh when nothing is selected.")
        solidify_form.addRow(self.solidify_preview_button)
        self.solidify_error = QLabel()
        self.solidify_error.setWordWrap(True)
        self.solidify_error.hide()
        solidify_form.addRow(self.solidify_error)
        self.solidify_body.hide()
        solidify_panel.addWidget(self.solidify_body)
        self.solidify_toggle.toggled.connect(lambda expanded: self._toggle_tool_section("Solidify", expanded))
        imprint_panel, _imprint_header = tool_section("CAST", "imprint", "optimization")
        imprint_panel.parentWidget().setProperty("imprintSection", True)
        self.imprint_section_toggle = QToolButton()
        self.imprint_section_toggle.setObjectName("sectionExpander")
        self.imprint_section_toggle.setAccessibleName("Expand Cast")
        self.imprint_section_toggle.setToolTip("Expand or collapse Cast")
        self.imprint_section_toggle.setCheckable(True)
        self.imprint_section_toggle.setChecked(False)
        self.imprint_section_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.imprint_section_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.imprint_section_toggle.toggled.connect(self._toggle_imprint_section)
        _imprint_header.addWidget(self.imprint_section_toggle)
        self.imprint_body = QFrame()
        self.imprint_body_layout = QVBoxLayout(self.imprint_body)
        self.imprint_body_layout.setContentsMargins(0, 0, 0, 0)
        self.imprint_body.hide()
        imprint_panel.addWidget(self.imprint_body)
        workflow_panel, _workflow_header = tool_section("ACTIONS", "actions", "optimization")
        general_panel, general_header = tool_section("GENERAL", "general", "general")
        stats_panel, _stats_header = tool_section("STATS", "stats", "stats")

        self.file_label = ElidedPathLabel("No file loaded")
        self.file_label.setObjectName("fileCard")
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        file_row = QHBoxLayout()
        file_row.setSpacing(4)
        file_row.addWidget(self.file_label, 1)
        self.open_file_folder_button = QPushButton("📁")
        self.open_file_folder_button.setFixedSize(34, 34)
        self.open_file_folder_button.setEnabled(False)
        self.open_file_folder_button.setProperty("iconTone", "general")
        self.open_file_folder_button.setToolTip("Open the folder containing this STL.")
        self.open_file_folder_button.clicked.connect(self._open_source_folder)
        self._set_theme_icon(self.open_file_folder_button, "folder")
        self.open_file_folder_button.setText("")
        file_row.addWidget(self.open_file_folder_button)
        self.copy_file_path_button = QPushButton("⧉")
        self.copy_file_path_button.setFixedSize(34, 34)
        self.copy_file_path_button.setEnabled(False)
        self.copy_file_path_button.setProperty("iconTone", "optimization")
        self.copy_file_path_button.setToolTip("Copy the full STL path to the clipboard.")
        self.copy_file_path_button.clicked.connect(self._copy_source_path)
        self._set_theme_icon(self.copy_file_path_button, "copy")
        self.copy_file_path_button.setText("")
        file_row.addWidget(self.copy_file_path_button)
        stats_panel.addLayout(file_row)

        stats = QGridLayout()
        stats.addWidget(field_label("Input triangles:", "Triangle count in the current working mesh."), 0, 0)
        self.original_count = QLabel("—")
        stats.addWidget(self.original_count, 0, 1)
        stats.addWidget(field_label("Input vertices:", "Vertex count in the current working mesh."), 1, 0)
        self.original_vertices = QLabel("—")
        stats.addWidget(self.original_vertices, 1, 1)
        stats.addWidget(field_label("Result triangles:", "Triangle count in the pending result, or an estimate for the optimization target."), 2, 0)
        self.preview_count = QLabel("—")
        stats.addWidget(self.preview_count, 2, 1)
        stats.addWidget(field_label("Result vertices:", "Vertex count in the pending result, or an estimate for the optimization target."), 3, 0)
        self.preview_vertices = QLabel("—")
        stats.addWidget(self.preview_vertices, 3, 1)
        self.dimensions_caption = QLabel("Dimensions (mm):")
        self.dimensions_caption.setToolTip("Overall X × Y × Z size of the original mesh.")
        stats.addWidget(self.dimensions_caption, 4, 0, Qt.AlignmentFlag.AlignTop)
        self.dimensions = QLabel("—")
        self.dimensions.setWordWrap(True)
        stats.addWidget(self.dimensions, 4, 1)
        stats.addWidget(
            field_label(
                "Dimension drift:",
                "Absolute change in the optimized result's X, Y, and Z bounding-box dimensions.",
            ),
            5,
            0,
        )
        self.drift = QLabel("—")
        stats.addWidget(self.drift, 5, 1)
        stats.addWidget(
            field_label(
                "Reduction:",
                "Triangle reduction relative to the current working mesh. A negative value means the result adds triangles.",
            ),
            6,
            0,
        )
        self.optimization_ratio = QLabel("—")
        stats.addWidget(self.optimization_ratio, 6, 1)
        stats.addWidget(field_label("Opened file:", "Size of the STL file opened from disk."), 7, 0)
        self.original_file_size = QLabel("—")
        stats.addWidget(self.original_file_size, 7, 1)
        stats.addWidget(field_label("Output size:", "Binary STL size of the pending result, or an estimate for the optimization target."), 8, 0)
        self.estimated_file_size = QLabel("—")
        stats.addWidget(self.estimated_file_size, 8, 1)
        for value in (
            self.original_count,
            self.original_vertices,
            self.preview_count,
            self.preview_vertices,
            self.drift,
            self.optimization_ratio,
            self.original_file_size,
            self.estimated_file_size,
        ):
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.dimensions.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        stats_panel.addLayout(stats)

        quality_row = QHBoxLayout()
        quality_row.setSpacing(3)
        self.quality_metric_labels: dict[str, QLabel] = {}
        quality_help_text = {
            "rms_percent": (
                "Average shape error: sampled distance from the original mesh to the result, shown "
                "as a percentage of the model diagonal. Lower preserves the overall surface better. "
                "A very low average can still hide isolated damaged areas, so compare it with P95."
            ),
            "p95_percent": (
                "High-end shape error: 95% of sampled original points are within this distance of "
                "the result, shown as a percentage of the model diagonal. Lower is better. This "
                "reveals localized damage that the average error can hide."
            ),
            "area_percent": (
                "Surface-area change: how much total triangle area changed. Lower is usually better. "
                "A large value can indicate collapsed features, folded surfaces, or duplicated geometry."
            ),
            "nonmanifold_percent": (
                "Topology risk: non-manifold edges per output triangle. Lower is better. High values "
                "can cause trouble in downstream editing or fabrication, but zero does not guarantee "
                "that an open scan is watertight."
            ),
        }
        quality_definitions = (
            ("p95_percent", "P95"),
            ("rms_percent", "RMS"),
            ("nonmanifold_percent", "TOPO"),
            ("area_percent", "AREA"),
        )
        for key, short_name in quality_definitions:
            metric = QLabel(f"{short_name}\n—")
            metric.setAlignment(Qt.AlignmentFlag.AlignCenter)
            metric.setToolTip(tr(quality_help_text[key]))
            metric.setMinimumWidth(38)
            metric.setObjectName("qualityMetric")
            metric.setProperty("qualityMetric", True)
            quality_row.addWidget(metric, 1)
            self.quality_metric_labels[key] = metric
        stats_panel.addLayout(quality_row)

        self.preview_button = QPushButton("Optimize")
        self.preview_button.setProperty("accent", True)
        self._set_primary_action(False, False)
        self.preview_button.clicked.connect(self._primary_optimize_action)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self._slider_changed)
        optimization_panel.addWidget(self.slider)

        target_row = QHBoxLayout()
        target_row.addWidget(
            field_label(
                "Target triangles",
                "Approximate triangle count requested by the optimization slider. Lower values "
                "produce lighter meshes; higher values preserve more detail.",
            )
        )
        self.target_display = QLabel("200,000")
        self.target_display.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.target_display.setToolTip(
            "Live target from the optimization slider. The optimizer may finish a few triangles "
            "above or below this value when preserving valid geometry requires it."
        )
        target_row.addStretch(1)
        target_row.addWidget(self.target_display)
        optimization_panel.addLayout(target_row)
        self.target = QSpinBox()
        self.target.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.target.setRange(1_000, 5_000_000)
        self.target.setSingleStep(10_000)
        self.target.setValue(200_000)
        self.target.valueChanged.connect(self._target_changed)

        smart_row = QHBoxLayout()
        smart_row.addWidget(
            field_label(
                "Smart quality",
                "Sets how aggressively MeshMill derives the triangle target. Draft favors a "
                "smaller mesh, Balanced splits the tradeoff, and Fine preserves more detail. "
                "Changing this value updates the target immediately.",
            )
        )
        self.smart_quality = ArrowComboBox()
        self.smart_quality.addItems(list(SMART_DIVISORS))
        initial_quality = (
            self.last_quality
            if self.startup_quality == "last"
            else self.startup_quality
        )
        if initial_quality not in SMART_DIVISORS:
            initial_quality = "Balanced"
        self.smart_quality.setCurrentText(initial_quality)
        self.smart_quality.currentTextChanged.connect(self._smart_quality_changed)
        smart_row.addWidget(self.smart_quality, 1)
        optimization_panel.addLayout(smart_row)

        algorithm_row = QHBoxLayout()
        algorithm_row.addWidget(field_label("Algorithm", "Geometry optimization method used for the next operation."))
        self.algorithm = ArrowComboBox()
        self.algorithm.addItems(ALGORITHMS)
        self.algorithm.setCurrentText(
            resolve_startup_algorithm(
                self.startup_algorithm,
                self.last_algorithm,
                self.gpu_preference,
            )
        )
        self.algorithm.setToolTip(
            "Fast QEM works on every supported system. Depth Adaptive QEM uses its bundled "
            "NVIDIA CUDA runtime to measure the surface before reduction and can preserve more "
            "shape detail. Depth + Density Adaptive QEM adds density- and normal-weighted "
            "reduction plus bounded surface finishing for the highest measured quality, with "
            "a longer runtime. GPU-guided algorithms are available when MeshMill detects "
            "compatible NVIDIA hardware."
        )
        if not gpu_backend_supported(self.gpu_preference):
            for gpu_algorithm in (DEPTH_ADAPTIVE_LABEL, DEPTH_DENSITY_ADAPTIVE_LABEL):
                gpu_index = self.algorithm.findText(gpu_algorithm)
                if gpu_index >= 0:
                    self.algorithm.setItemData(
                        gpu_index,
                        QColor("#9CA3AF"),
                        Qt.ItemDataRole.ForegroundRole,
                    )
        self.algorithm.currentTextChanged.connect(self._algorithm_changed)
        algorithm_row.addWidget(self.algorithm, 1)
        optimization_panel.addLayout(algorithm_row)

        self.display_type = ArrowComboBox(self)
        for mode in DISPLAY_MODES:
            shortcut = self.shortcuts[f"display_{mode.lower()}"]
            self.display_type.addItem(f"{mode} ({shortcut})", mode)
        saved_display = str(self.settings.value("display/type", "Shaded"))
        saved_display = {
            "Point matrix": "Vertices",
            "Point density matrix": "Density",
        }.get(saved_display, saved_display)
        saved_index = self.display_type.findData(saved_display)
        if saved_index >= 0:
            self.display_type.setCurrentIndex(saved_index)
        self.display_type.currentIndexChanged.connect(
            lambda _index: self._apply_display_mode(self._display_mode())
        )
        display_row = QHBoxLayout()
        display_row.addWidget(
            field_label("Display type", "Choose how mesh geometry and point density are drawn.")
        )
        display_row.addWidget(self.display_type, 1)
        workflow_panel.addLayout(display_row)

        units_row = QHBoxLayout()
        units_row.addWidget(
            field_label(
                "Model units",
                "Choose how dimensions and measurements are displayed. STL coordinates are unchanged.",
            )
        )
        self.units = ArrowComboBox()
        self.units.addItems(["mm", "cm", "m", "in", "ft"])
        self.units.currentTextChanged.connect(self._units_changed)
        self.units.setCurrentText(self.default_units)
        self._units_changed(self.default_units)
        units_row.addWidget(self.units, 1)
        workflow_panel.addLayout(units_row)

        processing_row = QHBoxLayout()
        processing_row.addWidget(
            field_label(
                "Processing",
                "Auto chooses the faster supported route. Full mesh keeps all geometry together. "
                "Tiled processing divides a mesh into bounded pieces for parallel CPU work.",
            )
        )
        self.processing_route = ArrowComboBox()
        self.processing_route.addItem("Auto", "auto")
        self.processing_route.addItem("Full mesh", "full")
        self.processing_route.addItem("Tiled", "tiled")
        route_index = self.processing_route.findData(self.large_mesh_mode)
        self.processing_route.setCurrentIndex(max(0, route_index))
        self.processing_route.setToolTip(
            "Auto checks the selected algorithm, mesh size, available memory, cached tile index, "
            "worker capacity, and matched route measurements. It uses tiles only when their setup "
            "and processing estimate is better than full mesh processing. Full mesh and Tiled override Auto."
        )
        self.processing_route.currentIndexChanged.connect(self._processing_route_changed)
        processing_row.addWidget(self.processing_route, 1)
        optimization_panel.addLayout(processing_row)

        self._processing_help = self.processing_route.toolTip()
        open_row = QHBoxLayout()
        open_row.setSpacing(5)
        open_row.addWidget(self.open_button, 1)
        self.recent_files_button = QPushButton(self.controls_content)
        self.recent_files_button.setObjectName("openRecentFiles")
        self.recent_files_button.setAccessibleName("Open recent files")
        self.recent_files_button.setToolTip("Open a recent STL file.")
        self.recent_files_button.setFixedWidth(34)
        self.recent_files_button.setStyleSheet("padding: 0px;")
        self._set_theme_icon(self.recent_files_button, "recent")
        self.recent_files_menu = QMenu(self.recent_files_button)
        self.recent_files_button.clicked.connect(self._show_recent_files)
        self.recent_files_button.setVisible(self.recent_files_enabled)
        open_row.addWidget(self.recent_files_button)
        workflow_panel.addLayout(open_row)
        self.imprint_form = QFormLayout()
        self.imprint_output = ArrowComboBox()
        self.imprint_output.addItems(["Clamshell", "Envelope"])
        self.imprint_output.setToolTip("Clamshell creates a hollow housing around the mesh, with an optional split. Envelope creates only the offset outer surface.")
        self.imprint_form.addRow("Output", self.imprint_output)
        self.imprint_prepare_solid = QCheckBox("Prepare with Solidify")
        self.imprint_prepare_solid.setChecked(True)
        self.imprint_prepare_solid.setToolTip("Fill missing scan geometry before building the clamshell. Closed solids skip this step.")
        self.imprint_form.addRow("Source", self.imprint_prepare_solid)
        self.imprint_prepare_solid.toggled.connect(self._imprint_settings_changed)

        from geometry_dimension_control import GeometryDimensionControl
        self.imprint_offset = GeometryDimensionControl("offset")
        self.imprint_form.addRow("Clearance", self.imprint_offset)
        surface_pick_row = QHBoxLayout()
        self.imprint_pick_surface = QPushButton("Pick surface")
        self.imprint_pick_surface.setCheckable(True)
        self.imprint_pick_surface.setToolTip("Click a known exterior surface to guide detection of its continuous surface sheet. Add picks where an incomplete scan is detected incorrectly.")
        self.imprint_pick_surface.toggled.connect(self._toggle_imprint_surface_pick)
        surface_pick_row.addWidget(self.imprint_pick_surface)
        self.imprint_clear_picks = QPushButton("Clear picks")
        self.imprint_clear_picks.clicked.connect(self._clear_imprint_surface_picks)
        surface_pick_row.addWidget(self.imprint_clear_picks)
        self.imprint_form.addRow("Detection", surface_pick_row)
        self._imprint_surface_seeds = []
        self._imprint_surface_seed_source = None
        self.imprint_shape = ArrowComboBox()
        self.imprint_shape.addItems(["Blob", "Box"])
        self.imprint_form.addRow("Outer shape", self.imprint_shape)
        self.imprint_wall = GeometryDimensionControl("wall")
        self.imprint_bridge = GeometryDimensionControl("gap")
        self.imprint_form.addRow("Wall thickness", self.imprint_wall)
        self.imprint_form.addRow("Gap bridging", self.imprint_bridge)
        self.imprint_wall.setToolTip("Auto scales the clamshell wall to the mesh. Drag the slider to set a manual thickness.")
        self.imprint_bridge.setToolTip("Auto scales gap bridging to the mesh. Manual zero leaves bridging off. Missing scan regions remain unknown space.")
        self.imprint_detail = QSlider(Qt.Orientation.Horizontal)
        self.imprint_detail.setRange(0, 4)
        self.imprint_detail.setValue(2)
        self.imprint_detail.setSingleStep(1)
        self.imprint_detail.setTickInterval(1)
        self.imprint_detail.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.imprint_detail.setToolTip("Left: coarser and faster. Right: finer detail. Actual resolution adapts to available memory.")
        self.imprint_form.addRow("Detail", self.imprint_detail)
        self.imprint_split = ArrowComboBox()
        self.imprint_split.addItems(["Unsplit", "Current view", "X", "Y", "Z", "Custom"])
        self.imprint_form.addRow("Split", self.imprint_split)
        self.imprint_body_layout.addLayout(self.imprint_form)
        self.imprint_error = QLabel()
        self.imprint_error.setWordWrap(True)
        self.imprint_error.hide()
        self.imprint_body_layout.addWidget(self.imprint_error)
        for widget in (self.imprint_output, self.imprint_shape, self.imprint_split):
            widget.currentIndexChanged.connect(self._imprint_settings_changed)
        for widget in (self.imprint_offset, self.imprint_wall, self.imprint_bridge, self.imprint_detail):
            widget.valueChanged.connect(self._imprint_settings_changed)
        self.envelope_button.setText("Cast")
        self.imprint_body_layout.addWidget(self.envelope_button)
        self.envelope_controls = QFrame()
        envelope_layout = QVBoxLayout(self.envelope_controls)
        envelope_layout.setContentsMargins(0, 0, 0, 0)
        envelope_row = QHBoxLayout()
        envelope_layout.addLayout(envelope_row)
        self.envelope_visible = QCheckBox("Overlay")
        self.envelope_visible.setChecked(True)
        self.envelope_visible.toggled.connect(self._update_envelope_overlay)
        envelope_row.addWidget(self.envelope_visible)
        self.envelope_original_visible = QCheckBox("Original")
        self.envelope_original_visible.setChecked(True)
        self.envelope_original_visible.setToolTip("Show the source mesh behind the cast preview.")
        self.envelope_original_visible.toggled.connect(self._update_envelope_overlay)
        envelope_row.addWidget(self.envelope_original_visible)
        self.envelope_separate_halves = QCheckBox("Separate parts")
        self.envelope_separate_halves.setToolTip("Separate the halves physically for selection and export. Turning this off returns them to their assembled positions, where STL importers may join touching faces.")
        self.envelope_separate_halves.toggled.connect(self._update_envelope_overlay)
        self.envelope_separate_halves.hide()
        envelope_layout.addWidget(self.envelope_separate_halves)
        self.envelope_opacity = QSlider(Qt.Orientation.Horizontal)
        self.envelope_opacity.setRange(1, 30)
        self.envelope_opacity.setValue(15)
        self.envelope_opacity.setToolTip("Cast preview opacity")
        self.envelope_opacity.valueChanged.connect(self._update_envelope_overlay)
        envelope_row.addWidget(self.envelope_opacity, 1)
        self.envelope_controls.hide()
        self.imprint_body_layout.addWidget(self.envelope_controls)
        self.envelope_stats = QLabel()
        self.envelope_stats.setWordWrap(True)
        self.envelope_stats.hide()
        self.imprint_body_layout.addWidget(self.envelope_stats)
        self._imprint_settings_changed()
        self.solidify_method.currentIndexChanged.connect(self._solidify_settings_changed)
        self.solidify_quality.currentIndexChanged.connect(self._solidify_settings_changed)
        self.solidify_gap.valueChanged.connect(self._solidify_settings_changed)
        self._solidify_settings_changed()
        self.units.currentTextChanged.connect(lambda _unit: self._refresh_solidify_info()
                                              if self.solidify_toggle.isChecked() else None)
        self.units.currentTextChanged.connect(lambda _unit: self._refresh_workflow_actions())
        workflow_panel.addWidget(self.reload_button)

        # Selection removal stays keyboard-driven (Delete and Escape). Keep internal controls for
        # shared enabled-state handling without adding redundant disabled buttons to the toolbox.
        self.delete_selection_button = QPushButton("Delete selected", self.controls_content)
        self.delete_selection_button.setEnabled(False)
        self.delete_selection_button.clicked.connect(self._delete_selected)
        self.delete_selection_button.hide()
        self.clear_selection_button = QPushButton("Clear selection", self.controls_content)
        self.clear_selection_button.setEnabled(False)
        self.clear_selection_button.clicked.connect(self._clear_selection)
        self.clear_selection_button.hide()
        workflow_panel.addWidget(self.history_actions)

        self.toggle_mesh_button = QPushButton("Show original")
        self.toggle_mesh_button.setEnabled(False)
        self.toggle_mesh_button.clicked.connect(self._toggle_mesh)

        self.previous_mesh_button = QPushButton("Show previous")
        self.previous_mesh_button.setEnabled(False)
        self.previous_mesh_button.clicked.connect(self._toggle_previous_mesh)

        self.comparison_actions = QFrame()
        comparison_row = QHBoxLayout(self.comparison_actions)
        comparison_row.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        comparison_row.setContentsMargins(2, 2, 2, 2)
        comparison_row.setSpacing(6)
        comparison_row.addWidget(self.toggle_mesh_button)
        comparison_row.addWidget(self.previous_mesh_button)

        self.status = QLabel("Open a mesh")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status.setProperty("muted", True)

        optimization_panel.addWidget(self.preview_button)
        self.workflow_preview_actions = QFrame()
        preview_actions_row = QHBoxLayout(self.workflow_preview_actions)
        preview_actions_row.setContentsMargins(0, 0, 0, 0)
        self.workflow_apply_button = QPushButton("Apply preview")
        self.workflow_apply_button.setProperty("accent", True)
        self.workflow_apply_button.clicked.connect(self._apply_workflow_preview)
        self.workflow_save_preview_button = QPushButton("Save preview")
        self.workflow_save_preview_button.clicked.connect(self._save_workflow_preview)
        self.workflow_save_preview_button.setToolTip("Export the preview without replacing the working mesh.")
        preview_actions_row.addWidget(self.workflow_apply_button)
        preview_actions_row.addWidget(self.workflow_save_preview_button)
        workflow_panel.addWidget(self.workflow_preview_actions)
        self.workflow_solid_summary = QWidget()
        solid_summary_row = QHBoxLayout(self.workflow_solid_summary)
        solid_summary_row.setContentsMargins(0, 0, 0, 0)
        solid_summary_row.setSpacing(4)
        self.solidify_result_labels = {}
        for key, title in (("retained_percent", "Surface kept"), ("rebuilt_percent", "Rebuilt area"), ("open_edges", "Open edges")):
            label = QLabel(title + "\n?")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setProperty("qualityMetric", True)
            label.setMinimumWidth(0)
            solid_summary_row.addWidget(label, 1)
            self.solidify_result_labels[key] = label
        workflow_panel.addWidget(self.workflow_solid_summary)
        self.workflow_solid_summary.hide()

        workflow_panel.addWidget(self.comparison_actions)

        self.save_button = QPushButton("Save current mesh…")
        self.save_button.setProperty("accent", True)
        self.save_button.setProperty("fileAction", True)
        self.save_button.setEnabled(self.mesh_modified)
        self.save_button.clicked.connect(self.save_file)
        self._set_theme_icon(self.save_button, "open")
        self.generate_script_button = QPushButton("Generate PowerShell script…")
        self.generate_script_button.setProperty("quiet", True)
        self.generate_script_button.setEnabled(False)
        self.generate_script_button.setToolTip(
            "Repeat committed Optimize, Solidify and Cast operations without opening the UI. "
            "Selections and crops use an edited-mesh input checkpoint."
        )
        self.generate_script_button.clicked.connect(self._generate_optimization_script)
        self._set_theme_icon(self.generate_script_button, "copy")
        workflow_panel.addWidget(self.generate_script_button)
        workflow_panel.addWidget(self.save_button)
        self._update_action_row_visibility()
        self._reorder_tool_sections()
        info_row = QHBoxLayout()
        self.settings_button = QPushButton("Settings")
        self.settings_button.setProperty("quiet", True)
        self.settings_button.clicked.connect(self._show_settings)
        self._set_theme_icon(self.settings_button, "settings")
        info_row.addWidget(self.settings_button)
        self.about_button = QPushButton("About")
        self.about_button.setProperty("quiet", True)
        self.about_button.clicked.connect(self._show_about)
        self._set_theme_icon(self.about_button, "info")
        info_row.addWidget(self.about_button)
        self.repository_button = QPushButton("GitHub")
        self.repository_button.setProperty("quiet", True)
        self.repository_button.setEnabled(bool(REPOSITORY_URL))
        self.repository_button.clicked.connect(self._open_repository)
        self._set_theme_icon(self.repository_button, "github")
        info_row.addWidget(self.repository_button)
        general_panel.addLayout(info_row)
        self.version_label = VersionButton(f"MeshMill {APP_VERSION}")
        self.version_label.setObjectName("versionLabel")
        self.version_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.version_label.setProperty("muted", True)
        self.version_label.setStyleSheet("")
        self.version_label.setToolTip(
            "Click to check GitHub for a newer MeshMill release. Right-click to copy the installed "
            "version. This does not enable automatic checks."
        )
        self.version_label.clicked.connect(self._version_action_clicked)
        self.version_label.rightClicked.connect(self._copy_version)
        general_header.addWidget(self.version_label)
        panel.addStretch(1)

        viewport_column = QWidget(root)
        self.viewport_column = viewport_column
        viewport_layout = QVBoxLayout(viewport_column)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(6)
        self.vtk_widget = QVTKRenderWindowInteractor(viewport_column)
        viewport_layout.addWidget(self.vtk_widget, 1)
        self._create_loading_panel()
        self._create_metrics_panel(viewport_layout)
        self._apply_tools_side(self.tools_side)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.043, 0.106, 0.204)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.GetRenderWindow().SetMultiSamples(8)
        self.vtk_widget.Initialize()
        self.navigation_style = MeshInteractorStyle(
            self.vtk_widget, self._show_loupe, self._crop_interaction, self._queue_keyboard_navigation,
            self._measurement_interaction,
        )
        self.navigation_style._surface_pick_callback = self._handle_imprint_surface_pick
        self.measure_picker = vtkHardwarePicker()
        self.keyboard_picker = vtkHardwarePicker()
        self.navigation_style.invert_horizontal = self.mouse_horizontal
        self.navigation_style.invert_vertical = self.mouse_vertical
        self.navigation_style.invert_zoom = self.mouse_zoom
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(self.navigation_style)
        self._create_view_pad()
        self._create_density_legend()
        self.vtk_widget.installEventFilter(self)
        QTimer.singleShot(0, self._position_viewport_overlays)
        self.metrics_timer = QTimer(self)
        # System metrics are sampled independently of geometry-progress updates so the
        # operation panel remains live while a worker is between progress callbacks.
        self.metrics_timer.setInterval(self.metrics_refresh_ms)
        self.metrics_timer.timeout.connect(self._update_system_metrics)
        self.metrics_timer.start()
        self.loading_timer = QTimer(self)
        self.loading_timer.setInterval(250)
        self.loading_timer.timeout.connect(self._update_loading_panel)
        self._apply_appearance()
        self._update_system_metrics()
        self._install_help_tooltips()
        localize_widget_tree(self)
        # Route application commands before focused editors consume standard text shortcuts.
        # The filter is active only while this MeshMill window is active.
        QApplication.instance().installEventFilter(self)
        if self.update_checks_enabled:
            QTimer.singleShot(1500, self._check_for_updates)

    @staticmethod
    def _resolved_locale(preference: str) -> str:
        if preference != "system":
            return preference
        system_name = QLocale.system().name().replace("_", "-")
        available = {code for code, _name in locale_manager.available_locales()}
        if system_name in available:
            return system_name
        language = system_name.split("-", 1)[0].lower()
        return next(
            (code for code in available if code.lower().split("-", 1)[0] == language),
            DEFAULT_LOCALE,
        )

    def eventFilter(self, watched, event) -> bool:
        if (event.type() in (QEvent.Type.WindowActivate, QEvent.Type.WindowDeactivate)
                and isinstance(watched, QWidget)):
            header_name = "appHeader" if watched is self else "dialogTitleBar"
            header = watched.findChild(WindowTitleBar, header_name)
            if header is not None:
                header.setProperty("windowActive", event.type() == QEvent.Type.WindowActivate)
                header.style().unpolish(header)
                header.style().polish(header)
                for label in header.findChildren(QLabel):
                    label.style().unpolish(label)
                    label.style().polish(label)
                header.update()
        if event.type() in (QEvent.Type.Polish, QEvent.Type.Show) and isinstance(watched, QDialog):
            owner = watched.parentWidget()
            while owner is not None and owner is not self:
                owner = owner.parentWidget()
            if owner is self:
                if event.type() == QEvent.Type.Polish:
                    self._install_dialog_title_bar(watched)
                QTimer.singleShot(0, watched, lambda: self._apply_native_title_bar(
                    appearance_theme(self.appearance_theme_name), watched
                ))
        if (event.type() == QEvent.Type.MouseButtonPress and isinstance(watched, QWidget)
                and event.button() == Qt.MouseButton.LeftButton):
            dialog = watched.window()
            if (isinstance(dialog, QDialog) and dialog.property("themedCaption")
                    and not dialog.isMaximized()):
                point = dialog.mapFromGlobal(event.globalPosition().toPoint())
                edges = Qt.Edge(0)
                if point.x() < 5:
                    edges |= Qt.Edge.LeftEdge
                elif point.x() >= dialog.width() - 5:
                    edges |= Qt.Edge.RightEdge
                if point.y() < 5:
                    edges |= Qt.Edge.TopEdge
                elif point.y() >= dialog.height() - 5:
                    edges |= Qt.Edge.BottomEdge
                if edges and dialog.windowHandle() and dialog.windowHandle().startSystemResize(edges):
                    return True
        controls = getattr(self, "controls_widget", None)
        in_toolbox = (controls is not None and isinstance(watched, QWidget)
                      and (watched is controls or controls.isAncestorOf(watched)))
        if in_toolbox and (
            event.type() in (QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.LayoutRequest)
            or (watched is controls.viewport() and event.type() == QEvent.Type.Resize)
        ):
            self._queue_toolbox_scroll_refresh()
        if (controls is not None and watched is controls.verticalScrollBar()
                and event.type() == QEvent.Type.MouseButtonPress):
            self._refresh_toolbox_scroll_range()
        if watched is getattr(self, "vtk_widget", None) and self._handle_imprint_plane_drag(event):
            return True
        if event.type() == QEvent.Type.Wheel and in_toolbox:
            # Scroll the toolbox over its fields as well as its empty background.
            # An open combo popup retains its normal item selection behavior.
            popup = QApplication.activePopupWidget()
            if popup is None:
                self._refresh_toolbox_scroll_range()
                bar = controls.verticalScrollBar()
                if bar.maximum() > bar.minimum():
                    pixel_delta = event.pixelDelta().y()
                    delta = (float(pixel_delta) if pixel_delta else
                             event.angleDelta().y() * bar.singleStep() * 3 / 120.0)
                    remainder = getattr(self, "_toolbox_wheel_remainder", 0.0) + delta
                    steps = int(remainder)
                    self._toolbox_wheel_remainder = remainder - steps
                    bar.setValue(bar.value() - steps)
                    event.accept()
                    return True
        if (event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
                and isinstance(watched, QWidget) and watched.window() is self
                and not in_toolbox
                and not self.isMaximized() and not self.isFullScreen()):
            edges = self._window_resize_edges(self.mapFromGlobal(event.globalPosition().toPoint()))
            if (edges and self.windowHandle() is not None
                    and self.windowHandle().startSystemResize(edges)):
                return True
        if (watched is self.vtk_widget and event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton and hasattr(self, "lathe")):
            render_width, render_height = self.vtk_widget.GetRenderWindow().GetSize()
            position = event.position()
            probe_position = (
                position.x()*render_width/max(1,self.vtk_widget.width()),
                (self.vtk_widget.height()-position.y()-1)*render_height/max(1,self.vtk_widget.height()),
            )
            if self.lathe.delete_probe(probe_position):
                return True
        if (watched is self.vtk_widget and event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            render_width, render_height = self.vtk_widget.GetRenderWindow().GetSize()
            position = event.position()
            self._select_connected_at((
                round(position.x()*render_width/max(1, self.vtk_widget.width())),
                round((self.vtk_widget.height()-position.y()-1)*render_height
                      /max(1, self.vtk_widget.height())),
            ))
            return True
        if (event.type() == QEvent.Type.ApplicationDeactivate
                or (event.type() == QEvent.Type.WindowDeactivate and watched is self)):
            self._stop_keyboard_navigation()
        if (event.type() == QEvent.Type.KeyRelease and not event.isAutoRepeat()
                and event.key() == self._held_navigation_key):
            self._stop_keyboard_navigation()
        if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Control:
            self.control_key_held = True
        elif event.type() == QEvent.Type.KeyRelease and event.key() == Qt.Key.Key_Control or event.type() == QEvent.Type.ApplicationDeactivate:
            self.control_key_held = False
        if watched is self.vtk_widget and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._position_viewport_overlays)
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.EnabledChange and watched in (
            self.undo_button,
            self.cancel_preview_button,
            self.redo_button,
            self.toggle_mesh_button,
            self.previous_mesh_button,
            self.reload_button,
            self.save_button,
            self.preview_button,
            self.open_file_folder_button,
            self.copy_file_path_button,
        ):
            QTimer.singleShot(0, self._update_action_row_visibility)
        if (
            event.type() not in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
            or not self.isActiveWindow()
        ):
            return super().eventFilter(watched, event)

        if (event.key() == Qt.Key.Key_Space and event.modifiers() == Qt.KeyboardModifier.NoModifier
                and hasattr(self, "lathe") and self.lathe.button.isChecked()
                and not self.lathe.calibration_locked and not self.busy):
            focus = QApplication.focusWidget()
            editing = focus is not None and any(focus.inherits(name) for name in
                ("QLineEdit", "QTextEdit", "QPlainTextEdit", "QAbstractSpinBox", "QComboBox"))
            if not editing:
                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                    self.lathe.set_spin(not self.lathe.timer.isActive())
                event.accept()
                return True

        modifiers = event.modifiers()
        raw_modifiers = int(modifiers.value)
        if event.key() in {
            Qt.Key.Key_Insert,
            Qt.Key.Key_Home,
            Qt.Key.Key_PageUp,
            Qt.Key.Key_Delete,
            Qt.Key.Key_End,
            Qt.Key.Key_PageDown,
        }:
            # Windows reports the six-key navigation cluster as keypad keys on
            # some keyboards. Treat that hardware flag as irrelevant so the
            # physical keys match the displayed shortcuts.
            modifiers &= ~Qt.KeyboardModifier.KeypadModifier
        elif event.key() in {Qt.Key.Key_Plus, Qt.Key.Key_Minus}:
            # Treat the main keyboard and numeric keypad + / - identically.
            modifiers &= ~Qt.KeyboardModifier.KeypadModifier
            if event.key() == Qt.Key.Key_Plus:
                modifiers &= ~Qt.KeyboardModifier.ShiftModifier
        combination = QKeyCombination(modifiers, Qt.Key(event.key()))
        sequence = QKeySequence(combination).toString(QKeySequence.SequenceFormat.PortableText)
        matching_actions = [
            shortcut_id for shortcut_id, configured in self.shortcuts.items()
            if configured == sequence and shortcut_id != "toggle_fullscreen"
        ]
        action = None
        callback = None
        for candidate in matching_actions:
            candidate_callback = self._shortcut_callback(candidate)
            if candidate_callback is not None:
                action = candidate
                callback = candidate_callback
                break
        self._diagnostic(
            "key_event",
            event_type=event.type().name,
            key=int(event.key()),
            raw_modifiers=raw_modifiers,
            normalized_sequence=sequence,
            matching_actions=matching_actions,
            resolved_action=action,
            watched=type(watched).__name__ if watched is not None else None,
            focus=type(QApplication.focusWidget()).__name__ if QApplication.focusWidget() else None,
        )
        if action is None or callback is None:
            return super().eventFilter(watched, event)
        navigation_actions = {
            "orbit_left", "orbit_right", "orbit_up", "orbit_down",
            "pan_left", "pan_right", "pan_up", "pan_down",
            "zoom_in", "zoom_out", "zoom_in_alt", "zoom_out_alt",
            "roll_counterclockwise", "roll_clockwise",
        }
        if event.isAutoRepeat() and action not in navigation_actions:
            event.accept()
            return True
        if event.type() == QEvent.Type.ShortcutOverride:
            event.accept()
            return True
        if action in navigation_actions:
            if event.isAutoRepeat() and self._held_navigation_key == event.key():
                event.accept()
                return True
            self._held_navigation_key = event.key()
        else:
            self._stop_keyboard_navigation()
        callback()
        event.accept()
        return True

    def _update_action_row_visibility(self) -> None:
        self._refresh_workflow_actions()
        if not self.busy:
            self.undo_button.setEnabled(bool(self.undo_stack))
            self.redo_button.setEnabled(bool(self.redo_stack))
            preview = self.preview_poly is not None and self.preview_poly is not self.source_poly
            self.toggle_mesh_button.setEnabled(preview)
            self.toggle_mesh_button.setText("Show optimized" if self.showing_original else "Show original")
            previous = self.undo_stack[-1].poly if self.undo_stack else None
            current = self.source_poly if self.showing_original or not preview else self.preview_poly
            self.previous_mesh_button.setEnabled(previous is not None and previous is not current)
            self.previous_mesh_button.setText("Show current" if self.showing_previous else "Show previous")
            for name in ("undo_button", "redo_button", "toggle_mesh_button", "previous_mesh_button"):
                button = getattr(self, name, None)
                if button is not None:
                    button.setVisible(button.isEnabled())
        if hasattr(self, "history_actions"):
            self.history_actions.setVisible(
                any(
                    button.isEnabled()
                    for button in (
                        self.undo_button, self.cancel_preview_button, self.redo_button
                    )
                )
            )
        if hasattr(self, "comparison_actions"):
            self.comparison_actions.setVisible(
                self.toggle_mesh_button.isEnabled() or self.previous_mesh_button.isEnabled()
            )
        for name in (
            "reload_button",
            "save_button",
            "generate_script_button",
            "preview_button",
            "open_file_folder_button",
            "copy_file_path_button",
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.setVisible(button.isEnabled())

    def _refresh_workflow_actions(self):
        self._refresh_solidify_availability()
        if not hasattr(self, "workflow_preview_actions") or getattr(self, "_refreshing_workflow", False):
            return
        self._refreshing_workflow = True
        try:
            imprint = getattr(self, "_envelope_poly", None) is not None
            solid = getattr(self, "_solidify_preview_context", None) is not None
            preview = self.preview_poly is not None and self.preview_poly is not self.source_poly
            pending = imprint or preview
            label = "cast" if imprint else "solid" if solid else "optimization"
            ready = (getattr(self, "_envelope_settings_key", None) == self._imprint_preview_settings_key()
                     if imprint else bool(preview and self.pass_ready))
            self.workflow_preview_actions.setVisible(pending)
            report = getattr(self, "_solidify_preview_context", None)
            completion = report.get("completion") if report is not None and not imprint else None
            self.workflow_solid_summary.setVisible(completion is not None)
            if completion is not None:
                self.solidify_result_labels["retained_percent"].setText(f"Surface kept\n{completion['retained_percent']:.2f}%")
                self.solidify_result_labels["retained_percent"].setToolTip(
                    f"{fmt_count(completion['retained_faces'])} of {fmt_count(completion['source_faces'])} measured triangles remain at STL precision. Percentage is weighted by measured surface area, not triangle count. It does not measure accuracy of reconstructed gaps.")
                self.solidify_result_labels["rebuilt_percent"].setText(f"Rebuilt area\n{completion['rebuilt_percent']:.2f}%")
                unit = self.units.currentText()
                area = completion['added_area'] / UNIT_MM[unit] ** 2
                self.solidify_result_labels["rebuilt_percent"].setToolTip(
                    f"{area:.3f} {unit}? of new surface, {fmt_count(completion['added_faces'])} triangles. Percentage of the completed surface reconstructed rather than measured. The scan cannot reveal the true missing shape.")
                self.solidify_result_labels["open_edges"].setText(f"Open edges\n{fmt_count(completion['open_edges'])}")
                self.solidify_result_labels["open_edges"].setToolTip("Boundary edges remaining in the solidified region. Zero means no open borders; the Solidify operation also validates intersections and topology before returning a result.")
            self.workflow_apply_button.setText(f"Apply {label}")
            self.workflow_apply_button.setEnabled(ready and not self.busy)
            self.workflow_save_preview_button.setEnabled(ready and not self.busy)
            self.cancel_preview_button.setVisible(pending)
            self.cancel_preview_button.setEnabled(pending and not self.busy)
        finally:
            self._refreshing_workflow = False

    def _apply_workflow_preview(self, _checked=False):
        if getattr(self, "_envelope_poly", None) is not None:
            self._replace_with_envelope()
        else:
            self._apply_optimized_pass()
        self._update_action_row_visibility()

    def _cancel_workflow_preview(self, _checked=False):
        if self.busy:
            return
        if getattr(self, "_envelope_poly", None) is not None:
            self._clear_envelope()
        elif getattr(self, "_solidify_preview_context", None) is not None:
            self._cancel_solidify_preview()
        else:
            self._cancel_optimized_preview()
        self._update_action_row_visibility()

    def _save_workflow_preview(self, _checked=False):
        if self.busy or not self.workflow_save_preview_button.isEnabled():
            return
        if getattr(self, "_envelope_poly", None) is not None:
            self._save_envelope()
            return
        if self.preview_poly is None or self.preview_poly is self.source_poly:
            return
        directory = self.source_path.parent if self.source_path else Path(self._file_open_directory())
        from mesh_export_ui import choose_mesh_export
        choice = choose_mesh_export(self, "Save preview", directory / "preview.stl", self.preview_poly)
        if choice is None:
            return
        destination, merge = choice
        self._export_meshes([(destination, self.preview_poly)], merge)

    def _sync_modified_mesh_actions(self) -> None:
        """Keep save and reload availability tied to the current history state."""
        available = bool(self.mesh_modified and self.source_poly is not None)
        self.save_button.setEnabled(available)
        self.reload_button.setEnabled(available)
        self.generate_script_button.setEnabled(available and bool(
            self.optimization_script_passes or self.script_base_poly is not None
            or self.export_calibration_transform is not None))
        self._update_action_row_visibility()

    def _toggle_full_screen(self) -> None:
        """Toggle immersive full screen while preserving a maximized return state."""
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()
        QTimer.singleShot(0, self._sync_full_screen_button)
        QTimer.singleShot(0, self._schedule_viewport_layout_refresh)
        QTimer.singleShot(100, self._schedule_viewport_layout_refresh)

    def _toggle_maximized(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        elif self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _window_resize_edges(self, point: QPoint):
        edges = Qt.Edge(0)
        if point.x() < 6:
            edges |= Qt.Edge.LeftEdge
        elif point.x() >= self.width() - 6:
            edges |= Qt.Edge.RightEdge
        if point.y() < 6:
            edges |= Qt.Edge.TopEdge
        elif point.y() >= self.height() - 6:
            edges |= Qt.Edge.BottomEdge
        return edges

    def nativeEvent(self, event_type, message):
        if os.name == "nt" and not self.isMaximized() and not self.isFullScreen():
            from ctypes import wintypes
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == 0x0084:  # WM_NCHITTEST: preserve native resize cursors/borders.
                rect = wintypes.RECT()
                if ctypes.windll.user32.GetWindowRect(wintypes.HWND(int(self.winId())), ctypes.byref(rect)):
                    x = ctypes.c_short(msg.lParam & 0xFFFF).value
                    y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
                    border = max(6, round(6 * self.devicePixelRatioF()))
                    left, right = x < rect.left + border, x >= rect.right - border
                    top, bottom = y < rect.top + border, y >= rect.bottom - border
                    hit = (13 if left else 14 if right else 12) if top else (
                        (16 if left else 17 if right else 15) if bottom else
                        (10 if left else 11 if right else 0))
                    if hit:
                        return True, hit
        return super().nativeEvent(event_type, message)

    def _sync_full_screen_button(self) -> None:
        sequence = getattr(self, "shortcuts", {}).get("toggle_fullscreen", "F11")
        shortcut = getattr(self, "fullscreen_shortcut", None)
        if shortcut is not None:
            shortcut.setKey(QKeySequence(sequence))
        button = getattr(self, "fullscreen_button", None)
        if button is not None:
            label = "Exit full screen" if self.isFullScreen() else "Full screen"
            button.setToolTip(f"{label} ({sequence})" if sequence else label)
            button.setAccessibleName(label)
        maximize = getattr(self, "maximize_button", None)
        if maximize is not None:
            restored = self.isMaximized() or self.isFullScreen()
            label = "Exit full screen" if self.isFullScreen() else "Restore" if restored else "Maximize"
            maximize.setToolTip(label)
            maximize.setAccessibleName(label)
            self._apply_theme_icon(maximize, "restore" if restored else "maximize")

    def _shortcut_callback(self, action: str):
        callbacks = {
            "save": self._save_shortcut_activated,
            "undo": self._undo_working_mesh,
            "redo": self._redo_working_mesh,
            "redo_alt": self._redo_working_mesh,
            "selection_optimize": lambda: self._selection_shortcut("optimize"),
            "selection_crop": lambda: self._selection_shortcut("crop"),
            "selection_add": lambda: self._selection_shortcut("add"),
            "selection_delete": (
                (lambda: self._selection_shortcut("delete"))
                if self._selected_triangle_count()
                else None
            ),
            "selection_clear": (
                self._clear_selection_and_ruler
                if self._selected_triangle_count() or getattr(self, "crop_points", []) or self.measure_points or self.measure_hover_world is not None
                else None
            ),
            "display_shaded": lambda: self._set_display_shortcut("Shaded"),
            "display_wireframe": lambda: self._set_display_shortcut("Wireframe"),
            "display_vertices": lambda: self._set_display_shortcut("Vertices"),
            "display_density": lambda: self._set_display_shortcut("Density"),
            "toggle_fullscreen": lambda: MainWindow._toggle_full_screen(self),
            "view_left": lambda: self._set_standard_view("left"),
            "view_front": lambda: self._set_standard_view("front"),
            "view_right": lambda: self._set_standard_view("right"),
            "view_top": lambda: self._set_standard_view("top"),
            "view_back": lambda: self._set_standard_view("back"),
            "view_bottom": lambda: self._set_standard_view("bottom"),
            "set_view_left": lambda: self._set_current_view_as("left"),
            "set_view_front": lambda: self._set_current_view_as("front"),
            "set_view_right": lambda: self._set_current_view_as("right"),
            "set_view_top": lambda: self._set_current_view_as("top"),
            "set_view_back": lambda: self._set_current_view_as("back"),
            "set_view_bottom": lambda: self._set_current_view_as("bottom"),
        }
        if action in callbacks:
            return callbacks[action]
        if (
            self.source_poly is None
            or self.busy
            or self.crop_start is not None
        ):
            return None
        navigation = {
            "orbit_left": ("left", False, False),
            "orbit_right": ("right", False, False),
            "orbit_up": ("up", False, False),
            "orbit_down": ("down", False, False),
            "pan_left": ("left", True, False),
            "pan_right": ("right", True, False),
            "pan_up": ("up", True, False),
            "pan_down": ("down", True, False),
            "zoom_in": ("up", True, True),
            "zoom_out": ("down", True, True),
            "zoom_in_alt": ("up", True, True),
            "zoom_out_alt": ("down", True, True),
            "roll_counterclockwise": ("left", True, True),
            "roll_clockwise": ("right", True, True),
        }
        if action not in navigation:
            return None
        direction, control, shift = navigation[action]
        return lambda: self._queue_keyboard_navigation(direction, control, shift)

    def _set_display_shortcut(self, mode: str) -> None:
        index = self.display_type.findData(mode)
        if index >= 0:
            self.display_type.setCurrentIndex(index)
        self.settings.setValue("display/type", mode)
        self.status.setText(f"Display: {mode}")

    def _display_mode(self) -> str:
        return str(self.display_type.currentData() or "Shaded")

    def _refresh_display_mode_labels(self) -> None:
        current_mode = self._display_mode()
        with QSignalBlocker(self.display_type):
            for index, mode in enumerate(DISPLAY_MODES):
                shortcut = self.shortcuts[f"display_{mode.lower()}"]
                self.display_type.setItemText(index, f"{mode} ({shortcut})")
            current_index = self.display_type.findData(current_mode)
            if current_index >= 0:
                self.display_type.setCurrentIndex(current_index)

    def _apply_tools_side(self, side: str) -> None:
        self.tools_side = "Left" if side == "Left" else "Right"
        self.outer_layout.removeWidget(self.controls_widget)
        self.outer_layout.removeWidget(self.viewport_column)
        if self.tools_side == "Left":
            self.outer_layout.addWidget(self.controls_widget)
            self.outer_layout.addWidget(self.viewport_column, 1)
        else:
            self.outer_layout.addWidget(self.viewport_column, 1)
            self.outer_layout.addWidget(self.controls_widget)
        metric_row = getattr(self, "metrics_toggle_row", None)
        if metric_row is not None:
            metric_row.setDirection(
                QBoxLayout.Direction.LeftToRight
                if self.tools_side == "Right"
                else QBoxLayout.Direction.RightToLeft
            )

    def _apply_appearance(self) -> None:
        """Apply the selected visual palette without changing controls or behavior."""
        app = QApplication.instance()
        if app is None:
            return
        theme = appearance_theme(self.appearance_theme_name)
        viewport_theme = self._viewport_theme()
        app.setStyleSheet(app_stylesheet(self.appearance_theme_name, self._font_pixel_size()))
        palette = app.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(theme["window"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(theme["field"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(theme["field"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(theme["text"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme["text"]))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme["panel"]))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme["text"]))
        app.setPalette(palette)
        QToolTip.setPalette(palette)
        graph = getattr(self, "geometry_activity_graph", None)
        if graph is not None:
            graph.apply_appearance(self.appearance_theme_name)
            for resource, bar in getattr(self, 'metric_bars', {}).items():
                series = 'gpu' if resource in ('gpu', 'vram') else 'cpu'
                color = graph.plot_colors[series].name()
                bar.setStyleSheet(
                    'QProgressBar { text-align: right; } '
                    f'QProgressBar::chunk {{ background: {color}; }}'
                )
            history_palette = self.metrics_history_scroll.viewport().palette()
            history_palette.setColor(QPalette.ColorRole.Window, QColor(theme["panel"]))
            history_palette.setColor(QPalette.ColorRole.Base, QColor(theme["panel"]))
            self.metrics_history_scroll.viewport().setPalette(history_palette)
            self.metrics_history_scroll.viewport().setAutoFillBackground(True)
        self._refresh_brand_logo()
        for target, icon_name in getattr(self, "_theme_icon_targets", []):
            if target is not None:
                self._apply_theme_icon(target, icon_name)
        self._refresh_theme_overlays(theme, viewport_theme)
        density_base = self._density_palette_key(viewport_theme)
        if (
            hasattr(self, "density_cache")
            and getattr(self, "_density_palette_base", None) != density_base
        ):
            self._density_palette_base = density_base
            # A theme switch can arrive while an old-palette density job is still running.
            # Invalidate its token before requesting the palette for the newly selected viewport.
            self.density_generation += 1
            if self.active_operation == "Analyzing point density":
                if self.operation_cancel_event is not None:
                    self.operation_cancel_event.set()
                self._end_mesh_operation()
            self.density_cache.clear()
            if hasattr(self, "display_type") and self._display_mode() == "Density":
                QTimer.singleShot(
                    0,
                    lambda: self._show_poly(
                        self._active_density_mesh()[0], reset_camera=False
                    )
                    if self._active_density_mesh()[0] is not None
                    else None,
                )
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            viewport_color = QColor(viewport_theme["viewport"])
            renderer.SetBackground(
                viewport_color.redF(), viewport_color.greenF(), viewport_color.blueF()
            )
            render_window = self.vtk_widget.GetRenderWindow()
            if render_window is not None:
                render_window.Render()
        self._update_toolbox_width()
        self._apply_native_title_bar(theme)
        if hasattr(self, "tooltip_filter"):
            self.tooltip_filter.apply_appearance(self.appearance_theme_name)
        QTimer.singleShot(0, self._position_viewport_overlays)

    def _custom_density_colors(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        if self.viewport_appearance_name != "custom":
            return None
        left = QColor(self.custom_density_gradient_left)
        right = QColor(self.custom_density_gradient_right)
        if not left.isValid() or not right.isValid():
            return None
        return (
            (left.red(), left.green(), left.blue()),
            (right.red(), right.green(), right.blue()),
        )

    def _density_palette_key(self, viewport_theme: dict[str, str]) -> tuple[object, ...]:
        return (viewport_theme["viewport"], self._custom_density_colors())

    def _viewport_theme(self) -> dict[str, str]:
        """Return the palette used by the render area and all of its floating aids."""
        name = self.appearance_theme_name
        if self.viewport_appearance_name not in {"match", "custom"}:
            name = self.viewport_appearance_name
        return appearance_theme(name)

    def _font_pixel_size(self) -> int:
        return (11, 12, 13, 14, 15)[self.font_size_position]

    def _update_toolbox_width(self) -> None:
        controls = getattr(self, "controls_widget", None)
        if controls is None:
            return
        # Wider text needs room before the toolbox starts wrapping labels or clipping controls.
        width = 384 + (self._font_pixel_size() - 12) * 28
        controls.setFixedWidth(max(356, min(480, width)))
        content = getattr(self, "controls_content", None)
        if content is not None:
            content.updateGeometry()

    def _set_theme_icon(self, target: QWidget, icon_name: str) -> None:
        """Register a control or label for palette-aware icon refreshes."""
        if not hasattr(self, "_theme_icon_targets"):
            self._theme_icon_targets: list[tuple[QWidget, str]] = []
        self._theme_icon_targets.append((target, icon_name))
        self._apply_theme_icon(target, icon_name)

    def _apply_theme_icon(self, target: QWidget, icon_name: str) -> None:
        if target is getattr(self, "maximize_button", None):
            icon_name = "restore" if self.isMaximized() or self.isFullScreen() else "maximize"
        theme = appearance_theme(self.appearance_theme_name)
        tone = str(target.property("iconTone") or "accent")
        color = theme.get(tone, theme["accent"])
        if isinstance(target, ViewButton):
            color = selected_control_text(theme) if target.isChecked() or target.isDown() else theme["text"]
        icon = draw_ui_icon(icon_name, color)
        if isinstance(target, QLabel):
            target.setPixmap(icon.pixmap(18, 18))
        elif isinstance(target, QPushButton):
            target.setIcon(icon)
            target.setIconSize(QSize(17, 17))

    def _refresh_brand_logo(self) -> None:
        logo = getattr(self, "brand_logo", None)
        logo_path = resource_path("assets/meshmill-logo.svg")
        if logo is None or not logo_path.exists():
            return
        theme = appearance_theme(getattr(self, "appearance_theme_name", DEFAULT_APPEARANCE))
        try:
            svg = logo_path.read_text(encoding="utf-8")
            svg = svg.replace("#0B1B34", theme["text"]).replace("#6B7280", theme["muted"])
            pixmap = QPixmap()
            pixmap.loadFromData(svg.encode("utf-8"), "SVG")
            logo.setPixmap(pixmap.scaled(310, 78, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except OSError:
            return

    def _refresh_theme_overlays(
        self, theme: dict[str, str], viewport_theme: dict[str, str]
    ) -> None:
        """Restyle floating viewport aids that own a local stylesheet."""
        floating = (
            "QFrame {{ color: {text}; background: {overlay}; border: 1px solid {border}; "
            "border-radius: 8px; }} QLabel {{ color: {text}; background: transparent; border: 0; }} "
            "QComboBox, QPushButton {{ color: {text}; background: {field}; border: 1px solid {border}; "
            "border-radius: 6px; }} QPushButton:hover, QComboBox:hover {{ background: {soft}; "
            "border-color: {accent}; }}"
        ).format(
            text=theme["text"], overlay=theme["overlay"], border=theme["border"],
            field=theme["field"], soft=theme["accent_soft"], accent=theme["accent"],
        )
        for name in ("measure_info_frame", "crop_info_frame"):
            frame = getattr(self, name, None)
            if frame is not None:
                frame.setStyleSheet(floating)
        legend = getattr(self, "density_legend", None)
        if legend is not None:
            legend.set_theme(theme, viewport_theme)
            legend.set_gradient_colors(self._custom_density_colors())
            legend.setText(
                f"POINT DENSITY   <span style='color:{theme['muted']}'>Sparse</span>  "
                "<span style='color:#199AD6'>●</span> <span style='color:#57E0B8'>●</span> "
                "<span style='color:#FACC15'>●</span> <span style='color:#E23730'>● Dense</span>"
            )
            legend.setStyleSheet(
                f"background: {theme['overlay']}; color: {theme['text']}; "
                f"border: 1px solid {theme['border']}; border-radius: 7px; padding: 7px;"
            )
            legend.setStyleSheet(f"background: {viewport_theme['viewport']};")
            self._apply_rounded_mask(legend, 9)
        indicator = getattr(self, "orientation_indicator", None)
        if indicator is not None:
            indicator.set_theme(theme, viewport_theme)
            indicator.setStyleSheet(f"background: {viewport_theme['viewport']};")
            self._apply_rounded_mask(indicator, 9)
        scale_bar = getattr(self, "scale_bar", None)
        if scale_bar is not None:
            scale_bar.set_theme(theme, viewport_theme)
            self._apply_rounded_mask(scale_bar, 8)
        view_pad = getattr(self, "view_pad", None)
        if view_pad is not None:
            view_pad.setStyleSheet(
                f"QFrame#viewPad {{ background: {theme['panel']}; border: 1px solid {theme['border']}; border-radius: 10px; }} "
                f"QFrame#viewPad QPushButton {{ background: {theme['surface']}; color: {theme['text']}; border: 1px solid {theme['border']}; border-radius: 6px; }} "
                f"QFrame#viewPad QPushButton:checked, QFrame#viewPad QPushButton:pressed {{ background: {theme['accent']}; color: {selected_control_text(theme)}; border-color: {theme['accent_hover']}; }}"
            )
            self._apply_rounded_mask(view_pad, 10)
        loading = getattr(self, "loading_frame", None)
        if loading is not None:
            loading.setStyleSheet(
                f"QFrame#loadingPanel {{ background: {theme['panel']}; border: 1px solid {theme['border']}; border-radius: 14px; }} "
                f"QFrame#loadingPanel QLabel {{ color: {theme['text']}; background: transparent; border: 0; }} "
                f"QFrame#loadingPanel QLabel[loadingMetric='true'] {{ background: transparent; border: 1px solid {theme['border']}; border-radius: 5px; padding: 4px 3px; }} "
                f"QFrame#loadingPanel QProgressBar {{ background: {theme['field']}; color: {theme['text']}; border: 1px solid {theme['border']}; }} "
                f"QFrame#loadingPanel QProgressBar::chunk {{ background: {theme['accent']}; }}"
            )
        loupe = getattr(self, "loupe_frame", None)
        if loupe is not None:
            loupe.setStyleSheet(
                f"QFrame {{ border: 3px solid {theme['accent']}; border-radius: 180px; background: #080b10; }}"
            )

    def _install_dialog_title_bar(self, dialog: QDialog) -> None:
        if dialog.property("themedCaption") or dialog.layout() is None:
            return
        dialog.setProperty("themedCaption", True)
        dialog.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        header = WindowTitleBar(dialog)
        header.setObjectName("dialogTitleBar")
        header.setProperty("windowActive", dialog.isActiveWindow())
        row = QHBoxLayout(header)
        row.setContentsMargins(10, 0, 0, 0)
        row.setSpacing(8)
        icon = QLabel(header)
        icon.setPixmap(self.windowIcon().pixmap(18, 18))
        icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        row.addWidget(icon)
        title = QLabel(dialog.windowTitle(), header)
        title.setObjectName("dialogCaptionText")
        title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        dialog.windowTitleChanged.connect(title.setText)
        row.addWidget(title, 1)
        close = QPushButton(header)
        close.setObjectName("dialogCaptionClose")
        close.setAccessibleName("Close")
        close.setToolTip("Close")
        close.setAutoDefault(False)
        close.clicked.connect(dialog.close)
        row.addWidget(close)
        dialog.layout().setMenuBar(header)
        header.setMinimumHeight(max(34, self._font_pixel_size() + 18))
        self._style_dialog_title_bar(dialog, appearance_theme(self.appearance_theme_name))

    def _style_dialog_title_bar(self, dialog: QWidget, theme: dict[str, str]) -> None:
        header = dialog.findChild(WindowTitleBar, "dialogTitleBar")
        if header is None:
            return
        header.setStyleSheet(
            f"QFrame#dialogTitleBar {{background:{theme['window']};border:0;"
            f"border-bottom:1px solid {theme['border']};border-radius:0;}}"
            f"QFrame#dialogTitleBar QLabel {{color:{theme['text']};background:transparent;"
            f"border:0;font-size:{self._font_pixel_size()}px;}}"
            f"QPushButton#dialogCaptionClose {{background:transparent;border:0;border-radius:0;"
            f"padding:0;min-width:38px;min-height:32px;}}"
            "QPushButton#dialogCaptionClose:hover {background:#C42B1C;}"
            f"QFrame#dialogTitleBar[windowActive='false'] {{background:{inactive_caption_color(theme)};}}"
            f"QFrame#dialogTitleBar[windowActive='false'] QLabel {{color:{theme['muted']};}}"
        )
        close = header.findChild(QPushButton, "dialogCaptionClose")
        close.setIcon(draw_ui_icon("caption-close", theme["text"]))
        close.setIconSize(QSize(17, 17))

    def _apply_native_title_bar(
        self, theme: dict[str, str], target: QWidget | None = None
    ) -> None:
        """Style native captions without replacing standard window controls."""
        if os.name != "nt":
            return
        try:
            dark = int(self.appearance_theme_name == "Dark Focus")
            value = ctypes.c_int(dark)
            targets = [target] if target is not None else [
                self, *[dialog for dialog in self.findChildren(QDialog) if dialog.isVisible()]
            ]
            dwmapi = ctypes.WinDLL("dwmapi")
            # 20 is supported by current Windows 10 and Windows 11. Older versions simply
            # ignore it, leaving the normal native caption intact.
            def colorref(hex_color: str) -> ctypes.c_uint:
                color = QColor(hex_color)
                return ctypes.c_uint(color.red() | (color.green() << 8) | (color.blue() << 16))

            caption = colorref(theme["window"])
            text = colorref(theme["text"])
            border = colorref(theme["border"])
            for widget in targets:
                self._style_dialog_title_bar(widget, theme)
                hwnd = ctypes.c_void_p(int(widget.winId()))
                dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
                for attribute, native_color in ((35, caption), (36, text), (34, border)):
                    dwmapi.DwmSetWindowAttribute(
                        hwnd, attribute, ctypes.byref(native_color), ctypes.sizeof(native_color)
                    )
        except (AttributeError, OSError, TypeError):
            pass

    def _show_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("MeshMill Settings")
        dialog.setModal(True)
        dialog.resize(620, 760)
        outer = QVBoxLayout(dialog)
        search = QLineEdit(dialog)
        search.setObjectName("preferencesSearch")
        search.setPlaceholderText("Search preferences")
        search.setAccessibleName("Search preferences")
        search.setClearButtonEnabled(True)
        search.setMaximumWidth(220)
        no_matches = QLabel("No matching preferences", dialog)
        no_matches.setObjectName("preferencesNoMatches")
        no_matches.hide()
        outer.addWidget(no_matches)
        sections = []
        scroll = QScrollArea(dialog)
        scroll.setObjectName("preferencesScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        def section(title: str) -> QVBoxLayout:
            frame = QFrame(content)
            sections.append(frame)
            frame.setObjectName("settingsSection")
            section_layout = QVBoxLayout(frame)
            section_layout.setContentsMargins(14, 12, 14, 12)
            heading = QLabel(title, frame)
            heading.setStyleSheet("font-weight: 700; border: 0;")
            section_layout.addWidget(heading)
            layout.addWidget(frame)
            return section_layout

        updates_layout = section("Updates")
        automatic_updates = QCheckBox("Check for updates when MeshMill opens", dialog)
        automatic_updates.setChecked(self.update_checks_enabled)
        automatic_updates.setToolTip(
            "Runs one GitHub release check inside MeshMill shortly after the app opens. "
            "MeshMill does not install an updater, service, scheduled task, startup item, or "
            "background process. Nothing runs while MeshMill is closed. A newer release prompts "
            "you once MeshMill is idle unless automatic installation is enabled."
        )
        updates_layout.addWidget(automatic_updates)
        auto_install = QCheckBox("Automatically install updates", dialog)
        auto_install.setObjectName("automaticUpdateInstall")
        auto_install.setChecked(self.auto_install_updates)
        auto_install.setEnabled(automatic_updates.isChecked())
        auto_install.setToolTip(
            "Download, verify, and install updates without another confirmation when MeshMill "
            "is idle and the mesh has no unsaved changes. Reopens the current STL after updating. "
            "Requires startup update checking; no separate background service is installed.")
        def sync_auto_install(enabled):
            auto_install.setEnabled(enabled)
            if not enabled:
                auto_install.setChecked(False)
        automatic_updates.toggled.connect(sync_auto_install)
        updates_layout.addWidget(auto_install)
        auto_install_scope = ArrowComboBox(dialog)
        auto_install_scope.setObjectName("automaticUpdateScope")
        auto_install_scope.addItem("Hotfixes and maintenance only", "maintenance")
        auto_install_scope.addItem("All releases (including major)", "all")
        auto_install_scope.setCurrentIndex(max(0, auto_install_scope.findData(self.auto_install_scope)))
        auto_install_scope.setEnabled(auto_install.isChecked())
        auto_install.toggled.connect(auto_install_scope.setEnabled)
        auto_install_scope.setToolTip(
            "Maintenance installs patch updates within the same release series, such as "
            "0.3.2 to 0.3.3. Updates to 0.4.0 or 1.0.0 still ask first. All releases also "
            "allows minor and major upgrades. Preview releases are excluded.")
        automatic_scope_form = QFormLayout()
        automatic_scope_form.addRow("Release types", auto_install_scope)
        updates_layout.addLayout(automatic_scope_form)
        update_row = QHBoxLayout()
        check_updates_button = QPushButton("Check now", dialog)
        check_updates_button.setProperty("quiet", True)
        check_updates_button.setToolTip(
            "Check GitHub once for a newer MeshMill release. This does not change the automatic setting."
        )
        updates_status = QLabel(self.update_check_status, dialog)
        updates_status.setProperty("muted", True)
        install_update_button = QPushButton("Install update", dialog)
        install_update_button.setObjectName("installUpdateButton")
        update_row.addWidget(check_updates_button)
        update_row.addWidget(install_update_button)
        update_row.addWidget(updates_status, 1)
        updates_layout.addLayout(update_row)

        def refresh_update_status() -> None:
            updates_status.setText(self.update_check_status)
            check_updates_button.setEnabled(self.update_future is None)
            available = self._update_available()
            install_update_button.setVisible(available)
            install_update_button.setEnabled(
                available and not self.busy and self.update_install_future is None
            )
            installable = update_asset_for_current_platform(self.latest_release)
            install_update_button.setText(
                f"Install {self.latest_release_version}" if installable
                else "Open downloads"
            )
            install_update_button.setToolTip(
                "Download and verify the update, then install it and reopen the current STL. "
                "Save changes and finish or cancel the current operation first."
                if installable else "Open the release downloads for your operating system."
            )

        check_updates_button.clicked.connect(lambda: self._check_for_updates(force=True))
        install_update_button.clicked.connect(self._install_latest_release)
        refresh_update_status()
        updates_timer = QTimer(dialog)
        updates_timer.setInterval(250)
        updates_timer.timeout.connect(refresh_update_status)
        updates_timer.start()

        interface_layout = section("Interface")
        form = QFormLayout()

        language = ArrowComboBox(dialog)
        language.addItem("System default", "system")
        for code, name in locale_manager.available_locales():
            language.addItem(name, code)
        locale_index = language.findData(self.locale_preference)
        language.setCurrentIndex(max(0, locale_index))
        language.setToolTip("Application language.")
        form.addRow("Language", language)

        open_mode = ArrowComboBox(dialog)
        open_mode.setObjectName("fileOpenMode")
        for label, value in (("Remember last folder", "last"),
                             ("Smart open (parent of last folder)", "smart"),
                             ("Specific folder", "fixed")):
            open_mode.addItem(label, value)
        open_mode.setCurrentIndex(max(0, open_mode.findData(
            self.settings.value("files/open_mode", "last"))))
        open_mode.setToolTip("Choose the starting folder for the Open dialog.")
        form.addRow("File open mode", open_mode)
        open_folder_row = QWidget(dialog)
        open_folder_layout = QHBoxLayout(open_folder_row)
        open_folder_layout.setContentsMargins(0, 0, 0, 0)
        open_folder = QLineEdit(str(self.settings.value("files/open_folder", "")), dialog)
        open_folder.setObjectName("fileOpenFolder")
        browse_open_folder = QPushButton("Browse…", dialog)
        def choose_open_folder():
            chosen = QFileDialog.getExistingDirectory(dialog, "Choose opening folder", open_folder.text())
            if chosen:
                open_folder.setText(chosen)
        browse_open_folder.clicked.connect(choose_open_folder)
        open_folder_layout.addWidget(open_folder, 1)
        open_folder_layout.addWidget(browse_open_folder)
        open_folder_row.setEnabled(open_mode.currentData() == "fixed")
        open_mode.currentIndexChanged.connect(
            lambda _: open_folder_row.setEnabled(open_mode.currentData() == "fixed"))
        form.addRow("Opening folder", open_folder_row)
        recent_files_enabled = QCheckBox("Enable recent files menu", dialog)
        recent_files_enabled.setObjectName("recentFilesEnabled")
        recent_files_enabled.setChecked(self.recent_files_enabled)
        recent_files_enabled.setToolTip(
            "Show an icon beside Open and remember up to ten successfully opened files "
            "on this computer. Off by default. Disabling clears this recent-file history.")
        form.addRow("Recent files", recent_files_enabled)
        if platform.system().lower() == "windows":
            associate_stl = QPushButton("Choose default app…", dialog)
            associate_stl.setObjectName("stlAssociationSettings")
            associate_stl.setToolTip(
                "Open Windows default-app settings to choose the app for .stl files. "
                "Installing MeshMill adds Open With without replacing your current default. "
                "No default changes until you choose one in Windows.")
            associate_stl.clicked.connect(self._open_stl_association_settings)
            form.addRow("STL file association", associate_stl)

        appearance = ArrowComboBox(dialog)
        for appearance_name in APPEARANCE_THEMES:
            appearance.addItem(appearance_name, appearance_name)
        appearance.setCurrentIndex(
            max(0, appearance.findData(self.appearance_theme_name))
        )
        appearance.setToolTip(
            "Changes MeshMill window colors, panel contrast, and controls. It does not change "
            "tools, layout, or mesh behavior."
        )
        form.addRow("Theme", appearance)

        order_editor = QWidget(dialog)
        order_layout = QHBoxLayout(order_editor)
        order_layout.setContentsMargins(0, 0, 0, 0)
        section_order = QListWidget(order_editor)
        section_order.setObjectName("toolboxOrder")
        section_order.addItems(self.toolbox_order)
        section_order.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        section_order.setFixedHeight(section_order.fontMetrics().height() * 7 + 12)
        section_order.setCurrentRow(0)
        section_order.setToolTip("Drag sections or use Move up and Move down. Active tool follows the expanded tool; Other tools contains the collapsed tools.")
        order_layout.addWidget(section_order, 1)
        order_buttons = QVBoxLayout()
        def move_section(delta):
            row = section_order.currentRow()
            destination = row + delta
            if 0 <= destination < section_order.count():
                item = section_order.takeItem(row)
                section_order.insertItem(destination, item)
                section_order.setCurrentRow(destination)
        for label, delta in (("Move up", -1), ("Move down", 1)):
            button = QPushButton(label, order_editor)
            button.clicked.connect(lambda _checked=False, step=delta: move_section(step))
            order_buttons.addWidget(button)
        order_buttons.addStretch(1)
        order_layout.addLayout(order_buttons)
        form.addRow("Toolbox order", order_editor)

        viewport_appearance = ArrowComboBox(dialog)
        viewport_appearance.addItem("Match theme", "match")
        for appearance_name in APPEARANCE_THEMES:
            viewport_appearance.addItem(appearance_name, appearance_name)
        viewport_appearance.addItem("Custom", "custom")
        viewport_appearance.setCurrentIndex(
            max(0, viewport_appearance.findData(self.viewport_appearance_name))
        )
        viewport_appearance.setToolTip(
            "Sets the viewport background and its floating controls. Match theme uses the "
            "same palette as the rest of MeshMill."
        )
        form.addRow("Viewport", viewport_appearance)

        gradient_colors = QWidget(dialog)
        gradient_row = QHBoxLayout(gradient_colors)
        gradient_row.setContentsMargins(0, 0, 0, 0)
        gradient_row.setSpacing(6)
        custom_gradient = {
            "left": self.custom_density_gradient_left,
            "right": self.custom_density_gradient_right,
        }
        left_gradient_button = QPushButton(gradient_colors)
        right_gradient_button = QPushButton(gradient_colors)

        def refresh_gradient_buttons() -> None:
            for side, button, label in (
                ("left", left_gradient_button, "Sparse"),
                ("right", right_gradient_button, "Dense"),
            ):
                color = QColor(custom_gradient[side])
                luminance = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000
                text = "#102A43" if luminance > 156 else "#FFFFFF"
                button.setText(f"{label}: {color.name().upper()}")
                button.setStyleSheet(
                    f"background: {color.name()}; color: {text}; border-color: {color.name()};"
                )

        def choose_gradient_color(side: str) -> None:
            color = QColorDialog.getColor(
                QColor(custom_gradient[side]), dialog, f"{side.title()} density color"
            )
            if color.isValid():
                custom_gradient[side] = color.name().upper()
                refresh_gradient_buttons()

        left_gradient_button.clicked.connect(lambda: choose_gradient_color("left"))
        right_gradient_button.clicked.connect(lambda: choose_gradient_color("right"))
        gradient_row.addWidget(left_gradient_button)
        gradient_row.addWidget(right_gradient_button)
        form.addRow("Density gradient", gradient_colors)
        gradient_label = form.labelForField(gradient_colors)

        def sync_custom_gradient_visibility() -> None:
            custom = viewport_appearance.currentData() == "custom"
            gradient_colors.setVisible(custom)
            gradient_label.setVisible(custom)

        gradient_colors._preferences_search_allowed = lambda: viewport_appearance.currentData() == "custom"
        gradient_label._preferences_search_allowed = gradient_colors._preferences_search_allowed

        viewport_appearance.currentIndexChanged.connect(sync_custom_gradient_visibility)
        refresh_gradient_buttons()
        sync_custom_gradient_visibility()

        font_size_control = QWidget(dialog)
        font_size_row = QHBoxLayout(font_size_control)
        font_size_row.setContentsMargins(0, 0, 0, 0)
        font_size_row.setSpacing(8)
        font_size = QSlider(Qt.Orientation.Horizontal, font_size_control)
        font_size.setRange(0, 4)
        font_size.setSingleStep(1)
        font_size.setPageStep(1)
        font_size.setTickInterval(1)
        font_size.setTickPosition(QSlider.TickPosition.TicksBelow)
        font_size.setValue(self.font_size_position)
        font_size.setToolTip(
            "Five fixed interface sizes. System default is the second position. "
            "The toolbox widens with larger text so controls and descriptions stay readable."
        )
        font_size_name = QLabel(font_size_control)
        font_size_name.setMinimumWidth(102)
        font_size_names = ("Smaller", "System default", "Larger", "Large", "Largest")

        def update_font_size_name(value: int) -> None:
            font_size_name.setText(font_size_names[max(0, min(4, value))])

        font_size.valueChanged.connect(update_font_size_name)
        update_font_size_name(font_size.value())
        font_size_row.addWidget(font_size, 1)
        font_size_row.addWidget(font_size_name)
        form.addRow("Font size", font_size_control)

        tools_side = ArrowComboBox(dialog)
        tools_side.addItems(["Right", "Left"])
        tools_side.setCurrentText(self.tools_side)
        tools_side.setToolTip("Choose which side of the window contains the tools pane.")
        form.addRow("Tools pane", tools_side)

        default_units = ArrowComboBox(dialog)
        default_units.addItems(list(UNIT_MM))
        default_units.setCurrentText(self.default_units)
        default_units.setToolTip("Units selected when MeshMill starts. STL coordinates are unchanged.")
        form.addRow("Default units", default_units)

        interface_layout.addLayout(form)

        overlays_layout = section("Viewport overlays")
        overlays_form = QFormLayout()
        overlay_controls: dict[str, tuple[QCheckBox, ArrowComboBox]] = {}
        overlay_refreshing = {"active": False}

        def refresh_overlay_corners() -> None:
            if overlay_refreshing["active"]:
                return
            overlay_refreshing["active"] = True
            try:
                selected = {
                    key: str(combo.currentData() or VIEWPORT_OVERLAYS[key][1])
                    for key, (_show, combo) in overlay_controls.items()
                }
                for key, (show, combo) in overlay_controls.items():
                    occupied = {
                        selected[other_key]
                        for other_key, (other_show, _other_combo) in overlay_controls.items()
                        if other_key != key and other_show.isChecked()
                    }
                    allowed = (
                        [corner for corner in VIEWPORT_CORNERS if corner not in occupied]
                        if show.isChecked()
                        else list(VIEWPORT_CORNERS)
                    )
                    choice = selected[key] if selected[key] in allowed else allowed[0]
                    with QSignalBlocker(combo):
                        combo.clear()
                        for corner in allowed:
                            combo.addItem(corner, corner)
                        combo.setCurrentIndex(max(0, combo.findData(choice)))
                    combo.setEnabled(show.isChecked())
            finally:
                overlay_refreshing["active"] = False

        for key, (label_text, default_corner) in VIEWPORT_OVERLAYS.items():
            control = QWidget(dialog)
            row = QHBoxLayout(control)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(7)
            show = QCheckBox("Show", control)
            show.setChecked(self.viewport_overlay_visible.get(key, True))
            combo = ArrowComboBox(control)
            combo.addItem(self.viewport_overlay_corners.get(key, default_corner))
            tooltip = (
                f"Show or hide the {label_text.lower()} overlay. Visible overlays use distinct "
                "viewport corners."
            )
            show.setToolTip(tooltip)
            combo.setToolTip("Choose a free viewport corner for this overlay.")
            row.addWidget(show)
            row.addWidget(combo, 1)
            overlays_form.addRow(label_text, control)
            overlay_controls[key] = (show, combo)
            show.toggled.connect(refresh_overlay_corners)
            combo.currentIndexChanged.connect(refresh_overlay_corners)
        refresh_overlay_corners()
        overlays_layout.addLayout(overlays_form)

        metrics_layout = section("Metrics history")
        metrics_form = QFormLayout()
        metrics_refresh = ArrowComboBox(dialog)
        for milliseconds, label in (
            (1000, "Once per second"),
            (500, "2 times per second"),
            (250, "4 times per second"),
            (2000, "Every 2 seconds"),
            (5000, "Every 5 seconds"),
        ):
            metrics_refresh.addItem(label, milliseconds)
        metrics_refresh.setCurrentIndex(
            max(0, metrics_refresh.findData(self.metrics_refresh_ms))
        )
        metrics_refresh.setToolTip(
            "How often MeshMill samples CPU, GPU, RAM, VRAM, and I/O. Faster updates make the "
            "graph smoother but retain more samples."
        )
        metrics_form.addRow("Refresh", metrics_refresh)

        metrics_history = ArrowComboBox(dialog)
        for minutes in (5, 15, 30, 60, 120):
            metrics_history.addItem(f"{minutes} minutes", minutes)
        metrics_history.setCurrentIndex(
            max(0, metrics_history.findData(self.metrics_history_minutes))
        )
        metrics_history.setToolTip(
            "How long CPU and GPU readings stay available. Recent readings are shown in detail; "
            "older readings are grouped into averages with their minimum and peak usage preserved."
        )
        metrics_form.addRow("Keep history", metrics_history)
        metrics_timeline = ArrowComboBox(dialog)
        for seconds in (10, 30, 60, 120, 300):
            metrics_timeline.addItem(f"{seconds} seconds", seconds)
        metrics_timeline.setCurrentIndex(
            max(0, metrics_timeline.findData(self.metrics_visible_seconds))
        )
        metrics_timeline.setToolTip(
            "The recent time range shown in full detail on the right. Older history is "
            "progressively compressed toward the left. A shorter range zooms in and enables scrolling."
        )
        metrics_form.addRow("Recent detail", metrics_timeline)
        metrics_memory = QLabel(dialog)
        metrics_memory.setProperty("muted", True)

        def update_metrics_memory_estimate() -> None:
            refresh = int(metrics_refresh.currentData() or 500)
            minutes = int(metrics_history.currentData() or 15)
            samples = math.ceil(minutes * 60_000 / refresh)
            estimated_bytes = samples * 160 + min(samples, 1600) * 320
            metrics_memory.setText(
                f"{samples:,} samples, about {fmt_bytes(estimated_bytes)} RAM"
            )

        metrics_refresh.currentIndexChanged.connect(update_metrics_memory_estimate)
        metrics_history.currentIndexChanged.connect(update_metrics_memory_estimate)
        metrics_timeline.currentIndexChanged.connect(update_metrics_memory_estimate)
        update_metrics_memory_estimate()
        metrics_form.addRow("Memory estimate", metrics_memory)
        metrics_layout.addLayout(metrics_form)

        optimization_defaults_layout = section("Optimization defaults")
        optimization_defaults_form = QFormLayout()

        startup_algorithm = ArrowComboBox(dialog)
        startup_algorithm.addItem("Last used", "last")
        startup_algorithm.addItem("Automatic (best available)", "auto")
        for algorithm_name in ALGORITHMS:
            label = algorithm_name
            if is_gpu_guided_algorithm(algorithm_name) and not gpu_backend_supported(
                self.gpu_preference
            ):
                label += "  (GPU unavailable)"
            startup_algorithm.addItem(label, algorithm_name)
        startup_algorithm_index = startup_algorithm.findData(self.startup_algorithm)
        startup_algorithm.setCurrentIndex(max(0, startup_algorithm_index))
        startup_algorithm.setToolTip(
            "Choose the algorithm selected when MeshMill opens. Last used remembers your most "
            "recent choice. Automatic selects Depth + Density Adaptive QEM when its GPU backend "
            "is available and Density balanced otherwise."
        )
        optimization_defaults_form.addRow("Startup algorithm", startup_algorithm)

        startup_quality = ArrowComboBox(dialog)
        startup_quality.addItem("Last used", "last")
        for quality_name in SMART_DIVISORS:
            startup_quality.addItem(quality_name, quality_name)
        startup_quality_index = startup_quality.findData(self.startup_quality)
        startup_quality.setCurrentIndex(max(0, startup_quality_index))
        startup_quality.setToolTip(
            "Choose the Smart quality setting selected when MeshMill opens. Draft removes more "
            "geometry, Balanced is the middle setting, and Fine preserves more detail."
        )
        optimization_defaults_form.addRow("Startup quality", startup_quality)
        optimization_defaults_layout.addLayout(optimization_defaults_form)

        guide_title = QLabel("Algorithm tradeoffs", dialog)
        guide_title.setStyleSheet("font-weight: 600; margin-top: 5px;")
        optimization_defaults_layout.addWidget(guide_title)
        guide = QGridLayout()
        guide.setHorizontalSpacing(6)
        guide.setVerticalSpacing(4)
        guide_headers = (
            "Method",
            "Surface detail",
            "Damaged areas",
            "Size accuracy",
            "Surface continuity",
            "Speed",
        )
        for column, heading_text in enumerate(guide_headers):
            heading = QLabel(heading_text, dialog)
            heading.setWordWrap(True)
            heading.setStyleSheet("font-weight: 600;")
            heading.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            guide.addWidget(heading, 0, column)
        tradeoffs = (
            ("Depth + Density", "Best", "Best", "Best", "Best", "Moderate"),
            ("Depth Adaptive", "Very good", "Very good", "Very good", "Very good", "Moderate"),
            ("Density balanced", "Good", "Good", "Very good", "Good", "Fast"),
            ("Fast QEM", "Good", "Fair", "Fair", "Good", "Fastest"),
        )
        tradeoff_colors = {
            "Best": ("#DCFCE7", "#166534"),
            "Very good": ("#ECFDF5", "#047857"),
            "Good": ("#FEF3C7", "#92400E"),
            "Fair": ("#FEE2E2", "#991B1B"),
            "Moderate": ("#FEF3C7", "#92400E"),
            "Fast": ("#ECFDF5", "#047857"),
            "Fastest": ("#DCFCE7", "#166534"),
        }
        for row, values in enumerate(tradeoffs, start=1):
            method = QLabel(values[0], dialog)
            method.setWordWrap(True)
            method.setStyleSheet("font-weight: 600;")
            guide.addWidget(method, row, 0)
            for column, value in enumerate(values[1:], start=1):
                background, foreground = tradeoff_colors[value]
                cell = QLabel(value, dialog)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setStyleSheet(
                    f"color: {foreground}; background: {background}; "
                    "border-radius: 4px; padding: 3px;"
                )
                guide.addWidget(cell, row, column)
        guide_note = QLabel(
            "These labels summarize the retained comparison mesh. Results vary with geometry. "
            "GPU-guided methods spend more time measuring the surface before reduction.",
            dialog,
        )
        guide_note.setWordWrap(True)
        guide_note.setProperty("muted", True)
        optimization_defaults_layout.addLayout(guide)
        optimization_defaults_layout.addWidget(guide_note)

        performance_layout = section("Performance")
        performance_form = QFormLayout()

        def performance_row(label_text: str, control: QWidget, help_text: str) -> None:
            label = QLabel(label_text, dialog)
            label.setToolTip(help_text)
            label.setMouseTracking(True)
            control.setToolTip(help_text)
            control.setMouseTracking(True)
            for child in control.findChildren(QWidget):
                child.setToolTip(help_text)
                child.setMouseTracking(True)
            performance_form.addRow(label, control)

        gpu_preference = ArrowComboBox(dialog)
        gpu_preference.addItem("Automatic (most available VRAM)", "system")
        for device in self.cuda_device_list:
            memory = fmt_bytes(int(device["total_memory"]))
            gpu_preference.addItem(
                f"{device['name']}  ({device['pci_bus_id']}, {memory})",
                device["pci_bus_id"],
            )
        selected_gpu = gpu_preference.findData(self.gpu_preference)
        gpu_preference.setCurrentIndex(max(0, selected_gpu))
        performance_row(
            "GPU",
            gpu_preference,
            "Selects the CUDA GPU used by GPU-guided optimization. Automatic checks available "
            "VRAM when work starts and chooses the least occupied compatible GPU. A specific "
            "selection is remembered and used until you change it. The viewport GPU is managed "
            "separately by the operating system.",
        )

        geometry_processes = ArrowComboBox(dialog)
        recommended_processes = min(8, max(1, self.logical_cpu_count // 2))
        geometry_processes.addItem(
            f"Automatic ({recommended_processes} detected)", 0
        )
        for process_count in range(1, self.logical_cpu_count + 1):
            geometry_processes.addItem(str(process_count), process_count)
        selected_processes = geometry_processes.findData(self.geometry_processes)
        geometry_processes.setCurrentIndex(max(0, selected_processes))
        performance_row(
            "Geometry processes",
            geometry_processes,
            f"Controls how many pieces of geometry MeshMill can analyze at the same time. "
            f"Automatic detected {recommended_processes} workers from this computer's "
            f"{self.logical_cpu_count} logical processors. Higher values can finish sooner but "
            "use more processor time and memory.",
        )

        large_mesh_mode = ArrowComboBox(dialog)
        large_mesh_mode.addItem("Auto (resource-aware)", "auto")
        large_mesh_mode.addItem("Full mesh (override)", "full")
        large_mesh_mode.addItem("Tiled processing", "tiled")
        large_mesh_mode.addItem("Overview only", "overview")
        large_mesh_mode.setCurrentIndex(max(0, large_mesh_mode.findData(self.large_mesh_mode)))
        performance_row(
            "Loading",
            large_mesh_mode,
            "Automatic checks the current memory budget at load time. Full mesh overrides that "
            "check and can exhaust memory. Tiled processing keeps full-resolution geometry in "
            "bounded disk-backed tiles and can use multiple CPU workers. Overview only opens a "
            "lighter navigation copy and leaves editing disabled.",
        )
        auto_tiling = QCheckBox("Use tiled processing automatically", dialog)
        auto_tiling.setChecked(self.auto_tiling_enabled)
        performance_row(
            "Automatic tiling",
            auto_tiling,
            "When enabled, Automatic can select tiled processing after comparing index setup and "
            "reduction estimates with full-mesh processing for the selected algorithm. Disable it "
            "to keep Automatic on full mesh when memory permits. Oversized meshes still open as an overview.",
        )

        tiled_recommendation = QLabel(dialog)
        tiled_recommendation.setWordWrap(True)
        tiled_recommendation.setToolTip(
            "Automatic uses tiled processing only when index setup plus tiled reduction is estimated "
            "to be faster than full-mesh reduction for the selected algorithm."
        )

        def update_tiled_recommendation(memory_percent: int) -> None:
            triangle_count = (
                self.overview_total_triangles
                if self.overview_total_triangles
                else len(self.source_faces)
                if self.source_faces is not None
                else 0
            )
            if not triangle_count:
                tiled_recommendation.setText(
                    "Tiled recommendation: evaluated when a mesh is opened. "
                    "Choose Tiled processing to require bounded full-resolution processing."
                )
                return
            budget = available_physical_memory() * memory_percent / 100.0
            estimated = triangle_count * ESTIMATED_BYTES_PER_TRIANGLE
            index_ready = False
            if self.source_path is not None:
                try:
                    index_ready = index_matches(
                        self.source_path,
                        load_manifest(default_index_directory(self.source_path)),
                    )
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    pass
            break_even = tiled_runtime_break_even(
                triangle_count,
                self.algorithm.currentText(),
                index_ready,
            )
            if self.tiled_mode:
                tiled_recommendation.setText(
                    "Tiled recommendation: active for this mesh. Full-resolution tiles are "
                    "bounded by the selected memory budget."
                )
            elif estimated > budget:
                tiled_recommendation.setText(
                    "Tiled recommendation: use Tiled processing. Estimated full working memory "
                    f"({fmt_bytes(estimated)}) exceeds the current budget ({fmt_bytes(budget)})."
                )
            elif is_gpu_guided_algorithm(self.algorithm.currentText()):
                tiled_recommendation.setText(
                    "Tiled GPU-guided reference: RMS 0.120% full / 0.133% tiled, "
                    "P95 0.245% / 0.264%. Full mesh selected."
                )
            elif break_even is not None and break_even[1] < break_even[0]:
                tiled_recommendation.setText(
                    "Tiled recommendation: use Tiled processing. Estimated reduction: "
                    f"{break_even[0]:.1f} s full, {break_even[1]:.1f} s tiled."
                )
            elif break_even is not None:
                tiled_recommendation.setText(
                    "Tiled recommendation: full mesh. Estimated reduction: "
                    f"{break_even[0]:.1f} s full, {break_even[1]:.1f} s tiled."
                )
            else:
                tiled_recommendation.setText(
                    "Tiled recommendation: optional. This algorithm does not yet have a matched "
                    "full-mesh and tiled runtime model."
                )

        memory_percent_control = QWidget(dialog)
        memory_percent_layout = QVBoxLayout(memory_percent_control)
        memory_percent_layout.setContentsMargins(0, 0, 0, 0)
        memory_percent_layout.setSpacing(3)
        memory_percent = QSlider(Qt.Orientation.Horizontal, memory_percent_control)
        memory_percent.setRange(20, 80)
        memory_percent.setSingleStep(5)
        memory_percent.setPageStep(10)
        memory_percent.setTickInterval(10)
        memory_percent.setTickPosition(QSlider.TickPosition.TicksBelow)
        memory_percent.setValue(self.large_mesh_memory_percent)
        memory_scale = QHBoxLayout()
        memory_minimum = QLabel("20%", memory_percent_control)
        memory_value = QLabel(memory_percent_control)
        memory_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        memory_value.setStyleSheet("font-weight: 600;")
        memory_maximum = QLabel("80%", memory_percent_control)
        memory_scale.addWidget(memory_minimum)
        memory_scale.addStretch(1)
        memory_scale.addWidget(memory_value)
        memory_scale.addStretch(1)
        memory_scale.addWidget(memory_maximum)

        def update_memory_budget(value: int) -> None:
            value = max(20, min(80, round(value / 5) * 5))
            if memory_percent.value() != value:
                with QSignalBlocker(memory_percent):
                    memory_percent.setValue(value)
            budget = round(available_physical_memory() * value / 100.0)
            memory_value.setText(f"{value}%  ({fmt_bytes(budget)})")
            update_tiled_recommendation(value)

        memory_percent.valueChanged.connect(update_memory_budget)
        self.algorithm.currentTextChanged.connect(
            lambda _text: update_tiled_recommendation(memory_percent.value())
        )
        update_memory_budget(memory_percent.value())
        memory_percent_layout.addWidget(memory_percent)
        memory_percent_layout.addLayout(memory_scale)
        performance_row(
            "Memory budget",
            memory_percent_control,
            "Sets how much currently available RAM MeshMill may use when deciding whether a mesh "
            "can be loaded in full. A higher value accepts larger meshes but leaves less memory "
            "for other applications.",
        )

        performance_layout.addLayout(performance_form)
        performance_layout.addWidget(tiled_recommendation)

        invert_layout = section("Invert")
        invert_grid = QGridLayout()
        invert_grid.setColumnStretch(0, 1)
        mouse_heading = QLabel("Mouse", dialog)
        mouse_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        invert_grid.addWidget(mouse_heading, 0, 1)
        keyboard_heading = QLabel("Keyboard", dialog)
        keyboard_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        invert_grid.addWidget(keyboard_heading, 0, 2)
        inversion_checks: dict[str, tuple[QCheckBox, QCheckBox]] = {}
        inversion_values = {
            "Horizontal": (self.mouse_horizontal, self.keyboard_horizontal),
            "Vertical": (self.mouse_vertical, self.keyboard_vertical),
            "Zoom": (self.mouse_zoom, self.keyboard_zoom),
        }

        def inversion_diagram(input_kind: str, axis: str, inverted: bool) -> str:
            if input_kind == "Mouse" and axis == "Horizontal":
                result = "→ / ←" if inverted else "← / →"
                return f"Drag ← / →   View {result}"
            if input_kind == "Mouse" and axis == "Vertical":
                result = "↓ / ↑" if inverted else "↑ / ↓"
                return f"Drag ↑ / ↓   View {result}"
            if input_kind == "Mouse":
                result = "− / +" if inverted else "+ / −"
                return f"Wheel ↑ / ↓   Zoom {result}"
            if axis == "Horizontal":
                result = "→ / ←" if inverted else "← / →"
                return f"Press ← / →   View {result}"
            if axis == "Vertical":
                result = "↓ / ↑" if inverted else "↑ / ↓"
                return f"Press ↑ / ↓   View {result}"
            result = "− / +" if inverted else "+ / −"
            return f"Press ↑ / ↓   Zoom {result}"

        from mesh_export_ui import readable_theme_color

        input_theme = appearance_theme(self.appearance_theme_name)
        input_accent = readable_theme_color(
            input_theme["accent"], [input_theme["window"], input_theme["panel"]], input_theme["text"]
        )

        def input_icon(input_kind: str, axis: str) -> QPixmap:
            pixmap = QPixmap(25, 23)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor(input_theme["text"]), 1.5))
            if input_kind == "Mouse":
                painter.drawRoundedRect(QRect(7, 2, 11, 18), 5, 5)
                painter.drawLine(12, 3, 12, 8)
                painter.drawLine(8, 9, 17, 9)
                if axis == "Zoom":
                    painter.drawLine(12, 4, 12, 7)
                    painter.drawLine(10, 5, 12, 3)
                    painter.drawLine(14, 5, 12, 3)
            else:
                first = QRect(1, 4, 11, 15)
                second = QRect(13, 4, 11, 15)
                painter.drawRoundedRect(first, 2, 2)
                painter.drawRoundedRect(second, 2, 2)
                arrows = ("←", "→") if axis == "Horizontal" else ("↑", "↓")
                painter.drawText(first, Qt.AlignmentFlag.AlignCenter, arrows[0])
                painter.drawText(second, Qt.AlignmentFlag.AlignCenter, arrows[1])
            painter.end()
            return pixmap

        def inversion_control(
            input_kind: str, axis: str, checked: bool, tooltip: str
        ) -> tuple[QWidget, QCheckBox]:
            cell = QWidget(dialog)
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(4, 1, 4, 1)
            cell_layout.setSpacing(7)
            check = QCheckBox(cell)
            check.setChecked(checked)
            check.setAccessibleName(f"{input_kind} {axis.lower()}")
            check.setToolTip(tooltip)
            icon = QLabel(cell)
            icon.setFixedSize(25, 23)
            icon.setPixmap(input_icon(input_kind, axis))
            icon.setToolTip(tooltip)
            diagram = QLabel(cell)
            diagram.setMinimumWidth(145)
            diagram.setStyleSheet(f"color: {input_theme['text']};")
            diagram.setToolTip(tooltip)

            def refresh(value: bool) -> None:
                diagram.setText(inversion_diagram(input_kind, axis, value))
                diagram.setStyleSheet(
                    f"color: {input_accent}; font-weight: 600;"
                    if value
                    else f"color: {input_theme['text']};"
                )

            refresh(checked)
            check.toggled.connect(refresh)
            cell_layout.addWidget(check)
            cell_layout.addWidget(icon)
            cell_layout.addWidget(diagram, 1)
            return cell, check

        for row, (axis, values) in enumerate(inversion_values.items(), start=1):
            invert_grid.addWidget(QLabel(axis, dialog), row, 0)
            mouse_tooltip = {
                "Horizontal": "Reverse horizontal mouse orbit, pan, and Ctrl+wheel roll.",
                "Vertical": "Reverse vertical mouse orbit and pan.",
                "Zoom": "Reverse scroll-wheel zoom direction.",
            }[axis]
            mouse_cell, mouse_check = inversion_control(
                "Mouse", axis, values[0], mouse_tooltip
            )
            invert_grid.addWidget(mouse_cell, row, 1)
            keyboard_tooltip = {
                "Horizontal": "Reverse horizontal orbit, pan, and roll shortcut directions.",
                "Vertical": "Reverse vertical orbit and pan shortcut directions.",
                "Zoom": "Reverse keyboard zoom-in and zoom-out directions.",
            }[axis]
            keyboard_cell, keyboard_check = inversion_control(
                "Keyboard", axis, values[1], keyboard_tooltip
            )
            invert_grid.addWidget(keyboard_cell, row, 2)
            inversion_checks[axis] = (mouse_check, keyboard_check)
        invert_layout.addLayout(invert_grid)

        shortcuts_layout = section("Keyboard shortcuts")
        shortcut_grid = QGridLayout()
        shortcut_grid.setColumnStretch(1, 1)
        shortcut_grid.addWidget(QLabel("Action", dialog), 0, 0)
        shortcut_grid.addWidget(QLabel("Shortcut", dialog), 0, 1)
        shortcut_edits: dict[str, QKeySequenceEdit] = {}
        for row, (shortcut_id, (label, default_sequence)) in enumerate(
            SHORTCUT_DEFINITIONS.items(), start=1
        ):
            shortcut_grid.addWidget(QLabel(label, dialog), row, 0)
            editor = QKeySequenceEdit(QKeySequence(self.shortcuts[shortcut_id]), dialog)
            editor.setMaximumSequenceLength(1)
            editor.setClearButtonEnabled(True)
            shortcut_grid.addWidget(editor, row, 1)
            reset_button = QPushButton("Reset", dialog)
            reset_button.setProperty("quiet", True)
            reset_button.setFixedWidth(64)
            reset_button.clicked.connect(
                lambda _=False, e=editor, default=default_sequence: e.setKeySequence(
                    QKeySequence(default)
                )
            )
            shortcut_grid.addWidget(reset_button, row, 2)
            shortcut_edits[shortcut_id] = editor
        shortcuts_layout.addLayout(shortcut_grid)

        reset_all_button = QPushButton("Reset", dialog)
        reset_all_button.setObjectName("preferencesReset")
        reset_all_button.setToolTip("Reset all settings to defaults after confirmation. Save to apply.")
        reset_all_button.setProperty("quiet", True)
        reset_all_requested = {"value": False}

        def reset_all() -> None:
            answer = QMessageBox.question(
                dialog,
                "Reset all settings",
                "Reset interface, navigation, display, and keyboard shortcut settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            reset_all_requested["value"] = True
            open_mode.setCurrentIndex(open_mode.findData("last"))
            open_folder.clear()
            recent_files_enabled.setChecked(False)
            automatic_updates.setChecked(False)
            auto_install_scope.setCurrentIndex(auto_install_scope.findData("maintenance"))
            language.setCurrentIndex(language.findData("system"))
            appearance.setCurrentIndex(appearance.findData(DEFAULT_APPEARANCE))
            section_order.clear()
            section_order.addItems(TOOLBOX_ORDER)
            section_order.setCurrentRow(0)
            viewport_appearance.setCurrentIndex(
                viewport_appearance.findData(DEFAULT_APPEARANCE)
            )
            custom_gradient["left"] = "#174A82"
            custom_gradient["right"] = "#E23730"
            refresh_gradient_buttons()
            for key, (_label, default_corner) in VIEWPORT_OVERLAYS.items():
                show, combo = overlay_controls[key]
                show.setChecked(True)
                index = combo.findData(default_corner)
                if index >= 0:
                    combo.setCurrentIndex(index)
            refresh_overlay_corners()
            font_size.setValue(1)
            metrics_refresh.setCurrentIndex(metrics_refresh.findData(1000))
            metrics_history.setCurrentIndex(metrics_history.findData(15))
            metrics_timeline.setCurrentIndex(metrics_timeline.findData(300))
            tools_side.setCurrentText("Right")
            default_units.setCurrentText("cm")
            startup_algorithm.setCurrentIndex(startup_algorithm.findData("auto"))
            startup_quality.setCurrentIndex(startup_quality.findData("last"))
            gpu_preference.setCurrentIndex(gpu_preference.findData("system"))
            geometry_processes.setCurrentIndex(geometry_processes.findData(0))
            large_mesh_mode.setCurrentIndex(large_mesh_mode.findData("auto"))
            auto_tiling.setChecked(True)
            memory_percent.setValue(50)
            for mouse_check, keyboard_check in inversion_checks.values():
                mouse_check.setChecked(False)
                keyboard_check.setChecked(False)
            for shortcut_id, (_label, default_sequence) in SHORTCUT_DEFINITIONS.items():
                shortcut_edits[shortcut_id].setKeySequence(QKeySequence(default_sequence))

        reset_all_button.clicked.connect(reset_all)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(search)
        save_settings_button = QPushButton("Save", dialog)
        save_settings_button.setDefault(True)
        cancel_settings_button = QPushButton("Cancel", dialog)
        footer.addWidget(save_settings_button)
        footer.addWidget(reset_all_button)
        footer.addWidget(cancel_settings_button)
        def accept_settings() -> None:
            assignments: dict[str, set[str]] = {}
            for shortcut_id, editor in shortcut_edits.items():
                sequence = editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
                if sequence:
                    assignments.setdefault(sequence, set()).add(shortcut_id)
            conflicts = [
                ids
                for ids in assignments.values()
                if len(ids) > 1 and ids != {"selection_delete", "view_top"}
            ]
            if conflicts:
                QMessageBox.warning(dialog, APP_NAME, "Each shortcut must be unique.")
                return
            dialog.accept()

        save_settings_button.clicked.connect(accept_settings)
        cancel_settings_button.clicked.connect(dialog.reject)
        outer.addLayout(footer)
        localize_widget_tree(dialog)
        def filter_settings(query: str) -> None:
            no_matches.setVisible(filter_preferences_sections(sections, query) == 0)
            scroll.verticalScrollBar().setValue(0)
        search.textChanged.connect(filter_settings)
        viewport_appearance.currentIndexChanged.connect(lambda _: filter_settings(search.text()))
        search.setFocus()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if reset_all_requested["value"]:
            self.settings.clear()
        self.locale_preference = str(language.currentData() or "system")
        self.settings.setValue("files/open_mode", open_mode.currentData())
        self.settings.setValue("files/open_folder", open_folder.text().strip())
        self._set_recent_files_enabled(recent_files_enabled.isChecked())
        self.appearance_theme_name = str(
            appearance.currentData() or DEFAULT_APPEARANCE
        )
        self.viewport_appearance_name = str(
            viewport_appearance.currentData() or "match"
        )
        self.toolbox_order = [section_order.item(i).text() for i in range(section_order.count())]
        self.settings.setValue("interface/toolbox_order", self.toolbox_order)
        self._reorder_tool_sections()
        self.custom_density_gradient_left = custom_gradient["left"]
        self.custom_density_gradient_right = custom_gradient["right"]
        self.viewport_overlay_visible = {
            key: show.isChecked() for key, (show, _combo) in overlay_controls.items()
        }
        self.viewport_overlay_corners = {
            key: str(combo.currentData() or VIEWPORT_OVERLAYS[key][1])
            for key, (_show, combo) in overlay_controls.items()
        }
        self._normalize_viewport_overlay_corners()
        self.font_size_position = font_size.value()
        self.metrics_refresh_ms = int(metrics_refresh.currentData() or 1000)
        self.metrics_history_minutes = int(metrics_history.currentData() or 15)
        self.metrics_visible_seconds = int(metrics_timeline.currentData() or 30)
        previous_update_setting = self.update_checks_enabled
        self.update_checks_enabled = automatic_updates.isChecked()
        self.auto_install_updates = self.update_checks_enabled and auto_install.isChecked()
        self.auto_install_scope = str(auto_install_scope.currentData() or "maintenance")
        self.locale_code = locale_manager.select(self._resolved_locale(self.locale_preference))
        self.default_units = default_units.currentText()
        self.startup_algorithm = str(startup_algorithm.currentData() or "auto")
        self.startup_quality = str(startup_quality.currentData() or "last")
        self.gpu_preference = str(gpu_preference.currentData() or "system")
        selected_cuda = next(
            (
                device
                for device in self.cuda_device_list
                if device["pci_bus_id"] == self.gpu_preference
            ),
            self.cuda_device_list[0] if self.cuda_device_list else None,
        )
        if selected_cuda is not None:
            self.vram_capacity_bytes = int(selected_cuda["total_memory"])
        self.geometry_processes = int(geometry_processes.currentData() or 0)
        self.large_mesh_mode = str(large_mesh_mode.currentData())
        self.auto_tiling_enabled = auto_tiling.isChecked()
        self.large_mesh_memory_percent = memory_percent.value()
        self._capacity_available_memory = 0
        self.mouse_horizontal = inversion_checks["Horizontal"][0].isChecked()
        self.keyboard_horizontal = inversion_checks["Horizontal"][1].isChecked()
        self.mouse_vertical = inversion_checks["Vertical"][0].isChecked()
        self.keyboard_vertical = inversion_checks["Vertical"][1].isChecked()
        self.mouse_zoom = inversion_checks["Zoom"][0].isChecked()
        self.keyboard_zoom = inversion_checks["Zoom"][1].isChecked()
        self.navigation_style.invert_horizontal = self.mouse_horizontal
        self.navigation_style.invert_vertical = self.mouse_vertical
        self.navigation_style.invert_zoom = self.mouse_zoom
        self.units.setCurrentText(self.default_units)
        self._apply_appearance()
        self._position_viewport_overlays()
        if hasattr(self, "density_legend"):
            self.density_legend.setVisible(
                self._display_mode() == "Density" and self._overlay_is_visible("density")
            )
        self.metrics_timer.setInterval(self.metrics_refresh_ms)
        self.geometry_activity_graph.configure_history(
            self.metrics_refresh_ms,
            self.metrics_history_minutes,
            self.metrics_visible_seconds,
        )
        self._configure_metrics_history()
        self._apply_tools_side(tools_side.currentText())
        with QSignalBlocker(self.processing_route):
            self.processing_route.setCurrentIndex(
                max(0, self.processing_route.findData(self.large_mesh_mode))
            )
        self._update_processing_plan()
        self.shortcuts = {
            shortcut_id: editor.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
            for shortcut_id, editor in shortcut_edits.items()
        }
        self._sync_full_screen_button()
        self._refresh_view_button_tooltips()
        self._refresh_display_mode_labels()

        self.settings.setValue("interface/tools_side", self.tools_side)
        self.settings.setValue("updates/automatic", self.update_checks_enabled)
        self.settings.setValue("updates/auto_install", self.auto_install_updates)
        self.settings.setValue("updates/auto_install_scope", self.auto_install_scope)
        self.settings.setValue("interface/locale", self.locale_preference)
        self.settings.setValue("interface/appearance_theme", self.appearance_theme_name)
        self.settings.setValue("interface/viewport_appearance", self.viewport_appearance_name)
        self.settings.setValue("interface/viewport_appearance_version", 1)
        self.settings.setValue("interface/density_gradient_left", self.custom_density_gradient_left)
        self.settings.setValue("interface/density_gradient_right", self.custom_density_gradient_right)
        for key in VIEWPORT_OVERLAYS:
            self.settings.setValue(
                f"interface/overlay_{key}_visible", self.viewport_overlay_visible[key]
            )
            self.settings.setValue(
                f"interface/overlay_{key}_corner", self.viewport_overlay_corners[key]
            )
        self.settings.setValue("interface/font_size_position", self.font_size_position)
        self.settings.setValue("metrics/refresh_ms", self.metrics_refresh_ms)
        self.settings.setValue("metrics/history_minutes", self.metrics_history_minutes)
        self.settings.setValue("metrics/detail_seconds", self.metrics_visible_seconds)
        self.settings.setValue("metrics/visible_seconds", self.metrics_visible_seconds)
        self.settings.setValue("metrics/default_version", 2)
        self.settings.setValue("performance/gpu_preference", self.gpu_preference)
        self.settings.remove("rendering/gpu_preference")
        self.settings.setValue("interface/default_units", self.default_units)
        self.settings.setValue("interface/units_default_version", 2)
        self.settings.setValue("optimization/startup_algorithm", self.startup_algorithm)
        self.settings.setValue("optimization/startup_quality", self.startup_quality)
        self.settings.setValue("performance/large_mesh_mode", self.large_mesh_mode)
        self.settings.setValue("performance/auto_tiling_enabled", self.auto_tiling_enabled)
        self.settings.setValue("performance/memory_percent", self.large_mesh_memory_percent)
        self.settings.remove("performance/geometry_compute")
        self.settings.setValue("performance/geometry_processes", self.geometry_processes)
        self.settings.remove("performance/overview_triangles")
        self.settings.setValue("navigation/mouse_horizontal", self.mouse_horizontal)
        self.settings.setValue("navigation/mouse_vertical", self.mouse_vertical)
        self.settings.setValue("navigation/mouse_zoom", self.mouse_zoom)
        self.settings.setValue("navigation/keyboard_horizontal", self.keyboard_horizontal)
        self.settings.setValue("navigation/keyboard_vertical", self.keyboard_vertical)
        self.settings.setValue("navigation/keyboard_zoom", self.keyboard_zoom)
        self.settings.remove("navigation/invert_mouse")
        self.settings.remove("navigation/invert_keyboard")
        self.settings.remove("navigation/invert_horizontal")
        self.settings.remove("navigation/invert_vertical")
        self.settings.remove("navigation/invert_zoom")
        self.settings.remove("optimization/automatic")
        self.settings.remove("display/show_edges")
        self.settings.setValue("display/type", self._display_mode())
        for shortcut_id, sequence in self.shortcuts.items():
            self.settings.setValue(f"shortcuts/{shortcut_id}", sequence)
        self.settings.sync()
        if self.update_checks_enabled and not previous_update_setting:
            self._check_for_updates()
        else:
            self._update_update_action()
        localize_widget_tree(self)

    def _install_help_tooltips(self) -> None:
        help_text = {
            self.open_button: "Open an STL mesh.",
            self.reload_button: "Reload source STL and discard edits.",
            self.undo_button: "Undo one mesh action.",
            self.redo_button: "Redo one mesh action.",
            self.units: "Dimension display units.",
            self.target_display: "Target triangle count derived from the slider and Smart quality.",
            self.preview_button: "Create an optimization preview using the current settings.",
            self.smart_quality: "Automatic target density: Draft 15%, Balanced 50%, Fine 75% maximum.",
            self.algorithm: "Mesh optimization algorithm.",
            self.slider: "Triangle target density.",
            self.display_type: "Choose shaded, wireframe, point, or color-mapped point-density display.",
            self.toggle_mesh_button: "Switch between the original and optimized meshes.",
            self.previous_mesh_button: "Switch between the current mesh and its previous undo state.",
            self.save_button: "Save the current mesh as STL, COLLADA, GLB, OBJ or PLY.",
            self.settings_button: "Open app settings.",
            self.version_label: (
                "Click to check GitHub for a newer MeshMill release. "
                "Right-click to copy the installed version. This does not enable automatic checks."
            ),
            self.metrics_toggle: "Show or hide CPU, memory, GPU, and mesh metrics.",
            self.about_button: "Version and license information.",
            self.repository_button: (
                "Open the project repository."
                if REPOSITORY_URL
                else "Project repository unavailable."
            ),
        }
        for widget, description in help_text.items():
            if not widget.toolTip():
                widget.setToolTip(description)
            widget.setMouseTracking(True)
        for widget in self.findChildren(QWidget):
            if widget.toolTip():
                widget.setMouseTracking(True)
        self.tooltip_filter = DeferredTooltipFilter(self)
        QApplication.instance().installEventFilter(self.tooltip_filter)

    def _create_metrics_panel(self, parent_layout: QVBoxLayout) -> None:
        self.metrics_toggle = QPushButton("Metrics  ▲")
        self.metrics_toggle.setCheckable(True)
        self.metrics_toggle.setChecked(False)
        self.metrics_toggle.setProperty("quiet", True)
        self.metrics_toggle.toggled.connect(self._toggle_metrics)
        self.metrics_toggle_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.metrics_toggle_row.setContentsMargins(0, 0, 0, 0)
        self.metrics_toggle_row.addStretch(1)
        self.metrics_toggle_row.addWidget(self.metrics_toggle)
        parent_layout.addLayout(self.metrics_toggle_row)

        frame = QFrame()
        self.metrics_frame = frame
        frame.setObjectName("metricsPanel")
        grid = QGridLayout(frame)
        grid.setContentsMargins(10, 7, 10, 7)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        self.metric_labels: dict[str, QLabel] = {}
        self.metric_bars: dict[str, QProgressBar] = {}
        for column, (key, title) in enumerate(
            (("cpu", "CPU"), ("ram", "RAM"), ("gpu", "GPU"), ("vram", "VRAM"))
        ):
            heading = QLabel(title)
            heading.setStyleSheet("font-weight: 600;")
            value = QLabel("Reading…")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            bar = QProgressBar()
            bar.setStyleSheet("QProgressBar { text-align: right; }")
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p% used")
            grid.addWidget(heading, 0, column)
            grid.addWidget(bar, 1, column)
            grid.addWidget(value, 2, column)
            self.metric_labels[key] = value
            self.metric_bars[key] = bar
        self.live_mesh_metric = QLabel("Mesh: no file loaded")
        self.live_mesh_metric.setProperty("muted", True)
        self.live_mesh_metric.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.io_metric = QLabel("I/O: reading…")
        self.io_metric.setProperty("muted", True)
        self.io_metric.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.io_metric.setToolTip(
            "Total data read and written by MeshMill since it opened. Active time counts sampled "
            "intervals in which I/O changed. Peak samples are the largest totals observed during "
            "one metrics interval."
        )
        grid.addWidget(self.io_metric, 3, 0, 1, 4)
        grid.addWidget(self.live_mesh_metric, 4, 0, 1, 4)
        self.capacity_metric = QLabel("Capacity: calculating…")
        self.capacity_metric.setProperty("muted", True)
        self.capacity_metric.setWordWrap(True)
        self.capacity_metric.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.capacity_metric.setToolTip(
            "Estimated from currently available memory, the configured memory budget, CPU "
            "count, and overview chunk size. Geometry operations currently run one at a time."
        )
        grid.addWidget(self.capacity_metric, 5, 0, 1, 4)
        self.runtime_metric = QLabel("Runtime: no completed operation")
        self.runtime_metric.setProperty("muted", True)
        self.runtime_metric.setWordWrap(False)
        self.runtime_metric.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.runtime_metric.setToolTip(
            "Left: completed jobs and total processing time for this session. Right: the most "
            "recent operation and time spent in its main stages. CPU/GPU totals count geometry inputs "
            "processed by each analysis or reduction pass, including repeated previews. Bytes are "
            "point and triangle array sizes, not file sizes, memory usage, or transfer traffic. "
            "A mesh processed on both devices contributes to both totals. Cached results do not "
            "add another pass. Totals reset when the app closes."
        )
        grid.addWidget(self.runtime_metric, 6, 0, 1, 4)
        self.geometry_activity_graph = GeometryActivityGraph(frame)
        self.geometry_activity_graph.configure_history(
            self.metrics_refresh_ms, self.metrics_history_minutes, self.metrics_visible_seconds
        )
        self.geometry_activity_graph.setToolTip(
            "Recent readings are detailed on the right. Older readings are progressively "
            "grouped toward the left. Lines show sampled averages; vertical ranges preserve "
            "minimum and peak usage. Shading marks geometry processing."
        )
        history_controls = QHBoxLayout()
        self.metrics_timeline_label = QLabel()
        self.metrics_timeline_label.setProperty("muted", True)
        history_controls.addStretch(1)
        history_controls.addWidget(self.metrics_timeline_label)
        self.metrics_zoom_out = QPushButton("−")
        self.metrics_zoom_out.setProperty("quiet", True)
        self.metrics_zoom_out.setFixedWidth(30)
        self.metrics_zoom_out.setToolTip("Show more time in the metrics graph.")
        self.metrics_zoom_in = QPushButton("+")
        self.metrics_zoom_in.setProperty("quiet", True)
        self.metrics_zoom_in.setFixedWidth(30)
        self.metrics_zoom_in.setToolTip("Zoom in to a shorter time range in the metrics graph.")
        self.metrics_zoom_out.clicked.connect(lambda: self._adjust_metrics_timeline(1))
        self.metrics_zoom_in.clicked.connect(lambda: self._adjust_metrics_timeline(-1))
        history_controls.addWidget(self.metrics_zoom_out)
        history_controls.addWidget(self.metrics_zoom_in)
        grid.addLayout(history_controls, 7, 0, 1, 4)
        self.metrics_history_scroll = QScrollArea(frame)
        self.metrics_history_scroll.setWidgetResizable(False)
        self.metrics_history_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.metrics_history_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.metrics_history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.metrics_history_scroll.setWidget(self.geometry_activity_graph)
        self.metrics_follow_latest = True
        self.metrics_timeline_dragging = False
        metrics_scrollbar = self.metrics_history_scroll.horizontalScrollBar()
        metrics_scrollbar.sliderPressed.connect(self._pause_metrics_follow)
        metrics_scrollbar.sliderMoved.connect(self._metrics_timeline_moved)
        metrics_scrollbar.sliderReleased.connect(self._resume_metrics_follow_if_latest)
        grid.addWidget(self.metrics_history_scroll, 8, 0, 1, 4)
        self._configure_metrics_history()
        parent_layout.addWidget(frame)
        frame.hide()

    def _configure_metrics_history(self) -> None:
        graph = getattr(self, "geometry_activity_graph", None)
        scroll = getattr(self, "metrics_history_scroll", None)
        if graph is None or scroll is None:
            return
        graph.configure_history(
            self.metrics_refresh_ms,
            self.metrics_history_minutes,
            self.metrics_visible_seconds,
            scroll.viewport().width(),
        )
        label = getattr(self, "metrics_timeline_label", None)
        if label is not None:
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label.setText(f"Detail: {self.metrics_visible_seconds} s")
        self._follow_latest_metrics_history()

    def _pause_metrics_follow(self) -> None:
        self.metrics_timeline_dragging = True
        self.metrics_follow_latest = False
        self.geometry_activity_graph.set_paused(True)

    def _metrics_timeline_moved(self, _position: int) -> None:
        self.metrics_follow_latest = False
        self.geometry_activity_graph.set_paused(True)

    def _resume_metrics_follow_if_latest(self) -> None:
        self.metrics_timeline_dragging = False
        scroll = getattr(self, "metrics_history_scroll", None)
        if scroll is None:
            return
        bar = scroll.horizontalScrollBar()
        self.metrics_follow_latest = bar.value() >= bar.maximum() - 1
        self.geometry_activity_graph.set_paused(not self.metrics_follow_latest)
        self._follow_latest_metrics_history()

    def _follow_latest_metrics_history(self) -> None:
        scroll = getattr(self, "metrics_history_scroll", None)
        if (
            scroll is None
            or not getattr(self, "metrics_follow_latest", True)
            or getattr(self, "metrics_timeline_dragging", False)
        ):
            return
        bar = scroll.horizontalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def _adjust_metrics_timeline(self, direction: int) -> None:
        levels = (10, 30, 60, 120, 300)
        index = min(range(len(levels)), key=lambda item: abs(levels[item] - self.metrics_visible_seconds))
        index = max(0, min(len(levels) - 1, index + direction))
        self.metrics_visible_seconds = levels[index]
        self._configure_metrics_history()
        self.settings.setValue("metrics/detail_seconds", self.metrics_visible_seconds)

    def _toggle_metrics(self, visible: bool) -> None:
        self.metrics_frame.setVisible(visible)
        if visible:
            QTimer.singleShot(0, self._configure_metrics_history)
        self.metrics_toggle.setText("Hide metrics  ▼" if visible else "Metrics  ▲")
        self._refresh_viewport_layout()
        # Windows can defer resizing the native VTK child in a maximized window. Repeat
        # after Qt has processed the visibility and layout requests.
        QTimer.singleShot(0, self._refresh_viewport_layout)
        QTimer.singleShot(75, self._refresh_viewport_layout)

    def _refresh_viewport_layout(self) -> None:
        if getattr(self, "_closing", False):
            return
        root = self.centralWidget()
        layouts = (
            root.layout() if root is not None else None,
            self.viewport_column.layout(),
            self.metrics_frame.layout(),
        )
        for layout in layouts:
            if layout is not None:
                layout.invalidate()
                layout.activate()
        if root is not None:
            root.updateGeometry()
        self.viewport_column.updateGeometry()
        self.metrics_frame.updateGeometry()
        self.vtk_widget.updateGeometry()
        QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
        if getattr(self, "_closing", False):
            return
        self._position_viewport_overlays()
        render_window = self.vtk_widget.GetRenderWindow()
        if render_window is not None and self.vtk_widget.width() > 0 and self.vtk_widget.height() > 0:
            render_window.SetSize(self.vtk_widget.width(), self.vtk_widget.height())
            render_window.Render()

    def _position_viewport_overlays(self) -> None:
        self._position_imprint_plane_tools()
        self._project_crop_outline()
        self._position_view_pad()
        self._position_orientation_overlays()
        self._position_loading_panel()
        if hasattr(self, "density_legend"):
            self.density_legend.adjustSize()
            self._position_overlay_in_corner(self.density_legend, "density")
            self._apply_rounded_mask(self.density_legend, 9)
        if self.measure_info_frame is not None and self.measure_info_frame.isVisible():
            self.measure_info_frame.adjustSize()
            self.measure_info_frame.move(
                14,
                max(
                    14,
                    self.vtk_widget.height()
                    - self.measure_info_frame.height()
                    - (self.view_pad.height() if hasattr(self, "view_pad") else 0)
                    - 28,
                ),
            )
            self._apply_rounded_mask(self.measure_info_frame, 5)
        if self.crop_info_frame is not None and self.crop_info_frame.isVisible():
            x = max(8, min(self.crop_info_frame.x(), max(8, self.vtk_widget.width() - self.crop_info_frame.width() - 8)))
            y = max(8, min(self.crop_info_frame.y(), max(8, self.vtk_widget.height() - self.crop_info_frame.height() - 8)))
            self.crop_info_frame.move(x, y)
        self._update_measure_overlay(render=False)
        self._update_scale_bar()

    @staticmethod
    def _apply_rounded_mask(widget: QWidget, radius: float) -> None:
        if widget.width() <= 0 or widget.height() <= 0:
            return
        path = QPainterPath()
        path.addRoundedRect(
            0.0, 0.0, float(widget.width()), float(widget.height()), radius, radius
        )
        widget.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def _normalize_viewport_overlay_corners(self) -> None:
        """Keep visible viewport overlays in distinct corners."""
        used: set[str] = set()
        for key, (_label, default_corner) in VIEWPORT_OVERLAYS.items():
            corner = self.viewport_overlay_corners.get(key, default_corner)
            if corner not in VIEWPORT_CORNERS:
                corner = default_corner
            if self.viewport_overlay_visible.get(key, True):
                if corner in used:
                    corner = next(candidate for candidate in VIEWPORT_CORNERS if candidate not in used)
                used.add(corner)
            self.viewport_overlay_corners[key] = corner

    def _overlay_is_visible(self, key: str) -> bool:
        return bool(self.viewport_overlay_visible.get(key, True))

    def _position_overlay_in_corner(self, widget: QWidget, key: str, margin: int = 14) -> None:
        corner = self.viewport_overlay_corners.get(key, VIEWPORT_OVERLAYS[key][1])
        x = margin if corner.endswith("left") else max(margin, self.vtk_widget.width() - widget.width() - margin)
        y = margin if corner.startswith("Top") else max(margin, self.vtk_widget.height() - widget.height() - margin)
        widget.move(x, y)

    def _create_density_legend(self) -> None:
        self.density_legend = DensityLegend(self.vtk_widget)
        self.density_legend.setText(
            "POINT DENSITY   <span style='color:#123070'>Sparse</span>  "
            "<span style='color:#199AD6'>●</span> "
            "<span style='color:#57E0B8'>●</span> "
            "<span style='color:#FACC15'>●</span> "
            "<span style='color:#E23730'>● Dense</span>"
        )
        self.density_legend.setTextFormat(Qt.TextFormat.RichText)
        self.density_legend.setStyleSheet(
            "background: rgba(247,245,241,232); color: #0B1B34; border: 1px solid #657384; "
            "border-radius: 6px; padding: 7px 10px; font-weight: 600;"
        )
        self.density_legend.adjustSize()
        self.density_legend.hide()

    @staticmethod
    def _filetime_value(value) -> int:
        return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

    def _update_system_metrics(self) -> None:
        if hasattr(self, "runtime_metric"):
            from geometry_workload import snapshot
            geometry = " ? ".join(
                f"{device} {fmt_count(v[0])} vertices / {fmt_count(v[1])} triangles / {fmt_bytes(v[2])}"
                for device, v in snapshot().items()
            )
            tail = self.runtime_metric.text().split("Session:", 1)
            self.runtime_metric.setText(geometry + ("  |  Session:" + tail[1] if len(tail) == 2 else ""))
        cpu = float(self.metric_bars["cpu"].value())
        gpu = float(self.metric_bars["gpu"].value())
        available_memory = available_physical_memory()
        total_memory = 0
        if os.name == "nt":
            class FILETIME(ctypes.Structure):
                _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]

            idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
            if ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                current = tuple(self._filetime_value(v) for v in (idle, kernel, user))
                if self._last_cpu_times is not None:
                    idle_delta = current[0] - self._last_cpu_times[0]
                    total_delta = (current[1] - self._last_cpu_times[1]) + (current[2] - self._last_cpu_times[2])
                    cpu = 100.0 * (1.0 - idle_delta / total_delta) if total_delta else 0.0
                    cpu = max(0.0, min(100.0, cpu))
                    self.metric_bars["cpu"].setValue(round(cpu))
                    self.metric_labels["cpu"].setText(
                        f"{cpu:.0f}% used  •  {100-cpu:.0f}% remaining  •  "
                        f"{self.logical_cpu_count} logical processors"
                    )
                self._last_cpu_times = current

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            memory = MEMORYSTATUSEX()
            memory.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                used = int(memory.dwMemoryLoad)
                available_memory = int(memory.ullAvailPhys)
                total_memory = int(memory.ullTotalPhys)
                self.metric_bars["ram"].setValue(used)
                self.metric_labels["ram"].setText(
                    f"{used}% used  •  {100-used}% remaining  •  "
                    f"{fmt_bytes(available_memory)} free / {fmt_bytes(total_memory)} total"
                )

        try:
            cuda_sample = (
                self.cuda_counters.sample(self.gpu_preference)
                if self.cuda_device_list
                else None
            )
            native = None
            try:
                load, dedicated_bytes = self.gpu_counters.sample()
                native = self.gpu_counters.last_sample
            except OSError:
                if cuda_sample is None:
                    raise
                load, dedicated_bytes = cuda_sample
            gpu = load
            self.metric_bars["gpu"].setValue(round(load))
            self.metric_bars["gpu"].setFormat("%p% used")
            if native is not None:
                self.metric_labels["gpu"].setText(
                    f"3D {native['graphics']:.0f}% ? Compute {native['compute']:.0f}%"
                    + (f" ? CUDA GPU {cuda_sample[0]:.0f}%" if cuda_sample is not None else ""))
                self.metric_labels["gpu"].setToolTip(
                    f"{native['name']}. Windows busiest-engine utilization on the adapter rendering MeshMill, including all applications. "
                    f"MeshMill process: {native['process_load']:.1f}%. Rendering and geometry processing may use different adapters."
                    + (f" Selected geometry GPU (NVIDIA): {cuda_sample[0]:.0f}%." if cuda_sample is not None else ""))
                vram_capacity = native["capacity"]
                self.metric_labels["vram"].setToolTip(f"Dedicated memory on {native['name']}, the adapter rendering MeshMill. Includes all applications on that adapter.")
            else:
                self.metric_labels["gpu"].setText(f"{load:.0f}% load ? NVIDIA fallback")
                self.metric_labels["gpu"].setToolTip("NVIDIA utilization for the selected geometry GPU. Windows rendering-engine counters are unavailable; viewport activity may use another adapter.")
                vram_capacity = self.vram_capacity_bytes
                self.metric_labels["vram"].setToolTip("Dedicated memory on the selected NVIDIA geometry GPU. Windows rendering-adapter counters are unavailable.")
            self.metric_bars["gpu"].setToolTip(self.metric_labels["gpu"].toolTip())
            if vram_capacity > 0:
                vram_percent = max(
                    0.0,
                    min(100.0, 100.0 * dedicated_bytes / vram_capacity),
                )
                self.metric_bars["vram"].setValue(round(vram_percent))
                self.metric_bars["vram"].setFormat("%p% used")
                self.metric_labels["vram"].setText(
                    f"{fmt_bytes(round(dedicated_bytes))} used  •  "
                    f"{fmt_bytes(max(0, vram_capacity - round(dedicated_bytes)))} free  •  "
                    f"{fmt_bytes(vram_capacity)} total"
                )
            else:
                self.metric_bars["vram"].setValue(0)
                self.metric_bars["vram"].setFormat("Unavailable")
                self.metric_labels["vram"].setText("Dedicated VRAM capacity unavailable")
        except (OSError, ValueError, IndexError):
            self.metric_bars["gpu"].setValue(0)
            self.metric_bars["gpu"].setFormat("Unavailable")
            self.metric_labels["gpu"].setText("Windows GPU performance counters unavailable")
            self.metric_bars["vram"].setValue(0)
            self.metric_bars["vram"].setFormat("Unavailable")
            self.metric_labels["vram"].setText("Windows GPU memory counters unavailable")

        if os.name == "nt":
            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.GetProcessIoCounters.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(IO_COUNTERS)
            ]
            kernel32.GetProcessIoCounters.restype = ctypes.c_int
            counters = IO_COUNTERS()
            if kernel32.GetProcessIoCounters(
                kernel32.GetCurrentProcess(), ctypes.byref(counters)
            ):
                now = time.monotonic()
                current_io = (int(counters.ReadTransferCount), int(counters.WriteTransferCount), now)
                if self._last_io_totals is not None:
                    elapsed = max(0.001, now - self._last_io_totals[2])
                    read_delta = max(0, current_io[0] - self._last_io_totals[0])
                    write_delta = max(0, current_io[1] - self._last_io_totals[1])
                    if read_delta or write_delta:
                        self._io_active_seconds += elapsed
                    self._peak_read_sample = max(self._peak_read_sample, read_delta)
                    self._peak_write_sample = max(self._peak_write_sample, write_delta)
                self.io_metric.setText(
                    f"I/O  •  Read {fmt_bytes(current_io[0])} total "
                    f"(peak sample {fmt_bytes(self._peak_read_sample)})  •  "
                    f"Written {fmt_bytes(current_io[1])} total "
                    f"(peak sample {fmt_bytes(self._peak_write_sample)})  •  "
                    f"Active {fmt_elapsed(self._io_active_seconds)}"
                )
                self._last_io_totals = current_io
            else:
                self.io_metric.setText("I/O counters unavailable")
        else:
            self.io_metric.setText("I/O counters unavailable")

        if self.busy or not self._capacity_available_memory:
            self._capacity_available_memory = available_memory
        capacity_available_memory = self._capacity_available_memory
        memory_budget = max(
            1, round(capacity_available_memory * self.large_mesh_memory_percent / 100.0)
        )
        estimated_triangles = max(1, memory_budget // ESTIMATED_BYTES_PER_TRIANGLE)
        estimated_vertices = estimated_triangles // 2
        overview_triangle_limit = automatic_overview_triangle_limit(
            memory_budget, self.vram_capacity_bytes
        )
        chunk_memory = max(1, overview_triangle_limit * ESTIMATED_BYTES_PER_TRIANGLE)
        configured_processes = self.geometry_processes or min(
            8, max(1, self.logical_cpu_count // 2)
        )
        parallel_cube_budget = max(
            1, min(configured_processes, memory_budget // chunk_memory)
        )
        self.capacity_metric.setText(
            f"Capacity estimate  •  Full mesh: ~{fmt_count(estimated_triangles)} triangles / "
            f"~{fmt_count(estimated_vertices)} vertices / "
            f"{fmt_bytes(binary_stl_size(estimated_triangles))} binary STL  •  "
            f"Geometry processes: {configured_processes}  •  "
            f"Parallel cube budget: ~{parallel_cube_budget}"
        )

        if self.source_faces is None:
            self.live_mesh_metric.setText("Mesh: no file loaded")
        else:
            preview = self.preview_poly.GetNumberOfPolys() if self.preview_poly is not None else 0
            operation = (
                self.active_operation
                if self.active_operation
                else "Optimizing"
                if self.busy
                else "Selection awaiting action"
                if self.crop_candidate
                else "Ready"
            )
            preview_text = fmt_count(preview) if preview else "not generated"
            if self.overview_mode:
                working_text = (
                    f"Overview: {fmt_count(len(self.source_faces))} sampled / "
                    f"{fmt_count(self.overview_total_triangles)} total triangles"
                )
            else:
                working_text = (
                    f"Original: {fmt_count(len(self.source_points))} vertices / "
                    f"{fmt_count(len(self.source_faces))} triangles"
                )
            self.live_mesh_metric.setText(
                f"{operation}  •  {working_text}  •  Optimized: {preview_text} triangles"
            )

        processing = self.busy or time.monotonic() < self.geometry_activity_until
        operation = (
            self.active_operation
            if self.active_operation
            else self.geometry_activity_label
            if processing
            else "Idle"
        )
        point_count = len(self.source_points) if self.source_points is not None else 0
        triangle_count = len(self.source_faces) if self.source_faces is not None else 0
        self.geometry_activity_graph.add_sample(
            cpu,
            gpu,
            processing,
            operation,
            point_count,
            triangle_count,
        )
        self._configure_metrics_history()

    def _mark_geometry_activity(self, label: str, seconds: float = 2.0) -> None:
        self.geometry_activity_label = label
        self.geometry_activity_until = max(self.geometry_activity_until, time.monotonic() + seconds)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.crop_points:
            self._update_crop_overlay()
        # A maximized native window can receive its final client size after the
        # first Qt resize event. Reposition every viewport overlay from the VTK
        # widget's settled geometry instead of updating only selected overlays.
        QTimer.singleShot(0, self._position_viewport_overlays)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_viewport_layout_refresh()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._sync_full_screen_button()
            self._schedule_viewport_layout_refresh()

    def _schedule_viewport_layout_refresh(self) -> None:
        """Refresh after Qt and Windows settle maximized client geometry."""
        if getattr(self, "_closing", False):
            return
        for delay in (0, 75, 250):
            QTimer.singleShot(delay, self._refresh_viewport_layout)

    def _create_loading_panel(self) -> None:
        self.loading_frame = QFrame(self.vtk_widget)
        self.loading_frame.setObjectName("loadingPanel")
        self.loading_frame.setFixedWidth(430)
        layout = QVBoxLayout(self.loading_frame)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(8)
        self.loading_heading = QLabel("Preparing mesh")
        self.loading_heading.setObjectName("loadingTitle")
        layout.addWidget(self.loading_heading)
        self.loading_file_label = QLabel("Reading STL…")
        self.loading_file_label.hide()
        self.loading_detail_label = QLabel("Reading triangles and welding shared vertices.")
        self.loading_detail_label.hide()
        self.loading_metrics_row = QWidget(self.loading_frame)
        metrics_layout = QHBoxLayout(self.loading_metrics_row)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(5)
        self.loading_metric_labels: dict[str, QLabel] = {}
        for key in ("gpu", "vram", "cpu", "ram"):
            label = QLabel(key.upper(), self.loading_metrics_row)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            label.setProperty("loadingMetric", True)
            metrics_layout.addWidget(label, 1)
            self.loading_metric_labels[key] = label
        layout.addWidget(self.loading_metrics_row)
        self.loading_steps_label = QLabel()
        self.loading_steps_label.hide()
        self.loading_overall_label = QLabel()
        self.loading_overall_label.hide()
        self.loading_overall_progress = QProgressBar()
        self.loading_overall_progress.setRange(0, 100)
        self.loading_overall_progress.setValue(0)
        self.loading_overall_progress.setFormat("Overall %p%")
        layout.addWidget(self.loading_overall_progress)
        self.loading_step_label = QLabel()
        self.loading_step_label.hide()
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 100)
        self.loading_progress.setValue(0)
        self.loading_progress.setFormat("Current step %p%")
        layout.addWidget(self.loading_progress)
        self.loading_elapsed_label = QLabel("Starting…")
        self.loading_elapsed_label.setProperty("muted", True)
        layout.addWidget(self.loading_elapsed_label)
        self.progress_cancel_button = QPushButton("Cancel", self.loading_frame)
        self.progress_cancel_button.setToolTip(
            "Cancel this operation and restore the preceding mesh, selection, and settings."
        )
        self.progress_cancel_button.clicked.connect(self._cancel_mesh_operation)
        self.progress_cancel_button.setMinimumHeight(32)
        self.progress_cancel_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.progress_cancel_button.setEnabled(False)
        layout.addWidget(self.progress_cancel_button)
        self.loading_frame.adjustSize()
        self._apply_rounded_mask(self.loading_frame, 12)
        self.loading_frame.hide()

    def _update_loading_metrics(self) -> None:
        """Render a compact, live resource row without duplicating operation text."""
        if not hasattr(self, "loading_metric_labels") or not hasattr(self, "metric_bars"):
            return
        device = next(
            (
                item
                for item in self.cuda_device_list
                if item["pci_bus_id"] == self.gpu_preference
            ),
            None,
        )
        gpu_name = f"GPU {device['id']}" if device is not None else "GPU Auto"
        self.loading_metric_labels["gpu"].setText(
            f"{gpu_name}  {self.metric_bars['gpu'].value()}%"
        )
        self.loading_metric_labels["vram"].setText(
            f"VRAM  {self.metric_bars['vram'].value()}%"
        )
        self.loading_metric_labels["cpu"].setText(
            f"CPU  {self.metric_bars['cpu'].value()}%"
        )
        self.loading_metric_labels["ram"].setText(
            f"RAM  {self.metric_bars['ram'].value()}%"
        )

    def _operation_step_plan(self, title: str) -> list[str]:
        """Return the user-visible stages for one operation in their execution order."""
        result_analysis = [
            "Preparing result analysis",
            "Analyzing results",
            "Measuring surface area",
            "Preparing topology edges",
            "Sorting topology edges",
            "Checking mesh topology",
            "Preparing optimized mesh",
            "Preparing viewport normals",
        ]
        if title == "Analyzing point density":
            return [
                "Measuring triangle density",
                "Smoothing density regions",
                "Coloring density regions",
            ]
        if title == "Loading":
            return [
                "Reading mesh",
                "Preparing triangles",
                "Measuring source geometry",
                "Preparing viewport normals",
                "Measuring mesh bounds",
                "Planning spatial tiles",
                "Writing spatial tiles",
                "Spatial index ready",
            ]
        if title in {"Optimizing", "Optimizing selection"}:
            first = (
                "Analyzing depth and density"
                if self.algorithm.currentText() == DEPTH_DENSITY_ADAPTIVE_LABEL
                else "Analyzing surface"
                if is_gpu_guided_algorithm(self.algorithm.currentText())
                else "Reducing mesh topology"
            )
            return list(dict.fromkeys([first, "Reducing mesh topology", "Finishing topology and surface", *result_analysis]))
        if title == "Optimizing tiles":
            return ["Optimizing tiles", "Assembling optimized mesh", "Loading optimized mesh", *result_analysis]
        if title == "Exporting":
            return ["Preparing export", "Finding planar faces", "Merging planar faces", "Writing geometry"]
        return ["Preparing mesh", title]

    def _reset_operation_steps(self, title: str) -> None:
        self.operation_steps = self._operation_step_plan(title)
        self.operation_step_index = 0
        self._render_operation_steps()
        self.loading_overall_progress.setRange(0, 100)
        self.loading_overall_progress.setValue(0)
        self.loading_progress.setRange(0, 100)
        self.loading_progress.setValue(0)

    @staticmethod
    def _runtime_bucket(stage: str) -> str:
        """Group progress stages into concise, user-facing runtime categories."""
        normalized = stage.casefold()
        if any(token in normalized for token in ("read", "load", "writing spatial", "planning spatial")):
            return "Reading and indexing"
        if any(token in normalized for token in ("assembling", "preparing optimized mesh")):
            return "Output assembly"
        if any(
            token in normalized
            for token in ("analyzing surface", "analyzing depth", "triangle density", "smoothing density")
        ):
            return "Surface analysis"
        if any(
            token in normalized
            for token in ("reducing", "optimizing tiles", "finishing topology")
        ):
            return "Mesh reduction"
        if any(
            token in normalized
            for token in (
                "analyzing results",
                "surface area",
                "topology edges",
                "sorting topology",
                "checking mesh topology",
                "bounds",
                "coloring density",
            )
        ):
            return "Result checks"
        return "Preparation"

    @staticmethod
    def _runtime_category_order() -> tuple[str, ...]:
        return (
            "Preparation",
            "Reading and indexing",
            "Surface analysis",
            "Mesh reduction",
            "Result checks",
            "Output assembly",
        )

    @staticmethod
    def _runtime_category_short_name(category: str) -> str:
        return {
            "Preparation": "Prep",
            "Reading and indexing": "Read/index",
            "Surface analysis": "Surface",
            "Mesh reduction": "Reduction",
            "Result checks": "Checks",
            "Output assembly": "Assembly",
        }.get(category, category)

    def _start_operation_runtime(self, initial_stage: str = "Preparing") -> None:
        started_at = time.monotonic()
        self.operation_runtime_started_at = started_at
        self.operation_runtime_stage_started_at = started_at
        self.operation_runtime_stage = initial_stage
        self.operation_runtime_seconds = {}

    def _transition_operation_runtime(self, stage: str) -> None:
        if not stage:
            return
        if self.operation_runtime_started_at <= 0.0:
            self._start_operation_runtime(stage)
            return
        if stage == self.operation_runtime_stage:
            return
        now = time.monotonic()
        elapsed = max(0.0, now - self.operation_runtime_stage_started_at)
        bucket = self._runtime_bucket(self.operation_runtime_stage)
        self.operation_runtime_seconds[bucket] = (
            self.operation_runtime_seconds.get(bucket, 0.0) + elapsed
        )
        self.operation_runtime_stage = stage
        self.operation_runtime_stage_started_at = now

    def _finish_operation_runtime(self) -> None:
        if self.operation_runtime_started_at <= 0.0:
            return
        now = time.monotonic()
        elapsed = max(0.0, now - self.operation_runtime_stage_started_at)
        bucket = self._runtime_bucket(self.operation_runtime_stage)
        self.operation_runtime_seconds[bucket] = (
            self.operation_runtime_seconds.get(bucket, 0.0) + elapsed
        )
        total = max(0.0, now - self.operation_runtime_started_at)
        self.last_operation_runtime = dict(self.operation_runtime_seconds)
        self.last_operation_name = self.active_operation
        self.runtime_operation_count += 1
        self.runtime_total_seconds += total
        for name, seconds in self.last_operation_runtime.items():
            if seconds <= 0.005:
                continue
            self.runtime_aggregate_seconds[name] = (
                self.runtime_aggregate_seconds.get(name, 0.0) + seconds
            )
            self.runtime_aggregate_counts[name] = self.runtime_aggregate_counts.get(name, 0) + 1
        if hasattr(self, "runtime_metric"):
            last_parts = []
            for name in self._runtime_category_order():
                seconds = self.last_operation_runtime.get(name, 0.0)
                if seconds > 0.005:
                    last_parts.append(
                        f"{self._runtime_category_short_name(name)} {seconds:.1f}s"
                    )
            detail = "  •  ".join(last_parts) if last_parts else "No measurable stages"
            from geometry_workload import snapshot
            workload = snapshot()
            geometry = " ? ".join(
                f"{device} {fmt_count(values[0])} vertices / {fmt_count(values[1])} triangles / {fmt_bytes(values[2])}"
                for device, values in workload.items()
            )
            self.runtime_metric.setText(
                f"{geometry}  |  Session: {self.runtime_operation_count} jobs • {self.runtime_total_seconds:.1f}s  |  "
                f"Latest: {self.last_operation_name} {total:.1f}s • {detail}"
            )
        self.operation_runtime_started_at = 0.0
        self.operation_runtime_stage_started_at = 0.0
        self.operation_runtime_stage = ""
        self.operation_runtime_seconds = {}

    def _render_operation_steps(self) -> None:
        # Step names remain in the operation plan for the overall calculation. The compact
        # panel shows their count and current stage instead of a verbose checklist.
        return

    def _update_operation_step_progress(self, stage: str, completed: int, total: int) -> None:
        self._transition_operation_runtime(stage)
        canonical_stage = (
            "Reducing mesh topology" if stage in {
                "Preparing QEM vertices", "Preparing QEM triangles", "Building QEM connectivity",
                "Scanning QEM triangles", "Checking QEM boundaries",
            } else stage
        )
        if stage.startswith("Surface probes:"):
            canonical_stage = "Surface probes"
        elif stage == "Building CPU ray index":
            canonical_stage = "Building ray index"
        if canonical_stage not in self.operation_steps:
            self.operation_steps.insert(self.operation_step_index, canonical_stage)
        self.operation_step_index = self.operation_steps.index(canonical_stage)
        step_fraction = max(0.0, min(1.0, completed / total)) if total > 0 else 0.0
        overall = round(100.0 * (self.operation_step_index + step_fraction) / len(self.operation_steps))
        overall = max(self.loading_overall_progress.value(), min(99, overall))
        self.loading_overall_progress.setValue(overall)
        self.loading_overall_progress.setFormat(
            f"Overall {overall}%  •  step {self.operation_step_index + 1}/{len(self.operation_steps)}"
        )
        if total > 0:
            self.loading_progress.setRange(0, 1000)
            self.loading_progress.setValue(round(1000 * step_fraction))
            amount = (f"{fmt_bytes(completed)} / {fmt_bytes(total)}" if stage == "Reading mesh"
                      else f"{completed:,} / {total:,}")
            self.loading_progress.setFormat(f"%p%  •  {amount}")
        else:
            self.loading_progress.setRange(0, 100)
            self.loading_progress.setValue(0)
            self.loading_progress.setFormat("Preparing worker")
        self._render_operation_steps()

    def _position_loading_panel(self) -> None:
        if not hasattr(self, "loading_frame"):
            return
        self.loading_frame.adjustSize()
        self._apply_rounded_mask(self.loading_frame, 12)
        x = max(12, (self.vtk_widget.width() - self.loading_frame.width()) // 2)
        y = max(12, (self.vtk_widget.height() - self.loading_frame.height()) // 2)
        self.loading_frame.move(x, y)
        if self.loading_frame.isVisible():
            self.loading_frame.raise_()

    def _update_loading_panel(self) -> None:
        if not self.busy:
            return
        elapsed = max(0.0, time.monotonic() - self.load_started_at)
        self.loading_elapsed_label.setText(f"Local  •  Elapsed: {elapsed:.1f} s")
        if self.operation_progress_path is not None:
            try:
                progress_state = json.loads(
                    self.operation_progress_path.read_text(encoding="utf-8")
                )
                stage = str(progress_state.get("stage", ""))
                completed = int(progress_state.get("completed", 0))
                total = int(progress_state.get("total", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                stage, completed, total = "", 0, 0
            if stage and stage != self.operation_progress_stage:
                self.operation_progress_stage = stage
                details = {
                    "Analyzing surface": "Surface analysis",
                    "Analyzing depth and density": "Depth and density analysis",
                    "Reducing mesh topology": "Reducing topology",
                    "Finishing topology and surface": "Finishing reduction",
                    "Preparing result analysis": "Preparing analysis",
                    "Analyzing results": "Measuring result quality",
                    "Measuring surface area": "Measuring surface area",
                    "Preparing topology edges": "Preparing topology",
                    "Sorting topology edges": "Sorting topology",
                    "Checking mesh topology": "Checking topology",
                    "Measuring triangle density": "Measuring density",
                    "Smoothing density regions": "Smoothing density",
                    "Coloring density regions": "Coloring density",
                    "Preparing optimized mesh": "Preparing viewport",
                    "Optimizing tiles": "Optimizing tiles",
                    "Assembling optimized mesh": "Assembling mesh",
                    "Loading optimized mesh": "Loading result",
                    "Reading mesh": "Reading mesh",
                    "Measuring mesh bounds": "Measuring bounds",
                    "Planning spatial tiles": "Planning tiles",
                    "Writing spatial tiles": "Writing tiles",
                    "Spatial index ready": "Tile index ready",
                }
                self.loading_heading.setText(stage)
                self.loading_detail_label.setText(details.get(stage, "Processing mesh data"))
            if stage:
                self._update_operation_step_progress(stage, completed, total)
        self._update_loading_metrics()
        if (
            self.loading
            and self.load_geometry_ready
            and self.gpu_warmup_future is not None
            and not self.gpu_warmup_future.done()
        ):
            self.loading_heading.setText("Preparing GPU optimization")
            self.loading_detail_label.setText("Initializing GPU runtime")
            self.loading_progress.setRange(0, 100)
            self.loading_progress.setValue(0)
            self.loading_progress.setFormat(f"GPU setup  •  {elapsed:.1f} s elapsed")

    def _set_controls_busy(self, busy: bool) -> None:
        split_widget = getattr(self, "_imprint_split_widget", None)
        if split_widget is not None:
            split_widget.SetProcessEvents(not busy and not self.imprint_pick_surface.isChecked()
                                          and getattr(self, "_imprint_plane_mode", "Rotate") == "Move")
        if hasattr(self, "imprint_plane_tools"):
            self.imprint_plane_tools.setEnabled(not busy)
        interactive_types = (QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QSlider)
        if busy:
            self._busy_control_states = {
                widget: widget.isEnabled()
                for widget in self.controls_content.findChildren(QWidget)
                if isinstance(widget, interactive_types)
            }
            for widget in self._busy_control_states:
                widget.setEnabled(False)
            self.progress_cancel_button.setEnabled(True)
            return
        for widget, enabled in self._busy_control_states.items():
            widget.setEnabled(enabled)
        self._busy_control_states.clear()
        self.progress_cancel_button.setEnabled(False)

    def _capture_operation_snapshot(self) -> None:
        self.operation_snapshot = {
            "quality": self.smart_quality.currentText(),
            "algorithm": self.algorithm.currentText(),
            "target": self.target.value(),
            "display": self._display_mode(),
            "units": self.units.currentText(),
            "status": self.status.text(),
            "selection": (
                self.selected_face_mask.copy() if self.selected_face_mask is not None else None
            ),
            "selection_undo_stack": [
                mask.copy() if mask is not None else None for mask in self.selection_undo_stack
            ],
            "crop_candidate": self.crop_candidate,
            "crop_points": [QPoint(point) for point in self.crop_points],
            "selection_target": self.selection_target,
            "selection_base_quality": self.selection_base_quality,
            "selection_base_algorithm": self.selection_base_algorithm,
            "selection_base_target": self.selection_base_target,
            "preview_dirty": self.preview_dirty,
            "pass_ready": self.pass_ready,
            "quality_metrics": getattr(self, "_last_quality_metrics", None),
            "quality_measured": getattr(self, "_last_quality_measured", False),
            "button_states": {
                widget: widget.isEnabled()
                for widget in (
                    self.preview_button,
                    self.save_button,
                    self.toggle_mesh_button,
                    self.previous_mesh_button,
                    self.reload_button,
                    self.undo_button,
                    self.cancel_preview_button,
                    self.redo_button,
                    self.delete_selection_button,
                    self.clear_selection_button,
                )
            },
        }
        self.operation_cancel_event = threading.Event()

    def _cancel_mesh_operation(self) -> None:
        if not self.busy:
            return
        cancelled_title = self.active_operation or "Operation"
        cancelled_selection_optimization = cancelled_title == "Optimizing selection"
        if self.operation_cancel_event is not None:
            self.operation_cancel_event.set()
        if self.operation_future is not None:
            self.operation_future.cancel()

        if self.operation_uses_process:
            self.optimization_executor.terminate_and_join()
            self.optimization_executor = OwnedProcessPool(max_workers=1)

        self.generation += 1
        self.edit_generation += 1
        self.load_generation += 1
        self.density_generation += 1
        self.pending_target = None
        self.running_target = None
        self.loading = False
        self.load_geometry_ready = False
        self.gpu_warmup_future = None

        # Native mesh routines may not expose an interruption hook. Retire their worker so a
        # cancelled native call cannot block the next requested operation.
        retired_executor = self.executor
        retired_executor.shutdown(wait=False, cancel_futures=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh-reducer")

        snapshot = self.operation_snapshot
        self._end_mesh_operation()
        if snapshot is not None:
            with QSignalBlocker(self.smart_quality):
                self.smart_quality.setCurrentText(snapshot["quality"])
            with QSignalBlocker(self.algorithm):
                self.algorithm.setCurrentText(snapshot["algorithm"])
            with QSignalBlocker(self.target):
                self.target.setValue(snapshot["target"])
            with QSignalBlocker(self.display_type):
                display_index = self.display_type.findData(snapshot["display"])
                if display_index >= 0:
                    self.display_type.setCurrentIndex(display_index)
            with QSignalBlocker(self.units):
                self.units.setCurrentText(snapshot["units"])
            self.selected_face_mask = snapshot["selection"]
            self.selection_undo_stack = snapshot["selection_undo_stack"]
            self.crop_candidate = snapshot["crop_candidate"]
            self.crop_points = snapshot["crop_points"]
            self._anchor_crop_outline()
            self.selection_target = snapshot["selection_target"]
            self.selection_base_quality = snapshot["selection_base_quality"]
            self.selection_base_algorithm = snapshot["selection_base_algorithm"]
            self.selection_base_target = snapshot["selection_base_target"]
            self.preview_dirty = snapshot["preview_dirty"]
            self._set_primary_action(
                snapshot["pass_ready"], snapshot["button_states"][self.preview_button]
            )
            for widget, enabled in snapshot["button_states"].items():
                widget.setEnabled(enabled)
            if cancelled_selection_optimization:
                self._clear_selection(message=None)
            elif self.crop_candidate is not None:
                self._show_crop_highlight(self.crop_candidate[0])
                self._update_crop_overlay()
            elif self.selected_face_mask is not None and self.selected_face_mask.any():
                self._refresh_persistent_selection()
            for actor in (self.source_actor, self.simplified_actor):
                if actor is not None:
                    self._set_actor_display_mode(actor)
            self._update_quality_metrics(snapshot["quality_metrics"], measured=snapshot["quality_measured"])
            self._update_estimated_size()
            self._update_dimension_labels()
            self._sync_slider_from_target()
            self._update_processing_plan()
            if not cancelled_selection_optimization and self._selected_triangle_count():
                self._show_selection_primary_action()
            self.vtk_widget.GetRenderWindow().Render()
        if cancelled_selection_optimization:
            self.status.setText("Selection optimization cancelled • selection cleared")
        else:
            self.status.setText(f"{cancelled_title} cancelled • previous state restored")

    def _begin_mesh_operation(self, title: str, detail: str) -> int:
        self.timer.stop()
        # The retained-selection action card is useful while editing. Once an operation starts,
        # the single progress panel is the only operation overlay.
        if self.crop_info_frame is not None:
            self.crop_info_frame.hide()
        self._capture_operation_snapshot()
        self._mark_geometry_activity(title, 2.0)
        self.edit_generation += 1
        self.busy = True
        self.operation_uses_process = False
        self._clear_operation_progress()
        self.active_operation = title
        self.load_started_at = time.monotonic()
        self._start_operation_runtime()
        self._last_loading_metric_refresh = 0.0
        self.load_geometry_ready = False
        self.gpu_warmup_future = None
        self._set_controls_busy(True)
        self._reset_operation_steps(title)
        self.loading_heading.setText(title)
        self.loading_file_label.setText(detail)
        self.loading_detail_label.setText("Preparing geometry")
        self.loading_elapsed_label.setText("Starting…")
        self.loading_frame.show()
        self._position_loading_panel()
        self.loading_frame.raise_()
        self.loading_timer.start()
        QApplication.processEvents()
        return self.edit_generation

    def _end_mesh_operation(self) -> None:
        self._finish_operation_runtime()
        self.loading_timer.stop()
        self.loading_frame.hide()
        self._set_controls_busy(False)
        self.busy = False
        self.active_operation = ""
        self.operation_snapshot = None
        self.operation_cancel_event = None
        self.operation_future = None
        self.operation_uses_process = False
        self._clear_operation_progress()
        self._sync_modified_mesh_actions()
        self._refresh_imprint_preview_actions()

    def _clear_operation_progress(self) -> None:
        progress_path = self.operation_progress_path
        self.operation_progress_path = None
        self.operation_progress_stage = ""
        if progress_path is not None:
            try:
                progress_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _start_operation_progress(self, token: int, stage: str) -> str:
        progress_path = Path(tempfile.gettempdir()) / (
            f"meshmill-progress-{os.getpid()}-{token}.txt"
        )
        self.operation_progress_path = progress_path
        self.operation_progress_stage = ""
        _write_process_progress(os.fspath(progress_path), stage, 0, 1)
        self._transition_operation_runtime(stage)
        return os.fspath(progress_path)

    def _create_view_pad(self) -> None:
        self.view_pad = QFrame(self.vtk_widget)
        self.view_pad.setObjectName("viewPad")
        grid = QGridLayout(self.view_pad)
        grid.setContentsMargins(7, 7, 7, 7)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(5)
        layout = [
            ("Left", "left", 0, 0),
            ("Front", "front", 0, 1),
            ("Right", "right", 0, 2),
            ("Top", "top", 1, 0),
            ("Back", "back", 1, 1),
            ("Bottom", "bottom", 1, 2),
        ]
        self.view_buttons: dict[str, ViewButton] = {}
        for label, direction, row, column in layout:
            button = ViewButton(label, self.view_pad)
            self._set_theme_icon(button, "view-" + direction)
            button.setCheckable(True)
            button.toggled.connect(lambda _, b=button, d=direction: self._apply_theme_icon(b, "view-" + d))
            button.pressed.connect(lambda b=button, d=direction: self._apply_theme_icon(b, "view-" + d))
            button.released.connect(lambda b=button, d=direction: self._apply_theme_icon(b, "view-" + d))
            button.clicked.connect(lambda _=False, d=direction: self._set_standard_view(d))
            button.orientationRequested.connect(lambda d=direction: self._set_current_view_as(d))
            grid.addWidget(button, row, column)
            self.view_buttons[direction] = button
        self._refresh_view_button_tooltips()
        from lathe_control import LatheController
        self.lathe = LatheController(self, grid, UNIT_MM)
        self.navigation_style._manual_zoom_callback = self.lathe.mark_manual_zoom
        self._reorder_tool_sections()
        self.navigation_style._lathe_callback = self.lathe.interaction
        self.orientation_indicator = OrientationIndicator(self.vtk_widget)
        self._orientation_mesh_cache = {}
        self._orientation_mesh_requested = None
        self._orientation_mesh_future = None
        self.orientation_mesh_timer = QTimer(self)
        self.orientation_mesh_timer.setInterval(60)
        self.orientation_mesh_timer.timeout.connect(self._poll_orientation_miniature)
        self.orientation_indicator.show()
        self.orientation_indicator.raise_()
        self.scale_bar = ViewportScaleBar(self.vtk_widget)
        self.scale_bar.hide()
        self.scale_bar.raise_()
        camera = self.renderer.GetActiveCamera()
        camera.AddObserver("ModifiedEvent", self._update_orientation_indicator)
        camera.AddObserver("ModifiedEvent", self._diagnostic_camera_modified)
        camera.AddObserver("ModifiedEvent", self._queue_crop_outline_projection)
        self.vtk_widget.GetRenderWindow().AddObserver("EndEvent", self._repaint_orientation_indicator)
        self._update_orientation_indicator()
        self.view_pad.adjustSize()
        self.view_pad.raise_()

    def _refresh_view_button_tooltips(self) -> None:
        if not hasattr(self, "view_buttons"):
            return
        for direction, button in self.view_buttons.items():
            shortcut = self.shortcuts.get(f"view_{direction}", "")
            set_shortcut = self.shortcuts.get(f"set_view_{direction}", "")
            shortcut_text = f" View shortcut: {shortcut}." if shortcut else ""
            set_shortcut_text = f" Set shortcut: {set_shortcut}." if set_shortcut else ""
            button.setToolTip(
                f"View {direction}.{shortcut_text} Ctrl+click sets the current view as {direction}."
                f"{set_shortcut_text}"
            )

    @staticmethod
    def _build_orientation_miniature(poly):
        from vtkmodules.vtkFiltersCore import vtkQuadricClustering
        geometry = vtkPolyData()
        geometry.SetPoints(poly.GetPoints())
        geometry.SetPolys(poly.GetPolys())
        for divisions in (12, 8, 5, 3):
            if geometry.GetNumberOfPolys() <= 240:
                break
            bounds = np.asarray(geometry.GetBounds()).reshape(3, 2)
            spans = np.ptp(bounds, axis=1)
            counts = np.maximum(2, np.rint(divisions * spans / max(float(spans.max()), 1e-12))).astype(int)
            cluster = vtkQuadricClustering()
            cluster.SetInputData(geometry)
            cluster.SetNumberOfDivisions(*map(int, counts))
            cluster.UseInputPointsOn()
            cluster.Update()
            geometry = cluster.GetOutput()
        points, faces = polydata_arrays(geometry)
        if not len(points) or not len(faces):
            return None
        points = np.asarray(points, dtype=float).copy()
        points -= (points.min(axis=0) + points.max(axis=0)) / 2
        points *= .82 / max(float(np.linalg.norm(points, axis=1).max()), 1e-12)
        if len(faces) > 240:
            faces = faces[np.linspace(0, len(faces) - 1, 240, dtype=int)]
        return points, faces.copy()

    def _request_orientation_miniature(self):
        if not hasattr(self, "orientation_mesh_timer") or getattr(self, "_closing", False):
            return
        cast = getattr(self, "_envelope_actor", None)
        actor = cast if cast is not None and cast.GetVisibility() else self.active_actor
        poly = actor.GetMapper().GetInput() if actor is not None and actor.GetMapper() is not None else None
        if poly is None or not poly.GetNumberOfPolys():
            self._orientation_mesh_requested = None
            self.orientation_indicator.set_miniature(None)
            return
        key = (id(poly), poly.GetPoints().GetMTime(), poly.GetPolys().GetMTime())
        requested = self._orientation_mesh_requested
        if requested is not None and requested[0] == key:
            return
        self._orientation_mesh_requested = (key, poly)
        self.orientation_indicator.set_miniature(self._orientation_mesh_cache.get(key))
        self._start_orientation_miniature()

    def _start_orientation_miniature(self):
        if self._orientation_mesh_future is not None or self._orientation_mesh_requested is None:
            return
        key, poly = self._orientation_mesh_requested
        if key in self._orientation_mesh_cache:
            return
        self._orientation_mesh_job_key = key
        # VTK filters may build or modify internal array/cell caches. Keep the
        # background miniature pipeline independent of the live OpenGL mapper.
        snapshot = vtkPolyData()
        snapshot.DeepCopy(poly)
        self._orientation_mesh_future = self.executor.submit(self._build_orientation_miniature, snapshot)
        self.orientation_mesh_timer.start()

    def _poll_orientation_miniature(self):
        if getattr(self, "_closing", False):
            self.orientation_mesh_timer.stop()
            return
        future = self._orientation_mesh_future
        if future is None or not future.done():
            return
        self.orientation_mesh_timer.stop()
        self._orientation_mesh_future = None
        try:
            miniature = future.result()
        except Exception:  # noqa: BLE001 - optional orientation preview does not affect the mesh
            miniature = None
        key = self._orientation_mesh_job_key
        self._orientation_mesh_cache[key] = miniature
        while len(self._orientation_mesh_cache) > 4:
            self._orientation_mesh_cache.pop(next(iter(self._orientation_mesh_cache)))
        if self._orientation_mesh_requested is not None and self._orientation_mesh_requested[0] == key:
            self.orientation_indicator.set_miniature(miniature)
        else:
            self._start_orientation_miniature()

    def _repaint_orientation_indicator(self, *_args) -> None:
        if getattr(self, "_closing", False) or not self.orientation_indicator.isVisible():
            return
        # Match the camera used by the completed VTK frame. Deferred Qt paints
        # can otherwise trail continuous interactive rendering.
        self._update_orientation_indicator()
        self.orientation_indicator.repaint()

    def _update_orientation_indicator(self, *_args, force: bool = False) -> None:
        if not hasattr(self, "orientation_indicator"):
            return
        if not force and getattr(getattr(self,"lathe",None),"_applying",False):
            return
        camera = self.renderer.GetActiveCamera()
        if hasattr(self, "lathe"):
            self.lathe.remember_camera()
        ruler_basis = np.asarray((*camera.GetDirectionOfProjection(), *camera.GetViewUp()))
        previous_basis = getattr(self, "_ruler_camera_basis", None)
        self._ruler_camera_basis = ruler_basis
        if (previous_basis is not None and not np.allclose(previous_basis, ruler_basis, atol=1e-8)
                and (self.measure_points or self.measure_hover_world is not None)
                and not getattr(self, "_ruler_rotation_check_pending", False)):
            self._ruler_rotation_check_pending = True
            def check_rotation():
                self._ruler_rotation_check_pending = False
                if getattr(self, "_closing", False):
                    return
                current = np.asarray((*camera.GetDirectionOfProjection(), *camera.GetViewUp()))
                if not np.allclose(previous_basis, current, atol=1e-8):
                    self._clear_measurements()
            QTimer.singleShot(0, check_rotation)
        self._update_active_view_button(camera)
        orientation = camera.GetOrientation()
        if hasattr(self, "lathe"):
            calibrated = self.lathe.display_orientation(camera)
            if calibrated is not None:
                orientation = calibrated
        if any(button.isChecked() for button in self.view_buttons.values()):
            orientation = (0.0, 0.0, 0.0)
        self.orientation_indicator.set_camera(
            camera.GetPosition(),
            camera.GetFocalPoint(),
            camera.GetViewUp(),
            orientation,
            getattr(self, "orientation_reference_basis", None),
        )
        self._update_scale_bar(camera)
        if hasattr(self, "lathe"):
            self.lathe.refresh_probe_positions()

    def _set_orientation_reference(self, camera=None) -> None:
        """Use the current camera frame as the local zero for the viewport indicator."""
        camera = camera or self.renderer.GetActiveCamera()
        position = np.asarray(camera.GetPosition(), dtype=float)
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        forward = focal - position
        forward /= max(float(np.linalg.norm(forward)), 1e-12)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        up -= forward * float(up @ forward)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        right = np.cross(forward, up)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        self.orientation_reference_basis = np.column_stack((right, up, forward))

    def _establish_opening_front(self) -> None:
        """Use the opening camera frame as this mesh's fixed view reference."""
        camera = self.renderer.GetActiveCamera()
        front = np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
        front /= max(float(np.linalg.norm(front)), 1e-12)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        up -= front * float(up @ front)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        left = np.cross(front, up)
        self.saved_standard_views = {
            "front": (front.copy(), up.copy()),
            "back": (-front, up.copy()),
            "left": (left, up.copy()),
            "right": (-left, up.copy()),
            "top": (up.copy(), front.copy()),
            "bottom": (-up, front.copy()),
        }
        self._set_orientation_reference(camera)
        self._update_orientation_indicator()

    def _update_scale_bar(self, camera=None) -> None:
        if not hasattr(self, "scale_bar") or self.source_poly is None:
            if hasattr(self, "scale_bar"):
                self.scale_bar.hide()
            return
        height = max(1, self.vtk_widget.height())
        camera = camera or self.renderer.GetActiveCamera()
        if camera.GetParallelProjection():
            world_per_pixel = (2.0 * float(camera.GetParallelScale())) / height
        else:
            position = np.asarray(camera.GetPosition(), dtype=float)
            focal = np.asarray(camera.GetFocalPoint(), dtype=float)
            distance = max(float(np.linalg.norm(position - focal)), 1e-9)
            visible_height = 2.0 * distance * math.tan(math.radians(camera.GetViewAngle()) * 0.5)
            world_per_pixel = visible_height / height
        unit = self.units.currentText() if hasattr(self, "units") else "mm"
        raw_value = world_per_pixel * 130.0 / UNIT_MM[unit]
        if raw_value <= 0 or not math.isfinite(raw_value):
            self.scale_bar.hide()
            return
        exponent = math.floor(math.log10(raw_value))
        scale = 10.0 ** exponent
        normalized = raw_value / scale
        nice = (5.0 if normalized >= 5.0 else 2.0 if normalized >= 2.0 else 1.0) * scale
        pixels = nice * UNIT_MM[unit] / world_per_pixel
        decimals = max(0, -exponent)
        label = f"{nice:,.{min(decimals, 3)}f} {unit}"
        self.scale_bar.set_scale(label, pixels)
        layout_key = (self.scale_bar.size(), self.vtk_widget.size(), self.pos(),
                      tuple(sorted(self.viewport_overlay_corners.items())),
                      tuple(sorted(self.viewport_overlay_visible.items())))
        if layout_key != getattr(self, "_scale_overlay_layout_key", None):
            self._scale_overlay_layout_key = layout_key
            self._apply_rounded_mask(self.scale_bar, 8)
            self._position_orientation_overlays()
            self.scale_bar.setVisible(self._overlay_is_visible("scale"))
            self.scale_bar.raise_()

    def _update_active_view_button(self, camera=None) -> None:
        if hasattr(self, "lathe"):
            self.lathe.button.setEnabled(self.source_poly is not None and not self.busy)
        if not hasattr(self, "view_buttons"):
            return
        if self.source_poly is None:
            for button in self.view_buttons.values():
                button.setChecked(False)
            return
        camera = camera or self.renderer.GetActiveCamera()
        position = np.asarray(camera.GetPosition(), dtype=float)
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        view = position - focal
        view /= max(float(np.linalg.norm(view)), 1e-12)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        up -= view * float(up @ view)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        active = None
        for direction, (expected_view, expected_up) in STANDARD_VIEWS.items():
            expected_view, expected_up = self.saved_standard_views.get(
                direction,
                (np.asarray(expected_view, dtype=float), np.asarray(expected_up, dtype=float)),
            )
            if (
                float(view @ expected_view) > 0.99999
                and float(up @ expected_up) > 0.99999
            ):
                active = direction
                break
        for direction, button in self.view_buttons.items():
            button.setChecked(direction == active)

    def _position_view_pad(self) -> None:
        if not hasattr(self, "view_pad"):
            return
        self.view_pad.adjustSize()
        if hasattr(self, "lathe"):
            self.lathe.refresh_probe_positions()
        self._apply_rounded_mask(self.view_pad, 10)
        self._position_overlay_in_corner(self.view_pad, "views")
        if self._overlay_is_visible("views") and not self.view_pad.isVisible():
            self.view_pad.show()
        elif not self._overlay_is_visible("views"):
            self.view_pad.hide()
        self.view_pad.raise_()

    def _position_orientation_overlays(self) -> None:
        if hasattr(self, "orientation_indicator"):
            self._position_overlay_in_corner(self.orientation_indicator, "orientation")
            self.orientation_indicator.setVisible(self._overlay_is_visible("orientation"))
            self.orientation_indicator.raise_()
        if hasattr(self, "scale_bar"):
            self._position_overlay_in_corner(self.scale_bar, "scale")
            if not self._overlay_is_visible("scale"):
                self.scale_bar.hide()
            self.scale_bar.raise_()

    def _fit_view_bounds(self, bounds, center=None) -> None:
        """Fit projected bounds with a three-percent margin on each viewport edge."""
        camera = self.renderer.GetActiveCamera()
        corners = np.array([(x, y, z) for x in bounds[:2]
                            for y in bounds[2:4] for z in bounds[4:6]], dtype=float)
        center = ((corners.min(axis=0)+corners.max(axis=0))/2
                  if center is None else np.asarray(center,dtype=float))
        view = np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
        view /= max(float(np.linalg.norm(view)), 1e-12)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        up -= view * float(up @ view)
        up /= max(float(np.linalg.norm(up)), 1e-12)
        right = np.cross(up, view)
        offsets = corners - center
        x, y, z = offsets @ right, offsets @ up, offsets @ view
        aspect = max(float(self.renderer.GetTiledAspectRatio()), 1e-9)
        usable = .94
        if camera.GetParallelProjection():
            camera.SetParallelScale(max(float(np.max(np.abs(y))),
                                        float(np.max(np.abs(x))) / aspect, 1e-9) / usable)
            distance = max(float(np.linalg.norm(camera.GetPosition() - center)),
                           float(np.max(z)) + 1)
        else:
            tangent = math.tan(math.radians(camera.GetViewAngle()) / 2)
            tan_x, tan_y = ((tangent, tangent / aspect) if camera.GetUseHorizontalViewAngle()
                            else (tangent * aspect, tangent))
            distance = max(float(np.max(z + np.abs(x) / (tan_x * usable))),
                           float(np.max(z + np.abs(y) / (tan_y * usable))), 1e-9)
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + view * distance))

    def _set_standard_view(self, direction: str) -> None:
        if direction not in STANDARD_VIEWS:
            return
        self._preserve_crop_outline_for_navigation()
        if self.active_actor is not None:
            bounds = self.active_actor.GetBounds()
        elif self.source_poly is not None:
            bounds = self.source_poly.GetBounds()
        else:
            for button in self.view_buttons.values():
                button.setChecked(False)
            return
        center = self._keyboard_object_pivot()
        if center is None:
            center = np.asarray(((bounds[0]+bounds[1])/2,(bounds[2]+bounds[3])/2,(bounds[4]+bounds[5])/2))
        diagonal = float(np.linalg.norm((bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])))
        distance = max(1.0, diagonal * 2.0)
        self._diagnostic("recall_view_begin", direction=direction, camera=self._camera_state())
        canonical_view, canonical_up = STANDARD_VIEWS[direction]
        view_direction, view_up = self.saved_standard_views.get(
            direction,
            (np.asarray(canonical_view, dtype=float), np.asarray(canonical_up, dtype=float)),
        )
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + distance * view_direction))
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCamera(*bounds)
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        self._fit_view_bounds(bounds,center=center)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
        self._update_orientation_indicator()
        self.view_pad.raise_()
        self._diagnostic(
            "recall_view_complete",
            direction=direction,
            used_saved_view=direction in self.saved_standard_views,
            camera=self._camera_state(),
        )

    def _set_current_view_as(self, direction: str) -> None:
        if self.source_poly is None:
            return
        self._preserve_crop_outline_for_navigation()
        camera = self.renderer.GetActiveCamera()
        old_position = np.asarray(camera.GetPosition(), dtype=float).copy()
        old_focal = np.asarray(camera.GetFocalPoint(), dtype=float).copy()
        current_view = old_position - old_focal
        current_view /= max(float(np.linalg.norm(current_view)), 1e-12)
        current_up = np.asarray(camera.GetViewUp(), dtype=float).copy()
        current_up -= current_view * float(current_up @ current_view)
        current_up /= max(float(np.linalg.norm(current_up)), 1e-12)
        self._diagnostic("save_view_requested", direction=direction, camera=self._camera_state())
        answer = QMessageBox.question(
            self,
            "Confirm viewport orientation",
            f"Set the current camera angle as {direction.title()}?\n\n"
            f"The current X, Y, and Z corrections become zero for {direction.title()}. "
            "Other saved views remain unchanged.",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer != QMessageBox.StandardButton.Save:
            self._diagnostic("save_view_cancelled", direction=direction)
            return
        self.saved_standard_views[direction] = (current_view.copy(), current_up.copy())
        opposite = OPPOSITE_VIEWS[direction]
        self.saved_standard_views[opposite] = (-current_view.copy(), current_up.copy())
        self._set_orientation_reference(camera)
        self._update_orientation_indicator()
        self.vtk_widget.GetRenderWindow().Render()
        self.view_pad.raise_()
        self.status.setText(f"{direction.title()} view saved")
        self._diagnostic(
            "save_view_complete",
            direction=direction,
            opposite_direction=opposite,
            saved_view=current_view.tolist(),
            saved_up=current_up.tolist(),
            camera=self._camera_state(),
        )

    def _camera_state(self) -> dict:
        camera = self.renderer.GetActiveCamera()
        return {
            "position": list(camera.GetPosition()),
            "focal_point": list(camera.GetFocalPoint()),
            "view_up": list(camera.GetViewUp()),
            "orientation": list(camera.GetOrientation()),
            "parallel_scale": float(camera.GetParallelScale()),
            "active_view": next(
                (
                    direction
                    for direction, button in getattr(self, "view_buttons", {}).items()
                    if button.isChecked()
                ),
                None,
            ),
        }

    def _diagnostic(self, event_name: str, **details) -> None:
        path = self.diagnostic_log_path
        if path is None:
            return
        record = {"time": time.time(), "event": event_name, **details}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError:
            pass

    def _diagnostic_camera_modified(self, *_args) -> None:
        now = time.monotonic()
        if now - self._last_diagnostic_camera_time < 0.08:
            return
        self._last_diagnostic_camera_time = now
        self._diagnostic("camera_modified", camera=self._camera_state())

    @staticmethod
    def _orientation_rotation(
        current_view: np.ndarray,
        current_up: np.ndarray,
        target_view: np.ndarray,
        target_up: np.ndarray,
    ) -> np.ndarray:
        """Return the proper rotation mapping one complete camera frame to another."""
        source_view = np.asarray(current_view, dtype=float)
        source_view /= max(float(np.linalg.norm(source_view)), 1e-12)
        source_up = np.asarray(current_up, dtype=float)
        source_up -= source_view * float(source_up @ source_view)
        source_up /= max(float(np.linalg.norm(source_up)), 1e-12)
        source_right = np.cross(source_up, source_view)
        source_right /= max(float(np.linalg.norm(source_right)), 1e-12)

        destination_view = np.asarray(target_view, dtype=float)
        destination_view /= max(float(np.linalg.norm(destination_view)), 1e-12)
        destination_up = np.asarray(target_up, dtype=float)
        destination_up -= destination_view * float(destination_up @ destination_view)
        destination_up /= max(float(np.linalg.norm(destination_up)), 1e-12)
        destination_right = np.cross(destination_up, destination_view)
        destination_right /= max(float(np.linalg.norm(destination_right)), 1e-12)

        source_frame = np.column_stack((source_right, source_up, source_view))
        target_frame = np.column_stack((destination_right, destination_up, destination_view))
        rotation = target_frame @ source_frame.T
        u, _singular_values, vh = np.linalg.svd(rotation)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vh
        return rotation

    @staticmethod
    def _anchored_orientation_rotation(
        current_view: np.ndarray, target_view: np.ndarray, anchor_view: np.ndarray
    ) -> np.ndarray:
        """Map a view onto its target using rotation only around an established axis."""
        axis = np.asarray(anchor_view, dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        source = np.asarray(current_view, dtype=float) - axis * float(current_view @ axis)
        target = np.asarray(target_view, dtype=float) - axis * float(target_view @ axis)
        source_length = float(np.linalg.norm(source))
        target_length = float(np.linalg.norm(target))
        if source_length <= 1e-9 or target_length <= 1e-9:
            return np.eye(3)
        source /= source_length
        target /= target_length
        sine = float(axis @ np.cross(source, target))
        cosine = max(-1.0, min(1.0, float(source @ target)))
        angle = math.atan2(sine, cosine)
        x, y, z = axis
        cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)

    def _keyboard_view_pivot(self) -> np.ndarray | None:
        """Return the visible mesh under the screen center for keyboard navigation."""
        if self.renderer is None:
            return None
        render_window = self.vtk_widget.GetRenderWindow()
        width, height = render_window.GetSize()
        if width > 0 and height > 0 and self.keyboard_picker.Pick(
            width * 0.5, height * 0.5, 0.0, self.renderer
        ):
            pivot = np.asarray(self.keyboard_picker.GetPickPosition(), dtype=float)
            if pivot.shape == (3,) and np.all(np.isfinite(pivot)):
                return pivot
        actor = self.active_actor
        if actor is not None:
            bounds = np.asarray(actor.GetBounds(), dtype=float)
            if (
                bounds.shape == (6,)
                and np.all(np.isfinite(bounds))
                and np.all(bounds[[1, 3, 5]] >= bounds[[0, 2, 4]])
            ):
                return (bounds[[0, 2, 4]] + bounds[[1, 3, 5]]) * 0.5
        return None

    def _keyboard_object_pivot(self) -> np.ndarray | None:
        lathe = getattr(self, "lathe", None)
        origin = lathe.navigation_origin() if lathe is not None else None
        if origin is not None:
            return origin
        actor = self.active_actor
        if actor is None:
            return None
        poly = actor.GetMapper().GetInput()
        cached = getattr(self,"_navigation_mass_cache",None)
        key = (poly,poly.GetMTime())
        if cached is None or cached[:2] != key:
            measured = None
            source_cache = self.source_measurement_cache
            if actor is self.source_actor and source_cache is not None:
                measured = source_cache[2].get("center_of_mass")
            if measured is None:
                points,faces = polydata_arrays(poly)
                measured = source_geometry_measurements(points,faces)["center_of_mass"]
            cached = (poly,poly.GetMTime(),np.asarray(measured,dtype=float))
            self._navigation_mass_cache = cached
        matrix = actor.GetMatrix()
        value = np.r_[cached[2],1.]
        transformed = np.array([[matrix.GetElement(i,j) for j in range(4)] for i in range(4)]) @ value
        return transformed[:3]

    def _keyboard_rotate_about_object(self, axis, degrees: float) -> None:
        camera = self.renderer.GetActiveCamera()
        pivot = self._keyboard_object_pivot()
        if pivot is None:
            pivot = np.asarray(camera.GetFocalPoint(), dtype=float)
        axis = np.asarray(axis, dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        angle = math.radians(degrees)
        def rotate(vector):
            return (vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle)
                    + axis * float(axis @ vector) * (1 - math.cos(angle)))
        # Rotate the entire camera frame so panning and zoom remain intact.
        camera.SetPosition(*(pivot + rotate(np.asarray(camera.GetPosition()) - pivot)))
        camera.SetFocalPoint(*(pivot + rotate(np.asarray(camera.GetFocalPoint()) - pivot)))
        camera.SetViewUp(*rotate(np.asarray(camera.GetViewUp())))
        camera.OrthogonalizeViewUp()

    def _keyboard_zoom_at_view_center(self, factor: float) -> None:
        """Zoom by a fixed camera-distance ratio without VTK Dolly saturation."""
        lathe = getattr(self, "lathe", None)
        if lathe is not None:
            lathe.mark_manual_zoom()
        camera = self.renderer.GetActiveCamera()
        pivot = self._keyboard_view_pivot()
        if pivot is None:
            pivot = np.asarray(camera.GetFocalPoint(), dtype=float)
        position = np.asarray(camera.GetPosition(), dtype=float)
        offset = position - pivot
        distance = float(np.linalg.norm(offset))
        if distance <= 1e-12:
            return
        bounds = np.asarray(self.renderer.ComputeVisiblePropBounds(), dtype=float)
        diagonal = (
            float(np.linalg.norm(bounds[[1, 3, 5]] - bounds[[0, 2, 4]]))
            if bounds.shape == (6,) and np.all(np.isfinite(bounds))
            else 0.0
        )
        minimum_distance = max(diagonal * 1e-6, 1e-7)
        if camera.GetParallelProjection():
            camera.SetParallelScale(max(minimum_distance, camera.GetParallelScale() / factor))
            camera.SetFocalPoint(*pivot)
            return
        next_distance = max(minimum_distance, distance / factor)
        camera.SetPosition(*(pivot + offset * (next_distance / distance)))
        camera.SetFocalPoint(*pivot)

    def _queue_keyboard_navigation(self, key: str, control: bool, shift: bool) -> bool:
        if key not in {"left", "right", "up", "down"} or self.busy:
            return False
        command = (key, control, shift)
        if command != self._navigation_command or not self.navigation_timer.isActive():
            self._navigation_steps = 0
            self._navigation_hold_started = time.monotonic()
        self._navigation_command = command
        # Real held keys use elapsed time, independent of OS repeat cadence.
        # Direct commands retain a short four-frame animation.
        self._navigation_steps = 4
        if not self.navigation_timer.isActive():
            self._navigation_last_tick = time.monotonic() - .016
            rotation = not control or (shift and key in ("left", "right"))
            if self._held_navigation_key is not None and rotation:
                self._keyboard_navigate(*command, fraction=.2)
                self._navigation_precision_steps = 1
                self._navigation_last_tick = time.monotonic()
            else:
                self._advance_navigation()
            if self._held_navigation_key is not None or self._navigation_steps:
                self.navigation_timer.start()
        return True

    def _stop_keyboard_navigation(self) -> None:
        self.navigation_timer.stop()
        self._navigation_steps = 0
        self._held_navigation_key = None
        self._navigation_last_tick = None
        self._navigation_hold_started = None
        self._navigation_precision_steps = 0

    def _advance_navigation(self) -> None:
        held = self._held_navigation_key is not None
        if (not held and self._navigation_steps <= 0) or self.busy or getattr(self, "_closing", False):
            self._stop_keyboard_navigation()
            return
        now = time.monotonic()
        elapsed = min(.05, max(.001, now - (self._navigation_last_tick or now)))
        self._navigation_last_tick = now
        if not held:
            self._navigation_steps -= 1
        key, control, shift = self._navigation_command
        rotation = not control or (shift and key in ("left", "right"))
        fraction = elapsed * 16 if held else .25
        if rotation:
            if held:
                started = self._navigation_hold_started
                duration = max(0., now - started) if started is not None else 0.
                if duration < .60:
                    # A quick press has no trailing motion. Initial held repeats
                    # are individual one-degree steps, independent of OS repeat.
                    target_steps = 1 + int(duration >= .30) + int(duration >= .45)
                    if self._navigation_precision_steps >= target_steps:
                        return
                    self._navigation_precision_steps = target_steps
                    fraction = .2
                else:
                    ramp = min(1., max(0., (duration - .60) / .50))
                    fraction *= .2 + .8 * ramp * ramp * (3. - 2. * ramp)
            else:
                fraction *= .2
        elif control:
            # Fine initial panning and zoom, reaching the existing speed on hold.
            started = self._navigation_hold_started
            duration = max(0., now - started) if held and started is not None else 0.
            ramp = min(1., duration / .8)
            fraction *= .12 + .88 * ramp * ramp * (3. - 2. * ramp)
        self._keyboard_navigate(*self._navigation_command, fraction=fraction)
        if not held and not self._navigation_steps:
            self.navigation_timer.stop()

    def _keyboard_navigate(self, key: str, control: bool, shift: bool, fraction: float = 1.) -> bool:
        if key not in {"left", "right", "up", "down"}:
            return False
        self._preserve_crop_outline_for_navigation()
        if key in ("left", "right") and self.keyboard_horizontal:
            key = {"left": "right", "right": "left"}[key]
        elif key in ("up", "down"):
            invert_axis = self.keyboard_zoom if control and shift else self.keyboard_vertical
            if invert_axis:
                key = {"up": "down", "down": "up"}[key]
        camera = self.renderer.GetActiveCamera()
        if control and shift and key in ("up", "down"):
            direction = fraction if key == "up" else -fraction
            lathe = getattr(self, "lathe", None)
            if lathe is None or not lathe.move_axis_depth(direction):
                self._keyboard_zoom_at_view_center(1.28 ** direction)
        elif control and shift and key == "left":
            self._keyboard_rotate_about_object(camera.GetDirectionOfProjection(), 5.0 * fraction)
        elif control and shift and key == "right":
            self._keyboard_rotate_about_object(camera.GetDirectionOfProjection(), -5.0 * fraction)
        elif key in ("left", "right", "up", "down"):
            if control:
                position = np.asarray(camera.GetPosition(), dtype=float)
                focal = np.asarray(camera.GetFocalPoint(), dtype=float)
                view_up = np.asarray(camera.GetViewUp(), dtype=float)
                forward = focal - position
                distance = float(np.linalg.norm(forward))
                if distance <= 0:
                    return True
                forward /= distance
                view_up /= max(float(np.linalg.norm(view_up)), 1e-12)
                right = np.cross(forward, view_up)
                right /= max(float(np.linalg.norm(right)), 1e-12)
                amount = distance * 0.035 * fraction
                if key == "left":
                    offset = right * amount
                elif key == "right":
                    offset = -right * amount
                elif key == "up":
                    offset = -view_up * amount
                else:
                    offset = view_up * amount
                camera.SetPosition(*(position + offset))
                camera.SetFocalPoint(*(focal + offset))
            else:
                up = np.asarray(camera.GetViewUp(), dtype=float)
                forward = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
                if key in ("left", "right"):
                    lathe = getattr(self, "lathe", None)
                    frame = lathe.navigation_frame() if lathe is not None else None
                    axis = frame[1] if frame is not None else up
                    degrees = 5.0 if key == "left" else -5.0
                else:
                    axis = -np.cross(forward, up)
                    degrees = -5.0 if key == "up" else 5.0
                self._keyboard_rotate_about_object(axis, degrees * fraction)
        else:
            return False
        # Keyboard navigation bypasses the interactor's mouse-update path. Mark the camera and
        # refresh the compact X/Y/Z indicator immediately instead of waiting for VTK to emit a
        # later render-side notification.
        camera.Modified()
        lathe = getattr(self, "lathe", None)
        if lathe is not None:
            # Apply the camera/axis change in this frame. A deferred second
            # render otherwise competes with the next navigation timer tick.
            lathe.apply_manual_camera(render=False)
        self._update_orientation_indicator()
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
        return True

    @staticmethod
    def _keyboard_camera_signature(camera) -> tuple:
        return (tuple(camera.GetPosition()), tuple(camera.GetFocalPoint()),
                tuple(camera.GetViewUp()), camera.GetParallelScale(),
                camera.GetParallelProjection(), camera.GetViewAngle())

    def _shutdown_background_workers(self) -> None:
        if getattr(self, "_background_workers_stopped", False):
            return
        self._closing = True
        if hasattr(self, "lathe"):
            self.lathe.close()
        self.metrics_timer.stop()
        self.solidify_info_timer.stop()
        self.orientation_mesh_timer.stop()
        split_widget = getattr(self, "_imprint_split_widget", None)
        if split_widget is not None:
            split_widget.SetEnabled(False)
        self.update_prompt_timer.stop()
        if self.operation_cancel_event is not None:
            self.operation_cancel_event.set()
        if self.operation_future is not None:
            self.operation_future.cancel()
        self.navigation_timer.stop()
        self.navigation_style._wheel_timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.update_executor.shutdown(wait=False, cancel_futures=True)
        self.optimization_executor.terminate_and_join()
        self._background_workers_stopped = True

    def closeEvent(self, event) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
            if hasattr(self, "tooltip_filter"):
                app.removeEventFilter(self.tooltip_filter)
        self._shutdown_background_workers()
        self.gpu_counters.close()
        self.cuda_counters.close()
        self.vtk_widget.Finalize()
        super().closeEvent(event)

    def _open_stl_association_settings(self) -> None:
        if platform.system().lower() != "windows":
            return
        url = "ms-settings:defaultapps"
        if getattr(sys, "frozen", False):
            import winreg
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\RegisteredApplications") as key:
                    winreg.QueryValueEx(key, APP_NAME)
                url += "?registeredAppUser=" + APP_NAME
            except OSError:
                pass
        if not QDesktopServices.openUrl(QUrl(url)):
            QMessageBox.warning(self, APP_NAME,
                                "Open Windows Settings > Apps > Default apps and search for .stl.")

    def _set_recent_files_enabled(self, enabled: bool) -> None:
        self.recent_files_enabled = enabled
        self.settings.setValue("files/recent_enabled", enabled)
        self.recent_files_button.setVisible(enabled)
        if not enabled:
            self._clear_recent_files()

    def _clear_recent_files(self) -> None:
        self.recent_files = []
        self.settings.remove("files/recent")
        self.recent_files_menu.clear()

    def _remember_recent_file(self, path: Path) -> None:
        if not self.recent_files_enabled:
            return
        value = os.fspath(path.resolve())
        self.recent_files = [value] + [item for item in self.recent_files
                                      if os.path.normcase(item) != os.path.normcase(value)]
        self.recent_files = self.recent_files[:10]
        self.settings.setValue("files/recent", self.recent_files)

    def _rebuild_recent_files_menu(self) -> None:
        self.recent_files_menu.clear()
        if not self.recent_files:
            self.recent_files_menu.addAction("No recent files").setEnabled(False)
            return
        for value in self.recent_files:
            path = Path(value)
            label = self.fontMetrics().elidedText(value, Qt.TextElideMode.ElideMiddle, 420)
            action = self.recent_files_menu.addAction(label.replace("&", "&&"))
            action.setToolTip(value)
            action.setEnabled(path.is_file())
            action.triggered.connect(lambda _checked=False, p=path: self.load_path(p))
        self.recent_files_menu.addSeparator()
        self.recent_files_menu.addAction("Clear recent files", self._clear_recent_files)

    def _show_recent_files(self) -> None:
        if not self.recent_files_enabled or self.busy:
            return
        self._rebuild_recent_files_menu()
        self.recent_files_menu.popup(self.recent_files_button.mapToGlobal(
            QPoint(0, self.recent_files_button.height())))

    def _file_open_directory(self) -> str:
        mode = self.settings.value("files/open_mode", "last")
        last = str(self.settings.value("files/last_open_folder", ""))
        if not last and self.source_path is not None:
            last = os.fspath(self.source_path.parent)
        chosen = (str(self.settings.value("files/open_folder", "")) if mode == "fixed"
                  else os.fspath(Path(last).parent) if mode == "smart" and last else last)
        folder = Path(chosen) if chosen else Path.home() / "Documents"
        while not folder.is_dir() and folder != folder.parent:
            folder = folder.parent
        return os.fspath(folder)

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open", self._file_open_directory(),
            "Mesh files (*.stl *.dae *.glb *.obj *.ply);;STL (*.stl);;COLLADA (*.dae);;GLB (*.glb);;OBJ (*.obj);;PLY (*.ply)")
        if path:
            self.load_path(Path(path))

    def _open_source_folder(self) -> None:
        if self.source_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.fspath(self.source_path.parent)))

    def _copy_source_path(self) -> None:
        if self.source_path is None:
            return
        QApplication.clipboard().setText(os.fspath(self.source_path))
        self.status.setText("Mesh path copied")

    def load_path(self, path: Path, *, restore_opening_view: bool = False) -> None:
        if self.busy:
            self.status.setText("Finish the current operation before loading another mesh.")
            return
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Could not load the mesh:\n\n{exc}")
            self.status.setText("Load failed.")
            return

        self._reload_opening_camera = (
            getattr(self, "_original_opening_camera", None) if restore_opening_view else None
        )
        self._probe_statistics_job = None
        self._capture_operation_snapshot()
        self._cancel_crop(None)
        self.timer.stop()
        self.load_generation += 1
        token = self.load_generation
        self.loading = True
        self.busy = True
        self.active_operation = "Loading"
        self._mark_geometry_activity("Loading geometry", 2.0)
        self.load_started_at = time.monotonic()
        self._start_operation_runtime()
        self._set_controls_busy(True)
        self._reset_operation_steps("Loading")
        self.loading_heading.setText("Preparing mesh")
        self.loading_file_label.setText(f"{path.name}  •  {fmt_bytes(file_size)}")
        self.loading_detail_label.setText("Reading triangles and welding vertices")
        self.loading_elapsed_label.setText("Starting…")
        self.loading_frame.show()
        self._position_loading_panel()
        self.loading_frame.raise_()
        self.loading_timer.start()
        self.status.setText(f"Loading {path.name}…")
        self.file_label.set_full_text(f"Loading: {path}")
        self.open_file_folder_button.setEnabled(False)
        self.copy_file_path_button.setEnabled(False)

        memory_budget = round(
            available_physical_memory() * self.large_mesh_memory_percent / 100.0
        )
        overview_triangle_limit = automatic_overview_triangle_limit(
            memory_budget, self.vram_capacity_bytes
        )
        if gpu_backend_supported(self.gpu_preference):
            self.gpu_warmup_future = self.optimization_executor.submit(
                prepare_gpu_backend,
                self.gpu_preference,
            )
            self.operation_uses_process = True
        progress_path = self._start_operation_progress(token, "Reading mesh")

        def load_progress(stage: str, completed: int, total: int) -> None:
            labels = {
                "bounds": "Measuring mesh bounds",
                "planning": "Planning spatial tiles",
                "partition": "Writing spatial tiles",
                "complete": "Spatial index ready",
            }
            _write_process_progress(
                progress_path, labels.get(stage, stage), completed, total
            )

        future = self.executor.submit(
            load_stl_smart,
            path,
            self.large_mesh_mode,
            memory_budget,
            overview_triangle_limit,
            self.operation_cancel_event,
            self.geometry_processes or None,
            load_progress,
            self.algorithm.currentText(),
            self.auto_tiling_enabled,
        )
        self.operation_future = future

        def completed(result_future) -> None:
            try:
                poly, points, faces, metadata = result_future.result()
                self.load_geometry_ready = True
                if self.gpu_warmup_future is not None and self.gpu_warmup_future.done():
                    metadata = dict(metadata or {})
                    try:
                        metadata["gpu_setup"] = self.gpu_warmup_future.result()
                    except Exception as gpu_error:  # noqa: BLE001 - CPU methods remain usable
                        metadata["gpu_setup"] = {
                            "backend": "unavailable",
                            "reason": type(gpu_error).__name__,
                        }
                error = None
            except Exception as exc:  # noqa: BLE001
                poly = points = faces = None
                metadata = None
                error = exc
            self.load_bridge.done.emit(token, path, poly, points, faces, metadata, error)

        future.add_done_callback(completed)

    def _finish_load(
        self,
        token: int,
        path: Path,
        poly: vtkPolyData | None,
        points: np.ndarray | None,
        faces: np.ndarray | None,
        metadata: dict | None,
        error: Exception | None,
    ) -> None:
        if token != self.load_generation:
            return
        self.loading = False
        self.load_geometry_ready = False
        self._end_mesh_operation()
        if error is not None or poly is None or points is None or faces is None:
            QMessageBox.critical(self, APP_NAME, f"Could not load the STL:\n\n{error}")
            self.status.setText("Load failed")
            return

        self.source_path = path
        self._remember_recent_file(path)
        self.settings.setValue("files/last_open_folder", os.fspath(path.resolve().parent))
        self.source_poly = poly
        self.source_points = points
        self.source_faces = faces
        from geometry_workload import record
        record("CPU", points, faces)
        self.source_measurement_cache = (
            (points, faces, metadata["measurements"])
            if metadata and metadata.get("measurements") is not None else None
        )
        self.overview_mode = bool(metadata and metadata.get("overview"))
        self.tiled_mode = bool(metadata and metadata.get("tiled_edit"))
        self.tiled_index_directory = (
            Path(metadata["index_directory"])
            if self.tiled_mode and metadata and metadata.get("index_directory")
            else None
        )
        self.overview_total_triangles = int(
            metadata.get("total_triangles", len(faces)) if metadata else len(faces)
        )
        self.preview_poly = None
        self.preview_is_selection = False
        self.density_cache.clear()
        self.resident_vertex_buffers.clear()
        self.measure_points.clear()
        self.measure_hover_world = None
        if self.measure_info_frame is not None:
            self.measure_info_frame.hide()
        self.mesh_modified = False
        self._solidification_applied = False
        self.saved_standard_views.clear()
        self._opening_front_pending = True
        self.orientation_reference_basis = None
        self.export_calibration_transform = None
        self.has_committed_optimization = False
        self.optimization_script_passes.clear()
        self.script_base_poly = None
        self.optimization_script_omissions.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.undo_bytes = 0
        self.redo_bytes = 0
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self._reset_previous_view()
        self.selected_face_mask = None
        self.selection_undo_stack.clear()
        self.crop_info_user_positioned = False
        self.delete_selection_button.setEnabled(False)
        self.clear_selection_button.setEnabled(False)
        self.source_actor = None
        self.simplified_actor = None
        self.file_label.set_full_text(os.fspath(path))
        self.open_file_folder_button.setEnabled(True)
        self.copy_file_path_button.setEnabled(True)
        self.loaded_triangle_count = self.overview_total_triangles
        self._update_save_button_label()
        self.original_count.setText(
            f"{fmt_count(len(faces))} overview / {fmt_count(self.overview_total_triangles)} total"
            if self.overview_mode
            else fmt_count(len(faces))
        )
        self.original_vertices.setText(
            f"{fmt_count(len(points))} overview" if self.overview_mode else fmt_count(len(points))
        )
        self.preview_count.setText("—")
        self.preview_vertices.setText("—")
        self.original_file_size.setText(fmt_bytes(path.stat().st_size))
        self._update_dimension_labels()
        self.drift.setText("—")
        self.optimization_ratio.setText("—")
        self.dimension_drift_model_units = None
        self._update_quality_metrics(None)
        with QSignalBlocker(self.target):
            self.target.setRange(1_000, max(1_000, self.overview_total_triangles))
        self._update_dynamic_presets()
        initial = smart_target(points, faces, self.smart_quality.currentText(),
                               self._cached_source_measurements(points, faces))
        if self.tiled_mode:
            initial = round(initial / max(1, len(faces)) * self.overview_total_triangles)
        with QSignalBlocker(self.target):
            self.target.setValue(initial)
        self._sync_slider_from_target()
        self.preview_dirty = not self.overview_mode or self.tiled_mode
        self._set_primary_action(False, not self.overview_mode or self.tiled_mode)
        self.toggle_mesh_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.reload_button.setEnabled(False)
        self._sync_modified_mesh_actions()
        self.showing_original = True
        self._update_estimated_size()
        self._update_processing_plan()
        self._show_poly(poly, prepared=metadata.get("render_poly") if metadata else None)
        QTimer.singleShot(500, self._precache_source_analysis)
        self.timer.stop()
        self.status.setText(
            f"Overview • {fmt_count(len(faces))} sampled of "
            f"{fmt_count(self.overview_total_triangles)} triangles • editing disabled"
            if self.overview_mode
            else f"Loaded • {fmt_count(len(faces))} triangles" +
                 (" • Geometry only" if path.suffix.lower() != '.stl' else "")
        )
        if self.verification_screenshot is not None:
            self.target.setValue(getattr(self, "verification_target", 200_000))
            self.timer.stop()
            self.preview()

    def _crop_interaction(self, phase: str, vtk_position) -> bool:
        if phase == "navigation":
            return self._preserve_crop_outline_for_navigation()
        if self.overview_mode:
            if phase == "start":
                self.status.setText("Overview mode is for navigation. Load the full mesh to edit.")
            return False
        if phase == "confirm" and self.crop_candidate is not None:
            self._confirm_crop()
            return True
        if phase == "cancel" and (
            self.crop_candidate is not None
            or self.crop_points
            or self.crop_selection_actor is not None
        ):
            self._cancel_crop("Crop cancelled.")
            return True
        if self.source_poly is None or self.busy or vtk_position is None:
            return False
        x, vtk_y = vtk_position
        qt_y = self.vtk_widget.height() - vtk_y
        point = QPoint(x, qt_y)
        if phase == "start":
            if self.crop_candidate is not None:
                self.selected_face_mask = self.crop_candidate[3].copy()
            if self.selection_base_quality is None:
                self.timer.stop()
                self.selection_base_quality = self.smart_quality.currentText()
                self.selection_base_algorithm = self.algorithm.currentText()
                self.selection_base_target = self.target.value()
                self.selection_target = None
            self.selection_undo_stack.append(
                self.selected_face_mask.copy() if self.selected_face_mask is not None else None
            )
            self._hide_crop_preview()
            self._hide_loupe()
            self._show_poly(self.source_poly, reset_camera=False)
            self._refresh_persistent_selection()
            self.crop_start = point
            self.crop_points = [point, point, point, point]
            self._update_crop_overlay()
        elif phase == "move" and self.crop_start is not None:
            rect = QRect(self.crop_start, point).normalized()
            self.crop_points = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
            self._update_crop_overlay()
        elif phase == "finish" and self.crop_start is not None:
            start = self.crop_start
            self.crop_start = None
            rect = QRect(start, point).normalized()
            if rect.width() < 8 or rect.height() < 8:
                self.crop_points = []
                self.crop_world_points = []
                self._update_crop_overlay()
                if self.selection_undo_stack:
                    self.selection_undo_stack.pop()
                if self.crop_candidate is not None:
                    self._present_selection(*self.crop_candidate)
                return True
            self.crop_points = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
            self._update_crop_overlay()
            self._prepare_polygon_crop()
        elif phase == "left_press" and self.crop_candidate is not None:
            vertex = self._nearest_crop_vertex(point, 12.0)
            if vertex is not None:
                self.crop_drag_index = vertex
                return True
            segment, distance = self._nearest_crop_segment(point)
            if segment is not None and distance <= 10.0:
                insert_at = segment + 1
                self.crop_points.insert(insert_at, point)
                self.crop_drag_index = insert_at
                self._update_crop_overlay()
                return True
            if QPolygonF([QPointF(p) for p in self.crop_points]).containsPoint(
                QPointF(point), Qt.FillRule.OddEvenFill
            ):
                return True
            self._preserve_crop_outline_for_navigation()
        elif phase == "edit_move" and self.crop_drag_index is not None:
            self.crop_points[self.crop_drag_index] = point
            self._update_crop_overlay()
            return True
        elif phase == "left_release" and self.crop_drag_index is not None:
            self.crop_points[self.crop_drag_index] = point
            self.crop_drag_index = None
            self._update_crop_overlay()
            self._prepare_polygon_crop()
            return True
        elif phase == "right_press" and self.crop_candidate is not None:
            segment, distance = self._nearest_crop_segment(point)
            if segment is not None and distance <= 12.0 and len(self.crop_points) > 3:
                next_index = (segment + 1) % len(self.crop_points)
                a = self.crop_points[segment]
                b = self.crop_points[next_index]
                remove_index = segment if (point - a).manhattanLength() < (point - b).manhattanLength() else next_index
                self.crop_points.pop(remove_index)
                self._update_crop_overlay()
                self._prepare_polygon_crop()
                return True
        return False

    def _preserve_crop_outline_for_navigation(self) -> bool:
        """Discard the view-specific outline while retaining selected mesh faces."""
        if self.crop_candidate is None or not self.crop_points:
            return False
        self.crop_points = []
        self.crop_world_points = []
        self.crop_drag_index = None
        self._update_crop_overlay(render=False)
        self.vtk_widget.update()
        return True

    def _anchor_crop_outline(self) -> None:
        if not self.crop_points:
            self.crop_world_points = []
            return
        points = self.crop_candidate[1] if self.crop_candidate is not None else None
        center = (points.mean(axis=0) if points is not None and len(points)
                  else self.renderer.GetActiveCamera().GetFocalPoint())
        self.renderer.SetWorldPoint(*center, 1.0)
        self.renderer.WorldToDisplay()
        depth = self.renderer.GetDisplayPoint()[2]
        anchors = []
        for point in self.crop_points:
            self.renderer.SetDisplayPoint(point.x(), self.vtk_widget.height() - point.y(), depth)
            self.renderer.DisplayToWorld()
            world = np.asarray(self.renderer.GetWorldPoint(), dtype=float)
            if not np.all(np.isfinite(world)) or abs(world[3]) < 1e-12:
                return
            anchors.append(world[:3] / world[3])
        self.crop_world_points = anchors

    def _queue_crop_outline_projection(self, *_args) -> None:
        if self.crop_world_points and not self._crop_projection_pending and not getattr(self, "_closing", False):
            self._crop_projection_pending = True
            def refresh():
                try:
                    self._preserve_crop_outline_for_navigation()
                finally:
                    self._crop_projection_pending = False
            QTimer.singleShot(0, refresh)

    def _project_crop_outline(self) -> None:
        if (getattr(self, "_closing", False) or not self.crop_world_points
                or self.crop_drag_index is not None or self.crop_start is not None):
            return
        projected = []
        for world in self.crop_world_points:
            display = self._world_to_display(world)
            if display is None:
                return
            projected.append(QPoint(round(display[0]), round(self.vtk_widget.height() - display[1])))
        if projected != self.crop_points:
            self.crop_points = projected
            self._update_crop_overlay(render=False)
            self.vtk_widget.update()

    def _update_crop_overlay(self, render: bool = True) -> None:
        for actor in self.crop_overlay_actors:
            self.renderer.RemoveViewProp(actor)
        self.crop_overlay_actors = []
        if len(self.crop_points) < 3:
            if render:
                self.vtk_widget.GetRenderWindow().Render()
            return
        vtk_points = vtkPoints()
        height = self.vtk_widget.height()
        for point in self.crop_points:
            vtk_points.InsertNextPoint(float(point.x()), float(height - point.y()), 0.0)
        coordinate = vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()

        fill_poly = vtkPolyData()
        fill_poly.SetPoints(vtk_points)
        fill_cells = vtkCellArray()
        fill_cells.InsertNextCell(len(self.crop_points))
        for index in range(len(self.crop_points)):
            fill_cells.InsertCellPoint(index)
        fill_poly.SetPolys(fill_cells)
        fill_mapper = vtkPolyDataMapper2D()
        fill_mapper.SetInputData(fill_poly)
        fill_mapper.SetTransformCoordinate(coordinate)
        fill_actor = vtkActor2D()
        fill_actor.SetMapper(fill_mapper)
        fill_actor.GetProperty().SetColor(0.15, 0.62, 1.0)
        fill_actor.GetProperty().SetOpacity(0.10)

        line_poly = vtkPolyData()
        line_poly.SetPoints(vtk_points)
        line_cells = vtkCellArray()
        line_cells.InsertNextCell(len(self.crop_points) + 1)
        for index in range(len(self.crop_points)):
            line_cells.InsertCellPoint(index)
        line_cells.InsertCellPoint(0)
        line_poly.SetLines(line_cells)
        line_mapper = vtkPolyDataMapper2D()
        line_mapper.SetInputData(line_poly)
        line_mapper.SetTransformCoordinate(coordinate)
        line_actor = vtkActor2D()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetColor(0.45, 0.88, 1.0)
        line_actor.GetProperty().SetLineWidth(3.0)

        handle_poly = vtkPolyData()
        handle_poly.SetPoints(vtk_points)
        handle_cells = vtkCellArray()
        for index in range(len(self.crop_points)):
            handle_cells.InsertNextCell(1)
            handle_cells.InsertCellPoint(index)
        handle_poly.SetVerts(handle_cells)
        handle_mapper = vtkPolyDataMapper2D()
        handle_mapper.SetInputData(handle_poly)
        handle_mapper.SetTransformCoordinate(coordinate)
        handle_actor = vtkActor2D()
        handle_actor.SetMapper(handle_mapper)
        handle_actor.GetProperty().SetColor(0.65, 0.94, 1.0)
        handle_actor.GetProperty().SetPointSize(11.0)

        self.crop_overlay_actors = [fill_actor, line_actor, handle_actor]
        for actor in self.crop_overlay_actors:
            self.renderer.AddViewProp(actor)
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def _world_to_display(self, point: np.ndarray) -> tuple[float, float] | None:
        self.renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        self.renderer.WorldToDisplay()
        display = self.renderer.GetDisplayPoint()
        if not all(math.isfinite(value) for value in display):
            return None
        return float(display[0]), float(display[1])

    def _nearest_measure_point(self, position: tuple[int, int], limit: float = 16.0) -> int | None:
        nearest, distance = None, float("inf")
        for index, world in enumerate(self.measure_points):
            display = self._world_to_display(world)
            if display is None:
                continue
            candidate = math.hypot(position[0] - display[0], position[1] - display[1])
            if candidate < distance:
                nearest, distance = index, candidate
        return nearest if distance <= limit else None

    def _pick_measure_point(self, position: tuple[int, int]) -> np.ndarray | None:
        if self.measure_picker is None or self.active_actor is None or self.busy:
            return None
        # Hardware picking follows rendered pixels. Temporarily pick the mesh as a surface so the
        # ruler remains usable in wireframe, vertices, and density displays without a CPU cell walk.
        mesh_property = self.active_actor.GetProperty()
        representation = mesh_property.GetRepresentation()
        mesh_property.SetRepresentationToSurface()
        self.measure_picker.InitializePickList()
        self.measure_picker.AddPickList(self.active_actor)
        self.measure_picker.PickFromListOn()
        try:
            picked = self.measure_picker.Pick(position[0], position[1], 0.0, self.renderer)
        finally:
            mesh_property.SetRepresentation(representation)
        if not picked:
            return None
        if self.measure_picker.GetActor() != self.active_actor:
            return None
        point = np.asarray(self.measure_picker.GetPickPosition(), dtype=float)
        return point if point.shape == (3,) and np.all(np.isfinite(point)) else None

    def _measurement_interaction(self, phase: str, position) -> bool:
        if phase == "hide":
            if self.measure_hover_world is not None or self.measure_pending_position is not None:
                self.measure_pending_position = None
                self.measure_pick_timer.stop()
                self.measure_hover_world = None
                self._update_measure_overlay()
            return False
        if phase == "refresh":
            if self.measure_points or self.measure_hover_world is not None:
                self._update_measure_overlay()
            return False
        if self.active_actor is None or self.busy:
            return False
        screen = (int(position[0]), int(position[1]))
        if phase == "move":
            self.measure_pending_position = screen
            if not self.measure_pick_timer.isActive():
                self.measure_pick_timer.start()
            return True
        if phase == "click":
            self.measure_pending_position = None
            self.measure_pick_timer.stop()
            existing = self._nearest_measure_point(screen)
            if existing is not None:
                self.measure_points.pop(existing)
                self.measure_hover_world = None
                self._update_measure_panel()
                self._update_measure_overlay()
                return True
            point = self._pick_measure_point(screen)
            if point is None:
                return False
            self.measure_points.append(point.copy())
            self.measure_hover_world = point
            self._update_measure_panel()
            self._update_measure_overlay()
            return True
        return False

    def _update_measure_hover(self) -> None:
        position = self.measure_pending_position
        self.measure_pending_position = None
        if position is None:
            return
        self.measure_hover_world = self._pick_measure_point(position)
        self._update_measure_overlay()

    def _clear_measurements(self) -> None:
        self.measure_points.clear()
        self.measure_hover_world = None
        self.measure_pending_position = None
        self.measure_pick_timer.stop()
        self._update_measure_panel()
        self._update_measure_overlay()

    def _clear_selection_and_ruler(self) -> None:
        had_selection = bool(self._selected_triangle_count() or self.crop_points)
        had_ruler = bool(self.measure_points) or self.measure_hover_world is not None
        if had_selection:
            self._clear_selection(message=None)
        if had_ruler:
            self._clear_measurements()
        if had_selection and had_ruler:
            self.status.setText("Selection and ruler cleared")
        elif had_selection:
            self.status.setText("Selection cleared")
        elif had_ruler:
            self.status.setText("Ruler cleared")

    def _ensure_measure_panel(self) -> None:
        if self.measure_info_frame is not None:
            return
        self.measure_info_frame = QFrame(self.vtk_widget)
        self.measure_info_frame.setStyleSheet(
            "QFrame { color: #0B1B34; background: rgba(251, 250, 248, 245); "
            "border: 1px solid #D8DCE5; border-radius: 7px; } "
            "QLabel { color: #0B1B34; border: none; background: transparent; font-size: 12px; } "
            "QComboBox, QPushButton { padding: 5px 8px; background: #FFFFFF; color: #0B1B34; "
            "border: 1px solid #D8DCE5; border-radius: 5px; } "
            "QComboBox:hover, QPushButton:hover { background: #F1F5FF; border-color: #2563FF; }"
        )
        layout = QVBoxLayout(self.measure_info_frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.measure_info_label = QLabel(self.measure_info_frame)
        layout.addWidget(self.measure_info_label)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Units", self.measure_info_frame))
        self.measure_units = ArrowComboBox(self.measure_info_frame)
        self.measure_units.addItems(list(UNIT_MM))
        preferred = self.units.currentText() if hasattr(self, "units") else "mm"
        self.measure_units.setCurrentText(preferred)
        self.measure_units.currentTextChanged.connect(self._measure_units_changed)
        self.measure_units.setToolTip("Choose the units used for ruler distances.")
        controls.addWidget(self.measure_units)
        clear_button = QPushButton("Clear", self.measure_info_frame)
        clear_button.clicked.connect(self._clear_measurements)
        clear_button.setToolTip(
            f"Remove all ruler points. Shortcut: {self.shortcuts['selection_clear']}."
        )
        controls.addWidget(clear_button)
        layout.addLayout(controls)
        self._refresh_theme_overlays(appearance_theme(self.appearance_theme_name), self._viewport_theme())

    def _measure_units_changed(self, _unit: str) -> None:
        self.measure_units_overridden = True
        self._update_measure_panel()

    def _update_measure_panel(self) -> None:
        if not self.measure_points:
            if self.measure_info_frame is not None:
                self.measure_info_frame.hide()
            return
        self._ensure_measure_panel()
        unit = self.measure_units.currentText() if self.measure_units is not None else "mm"
        segments = [
            float(np.linalg.norm(second - first)) / UNIT_MM[unit]
            for first, second in zip(self.measure_points, self.measure_points[1:])
        ]
        visible = segments[-8:]
        first_index = len(segments) - len(visible) + 1
        rows = ["RULER", f"Points: {len(self.measure_points)}"]
        if first_index > 1:
            rows.append(f"{first_index - 1} earlier segments")
        rows.extend(
            f"{index} to {index + 1}: {length:,.3f} {unit}"
            for index, length in enumerate(visible, start=first_index)
        )
        rows.append(f"Total: {sum(segments):,.3f} {unit}")
        self.measure_info_frame.setToolTip("Shift+click adds or removes a ruler point.")
        self.measure_info_label.setText("\n".join(rows))
        self.measure_info_frame.adjustSize()
        self.measure_info_frame.show()
        self._position_viewport_overlays()
        self.measure_info_frame.raise_()

    def _update_measure_overlay(self, render: bool = True) -> None:
        for overlay_actor in self.measure_overlay_actors:
            self.renderer.RemoveViewProp(overlay_actor)
        self.measure_overlay_actors = []
        displays = [self._world_to_display(point) for point in self.measure_points]
        displays = [point for point in displays if point is not None]
        hover = self._world_to_display(self.measure_hover_world) if self.measure_hover_world is not None else None
        if not displays and hover is None:
            if render:
                self.vtk_widget.GetRenderWindow().Render()
            return
        vtk_points = vtkPoints()
        lines = vtkCellArray()

        def add_segment(a: tuple[float, float], b: tuple[float, float]) -> None:
            first = vtk_points.InsertNextPoint(a[0], a[1], 0.0)
            second = vtk_points.InsertNextPoint(b[0], b[1], 0.0)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(first)
            lines.InsertCellPoint(second)

        for first, second in pairwise(displays):
            add_segment(first, second)
        for center, radius in [(point, 10.0) for point in displays] + ([(hover, 14.0)] if hover else []):
            ring = [
                (center[0] + radius * math.cos(index * math.tau / 24.0),
                 center[1] + radius * math.sin(index * math.tau / 24.0))
                for index in range(24)
            ]
            for first, second in zip(ring, ring[1:] + ring[:1]):
                add_segment(first, second)
        poly = vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetLines(lines)
        coordinate = vtkCoordinate()
        coordinate.SetCoordinateSystemToDisplay()
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        mapper.SetTransformCoordinate(coordinate)
        outline = vtkActor2D()
        outline.SetMapper(mapper)
        outline.GetProperty().SetColor(0.02, 0.07, 0.13)
        outline.GetProperty().SetLineWidth(7.0)
        foreground = vtkActor2D()
        foreground.SetMapper(mapper)
        foreground.GetProperty().SetColor(0.72, 0.96, 0.22)
        foreground.GetProperty().SetLineWidth(4.0)
        self.measure_overlay_actors = [outline, foreground]
        for overlay_actor in self.measure_overlay_actors:
            self.renderer.AddViewProp(overlay_actor)
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def _nearest_crop_vertex(self, point: QPoint, limit: float) -> int | None:
        if not self.crop_points:
            return None
        distances = [math.hypot(point.x() - p.x(), point.y() - p.y()) for p in self.crop_points]
        index = int(np.argmin(distances))
        return index if distances[index] <= limit else None

    def _nearest_crop_segment(self, point: QPoint) -> tuple[int | None, float]:
        best_index, best_distance = None, float("inf")
        p = np.asarray((point.x(), point.y()), dtype=float)
        for index, first in enumerate(self.crop_points):
            second = self.crop_points[(index + 1) % len(self.crop_points)]
            a = np.asarray((first.x(), first.y()), dtype=float)
            b = np.asarray((second.x(), second.y()), dtype=float)
            ab = b - a
            length_squared = float(ab @ ab)
            amount = 0.0 if length_squared == 0 else max(0.0, min(1.0, float((p - a) @ ab) / length_squared))
            distance = float(np.linalg.norm(p - (a + amount * ab)))
            if distance < best_distance:
                best_index, best_distance = index, distance
        return best_index, best_distance

    @staticmethod
    def _points_in_polygon(x: np.ndarray, y: np.ndarray, polygon: list[QPoint]) -> np.ndarray:
        inside = np.zeros(len(x), dtype=bool)
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = previous.x(), previous.y()
            x2, y2 = current.x(), current.y()
            crossing = ((y1 > y) != (y2 > y)) & (
                x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            )
            inside ^= crossing
            previous = current
        return inside

    def _select_connected_at(self, position) -> None:
        if self.busy or self.overview_mode or self.source_poly is None:
            return
        world = self._pick_measure_point(position)
        if world is None:
            return
        self.timer.stop()
        self.crop_start = None
        self.navigation_style._crop_down = False
        self.navigation_style._consume_left_release = True
        self.vtk_widget.setCursor(Qt.CursorShape.ArrowCursor)
        source_points, source_faces = self.source_points, self.source_faces
        previous = (self.crop_candidate[3].copy() if self.crop_candidate is not None
                    else self.selected_face_mask.copy() if self.selected_face_mask is not None
                    else np.zeros(len(source_faces), dtype=bool))
        render_width, render_height = self.vtk_widget.GetRenderWindow().GetSize()
        self._connected_selection_anchor = QPoint(
            round(position[0]*self.vtk_widget.width()/max(1, render_width)),
            round(self.vtk_widget.height()-position[1]*self.vtk_widget.height()/max(1, render_height)))
        token = self._begin_mesh_operation("Selecting connected geometry", "Local")

        def job():
            try:
                poly = arrays_polydata(source_points, source_faces)
                ids = numpy_to_vtk(np.arange(len(source_faces), dtype=np.int64), deep=True)
                ids.SetName("selection_face_id")
                poly.GetCellData().AddArray(ids)
                connectivity = vtkPolyDataConnectivityFilter()
                connectivity.SetInputData(poly)
                connectivity.SetExtractionModeToClosestPointRegion()
                connectivity.SetClosestPoint(*world)
                connectivity.Update()
                face_ids = vtk_to_numpy(connectivity.GetOutput().GetCellData().GetArray("selection_face_id"))
                mask = np.zeros(len(source_faces), dtype=bool)
                mask[face_ids] = True
                mask = previous & ~mask if np.all(previous[mask]) else previous | mask
                used, inverse = np.unique(source_faces[mask], return_inverse=True)
                points = np.ascontiguousarray(source_points[used])
                faces = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32)
                return {"kind": "connected-selection", "previous": previous, "candidate":
                        (arrays_polydata(points, faces), points, faces, mask)}
            except Exception:  # noqa: BLE001 - worker boundary returns the failure to the GUI
                return {"error": traceback.format_exc()}

        future = self.executor.submit(job)
        self.operation_future = future
        future.add_done_callback(lambda f: self.edit_bridge.done.emit(token, f.result()))

    def _prepare_polygon_crop(self) -> None:
        if self.source_poly is None or self.source_points is None or self.source_faces is None:
            return
        self.crop_world_points = []
        self._mark_geometry_activity("Analyzing selection", 3.0)
        if self.crop_info_frame is not None:
            self.crop_info_frame.hide()
        QApplication.processEvents()
        self.vtk_widget.GetRenderWindow().Render()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            width, height = max(1, self.vtk_widget.width()), max(1, self.vtk_widget.height())
            matrix_vtk = self.renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(
                self.renderer.GetTiledAspectRatio(), -1.0, 1.0
            )
            matrix = np.asarray(
                [[matrix_vtk.GetElement(row, column) for column in range(4)] for row in range(4)],
                dtype=float,
            )
            points_h = np.column_stack((self.source_points, np.ones(len(self.source_points))))
            clip = points_h @ matrix.T
            valid = np.abs(clip[:, 3]) > 1e-12
            screen_x = np.full(len(clip), -1e9)
            screen_y = np.full(len(clip), -1e9)
            screen_x[valid] = (clip[valid, 0] / clip[valid, 3] + 1.0) * width * 0.5
            screen_y[valid] = (1.0 - clip[valid, 1] / clip[valid, 3]) * height * 0.5
            selected_vertices = valid & self._points_in_polygon(screen_x, screen_y, self.crop_points)
            selected_face_mask = np.any(selected_vertices[self.source_faces], axis=1)
            if (
                self.selected_face_mask is not None
                and len(self.selected_face_mask) == len(selected_face_mask)
            ):
                selected_face_mask |= self.selected_face_mask
            selected_faces = self.source_faces[selected_face_mask]
            used = np.unique(selected_faces)
            remap = np.full(len(self.source_points), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
            points = np.ascontiguousarray(self.source_points[used])
            faces = np.ascontiguousarray(remap[selected_faces], dtype=np.int32)
            cropped = arrays_polydata(points, faces)
        finally:
            QApplication.restoreOverrideCursor()
        self._present_selection(cropped, points, faces, selected_face_mask)

    def _present_selection(self, cropped, points, faces, selected_face_mask) -> None:
        self.crop_candidate = (cropped, points, faces, selected_face_mask)
        self._anchor_crop_outline()
        self._show_crop_highlight(cropped)
        suggested_target = smart_target(points, faces, self.smart_quality.currentText()) if len(faces) else 0
        maximum = max(4, len(faces) - 1) if len(faces) else 0
        target = min(self.selection_target or suggested_target, maximum)
        self.selection_target = target
        with QSignalBlocker(self.target):
            self.target.setRange(min(1_000, maximum), maximum)
            self.target.setValue(target)
        self._update_dynamic_presets(len(faces))
        self._sync_slider_from_target(len(faces))
        self._update_estimated_size()
        self._show_selection_primary_action()
        self.preview_button.setEnabled(bool(len(faces)) and not self.busy)
        unit = self.units.currentText()
        size = np.asarray(bounds_size(cropped.GetBounds())) / UNIT_MM[unit] if len(faces) else np.zeros(3)
        if self.crop_info_frame is None:
            self.crop_info_frame = DraggableOverlayFrame(self.vtk_widget)
            self.crop_info_frame.movedByUser.connect(self._mark_crop_info_positioned)
            self.crop_info_frame.setStyleSheet(
                "QFrame { color: #0B1B34; background: rgba(251, 250, 248, 245); "
                "border: 1px solid #D8DCE5; border-radius: 7px; } "
                "QLabel { color: #0B1B34; border: none; background: transparent; font-size: 12px; } "
                "QPushButton { padding: 5px 8px; background: #FFFFFF; color: #0B1B34; "
                "border: 1px solid #D8DCE5; border-radius: 5px; } "
                "QPushButton:hover { background: #F1F5FF; border-color: #2563FF; }"
            )
            info_layout = QVBoxLayout(self.crop_info_frame)
            info_layout.setContentsMargins(8, 8, 8, 8)
            info_layout.setSpacing(6)
            self.crop_info_label = QLabel(self.crop_info_frame)
            self.crop_info_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            info_layout.addWidget(self.crop_info_label)
            action_row = QHBoxLayout()
            action_row.setSpacing(6)
            action_specs = (
                ("Crop", "crop", f"Keep only the highlighted triangles. Shortcut: {self.shortcuts['selection_crop']}."),
                ("Delete", "delete", f"Remove the highlighted triangles. Shortcut: {self.shortcuts['selection_delete']}."),
                ("Invert", "invert", "Select every triangle that is not selected, and deselect the current selection."),
                ("Step back", "undo", "Restore the previous selection."),
                ("Clear", "clear", f"Clear the current selection without changing the mesh. Shortcut: {self.shortcuts['selection_clear']}."),
            )
            for index, (caption, action, help_text) in enumerate(action_specs):
                button = QPushButton(caption, self.crop_info_frame)
                button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                button.setProperty("selectionAction", action)
                self._set_theme_icon(button, action)
                button.clicked.connect(lambda _=False, a=action: self._apply_selection_action(a))
                button.setToolTip(help_text)
                if action == "undo":
                    self.selection_undo_button = button
                action_row.addWidget(button)
            action_row.addStretch(1)
            info_layout.addLayout(action_row)
            self._refresh_theme_overlays(appearance_theme(self.appearance_theme_name), self._viewport_theme())
        self.crop_info_frame.setToolTip("Drag corners to edit the selection. Ctrl+drag adds another region. Shift+click measures.")
        self.crop_info_label.setText(
            f"SELECTION\n"
            f"Selected: {fmt_count(len(points))} vertices, {fmt_count(len(faces))} triangles\n"
            f"Mesh share: {self._selection_size_summary(len(faces))}\n"
            f"Dimensions: {fmt_size(tuple(size))} {unit}"
        )
        for button in self.crop_info_frame.findChildren(QPushButton):
            if button.property("selectionAction") in {"crop", "add", "optimize", "delete"}:
                button.setEnabled(bool(len(faces)))
        if self.selection_undo_button is not None:
            self.selection_undo_button.setVisible(self._can_step_back_selection())
        self.crop_info_frame.adjustSize()
        self._apply_rounded_mask(self.crop_info_frame, 5)
        polygon_rect = QPolygonF([QPointF(p) for p in self.crop_points]).boundingRect().toRect()
        if not self.crop_points and hasattr(self, "_connected_selection_anchor"):
            polygon_rect = QRect(self._connected_selection_anchor, QSize(1, 1))
        label_x = min(polygon_rect.left(), max(8, self.vtk_widget.width() - self.crop_info_frame.width() - 8))
        label_y = min(polygon_rect.bottom() + 8, max(8, self.vtk_widget.height() - self.crop_info_frame.height() - 8))
        if not self.crop_info_user_positioned:
            self.crop_info_frame.move(max(8, label_x), max(8, label_y))
        self.crop_info_frame.show()
        self.crop_info_frame.raise_()
        self.status.setText("Selection active")

    def _mark_crop_info_positioned(self) -> None:
        self.crop_info_user_positioned = True

    def _show_crop_highlight(self, poly: vtkPolyData) -> None:
        if self.crop_selection_actor is not None:
            self.renderer.RemoveActor(self.crop_selection_actor)
            self.crop_selection_actor = None
        if not poly.GetNumberOfCells():
            if self.source_actor is not None:
                self.source_actor.GetProperty().SetOpacity(1.0)
            return
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-3.0, -3.0)
        actor = vtkActor()
        actor.PickableOff()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.05, 0.95, 1.0)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetAmbient(0.85)
        actor.GetProperty().SetDiffuse(0.15)
        actor.GetProperty().SetEdgeColor(1.0, 0.55, 0.05)
        actor.GetProperty().EdgeVisibilityOn()
        self.crop_selection_actor = actor
        if self.source_actor is not None:
            self.source_actor.GetProperty().SetOpacity(0.12)
        self.renderer.AddActor(actor)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _confirm_crop(self) -> None:
        self._apply_selection_action("add")

    def _selection_shortcut(self, action: str) -> None:
        if self.busy:
            return
        if self.crop_candidate is not None:
            self._apply_selection_action(action)
            return
        if action == "optimize":
            self._optimize_saved_selection()
        elif action == "delete" and self.selected_face_mask is not None:
            self._delete_selected()

    def _reorder_tool_sections(self):
        sections = getattr(self, "_toolbox_sections", {})
        if not sections:
            return
        active = next((title for title, toggle in (
            ("OPTIMIZATION", "optimization_toggle"), ("SOLIDIFY", "solidify_toggle"),
            ("CAST", "imprint_section_toggle"), ("AXIAL CALIBRATION", "alignment_toggle"))
            if getattr(self, toggle, None) is not None and getattr(self, toggle).isChecked()), None)
        if active is None and hasattr(self, "optimization_toggle"):
            self.optimization_toggle.setChecked(True)
            active = "OPTIMIZATION"
        groups = {"Actions": ["ACTIONS"], "Active tool": [active] if active else [],
                  "Stats": ["STATS"], "General": ["GENERAL"],
                  "Other tools": [title for title in ("OPTIMIZATION", "SOLIDIFY", "CAST", "AXIAL CALIBRATION") if title != active]}
        order = [title for group in self.toolbox_order for title in groups[group]]
        for index, title in enumerate(title for title in order if title in sections):
            self._toolbox_layout.insertWidget(index + 1, sections[title])
        self._queue_toolbox_scroll_refresh()

    def _queue_toolbox_scroll_refresh(self):
        if getattr(self, "_toolbox_scroll_refresh_pending", False) or getattr(self, "_closing", False):
            return
        self._toolbox_scroll_refresh_pending = True
        QTimer.singleShot(0, self._refresh_toolbox_scroll_range)

    def _refresh_toolbox_scroll_range(self):
        self._toolbox_scroll_refresh_pending = False
        if getattr(self, "_closing", False):
            return
        layout = self.controls_content.layout()
        layout.activate()
        height = (layout.totalHeightForWidth(self.controls_widget.viewport().width())
                  if layout.hasHeightForWidth() else layout.totalSizeHint().height())
        height = max(0, height, layout.minimumSize().height())
        if self.controls_content.minimumHeight() != height:
            self.controls_content.setMinimumHeight(height)
        # Apply the new extent before scrolling. Collapsible child layouts can
        # otherwise leave the scroll area using the previous tool's range.
        self.controls_content.resize(
            self.controls_widget.viewport().width(),
            max(height, self.controls_widget.viewport().height()),
        )
        if getattr(self, "_toolbox_scroll_to_active", False):
            self._toolbox_scroll_to_active = False
            self.controls_widget.verticalScrollBar().setValue(0)

    def _toggle_tool_section(self, name: str, expanded: bool) -> None:
        if getattr(self, "_switching_tool_section", False):
            return
        self._switching_tool_section = True
        try:
            self._apply_tool_section(name, expanded)
        finally:
            self._switching_tool_section = False

    def _apply_tool_section(self, name: str, expanded: bool) -> None:
        sections = (("Optimize", "optimization_toggle", "optimization_body"),
                    ("Solidify", "solidify_toggle", "solidify_body"),
                    ("Cast", "imprint_section_toggle", "imprint_body"),
                    ("Axial calibration", "alignment_toggle", "alignment_body"))
        for label, toggle_name, body_name in sections:
            toggle, body = getattr(self, toggle_name, None), getattr(self, body_name, None)
            if toggle is None or body is None:
                continue
            if label == name:
                body.setVisible(expanded)
                toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
                toggle.setAccessibleName(f"{'Collapse' if expanded else 'Expand'} {label}")
            elif expanded:
                toggle.setChecked(False)
                body.hide()
                toggle.setArrowType(Qt.ArrowType.RightArrow)
                toggle.setAccessibleName(f"Expand {label}")
        if not expanded and not any(
            getattr(self, toggle_name, None) is not None
            and getattr(self, toggle_name).isChecked()
            for _, toggle_name, _ in sections
        ):
            self.optimization_toggle.setChecked(True)
            self._apply_tool_section("Optimize", True)
            return
        self._toolbox_scroll_to_active = expanded
        self._reorder_tool_sections()
        if expanded and name == "Solidify":
            self._refresh_solidify_info()

    def _refresh_solidify_availability(self):
        if hasattr(self, "solidify_preview_button"):
            marker = (self.source_poly.GetFieldData().GetArray("ImprintClosedOutput")
                      if self.source_poly is not None else None)
            imprint_closed = (getattr(self, "_envelope_closed", False)
                              and getattr(self, "_envelope_poly", None) is not None
                              and getattr(self, "_envelope_settings_key", None) == self._imprint_preview_settings_key())
            completed = (getattr(self, "_solidification_applied", False)
                         or (marker is not None and bool(marker.GetValue(0)))
                         or imprint_closed
                         or (getattr(self, "_solidify_preview_context", None) is not None and self.pass_ready))
            measured = getattr(self, "_solidify_info_result", None)
            report = (measured[1] if measured is not None
                      and measured[0] == self._solidify_input_key() else None)
            already_closed = (report is not None and report["open_edges"] == 0
                              and report["nonmanifold_edges"] == 0)
            available = self.source_poly is not None and not completed and not already_closed
            section = getattr(self, "_toolbox_sections", {}).get("SOLIDIFY")
            if section is not None:
                section.setVisible(available)
                self._queue_toolbox_scroll_refresh()
            self.solidify_preview_button.setVisible(available)
            self.solidify_preview_button.setEnabled(available and not self.busy and report is not None)
            self.solidify_toggle.setEnabled(available and not self.busy)
            if not available:
                self.solidify_toggle.setChecked(False)
            elif not self.busy:
                self._refresh_solidify_info()

    def _toggle_imprint_section(self, expanded: bool) -> None:
        self._toggle_tool_section("Cast", expanded)
        self.imprint_section_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.imprint_section_toggle.setAccessibleName("Collapse Cast" if expanded else "Expand Cast")
        self.imprint_body.setVisible(expanded)
        if not expanded:
            self.imprint_pick_surface.setChecked(False)
        self._refresh_imprint_plan()
        self._sync_imprint_split_widget()
        if hasattr(self, "renderer"):
            self._update_envelope_overlay()

    def _toggle_imprint_surface_pick(self, picking: bool) -> None:
        if not hasattr(self, "vtk_widget"):
            return
        if picking and (self.source_poly is None or self.busy):
            self.imprint_pick_surface.setChecked(False)
            return
        self.vtk_widget.setCursor(Qt.CursorShape.CrossCursor if picking else Qt.CursorShape.ArrowCursor)
        split_widget = getattr(self, "_imprint_split_widget", None)
        if split_widget is not None:
            split_widget.SetProcessEvents(not picking and not self.busy
                                          and getattr(self, "_imprint_plane_mode", "Rotate") == "Move")
        if picking:
            self.envelope_original_visible.setChecked(True)
            self._show_poly(self.source_poly, reset_camera=False)
            self.source_actor.SetVisibility(True)
            self.vtk_widget.GetRenderWindow().Render()
        self._sync_imprint_split_widget()

    def _clear_imprint_surface_picks(self) -> None:
        self._imprint_surface_seeds = []
        self._imprint_surface_seed_source = self.source_poly
        self.imprint_pick_surface.setChecked(False)
        self._imprint_settings_changed()

    def _handle_imprint_surface_pick(self, position) -> bool:
        if not self.imprint_pick_surface.isChecked() or self.busy:
            return False
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        picker = vtkCellPicker()
        picker.SetTolerance(.001)
        picker.PickFromListOn()
        picker.AddPickList(self.source_actor)
        picker.Pick(position[0], position[1], 0, self.renderer)
        index = picker.GetCellId()
        if index < 0 or self.source_faces is None or index >= len(self.source_faces):
            self.vtk_widget.setToolTip("Click directly on the mesh.")
            return True
        selected = self.crop_candidate[3] if self.crop_candidate is not None else self.selected_face_mask
        if selected is not None and not selected[index]:
            self.vtk_widget.setToolTip("Pick a surface within the selected region.")
            return True
        vertices = self.source_points[self.source_faces[index]]
        point = vertices.mean(axis=0)
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal /= max(float(np.linalg.norm(normal)), 1e-20)
        if float(normal @ (np.asarray(self.renderer.GetActiveCamera().GetPosition()) - point)) < 0:
            normal = -normal
        self._imprint_surface_seeds.append({"point": point.tolist(), "normal": normal.tolist()})
        self._imprint_surface_seed_source = self.source_poly
        self.imprint_pick_surface.setChecked(False)
        self._imprint_settings_changed()
        return True

    def _imprint_memory_budget(self) -> int:
        return max(1024**2, int(available_physical_memory() * .06))

    def _pending_solid_imprint_input(self):
        context = getattr(self, "_solidify_preview_context", None)
        if context is None or not self.pass_ready or self.preview_poly is None:
            return None
        selected = self.crop_candidate[3] if self.crop_candidate is not None else self.selected_face_mask
        previous = context.get("mask")
        if previous is None and (selected is None or not selected.any()):
            return self.preview_poly
        if previous is not None and selected is not None and np.array_equal(previous, selected):
            return context["solid_poly"]
        return None

    def _refresh_imprint_plan(self) -> None:
        if not hasattr(self, "imprint_detail"):
            return
        if getattr(self, "_imprint_surface_seed_source", None) is not self.source_poly:
            self._imprint_surface_seeds = []
            self._imprint_surface_seed_source = self.source_poly
            self.imprint_pick_surface.setChecked(False)
        picks = len(self._imprint_surface_seeds)
        self.imprint_clear_picks.setText(f"Clear picks ({picks})" if picks else "Clear picks")
        self.imprint_pick_surface.setEnabled(self.source_poly is not None and not self.busy)
        self.imprint_clear_picks.setEnabled(bool(picks) and not self.busy)
        self.imprint_clear_picks.setVisible(bool(picks))
        scale = (max(np.ptp(np.asarray(self.source_poly.GetBounds()).reshape(3, 2), axis=1))
                 if self.source_poly is not None else 100.)
        self.solidify_gap.set_geometry(scale)
        for control in (self.imprint_offset, self.imprint_wall, self.imprint_bridge):
            control.set_geometry(scale)
        mold = self.imprint_output.currentText() == "Clamshell"
        preparation_needed = (mold and self._pending_solid_imprint_input() is None
                              and not self._working_mesh_requires_closed_output())
        self.imprint_prepare_solid.setVisible(preparation_needed)
        self.imprint_form.labelForField(self.imprint_prepare_solid).setVisible(preparation_needed)
        for widget in (self.imprint_shape, self.imprint_wall, self.imprint_bridge, self.imprint_detail, self.imprint_split):
            widget.setVisible(mold)
            self.imprint_form.labelForField(widget).setVisible(mold)
        self._sync_imprint_split_widget()
        self.envelope_button.setEnabled(self.source_poly is not None and not self.busy)
        detail_help = "Left: coarser and faster. Right: finer detail. Resolution adapts to available memory."
        if mold and self.source_poly is not None:
            from clamshell_mold import mold_sampling_plan
            bounds = np.asarray(self.source_poly.GetBounds()).reshape(3, 2)
            bounds += np.array([-1, 1]) * self.imprint_offset.value()
            plan = mold_sampling_plan(bounds, self.imprint_wall.value(), self.imprint_bridge.value(),
                                      memory_budget_bytes=self._imprint_memory_budget(),
                                      detail=(.5, .75, 1, 1.5, 2)[self.imprint_detail.value()])
            adjusted = " · memory adjusted" if plan["memory_adjusted"] else ""
            detail_help += f"\nSampling pitch: {plan['pitch_mm']:.3f} mm. Estimated memory: {fmt_bytes(plan['estimated_bytes'])}{adjusted}."
        self.imprint_detail.setToolTip(detail_help)
        self._refresh_imprint_preview_actions()

    def _imprint_preview_settings_key(self):
        mold = self.imprint_output.currentText() == "Clamshell"
        pending_solid = self._pending_solid_imprint_input()
        key = (id(self.source_poly), id(pending_solid) if pending_solid is not None else None,
               self.imprint_output.currentText(), self.imprint_offset.value(),
               tuple((tuple(seed["point"]), tuple(seed["normal"])) for seed in self._imprint_surface_seeds))
        selected = self.crop_candidate[3] if self.crop_candidate is not None else self.selected_face_mask
        key += (selected.tobytes() if selected is not None else None,)
        if mold:
            key += (self.imprint_prepare_solid.isChecked(), self.imprint_shape.currentText(),
                    self.imprint_wall.value(), self.imprint_bridge.value(), self.imprint_detail.value(),
                    self.imprint_split.currentText())
            if self.imprint_prepare_solid.isChecked():
                key += (self.solidify_method.currentText(), self.solidify_gap.value(), self.solidify_quality.currentText())
            if self.imprint_split.currentText() == "Custom":
                rep = getattr(self, "_imprint_split_representation", None)
                if rep is not None:
                    key += (tuple(rep.GetNormal()), tuple(rep.GetOrigin()))
            elif self.imprint_split.currentText() == "Current view":
                key += (tuple(self.renderer.GetActiveCamera().GetDirectionOfProjection()),)
        return key

    def _refresh_imprint_preview_actions(self):
        if not hasattr(self, "envelope_controls"):
            return
        present = getattr(self, "_envelope_poly", None) is not None
        current = present and getattr(self, "_envelope_settings_key", None) == self._imprint_preview_settings_key()
        self.envelope_separate_halves.setVisible(present and self.imprint_output.currentText() == "Clamshell"
                                               and len(getattr(self, "_envelope_halves", [])) == 2)
        self.envelope_controls.setVisible(current)
        self.envelope_stats.setVisible(current)
        self._update_envelope_overlay()
        self.envelope_button.setText("Cast")
        self.envelope_button.setToolTip(
            "Regenerate the cast preview with the changed settings before applying or saving."
            if present and not current else "Create an cast preview using the current settings."
        )
        self._refresh_workflow_actions()
        self._update_estimated_size()

    def _imprint_settings_changed(self, *_args) -> None:
        self._refresh_imprint_plan()
        if hasattr(self, "imprint_error"):
            self.imprint_error.hide()
        self._refresh_imprint_preview_actions()

    def _sync_imprint_split_widget(self) -> None:
        if not hasattr(self, "renderer"):
            return
        visible = (self.imprint_section_toggle.isChecked() and self.source_poly is not None and self.imprint_output.currentText() == "Clamshell"
                   and self.imprint_split.currentText() == "Custom" and not getattr(self, "_closing", False))
        help_text = ("Rotate: drag in the viewport; drag around the edge to roll. "
                     "Move: drag the plane. Middle-drag rotates the camera.") if visible else ""
        if self.imprint_pick_surface.isChecked():
            help_text = "Click a known exterior surface."
        self.vtk_widget.setToolTip(help_text)
        self.imprint_split.setToolTip(help_text)
        widget = getattr(self, "_imprint_split_widget", None)
        if not visible:
            self._imprint_plane_drag = None
            if widget is not None:
                widget.SetEnabled(False)
                self.renderer.RemoveViewProp(self._imprint_split_representation)
            if hasattr(self, "imprint_plane_tools"):
                self.imprint_plane_tools.hide()
            self._schedule_imprint_split_render()
            return
        if widget is None:
            from vtkmodules.vtkInteractionWidgets import (
                vtkImplicitPlaneRepresentation,
                vtkImplicitPlaneWidget2,
            )
            representation = vtkImplicitPlaneRepresentation()
            representation.SetPlaceFactor(1.15)
            representation.DrawPlaneOn()
            representation.DrawOutlineOff()
            representation.OutlineTranslationOff()
            representation.ScaleEnabledOff()
            representation.LockNormalToCameraOff()
            representation.SetHandleSize(10)
            representation.SetHandleColor(.1, 1, .85)
            representation.GetPlaneProperty().SetColor(1, .12, .45)
            representation.GetPlaneProperty().SetOpacity(.08)
            representation.GetSelectedPlaneProperty().SetColor(1, .12, .45)
            representation.GetSelectedPlaneProperty().SetOpacity(.15)
            for prop in (representation.GetNormalProperty(), representation.GetEdgesProperty()):
                prop.SetColor(1, .12, .45)
                prop.SetLineWidth(4)
                prop.SetAmbient(1)
            representation.GetNormalProperty().SetColor(.1, 1, .85)
            representation.GetSelectedNormalProperty().SetColor(1, .85, .1)
            widget = vtkImplicitPlaneWidget2()
            widget.SetInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())
            widget.SetCurrentRenderer(self.renderer)
            widget.SetRepresentation(representation)
            widget.SetPriority(.95)
            widget.AddObserver("StartInteractionEvent", self._imprint_plane_started)
            widget.AddObserver("InteractionEvent", self._imprint_plane_moved)
            widget.AddObserver("EndInteractionEvent", self._imprint_plane_moved)
            self._imprint_split_widget = widget
            self._imprint_split_representation = representation
            self.imprint_plane_tools = QFrame(self.vtk_widget)
            self.imprint_plane_tools.setObjectName("splitPlaneTools")
            tools_row = QHBoxLayout(self.imprint_plane_tools)
            tools_row.setContentsMargins(7, 5, 7, 5)
            tools_row.setSpacing(5)
            tools_row.addWidget(QLabel("Split", self.imprint_plane_tools))
            self._imprint_plane_mode_buttons = {}
            for mode in ("Move", "Rotate"):
                button = QPushButton(mode, self.imprint_plane_tools)
                button.setCheckable(True)
                button.setProperty("toolMode", True)
                button.setToolTip("Drag the plane along its normal." if mode == "Move" else "Drag in the viewport to turn the plane. Drag around the edge to roll it clockwise or counterclockwise.")
                button.clicked.connect(lambda _checked=False, selected=mode: self._set_imprint_plane_mode(selected))
                self._imprint_plane_mode_buttons[mode] = button
                tools_row.addWidget(button)
            center_button = QPushButton("Center", self.imprint_plane_tools)
            center_button.setToolTip("Move the split plane to the model center, keeping its angle.")
            center_button.clicked.connect(self._center_imprint_plane)
            tools_row.addWidget(center_button)
            self._set_imprint_plane_mode("Rotate")
        representation = self._imprint_split_representation
        if getattr(self, "_imprint_widget_source", None) is not self.source_poly:
            representation.PlaceWidget(self.source_poly.GetBounds())
            representation.SetOrigin(*self.source_poly.GetCenter())
            representation.SetNormal(0, 0, 1)
            self._imprint_widget_source = self.source_poly
        widget.SetProcessEvents(not self.busy and not self.imprint_pick_surface.isChecked()
                                and getattr(self, "_imprint_plane_mode", "Rotate") == "Move")
        if not widget.GetEnabled():
            widget.SetEnabled(True)
        if not self.renderer.HasViewProp(representation):
            self.renderer.AddViewProp(representation)
        self.imprint_plane_tools.setEnabled(not self.busy)
        representation.SetVisibility(True)
        self.imprint_plane_tools.show()
        self._position_imprint_plane_tools()
        self._schedule_imprint_split_render()

    def _schedule_imprint_split_render(self) -> None:
        # Qt controls do not trigger VTK interaction renders. Coalesce their
        # updates and render after layout changes, outside VTK callbacks.
        if getattr(self, "_imprint_split_render_pending", False) or getattr(self, "_closing", False):
            return
        self._imprint_split_render_pending = True
        QTimer.singleShot(0, self._render_imprint_split)

    def _render_imprint_split(self) -> None:
        self._imprint_split_render_pending = False
        if getattr(self, "_closing", False):
            return
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _position_imprint_plane_tools(self) -> None:
        tools = getattr(self, "imprint_plane_tools", None)
        if tools is not None and tools.isVisible():
            tools.adjustSize()
            tools.move(max(8, (self.vtk_widget.width() - tools.width()) // 2),
                       max(8, self.vtk_widget.height() - tools.height() - 14))
            tools.raise_()
            self._apply_rounded_mask(tools, 8)

    def _set_imprint_plane_mode(self, mode: str) -> None:
        self._imprint_plane_mode = mode
        for name, button in self._imprint_plane_mode_buttons.items():
            button.setChecked(name == mode)
        widget = getattr(self, "_imprint_split_widget", None)
        if widget is not None:
            widget.SetProcessEvents(mode == "Move" and not self.busy
                                    and not self.imprint_pick_surface.isChecked())
        self._imprint_plane_drag = None

    def _center_imprint_plane(self) -> None:
        if self.busy or self.source_poly is None:
            return
        self._imprint_split_representation.SetOrigin(*self.source_poly.GetCenter())
        self._imprint_plane_moved()
        self.vtk_widget.GetRenderWindow().Render()

    def _imprint_trackball_vector(self, position, center, radius):
        x, y = (position.x() - center[0]) / radius, (center[1] - position.y()) / radius
        length = x*x + y*y
        vector = np.array([x, y, np.sqrt(max(0., 1. - length))])
        return vector / max(np.linalg.norm(vector), 1e-12)

    def _handle_imprint_plane_drag(self, event) -> bool:
        mouse_types = (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove, QEvent.Type.MouseButtonRelease)
        if event.type() not in mouse_types:
            return False
        widget = getattr(self, "_imprint_split_widget", None)
        active = (widget is not None and widget.GetEnabled() and not self.busy
                  and getattr(self, "_imprint_plane_mode", "Rotate") == "Rotate"
                  and not self.imprint_pick_surface.isChecked())
        if not active:
            self._imprint_plane_drag = None
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() != Qt.MouseButton.LeftButton or event.modifiers() != Qt.KeyboardModifier.NoModifier:
                return False
            representation = self._imprint_split_representation
            self.renderer.SetWorldPoint(*representation.GetOrigin(), 1.)
            self.renderer.WorldToDisplay()
            dx, dy, _ = self.renderer.GetDisplayPoint()
            rw, rh = self.vtk_widget.GetRenderWindow().GetSize()
            center = (dx*self.vtk_widget.width()/max(1,rw),
                      (rh-dy)*self.vtk_widget.height()/max(1,rh))
            radius = max(40., min(self.vtk_widget.width(), self.vtk_widget.height())*.32)
            camera = self.renderer.GetActiveCamera()
            forward = np.array(camera.GetDirectionOfProjection())
            right = np.cross(forward, camera.GetViewUp()); right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            self._imprint_plane_drag = (self._imprint_trackball_vector(event.position(), center, radius),
                np.array(representation.GetNormal()), np.column_stack((right, up, -forward)), center, radius)
            return True
        drag = getattr(self, "_imprint_plane_drag", None)
        if drag is None:
            return False
        if event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            self._imprint_plane_drag = None
            return True
        start, normal, basis, center, radius = drag
        end = self._imprint_trackball_vector(event.position(), center, radius)
        axis = np.cross(start, end)
        sine = np.linalg.norm(axis); cosine = np.clip(start @ end, -1., 1.)
        if sine > 1e-10:
            axis = basis @ (axis / sine)
            rotated = normal*cosine + np.cross(axis, normal)*sine + axis*(axis @ normal)*(1.-cosine)
            self._imprint_split_representation.SetNormal(*(rotated / np.linalg.norm(rotated)))
            self._imprint_plane_moved()
            self._schedule_imprint_split_render()
        return True

    def _imprint_plane_started(self, _widget=None, _event=None) -> None:
        representation = self._imprint_split_representation
        if representation.GetInteractionState() != representation.Outside:
            state = representation.Rotating if self._imprint_plane_mode == "Rotate" else representation.Pushing
            representation.SetInteractionState(state)

    def _imprint_plane_moved(self, _widget=None, _event=None) -> None:
        # The VTK widget renders its own interaction. Do not re-enable it or
        # enter Render() recursively from this callback.
        self._refresh_imprint_preview_actions()

    @staticmethod
    def _solidify_pitch(poly, quality):
        scale = float(np.ptp(np.asarray(poly.GetBounds()).reshape(3, 2), axis=1).max())
        return scale / {"Draft": 96, "Balanced": 160, "Fine": 224}[quality]

    def _solidify_input_key(self):
        mask = self.crop_candidate[3] if self.crop_candidate is not None else self.selected_face_mask
        selection = hash(mask.tobytes()) if mask is not None and mask.any() else None
        return (self.generation, id(self.source_poly), selection)

    def _refresh_solidify_info(self):
        if self.source_poly is None or not hasattr(self, "solidify_info_labels") or getattr(self, "_closing", False):
            return
        key = self._solidify_input_key()
        if self._solidify_info_result is not None and self._solidify_info_result[0] == key:
            report = self._solidify_info_result[1]
            unit = self.units.currentText()
            self.solidify_info_labels["open_rims"].setText(f"Open rims\n{fmt_count(report['open_rims'])}")
            self.solidify_info_labels["open_rims"].setToolTip(
                f"Connected open-border groups in the current input: {report['closed_loops']} closed loops. Cropped selection boundaries also count. A rim is evidence of an opening, not a known amount of missing wall.")
            self.solidify_info_labels["open_edges"].setText(f"Open edges\n{fmt_count(report['open_edges'])}")
            self.solidify_info_labels["open_edges"].setToolTip(
                f"Edges with a face on just one side, measured at STL precision. {report['nonmanifold_edges']} edges have more than two faces. Repairability is established by generating and validating a preview.")
            self.solidify_info_labels["rim_length"].setText(f"Rim length\n{report['rim_length'] / UNIT_MM[unit]:.2f} {unit}")
            self.solidify_info_labels["rim_length"].setToolTip("Combined length of open borders. This is perimeter, not missing area or wall thickness. The completed preview reports how much surface was rebuilt.")
            return
        if self._solidify_info_key != key:
            for label, title in zip(self.solidify_info_labels.values(), ("Open rims", "Open edges", "Rim length")):
                label.setText(title + "\n?")
                label.setToolTip("Measuring open boundaries in the current mesh or selection.")
        if self._solidify_info_future is not None or self.busy:
            return
        self._solidify_info_key = key
        points, faces = self.source_points, self.source_faces
        mask = self.crop_candidate[3] if self.crop_candidate is not None else self.selected_face_mask
        mask = mask.copy() if mask is not None and mask.any() else None
        def measure():
            from solidify_metrics import input_metrics
            return input_metrics(points, faces[mask] if mask is not None else faces)
        self._solidify_info_future = self.executor.submit(measure)
        self.solidify_info_timer.start()

    def _poll_solidify_info(self):
        future = self._solidify_info_future
        if getattr(self, "_closing", False) or future is None:
            self.solidify_info_timer.stop()
            return
        if not future.done():
            return
        self.solidify_info_timer.stop()
        self._solidify_info_future = None
        try:
            self._solidify_info_result = (self._solidify_info_key, future.result())
        except Exception:  # noqa: BLE001 - optional boundary measurements do not prevent repair
            for label in self.solidify_info_labels.values():
                label.setText("Unavailable")
            return
        # Availability follows the analysis even when the section is collapsed.
        # Stale results from a previous mesh or selection cannot hide this input.
        self._refresh_solidify_availability()

    def _solidify_settings_changed(self, *_args):
        method = self.solidify_method.currentText()
        for control in (self.solidify_gap, self.solidify_quality):
            control.setVisible(method == "Voxel gap closing")
            self.solidify_form.labelForField(control).setVisible(method == "Voxel gap closing")
        self.solidify_error.hide()
        if getattr(self, "_solidify_preview_context", None) is not None:
            self.solidify_preview_button.setText("Solidify")
            self._set_primary_action(False, False)
        self._imprint_settings_changed()

    def _cancel_solidify_preview(self, _checked=False):
        if self.busy:
            if self.active_operation.startswith("Solidifying"):
                self._cancel_mesh_operation()
            else:
                return
        self._cancel_optimized_preview()
        self._solidify_preview_context = None
        self.solidify_preview_button.setText("Solidify")
        self.status.setText("Solid preview discarded")

    def _preview_solidify(self, _checked=False) -> None:
        if self.busy or self.source_poly is None or getattr(self, "_solidification_applied", False):
            return
        measured = self._solidify_info_result
        if measured is not None and measured[0] == self._solidify_input_key():
            report = measured[1]
            if report["open_edges"] == 0 and report["nonmanifold_edges"] == 0:
                self._refresh_solidify_availability()
                return
        poly = self.source_poly
        mask = None
        if self.crop_candidate is not None:
            poly, _, _, mask = self.crop_candidate
            mask = mask.copy()
        elif self.selected_face_mask is not None and self.selected_face_mask.any():
            mask = self.selected_face_mask.copy()
            poly = self._poly_from_face_mask(mask)[0]
        snapshot = vtkPolyData()
        snapshot.DeepCopy(poly)
        source_points, source_faces = self.source_points, self.source_faces
        self._clear_envelope()
        self.imprint_section_toggle.setChecked(False)
        if self.preview_poly is not None:
            self._cancel_optimized_preview()
        method = self.solidify_method.currentText()
        solid_quality = self.solidify_quality.currentText()
        gap = self.solidify_gap.value()
        pitch = self._solidify_pitch(snapshot, self.solidify_quality.currentText())
        budget = self._imprint_memory_budget()
        script_step = None if mask is not None else {"kind": "solidify", "method": method,
            "quality": solid_quality.lower(), "gap_mm": gap, "gpu": self.gpu_preference}
        self.solidify_error.hide()
        token = self._begin_mesh_operation("Solidifying selection" if mask is not None else "Solidifying mesh", "Local")
        cancel = self.operation_cancel_event
        progress_path = self._start_operation_progress(token, "Preparing solid")
        self.operation_steps = ["Checking surface intersections", "Repairing local surface defects",
                                "Building constrained volume", "Classifying volume", "Resolving surface contacts",
                                "Sampling solid surface", "Closing solid gaps", "Filling solid interior",
                                "Creating solid mesh", "Validating solid", "Analyzing results"]

        def progress(stage, completed, total):
            _write_process_progress(progress_path, stage, completed, total)

        def job():
            try:
                from solidify import solidify_mesh
                solid, report = solidify_mesh(snapshot, method, gap, pitch, budget, cancel, progress,
                                              gpu_preference=self.gpu_preference)
                completion = report["completion"]
                from solidify import combine_selected_solid
                preview = combine_selected_solid(source_points, source_faces, mask, solid)
                points, faces = polydata_arrays(preview)
                metrics = mesh_quality_metrics(source_points, source_faces, points, faces,
                                               progress=progress, cancel_event=cancel)
                return {"kind": "solidify", "poly": preview, "solid_poly": solid,
                        "points": points, "faces": faces, "metrics": metrics,
                        "mask": mask, "source_points": source_points, "source_faces": source_faces,
                        "report": report, "completion": completion, "script_step": script_step}
            except Exception as error:  # noqa: BLE001 - report worker failures inline
                return {"kind": "solidify", "error": str(error).strip() or type(error).__name__}
        future = self.executor.submit(job)
        self.operation_future = future
        future.add_done_callback(lambda result: self._deliver_mesh_edit_future(token, "solidify", result))

    def _create_envelope(self) -> None:
        if self.busy or self.source_poly is None or getattr(self, "_closing", False):
            return
        pending_solid = self._pending_solid_imprint_input()
        if self.preview_poly is not None and self.preview_poly is not self.source_poly and pending_solid is None:
            self._cancel_optimized_preview()
        self.imprint_error.hide()
        mode, offset = self.imprint_output.currentText(), self.imprint_offset.value()
        shape, split = self.imprint_shape.currentText(), self.imprint_split.currentText()
        self._sync_imprint_split_widget()
        representation = getattr(self, "_imprint_split_representation", None)
        settings_key = self._imprint_preview_settings_key()
        custom_normal = representation.GetNormal() if representation is not None else (0, 0, 1)
        custom_origin = representation.GetOrigin() if representation is not None else self.source_poly.GetCenter()
        from clamshell_mold import mold_sampling_plan
        bounds = np.asarray(self.source_poly.GetBounds()).reshape(3, 2)
        bounds += np.array([-1, 1]) * offset
        memory_budget = self._imprint_memory_budget()
        plan = mold_sampling_plan(bounds, self.imprint_wall.value(), self.imprint_bridge.value(),
                                  memory_budget_bytes=memory_budget,
                                  detail=(.5, .75, 1, 1.5, 2)[self.imprint_detail.value()])
        wall, bridge, pitch = plan['wall_mm'], plan['bridge_mm'], plan['pitch_mm']
        poly = pending_solid if pending_solid is not None else self.source_poly
        if pending_solid is None and self.crop_candidate is not None:
            poly = self.crop_candidate[0]
        elif pending_solid is None and self.selected_face_mask is not None and self.selected_face_mask.any():
            poly = self._poly_from_face_mask(self.selected_face_mask)[0]
        from vtkmodules.vtkCommonDataModel import vtkPolyData
        snapshot = vtkPolyData()
        snapshot.DeepCopy(poly)
        surface_seeds = tuple({"point": tuple(seed["point"]), "normal": tuple(seed["normal"])}
                              for seed in self._imprint_surface_seeds)
        selected = self.crop_candidate[3] if self.crop_candidate is not None else self.selected_face_mask
        surface_key = (id(self.source_poly), id(pending_solid) if pending_solid is not None else None,
                       hash(selected.tobytes()) if selected is not None else None,
                       offset, self.gpu_preference, tuple((seed["point"], seed["normal"]) for seed in surface_seeds))
        solidify_settings = (self.solidify_method.currentText(), self.solidify_gap.value(),
                             self.solidify_quality.currentText()) if mode == "Clamshell" and self.imprint_prepare_solid.isChecked() else None
        surface_key += (solidify_settings,)
        surface_cache = getattr(self, "_imprint_surface_cache", None)
        camera = self.renderer.GetActiveCamera()
        imprint_direction = np.asarray(camera.GetFocalPoint()) - np.asarray(camera.GetPosition())
        token = self._begin_mesh_operation("Cast", "Building preview")
        token = self.edit_generation

        envelope_cancel = self.operation_cancel_event
        envelope_gpu = self.gpu_preference
        progress_path = self._start_operation_progress(token, "Preparing surface normals")
        self.operation_steps = ["Preparing surface normals", "Detecting supported surface sheets", "Measuring wall thickness", "Building ray index", "Preparing GPU kernel",
                                "Surface probes", "Creating offset mesh"]
        if mode == "Clamshell":
            self.operation_steps.extend(["Measuring exterior depth", "Bridging cavity volume", "Removing interior islands",
                                         "Building outer blob", "Sampling outer blob", "Adding clamshell walls", "Removing detached material",
                                         "Creating hollow clamshell", "Splitting clamshell half 1", "Splitting clamshell half 2", "Cleaning generated geometry"])
        last_report = [0.0, ""]

        def envelope_progress(stage, completed, total):
            now = time.monotonic()
            if stage != last_report[1] or now - last_report[0] >= 0.08 or completed == total:
                _write_process_progress(progress_path, stage, completed, total)
                last_report[:] = [now, stage]

        split_normal = None if split == "Unsplit" else (
            custom_normal if split == "Custom" else
            imprint_direction if split == "Current view" else np.eye(3)[["X", "Y", "Z"].index(split)])
        prepared_settings = solidify_settings
        if pending_solid is not None:
            context = getattr(self, "_solidify_preview_context", {}) or {}
            pending_step = context.get("script_step")
            if pending_step:
                prepared_settings = (pending_step['method'], pending_step['gap_mm'], pending_step['quality'].title())
        script_step = None if (surface_seeds or (selected is not None and selected.any())) else {
            "kind": "cast", "mode": mode, "offset_mm": offset, "shape": shape,
            "wall_mm": self.imprint_wall.value(), "bridge_mm": self.imprint_bridge.value(),
            "detail": (.5, .75, 1, 1.5, 2)[self.imprint_detail.value()], "gpu": envelope_gpu,
            "split_normal": np.asarray(split_normal).tolist() if split_normal is not None else None,
            "split_origin": list(custom_origin) if split == "Custom" else None,
            "prepare_solid": prepared_settings is not None,
            "method": prepared_settings[0] if prepared_settings else "Surface repair",
            "gap_mm": prepared_settings[1] if prepared_settings else 0,
            "solidify_quality": prepared_settings[2].lower() if prepared_settings else "balanced"}

        def build():
            try:
                from exposed_surface import exposed_surface
                if surface_cache is not None and surface_cache[0] == surface_key:
                    _, result, report = surface_cache
                    envelope_progress("Creating offset mesh", 3, 3)
                else:
                    prepared = snapshot
                    from solidify import needs_solidification
                    needs_preparation = needs_solidification(snapshot)
                    if solidify_settings is not None:
                        from solidify import solidify_mesh
                        if needs_preparation:
                            prepared, _ = solidify_mesh(snapshot, solidify_settings[0], solidify_settings[1],
                                self._solidify_pitch(snapshot, solidify_settings[2]), memory_budget,
                                cancel=envelope_cancel, progress=envelope_progress,
                                gpu_preference=envelope_gpu)
                            needs_preparation = False
                    elif mode == "Clamshell" and needs_preparation:
                        raise ValueError("The scan is open. Enable Prepare with Solidify and review the Solidify method and gap settings.")
                    marker = prepared.GetFieldData().GetArray("ImprintClosedOutput")
                    if not needs_preparation or (marker is not None and marker.GetValue(0)):
                        from solidify import offset_solid
                        result, report = offset_solid(prepared, offset)
                    else:
                        result, report = exposed_surface(prepared, offset, cancel=envelope_cancel,
                                                         gpu_preference=envelope_gpu, progress=envelope_progress,
                                                         surface_seeds=surface_seeds)
                from closed_reduction import is_closed_mesh
                cached_surface = (surface_key, result, report)
                if mode == "Envelope":
                    output_points, output_faces = polydata_arrays(result)
                    closed_output = is_closed_mesh(output_points, output_faces)
                    return {"kind": "envelope", "settings_key": settings_key, "poly": result, "spacing": None, "mode": mode,
                            "report": report, "surface_cache": cached_surface, "closed_output": closed_output, "script_step": script_step}
                from vtkmodules.vtkFiltersCore import vtkAppendPolyData

                from clamshell_mold import clamshell_mold
                split_normal = None if split == "Unsplit" else (
                    custom_normal if split == "Custom" else
                    imprint_direction if split == "Current view" else np.eye(3)[["X", "Y", "Z"].index(split)])
                halves, mold_report = clamshell_mold(result, wall, bridge, pitch, split_normal, shape,
                                                     cancel=envelope_cancel, progress=envelope_progress,
                                                     memory_budget_bytes=memory_budget,
                                                     split_origin=custom_origin if split == "Custom" else None)
                combined = vtkAppendPolyData()
                for half in halves:
                    combined.AddInputData(half)
                combined.Update()
                return {"kind": "envelope", "settings_key": settings_key, "poly": combined.GetOutput(), "spacing": mold_report['pitch_mm'],
                        "mode": f"{shape} clamshell", "halves": halves, "report": mold_report, "script_step": script_step,
                        "closed_output": True,
                        "surface_cache": cached_surface, "split_normal": split_normal,
                        "split_origin": custom_origin if split == "Custom" else result.GetCenter()}
            except Exception as exc:  # noqa: BLE001 - relay experimental worker errors to the UI.
                return {"kind": "envelope", "error": str(exc).strip() or type(exc).__name__}

        future = self.executor.submit(build)
        self.operation_future = future
        future.add_done_callback(lambda f: self._deliver_mesh_edit_future(token, "envelope", f))

    def _clear_envelope(self) -> None:
        self.envelope_button.setText("Cast")
        for original, visibility in getattr(self, "_envelope_original_actors", []):
            original.SetVisibility(visibility)
        self._envelope_original_actors = []
        for rim in getattr(self, "_envelope_rim_actors", []):
            self.renderer.RemoveActor(rim)
        self._envelope_rim_actors = []
        actor = getattr(self, "_envelope_actor", None)
        if actor is not None:
            self.renderer.RemoveActor(actor)
        self._envelope_actor = None
        self._envelope_poly = None
        self._request_orientation_miniature()
        self._envelope_closed = False
        self._envelope_settings_key = None
        self._envelope_halves = []
        self._envelope_spaced_halves = []
        self._envelope_spaced_poly = None
        self.envelope_separate_halves.hide()
        self.envelope_controls.hide()
        self.envelope_stats.hide()
        self.envelope_stats.clear()
        self._refresh_imprint_preview_actions()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_envelope_overlay(self, *_args) -> None:
        actor = getattr(self, "_envelope_actor", None)
        if actor is not None:
            expanded = (self.imprint_section_toggle.isChecked()
                        and getattr(self, "_envelope_settings_key", None) == self._imprint_preview_settings_key())
            halves = getattr(self, "_envelope_halves", [])
            if len(halves) == 2 and self.envelope_separate_halves.isChecked():
                actor.GetMapper().SetInputData(self._envelope_spaced_poly)
            else:
                actor.GetMapper().SetInputData(self._envelope_poly)
            self._set_actor_display_mode(actor)
            actor.GetMapper().ScalarVisibilityOff()
            actor.GetProperty().SetColor(1.0, 0.72, 0.15)
            for original, visibility in getattr(self, "_envelope_original_actors", []):
                original.SetVisibility(visibility if not expanded or self.envelope_original_visible.isChecked() else False)
            actor.SetVisibility(expanded and self.envelope_visible.isChecked())
            actor.GetProperty().SetOpacity(self.envelope_opacity.value() / 100)
            for index, rim in enumerate(getattr(self, "_envelope_rim_actors", [])):
                rim.SetVisibility(expanded and self.envelope_visible.isChecked())
                if self.envelope_separate_halves.isChecked():
                    shift = (np.asarray(self._envelope_spaced_halves[index].GetPoint(0))
                             - np.asarray(self._envelope_halves[index].GetPoint(0)))
                    rim.SetPosition(*shift)
                else:
                    rim.SetPosition(0, 0, 0)
                self.renderer.AddActor(rim)
            self._request_orientation_miniature()
            self.vtk_widget.GetRenderWindow().Render()

    def _replace_with_envelope(self) -> None:
        poly = getattr(self, "_envelope_poly", None)
        if getattr(self, "_envelope_spaced_poly", None) is not None and self.envelope_separate_halves.isChecked():
            poly = self._envelope_spaced_poly
        if (self.busy or poly is None or not poly.GetNumberOfPolys()
                or getattr(self, "_envelope_settings_key", None) != self._imprint_preview_settings_key()):
            return
        replacement = vtkPolyData()
        replacement.DeepCopy(poly)
        points, faces = polydata_arrays(replacement)
        closed_output = getattr(self, "_envelope_closed", False)
        script_step = getattr(self, "_envelope_script_step", None)
        if script_step is not None:
            script_step = dict(script_step, separate_halves=self.envelope_separate_halves.isChecked())
        if closed_output:
            from vtkmodules.vtkCommonCore import vtkIntArray
            marker = vtkIntArray()
            marker.SetName("ImprintClosedOutput")
            marker.InsertNextValue(1)
            replacement.GetFieldData().AddArray(marker)
        self._clear_envelope()
        if getattr(self, "_solidify_preview_context", None) is not None:
            self._cancel_optimized_preview()
        for control, value in ((self.imprint_output, "Clamshell"),
                               (self.imprint_shape, "Blob"), (self.imprint_split, "Unsplit")):
            with QSignalBlocker(control):
                control.setCurrentText(value)
        for control in (self.imprint_offset, self.imprint_wall, self.imprint_bridge):
            with QSignalBlocker(control):
                control.reset_auto()
        for control, value in ((self.imprint_detail, 2), (self.envelope_opacity, 15)):
            with QSignalBlocker(control):
                control.setValue(value)
        self._imprint_surface_cache = None
        self._imprint_surface_seeds = []
        self.imprint_pick_surface.setChecked(False)
        self._imprint_widget_source = None
        if hasattr(self, "_imprint_plane_mode_buttons"):
            self._set_imprint_plane_mode("Rotate")
        self.imprint_error.clear()
        self.imprint_error.hide()
        self.imprint_section_toggle.setChecked(False)
        self._sync_imprint_split_widget()
        self._refresh_imprint_plan()
        self._set_working_mesh(
            replacement, points, faces, "Replaced working mesh with cast",
            committed_optimization=False, script_pass=script_step,
            script_omission="Selection or guided cast" if script_step is None else None,
            solidification_applied=closed_output,
        )
        self.optimization_toggle.setChecked(True)
        self._refresh_imprint_plan()
        self._refresh_imprint_preview_actions()

    def _save_envelope(self) -> None:
        poly = getattr(self, "_envelope_poly", None)
        if (self.busy or poly is None
                or getattr(self, "_envelope_settings_key", None) != self._imprint_preview_settings_key()):
            return
        halves = (self._envelope_spaced_halves if self.envelope_separate_halves.isChecked()
                  else getattr(self, "_envelope_halves", []))
        title = "Save clamshell halves" if len(halves) == 2 else "Save cast"
        remembered = str(self.settings.value("imprint/save_directory", ""))
        directory = Path(remembered) if remembered else None
        if directory is None or not directory.is_dir():
            directory = self.source_path.parent if self.source_path else Path(self._file_open_directory())
        from mesh_export_ui import choose_mesh_export
        choice = choose_mesh_export(self, title, directory / "cast.stl", poly)
        if choice is None:
            return
        base, merge = choice
        outputs = [(base, poly)]
        if len(halves) == 2:
            outputs = [(base.with_name(f"{base.stem}-half-{i + 1}{base.suffix}"), half) for i, half in enumerate(halves)]
            if (any(destination.exists() for destination, _ in outputs)
                    and QMessageBox.question(self, APP_NAME, "Replace existing clamshell half files?") != QMessageBox.StandardButton.Yes):
                return
        if self.source_path and any(destination.resolve() == Path(self.source_path).resolve() for destination, _ in outputs):
            QMessageBox.warning(self, APP_NAME, "Choose a separate output file.")
            return
        marker = poly.GetFieldData().GetArray("ImprintClosedOutput")
        self._export_meshes(outputs, merge, closed=bool(marker is not None and marker.GetValue(0)))
        self.settings.setValue("imprint/save_directory", str(base.resolve().parent))

    def _apply_selection_action(self, action: str) -> None:
        if action == "invert":
            self._invert_selection()
            return
        if action == "clear":
            self._clear_selection()
            return
        if action == "undo":
            self._undo_selection()
            return
        if self.crop_candidate is None:
            if self.selected_face_mask is None or not self.selected_face_mask.any():
                return
            if action == "optimize":
                self._optimize_saved_selection()
            elif action == "delete":
                self._delete_selected(confirm=False)
            elif action == "crop":
                cropped, points, faces = self._poly_from_face_mask(self.selected_face_mask)
                self._set_working_mesh(
                    cropped, points, faces, "Cropped mesh", script_omission="Crop"
                )
            return
        cropped, points, faces, selected_face_mask = self.crop_candidate
        if not len(faces):
            return
        if action not in {"add", "crop", "optimize", "delete"}:
            return
        self.crop_candidate = None
        self._hide_crop_preview()
        self.selected_face_mask = selected_face_mask.copy()
        self.delete_selection_button.setEnabled(True)
        self.clear_selection_button.setEnabled(True)
        if action == "add":
            self._refresh_persistent_selection()
            self._show_selection_primary_action()
            self._show_selection_action_window()
            self.status.setText(
                f"Selection saved • {fmt_count(int(selected_face_mask.sum()))} triangles • "
                f"{self._selection_size_summary(int(selected_face_mask.sum()))}"
            )
            return
        if action == "delete":
            self._delete_selected(confirm=False)
            return
        if action == "crop":
            self._set_working_mesh(
                cropped, points, faces, "Cropped mesh", script_omission="Crop"
            )
            return
        self._optimize_selected_region(points, faces, selected_face_mask)

    def _optimize_selected_region(
        self, selected_points: np.ndarray, selected_faces: np.ndarray, selected_face_mask: np.ndarray
    ) -> None:
        if self.source_points is None or self.source_faces is None:
            return
        if len(selected_faces) < 5:
            self.status.setText("Selection is too small to optimize.")
            return
        target = min(
            self.selection_target
            or smart_target(selected_points, selected_faces, self.smart_quality.currentText()),
            len(selected_faces) - 1,
        )
        algorithm = self.algorithm.currentText()
        source_points = self.source_points
        source_faces = self.source_faces
        preserve_closed = self._working_mesh_requires_closed_output()
        if getattr(self, "_envelope_poly", None) is not None:
            self._clear_envelope()
        if getattr(self, "_solidify_preview_context", None) is not None:
            self._cancel_optimized_preview()
        token = self._begin_mesh_operation(
            "Optimizing selection",
            (
                depth_adaptive_progress_text(self.gpu_preference)
                if is_gpu_guided_algorithm(algorithm)
                else f"{fmt_count(len(selected_faces))} selected triangles → about {fmt_count(target)}"
            ),
        )
        self._set_primary_action(False, False)
        cancel_event = self.operation_cancel_event
        progress_path = self._start_operation_progress(
            token,
            "Analyzing depth and density"
            if algorithm == DEPTH_DENSITY_ADAPTIVE_LABEL
            else "Analyzing surface"
            if is_gpu_guided_algorithm(algorithm)
            else "Reducing mesh topology",
        )
        self.status.setText(
            depth_adaptive_progress_text(self.gpu_preference) + "…"
            if is_gpu_guided_algorithm(algorithm)
            else (
                f"Optimizing selected region: {fmt_count(len(selected_faces))} to about "
                f"{fmt_count(target)} triangles…"
            )
        )

        def job():
            try:
                reduced_points, reduced_faces = simplify_arrays(
                    selected_points,
                    selected_faces,
                    target,
                    algorithm,
                    gpu_preference=self.gpu_preference,
                    progress=lambda stage, completed, total: _write_process_progress(
                        progress_path, stage, completed, total
                    ),
                )
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled
                untouched_faces = np.ascontiguousarray(source_faces[~selected_face_mask], dtype=np.int32)
                point_offset = len(source_points)
                combined_points = np.ascontiguousarray(
                    np.vstack((source_points, reduced_points)), dtype=np.float64
                )
                combined_faces = np.ascontiguousarray(
                    np.vstack((untouched_faces, reduced_faces + point_offset)), dtype=np.int64
                )
                used = np.unique(combined_faces)
                remap = np.full(len(combined_points), -1, dtype=np.int64)
                remap[used] = np.arange(len(used))
                merged_points = np.ascontiguousarray(combined_points[used])
                merged_faces = np.ascontiguousarray(remap[combined_faces], dtype=np.int32)
                from closed_reduction import validate_closed_reduction
                closed_output = preserve_closed
                if closed_output:
                    merged_points, merged_faces = validate_closed_reduction(merged_points, merged_faces,
                        lambda stage, done, total: _write_process_progress(progress_path, stage, done, total))
                metrics = mesh_quality_metrics(
                    source_points,
                    source_faces,
                    merged_points,
                    merged_faces,
                    progress=lambda stage, completed, total: _write_process_progress(
                        progress_path, stage, completed, total
                    ),
                    cancel_event=cancel_event,
                )
                metrics["closed_output"] = closed_output
                return {
                    "kind": "optimize",
                    "poly": arrays_polydata(merged_points, merged_faces),
                    "points": merged_points,
                    "faces": merged_faces,
                    "removed": len(selected_faces) - len(reduced_faces),
                    "metrics": metrics,
                }
            except OperationCancelled:
                return {"kind": "optimize", "cancelled": True}
            except Exception:  # noqa: BLE001 - worker boundary reports failures through the result payload
                return {"kind": "optimize", "error": traceback.format_exc()}

        future = self.executor.submit(job)
        self.operation_future = future
        future.add_done_callback(lambda f: self.edit_bridge.done.emit(token, f.result()))

    def _optimize_saved_selection(self) -> None:
        if (
            self.selected_face_mask is None
            or self.source_faces is None
            or len(self.selected_face_mask) != len(self.source_faces)
            or not self.selected_face_mask.any()
        ):
            self.status.setText("No selection")
            return
        mask = self.selected_face_mask.copy()
        _, points, faces = self._poly_from_face_mask(mask)
        self._optimize_selected_region(points, faces, mask)

    def _invert_selection(self) -> None:
        if self.busy or self.source_faces is None or self.source_points is None:
            return
        previous = (self.crop_candidate[3].copy() if self.crop_candidate is not None else
                    self.selected_face_mask.copy() if self.selected_face_mask is not None else
                    np.zeros(len(self.source_faces), dtype=bool))
        if len(previous) != len(self.source_faces):
            return
        self._discard_selection_preview_for_retry()
        points, faces = self.source_points, self.source_faces
        token = self._begin_mesh_operation("Inverting selection", "Selecting unselected triangles")
        cancel = self.operation_cancel_event

        def job():
            try:
                mask = np.logical_not(previous)
                selected = faces[mask]
                used, inverse = np.unique(selected, return_inverse=True)
                if cancel.is_set():
                    return {"kind": "invert-selection", "cancelled": True}
                selected_points = np.ascontiguousarray(points[used])
                selected_faces = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32)
                return {"kind": "invert-selection", "previous": previous,
                        "candidate": (arrays_polydata(selected_points, selected_faces),
                                      selected_points, selected_faces, mask)}
            except Exception:  # noqa: BLE001 - worker errors are delivered to the GUI
                return {"kind": "invert-selection", "error": traceback.format_exc()}

        future = self.executor.submit(job)
        self.operation_future = future
        future.add_done_callback(lambda f: self.edit_bridge.done.emit(token, f.result()))

    def _poly_from_face_mask(self, mask: np.ndarray) -> tuple[vtkPolyData, np.ndarray, np.ndarray]:
        selected_faces = self.source_faces[mask]
        used = np.unique(selected_faces)
        remap = np.full(len(self.source_points), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        points = np.ascontiguousarray(self.source_points[used])
        faces = np.ascontiguousarray(remap[selected_faces], dtype=np.int32)
        return arrays_polydata(points, faces), points, faces

    def _selected_triangle_count(self) -> int:
        if self.crop_candidate is not None:
            return len(self.crop_candidate[2])
        if self.selected_face_mask is not None and self.selected_face_mask.any():
            return int(self.selected_face_mask.sum())
        return 0

    def _selection_size_summary(self, selected_triangles: int) -> str:
        total_triangles = len(self.source_faces) if self.source_faces is not None else 0
        if total_triangles <= 0:
            return "0% • 0 B of 0 B"
        selected_triangles = min(max(0, int(selected_triangles)), total_triangles)
        percentage = selected_triangles * 100.0 / total_triangles
        selected_size = binary_stl_size(selected_triangles)
        total_size = binary_stl_size(total_triangles)
        return f"{percentage:.1f}% • {fmt_bytes(selected_size)} of {fmt_bytes(total_size)}"

    def _refresh_persistent_selection(self) -> None:
        if (
            self.selected_face_mask is None
            or self.source_faces is None
            or len(self.selected_face_mask) != len(self.source_faces)
            or not self.selected_face_mask.any()
        ):
            return
        poly, _, _ = self._poly_from_face_mask(self.selected_face_mask)
        self._show_crop_highlight(poly)

    def _show_selection_action_window(self) -> None:
        """Keep selection actions visible whenever geometry remains selected."""
        if (
            self.crop_info_frame is None
            or self.crop_info_label is None
            or self.selected_face_mask is None
            or self.source_faces is None
            or len(self.selected_face_mask) != len(self.source_faces)
            or not self.selected_face_mask.any()
        ):
            return
        poly, points, faces = self._poly_from_face_mask(self.selected_face_mask)
        unit = self.units.currentText()
        size = np.asarray(bounds_size(poly.GetBounds())) / UNIT_MM[unit]
        self.crop_info_label.setText(
            "SELECTION\n"
            f"Selected: {fmt_count(len(points))} vertices, {fmt_count(len(faces))} triangles\n"
            f"Mesh share: {self._selection_size_summary(len(faces))}\n"
            f"Dimensions: {fmt_size(tuple(size))} {unit}"
        )
        if self.selection_undo_button is not None:
            self.selection_undo_button.setVisible(self._can_step_back_selection())
        self.crop_info_frame.adjustSize()
        self._apply_rounded_mask(self.crop_info_frame, 5)
        self.crop_info_frame.show()
        self.crop_info_frame.raise_()

    def _can_step_back_selection(self) -> bool:
        if not self.selection_undo_stack:
            return False
        previous = self.selection_undo_stack[-1]
        return previous is not None

    def _undo_selection(self) -> None:
        """Restore the selection before the most recent selection change."""
        if not self.selection_undo_stack:
            return
        previous = self.selection_undo_stack.pop()
        self.crop_candidate = None
        self._hide_crop_preview()
        self.selected_face_mask = previous.copy() if previous is not None else None
        if self.selected_face_mask is not None and self.selected_face_mask.any():
            self._refresh_persistent_selection()
            self._show_selection_primary_action()
            self._show_selection_action_window()
            self.status.setText("Selection stepped back")
        else:
            self.selected_face_mask = None
            self.delete_selection_button.setEnabled(False)
            self.clear_selection_button.setEnabled(False)
            self.status.setText("Selection cleared")

    def _clear_selection(self, _checked=False, message: str | None = "Selection cleared.") -> None:
        self.selected_face_mask = None
        self.crop_candidate = None
        self.selection_undo_stack.clear()
        self._hide_crop_preview()
        if self.selection_base_quality is not None:
            with QSignalBlocker(self.smart_quality):
                self.smart_quality.setCurrentText(self.selection_base_quality)
            with QSignalBlocker(self.algorithm):
                self.algorithm.setCurrentText(self.selection_base_algorithm or self.algorithm.currentText())
            if self.source_faces is not None:
                with QSignalBlocker(self.target):
                    self.target.setRange(1_000, max(1_000, len(self.source_faces)))
                    self.target.setValue(
                        min(self.selection_base_target or 1_000, max(1_000, len(self.source_faces)))
                    )
            self.selection_base_quality = None
            self.selection_base_algorithm = None
            self.selection_base_target = None
            self.selection_target = None
            self._update_dynamic_presets()
            self._sync_slider_from_target()
            self._update_estimated_size()
        self.delete_selection_button.setEnabled(False)
        self.clear_selection_button.setEnabled(False)
        if self.source_faces is not None and not self.busy:
            unlocked_result = (
                self.preview_poly is not None and self.preview_poly is not self.source_poly
            )
            self._set_primary_action(
                self.pass_ready and unlocked_result,
                unlocked_result or len(self.source_faces) > 1_000,
            )
        if message:
            self.status.setText(message)

    def _delete_selected(self, _checked=False, confirm: bool = True) -> None:
        if (
            self.selected_face_mask is None
            or self.source_faces is None
            or self.source_points is None
            or not self.selected_face_mask.any()
        ):
            return
        selected_count = int(self.selected_face_mask.sum())
        if confirm:
            answer = QMessageBox.question(
                self,
                "Delete selected triangles",
                f"Delete {selected_count:,} highlighted triangles from the current mesh?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        mask = self.selected_face_mask.copy()
        source_points = self.source_points
        source_faces = self.source_faces
        kept_faces_source = source_faces[~mask]
        if len(kept_faces_source) == 0:
            QMessageBox.warning(self, APP_NAME, "The selection contains the entire mesh. Use Crop instead.")
            return
        token = self._begin_mesh_operation(
            "Deleting selection", f"Removing {fmt_count(selected_count)} selected triangles"
        )
        self.status.setText(f"Deleting {fmt_count(selected_count)} selected triangles…")

        cancel_event = self.operation_cancel_event

        def job():
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled
                used = np.unique(kept_faces_source)
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled
                remap = np.full(len(source_points), -1, dtype=np.int64)
                remap[used] = np.arange(len(used))
                points = np.ascontiguousarray(source_points[used])
                faces = np.ascontiguousarray(remap[kept_faces_source], dtype=np.int32)
                return {
                    "kind": "delete",
                    "poly": arrays_polydata(points, faces),
                    "points": points,
                    "faces": faces,
                    "removed": selected_count,
                }
            except OperationCancelled:
                return {"kind": "delete", "cancelled": True}
            except Exception:  # noqa: BLE001 - worker boundary reports failures through the result payload
                return {"kind": "delete", "error": traceback.format_exc()}

        future = self.executor.submit(job)
        self.operation_future = future
        future.add_done_callback(lambda f: self.edit_bridge.done.emit(token, f.result()))

    def _deliver_mesh_edit_future(self, token, kind, future):
        if future.cancelled() or getattr(self, "_closing", False):
            return
        try:
            result = future.result()
        except Exception as error:  # noqa: BLE001 - worker failures are presented by the owning tool
            result = {"kind": kind, "error": str(error).strip() or type(error).__name__}
        self.edit_bridge.done.emit(token, result)

    def _finish_mesh_edit(self, token: int, result: dict) -> None:
        if token != self.edit_generation:
            return
        error = result.get("error")
        if "error" in result:
            error = str(error or "The operation returned no validated result.")
            self._end_mesh_operation()
            if result.get("kind") == "solidify":
                self._solidify_preview_context = None
                self.solidify_toggle.setChecked(True)
                if "intersect" in error.lower():
                    message = "Input surfaces intersect.\nCrop scan noise or select the intended region, then retry."
                elif "already closed" in error.lower():
                    message = error
                else:
                    message = "Solidify could not produce a validated solid.\nCrop suspect geometry or try a smaller selection."
                self.solidify_error.setText(message)
                self.solidify_error.setToolTip(error)
                self.solidify_error.show()
                self.status.setText("Solidify failed; mesh unchanged")
                self._refresh_workflow_actions()
                QTimer.singleShot(0, lambda: self.controls_widget.ensureWidgetVisible(self.solidify_error, 8, 16))
                return
            if result.get("kind") == "envelope":
                self._clear_envelope()
                self.imprint_section_toggle.setChecked(True)
                if "intersect" in error.lower():
                    message = "Input surfaces intersect.\nCrop scan noise or select the intended region, then retry Cast."
                elif "already closed" in error.lower():
                    message = "The preparation step made no changes.\nReview the source solid before retrying Cast."
                else:
                    message = "Cast could not produce a validated preview.\nCrop suspect geometry or try a smaller selection."
                self.imprint_error.setText(message)
                self.imprint_error.setToolTip(error)
                self.imprint_error.show()
                self.envelope_button.setText("Cast")
                self.status.setText("Cast failed; mesh unchanged")
                self._refresh_workflow_actions()
                QTimer.singleShot(0, lambda: self.controls_widget.ensureWidgetVisible(self.imprint_error, 8, 16))
                return
            QMessageBox.critical(self, APP_NAME, f"Mesh operation failed:\n\n{error}")
            self.status.setText("Mesh operation failed.")
            return
        if result.get("cancelled"):
            self._end_mesh_operation()
            self.status.setText("Mesh operation cancelled")
            return
        kind = result["kind"]
        if kind == "envelope" and (result.get("poly") is None or not result["poly"].GetNumberOfPolys()):
            self._finish_mesh_edit(token, {"kind": "envelope", "error": "Cast returned an empty mesh."})
            return
        if kind == "solidify":
            self._finish_simplify(self.generation, result["points"], result["faces"], result["metrics"], None)
            self.preview_poly = result["poly"]
            self._solidify_preview_context = result
            self.solidify_preview_button.setText("Solidify")
            self.solidify_toggle.setChecked(False)
            self._refresh_imprint_plan()
            self._refresh_workflow_actions()
            self.status.setText("Solid preview ready")
            return
        if kind == "envelope":
            self._end_mesh_operation()
            if result.get("surface_cache") is not None:
                self._imprint_surface_cache = result["surface_cache"]
            self._clear_envelope()
            self._envelope_poly = result["poly"]
            self._envelope_closed = bool(result.get("closed_output", False))
            self._envelope_settings_key = result.get("settings_key")
            self._envelope_script_step = result.get("script_step")
            self._envelope_halves = result.get("halves", [])
            if len(self._envelope_halves) == 2 and result.get("split_normal") is not None:
                from clamshell_mold import split_rim_lines
                for half in self._envelope_halves:
                    lines = split_rim_lines(half, result["split_normal"], result["split_origin"],
                                            max(result["spacing"] * 1e-4, 1e-6))
                    line_mapper = vtkPolyDataMapper()
                    line_mapper.SetInputData(lines)
                    line_mapper.ScalarVisibilityOff()
                    rim = vtkActor()
                    rim.SetMapper(line_mapper)
                    rim.PickableOff()
                    rim.GetProperty().SetColor(1.0, .12, .45)
                    rim.GetProperty().SetLineWidth(4.0)
                    rim.GetProperty().SetAmbient(1)
                    rim.GetProperty().SetDiffuse(0)
                    self._envelope_rim_actors.append(rim)
            if len(self._envelope_halves) == 2:
                from vtkmodules.vtkFiltersCore import vtkAppendPolyData

                from clamshell_mold import separated_mold_halves
                self._envelope_spaced_halves = separated_mold_halves(self._envelope_halves)
                joined = vtkAppendPolyData()
                for half in self._envelope_spaced_halves:
                    joined.AddInputData(half)
                joined.Update()
                self._envelope_spaced_poly = joined.GetOutput()
            self.envelope_separate_halves.setVisible(len(self._envelope_halves) == 2)
            self.envelope_separate_halves.setChecked(len(self._envelope_halves) == 2)
            triangle_count = self._envelope_poly.GetNumberOfCells()
            vertex_count = self._envelope_poly.GetNumberOfPoints()
            self.envelope_stats.setText(
                f"{fmt_count(triangle_count)} triangles · "
                f"{fmt_count(vertex_count)} vertices · "
                f"{fmt_bytes(binary_stl_size(triangle_count))} STL"
            )
            self.envelope_stats.setToolTip("Estimated binary STL file size, not memory usage.")
            if "halves" in result:
                self.envelope_stats.setToolTip(
                    f"Sampling pitch: {result['spacing']:.3f} mm. Cavity fit and gap bridging are approximate. "
                    "Separate parts applies to preview, replacement and export, preserving independent selection after STL import."
                )
            self.envelope_stats.show()
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(self._envelope_poly)
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.PickableOff()
            actor.GetProperty().SetColor(1.0, 0.72, 0.15)
            self._envelope_actor = actor
            self._envelope_source_poly = self.source_poly
            self._envelope_original_actors = [
                (original, original.GetVisibility())
                for original in (self.active_actor, self.crop_selection_actor)
                if original is not None
            ]
            self.renderer.AddActor(actor)
            self.envelope_original_visible.setChecked(True)
            self.envelope_visible.setChecked(True)
            self.envelope_controls.show()
            self.renderer.ResetCameraClippingRange()
            self._update_envelope_overlay()
            self.envelope_button.setText("Cast")
            self.imprint_error.hide()
            self._refresh_imprint_preview_actions()
            self._sync_imprint_split_widget()
            if result["spacing"] is None:
                report = result["report"]
                self.status.setText(
                    f"{report.get('device', 'CPU')} · {report.get('seconds', 0):.1f}s · "
                    f"{report['kept']:,} triangles · {report['removed']:,} hidden triangles removed"
                )
            else:
                self.status.setText(f"Clamshell preview: {result['spacing']:.3f} mm voxel spacing")
            return
        if kind in {"connected-selection", "invert-selection"}:
            self._end_mesh_operation()
            if kind == "invert-selection":
                self.selection_target = None
            self.selected_face_mask = None
            self.selection_undo_stack.append(result["previous"])
            self.crop_points = []
            self.crop_world_points = []
            self._update_crop_overlay(render=False)
            self._present_selection(*result["candidate"])
            return
        removed = int(result["removed"])
        if kind == "optimize":
            self._end_mesh_operation()
            self._show_selection_optimization_preview(result, removed)
            return
        else:
            message = f"Deleted {fmt_count(removed)} selected triangles"
        # Restore controls before applying the completed edit. Otherwise the pre-operation button
        # snapshot overwrites the Undo, Previous, Reload Original, and Save states established by
        # the new working mesh.
        self._end_mesh_operation()
        self._set_working_mesh(
            result["poly"],
            result["points"],
            result["faces"],
            message,
            preserve_target=(kind == "optimize"),
            committed_optimization=True if kind == "optimize" else None,
            script_omission="Selection delete",
        )

    def _show_selection_optimization_preview(self, result: dict, removed: int) -> None:
        if self.source_poly is None:
            return
        self.preview_poly = result["poly"]
        if result.get("metrics", {}).get("closed_output"):
            from vtkmodules.vtkCommonCore import vtkIntArray
            marker = vtkIntArray(); marker.SetName("ImprintClosedOutput"); marker.InsertNextValue(1)
            self.preview_poly.GetFieldData().AddArray(marker)
        self.simplified_actor = None
        self.preview_is_selection = True
        self.preview_dirty = False
        preview_points = result["points"]
        preview_faces = result["faces"]
        self.preview_count.setText(fmt_count(len(preview_faces)))
        self.preview_vertices.setText(fmt_count(len(preview_points)))
        self._update_quality_metrics(result.get("metrics"), measured=True)
        self._update_optimization_ratio(len(preview_faces))
        self.estimated_file_size.setText(fmt_bytes(binary_stl_size(len(preview_faces))))
        source_size = np.asarray(bounds_size(self.source_poly.GetBounds()))
        preview_size = np.asarray(bounds_size(self.preview_poly.GetBounds()))
        self.dimension_drift_model_units = np.abs(preview_size - source_size)
        self._update_dimension_labels()
        self._show_poly(self.preview_poly, reset_camera=False)
        self.showing_original = False
        self.toggle_mesh_button.setText("Show original")
        self.toggle_mesh_button.setEnabled(True)
        self._set_primary_action(True, True)
        self.save_button.setEnabled(self.mesh_modified)
        self.status.setText(
            f"Selection optimization ready, removed {fmt_count(removed)} triangles â€¢ "
            "Apply or Cancel"
        )

    def _cancel_crop(self, message: str | None) -> None:
        self.crop_start = None
        self._clear_selection(message=message)

    def _hide_crop_preview(self) -> None:
        for actor in self.crop_overlay_actors:
            self.renderer.RemoveViewProp(actor)
        self.crop_overlay_actors = []
        self.crop_points = []
        self.crop_world_points = []
        self.crop_drag_index = None
        if self.crop_selection_actor is not None:
            self.renderer.RemoveActor(self.crop_selection_actor)
            self.crop_selection_actor = None
        if self.source_actor is not None:
            self.source_actor.GetProperty().SetOpacity(1.0)
        if self.crop_info_frame is not None:
            self.crop_info_frame.hide()
        if hasattr(self, "vtk_widget"):
            QApplication.processEvents()
            self.vtk_widget.GetRenderWindow().Render()

    def _reload_original(self) -> None:
        if self.source_path is None or not self.mesh_modified:
            return
        answer = QMessageBox.question(
            self,
            "Reload original mesh",
            "Discard all crops, orientation changes, and the optimized result, then reload the "
            "untouched source STL?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._cancel_crop(None)
        self.load_path(self.source_path, restore_opening_view=True)

    def _capture_history_state(self) -> MeshHistoryState:
        return MeshHistoryState(
            self.source_poly,
            self.source_points,
            self.source_faces,
            self.source_actor,
            self.mesh_modified,
            self.preview_poly,
            self.simplified_actor,
            (
                self.dimension_drift_model_units.copy()
                if self.dimension_drift_model_units is not None
                else None
            ),
            self.showing_original,
            self.preview_dirty,
            self.has_committed_optimization,
            tuple(self.optimization_script_passes),
            tuple(self.optimization_script_omissions),
            getattr(self, "_solidification_applied", False),
            self.target.value(), self.smart_quality.currentText(),
            getattr(self, "_last_quality_metrics", None),
            getattr(self, "_last_quality_measured", False),
            self.script_base_poly,
            getattr(self, "_preview_script_step", None),
        )

    def _record_history_state(self) -> None:
        if self.source_poly is None or self.source_points is None or self.source_faces is None:
            return
        state = self._capture_history_state()
        self.undo_stack.append(state)
        self.undo_bytes += state.points.nbytes + state.faces.nbytes
        self.redo_stack.clear()
        self.redo_bytes = 0
        self.undo_button.setEnabled(True)
        self.redo_button.setEnabled(False)
        self.previous_view_poly = None
        self.previous_view_actor = None
        self.previous_mesh_button.setEnabled(True)

    def _set_working_mesh(
        self,
        poly: vtkPolyData,
        points: np.ndarray,
        faces: np.ndarray,
        message: str,
        record_undo: bool = True,
        preserve_target: bool = False,
        cached_actor: vtkActor | None = None,
        modified_state: bool | None = None,
        committed_optimization: bool | None = None,
        script_pass: dict[str, object] | None = None,
        script_omission: str | None = None,
        solidification_applied: bool = False,
    ) -> None:
        self._mark_geometry_activity(message, 3.0)
        self._solidify_preview_context = None
        self.solidify_preview_button.setText("Solidify")
        self.solidify_error.clear()
        self.solidify_error.hide()
        previous_target = (
            self.selection_base_target
            if preserve_target and self.selection_base_target is not None
            else self.target.value()
        )
        if (
            record_undo
            and self.source_poly is not None
            and self.source_points is not None
            and self.source_faces is not None
        ):
            # Working arrays are replaced, never modified in place, so retaining references makes
            # undo instantaneous even for multi-million-triangle scans.
            self._record_history_state()
        self._solidification_applied = solidification_applied
        if script_pass is not None:
            self.optimization_script_passes.append(script_pass)
        if script_pass is None:
            self.script_base_poly = poly
            self.optimization_script_passes.clear()
            checkpoint_edit = script_omission or "Mesh edit"
            if checkpoint_edit not in self.optimization_script_omissions:
                self.optimization_script_omissions.append(checkpoint_edit)
        self.timer.stop()
        self._cancel_crop(None)
        self._reset_previous_view()
        self.generation += 1
        self.pending_target = None
        self._probe_statistics_job = None
        self.source_poly = poly
        self.source_points = points
        self.source_faces = faces
        self.preview_poly = None
        self.preview_is_selection = False
        if committed_optimization is not None:
            self.has_committed_optimization = committed_optimization
        self.mesh_modified = True if modified_state is None else modified_state
        self._update_save_button_label()
        self.reload_button.setEnabled(self.mesh_modified)
        self.source_actor = cached_actor
        self.simplified_actor = None
        self.preview_count.setText("—")
        self.preview_vertices.setText("—")
        self.original_count.setText(fmt_count(len(faces)))
        self.original_vertices.setText(fmt_count(len(points)))
        self._update_optimization_ratio(len(faces) if self.mesh_modified else None)
        with QSignalBlocker(self.target):
            self.target.setMaximum(max(1_000, len(faces)))
        self._update_dynamic_presets()
        target = (
            min(previous_target, max(1_000, len(faces)))
            if preserve_target
            else min(
                smart_target(points, faces, self.smart_quality.currentText()),
                max(1_000, len(faces)),
            )
        )
        with QSignalBlocker(self.target):
            self.target.setValue(target)
        self._sync_slider_from_target()
        self.dimension_drift_model_units = None
        self.preview_dirty = True
        self._set_primary_action(False, len(faces) > 1_000)
        self.toggle_mesh_button.setEnabled(False)
        self.toggle_mesh_button.setText("Show original")
        self._sync_modified_mesh_actions()
        self.showing_original = True
        self._update_dimension_labels()
        self._update_estimated_size()
        self._update_processing_plan()
        self._show_poly(poly)
        QTimer.singleShot(500, self._precache_source_analysis)
        self._reset_previous_view()
        self.status.setText(f"{message} • {fmt_count(len(faces))} triangles")

    def _undo_working_mesh(self, _checked=False) -> None:
        if self.busy or not self.undo_stack:
            return
        token = self._begin_mesh_operation("Undo", "Restoring previous mesh state")
        QTimer.singleShot(50, lambda: self._finish_history_navigation("undo", token))

    def _redo_working_mesh(self, _checked=False) -> None:
        if self.busy or not self.redo_stack:
            return
        token = self._begin_mesh_operation("Redo", "Restoring next mesh state")
        QTimer.singleShot(50, lambda: self._finish_history_navigation("redo", token))

    def _finish_history_navigation(self, direction: str, token: int) -> None:
        if token != self.edit_generation:
            return
        if (
            self.source_poly is None
            or self.source_points is None
            or self.source_faces is None
        ):
            self._end_mesh_operation()
            return
        current = self._capture_history_state()
        if direction == "undo":
            self.redo_stack.append(current)
            self.redo_bytes += current.points.nbytes + current.faces.nbytes
            state = self.undo_stack.pop()
            self.undo_bytes -= state.points.nbytes + state.faces.nbytes
            message = "Undo"
        else:
            self.undo_stack.append(current)
            self.undo_bytes += current.points.nbytes + current.faces.nbytes
            state = self.redo_stack.pop()
            self.redo_bytes -= state.points.nbytes + state.faces.nbytes
            message = "Redo"
        self._restore_history_state(state, message)
        self._end_mesh_operation()
        self.undo_button.setEnabled(bool(self.undo_stack))
        self.redo_button.setEnabled(bool(self.redo_stack))
        # Ending the busy state restores the controls that were enabled before navigation.
        # Reapply state-derived availability afterward so undoing back to the loaded mesh does
        # not leave Reload Original enabled from the edited state.
        self._reset_previous_view()
        self._sync_modified_mesh_actions()

    def _restore_history_state(self, state: MeshHistoryState, message: str) -> None:
        self._solidify_preview_context = None
        self.solidify_preview_button.setText("Solidify")
        self.timer.stop()
        self._cancel_crop(None)
        self.showing_previous = False
        self.previous_view_poly = None
        self.previous_view_actor = None
        self.previous_mesh_button.setText("Show previous")
        self.generation += 1
        self.pending_target = None
        self.source_poly = state.poly
        self._solidification_applied = state.solidification_applied
        self.source_points = state.points
        self.source_faces = state.faces
        self.source_actor = state.actor
        self.mesh_modified = state.modified or self.export_calibration_transform is not None
        self.preview_poly = state.preview_poly
        self.preview_is_selection = False
        self.simplified_actor = state.preview_actor
        self.dimension_drift_model_units = (
            state.dimension_drift.copy() if state.dimension_drift is not None else None
        )
        self.showing_original = state.showing_original
        self.preview_dirty = state.preview_dirty
        self.has_committed_optimization = state.committed_optimization
        self.script_base_poly = state.script_base_poly
        self._preview_script_step = state.preview_script_step
        self.optimization_script_passes = list(state.optimization_script_passes)
        self.optimization_script_omissions = list(state.optimization_script_omissions)
        self._update_save_button_label()

        if state.quality is not None:
            with QSignalBlocker(self.smart_quality):
                self.smart_quality.setCurrentText(state.quality)
        self.original_count.setText(fmt_count(len(state.faces)))
        self.original_vertices.setText(fmt_count(len(state.points)))
        self.reload_button.setEnabled(self.mesh_modified)
        with QSignalBlocker(self.target):
            self.target.setMaximum(max(1_000, len(state.faces)))
            self.target.setValue(state.target_value if state.target_value is not None else
                                 smart_target(state.points, state.faces, self.smart_quality.currentText()))
        self._update_dynamic_presets()
        self._sync_slider_from_target()
        self._update_dimension_labels()
        self._update_estimated_size()

        if state.preview_poly is not None:
            preview_points, preview_faces = polydata_arrays(state.preview_poly)
            self.preview_count.setText(fmt_count(len(preview_faces)))
            self.preview_vertices.setText(fmt_count(len(preview_points)))
            self._update_optimization_ratio(len(preview_faces))
            self.estimated_file_size.setText(fmt_bytes(binary_stl_size(len(preview_faces))))
            self.toggle_mesh_button.setEnabled(state.preview_poly is not state.poly)
            self.toggle_mesh_button.setText(
                "Show optimized" if state.showing_original else "Show original"
            )
            self.save_button.setEnabled(self.mesh_modified)
            self._set_primary_action(
                not state.preview_dirty and state.preview_poly is not state.poly,
                (not state.preview_dirty and state.preview_poly is not state.poly)
                or (state.preview_dirty and len(state.faces) > 1_000),
            )
            shown = state.poly if state.showing_original else state.preview_poly
            self._show_poly(shown, reset_camera=False)
        else:
            self.preview_count.setText("—")
            self.preview_vertices.setText("—")
            self._update_optimization_ratio(len(state.faces) if state.modified else None)
            self.toggle_mesh_button.setEnabled(False)
            self.toggle_mesh_button.setText("Show original")
            self.save_button.setEnabled(state.modified)
            self._set_primary_action(False, state.preview_dirty and len(state.faces) > 1_000)
            self._show_poly(state.poly, reset_camera=False)
        if state.preview_poly is not None:
            self._update_quality_metrics(state.quality_metrics, measured=state.quality_measured)
        self._update_processing_plan()
        self.status.setText(message)

    def _discard_selection_preview_for_retry(self) -> None:
        if not self.preview_is_selection or self.preview_poly is None:
            return
        if self.simplified_actor is not None:
            self.renderer.RemoveActor(self.simplified_actor)
        self.preview_poly = None
        self.simplified_actor = None
        self.preview_is_selection = False
        self.pass_ready = False
        self.showing_original = True
        self.toggle_mesh_button.setText("Show original")
        self.toggle_mesh_button.setEnabled(False)
        self.dimension_drift_model_units = None
        self.drift.setText("â€”")
        self._show_poly(self.source_poly, reset_camera=False)
        self._show_selection_primary_action()

    def _target_changed(self, _value: int) -> None:
        if self.crop_candidate is not None or (
            self.selected_face_mask is not None and self.selected_face_mask.any()
        ):
            self._discard_selection_preview_for_retry()
            self.timer.stop()
            self.selection_target = self.target.value()
            self._sync_slider_from_target(self._selected_triangle_count())
            self._update_estimated_size()
            self.status.setText(f"Selection target • {fmt_count(self.selection_target)} triangles")
            return
        self.preview_dirty = True
        self._set_primary_action(False, not self.busy)
        self.toggle_mesh_button.setEnabled(
            self.preview_poly is not None and self.preview_poly is not self.source_poly
        )
        self._update_estimated_size()
        self._update_processing_plan()
        self._sync_slider_from_target()

    def _update_dynamic_presets(self, selection_total: int | None = None) -> None:
        if hasattr(self, "target_display"):
            target = (
                self.selection_target
                if selection_total and self.selection_target
                else self.target.value()
            )
            self.target_display.setText(fmt_count(int(target)))

    def _smart_quality_changed(self, quality: str) -> None:
        if quality in SMART_DIVISORS:
            self.last_quality = quality
            self.settings.setValue("optimization/last_quality", quality)
        self._apply_smart_target()
        self._update_processing_plan()

    def _processing_route_changed(self, _index: int) -> None:
        self.large_mesh_mode = str(self.processing_route.currentData() or "auto")
        self.settings.setValue("performance/large_mesh_mode", self.large_mesh_mode)
        self._update_processing_plan()

    def _set_processing_summary(self, text: str) -> None:
        self.processing_route.setToolTip(self._processing_help + "\n\n" + text)

    def _update_processing_plan(self) -> None:
        if not hasattr(self, "processing_route"):
            return
        triangle_count = (
            self.overview_total_triangles if self.tiled_mode
            else len(self.source_faces) if self.source_faces is not None else 0
        )
        if not triangle_count:
            self._set_processing_summary("Processing: open a mesh to estimate time and quality")
            return

        workers = self.geometry_processes or min(8, max(1, self.logical_cpu_count // 2))
        algorithm = self.algorithm.currentText()
        index_ready = False
        if self.source_path is not None:
            try:
                index_ready = index_matches(
                    self.source_path,
                    load_manifest(default_index_directory(self.source_path)),
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass

        route = self.large_mesh_mode
        if is_gpu_guided_algorithm(algorithm):
            scale = triangle_count / 4_126_315
            full_seconds = 12.55 * scale
            tiled_seconds = 27.30 * scale
            use_tiled = route == "tiled"
            if route == "auto":
                use_tiled = False
            if use_tiled:
                text = (
                    f"Tiled • {workers} workers • est. {tiled_seconds:.1f} s • "
                    "avg error +11% • peak error +8%"
                )
            else:
                route_label = "Auto → full mesh" if route == "auto" else "Full mesh"
                text = (
                    f"{route_label} • single mesh • est. {full_seconds:.1f} s • "
                    "baseline quality"
                )
            self._set_processing_summary(text)
            return

        break_even = tiled_runtime_break_even(triangle_count, algorithm, index_ready)
        if break_even is None:
            self._set_processing_summary(
                f"{route.title()} • {workers} workers • time and quality estimate unavailable"
            )
            return
        full_seconds, tiled_seconds, setup_seconds = break_even
        use_tiled = route == "tiled" or (
            route == "auto" and self.auto_tiling_enabled and tiled_seconds < full_seconds
        )
        if use_tiled:
            setup_text = "cached index" if setup_seconds == 0 else f"{setup_seconds:.1f} s setup"
            self._set_processing_summary(
                f"Tiled • {workers} workers • est. {tiled_seconds:.1f} s ({setup_text}) • "
                "avg error +6% • peak error +4%"
            )
        else:
            route_label = "Auto → full mesh" if route == "auto" else "Full mesh"
            self._set_processing_summary(
                f"{route_label} • single mesh • est. {full_seconds:.1f} s • baseline quality"
            )

    def _apply_smart_target(self, _checked=False) -> None:
        if self.source_points is None or self.source_faces is None:
            return
        if self.crop_candidate is not None or (
            self.selected_face_mask is not None and self.selected_face_mask.any()
        ):
            self._discard_selection_preview_for_retry()
            self.timer.stop()
            if self.crop_candidate is not None:
                _, points, faces, _ = self.crop_candidate
            else:
                _, points, faces = self._poly_from_face_mask(self.selected_face_mask)
            target = min(
                smart_target(points, faces, self.smart_quality.currentText()),
                max(1, len(faces) - 1),
            )
            self.selection_target = target
            with QSignalBlocker(self.target):
                self.target.setValue(target)
            self._sync_slider_from_target(len(faces))
            self._update_estimated_size()
            self.status.setText(f"Selection target • {fmt_count(target)} triangles")
            return
        target = smart_target(
            self.source_points, self.source_faces, self.smart_quality.currentText(),
            self._cached_source_measurements(self.source_points, self.source_faces),
        )
        if self.tiled_mode:
            target = round(target / max(1, len(self.source_faces)) * self.overview_total_triangles)
        self.preview_dirty = True
        self._set_primary_action(False, not self.busy)
        self.target.setValue(target)
        self._sync_slider_from_target()
        self._update_estimated_size()

    def _algorithm_changed(self, _algorithm: str) -> None:
        if is_gpu_guided_algorithm(_algorithm) and not gpu_backend_supported(self.gpu_preference):
            message = QMessageBox(self)
            message.setWindowTitle("GPU-guided optimization")
            message.setIcon(QMessageBox.Icon.Information)
            message.setText("This GPU-guided algorithm is unavailable on this system.")
            message.setInformativeText(
                "Its runtime is included with MeshMill, but the algorithm requires a compatible "
                "NVIDIA GPU and driver. MeshMill will use Density balanced instead."
            )
            message.addButton(QMessageBox.StandardButton.Ok)
            message.exec()
            with QSignalBlocker(self.algorithm):
                self.algorithm.setCurrentText("Density balanced")
            _algorithm = "Density balanced"
        self.last_algorithm = _algorithm
        QTimer.singleShot(500, self._precache_source_analysis)
        self.settings.setValue("optimization/last_algorithm", _algorithm)
        if self.source_faces is None:
            return
        if self.crop_candidate is not None or (
            self.selected_face_mask is not None and self.selected_face_mask.any()
        ):
            self._discard_selection_preview_for_retry()
            self.timer.stop()
            self._apply_smart_target()
            self.status.setText(f"Selection algorithm • {_algorithm}")
            return
        self.preview_dirty = True
        self._set_primary_action(False, not self.busy)
        self.toggle_mesh_button.setEnabled(
            self.preview_poly is not None and self.preview_poly is not self.source_poly
        )
        self._apply_smart_target()
        self._update_processing_plan()

    def _optimization_slider_knots(self, total):
        candidate = self.crop_candidate
        mask = self.selected_face_mask
        selected = self._selected_triangle_count()
        key = (id(self.source_points), id(self.source_faces), total,
               id(candidate), id(mask), selected)
        cached = getattr(self, "_optimization_slider_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1:]
        if candidate is not None and len(candidate[2]):
            _, points, faces, _ = candidate
            measurements = None
        elif selected and mask is not None:
            _, points, faces = self._poly_from_face_mask(mask)
            measurements = None
        else:
            points, faces = self.source_points, self.source_faces
            measurements = self._cached_source_measurements(points, faces)
        if measurements is None:
            measurements = {"bounds": np.ptp(points, axis=0), "area": mesh_surface_area(points, faces)}
        minimum = min(1_000, max(1, total - 1))
        maximum = max(minimum + 1, total)
        scale = total / max(1, len(faces))
        presets = [np.clip(round(smart_target(points, faces, quality, measurements) * scale),
                           minimum, maximum) for quality in SMART_DIVISORS]
        positions = np.array((0., 300., 600., 900., 1000.))
        counts = np.maximum.accumulate(np.array((minimum, *presets, maximum), dtype=float))
        self._optimization_slider_cache = (key, positions, counts)
        return positions, counts

    def _slider_changed(self, value: int) -> None:
        if self.source_faces is None:
            return
        selection_total = self._selected_triangle_count()
        total = selection_total or (
            self.overview_total_triangles if self.tiled_mode else len(self.source_faces)
        )
        positions, counts = self._optimization_slider_knots(total)
        count = round(np.exp(np.interp(value, positions, np.log(counts))))
        with QSignalBlocker(self.target):
            self.target.setValue(count)
        if selection_total:
            self._discard_selection_preview_for_retry()
            self.timer.stop()
            self.selection_target = min(count, max(4, selection_total - 1))
            self._update_estimated_size()
            self.status.setText(f"Selection target • {fmt_count(self.selection_target)} triangles")
            return
        self.preview_dirty = True
        self._set_primary_action(False, not self.busy)
        self.toggle_mesh_button.setEnabled(
            self.preview_poly is not None and self.preview_poly is not self.source_poly
        )
        self._update_estimated_size()

    def _sync_slider_from_target(self, selection_total: int | None = None) -> None:
        if self.source_faces is None:
            return
        total = selection_total or (
            self.overview_total_triangles if self.tiled_mode else len(self.source_faces)
        )
        positions, counts = self._optimization_slider_knots(total)
        target = self.selection_target if selection_total and self.selection_target else self.target.value()
        selected_index = list(SMART_DIVISORS).index(self.smart_quality.currentText()) + 1
        if abs(target - counts[selected_index]) < 1:
            pos = int(positions[selected_index])
        else:
            pos = round(float(np.interp(np.log(max(1, target)), np.log(counts), positions)))
        with QSignalBlocker(self.slider):
            self.slider.setValue(max(0, min(1000, pos)))

    def preview(self) -> None:
        if self.source_points is None or self.source_faces is None:
            return
        if self.crop_candidate is not None and not len(self.crop_candidate[2]):
            self.status.setText("Selection is empty. Resize the outline or clear it.")
            self.preview_button.setEnabled(False)
            return
        if self.tiled_mode:
            self._preview_tiled()
            return
        if self._selected_triangle_count():
            self.timer.stop()
            self.status.setText(
                f"Selection active • {self.shortcuts['selection_optimize']} to optimize"
            )
            return
        if not self.preview_dirty:
            return
        target = min(self.target.value(), len(self.source_faces))
        if target < 4:
            return
        if target >= len(self.source_faces):
            self.timer.stop()
            self.preview_poly = self.source_poly
            self.simplified_actor = self.source_actor
            self.preview_count.setText(fmt_count(len(self.source_faces)))
            self.preview_vertices.setText(fmt_count(len(self.source_points)))
            self._update_optimization_ratio(len(self.source_faces))
            self.dimension_drift_model_units = np.zeros(3)
            self._update_dimension_labels()
            self.estimated_file_size.setText(fmt_bytes(binary_stl_size(len(self.source_faces))))
            self.preview_dirty = False
            self._set_primary_action(False, False)
            self.toggle_mesh_button.setEnabled(False)
            self.save_button.setEnabled(self.mesh_modified)
            self.showing_original = True
            self.status.setText(f"No optimization needed • {fmt_count(len(self.source_faces))} triangles")
            return
        if self.busy:
            self.pending_target = target
            self.status.setText(f"Finishing current optimization; queued {fmt_count(target)} triangles…")
            return
        self.pending_target = None
        if getattr(self, "_envelope_poly", None) is not None:
            self._clear_envelope()
        if getattr(self, "_solidify_preview_context", None) is not None:
            self._cancel_optimized_preview()
        self.generation += 1
        self.running_target = target
        generation = self.generation
        algorithm = self.algorithm.currentText()
        self._preview_script_step = self._optimization_script_settings()
        self._begin_mesh_operation(
            "Optimizing",
            (
                depth_adaptive_progress_text(self.gpu_preference)
                if is_gpu_guided_algorithm(algorithm)
                else (
                    f"Optimizing {fmt_count(len(self.source_faces))} triangles to about "
                    f"{fmt_count(target)}"
                )
            ),
        )
        self._set_primary_action(False, False)
        self.save_button.setEnabled(self.mesh_modified)
        self.status.setText(
            depth_adaptive_progress_text(self.gpu_preference) + "…"
            if is_gpu_guided_algorithm(algorithm)
            else f"Optimizing to about {fmt_count(target)} triangles…"
        )
        points = self.source_points
        faces = self.source_faces
        progress_path = self._start_operation_progress(
            generation,
            "Analyzing depth and density"
            if algorithm == DEPTH_DENSITY_ADAPTIVE_LABEL
            else "Analyzing surface"
            if is_gpu_guided_algorithm(algorithm)
            else "Reducing mesh topology",
        )
        self.operation_uses_process = True
        future = self.optimization_executor.submit(
            simplify_process_job,
            points,
            faces,
            target,
            algorithm,
            self.gpu_preference,
            progress_path,
            self._cached_source_measurements(points, faces),
            self._cached_probe_statistics(points, faces),
            self._working_mesh_requires_closed_output(),
        )
        self.operation_future = future

        def completed(result_future) -> None:
            try:
                result = result_future.result()
            except Exception:  # noqa: BLE001 - Future may surface any worker exception type
                result = (None, None, None, traceback.format_exc())
            self._emit_prepared_simplify_result(generation, result, progress_path)

        future.add_done_callback(completed)

    def _preview_tiled(self) -> None:
        if (
            self.tiled_index_directory is None
            or self.source_points is None
            or self.source_faces is None
            or self.busy
        ):
            return
        self._preview_script_step = self._optimization_script_settings()
        target = min(self.target.value(), self.overview_total_triangles)
        self.generation += 1
        generation = self.generation
        self.running_target = target
        token = self._begin_mesh_operation(
            "Optimizing tiles",
            f"{fmt_count(self.overview_total_triangles)} triangles to about {fmt_count(target)}",
        )
        progress_path = self._start_operation_progress(token, "Optimizing tiles")
        output_path = self.tiled_index_directory / "optimized-preview.stl"
        source_points = self.source_points.copy()
        source_faces = self.source_faces.copy()
        preserve_closed = self._working_mesh_requires_closed_output()
        cancel_event = self.operation_cancel_event

        def job():
            try:
                def progress(stage: str, completed: int, total: int) -> None:
                    labels = {
                        "optimizing tiles": "Optimizing tiles",
                        "assembling": "Assembling optimized mesh",
                    }
                    _write_process_progress(
                        progress_path,
                        labels.get(stage, stage.capitalize()),
                        completed,
                        total,
                    )

                optimize_indexed_mesh(
                    self.tiled_index_directory,
                    output_path,
                    target,
                    aggressiveness=7.0,
                    algorithm=self.algorithm.currentText(),
                    gpu_preference=self.gpu_preference,
                    workers=self.geometry_processes or None,
                    memory_budget_bytes=max(
                        64 * 1024**2,
                        round(
                            available_physical_memory()
                            * self.large_mesh_memory_percent
                            / 100.0
                        ),
                    ),
                    cancel_event=cancel_event,
                    progress_callback=progress,
                )
                _write_process_progress(progress_path, "Loading optimized mesh", 0, 1)
                _poly, points, faces = load_stl(output_path)
                if preserve_closed:
                    from closed_reduction import validate_closed_reduction
                    points, faces = validate_closed_reduction(points, faces, progress)
                metrics = mesh_quality_metrics(
                    source_points,
                    source_faces,
                    points,
                    faces,
                    progress=lambda stage, completed, total: _write_process_progress(
                        progress_path, stage, completed, total
                    ),
                    cancel_event=cancel_event,
                )
                _write_process_progress(progress_path, "Loading optimized mesh", 1, 1)
                metrics["closed_output"] = preserve_closed
                return points, faces, metrics, None
            except Exception:  # noqa: BLE001 - worker boundary returns diagnostics to the UI
                return None, None, None, traceback.format_exc()

        future = self.executor.submit(job)
        self.operation_future = future

        def completed(result_future) -> None:
            try:
                result = result_future.result()
            except Exception:  # noqa: BLE001 - Future errors are reported on the GUI thread
                result = (None, None, None, traceback.format_exc())
            self._emit_prepared_simplify_result(generation, result, progress_path)

        future.add_done_callback(completed)

    def _emit_prepared_simplify_result(self, generation, result, progress_path):
        if generation != self.generation:
            return
        points, faces, metrics, error = result
        prepared_viewport = None
        if error is None:
            try:
                poly = arrays_polydata(points, faces)
                prepared = prepare_viewport_poly(poly, lambda stage, done, total:
                    _write_process_progress(progress_path, stage, done, total))
                prepared_viewport = (poly, prepared)
            except Exception:  # noqa: BLE001 - viewport preparation reports through the result bridge
                error = traceback.format_exc()
        self.bridge.done.emit(generation, points, faces, metrics, error, prepared_viewport)

    def _finish_simplify(self, generation: int, points, faces, metrics, error, prepared_viewport=None) -> None:
        self._solidify_preview_context = None
        if generation != self.generation:
            return
        if metrics and metrics.get("geometry_workload"):
            from geometry_workload import merge
            merge(metrics["geometry_workload"])
        self._end_mesh_operation()
        target_changed = self.running_target is not None and self.target.value() != self.running_target
        self.preview_dirty = target_changed
        self._set_primary_action(False, self.preview_dirty)
        self.running_target = None
        if error:
            self.status.setText("Optimization failed.")
            self.preview_dirty = True
            self._set_primary_action(False, True)
            QMessageBox.critical(self, APP_NAME, error)
        elif generation == self.generation:
            self.preview_poly = (prepared_viewport[0] if prepared_viewport is not None
                                 else arrays_polydata(points, faces))
            if metrics and metrics.get('closed_output'):
                from vtkmodules.vtkCommonCore import vtkIntArray
                marker = vtkIntArray(); marker.SetName('ImprintClosedOutput'); marker.InsertNextValue(1)
                self.preview_poly.GetFieldData().AddArray(marker)
            self.preview_is_selection = False
            self.simplified_actor = None
            self.preview_count.setText(fmt_count(len(faces)))
            self.preview_vertices.setText(fmt_count(len(points)))
            self._update_optimization_ratio(len(faces))
            self._update_quality_metrics(metrics, measured=True)
            self.estimated_file_size.setText(fmt_bytes(binary_stl_size(len(faces))))
            source_size = np.asarray(bounds_size(self.source_poly.GetBounds()))
            preview_size = np.asarray(bounds_size(self.preview_poly.GetBounds()))
            drift = np.abs(preview_size - source_size)
            self.dimension_drift_model_units = drift
            self._update_dimension_labels()
            self._show_poly(self.preview_poly, reset_camera=False,
                            prepared=prepared_viewport[1] if prepared_viewport is not None else None)
            self.showing_original = False
            self.toggle_mesh_button.setText("Show original")
            self.toggle_mesh_button.setEnabled(True)
            self._set_primary_action(True, True)
            self.save_button.setEnabled(self.mesh_modified)
            self.status.setText(f"Optimized result ready • {fmt_count(len(faces))} triangles")
            if self.verification_screenshot is not None:
                capture_path = self.verification_screenshot
                self.verification_screenshot = None
                if getattr(self, "verification_selection", False):
                    QTimer.singleShot(750, lambda: self._capture_selection_and_exit(capture_path))
                elif getattr(self, "verification_loupe", False):
                    QTimer.singleShot(750, lambda: self._capture_loupe_and_exit(capture_path))
                else:
                    QTimer.singleShot(750, lambda: self._capture_and_exit(capture_path))
        if self.pending_target is not None:
            self.pending_target = None
            self.preview_dirty = True
            QTimer.singleShot(0, self.preview)

    def _units_changed(self, _unit: str) -> None:
        self._update_dimension_labels()
        self._update_scale_bar()
        if self.measure_units is not None and not self.measure_units_overridden:
            with QSignalBlocker(self.measure_units):
                self.measure_units.setCurrentText(_unit)
            self._update_measure_panel()

    def _update_optimization_ratio(self, optimized_triangles: int | None) -> None:
        working_count = (self.overview_total_triangles if self.tiled_mode else
                         len(self.source_faces) if self.source_faces is not None else 0)
        if working_count <= 0 or not optimized_triangles:
            self.optimization_ratio.setText("—")
            return
        reduction = 100.0 * (1.0 - optimized_triangles / working_count)
        self.optimization_ratio.setText(f"{reduction:.1f}%")

    def _update_quality_metrics(
        self, metrics: dict[str, float] | None, *, measured: bool = False
    ) -> None:
        self._last_quality_metrics = dict(metrics) if metrics else None
        self._last_quality_measured = measured
        if not hasattr(self, "quality_metric_labels"):
            return
        abbreviations = {
            "rms_percent": "RMS",
            "p95_percent": "P95",
            "area_percent": "AREA",
            "nonmanifold_percent": "TOPO",
        }
        thresholds = {
            "rms_percent": (0.25, 0.75),
            "p95_percent": (0.50, 1.50),
            "area_percent": (1.0, 5.0),
            "nonmanifold_percent": (0.01, 0.10),
        }
        for key, label in self.quality_metric_labels.items():
            if not metrics or key not in metrics:
                label.setText(f"{abbreviations[key]}\n—")
                label.setProperty("qualityState", "")
                label.style().unpolish(label)
                label.style().polish(label)
                continue
            value = float(metrics[key])
            good, caution = thresholds[key]
            label.setText(f"{abbreviations[key]}\n{value:.2f}%")
            label.setProperty(
                "qualityState", "good" if value <= good else "caution" if value <= caution else "warning"
            )
            label.style().unpolish(label)
            label.style().polish(label)

    def _update_save_button_label(self) -> None:
        label = "Save current mesh"
        if self.loaded_triangle_count > 0 and self.source_faces is not None:
            reduction = 100.0 * (1.0 - len(self.source_faces) / self.loaded_triangle_count)
            if reduction >= 0.05:
                percentage = f"{reduction:.1f}".rstrip("0").rstrip(".")
                label += f" (-{percentage}%)"
        self.save_button.setText(label + "…")

    def _update_dimension_labels(self) -> None:
        unit = self.units.currentText()
        divisor = UNIT_MM[unit]
        self.dimensions_caption.setText(f"Dimensions ({unit}):")
        if self.source_poly is not None:
            size = np.asarray(bounds_size(self.source_poly.GetBounds())) / divisor
            self.dimensions.setText(fmt_size(tuple(size)))
        if self.dimension_drift_model_units is not None:
            drift = self.dimension_drift_model_units / divisor
            self.drift.setText(" x ".join(f"{v:.4f}" for v in drift) + f" {unit}")
        else:
            self.drift.setText("—")

    def _display_pending_result_statistics(self) -> bool:
        cast = getattr(self, "_envelope_poly", None)
        if cast is not None and getattr(self, "_envelope_settings_key", None) != self._imprint_preview_settings_key():
            cast = None
        solid = getattr(self, "_solidify_preview_context", None)
        ready_preview = (self.preview_poly is not None and self.preview_poly is not self.source_poly
                         and (solid is not None or not self.preview_dirty))
        poly = cast if cast is not None else self.preview_poly if ready_preview else None
        if poly is None:
            return False
        count = poly.GetNumberOfPolys()
        self.preview_count.setText(fmt_count(count))
        self.preview_vertices.setText(fmt_count(poly.GetNumberOfPoints()))
        self.estimated_file_size.setText(fmt_bytes(binary_stl_size(count)))
        self._update_optimization_ratio(count)
        metrics = (solid.get("metrics") if solid is not None else
                   getattr(self, "_last_quality_metrics", None)) if cast is None else None
        self._update_quality_metrics(metrics, measured=True)
        self.dimension_drift_model_units = np.abs(
            np.asarray(bounds_size(poly.GetBounds())) - np.asarray(bounds_size(self.source_poly.GetBounds())))
        self._update_dimension_labels()
        return True

    def _update_estimated_size(self) -> None:
        if self.source_faces is None or self.source_points is None:
            self.preview_count.setText("—")
            self.preview_vertices.setText("—")
            self.estimated_file_size.setText("—")
            return
        self._refresh_solidify_availability()
        self.target_display.setText(fmt_count(
            int(self.selection_target or self.target.value()) if self._selected_triangle_count() else self.target.value()))
        if self._display_pending_result_statistics():
            return
        if self.overview_mode and not self.tiled_mode:
            self.preview_count.setText("—")
            self.preview_vertices.setText("—")
            self.optimization_ratio.setText("—")
            self.estimated_file_size.setText("—")
            return
        working_triangles = (
            self.overview_total_triangles if self.tiled_mode else len(self.source_faces)
        )
        selected_triangles = self._selected_triangle_count()
        requested = self.selection_target if selected_triangles else self.target.value()
        target = min(
            max(1, int(requested or self.target.value())),
            selected_triangles or working_triangles,
        )
        estimated_triangles = (
            working_triangles - selected_triangles + target
            if selected_triangles
            else target
        )
        vertex_ratio = len(self.source_points) / max(1, working_triangles)
        estimated_vertices = max(4, round(estimated_triangles * vertex_ratio))
        self.preview_count.setText(f"~{fmt_count(estimated_triangles)}")
        self.preview_vertices.setText(f"~{fmt_count(estimated_vertices)}")
        self.target_display.setText(fmt_count(target))
        self._update_optimization_ratio(estimated_triangles)
        self._update_quality_metrics(
            mesh_quality_baseline(
                self.source_points,
                self.source_faces,
                estimated_triangles,
                self.algorithm.currentText(),
                self.smart_quality.currentText(),
            )
        )
        self.dimension_drift_model_units = None
        self.drift.setText("—")
        self.estimated_file_size.setText(fmt_bytes(binary_stl_size(estimated_triangles)))

    def _set_primary_action(self, pass_ready: bool, enabled: bool) -> None:
        self.pass_ready = pass_ready or getattr(self, "_solidify_preview_context", None) is not None
        self.preview_button.setText("Optimize")
        self.preview_button.setToolTip("Generate an optimization preview using the current settings.")
        self.preview_button.setEnabled(enabled)
        if hasattr(self, "cancel_preview_button"):
            has_unlocked_result = (
                self.preview_poly is not None
                and self.source_poly is not None
                and self.preview_poly is not self.source_poly
            )
            self.cancel_preview_button.setEnabled(has_unlocked_result and not self.busy)

    def _working_mesh_requires_closed_output(self) -> bool:
        if self.source_poly is None:
            return False
        marker = self.source_poly.GetFieldData().GetArray("ImprintClosedOutput")
        if marker is not None and marker.GetValue(0):
            return True
        if getattr(self, "_solidification_applied", False):
            # Solidifying a selection does not close unrelated open scan regions.
            from closed_reduction import is_closed_mesh
            return is_closed_mesh(self.source_points, self.source_faces)
        return False

    def _primary_optimize_action(self, _checked=False) -> None:
        if self._selected_triangle_count():
            self._selection_shortcut("optimize")
        else:
            self.preview()

    def _show_selection_primary_action(self) -> None:
        if not hasattr(self, "preview_button") or not self._selected_triangle_count():
            return
        self.preview_button.setText("Optimize")
        self.preview_button.setToolTip(
            "Optimize only the highlighted triangles using the current settings."
        )
        self.preview_button.setEnabled(not self.busy)

    def _cancel_optimized_preview(self, _checked=False) -> None:
        was_solidify = getattr(self, "_solidify_preview_context", None) is not None
        self._solidify_preview_context = None
        if (
            self.busy
            or self.source_poly is None
            or self.preview_poly is None
            or self.preview_poly is self.source_poly
        ):
            return
        was_selection_preview = self.preview_is_selection
        if self.simplified_actor is not None:
            self.renderer.RemoveActor(self.simplified_actor)
        self.preview_poly = None
        self.preview_is_selection = False
        self.simplified_actor = None
        self.pending_target = None
        self.running_target = None
        self.showing_original = True
        self.showing_previous = False
        self.toggle_mesh_button.setText("Show original")
        self.toggle_mesh_button.setEnabled(False)
        self.previous_mesh_button.setText("Show previous")
        self.dimension_drift_model_units = None
        self.drift.setText("â€”")
        self.preview_dirty = True
        self._update_estimated_size()
        self._set_primary_action(
            False, self.source_faces is not None and len(self.source_faces) > 1_000
        )
        self.save_button.setEnabled(self.mesh_modified)
        self._show_poly(self.source_poly, reset_camera=False)
        self.solidify_preview_button.setText("Solidify")
        if was_selection_preview:
            self._clear_selection(message=None)
            self.status.setText("Selection optimization cancelled • selection cleared")
        else:
            self.status.setText("Solid preview discarded" if was_solidify else "Optimization cancelled")

    def _optimization_script_settings(self):
        return {"kind": "optimize", "algorithm": CLI_ALGORITHM_KEYS.get(self.algorithm.currentText(), "auto"),
                "quality": self.smart_quality.currentText().lower(), "aggressiveness": 7.0,
                "gpu": self.gpu_preference, "processing": self.large_mesh_mode,
                "workers": self.geometry_processes, "auto_tiling": self.auto_tiling_enabled,
                "memory_percent": self.large_mesh_memory_percent,
                "preserve_closed": self._working_mesh_requires_closed_output()}

    def _apply_optimized_pass(self, _checked=False) -> None:
        solid = getattr(self, "_solidify_preview_context", None)
        if solid is not None and not self.busy:
            from solidify import combine_selected_solid
            poly = combine_selected_solid(solid["source_points"], solid["source_faces"],
                                           solid.get("mask"), solid["solid_poly"])
            points, faces = polydata_arrays(poly)
            # History should capture the unchanged source, not an uncommitted
            # preview that would reappear on top of it after Undo.
            self._cancel_optimized_preview()
            self._solidify_preview_context = None
            self._set_working_mesh(poly, points, faces, "Applied solidification", committed_optimization=False,
                                   script_pass=solid.get("script_step"),
                                   script_omission="Selection solidification" if solid.get("script_step") is None else None,
                                   solidification_applied=True)
            self.solidify_preview_button.setText("Solidify")
            self.solidify_toggle.setChecked(False)
            self.optimization_toggle.setChecked(True)
            self._refresh_workflow_actions()
            return
        if (
            self.busy
            or self.preview_poly is None
            or self.source_poly is None
            or self.preview_poly is self.source_poly
        ):
            return
        points, faces = polydata_arrays(self.preview_poly)
        applied_poly = self.preview_poly
        applied_actor = self.simplified_actor
        selection_pass = self.preview_is_selection
        script_pass = None
        if not selection_pass:
            script_pass = dict(getattr(self, "_preview_script_step", None) or self._optimization_script_settings())
        if selection_pass:
            self.preview_poly = None
            self.simplified_actor = None
            self.preview_is_selection = False
        self._set_working_mesh(
            applied_poly,
            points,
            faces,
            "Applied optimization",
            preserve_target=selection_pass,
            cached_actor=applied_actor,
            committed_optimization=True,
            script_pass=script_pass,
            script_omission="Selection optimization" if selection_pass else None,
        )

    def _generate_optimization_script(self, _checked=False) -> None:
        if self.source_path is None or not self.mesh_modified:
            return
        script_source = self.source_path
        if self.script_base_poly is not None:
            try:
                from mesh_export import write_polygon_mesh
                handle, filename = tempfile.mkstemp(prefix=self.source_path.stem + "-workflow-input-", suffix=".ply",
                                                    dir=self.source_path.parent)
                os.close(handle)
                script_source = Path(filename)
                points, faces = polydata_arrays(self.script_base_poly)
                write_polygon_mesh(points, faces, script_source, merge=False)
            except (OSError, ValueError) as error:
                if script_source != self.source_path:
                    script_source.unlink(missing_ok=True)
                QMessageBox.warning(self, APP_NAME, f"Could not save the workflow input mesh.\n{error}")
                return
        script_steps = list(self.optimization_script_passes)
        if self.export_calibration_transform is not None:
            script_steps.append({"kind": "convert", "export_transform": self.export_calibration_transform.tolist()})
        generated = render_optimization_script(
            script_source,
            script_steps,
            self.optimization_script_omissions,
        )
        suggested = self.source_path.with_name(f"{self.source_path.stem}_meshmill_passes.ps1")
        script_dialog = QDialog(self)
        script_dialog.setWindowTitle("PowerShell script")
        script_dialog.resize(max(760, self.width() * 2 // 3), max(520, self.height() * 2 // 3))
        layout = QVBoxLayout(script_dialog)
        editor = QPlainTextEdit(script_dialog)
        editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        editor.setPlainText(generated)
        editor.setToolTip("Edit the commands before copying or saving the script.")
        layout.addWidget(editor, 1)

        actions = QHBoxLayout()
        copy_button = QPushButton("Copy", script_dialog)
        reset_button = QPushButton("Reset", script_dialog)
        save_button = QPushButton("Save As…", script_dialog)
        actions.addWidget(copy_button)
        actions.addWidget(reset_button)
        actions.addStretch(1)
        actions.addWidget(save_button)
        close_button = QPushButton("Close", script_dialog)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(editor.toPlainText()))
        reset_button.clicked.connect(lambda: editor.setPlainText(generated))

        def save_script() -> None:
            path, _ = QFileDialog.getSaveFileName(
                script_dialog,
                "Save PowerShell script",
                os.fspath(suggested),
                "PowerShell script (*.ps1)",
            )
            if not path:
                return
            destination = Path(path)
            if destination.suffix.lower() != ".ps1":
                destination = destination.with_suffix(".ps1")
            try:
                destination.write_text(editor.toPlainText(), encoding="utf-8-sig", newline="\r\n")
            except OSError as exc:
                QMessageBox.critical(
                    script_dialog,
                    APP_NAME,
                    "Could not save the PowerShell script.\n\n" + str(exc),
                )
                return
            self.status.setText(f"PowerShell script saved: {destination}")

        save_button.clicked.connect(save_script)
        close_button.clicked.connect(script_dialog.accept)
        center = self.geometry().center()
        script_dialog.move(
            center.x() - script_dialog.width() // 2,
            center.y() - script_dialog.height() // 2,
        )
        script_dialog.exec()

    def _toggle_mesh(self) -> None:
        if self.source_poly is None or self.preview_poly is None:
            return
        self.showing_previous = False
        self.previous_mesh_button.setText("Show previous")
        if self.showing_original:
            self.toggle_mesh_button.setText("Show original")
            self.showing_original = False
            self._show_poly(self.preview_poly, reset_camera=False)
        else:
            self.toggle_mesh_button.setText("Show optimized")
            self.showing_original = True
            self._show_poly(self.source_poly, reset_camera=False)
        self._update_action_row_visibility()

    def _reset_previous_view(self) -> None:
        self.showing_previous = False
        self.previous_view_poly = None
        self.previous_view_actor = None
        if hasattr(self, "previous_mesh_button"):
            self.previous_mesh_button.setText("Show previous")
            previous = self.undo_stack[-1].poly if self.undo_stack else None
            current = (
                self.source_poly
                if self.showing_original or self.preview_poly is None
                else self.preview_poly
            )
            self.previous_mesh_button.setEnabled(
                previous is not None and previous is not current and not self.busy
            )
        self._update_action_row_visibility()

    def _toggle_previous_mesh(self) -> None:
        if self.source_poly is None or not self.undo_stack:
            return
        if self.showing_previous:
            self.showing_previous = False
            self.previous_mesh_button.setText("Show previous")
            current = (
                self.source_poly
                if self.showing_original or self.preview_poly is None
                else self.preview_poly
            )
            self._show_poly(current, reset_camera=False)
            previous = self.undo_stack[-1].poly if self.undo_stack else None
            self.previous_mesh_button.setEnabled(previous is not current and not self.busy)
            self._update_action_row_visibility()
            return

        previous = self.undo_stack[-1].poly
        if self.previous_view_poly is not previous or self.previous_view_actor is None:
            self.previous_view_poly = previous
            self._show_poly(previous, reset_camera=False)
            self.previous_view_actor = self.active_actor
        else:
            self._set_actor_display_mode(self.previous_view_actor)
            self._show_cached_actor(self.previous_view_actor, reset_camera=False)
        self.showing_previous = True
        self.previous_mesh_button.setText("Show current")
        current = (
            self.source_poly
            if self.showing_original or self.preview_poly is None
            else self.preview_poly
        )
        self.previous_mesh_button.setEnabled(current is not previous and not self.busy)
        self._update_action_row_visibility()

    def _capture_and_exit(self, path: Path) -> None:
        # Verification images document the settled application state. Density coloring runs in a
        # separate worker, so wait for it instead of capturing the progress panel.
        if self.busy:
            QTimer.singleShot(250, lambda: self._capture_and_exit(path))
            return
        if self._display_mode() == "Density":
            density_poly, _points, _faces = self._active_density_mesh()
            if density_poly is not None and id(density_poly) not in self.density_cache:
                self._request_density_analysis(density_poly)
                QTimer.singleShot(250, lambda: self._capture_and_exit(path))
                return
            if density_poly is not None:
                self._show_poly(density_poly, reset_camera=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.vtk_widget.GetRenderWindow().Render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(self.vtk_widget.GetRenderWindow())
        capture.SetInputBufferTypeToRGBA()
        capture.ReadFrontBufferOff()
        capture.Update()
        rendered = capture.GetOutput()
        image_width, image_height, _ = rendered.GetDimensions()
        rgba = vtk_to_numpy(rendered.GetPointData().GetScalars()).reshape(
            image_height, image_width, 4
        )
        rgba = np.ascontiguousarray(np.flipud(rgba))
        viewport_image = QImage(
            rgba.data,
            image_width,
            image_height,
            image_width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()

        screenshot = self.grab()
        painter = QPainter(screenshot)
        viewport_position = self.vtk_widget.mapTo(self, QPoint(0, 0))
        painter.drawImage(
            QRect(
                viewport_position.x(),
                viewport_position.y(),
                self.vtk_widget.width(),
                self.vtk_widget.height(),
            ),
            viewport_image,
        )
        for overlay in (
            self.view_pad,
            self.orientation_indicator,
            self.scale_bar,
            self.density_legend,
            self.loading_frame,
            self.crop_info_frame,
            self.measure_info_frame,
            self.loupe_frame,
        ):
            if overlay is not None and overlay.isVisible():
                position = overlay.mapTo(self, QPoint(0, 0))
                painter.drawPixmap(position, overlay.grab())
        painter.end()
        if not screenshot.save(os.fspath(path), "PNG"):
            QMessageBox.critical(self, APP_NAME, f"Could not save verification screenshot:\n{path}")
            return
        QApplication.instance().quit()

    def _capture_selection_and_exit(self, path: Path) -> None:
        """Build a stable retained-selection state for release verification images."""
        if self.source_poly is None:
            self._capture_and_exit(path)
            return
        self._show_poly(self.source_poly, reset_camera=False)
        self.showing_original = True
        width = max(1, self.vtk_widget.width())
        height = max(1, self.vtk_widget.height())
        self.crop_points = [
            QPoint(round(width * 0.46), round(height * 0.30)),
            QPoint(round(width * 0.70), round(height * 0.30)),
            QPoint(round(width * 0.70), round(height * 0.67)),
            QPoint(round(width * 0.46), round(height * 0.67)),
        ]
        self._prepare_polygon_crop()
        QTimer.singleShot(250, lambda: self._capture_and_exit(path))

    def _capture_loupe_and_exit(self, path: Path) -> None:
        """Show the pointer magnifier in a deterministic position for release images."""
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        width, height = self.vtk_widget.GetRenderWindow().GetSize()
        interactor.SetEventInformation(max(1, width // 2), max(1, height // 2), 0, 0, "0", 0, None)
        self._show_loupe(True)
        QTimer.singleShot(250, lambda: self._capture_and_exit(path))

    def _precache_source_analysis(self) -> None:
        if getattr(self, "_closing", False):
            return
        if self.busy:
            QTimer.singleShot(500, self._precache_source_analysis)
            return
        if (self.source_points is None or self.source_faces is None or self.tiled_mode
                or not is_gpu_guided_algorithm(self.algorithm.currentText())):
            return
        points, faces = self.source_points, self.source_faces
        cached = getattr(self, "_probe_statistics_job", None)
        if cached is not None and cached[0] is points and cached[1] is faces:
            return
        # Bound speculative allocations and use the existing single CPU worker.
        if available_physical_memory() < min(len(faces), 1_000_000) * 256:
            return
        self._probe_statistics_job = (points, faces,
                                      self.executor.submit(probe_source_statistics, points, faces))

    def _cached_probe_statistics(self, points, faces):
        cached = getattr(self, "_probe_statistics_job", None)
        if cached is not None and cached[0] is points and cached[1] is faces and cached[2].done():
            try:
                return cached[2].result()
            except Exception:  # noqa: BLE001 - optional preparation falls back to the normal path
                return None
        return None

    def _cached_source_measurements(self, points, faces):
        cached = self.source_measurement_cache
        if cached is not None and cached[0] is points and cached[1] is faces:
            return cached[2]
        return None

    def _show_poly(self, poly: vtkPolyData, reset_camera: bool = True, prepared=None) -> None:
        if poly is self.source_poly or poly is self.preview_poly:
            self.showing_previous = False
            self.showing_original = poly is self.source_poly
            self._update_action_row_visibility()
        self._refresh_imprint_plan()
        if (getattr(self, "_envelope_actor", None) is not None
                and self.source_poly is not getattr(self, "_envelope_source_poly", None)):
            self._clear_envelope()
        if poly is self.source_poly and self.source_actor is not None:
            actor = self.source_actor
        elif poly is self.preview_poly and self.simplified_actor is not None:
            actor = self.simplified_actor
        else:
            if prepared is None:
                prepared = prepare_viewport_poly(poly)
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(prepared)
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.76, 0.80, 0.84)
            actor.GetProperty().SetInterpolationToPhong()
            actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.12)
            actor.GetProperty().SetEdgeVisibility(False)
            if poly is self.source_poly:
                self.source_actor = actor
            elif poly is self.preview_poly:
                self.simplified_actor = actor
            elif poly is self.previous_view_poly:
                self.previous_view_actor = actor
        if (
            self._display_mode() == "Density"
            and id(poly) not in self.density_cache
        ):
            self.pending_density_poly = poly
            self.density_legend.setVisible(self._overlay_is_visible("density"))
            self.density_legend.raise_()
            # Establish the same framing as other modes before asynchronous coloring.
            self._set_actor_display_mode(actor)
            self._show_cached_actor(actor, reset_camera)
            QTimer.singleShot(0, lambda requested=poly: self._request_density_analysis(requested))
            return
        self.pending_density_poly = None
        self._set_actor_display_mode(actor)
        self._show_cached_actor(actor, reset_camera)

    def _show_cached_actor(self, actor: vtkActor | None, reset_camera: bool = False) -> None:
        if actor is None:
            return
        self._hide_loupe()
        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.preview_actor = actor
        self.active_actor = actor
        self._request_orientation_miniature()
        self._sync_imprint_split_widget()
        envelope = getattr(self, "_envelope_actor", None)
        if envelope is not None:
            self.renderer.AddActor(envelope)
            self._update_envelope_overlay()
        if (
            actor is self.source_actor
            and self.crop_selection_actor is not None
            and self.selected_face_mask is not None
        ):
            self.renderer.AddActor(self.crop_selection_actor)
            actor.GetProperty().SetOpacity(0.12)
        if reset_camera:
            self.renderer.ResetCamera()
            self.renderer.GetActiveCamera().Zoom(1.28)
            self.renderer.ResetCameraClippingRange()
        if actor is self.source_actor and getattr(self, "_opening_front_pending", False):
            self._opening_front_pending = False
            camera = self.renderer.GetActiveCamera()
            opening = getattr(self, "_reload_opening_camera", None)
            if opening is not None:
                camera.DeepCopy(opening)
            else:
                self._original_opening_camera = camera.NewInstance()
                self._original_opening_camera.DeepCopy(camera)
            self._reload_opening_camera = None
            self.renderer.ResetCameraClippingRange()
            self._establish_opening_front()
        self._update_measure_overlay(render=False)
        # Cached history can have different depth extents. Refresh near/far
        # planes before its first frame while keeping the user's camera pose.
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()
        resident = inspect_resident_vertices(actor)
        if resident is not None:
            self.resident_vertex_buffers[id(actor)] = resident
        else:
            self.resident_vertex_buffers.pop(id(actor), None)
        density_visible = self._display_mode() == "Density" and self._overlay_is_visible("density")
        self.density_legend.setVisible(density_visible)
        if density_visible:
            self.density_legend.raise_()

    def _show_loupe(self, visible: bool) -> None:
        if not visible:
            self._hide_loupe()
            return
        if self.active_actor is None:
            return
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        x, y = interactor.GetEventPosition()
        width, height = self.vtk_widget.GetRenderWindow().GetSize()
        if width <= 0 or height <= 0:
            return
        if self.loupe_frame is None:
            diameter = 360
            self.loupe_frame = QFrame(self.vtk_widget)
            self.loupe_frame.setFixedSize(diameter, diameter)
            self.loupe_frame.setMask(QRegion(0, 0, diameter, diameter, QRegion.RegionType.Ellipse))
            self.loupe_frame.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.loupe_frame.setStyleSheet(
                "QFrame { border: 3px solid #2563FF; border-radius: 180px; background: #080b10; }"
            )
            self.loupe_label = QLabel(self.loupe_frame)
            self.loupe_label.setGeometry(3, 3, diameter - 6, diameter - 6)
            self.loupe_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.vtk_widget.GetRenderWindow().Render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(self.vtk_widget.GetRenderWindow())
        capture.SetInputBufferTypeToRGBA()
        capture.ReadFrontBufferOff()
        capture.Update()
        rendered = capture.GetOutput()
        image_width, image_height, _ = rendered.GetDimensions()
        rgba = vtk_to_numpy(rendered.GetPointData().GetScalars()).reshape(image_height, image_width, 4)
        rgba = np.ascontiguousarray(np.flipud(rgba))
        image = QImage(
            rgba.data, image_width, image_height, image_width * 4, QImage.Format.Format_RGBA8888
        ).copy()

        diameter = self.loupe_frame.width()
        inner = diameter - 6
        crop_size = max(20, inner // 4)
        image_y = image_height - y
        crop_left = max(0, min(image_width - crop_size, x - crop_size // 2))
        crop_top = max(0, min(image_height - crop_size, image_y - crop_size // 2))
        crop = image.copy(crop_left, crop_top, crop_size, crop_size)
        pixmap = QPixmap.fromImage(crop).scaled(
            inner,
            inner,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.loupe_label.setPixmap(pixmap)

        margin = 10
        qt_y = height - y
        left = min(max(margin, x - diameter // 2), max(margin, width - diameter - margin))
        top = min(max(margin, qt_y - diameter // 2), max(margin, height - diameter - margin))
        self.loupe_frame.move(left, top)
        self.loupe_frame.show()
        self.loupe_frame.raise_()

    def _hide_loupe(self) -> None:
        if self.loupe_frame is not None:
            self.loupe_frame.hide()

    def _set_actor_display_mode(self, actor: vtkActor) -> None:
        mode = self._display_mode()
        mapper = actor.GetMapper()
        mapper.ScalarVisibilityOff()
        mapper.SetScalarModeToDefault()
        actor.GetProperty().SetColor(0.76, 0.80, 0.84)
        actor.GetProperty().SetAmbient(0.0)
        actor.GetProperty().SetDiffuse(1.0)
        actor.GetProperty().SetInterpolationToPhong()
        if mode == "Wireframe":
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetLineWidth(1.0)
            actor.GetProperty().SetPointSize(1.0)
            actor.GetProperty().SetEdgeVisibility(False)
        elif mode == "Vertices":
            actor.GetProperty().SetRepresentationToPoints()
            actor.GetProperty().SetPointSize(2.5)
            actor.GetProperty().SetEdgeVisibility(False)
        elif mode == "Density":
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetEdgeVisibility(False)
            actor.GetProperty().SetAmbient(0.72)
            actor.GetProperty().SetDiffuse(0.28)
            actor.GetProperty().SetInterpolationToFlat()
            if actor is self.source_actor:
                poly = self.source_poly
            elif actor is self.previous_view_actor:
                poly = self.previous_view_poly
            else:
                poly = self.preview_poly
            density_colors = self.density_cache.get(id(poly)) if poly is not None else None
            if density_colors is not None:
                # Accept the former (point, cell) cache shape while restoring older history in
                # development sessions; new entries store only the smaller point-color array.
                colors = density_colors[0] if isinstance(density_colors, tuple) else density_colors
                render_poly = mapper.GetInput()
                if render_poly is None:
                    mapper.Update()
                    render_poly = mapper.GetInput()
                point_data = render_poly.GetPointData()
                # Theme changes produce a fresh palette for the same mesh. Replace the
                # existing VTK array rather than leaving the first palette attached.
                point_data.RemoveArray("PointDensityColors")
                vtk_colors = numpy_to_vtk(colors, deep=True)
                vtk_colors.SetName("PointDensityColors")
                point_data.AddArray(vtk_colors)
                point_data.SetActiveScalars("PointDensityColors")
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray("PointDensityColors")
                mapper.SetColorModeToDirectScalars()
                mapper.InterpolateScalarsBeforeMappingOff()
                mapper.ScalarVisibilityOn()
                mapper.Modified()
        else:
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetLineWidth(1.0)
            actor.GetProperty().SetPointSize(1.0)
            actor.GetProperty().SetEdgeVisibility(False)

    def _active_density_mesh(self):
        if self.showing_previous and self.previous_view_poly is not None:
            return self.previous_view_poly, *polydata_arrays(self.previous_view_poly)
        if self.active_actor is self.simplified_actor and self.preview_poly is not None:
            return self.preview_poly, *polydata_arrays(self.preview_poly)
        if self.source_poly is not None and self.source_points is not None and self.source_faces is not None:
            return self.source_poly, self.source_points, self.source_faces
        return None, None, None

    def _request_density_analysis(self, requested_poly: vtkPolyData | None = None) -> None:
        if self._display_mode() != "Density":
            return
        if requested_poly is self.source_poly:
            poly, points, faces = self.source_poly, self.source_points, self.source_faces
        elif requested_poly is self.preview_poly and self.preview_poly is not None:
            poly, points, faces = self.preview_poly, *polydata_arrays(self.preview_poly)
        elif requested_poly is self.previous_view_poly and self.previous_view_poly is not None:
            poly, points, faces = self.previous_view_poly, *polydata_arrays(self.previous_view_poly)
        else:
            poly, points, faces = self._active_density_mesh()
        if poly is None or points is None or faces is None:
            return
        if id(poly) in self.density_cache:
            self._show_poly(poly, reset_camera=False)
            return
        if self.busy:
            QTimer.singleShot(250, lambda requested=poly: self._request_density_analysis(requested))
            return
        self.density_generation += 1
        token = self.density_generation
        poly_key = id(poly)
        self._begin_mesh_operation(
            "Analyzing point density",
            f"Mapping {fmt_count(len(points))} vertices across {fmt_count(len(faces))} triangles",
        )
        cancel_event = self.operation_cancel_event
        viewport_color = QColor(self._viewport_theme()["viewport"])
        density_base = (
            viewport_color.red(), viewport_color.green(), viewport_color.blue()
        )
        custom_gradient = self._custom_density_colors()
        if custom_gradient is not None:
            density_base, density_dense = custom_gradient
        else:
            density_dense = None
        progress_path = self._start_operation_progress(token, "Measuring triangle density")
        _write_process_progress(progress_path, "Measuring triangle density", 0, len(faces))
        density_gpu_preference = self.gpu_preference

        def job():
            try:
                return (
                    point_density_colors(
                        points,
                        faces,
                        cancel_event,
                        progress=lambda stage, completed, total: _write_process_progress(
                            progress_path, stage, completed, total
                        ),
                        base_color=density_base,
                        dense_color=density_dense,
                        gpu_preference=density_gpu_preference,
                    ),
                    None,
                )
            except OperationCancelled:
                return None, "cancelled"
            except Exception:  # noqa: BLE001 - worker boundary reports failures through the result payload
                return None, traceback.format_exc()

        future = self.executor.submit(job)
        self.operation_future = future
        future.add_done_callback(
            lambda f: self.density_bridge.done.emit(token, poly_key, *f.result())
        )

    def _finish_density_analysis(self, token: int, poly_key: int, colors, error) -> None:
        if token != self.density_generation:
            return
        self._end_mesh_operation()
        if error:
            self.status.setText("Point density analysis failed")
            QMessageBox.critical(self, APP_NAME, error)
            return
        self.density_cache[poly_key] = colors
        if self._display_mode() == "Density":
            requested = self.pending_density_poly
            if requested is not None and id(requested) == poly_key:
                self._show_poly(requested, reset_camera=False)
            self.status.setText("Ready")

    def _apply_display_mode(self, mode: str) -> None:
        self.settings.setValue("display/type", mode)
        self.density_legend.setVisible(mode == "Density" and self._overlay_is_visible("density"))
        if mode == "Density":
            self.density_legend.raise_()
            poly, _points, _faces = self._active_density_mesh()
            if poly is not None:
                self._show_poly(poly, reset_camera=False)
            return
        self.pending_density_poly = None
        for actor in (self.source_actor, self.simplified_actor, self.previous_view_actor):
            if actor is not None:
                self._set_actor_display_mode(actor)
        shown = (
            self.previous_view_actor
            if self.showing_previous
            else self.source_actor if self.showing_original else self.simplified_actor
        )
        self._show_cached_actor(shown, reset_camera=False)

    def _save_shortcut_activated(self) -> None:
        if self.save_button.isEnabled():
            self.save_file()

    def _check_for_updates(self, force: bool = False) -> None:
        if getattr(self, "_closing", False):
            return
        if not force and not self.update_checks_enabled:
            return
        if self.update_future is not None:
            return
        self._update_prompt_requested = not force
        if force:
            self._pending_update_prompt = ""
            self.update_prompt_timer.stop()
        self.update_check_status = "Checking…"
        self.version_label.setText("Checking for updates…")
        self.version_label.setEnabled(False)
        future = self.update_executor.submit(fetch_latest_release)
        self.update_future = future

        def completed(result_future) -> None:
            try:
                release = result_future.result()
                release_url = release["release_url"]
                error = None
            except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
                release_url = release = None
                error = exc
            self.update_bridge.done.emit(release, release_url, error)

        future.add_done_callback(completed)

    def _finish_update_check(self, release, release_url, error) -> None:
        self.update_future = None
        prompt_requested = self._update_prompt_requested
        self._update_prompt_requested = False
        if getattr(self, "_closing", False):
            return
        version = release.get("version") if isinstance(release, dict) else None
        if error is not None or not version or not release_url:
            self.update_check_status = "Could not check GitHub"
            self._update_update_action()
            return
        self.latest_release_version = str(version)
        self.latest_release_url = str(release_url)
        self.latest_release = release
        if version_key(self.latest_release_version) > version_key(APP_VERSION):
            self.update_check_status = f"Version {self.latest_release_version} is available"
        else:
            self.update_check_status = "MeshMill is up to date"
        self._update_update_action()
        if (prompt_requested and self.update_checks_enabled and self._update_available()
                and self.latest_release_version != self.skipped_release_version
                and self.latest_release_version not in self._prompted_update_versions):
            self._pending_update_prompt = self.latest_release_version
            self.update_prompt_timer.start()

    def _update_available(self) -> bool:
        return (bool(self.latest_release_url)
                and version_key(self.latest_release_version) > version_key(APP_VERSION))

    def _offer_startup_update(self) -> None:
        if self._ready_auto_installer is not None:
            if not self._auto_install_release_allowed() or not self.update_checks_enabled:
                self._automatic_update_in_progress = False
                self.update_prompt_timer.stop()
            elif self._can_auto_install_update():
                installer = self._ready_auto_installer
                self._ready_auto_installer = None
                self.update_prompt_timer.stop()
                self._finish_update_download(installer, None)
            return
        version = self._pending_update_prompt
        if (getattr(self, "_closing", False) or not self.update_checks_enabled
                or not version or not self._update_available()
                or version != self.latest_release_version
                or version == self.skipped_release_version
                or version in self._prompted_update_versions):
            self._pending_update_prompt = ""
            self.update_prompt_timer.stop()
            return
        if self.busy or QApplication.activeModalWidget() is not None:
            return
        automatic = self._auto_install_release_allowed()
        if automatic and not self._can_auto_install_update():
            return
        self._pending_update_prompt = ""
        self.update_prompt_timer.stop()
        self._install_latest_release(automatic=automatic)

    def _auto_install_release_allowed(self) -> bool:
        if not self.auto_install_updates:
            return False
        if self.auto_install_scope == "all":
            return True
        current = (*version_key(APP_VERSION), 0, 0)
        latest = (*version_key(self.latest_release_version), 0, 0)
        return current[:2] == latest[:2]

    def _can_auto_install_update(self) -> bool:
        return (not getattr(self, "_closing", False) and self.update_checks_enabled
                and self._auto_install_release_allowed() and not self.busy and not self.mesh_modified
                and QApplication.activeModalWidget() is None)

    def _update_update_action(self) -> None:
        update_available = (
            bool(self.latest_release_url)
            and version_key(self.latest_release_version) > version_key(APP_VERSION)
        )
        self.version_label.setEnabled(True)
        if update_available:
            self.version_label.setText(f"Upgrade to {self.latest_release_version}")
            installable = update_asset_for_current_platform(self.latest_release)
            action = (
                "download, verify, install, and reopen MeshMill"
                if installable
                else "open the release downloads"
            )
            self.version_label.setToolTip(
                f"Click to {action}. Right-click to copy the installed version."
            )
        elif self.update_check_status == "MeshMill is up to date":
            self.version_label.setText(f"MeshMill {APP_VERSION}  •  Up to date")
            self.version_label.setToolTip(
                "This is the newest MeshMill release. Click to check again. "
                "Right-click to copy the installed version."
            )
        else:
            self.version_label.setText(f"MeshMill {APP_VERSION}")
            self.version_label.setToolTip(
                "Click to check GitHub for a newer MeshMill release. "
                "Right-click to copy the installed version. This does not enable automatic checks."
            )

    def _version_action_clicked(self) -> None:
        if (
            self.latest_release_url
            and version_key(self.latest_release_version) > version_key(APP_VERSION)
        ):
            self._install_latest_release()
            return
        self._check_for_updates(force=True)

    def _copy_version(self) -> None:
        QApplication.clipboard().setText(APP_VERSION)
        self.status.setText(f"Version {APP_VERSION} copied")

    def _open_latest_release(self) -> None:
        if self.latest_release_url:
            QDesktopServices.openUrl(QUrl(self.latest_release_url))

    def _install_latest_release(self, automatic: bool = False) -> None:
        if (getattr(self, "_closing", False) or not self._update_available()
                or self.update_install_future is not None or self.busy):
            return
        asset = update_asset_for_current_platform(self.latest_release)
        installable = asset is not None and getattr(sys, "frozen", False)
        if automatic and (not installable or not self._can_auto_install_update()):
            return
        self._prompted_update_versions.add(self.latest_release_version)
        self._pending_update_prompt = ""
        self.update_prompt_timer.stop()
        choice = "install" if automatic else self._confirm_update_install(installable)
        if choice == "skip":
            self.skipped_release_version = self.latest_release_version
            self.settings.setValue("updates/skipped_version", self.skipped_release_version)
            return
        if choice != "install":
            return
        if not installable:
            self._open_latest_release()
            return
        self._automatic_update_in_progress = automatic
        update_dir = (
            Path(tempfile.gettempdir()) / "MeshMill" / "updates" / self.latest_release_version
        )
        update_dir.mkdir(parents=True, exist_ok=True)
        destination = update_dir / str(asset["name"])
        self.version_label.setText("Downloading update...")
        self.version_label.setEnabled(False)
        self.update_check_status = "Downloading update..."
        future = self.update_executor.submit(download_verified_update, asset, destination)
        self.update_install_future = future

        def completed(result_future) -> None:
            try:
                installer = result_future.result()
                error = None
            except (HTTPError, URLError, OSError, ValueError) as exc:
                installer = None
                error = exc
            self.update_bridge.install_done.emit(installer, error)

        future.add_done_callback(completed)

    def _confirm_update_install(self, installable: bool) -> str:
        prompt = QMessageBox(QApplication.activeModalWidget() or self)
        prompt.setWindowTitle("MeshMill update")
        prompt.setIcon(QMessageBox.Icon.Information)
        prompt.setText(
            (f"Install MeshMill {self.latest_release_version}?\n\n"
             "The package will be verified, MeshMill will close, and the open STL "
             "will reopen after the upgrade."
             if installable else
             f"MeshMill {self.latest_release_version} is available. Open the release downloads?")
            + (
                "\n\nThe current mesh has unapplied or unsaved changes. Save it before "
                "continuing if you want to keep them."
                if self.mesh_modified
                else ""
            )
        )
        install = prompt.addButton(
            "Install now" if installable else "Open downloads", QMessageBox.ButtonRole.AcceptRole
        )
        remind = prompt.addButton("Remind me next time", QMessageBox.ButtonRole.RejectRole)
        skip = prompt.addButton("Skip this version", QMessageBox.ButtonRole.NoRole)
        remind.setToolTip("Ask again on the next app launch if startup checks are enabled.")
        skip.setToolTip("Stop startup reminders for this version. Settings can still install it.")
        prompt.setDefaultButton(remind)
        prompt.setEscapeButton(remind)
        prompt.exec()
        if prompt.clickedButton() == install:
            return "install"
        if prompt.clickedButton() == skip:
            return "skip"
        return "remind"

    def _finish_update_download(self, installer: Path | None, error) -> None:
        self.update_install_future = None
        if error is not None or installer is None:
            self._automatic_update_in_progress = False
            self.update_check_status = "Update download failed"
            self._update_update_action()
            QMessageBox.warning(self, APP_NAME, f"The update was not installed.\n\n{error}")
            return
        if self._automatic_update_in_progress and not self._can_auto_install_update():
            self._ready_auto_installer = installer
            self.update_check_status = "Update downloaded; waiting for saved work and idle state"
            self.update_prompt_timer.start()
            return
        reopen = os.fspath(self.source_path) if self.source_path is not None else ""
        if platform.system().lower() == "windows":
            command = [
                os.fspath(installer),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/CLOSEAPPLICATIONS",
                "/REOPEN=1",
                f"/REOPENFILE={reopen}",
            ]
        elif platform.system().lower() == "darwin":
            command = ["open", os.fspath(installer)]
        else:
            command = [os.fspath(installer)]
        try:
            subprocess.Popen(command, close_fds=True)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"The installer could not start.\n\n{exc}")
            self._update_update_action()
            return
        QApplication.quit()

    def _anonymous_bug_diagnostics(self) -> str:
        """Return reviewable diagnostics without paths, identifiers, or raw settings."""
        loaded_triangles = (
            self.overview_total_triangles
            if self.tiled_mode
            else len(self.source_faces)
            if self.source_faces is not None
            else 0
        )
        gpu_selection = "Automatic" if self.gpu_preference == "system" else "Manually selected"
        geometry_workers = "Automatic" if self.geometry_processes == 0 else str(self.geometry_processes)
        installation = "Packaged application" if getattr(sys, "frozen", False) else "Python source run"
        cpu_model = platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "Unknown")
        selected_gpu = next(
            (
                device
                for device in self.cuda_device_list
                if device["pci_bus_id"] == self.gpu_preference
            ),
            None,
        )
        if selected_gpu is not None:
            gpu_model = str(selected_gpu["name"])
            gpu_memory = fmt_bytes(int(selected_gpu["total_memory"]))
        elif len(self.cuda_device_list) == 1:
            gpu_model = str(self.cuda_device_list[0]["name"])
            gpu_memory = fmt_bytes(int(self.cuda_device_list[0]["total_memory"]))
        elif self.cuda_device_list:
            gpu_model = ", ".join(str(device["name"]) for device in self.cuda_device_list)
            gpu_memory = "Automatic selection"
        else:
            gpu_model = "No CUDA GPU detected"
            gpu_memory = "Not available"

        lines = [
            f"MeshMill version: {APP_VERSION}",
            f"Installation: {installation}",
            f"Operating system: {platform.system()}",
            f"CPU model: {cpu_model}",
            f"Logical processors: {os.cpu_count() or 'Unknown'}",
            f"GPU-guided algorithms available: {'Yes' if gpu_backend_supported(self.gpu_preference) else 'No'}",
            f"GPU selection: {gpu_selection}",
            f"GPU model: {gpu_model}",
            f"GPU memory: {gpu_memory}",
            f"Algorithm: {self.algorithm.currentText()}",
            f"Smart quality: {self.smart_quality.currentText()}",
            f"Display mode: {self._display_mode()}",
            f"Model units: {self.units.currentText()}",
            f"Large-mesh mode: {self.large_mesh_mode}",
            f"Geometry workers: {geometry_workers}",
            f"Memory budget: {self.large_mesh_memory_percent}%",
            f"Model loaded: {'Yes' if self.source_faces is not None else 'No'}",
            f"Loaded triangles: {loaded_triangles:,}" if loaded_triangles else "Loaded triangles: none",
        ]
        if self.last_operation_runtime:
            runtime_total = sum(self.last_operation_runtime.values())
            runtime_parts = []
            for name in (
                "Preparation",
                "Reading and indexing",
                "Surface analysis",
                "Mesh reduction",
                "Result checks",
                "Output assembly",
            ):
                seconds = self.last_operation_runtime.get(name, 0.0)
                if seconds <= 0.005:
                    continue
                percentage = round(100.0 * seconds / runtime_total) if runtime_total else 0
                runtime_parts.append(f"{name} {percentage}% ({seconds:.1f} s)")
            if runtime_parts:
                lines.append(
                    f"Last operation runtime: {self.last_operation_name} {runtime_total:.1f} s"
                )
                lines.append("Runtime breakdown: " + "; ".join(runtime_parts))
        if self.runtime_operation_count:
            aggregate_parts = []
            for name in self._runtime_category_order():
                seconds = self.runtime_aggregate_seconds.get(name, 0.0)
                count = self.runtime_aggregate_counts.get(name, 0)
                if seconds > 0.005:
                    aggregate_parts.append(f"{name} {count}x ({seconds:.1f} s)")
            lines.append(
                f"Runtime session: {self.runtime_operation_count} completed operations, "
                f"{self.runtime_total_seconds:.1f} s total"
            )
            if aggregate_parts:
                lines.append("Runtime aggregates: " + "; ".join(aggregate_parts))
        if self.source_points is not None:
            lines.append(f"Loaded vertices: {len(self.source_points):,}")
        if self.source_poly is not None:
            divisor = UNIT_MM[self.units.currentText()]
            size = np.asarray(bounds_size(self.source_poly.GetBounds())) / divisor
            lines.append(f"Model dimensions: {fmt_size(tuple(size))} {self.units.currentText()}")
        return "\n".join(lines)

    def _show_bug_report_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Report a bug")
        dialog.setModal(True)
        dialog.resize(620, 590)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)

        intro = QLabel(
            "Describe the problem below. Create opens a prefilled GitHub issue for final review and submission.",
            dialog,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        title_label = QLabel("Title", dialog)
        title = QLineEdit(dialog)
        title.setPlaceholderText("Short description of the problem")
        title.setMaxLength(160)
        layout.addWidget(title_label)
        layout.addWidget(title)

        details_label = QLabel("What happened", dialog)
        details = QPlainTextEdit(dialog)
        details.setPlaceholderText(
            "What were you doing? What did you expect? What happened instead?"
        )
        details.setMinimumHeight(145)
        layout.addWidget(details_label)
        layout.addWidget(details, 1)

        include_diagnostics = QCheckBox("Include anonymous diagnostics", dialog)
        include_diagnostics.setToolTip(
            "Includes MeshMill version, broad operating-system information, capability status, "
            "and selected MeshMill options. It excludes paths, usernames, device names, hardware "
            "identifiers, driver details, and raw settings files."
        )
        layout.addWidget(include_diagnostics)

        diagnostics = QPlainTextEdit(dialog)
        diagnostics.setPlainText(self._anonymous_bug_diagnostics())
        diagnostics.setMinimumHeight(145)
        diagnostics.setEnabled(False)
        diagnostics.setToolTip(
            "Review and edit this text before creating the GitHub issue.")
        layout.addWidget(diagnostics)

        privacy_note = QLabel(
            "Diagnostics are optional and editable. File paths, account information, and hardware "
            "identifiers are not collected.",
            dialog,
        )
        privacy_note.setWordWrap(True)
        privacy_note.setProperty("muted", True)
        layout.addWidget(privacy_note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        create_button = buttons.addButton("Create GitHub issue", QDialogButtonBox.ButtonRole.AcceptRole)
        create_button.setProperty("accent", True)
        layout.addWidget(buttons)

        include_diagnostics.toggled.connect(diagnostics.setEnabled)

        def create_issue() -> None:
            report_details = details.toPlainText().strip()
            if not report_details:
                QMessageBox.information(dialog, APP_NAME, "Describe the problem before creating the issue.")
                return
            QDesktopServices.openUrl(
                QUrl(
                    build_bug_report_url(
                        title.text(),
                        report_details,
                        diagnostics.toPlainText() if include_diagnostics.isChecked() else None,
                    )
                )
            )
            dialog.accept()

        buttons.rejected.connect(dialog.reject)
        create_button.clicked.connect(create_issue)
        dialog.exec()

    def _show_about(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"About {APP_NAME}")
        dialog.setMinimumWidth(390)

        layout = QVBoxLayout(dialog)
        heading = QHBoxLayout()
        icon = QLabel(dialog)
        icon.setPixmap(self.windowIcon().pixmap(52, 52))
        heading.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)

        title_layout = QVBoxLayout()
        title = QLabel(f"{APP_NAME} {APP_VERSION}", dialog)
        title.setObjectName("aboutTitle")
        title_layout.addWidget(title)
        description = QLabel("Inspect, reduce, solidify, and cast STL meshes.", dialog)
        description.setWordWrap(True)
        title_layout.addWidget(description)
        heading.addLayout(title_layout, 1)
        layout.addLayout(heading)

        changes_row = QHBoxLayout()
        version_changes = QPushButton(f"What’s new in {APP_VERSION}", dialog)
        all_changes = QPushButton("All changes", dialog)
        changes_row.addWidget(version_changes)
        changes_row.addWidget(all_changes)
        layout.addLayout(changes_row)

        action_row = QHBoxLayout()
        report_bug = QPushButton("Report a bug", dialog)
        request_feature = QPushButton("Request a feature", dialog)
        action_row.addWidget(report_bug)
        action_row.addWidget(request_feature)
        layout.addLayout(action_row)

        support = QPushButton("Support MeshMill", dialog)
        support.setProperty("support", True)
        support.setIcon(QIcon(os.fspath(resource_path("assets/support-profile.webp"))))
        support.setIconSize(QSize(26, 26))
        layout.addWidget(support)

        license_label = QLabel("GNU General Public License v3.0 or later.", dialog)
        license_label.setProperty("muted", True)
        layout.addWidget(license_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def changelog_text(current_only: bool) -> str:
            try:
                text = resource_path("CHANGELOG.md").read_text(encoding="utf-8")
            except OSError:
                return "Changelog is unavailable in this build."
            if not current_only:
                return text
            heading_text = f"## {APP_VERSION}"
            start = text.find(heading_text)
            if start < 0:
                return text
            end = text.find("\n## ", start + len(heading_text))
            return text[start:] if end < 0 else text[start:end]

        def show_changelog(current_only: bool) -> None:
            changelog = QDialog(dialog)
            changelog.setWindowTitle(
                f"What’s new in {APP_VERSION}" if current_only else "Changelog"
            )
            screen = self.screen() or QApplication.primaryScreen()
            available = screen.availableGeometry()
            changelog.resize(
                min(
                    max(520, available.width() - 64),
                    min(1120, max(860, round(available.width() * 0.72))),
                ),
                min(
                    max(420, available.height() - 64),
                    min(820, max(620, round(available.height() * 0.76))),
                ),
            )
            changelog_layout = QVBoxLayout(changelog)
            viewer = create_changelog_viewer(
                changelog_text(current_only), self.appearance_theme_name, changelog)
            changelog_layout.addWidget(viewer)
            close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=changelog)
            close.rejected.connect(changelog.reject)
            changelog_layout.addWidget(close)
            center = available.center()
            changelog.move(center.x() - changelog.width() // 2, center.y() - changelog.height() // 2)
            changelog.exec()

        version_changes.clicked.connect(lambda: show_changelog(True))
        all_changes.clicked.connect(lambda: show_changelog(False))

        def open_bug_report() -> None:
            dialog.accept()
            self._show_bug_report_dialog()

        report_bug.clicked.connect(open_bug_report)
        request_feature.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(FEATURE_REQUEST_URL))
        )
        support.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(SUPPORT_URL)))
        dialog.exec()

    def _open_repository(self) -> None:
        if REPOSITORY_URL:
            QDesktopServices.openUrl(QUrl(REPOSITORY_URL))

    def save_file(self) -> None:
        if (
            not self.mesh_modified
            or self.source_poly is None
            or self.source_faces is None
            or self.source_path is None
        ):
            return
        suggested = suggested_mesh_path(self.source_path, len(self.source_faces))
        from mesh_export_ui import choose_mesh_export
        choice = choose_mesh_export(self, "Save current mesh", suggested, self.source_poly)
        if choice is None:
            return
        destination, merge = choice
        self._export_meshes([(destination, self.source_poly)], merge)

    def _export_meshes(self, outputs, merge: bool, closed=None) -> None:
        if self.busy:
            return
        if self.source_path and any(path.resolve() == Path(self.source_path).resolve()
                                    for path, _ in outputs):
            QMessageBox.warning(self, APP_NAME, "Choose a separate output file.")
            return
        # Each export owns its VTK data. The viewport and history remain editable
        # geometry sources rather than inputs to background VTK filters.
        snapshots = []
        from mesh_export import calibrated_export_mesh
        for path, poly in outputs:
            snapshot = vtkPolyData()
            snapshot.DeepCopy(poly)
            snapshots.append((path, calibrated_export_mesh(snapshot, self.export_calibration_transform)))
        token = self._begin_mesh_operation("Exporting", outputs[0][0].name)
        cancel = threading.Event()
        self.operation_cancel_event = cancel
        progress_path = self._start_operation_progress(token, "Preparing export")
        def work():
            from mesh_export import write_polygon_mesh
            reports = []
            def progress(stage, completed, total):
                if cancel.is_set():
                    raise OperationCancelled
                _write_process_progress(progress_path, stage, completed, total)
            for path, poly in snapshots:
                progress("Preparing export", 0, 1)
                marker = poly.GetFieldData().GetArray("ImprintClosedOutput")
                if marker is not None or closed is not None:
                    from imprint_cleanup import clean_generated_mesh
                    is_closed = closed if closed is not None else bool(marker.GetValue(0))
                    poly, _ = clean_generated_mesh(poly, closed=is_closed)
                    # This isolated export copy has already passed cleanup.
                    poly.GetFieldData().RemoveArray("ImprintClosedOutput")
                if path.suffix.lower() == '.stl':
                    import tempfile
                    fd, temporary = tempfile.mkstemp(prefix='.meshmill-export-', suffix='.stl', dir=path.parent)
                    os.close(fd)
                    try:
                        progress("Writing geometry", 0, poly.GetNumberOfPolys())
                        save_stl(poly, Path(temporary))
                        progress("Writing geometry", poly.GetNumberOfPolys(), poly.GetNumberOfPolys())
                        os.replace(temporary, path)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
                    reports.append({"polygons": poly.GetNumberOfPolys()})
                else:
                    points, faces = polydata_arrays(poly)
                    reports.append(write_polygon_mesh(points, faces, path, merge=merge,
                                                       unit_meter=.001, progress=progress))
            return reports
        future = self.executor.submit(work)
        self.operation_future = future
        poll = QTimer(self)
        poll.setInterval(60)
        def finished():
            if token != self.edit_generation or getattr(self, "_closing", False):
                poll.stop()
                poll.deleteLater()
                return
            if not future.done():
                return
            poll.stop()
            poll.deleteLater()
            try:
                reports = future.result()
            except OperationCancelled:
                self._end_mesh_operation()
                self.status.setText("Export cancelled")
            except Exception as error:  # noqa: BLE001 - export worker boundary
                self._end_mesh_operation()
                QMessageBox.warning(self, APP_NAME, str(error) or "Export failed")
            else:
                self._end_mesh_operation()
                total = sum(report['polygons'] for report in reports)
                self.status.setText(f"Saved {fmt_count(total)} faces ({outputs[0][0].suffix})")
        poll.timeout.connect(finished)
        poll.start()


def main() -> int:
    preserved_workers = tuple(int(value) for value in
                              os.environ.get("MESHMILL_PRESERVE_WORKER_PIDS", "").split(",")
                              if value.strip().isdecimal())
    threading.Thread(target=recover_orphan_workers,
                     kwargs={"excluded_pids": preserved_workers},
                     name="worker-recovery", daemon=True).start()
    cli_name = Path(sys.executable).stem.lower().endswith("cli")
    if "--cli" in sys.argv or cli_name:
        args = [arg for arg in sys.argv[1:] if arg != "--cli"]
        return cli_main(args)
    verify_parser = argparse.ArgumentParser(add_help=False)
    verify_parser.add_argument("--verification-screenshot", type=Path)
    verify_parser.add_argument("--verification-target", type=int, default=200_000)
    verify_parser.add_argument("--verification-display", choices=DISPLAY_MODES)
    verify_parser.add_argument("--verification-selection", action="store_true")
    verify_parser.add_argument("--verification-loupe", action="store_true")
    verify_parser.add_argument("--diagnostic-log", type=Path)
    verify_args, qt_args = verify_parser.parse_known_args(sys.argv[1:])
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(APP_STYLESHEET)
    tooltip_palette = app.palette()
    tooltip_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    tooltip_palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#111827"))
    app.setPalette(tooltip_palette)
    QToolTip.setPalette(tooltip_palette)
    app_icon = resource_path("assets/meshmill-mark.svg")
    if app_icon.exists():
        app.setWindowIcon(QIcon(os.fspath(app_icon)))
    window = MainWindow()
    app.aboutToQuit.connect(window._shutdown_background_workers)
    window.diagnostic_log_path = verify_args.diagnostic_log
    if window.diagnostic_log_path is not None:
        window.diagnostic_log_path.unlink(missing_ok=True)
        window._diagnostic("session_start", argv=sys.argv[1:])
    window.verification_screenshot = verify_args.verification_screenshot
    window.verification_target = verify_args.verification_target
    window.verification_selection = verify_args.verification_selection
    window.verification_loupe = verify_args.verification_loupe
    window.showMaximized()
    # The Windows frame and the embedded native VTK window settle in separate
    # event-loop passes during startup. Refresh against the final client area.
    window._schedule_viewport_layout_refresh()
    def settle_startup_window() -> None:
        if not window.isMaximized():
            window.showMaximized()
        window._schedule_viewport_layout_refresh()

    for delay in (250, 1000, 2500):
        QTimer.singleShot(delay, settle_startup_window)
    paths = [arg for arg in qt_args if not arg.startswith("-")]
    if paths:
        def load_initial() -> None:
            if verify_args.verification_display:
                display_index = window.display_type.findData(verify_args.verification_display)
                if display_index >= 0:
                    window.display_type.setCurrentIndex(display_index)
            window.load_path(Path(paths[0]))
        QTimer.singleShot(0, load_initial)
    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
