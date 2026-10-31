"""Themed format comparison and save controls for geometry exports."""

import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from mesh_export import coplanar_polygons

FORMAT_ORDER = (".dae", ".glb", ".ply", ".obj", ".stl")
FORMAT_INFO = {
    ".dae": (
        "COLLADA",
        "Larger flat faces for SketchUp editing",
        "Polygons and explicit units",
        "XML overhead; legacy support elsewhere",
        "Native import",
        "Up to 4.5 LTS",
        "Not listed",
        "Polygons / millimeters",
    ),
    ".glb": (
        "GLB",
        "Compact, self-contained delivery",
        "Shared vertices; modern app support",
        "Still triangles; fewer bytes does not mean fewer edges",
        "Recent versions",
        "Native import",
        "Supported upload",
        "Triangles / meters",
    ),
    ".ply": (
        "PLY",
        "Compact scanned geometry",
        "Binary storage and polygon faces",
        "Some applications need an importer",
        "Extension / conversion",
        "Native import",
        "Not listed",
        "Polygons / unitless",
    ),
    ".obj": (
        "OBJ",
        "General modeling exchange",
        "Readable text and polygon faces",
        "Text parsing; no standard units",
        "Extension / conversion",
        "Native import",
        "Supported upload",
        "Polygons / unitless",
    ),
    ".stl": (
        "STL",
        "Slicer compatibility",
        "Widely supported; predictable output",
        "Repeated coordinates; triangles only",
        "Native import",
        "Native import",
        "Supported upload",
        "Triangles / unitless",
    ),
}
RECOMMENDATIONS = {
    "SketchUp": (".dae", "COLLADA for fewer flat-face edges"),
    "Blender": (".glb", "GLB for compact interchange"),
    "Meshy": (".glb", "GLB for a single-file upload"),
    "3D printing / slicer": (".stl", "STL for broad slicer compatibility"),
    "Scan processing": (".ply", "PLY for compact indexed geometry"),
}
PRODUCT_LINKS = {
    'SketchUp': 'https://sketchup.trimble.com/en?ref=meshmill',
    'Blender': 'https://www.blender.org/?ref=meshmill',
    'Meshy': 'https://www.meshy.ai/?ref=meshmill',
}


def product_link(product, theme, parent=None):
    label = QLabel(parent)
    color = readable_theme_color(theme['accent'], [theme['window'], theme['panel']], theme['text'])
    url = PRODUCT_LINKS[product]
    label.setText(f'<a href="{url}" style="color:{color};text-decoration:underline;">{product}</a>')
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    label.setOpenExternalLinks(True)
    label.setToolTip(url)
    label.setAccessibleName(product + ' official website')
    return label


def export_preferences(parent):
    settings = parent.settings
    suffix = str(settings.value('export/preferred_format', '.dae'))
    return {
        'format': suffix if suffix in FORMAT_ORDER else '.dae',
        'merge': settings.value('export/preferred_merge', False, type=bool),
        'software': str(settings.value('export/preferred_software', 'SketchUp')),
        'compact': settings.value('export/compact_save', False, type=bool),
    }


def estimate_geometry(points, faces, cancel, progress, merge=True):
    def notify(stage, done, total):
        if cancel.is_set():
            raise RuntimeError("Cancelled")
        progress[:] = [stage, done, total]

    if merge:
        polygons, report = coplanar_polygons(points, faces, notify)
        ply_polygons, ply_report = coplanar_polygons(points, faces, notify, max_vertices=255)
    else:
        polygons = ply_polygons = faces
        report = {'polygons': len(faces), 'removed_edges': 0}
        ply_report = report
    report["ply_polygons"] = ply_report["polygons"]
    report["ply_removed_edges"] = ply_report["removed_edges"]
    used = np.unique(np.concatenate(polygons))
    mapping = np.full(len(points), -1, dtype=int)
    mapping[used] = np.arange(len(used))
    polygons = [mapping[p] for p in polygons]
    coordinates = sum(len(format(float(v), ".17g")) + 1 for v in points[used].flat)
    indices = sum(sum(len(str(int(i))) + 1 for i in p) for p in polygons)
    sizes = {
        ".stl": 84 + 50 * len(faces),
        ".glb": 12 * len(points) + 12 * len(faces) + 1024,
        ".ply": 24 * len(np.unique(np.concatenate(ply_polygons)))
        + sum(1 + 4 * len(p) for p in ply_polygons)
        + 256,
        ".obj": coordinates + len(used) * 2 + indices + len(polygons) * 3,
        ".dae": coordinates + indices + sum(len(str(len(p))) + 1 for p in polygons) + 1600,
    }
    return report, sizes


def readable_theme_color(color, backgrounds, text):
    def luminance(value):
        channels = (value.redF(), value.greenF(), value.blueF())
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return sum(a * b for a, b in zip(linear, (0.2126, 0.7152, 0.0722)))

    original, target = QColor(color), QColor(text)
    for step in range(21):
        fraction = step / 20
        candidate = QColor.fromRgbF(
            *(
                a + (b - a) * fraction
                for a, b in zip(
                    (original.redF(), original.greenF(), original.blueF()),
                    (target.redF(), target.greenF(), target.blueF()),
                )
            )
        )
        light = luminance(candidate)
        if all(
            (max(light, luminance(QColor(bg))) + 0.05) / (min(light, luminance(QColor(bg))) + 0.05)
            >= 4.5
            for bg in backgrounds
        ):
            return candidate.name()
    return target.name()


class FormatComparison(QFrame):
    selected = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("formatComparison")
        self.cells = []
        self.buttons = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(1, 1, 1, 1)
        grid.setSpacing(0)
        grid.setColumnMinimumWidth(0, 100)
        for col in range(1, 6):
            grid.setColumnStretch(col, 1)
            grid.setColumnMinimumWidth(col, 120)
        subtitles = (
            ".dae · polygon editing",
            ".glb · modern exchange",
            ".ply · compact scans",
            ".obj · interchange",
            ".stl · printing",
        )
        rows = (
            "Compare",
            "Does best",
            "Strength",
            "Tradeoff",
            "SketchUp",
            "Blender",
            "Meshy",
            "MeshMill output",
        )
        for row, caption in enumerate(rows):
            for col in range(6):
                cell = QFrame(self)
                cell.setObjectName("formatCell")
                cell.setProperty("header", row == 0)
                cell.setProperty("lastRow", row == 7)
                cell.setProperty("selected", False)
                layout = QVBoxLayout(cell)
                layout.setContentsMargins(9, 10, 9, 10)
                layout.setSpacing(3)
                if row == 0 and col:
                    suffix = FORMAT_ORDER[col - 1]
                    button = QPushButton(FORMAT_INFO[suffix][0], cell)
                    button.setObjectName("formatPick")
                    button.setCheckable(True)
                    button.setAccessibleName("Choose " + FORMAT_INFO[suffix][0])
                    button.clicked.connect(lambda _checked=False, f=suffix: self.selected.emit(f))
                    self.buttons[suffix] = button
                    layout.addWidget(button)
                    label = QLabel(subtitles[col - 1], cell)
                    label.setObjectName("formatSubtitle")
                    label.setWordWrap(True)
                    layout.addWidget(label)
                else:
                    text = caption if col == 0 else FORMAT_INFO[FORMAT_ORDER[col - 1]][row]
                    label = (product_link(caption, parent.theme, cell)
                             if col == 0 and caption in PRODUCT_LINKS else QLabel(text, cell))
                    label.setObjectName("formatCriterion" if col == 0 else "formatText")
                    label.setWordWrap(True)
                    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                    if row in (4, 5, 6) and text in (
                        "Native import",
                        "Recent versions",
                        "Supported upload",
                    ):
                        label.setProperty("compatible", True)
                    layout.addWidget(label)
                grid.addWidget(cell, row, col)
                self.cells.append((col, cell))

    def select_format(self, suffix):
        selected = FORMAT_ORDER.index(suffix) + 1
        for col, cell in self.cells:
            cell.setProperty("selected", col == selected)
            cell.style().unpolish(cell)
            cell.style().polish(cell)
        for key, button in self.buttons.items():
            button.setChecked(key == suffix)


class MeshExportDialog(QDialog):
    def __init__(self, parent, title, suggested, poly=None, initial_options=None):
        super().__init__(parent)
        from meshmill import appearance_theme, polydata_arrays, resource_path

        self.setWindowTitle(title)
        self.setObjectName("meshExportDialog")
        self.theme = appearance_theme(parent.appearance_theme_name)
        self.folder = Path(suggested).parent
        self.selected_format = ".dae"
        self.result_choice = None
        self.compact_requested = False
        self.analysis = None
        self.executor = None
        self.cancel_analysis = threading.Event()
        self.progress = ["Finding planar faces", 0, 1]
        self.points, self.faces = (
            polydata_arrays(poly)
            if poly is not None
            else (np.empty((0, 3)), np.empty((0, 3), dtype=int))
        )
        self.points, self.faces = self.points.copy(), self.faces.copy()
        self.setFont(parent.font())
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        chrome = QFrame(self)
        chrome.setObjectName("exportChrome")
        chrome_row = QHBoxLayout(chrome)
        chrome_row.setContentsMargins(18, 12, 18, 12)
        mark = QLabel()
        mark.setPixmap(QIcon(str(resource_path("assets/meshmill-mark.svg"))).pixmap(22, 22))
        chrome_row.addWidget(mark)
        brand = QLabel("MeshMill")
        brand.setObjectName("exportBrand")
        chrome_row.addWidget(brand)
        chrome_row.addStretch(1)
        chrome_caption = QLabel("SAVE AS")
        chrome_caption.setObjectName("exportMuted")
        chrome_row.addWidget(chrome_caption)
        root.addWidget(chrome)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(18, 18, 18, 18)
        body.setSpacing(17)
        introduction = QHBoxLayout()
        introduction.setSpacing(14)
        title_group = QVBoxLayout()
        title_group.setSpacing(5)
        heading = QLabel("Save mesh")
        heading.setObjectName("exportHeading")
        font = heading.font()
        if font.pixelSize() > 0:
            font.setPixelSize(round(font.pixelSize() * 1.5))
        else:
            font.setPointSizeF(font.pointSizeF() * 1.5)
        font.setBold(True)
        heading.setFont(font)
        title_group.addWidget(heading)
        title_group.addWidget(
            QLabel(
                f"{Path(suggested).stem}  •  {len(self.faces):,} triangles  •  {len(self.points):,} vertices"
            )
        )
        introduction.addLayout(title_group, 1)
        badge = QLabel("Working mesh stays unchanged")
        badge.setObjectName("exportBadge")
        introduction.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
        body.addLayout(introduction)
        target = QHBoxLayout()
        target.addWidget(QLabel("Opening in"))
        self.destination = QComboBox()
        self.destination.addItems(RECOMMENDATIONS)
        target.addWidget(self.destination)
        self.recommendation = QLabel()
        self.recommendation.setWordWrap(True)
        target.addWidget(self.recommendation, 1)
        body.addLayout(target)
        self.table = FormatComparison(self)
        self.table.selected.connect(self.select_format)
        body.addWidget(self.table)
        self.merge = QCheckBox("Merge coplanar triangles")
        self.merge.setObjectName("exportMergeCoplanar")
        self.merge.setChecked(False)
        self.merge.setToolTip(
            "Combine adjoining triangles on the same plane. Preserve curves, sharp edges, openings and shared boundary vertices. STL and GLB export triangles."
        )
        stats_frame = QFrame()
        stats_frame.setObjectName("exportStats")
        options = QVBoxLayout(stats_frame)
        options.setContentsMargins(13, 13, 13, 13)
        options.setSpacing(13)
        merge_row = QHBoxLayout()
        merge_row.addWidget(self.merge)
        merge_row.addStretch(1)
        self.merge_detail = QLabel("Preserve curves, creases and openings")
        self.merge_detail.setObjectName("exportMuted")
        merge_row.addWidget(self.merge_detail)
        options.addLayout(merge_row)
        stats = QGridLayout()
        stats.setHorizontalSpacing(18)
        options.addLayout(stats)
        self.face_value, self.size_value = QLabel(), QLabel()
        for col, (label, value) in enumerate(
            (
                ("Export faces", self.face_value),
                ("Estimated size", self.size_value),
                ("Geometry", QLabel("Preserved")),
            )
        ):
            caption = QLabel(label.upper())
            caption.setObjectName("exportMuted")
            stats.addWidget(caption, 0, col)
            stats.addWidget(value, 1, col)
            stats.setColumnStretch(col, 1)
            font = value.font()
            if font.pixelSize() > 0:
                font.setPixelSize(round(font.pixelSize() * 1.25))
            else:
                font.setPointSizeF(font.pointSizeF() * 1.25)
            font.setBold(True)
            value.setFont(font)
            value.setObjectName("exportValue")
            value.setToolTip(
                (
                    "Faces written to the export. Receiving applications may triangulate polygon faces.",
                    "Estimated geometry file size for MeshMill encoding. Actual size can differ.",
                    "No additional mesh reduction. GLB uses 32-bit positions in meters, which can introduce coordinate rounding.",
                )[col]
            )
        self.face_detail, self.size_detail = QLabel(), QLabel("Estimated, uncompressed")
        self.face_detail.setObjectName("exportMuted")
        self.size_detail.setObjectName("exportMuted")
        stats.addWidget(self.face_detail, 2, 0)
        stats.addWidget(self.size_detail, 2, 1)
        unchanged = QLabel("Same optimized mesh")
        unchanged.setObjectName("exportMuted")
        stats.addWidget(unchanged, 2, 2)
        self.size_bar = QProgressBar()
        self.size_bar.setObjectName("exportSizeBar")
        self.size_bar.setRange(0, 100)
        self.size_bar.setTextVisible(False)
        self.size_bar.setFixedHeight(5)
        stats.addWidget(self.size_bar, 3, 1)
        body.addWidget(stats_frame)
        self.analysis_status = QLabel()
        self.analysis_status.setWordWrap(True)
        body.addWidget(self.analysis_status)
        row = QHBoxLayout()
        file_column = QVBoxLayout()
        file_column.setSpacing(5)
        filename_caption = QLabel("FILENAME")
        filename_caption.setObjectName("exportMuted")
        file_column.addWidget(filename_caption)
        self.filename = QLineEdit(Path(suggested).stem + ".dae")
        file_column.addWidget(self.filename)
        row.addLayout(file_column, 1)
        folder = QPushButton("Choose folder…")
        folder.clicked.connect(self.choose_folder)
        row.addWidget(folder, 0, Qt.AlignmentFlag.AlignBottom)
        body.addLayout(row)
        self.folder_label = QLabel()
        self.folder_label.setObjectName("exportMuted")
        body.addWidget(self.folder_label)
        body.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_button = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setProperty("accent", True)
        self.save_button.setDefault(True)
        self.buttons.accepted.connect(self.confirm)
        self.buttons.rejected.connect(self.reject)
        bottom = QFrame(self)
        bottom.setObjectName("exportFooter")
        footer_row = QHBoxLayout(bottom)
        footer_row.setContentsMargins(18, 13, 18, 13)
        self.prefer_button = QPushButton("Use these settings next time")
        self.prefer_button.setAutoDefault(False)
        self.prefer_button.clicked.connect(self.remember_preferences)
        footer_row.addWidget(self.prefer_button)
        footer_row.addStretch(1)
        # Match the reviewed Cancel, Save ordering on every operating system.
        cancel_button = self.buttons.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.setAutoDefault(False)
        self.buttons.removeButton(cancel_button)
        self.buttons.removeButton(self.save_button)
        cancel_button.clicked.connect(self.reject)
        self.save_button.clicked.connect(self.confirm)
        footer_row.addWidget(cancel_button)
        footer_row.addWidget(self.save_button)
        root.addWidget(bottom)
        t = self.theme
        pixels = parent._font_pixel_size()
        accent_text = readable_theme_color(t["accent"], [t["accent_soft"]], t["text"])
        positive_text = readable_theme_color(t["stats"], [t["window"], t["accent_soft"]], t["text"])
        badge_text = readable_theme_color(t["stats"], [t["stats_soft"]], t["text"])
        self.setStyleSheet(
            f"QDialog#meshExportDialog, QDialog#meshExportDialog QScrollArea, QDialog#meshExportDialog QScrollArea > QWidget > QWidget {{background:{t['window']};color:{t['text']};}}"
            f"QLabel#exportHeading {{font-size:{round(pixels * 1.7)}px;font-weight:600;}}"
            f"QFrame#exportChrome, QFrame#exportFooter {{background:{t['panel']};border:0;border-bottom:1px solid {t['border']};}}"
            f"QLabel#exportBrand {{font-size:{pixels + 2}px;font-weight:600;background:transparent;}}"
            f"QLabel#exportMuted, QLabel#formatCriterion, QLabel#formatSubtitle {{color:{t['muted']};background:transparent;}}"
            f"QLabel#formatCriterion, QLabel#exportMuted {{font-size:{max(11, pixels - 1)}px;}}"
            f"QLabel#formatSubtitle {{font-size:{max(11, pixels - 2)}px;}}"
            f"QLabel#exportBadge {{background:{t['stats_soft']};color:{badge_text};border-radius:6px;padding:5px 9px;}}"
            f"QFrame#formatComparison {{background:{t['panel']};border:1px solid {t['border']};border-radius:9px;}}"
            f"QFrame#formatCell {{background:{t['window']};border:0;border-bottom:1px solid {t['border']};}}"
            f"QFrame#formatCell[header=true] {{background:{t['panel']};}}"
            f"QFrame#formatCell[selected=true] {{background:{t['accent_soft']};}}"
            f"QFrame#formatCell[lastRow=true] {{border-bottom:0;}}"
            f"QFrame#formatCell QLabel {{background:transparent;border:0;padding:0;}}"
            f"QLabel#formatText[compatible=true] {{color:{positive_text};}}"
            f"QPushButton#formatPick {{background:transparent;color:{t['text']};border:0;padding:0;text-align:left;font-size:{pixels + 2}px;font-weight:600;min-height:22px;}}"
            f"QPushButton#formatPick:checked {{color:{accent_text};}}"
            f"QPushButton#formatPick:hover {{color:{t['accent_hover']};}}"
            f"QProgressBar#exportSizeBar {{background:{t['disabled']};border:0;border-radius:2px;padding:0;min-height:5px;}}"
            f"QProgressBar#exportSizeBar::chunk {{background:{t['accent']};border:0;border-radius:2px;}}"
            f"QFrame#exportStats {{background:{t['panel']};border:1px solid {t['border']};border-radius:9px;}}"
            f"QFrame#exportStats QLabel {{background:transparent;border:0;}}"
            f"QLabel#exportValue {{font-size:{round(pixels * 1.5)}px;font-weight:600;}}"
        )
        self.merge.toggled.connect(self.update_summary)
        self.destination.currentTextChanged.connect(self.recommend)
        self.recommend(self.destination.currentText())
        options = initial_options or export_preferences(parent)
        if options['software'] in RECOMMENDATIONS:
            self.destination.setCurrentText(options['software'])
        self.merge.setChecked(options['merge'])
        self.select_format(options['format'])
        self.future = None
        self.poll = QTimer(self)
        self.poll.setInterval(80)
        self.poll.timeout.connect(self.poll_analysis)
        self.finished.connect(self.cleanup)
        screen = parent.screen().availableGeometry()
        self.resize(min(1040, int(screen.width() * 0.92)), min(1000, int(screen.height() * 0.92)))
        QTimer.singleShot(0, self.fit_rows)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "table"):
            self.fit_rows()

    def fit_rows(self):
        self.table.updateGeometry()

    def current_options(self):
        return {'format': self.selected_format, 'merge': self.merge.isChecked(),
                'software': self.destination.currentText()}

    def remember_preferences(self):
        settings = self.parent().settings
        for key, value in self.current_options().items():
            settings.setValue('export/preferred_' + key, value)
        settings.setValue('export/compact_save', True)
        settings.sync()
        self.compact_requested = True
        self.reject()

    def recommend(self, software):
        suffix, label = RECOMMENDATIONS[software]
        self.recommendation.setText("Recommended: " + label)
        self.select_format(suffix)

    def select_format(self, suffix):
        self.selected_format = suffix
        self.filename.setText(Path(self.filename.text()).stem + suffix)
        self.merge.setEnabled(suffix not in (".stl", ".glb"))
        self.table.select_format(suffix)
        self.update_summary()

    def update_summary(self):
        self.prefer_button.setText('Use these settings next time')
        suffix = self.selected_format
        merge = self.merge.isEnabled() and self.merge.isChecked()
        count = (
            self.analysis[0]["ply_polygons" if suffix == ".ply" else "polygons"]
            if self.analysis is not None and merge
            else len(self.faces)
        )
        self.face_value.setText(
            f"{count:,}" if self.analysis is not None or not merge else "Calculated on save"
        )
        if suffix == ".stl":
            size = 84 + 50 * len(self.faces)
        elif suffix == ".glb":
            size = 12 * len(self.points) + 12 * len(self.faces) + 1024
        elif self.analysis is not None and merge:
            size = self.analysis[1][suffix]
        else:
            size = None
        self.size_value.setText(
            (f"~{size / 1_000_000:.2f} MB" if size >= 1_000_000 else f"~{size / 1000:.1f} KB")
            if size is not None
            else "Calculated on save"
            if merge
            else "Depends on face encoding"
        )
        self.save_button.setText("Save " + FORMAT_INFO[suffix][0])
        self.folder_label.setText(str(self.folder))
        self.folder_label.setToolTip(str(self.folder))
        self.face_detail.setText("Polygon faces" if merge else "Triangle faces")
        self.merge_detail.setText(
            "Preserve curves, creases and openings"
            if self.merge.isEnabled()
            else "This format exports triangles"
        )
        self.size_bar.setValue(
            min(100, round(100 * size / max(84 + 50 * len(self.faces), 1)))
            if size is not None
            else 0
        )
        if self.analysis is not None:
            removed = (
                self.analysis[0]["ply_removed_edges" if suffix == ".ply" else "removed_edges"]
                if merge
                else 0
            )
            self.analysis_status.setText(f"Planar edges removed: {removed:,}")

    def poll_analysis(self):
        if self.future is None:
            return
        if not self.future.done():
            stage, done, total = self.progress
            self.analysis_status.setText(f"{stage}  {done:,} / {total:,}")
            return
        self.poll.stop()
        try:
            self.analysis = self.future.result()
            self.analysis_status.setText(
                f"{self.analysis[0]['removed_edges']:,} internal planar edges can be removed. Working mesh stays unchanged."
            )
        except Exception:  # noqa: BLE001 - estimates do not prevent export
            self.analysis_status.setText("Size analysis unavailable. Export remains available.")
        self.update_summary()

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Export folder", str(self.folder))
        if folder:
            self.folder = Path(folder)
            self.update_summary()

    def confirm(self):
        name = self.filename.text().strip()
        if not name or Path(name).name != name or any(c in name for c in '<>:"/\\|?*'):
            QMessageBox.warning(
                self, "Save mesh", "Enter a filename without a folder path or reserved characters."
            )
            return
        destination = self.folder / Path(name).with_suffix(self.selected_format)
        if (
            self.parent().source_path
            and destination.resolve() == self.parent().source_path.resolve()
        ):
            QMessageBox.warning(self, "Save mesh", "Choose a separate output file.")
            return
        if (
            destination.exists()
            and QMessageBox.question(self, "Save mesh", "Replace the existing file?")
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.result_choice = (destination, self.merge.isEnabled() and self.merge.isChecked())
        self.accept()

    def cleanup(self, _result):
        self.poll.stop()
        self.cancel_analysis.set()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)


class CompactMeshExportDialog(QDialog):
    """Filename, format and options without comparison or background analysis."""

    def __init__(self, parent, title, suggested, options=None, poly=None, analysis=None):
        super().__init__(parent)
        from meshmill import appearance_theme

        self.setWindowTitle(title)
        self.setFont(parent.font())
        self.setObjectName('compactMeshExportDialog')
        self.options = options or export_preferences(parent)
        self.folder = Path(suggested).parent
        self.selected_format = self.options['format']
        self.result_choice = None
        self.compare_requested = False
        from meshmill import polydata_arrays
        self.points, self.faces = (polydata_arrays(poly) if poly is not None
                                  else (np.empty((0, 3)), np.empty((0, 3), dtype=int)))
        self.points, self.faces = self.points.copy(), self.faces.copy()
        self.analysis_cache = {True: analysis} if analysis is not None else {}
        self.executor = None
        self.future = None
        self.pending_merge = None
        self.cancel_analysis = threading.Event()
        self.progress = ['', 0, 1]
        self.poll = QTimer(self)
        self.poll.setInterval(80)
        self.poll.timeout.connect(self.poll_stats)
        self.finished.connect(self.cleanup)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        fields = QGridLayout()
        fields.setHorizontalSpacing(12)
        fields.setVerticalSpacing(10)
        self.filename = QLineEdit(Path(suggested).stem + self.selected_format)
        fields.addWidget(QLabel('Filename'), 0, 0)
        fields.addWidget(self.filename, 0, 1, 1, 2)
        self.folder_label = QLineEdit(str(self.folder))
        self.folder_label.setReadOnly(True)
        self.folder_label.setToolTip(str(self.folder))
        folder_button = QPushButton('Choose folder…')
        folder_button.setAutoDefault(False)
        folder_button.clicked.connect(self.choose_folder)
        fields.addWidget(QLabel('Folder'), 1, 0)
        fields.addWidget(self.folder_label, 1, 1)
        fields.addWidget(folder_button, 1, 2)
        self.format = QComboBox()
        for suffix in FORMAT_ORDER:
            self.format.addItem(f'{FORMAT_INFO[suffix][0]} ({suffix})', suffix)
        self.format.setCurrentIndex(self.format.findData(self.selected_format))
        fields.addWidget(QLabel('Format'), 2, 0)
        fields.addWidget(self.format, 2, 1, 1, 2)
        self.merge = QCheckBox('Merge coplanar triangles')
        self.merge.setChecked(self.options['merge'])
        stats_row = QHBoxLayout()
        stats_row.addWidget(self.merge)
        stats_row.addStretch(1)
        self.stats = QLabel()
        self.stats.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.stats.setToolTip(f'{len(self.points):,} input vertices · {len(self.faces):,} input triangles')
        stats_row.addWidget(self.stats)
        fields.addLayout(stats_row, 3, 1, 1, 2)
        layout.addLayout(fields)
        buttons = QHBoxLayout()
        self.compare_button = QPushButton('Compare formats…')
        self.compare_button.setAutoDefault(False)
        self.compare_button.clicked.connect(self.show_comparison)
        buttons.addWidget(self.compare_button)
        buttons.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.save_button = QPushButton()
        self.save_button.setProperty('accent', True)
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.confirm)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        self.format.currentIndexChanged.connect(self.format_changed)
        self.merge.toggled.connect(self.update_stats)
        self.format_changed()
        t = appearance_theme(parent.appearance_theme_name)
        self.setStyleSheet(
            f"QDialog#compactMeshExportDialog {{background:{t['window']};color:{t['text']};}}"
            f"QDialog#compactMeshExportDialog QLabel {{color:{t['text']};background:transparent;}}"
        )
        self.resize(max(580, self.sizeHint().width()), self.sizeHint().height())

    def format_changed(self):
        self.selected_format = self.format.currentData()
        self.filename.setText(Path(self.filename.text()).stem + self.selected_format)
        self.merge.setEnabled(self.selected_format not in ('.stl', '.glb'))
        self.save_button.setText('Save ' + FORMAT_INFO[self.selected_format][0])
        self.update_stats()

    def update_stats(self):
        if not len(self.faces):
            self.stats.clear()
            return
        suffix = self.selected_format
        merge = self.merge.isEnabled() and self.merge.isChecked()
        count = len(self.faces)
        size = None
        if suffix == '.stl':
            size = 84 + 50 * count
        elif suffix == '.glb':
            size = 12 * len(self.points) + 12 * count + 1024
        elif merge in self.analysis_cache:
            report, sizes = self.analysis_cache[merge]
            count = report['ply_polygons' if suffix == '.ply' else 'polygons']
            size = sizes[suffix]
        units = ('faces' if size is not None else 'triangles')
        amount = ('calculated on save' if merge and size is None else 'estimated on save' if size is None else f'~{size / 1_000_000:.2f} MB' if size >= 1_000_000
                  else f'~{size / 1000:.1f} KB' if size >= 1000 else f'{size:,} B')
        self.stats.setText(f'{count:,} {units} · {amount}')

    def poll_stats(self):
        if self.future is None or not self.future.done():
            return
        self.poll.stop()
        try:
            self.analysis_cache[self.pending_merge] = self.future.result()
        except Exception:  # noqa: BLE001 - size estimates do not prevent saving
            self.stats.setText(f'{len(self.faces):,} triangles · size unavailable')
        else:
            self.update_stats()

    def cleanup(self, _result):
        self.poll.stop()
        self.cancel_analysis.set()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Export folder', str(self.folder))
        if folder:
            self.folder = Path(folder)
            self.folder_label.setText(str(self.folder))
            self.folder_label.setToolTip(str(self.folder))

    def current_options(self):
        return dict(self.options, format=self.selected_format, merge=self.merge.isChecked())

    def show_comparison(self):
        self.parent().settings.setValue('export/compact_save', False)
        self.parent().settings.sync()
        self.compare_requested = True
        self.reject()

    def confirm(self):
        # Both views share filename, overwrite and source-file protection.
        MeshExportDialog.confirm(self)


def choose_mesh_export(parent, title, suggested, poly=None):
    options = export_preferences(parent)
    compact_mode = options['compact']
    analysis = None
    while True:
        dialog = (CompactMeshExportDialog(parent, title, suggested, options, poly, analysis) if compact_mode
                  else MeshExportDialog(parent, title, suggested, poly, options))
        dialog.exec()
        switch = dialog.compare_requested if compact_mode else dialog.compact_requested
        options = dialog.current_options()
        suggested = dialog.folder / dialog.filename.text()
        choice = dialog.result_choice
        if not compact_mode:
            analysis = dialog.analysis
        dialog.deleteLater()
        if not switch:
            return choice
        compact_mode = not compact_mode
