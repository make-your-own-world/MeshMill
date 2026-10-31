"""Fixed-spindle finger interaction with rigid centering and levelling."""

import math
import time

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkCoordinate,
    vtkPolyDataMapper,
    vtkPolyDataMapper2D,
    vtkTextActor,
)

from axis_datum import fit_probe_datum
from lathe_contact import build_contact_index, surface_contact
from lathe_spindle import pose_matrix, spin_matrix


def viewport_axis_ends(first, second, width, height):
    origin = np.asarray(first, dtype=float)
    direction = np.asarray(second, dtype=float) - origin
    if np.linalg.norm(direction) < 1e-8:
        return None
    low, high = -np.inf, np.inf
    for coordinate, delta, limit in zip(origin, direction, (width - 1., height - 1.)):
        if abs(delta) < 1e-12:
            if not 0 <= coordinate <= limit:
                return None
            continue
        a, b = sorted((-coordinate / delta, (limit - coordinate) / delta))
        low, high = max(low, a), min(high, b)
    if low > high:
        return None
    return origin + direction * low, origin + direction * high


class CalibrationProgressBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.rotation_text = ""
        self.setTextVisible(False)

    def set_rotation(self, text):
        if text != self.rotation_text:
            self.rotation_text = text
            self.update()
        self.setMinimumHeight(self.fontMetrics().height() * 2 + 12)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        text = self.format().replace("%p%", str(self.value()) + "%")
        if self.rotation_text:
            text += "\n" + self.rotation_text
        width = round(self.width() * self.value() / max(1, self.maximum()))
        for left, right, role in ((0, width, QPalette.ColorRole.HighlightedText),
                                   (width, self.width(), QPalette.ColorRole.Text)):
            painter.save()
            painter.setClipRect(left, 0, right-left, self.height())
            painter.setPen(self.palette().color(role))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            painter.restore()


class LatheController:
    def __init__(self, owner, grid, unit_scales):
        self.owner, self.unit_scales = owner, unit_scales
        self.center = self.axis = self.mesh = self.position = None
        self.probes, self.probe_worlds, self.probe_frames, self.actors = [], [], [], []
        self._probe_drawings = []
        self._pending_manual_camera = False
        self.manual_zoom = False
        self.coverage = np.zeros((0, 72), dtype=bool)
        self.calibration_locked = False
        self.detection_complete = False
        self.has_calibration = False
        self.retained_probe_frames = []
        self.samples, self.sample_groups = [], []
        self.rig = self.world_to_rig = self.model_matrix = None
        self.parameters = np.zeros(4)
        self.target_parameters = np.zeros(4)
        self.base_camera = None
        self.extent = 1.0
        self.future = self.locator_future = self.locator = self.locator_mesh = None
        self.generation = self.future_generation = 0
        self.sweep, self.direction = 0.0, 1
        self.dragging, self.active_probe = False, None
        self._resume_after_probe = False
        self.reference_basis = None
        self.previous_reference = self.previous_views = None
        self.revision, self.calibrated_revision = 0, -1
        self._calibrated_parameters = None
        self.stable_fits = 0
        self.last_fit_time = -math.inf
        self.last_result = None
        self.wobble_baseline = None
        self.solve_complete = False
        self.probe_travel = []
        self._applying = False
        self._camera_signature = None
        self._framed_parameters = np.zeros(4)
        self.accepted_frame = None
        self.original_reference_frame = None
        section, header = owner._make_tool_section("AXIAL CALIBRATION", "settings", "optimization")
        self.button = QToolButton()
        self.button.setObjectName("sectionExpander")
        self.button.setArrowType(Qt.ArrowType.RightArrow)
        self.button.setAccessibleName("Expand Axial calibration")
        self.button.setCheckable(True)
        self.button.setEnabled(owner.active_actor is not None)
        self.button.setToolTip(
            "Spin like the Left/Right arrows. Add contact fingers at different heights to refine the axis. Drag a circle to reposition it; double-click to remove it. Turning Axial Calibration off keeps alignment. Reset restores object-center rotation."
        )
        self.button.toggled.connect(self.toggle)
        header.addWidget(self.button)
        self.controls = QWidget()
        owner.alignment_toggle = self.button
        owner.alignment_body = self.controls
        self.button.toggled.connect(lambda enabled: owner._toggle_tool_section("Axial calibration", enabled))
        layout = QVBoxLayout(self.controls)
        layout.setContentsMargins(0, 0, 0, 0)
        axis_row = QHBoxLayout()
        axis_row.addWidget(QLabel("Viewport axis"))
        self.axis_choice = QComboBox()
        self.axis_choice.addItem("Vertical (turntable)", "vertical")
        self.axis_choice.addItem("Left / Right", "horizontal")
        self.axis_choice.addItem("In / Out", "depth")
        saved_axis = str(owner.settings.value("alignment/viewport_axis", "vertical"))
        self.axis_choice.setCurrentIndex(max(0, self.axis_choice.findData(saved_axis)))
        self.axis_choice.setToolTip("Rotation axis relative to the current view, through the model center. Up/Down spins around screen vertical; Left/Right around screen horizontal; In/Out around screen depth. Reverse changes rotation direction. Changing this clears probes and their fit.")
        self.axis_choice.currentIndexChanged.connect(self.change_viewport_axis)
        axis_row.addWidget(self.axis_choice, 1)
        layout.addLayout(axis_row)
        row = QHBoxLayout()
        self.pick = QPushButton("Add probes")
        self.pick.setCheckable(True)
        self.pick.toggled.connect(self.set_probe_mode)
        self.pick.setToolTip(
            "Enable placement, then click the mesh to add contact probes. Finish probes returns to pan. Existing probes can still be dragged or double-clicked to remove them. Rotate starts rotation and automatic correction."
        )
        self.spin = QPushButton("Rotate")
        self.spin.setCheckable(True)
        self.spin.setToolTip("Start or stop rotation (Space).")
        self.spin.toggled.connect(self.set_spin)
        self.reverse_button = QPushButton("Reverse")
        self.reverse_button.clicked.connect(self.reverse)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_export_calibration)
        self.pick.hide()
        for widget in (self.spin, self.reverse_button, self.reset_button):
            row.addWidget(widget)
        layout.addLayout(row)
        probe_actions = QHBoxLayout()
        self.clear_probes_button = QPushButton("Clear probes")
        self.clear_probes_button.setToolTip("Remove all probes and stop detection. Keep the current alignment and rotation.")
        self.clear_probes_button.clicked.connect(self.clear_all_probes)
        probe_actions.addWidget(self.clear_probes_button)
        layout.addLayout(probe_actions)
        self.status = QLabel("Place probe")
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.status)
        self.calibration_progress = CalibrationProgressBar()
        self.calibration_progress.setRange(0, 100)
        self.calibration_progress.setValue(0)
        self.calibration_progress.setFormat("Detecting %p%")
        self.calibration_progress.setToolTip("Full-turn coverage sampled every five degrees for every probe at the current fitted pose. Missing contact or a changed fit requires more coverage.")
        self.calibration_progress.setMinimumHeight(self.calibration_progress.fontMetrics().height() + 10)
        layout.addWidget(self.calibration_progress)
        self.finish = QPushButton("Finalize")
        self.finish.setEnabled(False)
        self.finish.setProperty("accent", True)
        self.finish.clicked.connect(self.finish_calibration)
        layout.addWidget(self.finish)
        section.addWidget(self.controls)
        self.axis_overlays = []
        self.axis_end_labels = {}
        for row_index, title in enumerate(("Initial axes", "Alignment axes")):
            check = QCheckBox(title, self.controls)
            check.setToolTip("Initial model X/Y/Z reference through the model center. Does not control rotation." if row_index == 0 else "Viewport axes through the alignment center. Rotation and tilt adjust their direction; Ctrl+Shift+Up/Down moves them in and out.")
            layout.addWidget(check)
            axes = []
            for name, color in zip("XYZ", ((1., .3, .3), (.3, 1., .4), (.3, .65, 1.))):
                points = vtkPoints()
                points.InsertNextPoint(0, 0, 0)
                points.InsertNextPoint(0, 0, 0)
                cells = vtkCellArray()
                if row_index == 0:
                    points.SetNumberOfPoints(50)
                    for dash in range(24):
                        cells.InsertNextCell(2)
                        cells.InsertCellPoint(dash * 2 + 2)
                        cells.InsertCellPoint(dash * 2 + 3)
                else:
                    cells.InsertNextCell(2)
                    cells.InsertCellPoint(0)
                    cells.InsertCellPoint(1)
                poly = vtkPolyData()
                poly.SetPoints(points)
                poly.SetLines(cells)
                if row_index == 0:
                    mapper = vtkPolyDataMapper2D()
                    coordinate = vtkCoordinate()
                    coordinate.SetCoordinateSystemToDisplay()
                    mapper.SetTransformCoordinate(coordinate)
                    actor = vtkActor2D()
                    actor.GetProperty().SetDisplayLocationToForeground()
                else:
                    mapper = vtkPolyDataMapper()
                    actor = vtkActor()
                    actor.GetProperty().LightingOff()
                mapper.SetInputData(poly)
                actor.SetMapper(mapper)
                display_color = tuple(.42 + .18 * value for value in color) if row_index == 0 else color
                actor.GetProperty().SetColor(*display_color)
                actor.GetProperty().SetOpacity(.65 if row_index == 0 else .4)
                actor.GetProperty().SetLineWidth(1.5)
                label = vtkTextActor()
                label.SetInput(("Initial " if row_index == 0 else "Viewport ") + name)
                label.GetTextProperty().SetColor(*display_color)
                label.GetTextProperty().SetFontSize(14 if row_index == 0 else 18)
                label.GetTextProperty().SetBold(row_index == 1)
                label.GetTextProperty().ShadowOn()
                for prop in (actor, label):
                    prop.PickableOff()
                    prop.SetVisibility(False)
                    owner.renderer.AddViewProp(prop)
                opposite_label = vtkTextActor()
                opposite_label.GetTextProperty().ShallowCopy(label.GetTextProperty())
                opposite_label.PickableOff()
                opposite_label.SetVisibility(False)
                owner.renderer.AddViewProp(opposite_label)
                self.axis_end_labels[(row_index, len(axes))] = opposite_label
                axes.append((points, actor, label))
            self.axis_overlays.append((check, axes))
            check.setChecked(True)
            check.toggled.connect(self.render_axes)
        self.fit_metrics = QLabel("Center shift -- | Tilt --")
        self.fit_metrics.setWordWrap(True)
        self.fit_metrics.setToolTip("Offset: lateral correction from the initial mass-center axis. Tilt: angular correction from the starting viewport orientation.")
        layout.addWidget(self.fit_metrics)
        self.calibration_values = QLabel("Rotation (°): X -- | Y -- | Z --\nCenter: X -- | Y -- | Z --")
        self.calibration_values.setWordWrap(True)
        self.calibration_values.setToolTip("Signed probe correction in the original model coordinate directions. Rotation is the correction rotation vector in degrees; its direction is the tilt axis and its length is the total tilt. Center is displacement of the fitted rotation center from the starting center. These values exclude turntable spin and camera movement.")
        layout.addWidget(self.calibration_values)
        self.controls.hide()
        self.timer = QTimer(owner)
        self.timer.setInterval(16)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.last_tick = None
        self.sync_controls()

    @staticmethod
    def same_mesh(first, second):
        return first is second or (first is not None and second is not None
            and first.GetAddressAsString("") == second.GetAddressAsString(""))

    def sync_controls(self):
        has_mesh = self.owner.active_actor is not None
        running = self.timer.isActive()
        adjusted = bool(np.any(np.abs(self.parameters) > 1e-10))
        if self.rig is not None and hasattr(self, "starting_rig"):
            adjusted = adjusted or not np.allclose(self.rig, self.starting_rig, atol=1e-10)
        states = (
            (self.spin, has_mesh and not self.calibration_locked),
            (self.reverse_button, running),
            (self.clear_probes_button, bool(self.probes)),
            (self.finish, self.finish.isEnabled() and not self.calibration_locked),
            (self.reset_button, bool(self.probes) or adjusted or abs(self.sweep) > 1e-8
             or self.detection_complete or self.calibration_locked),
            (self.calibration_progress, bool(self.probes) or self.detection_complete or self.calibration_locked),
        )
        changed = False
        for widget, visible in states:
            if widget.isHidden() == visible:
                widget.setVisible(visible)
                changed = True
        if changed:
            self.owner._queue_toolbox_scroll_refresh()

    def current_mesh(self):
        actor = self.owner.active_actor
        return actor.GetMapper().GetInput() if actor is not None else None

    def viewport_frame(self):
        camera = self.owner.renderer.GetActiveCamera()
        depth = -np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        depth /= np.linalg.norm(depth)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        up -= depth * float(up @ depth)
        up /= np.linalg.norm(up)
        horizontal = np.cross(up, depth)
        mode = self.axis_choice.currentData()
        axis = {"vertical": up, "horizontal": horizontal, "depth": depth}[mode]
        forward = depth if mode == "vertical" else up
        return axis, forward

    def change_viewport_axis(self, *_args):
        self.owner.settings.setValue("alignment/viewport_axis", self.axis_choice.currentData())
        self.reset()
        self.update_axes()

    def capture_rig(self):
        center = self.owner._keyboard_object_pivot()
        axis, forward = self.viewport_frame()
        right = np.cross(forward, axis)
        self.rig = np.eye(4)
        self.rig[:3, :3] = np.column_stack((right, forward, axis))
        self.rig[:3, 3] = center
        self.starting_rig = self.rig.copy()
        self.world_to_rig = np.linalg.inv(self.rig)
        matrix = self.owner.active_actor.GetMatrix()
        actor_matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
        self.model_matrix = self.world_to_rig @ actor_matrix
        if self.original_reference_frame is None:
            self.original_reference_frame = actor_matrix.copy()
            self.original_reference_frame[:3, 3] = center
        bounds = np.asarray(self.owner.active_actor.GetBounds()).reshape(3, 2)
        self.extent = max(float(np.linalg.norm(np.diff(bounds, axis=1))) * 2, 1e-6)
        self.mesh = self.current_mesh()
        self.center, self.axis = center, axis
        self._pending_manual_camera = False
        self._camera_signature = None
        self._framed_parameters = self.parameters.copy()
        self.remember_camera()

    def spin_pose(self):
        return spin_matrix(-math.radians(self.sweep)) @ pose_matrix(self.parameters)

    def original_to_spindle(self, point):
        return (self.spin_pose() @ self.world_to_rig @ np.r_[point, 1.0])[:3]

    def spindle_to_original(self, point):
        return (self.rig @ np.linalg.inv(self.spin_pose()) @ np.r_[point, 1.0])[:3]

    def remember_camera(self):
        if self.rig is None or self._applying:
            return
        camera = self.owner.renderer.GetActiveCamera()
        signature = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())
        if self._camera_signature is not None and np.allclose(
            np.asarray(signature), np.asarray(self._camera_signature), rtol=1e-12, atol=1e-9
        ):
            return
        if self._camera_signature is not None and self.button.isChecked() and not self.calibration_locked:
            if not self._pending_manual_camera:
                self._pending_manual_camera = True
                QTimer.singleShot(0, self.owner, self.apply_manual_camera)
            return
        mapping = self.spin_pose() @ self.world_to_rig
        self.base_camera = (
            (mapping @ np.r_[camera.GetPosition(), 1.0])[:3],
            (mapping @ np.r_[camera.GetFocalPoint(), 1.0])[:3],
            mapping[:3, :3] @ np.asarray(camera.GetViewUp()),
        )
        self._camera_signature = signature

    def apply_manual_camera(self, render=True, axis_offset=None):
        if not self._pending_manual_camera and axis_offset is None:
            return
        self._pending_manual_camera = False
        if self.rig is None or getattr(self.owner, "_closing", False):
            return
        camera = self.owner.renderer.GetActiveCamera()
        current = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())
        previous = self._camera_signature
        if previous is None:
            return
        def basis(state):
            position, focal, up = map(np.asarray, state)
            depth = position - focal
            depth /= np.linalg.norm(depth)
            up = up - depth * float(up @ depth)
            up /= np.linalg.norm(up)
            return np.column_stack((np.cross(up, depth), up, depth))
        rotation = basis(current) @ basis(previous).T
        delta = np.eye(4)
        delta[:3, :3] = rotation
        delta[:3, 3] = np.asarray(current[1]) - rotation @ np.asarray(previous[1])
        if axis_offset is not None:
            delta[:3, 3] += axis_offset
        self.rig = delta @ self.rig
        if not np.allclose(delta, np.eye(4), atol=1e-10):
            self.has_calibration = True
        self.world_to_rig = np.linalg.inv(self.rig)
        matrix = self.owner.active_actor.GetMatrix()
        actor_matrix = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
        self.model_matrix = self.world_to_rig @ actor_matrix
        self.probe_frames = [self.probe_frame(point) for point in self.probe_worlds]
        if not self.detection_complete:
            self.invalidate_coverage()
            self.solve_complete = False
            self.stable_fits = 0
            self.last_result = None
        self.generation += 1
        self.last_fit_time = time.monotonic()
        self._camera_signature = current
        mapping = self.spin_pose() @ self.world_to_rig
        self.base_camera = (
            (mapping @ np.r_[current[0], 1.])[:3],
            (mapping @ np.r_[current[1], 1.])[:3],
            mapping[:3, :3] @ np.asarray(current[2]),
        )
        if not np.allclose(rotation, np.eye(3), rtol=0., atol=1e-10):
            self.fit_rotation_view()
        self.rotation_frame()
        if self.reference_basis is not None:
            self.calibrate_orientation()
        self.update_axes()
        self.owner._update_orientation_indicator(force=True)
        if render:
            self.owner.renderer.ResetCameraClippingRange()
            self.owner.vtk_widget.GetRenderWindow().Render()

    def mark_manual_zoom(self):
        if self.button.isChecked():
            self.manual_zoom = True

    def fit_rotation_view(self):
        if (self.manual_zoom or not self.button.isChecked()
                or self.rig is None or self.owner.active_actor is None):
            return
        was_applying = self._applying
        self._applying = True
        try:
            camera = self.owner.renderer.GetActiveCamera()
            self.owner._fit_view_bounds(self.owner.active_actor.GetBounds())
            mapping = self.spin_pose() @ self.world_to_rig
            self.base_camera = (
                (mapping @ np.r_[camera.GetPosition(), 1.])[:3],
                (mapping @ np.r_[camera.GetFocalPoint(), 1.])[:3],
                mapping[:3, :3] @ np.asarray(camera.GetViewUp()),
            )
            self._camera_signature = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())
        finally:
            self._applying = was_applying

    def move_axis_depth(self, fraction):
        if not self.button.isChecked() or self.calibration_locked or self.rig is None:
            return False
        self.apply_manual_camera(render=False)
        camera = self.owner.renderer.GetActiveCamera()
        depth = np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        depth /= max(np.linalg.norm(depth), 1e-12)
        self.apply_manual_camera(render=False, axis_offset=depth * self.extent * .005 * fraction)
        return True

    def rotation_frame(self):
        if self.rig is None:
            return None
        if not self.same_mesh(self.current_mesh(), self.mesh):
            self.reset()
            return None
        inverse = self.rig @ np.linalg.inv(pose_matrix(self.parameters))
        self.center = inverse[:3, 3].copy()
        self.axis = inverse[:3, 2].copy()
        return self.center, self.axis

    def navigation_origin(self):
        if self.accepted_frame is None:
            return None
        if not self.same_mesh(self.current_mesh(), self.mesh):
            self.reset()
            return None
        return self.accepted_frame[0].copy()

    def navigation_frame(self):
        if self.navigation_origin() is None:
            return None
        up = np.asarray(self.owner.renderer.GetActiveCamera().GetViewUp())
        if abs(float(up @ self.accepted_frame[1])) < math.cos(math.radians(0.5)):
            return None
        return self.accepted_frame

    def toggle(self, enabled):
        self.controls.setVisible(enabled)
        if enabled:
            if self.owner.active_actor is None or self.owner.busy:
                self.button.setChecked(False)
                return
            self.manual_zoom = False
            self.set_spin(False)
            self.fit_current_view()
            if self.rotation_frame() is None:
                self.capture_rig()
            self.remember_camera()
            self.ensure_contact_index()
            self.pick.setChecked(True)
            self.set_spin(False)
            self.status.setText("Place probes")
            self.draw_probe()
        else:
            self.pick.setChecked(False)
            self.set_spin(False)
            self.remove_probe()
        self.update_axes()
        self.owner._position_view_pad()
        if enabled:
            QTimer.singleShot(0, self.owner, self.fit_current_view)

    def fit_current_view(self):
        if getattr(self.owner, "_closing", False):
            return
        if not self.button.isChecked() or self.owner.active_actor is None:
            return
        self._applying = True
        try:
            self.owner._fit_view_bounds(self.owner.active_actor.GetBounds())
            self.owner.renderer.ResetCameraClippingRange()
        finally:
            self._applying = False
        if self.rig is not None:
            self._camera_signature = None
            self.remember_camera()
        self.owner._update_orientation_indicator()
        self.owner.vtk_widget.GetRenderWindow().Render()

    def set_probe_mode(self, enabled):
        self.pick.setText("Finish probes" if enabled else "Add probes")
        self.owner.vtk_widget.setCursor(Qt.CursorShape.ArrowCursor)

    def set_spin(self, spinning):
        if (spinning and self.rig is not None and not self.probes and not self.detection_complete
                and not np.any(self.parameters)):
            # Capture at Rotate, after the user has oriented the viewport.
            self.sweep = 0.0
            self.parameters = np.zeros(4)
            self.target_parameters = np.zeros(4)
            self.capture_rig()
        with QSignalBlocker(self.spin):
            self.spin.setChecked(spinning)
        self.spin.setText("Stop rotation" if spinning and self.rig is not None else "Rotate")
        if spinning and self.rig is not None:
            self.last_tick = time.monotonic()
            self.timer.start()
        else:
            self.timer.stop()
        self.sync_controls()

    def reverse(self):
        self.direction *= -1

    def reset_export_calibration(self):
        self.owner.export_calibration_transform = None
        self.reset()

    def reset(self):
        self.set_spin(False)
        self.generation += 1
        self.center = self.axis = self.rig = self.world_to_rig = self.model_matrix = None
        self.mesh = self.position = None
        self.probes.clear()
        self.probe_frames.clear()
        self.probe_worlds.clear()
        self.probe_travel.clear()
        self.samples.clear()
        self.sample_groups.clear()
        self.parameters = np.zeros(4)
        self.target_parameters = np.zeros(4)
        self.accepted_frame = None
        self.original_reference_frame = None
        self.has_calibration = False
        self.retained_probe_frames = []
        self.dragging, self.active_probe = False, None
        self.sweep = 0.0
        self.stable_fits = 0
        self.last_result = None
        self.wobble_baseline = None
        self.solve_complete = False
        self.reference_basis = None
        if self.previous_views is not None:
            self.owner.orientation_reference_basis = self.previous_reference
            self.owner.saved_standard_views = self.previous_views
        self.previous_views = self.previous_reference = None
        self.calibrated_revision = -1
        self._calibrated_parameters = None
        self.revision = 0
        self.remove_probe()
        self.invalidate_coverage()
        self.status.setText("Place probes")
        self.pick.setChecked(False)
        if self.button.isChecked() and self.owner.active_actor is not None:
            self.capture_rig()

    def probe_frame(self, point):
        point = self.original_to_spindle(point)
        radius = max(float(np.linalg.norm(point[:2])), 1e-9)
        return (float(point[2]), float(point[0] / radius), float(point[1] / radius), radius)

    def interaction(self, phase, position):
        if not self.button.isChecked() or self.owner.busy or self.calibration_locked:
            return False
        position = np.asarray(position, dtype=float)
        if phase == "press":
            resumed = self.detection_complete
            if resumed:
                if self.owner._pick_measure_point(tuple(position)) is None:
                    return True
                self.probe_frames = list(self.retained_probe_frames)
                self.probe_worlds = [self.spindle_to_original(np.array((r*x,r*y,h))) for h,x,y,r in self.probe_frames]
                self.probes = [np.asarray(self.owner._world_to_display(p)) for p in self.probe_worlds]
                self.probe_travel = [[f[3],time.monotonic()] for f in self.probe_frames]
            nearest = next(
                (i for i, point in enumerate(self.probes) if np.linalg.norm(position - point) < 18),
                None,
            )
            if resumed:
                nearest = None
            self._probe_edit_snapshot = None
            if nearest is not None:
                self._probe_edit_snapshot = (
                    self.probes[nearest].copy(),
                    self.probe_worlds[nearest].copy(),
                    self.probe_frames[nearest],
                    list(self.probe_travel[nearest]),
                )
            if nearest is None:
                if not self.probes:
                    up, _ = self.viewport_frame()
                    if not resumed and not np.allclose(up, self.rig[:3, 2], atol=1e-8):
                        self.sweep = 0.0
                        self.capture_rig()
                point = self.owner._pick_measure_point(tuple(position))
                if point is None:
                    return True
                self.active_probe = len(self.probes)
                self.probes.append(position.copy())
                self.probe_worlds.append(point.copy())
                self.probe_frames.append(self.probe_frame(point))
                self.probe_travel.append([self.probe_frames[-1][3], time.monotonic()])
            else:
                self.active_probe = nearest
                self._probe_click = (nearest, position.copy(), time.monotonic())
            self._probe_press_position = position.copy()
            self._resume_after_probe = self.timer.isActive()
            self.set_spin(False)
            self.position = position.copy()
            self.dragging = True
            self.changed_probes()
            self.draw_probe()
            return True
        if not self.dragging:
            return False
        if phase == "move":
            self.position = position.copy()
            self.probes[self.active_probe] = position.copy()
            self.draw_probe()
        elif phase == "release":
            self.dragging = False
            if (self._probe_edit_snapshot is not None
                    and np.linalg.norm(position - self._probe_press_position) < 3.):
                index = self.active_probe
                screen, world, frame, travel = self._probe_edit_snapshot
                self.probes[index], self.probe_worlds[index] = screen, world
                self.probe_frames[index], self.probe_travel[index] = frame, travel
                self.active_probe = None
                self.draw_probe()
                self.set_spin(self._resume_after_probe)
                return True
            self._probe_click = None
            point = self.owner._pick_measure_point(tuple(position))
            if point is None:
                index = self.active_probe
                if self._probe_edit_snapshot is None:
                    for values in (
                        self.probes,
                        self.probe_worlds,
                        self.probe_frames,
                        self.probe_travel,
                    ):
                        values.pop(index)
                else:
                    screen, world, frame, travel = self._probe_edit_snapshot
                    self.probes[index], self.probe_worlds[index] = screen, world
                    self.probe_frames[index], self.probe_travel[index] = frame, travel
                self.active_probe = None
                self.position = self.probes[-1].copy() if self.probes else None
                self.draw_probe()
                self.set_spin(self._resume_after_probe)
                self.status.setText("No surface contact")
                return True
            self.probe_worlds[self.active_probe] = point.copy()
            self.probe_frames[self.active_probe] = self.probe_frame(point)
            self.probe_travel[self.active_probe] = [
                self.probe_frames[self.active_probe][3],
                time.monotonic(),
            ]
            self.samples.append(point)
            self.sample_groups.append(self.active_probe)
            self.changed_probes()
            self.draw_probe()
            self.set_spin(self._resume_after_probe)
        return True

    def invalidate_coverage(self):
        self.coverage = np.zeros((len(self.probes), 72), dtype=bool)
        self.calibration_locked = False
        self.detection_complete = False
        self.timer.setInterval(16)
        self.owner.navigation_style._calibrated_inspection = False
        self.finish.setEnabled(False)
        self.spin.setEnabled(True)
        self.calibration_progress.setValue(0)
        self.calibration_progress.setFormat("Detecting %p%")

    def update_calibration_state(self, settled):
        if self.calibration_locked or self.detection_complete:
            return
        if self.coverage.shape != (len(self.probes), 72):
            self.invalidate_coverage()
        if not self.probes:
            self.calibration_progress.setFormat("Add probes")
            return
        if settled and not self.dragging:
            phase_bin = int(self.sweep // 5) % 72
            for index in range(len(self.probes)):
                if not self.coverage[index, phase_bin]:
                    self.coverage[index, phase_bin] = self.finger_contact(index) is not None
        percent = int(100 * self.coverage.sum() / self.coverage.size)
        self.calibration_progress.setValue(percent)
        # Completion means the validated solver has no useful work left, not
        # that an imperfect scanned surface can attain numerically zero wobble.
        stable = (percent == 100 and settled and self.last_result is not None
                  and self.solve_complete
                  and self.future is None)
        moving = not settled
        state = "Stable" if stable else ("Adjusting" if moving else "Detecting")
        if stable and self.last_result.rank < 4:
            state = "Stable (center only)"
        elif percent == 100 and not stable:
            state = "Adjusting" if moving else "Validating"
        self.calibration_progress.setFormat(state + ("" if stable else " %p%"))
        self.finish.setEnabled(stable)
        if stable:
            self.calibrate_orientation()
            self.detection_complete = True
            self.retained_probe_frames = list(self.probe_frames)
            self.timer.setInterval(8)
            self.owner.navigation_style._calibrated_inspection = True
            self.clear_probe_visuals()
            self.status.setText("Ready to finalize")
        self.sync_controls()

    def clear_probe_visuals(self):
        self.probes.clear()
        self.probe_worlds.clear()
        self.probe_frames.clear()
        self.probe_travel.clear()
        self.samples.clear()
        self.sample_groups.clear()
        self.position = None
        self.remove_probe(render=False)

    def finish_calibration(self):
        if not self.finish.isEnabled():
            return
        self.calibrate_orientation()
        self.set_spin(False)
        # The corrected frame, independent of the inspection spin angle, becomes
        # the common document-to-export frame for every supported file format.
        self.owner.export_calibration_transform = (
            pose_matrix(self.parameters) @ np.linalg.inv(self.rig))
        self.owner.mesh_modified = True
        self.owner._sync_modified_mesh_actions()
        self.calibration_locked = True
        self.spin.setEnabled(False)
        self.finish.setEnabled(False)
        self.calibration_progress.setFormat("Calibrated")
        self.status.setText("Calibrated")
        self.clear_probe_visuals()
        self.button.setChecked(False)
        self.owner.optimization_toggle.setChecked(True)
        self.update_axes()
        self.owner.vtk_widget.GetRenderWindow().Render()

    def clear_all_probes(self):
        self.generation += 1
        for name in ("future",):
            future = getattr(self, name, None)
            if future is not None:
                future.cancel()
            setattr(self, name, None)
        self.dragging, self.active_probe = False, None
        self.clear_probe_visuals()
        self.target_parameters = self.parameters.copy()
        self.coverage = np.zeros((0, 72), dtype=bool)
        self.solve_complete = True
        self.stable_fits = 0
        self.detection_complete = False
        self.finish.setEnabled(False)
        self.calibration_progress.setValue(0)
        self.calibration_progress.setFormat("Add probes")
        self.status.setText("Probes cleared")
        self.update_axes()
        self.owner.vtk_widget.GetRenderWindow().Render()

    def changed_probes(self):
        self.invalidate_coverage()
        self.generation += 1
        self.revision += 1
        self.wobble_baseline = None
        self.stable_fits = 0
        self.last_fit_time = -math.inf
        self.solve_complete = False
        self.sync_controls()

    def ensure_contact_index(self):
        mesh = self.current_mesh()
        if mesh is None:
            return False
        if not self.same_mesh(mesh, self.locator_mesh):
            if self.locator_future is not None:
                self.locator_future.cancel()
            self.locator_mesh, self.locator = mesh, None
            self.locator_future = self.owner.executor.submit(build_contact_index, mesh)
        if self.locator_future is not None and self.locator_future.done():
            self.locator = self.locator_future.result()
            self.locator_future = None
        return self.locator is not None

    def finger_contact(self, index, offset=0.0):
        if not self.ensure_contact_index():
            return None
        height, x, y, radius = self.probe_frames[index]
        direction = np.array((x, y, 0.0))
        direction /= np.linalg.norm(direction)
        station = np.array((0.0, 0.0, height + radius * offset))
        contact = surface_contact(
            self.locator,
            station + direction * self.extent,
            station - direction * self.extent,
            self.spin_pose() @ self.model_matrix,
        )
        if contact is None or float(contact @ direction) <= 0:
            return None
        return self.spindle_to_original(contact)

    def apply_camera(self):
        mapping = self.rig @ np.linalg.inv(self.spin_pose())
        camera = self.owner.renderer.GetActiveCamera()
        self._applying = True
        try:
            camera.SetPosition(*(mapping @ np.r_[self.base_camera[0], 1.0])[:3])
            camera.SetFocalPoint(*(mapping @ np.r_[self.base_camera[1], 1.0])[:3])
            camera.SetViewUp(*(mapping[:3, :3] @ self.base_camera[2]))
            camera.OrthogonalizeViewUp()
            if not np.allclose(self.parameters, self._framed_parameters, rtol=0., atol=1e-10):
                self.has_calibration = True
                self._framed_parameters = self.parameters.copy()
            self.fit_rotation_view()
            self._camera_signature = (
                camera.GetPosition(),
                camera.GetFocalPoint(),
                camera.GetViewUp(),
            )
            self.rotation_frame()
            if self.reference_basis is not None:
                self.reference_basis = (self.rig @ np.linalg.inv(pose_matrix(self.parameters)))[
                    :3, :3
                ]
                self.owner.orientation_reference_basis = self.reference_basis.copy()
            self.owner._update_orientation_indicator(force=True)
            self.owner.renderer.ResetCameraClippingRange()
            self.owner.vtk_widget.GetRenderWindow().Render()
        finally:
            self._applying = False

    def tick(self):
        self.apply_manual_camera(render=False)
        if (
            self.owner.busy
            or getattr(self.owner, "_closing", False)
            or self.rotation_frame() is None
        ):
            self.set_spin(False)
            return
        now = time.monotonic()
        elapsed = min(0.08, max(0.0, now - (self.last_tick or now)))
        self.last_tick = now
        sign = -1 if self.owner.keyboard_horizontal else 1
        self.sweep = (self.sweep + elapsed * 36 * self.direction * sign) % 360
        self.parameters += (self.target_parameters - self.parameters) * (
            1 - math.exp(-elapsed / 0.12)
        )
        scale = np.array((self.extent, self.extent, 1.0, 1.0))
        settled = np.linalg.norm((self.target_parameters - self.parameters) / scale) < 1e-6
        if settled:
            self.parameters = self.target_parameters.copy()
        self.apply_camera()
        if self.detection_complete:
            return
        if not self.probes:
            self.status.setText(f"{int(self.sweep)}\N{DEGREE SIGN} | Place probes")
            return
        if not self.ensure_contact_index():
            self.status.setText("Preparing probes")
            return
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                if self.future_generation == self.generation:
                    self.last_result = result
                    if self.wobble_baseline is None:
                        self.wobble_baseline = float(max(result.before))
                    if np.linalg.norm((result.parameters - self.target_parameters) / scale) > 1e-6:
                        self.invalidate_coverage()
                    self.target_parameters = result.parameters.copy()
                    noise = result.noise_floor
                    limits = np.full_like(result.after, self.extent * 1e-4)
                    if noise is not None:
                        limits = np.maximum(limits, noise)
                    converged = bool(np.all(result.after <= limits))
                    self.stable_fits = self.stable_fits + 1 if result.rank == 4 and converged else 0
                    self.solve_complete = not result.accepted
            except (ValueError, np.linalg.LinAlgError) as error:
                self.status.setText(str(error))
                self.last_result = None
            self.future = None
        if self.last_result is not None:
            unit = self.owner.units.currentText()
            factor = self.unit_scales[unit]
            wobble = max(self.last_result.after) / factor
            surface = max(self.last_result.total_runout) / factor
            height_note = " | 1 height" if self.last_result.rank < 4 else ""
            self.status.setText(
                f"{int(self.sweep)}\N{DEGREE SIGN} | Wobble {wobble:.3f} {unit}{height_note}"
            )
            self.status.setToolTip(
                f"Total surface variation: {surface:.3f} {unit}. Probes at separated heights determine axial tilt; probes at one height determine centering. Surface shape is preserved."
            )
        settled_after_result = np.linalg.norm((self.target_parameters - self.parameters) / scale) < 1e-6
        self.update_calibration_state(settled_after_result)
        accepted_axis = (self.last_result is not None and self.last_result.rank == 4
                         and (self.last_result.accepted or self.stable_fits >= 2))
        axis_changed = (self.calibrated_revision != self.revision
                        or self._calibrated_parameters is None
                        or not np.allclose(self.parameters, self._calibrated_parameters, rtol=0., atol=1e-10))
        if accepted_axis and settled_after_result and axis_changed:
            self.calibrate_orientation()
        if (
            not self.dragging
            and not self.solve_complete
            and settled
            and self.future is None
            and now - self.last_fit_time > 0.25
        ):
            self.future_generation = self.generation
            self.last_fit_time = now
            self.future = self.owner.executor.submit(
                fit_probe_datum,
                self.locator,
                self.model_matrix.copy(),
                self.parameters.copy(),
                tuple(self.probe_frames),
                self.extent,
            )

    def calibrate_orientation(self):
        if self.previous_views is None:
            previous = self.owner.orientation_reference_basis
            self.previous_reference = previous.copy() if previous is not None else None
            self.previous_views = {
                key: (np.asarray(view).copy(), np.asarray(up).copy())
                for key, (view, up) in self.owner.saved_standard_views.items()
            }
        mapping = self.rig @ np.linalg.inv(pose_matrix(self.parameters))
        self.reference_basis = mapping[:3, :3].copy()
        self.owner.orientation_reference_basis = self.reference_basis.copy()
        right, forward, axis = self.reference_basis.T
        self.owner.saved_standard_views = {
            "front": (-forward, axis),
            "back": (forward, axis),
            "left": (-right, axis),
            "right": (right, axis),
            "top": (axis, -forward),
            "bottom": (-axis, -forward),
        }
        self.calibrated_revision = self.revision
        self._calibrated_parameters = self.parameters.copy()
        self.rotation_frame()
        self.accepted_frame = (self.center.copy(), self.axis.copy())
        self.owner._update_orientation_indicator()

    def display_orientation(self, camera):
        if self.reference_basis is None or self.rotation_frame() is None:
            return None
        forward = np.asarray(camera.GetDirectionOfProjection())
        up = np.asarray(camera.GetViewUp())
        axis = self.reference_basis[:, 2]
        expected_up = axis - forward * float(axis @ forward)
        expected_up /= max(np.linalg.norm(expected_up), 1e-12)
        right = np.cross(forward, expected_up)
        pitch = np.degrees(np.arcsin(np.clip(forward @ axis, -1.0, 1.0)))
        roll = np.degrees(np.arctan2(up @ right, up @ expected_up))
        heading = np.degrees(
            np.arctan2(-forward @ self.reference_basis[:, 0], forward @ self.reference_basis[:, 1])
        )
        return float(pitch), float(roll), float(heading)

    def delete_probe(self, position):
        if not self.button.isChecked():
            return False
        index = next(
            (
                i
                for i, point in enumerate(self.probes)
                if np.linalg.norm(np.asarray(position) - point) < 18
            ),
            None,
        )
        if index is None:
            clicked = getattr(self, "_probe_click", None)
            if (clicked is not None and clicked[0] < len(self.probes)
                    and time.monotonic() - clicked[2] <= QApplication.doubleClickInterval() / 1000.
                    and np.linalg.norm(np.asarray(position) - clicked[1]) < 18):
                index = clicked[0]
            else:
                return False
        self._probe_click = None
        resume = self.timer.isActive() or (self.dragging and self._resume_after_probe)
        self.probes.pop(index)
        self.probe_worlds.pop(index)
        self.probe_frames.pop(index)
        self.probe_travel.pop(index)
        retained = [
            (point, group - (group > index))
            for point, group in zip(self.samples, self.sample_groups)
            if group != index
        ]
        self.samples = [point for point, _ in retained]
        self.sample_groups = [group for _, group in retained]
        self.dragging, self.active_probe = False, None
        self.position = self.probes[-1].copy() if self.probes else None
        self.changed_probes()
        self.draw_probe()
        self.set_spin(resume)
        return True

    def render_axes(self, *_args):
        self.update_axes()
        if not self._applying:
            self.owner.vtk_widget.GetRenderWindow().Render()

    def probe_calibration_values(self):
        if self.rig is None or self.original_reference_frame is None:
            return None
        model_basis = self.original_reference_frame[:3, :3].copy()
        model_basis /= np.linalg.norm(model_basis, axis=0)
        corrected_frame = self.rig @ np.linalg.inv(pose_matrix(self.parameters))
        baseline = getattr(self, "starting_rig", self.rig)
        relative = corrected_frame[:3, :3] @ baseline[:3, :3].T
        angle = math.acos(float(np.clip((np.trace(relative) - 1) / 2, -1., 1.)))
        vector = np.array((relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]))
        rotation = model_basis.T @ (vector * (.5 if angle < 1e-8 else angle / (2 * math.sin(angle))))
        shift = model_basis.T @ (corrected_frame[:3, 3] - baseline[:3, 3])
        return np.degrees(rotation), shift

    def update_axes(self):
        self.sync_controls()
        axis_names = "XYZ"
        calibrated = (self.rig @ np.linalg.inv(pose_matrix(self.parameters))
                      if self.rig is not None else None)
        camera = self.owner.renderer.GetActiveCamera()
        depth = -np.asarray(camera.GetDirectionOfProjection(), dtype=float)
        depth /= max(np.linalg.norm(depth), 1e-12)
        up = np.asarray(camera.GetViewUp(), dtype=float)
        up -= depth * float(up @ depth)
        up /= max(np.linalg.norm(up), 1e-12)
        viewport_basis = np.column_stack((np.cross(up, depth), up, depth))
        calibration = self.probe_calibration_values()
        if calibration is None:
            self.calibration_values.setText("Center: X -- | Y -- | Z --")
            self.calibration_progress.set_rotation("X -- | Y -- | Z --")
        else:
            angles, shift = calibration
            unit = self.owner.units.currentText()
            shift = shift / self.unit_scales[unit]
            rotation_text = " | ".join(f"{name} {value:+.3f}" for name, value in zip("XYZ", angles))
            shift_text = " | ".join(f"{name} {value:+.3f}" for name, value in zip("XYZ", shift))
            self.calibration_values.setText(f"Center ({unit}): {shift_text}")
            self.calibration_progress.set_rotation(rotation_text + "°")
        if self.rig is not None:
            unit = self.owner.units.currentText()
            offset = np.linalg.norm(self.parameters[:2]) / self.unit_scales[unit]
            tilt = np.degrees(np.arccos(np.clip(pose_matrix(self.parameters)[2, 2], -1, 1)))
            change = f"Center shift {offset:.3f} {unit} | Tilt {tilt:.2f}°"
            if self.last_result is not None and self.wobble_baseline is not None:
                before = self.wobble_baseline / self.unit_scales[unit]
                after = float(max(self.last_result.after)) / self.unit_scales[unit]
                change += f"\nWobble {before:.3f} -> {after:.3f} {unit}"
                if before > 1e-12:
                    change += f" ({(after / before - 1) * 100:+.1f}%)"
            self.fit_metrics.setText(change)
        else:
            self.fit_metrics.setText("Center shift -- | Tilt --")
        for index, (check, axes) in enumerate(self.axis_overlays):
            active = self.button.isChecked() and check.isChecked() and self.rig is not None
            frame = None
            if active:
                if index == 0:
                    # Model coordinate axes are a reference overlay, not the spindle frame.
                    frame = self.original_reference_frame.copy()
                else:
                    frame = calibrated.copy()
                    frame[:3, :3] = viewport_basis
            for axis_index, (points, actor, label) in enumerate(axes):
                opposite_label = self.axis_end_labels[(index, axis_index)]
                if active:
                    for prop in (actor, label, opposite_label):
                        if not self.owner.renderer.HasViewProp(prop):
                            self.owner.renderer.AddViewProp(prop)
                visible = active and (index == 0 or axis_index < 2)
                actor.SetVisibility(visible)
                label.SetVisibility(visible and index == 0)
                opposite_label.SetVisibility(visible and index == 0)
                if not visible:
                    continue
                name = axis_names[axis_index]
                reference_index = "XYZ".index(name)
                center = frame[:3, 3]
                direction = frame[:3, reference_index if index == 0 else axis_index]
                first = self.owner._world_to_display(center - direction * self.extent * .3)
                second = self.owner._world_to_display(center + direction * self.extent * .3)
                if first is None or second is None:
                    actor.SetVisibility(False)
                    label.SetVisibility(False)
                    opposite_label.SetVisibility(False)
                    continue
                width, height = self.owner.renderer.GetSize()
                ends = viewport_axis_ends(first, second, width, height)
                if ends is not None:
                    first, second = ends
                if index == 0:
                    points.SetPoint(0, *first, 0.)
                    points.SetPoint(1, *second, 0.)
                    start, end = np.asarray(first), np.asarray(second)
                    for dash in range(24):
                        points.SetPoint(2 * dash + 2, *(start + (end - start) * (dash / 24)), 0.)
                        points.SetPoint(2 * dash + 3, *(start + (end - start) * ((dash + .55) / 24)), 0.)
                else:
                    self.owner.renderer.SetWorldPoint(*center, 1.)
                    self.owner.renderer.WorldToDisplay()
                    depth_value = self.owner.renderer.GetDisplayPoint()[2]
                    for point_index, endpoint in enumerate((first, second)):
                        self.owner.renderer.SetDisplayPoint(*endpoint, depth_value)
                        self.owner.renderer.DisplayToWorld()
                        world = np.asarray(self.owner.renderer.GetWorldPoint())
                        points.SetPoint(point_index, *(world[:3] / world[3]))
                points.Modified()
                if index == 0:
                    if axis_index == 2:
                        cosine = float(direction @ viewport_basis[:, 2]) / max(np.linalg.norm(direction), 1e-12)
                        angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
                    else:
                        screen_direction = viewport_basis[:, :2].T @ direction
                        angle = np.degrees(np.arctan2(screen_direction[1], screen_direction[0])) - axis_index * 90.
                        angle = (angle + 180.) % 360. - 180.
                    label.SetInput(f"{name}:{angle:.2f}".rstrip("0").rstrip(".") + "\N{DEGREE SIGN}")
                else:
                    label.SetInput(name)
                opposite_label.SetInput(label.GetInput())
                for endpoint_label, endpoint in ((label, second), (opposite_label, first)):
                    label_size = [0., 0.]
                    endpoint_label.GetSize(self.owner.renderer, label_size)
                    margin = 8.
                    x = np.clip(endpoint[0] + 5, margin, max(margin, width - label_size[0] - margin))
                    y = np.clip(endpoint[1] + 5, margin, max(margin, height - label_size[1] - margin))
                    endpoint_label.SetPosition(float(x), float(y))

    def refresh_probe_positions(self):
        self.update_axes()
        if self.button.isChecked() and self.probes:
            self.draw_probe(render=False)

    def remove_probe(self, render=True):
        for actor in self.actors:
            self.owner.renderer.RemoveViewProp(actor)
        self.actors.clear()
        self._probe_drawings.clear()
        if render and not getattr(self.owner, "_closing", False):
            self.owner.vtk_widget.GetRenderWindow().Render()

    def draw_probe(self, render=True):
        if self.position is None or not self.button.isChecked():
            self.remove_probe(render=False)
            return
        if len(self._probe_drawings) != len(self.probe_frames):
            self.remove_probe(render=False)
            for _ in self.probe_frames:
                positions = np.zeros((70, 3), dtype=np.float32)
                points = vtkPoints()
                points.SetData(numpy_to_vtk(positions, deep=False))
                cells = vtkCellArray()
                for first in range(0, 70, 2):
                    cells.InsertNextCell(2)
                    cells.InsertCellPoint(first)
                    cells.InsertCellPoint(first + 1)
                poly = vtkPolyData()
                poly.SetPoints(points)
                poly.SetLines(cells)
                coordinate = vtkCoordinate()
                coordinate.SetCoordinateSystemToDisplay()
                mapper = vtkPolyDataMapper2D()
                mapper.SetInputData(poly)
                mapper.SetTransformCoordinate(coordinate)
                actors = []
                for width in (7, 3):
                    actor = vtkActor2D()
                    actor.SetMapper(mapper)
                    actor.PickableOff()
                    actor.GetProperty().SetLineWidth(width)
                    actors.append(actor)
                    self.actors.append(actor)
                    self.owner.renderer.AddViewProp(actor)
                actors[0].GetProperty().SetColor(0.02, 0.07, 0.13)
                self._probe_drawings.append((positions, points, actors))
        for index, frame in enumerate(self.probe_frames):
            height, x, y, radius = frame
            direction = np.array((x, y, 0.0))
            direction /= np.linalg.norm(direction)
            datum = np.array((radius * x, radius * y, height))
            datum_screen = self.owner._world_to_display(self.spindle_to_original(datum))
            valid = True
            if self.dragging and index == self.active_probe:
                hit = self.position.copy()
            else:
                contact = self.finger_contact(index)
                valid = contact is not None
                while len(self.probe_travel) <= index:
                    self.probe_travel.append([radius, time.monotonic()])
                travel, last = self.probe_travel[index]
                now = time.monotonic()
                if valid:
                    # Smooth the displayed finger only. Datum fitting uses the
                    # exact intersections, not these visual travel values.
                    distance = float(self.original_to_spindle(contact) @ direction)
                    travel += (distance - travel) * (
                        1 - math.exp(-min(0.05, max(0.0, now - last)) / 0.08)
                    )
                    self.probe_worlds[index] = contact
                self.probe_travel[index] = [travel, now]
                shown = self.spindle_to_original(np.array((0.0, 0.0, height)) + direction * travel)
                screen = self.owner._world_to_display(shown)
                if screen is None:
                    for actor in self._probe_drawings[index][2]:
                        actor.SetVisibility(False)
                    continue
                hit = np.asarray(screen)
                self.probes[index] = hit
            body = (
                np.asarray(datum_screen) + (-40.0, 25.0)
                if datum_screen is not None
                else hit + (-40.0, 25.0)
            )
            positions, points, actors = self._probe_drawings[index]
            angles = np.arange(33) * (math.tau / 32)
            ring = hit + 14 * np.column_stack((np.cos(angles), np.sin(angles)))
            positions[:64:2, :2] = ring[:-1]
            positions[1:64:2, :2] = ring[1:]
            arrow = (hit - body) / max(np.linalg.norm(hit - body), 1e-12)
            side = np.array((-arrow[1], arrow[0]))
            positions[64:, :2] = (
                body, hit, hit, hit - arrow * 9 + side * 4,
                hit, hit - arrow * 9 - side * 4,
            )
            points.GetData().Modified()
            points.Modified()
            foreground = (0.75, 0.97, 0.25) if valid else (1.0, 0.65, 0.15)
            actors[1].GetProperty().SetColor(*foreground)
            for actor in actors:
                actor.SetVisibility(True)
        if render:
            self.owner.vtk_widget.GetRenderWindow().Render()

    def close(self):
        self.timer.stop()
        self.generation += 1
        if self.future is not None:
            self.future.cancel()
        if self.locator_future is not None:
            self.locator_future.cancel()
