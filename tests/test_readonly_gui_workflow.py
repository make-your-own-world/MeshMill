"""Exercise real Qt/VTK state transitions with isolated preferences."""
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSettings, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QLabel, QLineEdit, QMessageBox
from test_operation_progress import sphere
from vtkmodules.vtkIOGeometry import vtkSTLWriter
from worker_lifecycle_cases import slow_task

import meshmill as mm


def wait(app, predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("GUI operation did not settle")
        QTest.qWait(5)
    app.processEvents()


def settled(window):
    return (not window.busy and getattr(window, "viewport_pending", None) is None
            and window.pending_density_poly is None and window.active_actor is not None)


@pytest.fixture
def gui(tmp_path, monkeypatch, request):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    settings.setValue("optimization/last_algorithm", "Fast QEM")
    settings.setValue("display/type", "Shaded")
    if hasattr(request, "param"):
        settings.setValue("shortcuts/toggle_fullscreen", request.param)
    monkeypatch.setattr(mm, "QSettings", lambda *_: settings)
    monkeypatch.setattr(mm, "cuda_devices", lambda: [])
    monkeypatch.setattr(mm, "gpu_backend_supported", lambda *_: False)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(str(args[-1])))
    window = mm.MainWindow()
    window.resize(1300, 850)
    window.show()
    points, faces = sphere()
    path = tmp_path / "mesh.stl"
    writer = vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(mm.arrays_polydata(points, faces))
    writer.SetFileTypeToBinary()
    assert writer.Write()
    window.load_path(path)
    wait(app, lambda: settled(window))
    window.timer.stop()
    yield app, window, errors
    window.timer.stop()
    window.close()
    app.processEvents()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not errors


def test_ctrl_double_click_selects_only_connected_component(gui, monkeypatch):
    app, window, _ = gui
    points, faces = window.source_points.copy(), window.source_faces.copy()
    offset = np.array([np.ptp(points[:, 0])*3, 0., 0.])
    window.source_points = np.vstack((points, points+offset))
    window.source_faces = np.vstack((faces, faces+len(points)))
    window.source_poly = mm.arrays_polydata(window.source_points, window.source_faces)
    monkeypatch.setattr(window, "_pick_measure_point", lambda _: points[0])
    QTest.mouseDClick(window.vtk_widget, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ControlModifier, QPoint(100, 100))
    wait(app, lambda: not window.busy and window.crop_candidate is not None)
    mask = window.crop_candidate[3]
    assert mask[:len(faces)].all()
    assert not mask[len(faces):].any()
    monkeypatch.setattr(window, "_pick_measure_point", lambda _: points[0]+offset)
    window._select_connected_at((100,100))
    wait(app, lambda: not window.busy and window.crop_candidate[3][len(faces):].all())
    assert window.crop_candidate[3][:len(faces)].all()
    window._select_connected_at((100,100))
    wait(app, lambda: not window.busy and not window.crop_candidate[3][len(faces):].any())
    assert window.crop_candidate[3][:len(faces)].all()
    assert not window.crop_points
    assert np.array_equal(window.source_faces, np.vstack((faces, faces+len(points))))


def test_camera_rotation_clears_ruler_but_preserves_selection(gui):
    app, window, _ = gui
    mask = np.zeros(len(window.source_faces), dtype=bool)
    mask[:20] = True
    window.selected_face_mask = mask.copy()
    window.measure_points = [window.source_points[0].copy(), window.source_points[1].copy()]
    window._update_orientation_indicator()
    camera = window.renderer.GetActiveCamera()
    camera.Azimuth(10)
    app.processEvents()
    assert not window.measure_points
    assert window.measure_hover_world is None
    assert np.array_equal(window.selected_face_mask, mask)


def test_repeated_arrow_rotation_keeps_pivot_and_zoom(gui, monkeypatch):
    app, window, _ = gui
    camera = window.renderer.GetActiveCamera()
    calls = []
    pivot = np.asarray(camera.GetFocalPoint()) + np.array([.1, .2, .3])
    def pick():
        calls.append(True)
        return pivot.copy() + np.array([0., 0., len(calls)*.01])
    monkeypatch.setattr(window, "_keyboard_view_pivot", pick)
    window._keyboard_navigate("right", False, False)
    fixed = np.asarray(camera.GetFocalPoint())
    position = np.asarray(camera.GetPosition())
    radius, scale, angle = camera.GetDistance(), camera.GetParallelScale(), camera.GetViewAngle()
    for key in ("right", "up"):
        start = np.asarray(camera.GetPosition())
        for _ in range(72):
            assert window._keyboard_navigate(key, False, False)
            assert np.allclose(camera.GetFocalPoint(), fixed, atol=1e-9)
            assert np.isclose(camera.GetDistance(), radius, rtol=1e-10)
            assert camera.GetParallelScale() == scale
            assert camera.GetViewAngle() == angle
        assert np.allclose(camera.GetPosition(), start, atol=1e-8)
    assert len(calls) == 1
    window._keyboard_navigate("right", True, False)
    window._keyboard_navigate("left", False, False)
    assert len(calls) == 2


def test_navigation_smoothing_is_bounded_and_stops(gui, monkeypatch):
    app, window, _ = gui
    steps = []
    monkeypatch.setattr(window, "_keyboard_navigate",
                        lambda *args, **kwargs: steps.append((args, kwargs["fraction"])))
    window._queue_keyboard_navigation("right", False, False)
    wait(app, lambda: not window.navigation_timer.isActive())
    assert len(steps) == 4
    assert sum(fraction for _, fraction in steps) == 1
    window._queue_keyboard_navigation("right", False, False)
    window._queue_keyboard_navigation("left", False, False)
    start = len(steps)
    wait(app, lambda: not window.navigation_timer.isActive())
    assert all(args[0] == "left" for args, _ in steps[start:])
    before = len(steps)
    QTest.qWait(100)
    assert len(steps) == before
    style = window.navigation_style
    zoom = []
    monkeypatch.setattr(style, "_zoom_at_cursor",
                        lambda factor, **kwargs: zoom.append(factor))
    style._queue_wheel("zoom", 1.2)
    wait(app, lambda: not style._wheel_timer.isActive())
    assert len(zoom) == 4
    assert np.isclose(np.prod(zoom), 1.2)


def test_update_indicator_manual_check_and_installer_handoff(gui, monkeypatch, tmp_path):
    app, window, _ = gui
    assert not window.update_checks_enabled
    release = {"version": "0.3.1", "release_url": mm.REPOSITORY_URL+"/releases/tag/v0.3.1",
               "assets": [{"name": "MeshMill-0.3.1-windows-x64-setup.exe",
                           "url": mm.REPOSITORY_URL+"/releases/download/v0.3.1/setup.exe",
                           "digest": "sha256:"+"a"*64}]}
    monkeypatch.setattr(mm, "APP_VERSION", "0.3.0")
    monkeypatch.setattr(mm, "fetch_latest_release", lambda: release)
    window._check_for_updates()
    assert window.update_future is None
    window._check_for_updates(force=True)
    wait(app, lambda: window.update_future is None)
    assert window.version_label.text() == "Upgrade to 0.3.1"
    assert window.version_label.isEnabled()
    assert not window.update_checks_enabled
    launches, exits = [], []
    monkeypatch.setattr(mm.subprocess, "Popen", lambda command, **_: launches.append(command))
    monkeypatch.setattr(mm.QApplication, "quit", lambda: exits.append(True))
    installer = tmp_path/"update with spaces.exe"
    window._finish_update_download(installer, None)
    assert launches[0][0] == str(installer)
    assert "/REOPEN=1" in launches[0]
    assert f"/REOPENFILE={window.source_path}" in launches[0]
    assert exits == [True]


def test_toolbox_button_rows_fit_all_font_sizes(gui):
    app, window, _ = gui
    for position in range(5):
        window.font_size_position = position
        window._apply_appearance()
        for button in (window.undo_button, window.cancel_preview_button, window.redo_button,
                       window.preview_button, window.toggle_mesh_button, window.previous_mesh_button):
            button.setEnabled(True)
        window._update_action_row_visibility()
        app.processEvents()
        for frame in (window.history_actions, window.comparison_actions):
            frame.layout().activate()
            for button in frame.findChildren(mm.QPushButton):
                assert frame.rect().contains(button.geometry().adjusted(-1, -1, 1, 1))
                assert button.width() >= button.minimumSizeHint().width()
                assert button.height() >= button.minimumSizeHint().height()
        for frame in window.controls_content.findChildren(QFrame, "toolSection"):
            margins = frame.layout().contentsMargins()
            assert min(margins.left(), margins.right(), margins.top(), margins.bottom()) >= 11


def test_load_display_switch_and_geometry_unchanged(gui):
    app, window, _ = gui
    original = window.source_points.copy(), window.source_faces.copy()
    assert window._cached_source_measurements(window.source_points, window.source_faces)
    for mode in ("Wireframe", "Vertices", "Density", "Shaded", "Density"):
        window.display_type.setCurrentIndex(window.display_type.findData(mode))
        wait(app, lambda: settled(window))
        assert window.active_actor is window.source_actor
        assert np.array_equal(original[0], window.source_points)
        assert np.array_equal(original[1], window.source_faces)


def test_unified_caption_controls_preserve_window_states_and_geometry(gui):
    app, window, _ = gui
    original = window.source_points.copy(), window.source_faces.copy()
    buttons = [window.fullscreen_button, window.minimize_button,
               window.maximize_button, window.close_button]
    assert window.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert all(button.parent() is window.app_header for button in buttons)
    assert len({button.geometry().center().y() for button in buttons}) == 1
    assert all(button.width() >= 38 and button.height() >= 28 for button in buttons)
    window.setWindowTitle("MeshMill caption test")
    assert window.caption_label.text() == "MeshMill caption test"
    QTest.mouseClick(window.maximize_button, Qt.MouseButton.LeftButton)
    wait(app, window.isMaximized)
    assert window.maximize_button.toolTip() == "Restore"
    QTest.mouseClick(window.maximize_button, Qt.MouseButton.LeftButton)
    wait(app, lambda: not window.isMaximized())
    QTest.mouseDClick(window.app_header, Qt.MouseButton.LeftButton, pos=QPoint(150, 15))
    wait(app, window.isMaximized)
    QTest.mouseClick(window.fullscreen_button, Qt.MouseButton.LeftButton)
    wait(app, window.isFullScreen)
    QTest.mouseClick(window.fullscreen_button, Qt.MouseButton.LeftButton)
    wait(app, window.isMaximized)
    QTest.mouseClick(window.minimize_button, Qt.MouseButton.LeftButton)
    wait(app, window.isMinimized)
    window.showMaximized()
    wait(app, window.isMaximized)
    assert all(np.array_equal(a, b) for a, b in zip(
        original, (window.source_points, window.source_faces)))
    assert window._window_resize_edges(QPoint(0, 0)) == (Qt.Edge.LeftEdge | Qt.Edge.TopEdge)
    assert not window._window_resize_edges(QPoint(window.width() // 2, window.height() // 2))


def test_f11_enters_and_exits_fullscreen_with_viewport_and_control_focus(gui):
    app, window, _ = gui
    assert mm.SHORTCUT_DEFINITIONS["toggle_fullscreen"][1] == "F11"
    assert window.shortcuts["toggle_fullscreen"] == "F11"
    window.activateWindow()
    wait(app, window.isActiveWindow)
    for widget in (window.vtk_widget, window.algorithm):
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key.Key_F11)
        wait(app, window.isFullScreen)
        assert "F11" in window.fullscreen_button.toolTip()
        QTest.keyClick(widget, Qt.Key.Key_F11)
        wait(app, lambda: window.isMaximized() and not window.isFullScreen())
    window.shortcuts["toggle_fullscreen"] = "Ctrl+F12"
    window._sync_full_screen_button()
    assert window.fullscreen_shortcut.key() == QKeySequence("Ctrl+F12")
    assert "Ctrl+F12" in window.fullscreen_button.toolTip()
    QTest.keyClick(window.vtk_widget, Qt.Key.Key_F12, Qt.KeyboardModifier.ControlModifier)
    wait(app, window.isFullScreen)
    window._toggle_full_screen()


def test_preferences_search_filters_clears_and_cancel_preserves_settings(gui):
    app, window, _ = gui
    settings_before = {key: window.settings.value(key) for key in window.settings.allKeys()}
    failures = []
    def exercise():
        dialog = app.activeModalWidget()
        try:
            search = dialog.findChild(QLineEdit, "preferencesSearch")
            assert search.isClearButtonEnabled()
            sections = dialog.findChildren(QFrame, "settingsSection")
            assert len(sections) > 2
            search.setText("kEyBoArD shortcuts")
            assert any(not section.isHidden() for section in sections)
            assert any(section.isHidden() for section in sections)
            def label(text):
                return next(widget for widget in dialog.findChildren(QLabel) if widget.text() == text)
            search.setText("Font size")
            assert label("Font size").isVisible()
            assert not label("Tools pane").isVisible()
            assert not label("Default units").isVisible()
            search.setText("performance GPU")
            assert label("GPU").isVisible()
            assert not label("Geometry processes").isVisible()
            search.setText("Toggle full screen")
            assert label("Toggle full screen").isVisible()
            assert not label("Orbit left").isVisible()
            label("GPU").setToolTip("synthetic-tooltip-only-match")
            combo = dialog.findChildren(QComboBox)[0]
            combo.addItem("synthetic-value-only-match")
            for query in ("synthetic-tooltip-only-match", "synthetic-value-only-match", "F11"):
                search.setText(query)
                assert all(section.isHidden() for section in sections), query
            search.setText("no-matching-preference-xyz")
            assert all(section.isHidden() for section in sections)
            assert dialog.findChild(QLabel, "preferencesNoMatches").isVisible()
            search.clear()
            assert all(not section.isHidden() for section in sections)
            assert label("Tools pane").isVisible()
            assert label("Geometry processes").isVisible()
            assert not dialog.findChild(QLabel, "preferencesNoMatches").isVisible()
        except Exception as error:
            failures.append(error)
        finally:
            dialog.reject()
    QTimer.singleShot(100, exercise)
    window._show_settings()
    assert not failures, failures
    assert {key: window.settings.value(key) for key in window.settings.allKeys()} == settings_before


@pytest.mark.parametrize("gui", ["Ctrl+Enter", "Ctrl+Return", "Ctrl+F12"], indirect=True)
def test_fullscreen_shortcut_migrates_old_default_preserves_custom_mapping(gui):
    _, window, _ = gui
    saved = str(window.settings.value("shortcuts/toggle_fullscreen"))
    assert saved in {"F11", "Ctrl+F12"}
    assert window.shortcuts["toggle_fullscreen"] == saved
    assert window.fullscreen_shortcut.key() == QKeySequence(saved)
    assert saved in window.fullscreen_button.toolTip()


def test_preview_apply_undo_redo_and_cache_revision(gui, monkeypatch):
    app, window, _ = gui
    original = window.source_points.copy(), window.source_faces.copy()
    with QSignalBlocker(window.target):
        window.target.setValue(1500)
    window.timer.stop()
    window.preview()
    wait(app, lambda: settled(window) and window.preview_poly is not None)
    optimized = tuple(array.copy() for array in mm.polydata_arrays(window.preview_poly))
    assert window.active_actor is window.simplified_actor
    window._toggle_mesh()
    wait(app, lambda: settled(window))
    assert window.active_actor is window.source_actor
    window._toggle_mesh()
    wait(app, lambda: settled(window))
    assert window.active_actor is window.simplified_actor
    window._apply_optimized_pass()
    window.timer.stop()
    wait(app, lambda: settled(window))
    assert window.mesh_modified
    assert window.save_button.isEnabled()
    assert window._cached_source_measurements(window.source_points, window.source_faces) is None
    assert all(np.array_equal(a, b) for a, b in zip(optimized, (window.source_points, window.source_faces)))
    window._toggle_previous_mesh()
    wait(app, lambda: settled(window))
    assert window.showing_previous
    assert window.active_actor is window.previous_view_actor
    window._toggle_previous_mesh()
    wait(app, lambda: settled(window))
    assert not window.showing_previous
    assert window.active_actor is window.source_actor
    window._undo_working_mesh()
    wait(app, lambda: settled(window))
    assert all(np.array_equal(a, b) for a, b in zip(original, (window.source_points, window.source_faces)))
    assert not window.mesh_modified
    assert not window.save_button.isEnabled()
    assert window._cached_source_measurements(window.source_points, window.source_faces)
    window._redo_working_mesh()
    wait(app, lambda: settled(window))
    assert all(np.array_equal(a, b) for a, b in zip(optimized, (window.source_points, window.source_faces)))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes)
    window._reload_original()
    wait(app, lambda: settled(window))
    assert not window.mesh_modified
    assert all(np.array_equal(a, b) for a, b in zip(original, (window.source_points, window.source_faces)))


def test_cancel_optimization_and_ignore_late_result(gui):
    app, window, _ = gui
    original = window.source_points.copy(), window.source_faces.copy()
    with QSignalBlocker(window.target):
        window.target.setValue(1500)
    window.timer.stop()
    window.preview()
    assert window.busy
    generation = window.generation
    window._cancel_mesh_operation()
    wait(app, lambda: settled(window))
    window._finish_simplify(generation, original[0][:4], original[1][:1], {}, None)
    QTest.qWait(100)
    app.processEvents()
    assert not window.busy
    assert window.preview_poly is None
    assert not window.mesh_modified
    assert all(np.array_equal(a, b) for a, b in zip(original, (window.source_points, window.source_faces)))


def test_loading_stays_busy_until_viewport_ready(gui, monkeypatch):
    app, window, _ = gui
    prepare = mm.prepare_viewport_poly
    started, release = threading.Event(), threading.Event()
    def delayed(poly, progress=None):
        started.set()
        assert release.wait(5)
        return prepare(poly, progress)
    monkeypatch.setattr(mm, "prepare_viewport_poly", delayed)
    try:
        window.load_path(window.source_path)
        wait(app, started.is_set)
        assert window.busy
        assert not window.preview_button.isEnabled()
    finally:
        release.set()
    wait(app, lambda: settled(window))
    assert window.active_actor is window.source_actor


def test_optimization_stays_busy_until_viewport_ready(gui, monkeypatch):
    app, window, _ = gui
    prepare = mm.prepare_viewport_poly
    started, release = threading.Event(), threading.Event()
    def delayed(poly, progress=None):
        started.set()
        assert release.wait(5)
        return prepare(poly, progress)
    monkeypatch.setattr(mm, "prepare_viewport_poly", delayed)
    with QSignalBlocker(window.target):
        window.target.setValue(1500)
    window.timer.stop()
    try:
        window.preview()
        wait(app, started.is_set)
        assert window.busy
        assert window.preview_poly is None
    finally:
        release.set()
    wait(app, lambda: settled(window))
    assert window.active_actor is window.simplified_actor


def test_selection_preview_preserves_unselected_geometry_and_cancel(gui):
    app, window, _ = gui
    original = window.source_points.copy(), window.source_faces.copy()
    mask = np.zeros(len(window.source_faces), dtype=bool)
    mask[:len(mask) // 2] = True
    window.selected_face_mask = mask
    window.selection_target = 1000
    window.preview_button.click()
    wait(app, lambda: settled(window) and window.preview_poly is not None)
    points, faces = mm.polydata_arrays(window.preview_poly)
    assert window.preview_is_selection
    untouched = original[0][original[1][~mask]]
    assert np.array_equal(points[faces[:len(untouched)]], untouched)
    assert all(np.array_equal(a, b) for a, b in zip(original, (window.source_points, window.source_faces)))
    window._cancel_optimized_preview()
    wait(app, lambda: settled(window))
    assert window.preview_poly is None
    assert window.selected_face_mask is None
    assert not window.mesh_modified


def draw_selection(window, left, top, right, bottom):
    height = window.vtk_widget.height()
    window._crop_interaction("start", (left, height - top))
    window._crop_interaction("move", (right, height - bottom))
    window._crop_interaction("finish", (right, height - bottom))
    assert window.crop_candidate is not None


def test_selection_shrinks_preserves_added_region_and_empty_outline(gui):
    app, window, _ = gui
    width, height = window.vtk_widget.width(), window.vtk_widget.height()
    original = window.source_points.copy(), window.source_faces.copy()
    draw_selection(window, int(width*.1), int(height*.1), int(width*.9), int(height*.9))
    large = window.crop_candidate[3].copy()
    assert large.any()
    # Modifiers and non-navigation keys do not detach or commit the draft.
    outline = [QPoint(point) for point in window.crop_points]
    for key in ("control_l", "shift_l", "alt_l", "x"):
        assert not window._keyboard_navigate(key, False, False)
    for key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt):
        QTest.keyPress(window.vtk_widget, key)
        QTest.keyRelease(window.vtk_widget, key)
    assert window.crop_points == outline
    assert window.selected_face_mask is None
    # Shift+click on empty space remains a ruler action, not a crop edit or pan.
    interactor = window.navigation_style.GetInteractor()
    interactor.SetEventInformation(4, 4, 0, 1)
    window.navigation_style._left_press(None, None)
    window.navigation_style._left_release(None, None)
    interactor.SetShiftKey(0)
    assert window.crop_points == outline
    assert not window.navigation_style._left_pan_down
    # Replacing the editable outline recalculates the draft rather than unioning its old shape.
    window.crop_points = [QPoint(int(width*x), int(height*y)) for x,y in
                          ((.47,.4),(.55,.4),(.55,.6),(.47,.6))]
    window._prepare_polygon_crop()
    smaller = window.crop_candidate[3].copy()
    assert 0 < smaller.sum() < large.sum()
    assert not np.any(smaller & ~large)
    window._apply_selection_action("add")
    committed = window.selected_face_mask.copy()
    draw_selection(window, 4, 4, 20, 20)
    assert np.array_equal(window.crop_candidate[3], committed)
    window._clear_selection()
    draw_selection(window, 4, 4, 20, 20)
    assert window.crop_candidate is not None and not window.crop_candidate[3].any()
    assert len(window.crop_points) == 4
    assert not window.preview_button.isEnabled()
    window.preview()
    assert not window.busy
    window._apply_selection_action("crop")
    assert all(np.array_equal(a,b) for a,b in zip(original,(window.source_points,window.source_faces)))
    window._clear_selection()


def test_selection_navigation_preserves_faces_and_clears_view_specific_outline(gui):
    app, window, _ = gui
    width, height = window.vtk_widget.width(), window.vtk_widget.height()
    draw_selection(window, int(width*.25), int(height*.25), int(width*.75), int(height*.75))
    mask = window.crop_candidate[3].copy()
    outline = [QPoint(point) for point in window.crop_points]
    center=(width//2,height//2)
    assert window._measurement_interaction("click", center)
    assert len(window.measure_points) == 1
    assert window.crop_points == outline
    assert window.selected_face_mask is None
    # An ordinary click inside no longer applies/adds the selection.
    window._crop_interaction("left_press", center)
    assert window.crop_points == outline
    assert window.selected_face_mask is None
    assert window._keyboard_navigate("right", True, False)
    app.processEvents()
    assert not window.crop_points
    assert not window.crop_overlay_actors
    assert np.array_equal(window.crop_candidate[3], mask)
    assert window.selected_face_mask is None
    interactor = window.navigation_style.GetInteractor()
    interactor.SetEventPosition(width//2, height//2)
    interactor.SetControlKey(0)
    interactor.SetShiftKey(0)
    window.navigation_style._wheel_forward(None, None)
    app.processEvents()
    assert np.array_equal(window.crop_candidate[3], mask)
    assert not window.crop_points
    window._clear_selection()
    assert len(window.measure_points) == 1
    window._clear_measurements()


def test_multiple_selection_drags_crop_and_undo_preserve_exact_coordinates(gui):
    app, window, _ = gui
    width, height = window.vtk_widget.width(), window.vtk_widget.height()
    original = window.source_points.copy(), window.source_faces.copy()
    draw_selection(window, int(width*.35), int(height*.3), int(width*.5), int(height*.7))
    first = window.crop_candidate[3].copy()
    assert first.any()
    actions = {button.property("selectionAction")
               for button in window.crop_info_frame.findChildren(mm.QPushButton)}
    assert "add" not in actions
    assert "optimize" not in actions
    assert {"crop", "delete", "undo", "clear"} <= actions
    draw_selection(window, int(width*.5), int(height*.3), int(width*.65), int(height*.7))
    assert np.array_equal(window.selected_face_mask, first)
    combined = window.crop_candidate[3].copy()
    assert combined.sum() > first.sum()
    assert not np.any(first & ~combined)
    expected = tuple(array.copy() for array in window.crop_candidate[1:3])
    window._apply_selection_action("crop")
    wait(app, lambda: settled(window))
    assert window.mesh_modified
    assert all(np.array_equal(a,b) for a,b in zip(expected,(window.source_points,window.source_faces)))
    window._undo_working_mesh()
    wait(app, lambda: settled(window))
    assert all(np.array_equal(a,b) for a,b in zip(original,(window.source_points,window.source_faces)))


def test_close_joins_active_optimization_worker(gui, tmp_path):
    app, window, _ = gui
    window.optimization_executor.submit(slow_task, str(tmp_path), "worker")
    wait(app, lambda: (tmp_path / "worker").exists())
    processes = list(window.optimization_executor._processes.values())
    window.close()
    assert all(not process.is_alive() for process in processes)


def test_cancel_joins_worker_and_replacement_accepts_work(gui, tmp_path):
    app, window, _ = gui
    window._begin_mesh_operation("Optimizing", "Test mesh")
    window.operation_future = window.optimization_executor.submit(
        slow_task, str(tmp_path), "worker"
    )
    window.operation_uses_process = True
    wait(app, lambda: (tmp_path / "worker").exists())
    processes = list(window.optimization_executor._processes.values())
    window._cancel_mesh_operation()
    assert all(not process.is_alive() for process in processes)
    future = window.optimization_executor.submit(abs, -4)
    wait(app, future.done)
    assert future.result() == 4
    assert not window.busy


def test_application_quit_stops_worker_without_window_close(gui, tmp_path):
    app, window, _ = gui
    window.optimization_executor.submit(slow_task, str(tmp_path), "worker")
    wait(app, lambda: (tmp_path / "worker").exists())
    processes = list(window.optimization_executor._processes.values())
    app.aboutToQuit.connect(window._shutdown_background_workers)
    app.aboutToQuit.emit()
    assert all(not process.is_alive() for process in processes)
    assert window._background_workers_stopped
