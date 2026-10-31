"""Exercise real Qt/VTK state transitions with isolated preferences."""
import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSettings, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
)
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


def test_update_remind_and_skip_choices(gui, monkeypatch):
    _, window, _ = gui
    window.latest_release_version = "99.0.0"
    window.latest_release_url = "https://example.org/release"
    window.latest_release = {}
    monkeypatch.setattr(window, "_confirm_update_install", lambda _: "remind")
    window._install_latest_release()
    assert window.settings.value("updates/skipped_version", "") == ""
    assert window._update_available()
    assert "99.0.0" in window._prompted_update_versions

    # A new application session has no session-only prompt suppression.
    window._prompted_update_versions.clear()
    window.update_checks_enabled = True
    window._pending_update_prompt = "99.0.0"
    monkeypatch.setattr(window, "_confirm_update_install", lambda _: "skip")
    window._offer_startup_update()
    assert window.settings.value("updates/skipped_version") == "99.0.0"
    assert window._update_available()  # Manual installation remains available.


def test_auto_install_requires_checking_and_saved_idle_mesh(gui, monkeypatch):
    _, window, _ = gui
    assert not window.update_checks_enabled
    assert not window.auto_install_updates
    window.update_checks_enabled = True
    window.auto_install_updates = True
    window.auto_install_scope = "all"
    window.latest_release_version = "99.0.0"
    window.latest_release_url = "https://example.org/release"
    window._pending_update_prompt = "99.0.0"
    calls = []
    monkeypatch.setattr(window, "_install_latest_release", lambda **kw: calls.append(kw))
    window.mesh_modified = True
    window._offer_startup_update()
    assert not calls
    window.mesh_modified = False
    window._offer_startup_update()
    assert calls == [{"automatic": True}]


def test_auto_install_rechecks_unsaved_work_after_download(gui, monkeypatch, tmp_path):
    _, window, _ = gui
    window.update_checks_enabled = True
    window.auto_install_updates = True
    window.auto_install_scope = "all"
    window._automatic_update_in_progress = True
    installer = tmp_path / "setup.exe"
    launched = []
    monkeypatch.setattr(mm.subprocess, "Popen", lambda *a, **kw: launched.append(a))
    monkeypatch.setattr(QApplication, "quit", lambda: None)
    window.mesh_modified = True
    window._finish_update_download(installer, None)
    assert not launched
    assert window._ready_auto_installer == installer
    window.mesh_modified = False
    window._offer_startup_update()
    assert len(launched) == 1
    assert window._ready_auto_installer is None


def test_maintenance_update_scope_prompts_for_minor_and_major_releases(gui, monkeypatch):
    _, window, _ = gui
    monkeypatch.setattr(mm, "APP_VERSION", "0.3.2")
    window.update_checks_enabled = True
    window.auto_install_updates = True
    window.auto_install_scope = "maintenance"
    window.latest_release_url = "https://example.org/release"
    calls = []
    monkeypatch.setattr(window, "_install_latest_release", lambda **kw: calls.append(kw))
    for version, automatic in (("0.3.3", True), ("0.4.0", False), ("1.0.0", False)):
        window.latest_release_version = version
        window._pending_update_prompt = version
        window._offer_startup_update()
        assert calls[-1] == {"automatic": automatic}
    window.auto_install_scope = "all"
    assert window._auto_install_release_allowed()


def test_empty_view_buttons_and_open_folder_modes(gui, tmp_path):
    _, window, _ = gui
    window.active_actor = None
    window.source_poly = None
    window._update_active_view_button()
    assert not any(b.isChecked() for b in window.view_buttons.values())
    window.renderer.GetActiveCamera().Azimuth(12)
    assert not any(b.isChecked() for b in window.view_buttons.values())
    for theme in mm.APPEARANCE_THEMES:
        window.appearance_theme_name = theme
        for direction, button in window.view_buttons.items():
            button.click()
            assert not any(b.isChecked() for b in window.view_buttons.values())
            window._apply_theme_icon(button, "view-" + direction)
            assert not button.icon().isNull()
        palette = mm.appearance_theme(theme)
        foreground = mm.selected_control_text(palette)
        assert foreground in ("#FFFFFF", "#000000")
    folder = tmp_path / "scans" / "part"
    folder.mkdir(parents=True)
    window.settings.setValue("files/last_open_folder", str(folder))
    window.settings.setValue("files/open_mode", "last")
    assert window._file_open_directory() == str(folder)
    window.settings.setValue("files/open_mode", "smart")
    assert window._file_open_directory() == str(folder.parent)
    window.settings.setValue("files/open_mode", "fixed")
    window.settings.setValue("files/open_folder", str(tmp_path))
    assert window._file_open_directory() == str(tmp_path)


def test_recent_files_opt_in_deduplication_limit_and_clear(gui, tmp_path, monkeypatch):
    _, window, _ = gui
    assert not window.recent_files_enabled
    assert window.recent_files_button.isHidden()
    window._remember_recent_file(window.source_path)
    assert not window.settings.contains("files/recent")
    window._set_recent_files_enabled(True)
    assert not window.recent_files_button.isHidden()
    assert window.recent_files_button.text() == ""
    assert not window.recent_files_button.icon().isNull()
    for index in range(12):
        window._remember_recent_file(tmp_path / f"mesh-{index}.stl")
    window._remember_recent_file(window.source_path)
    window._remember_recent_file(window.source_path)
    assert len(window.recent_files) == 10
    assert window.recent_files[0] == str(window.source_path.resolve())
    assert len(set(window.recent_files)) == 10
    assert window.settings.value("files/recent", [], type=list) == window.recent_files
    window._rebuild_recent_files_menu()
    actions = window.recent_files_menu.actions()
    assert actions[0].isEnabled()
    assert not actions[1].isEnabled()  # Missing files remain visible but cannot be opened.
    opened = []
    monkeypatch.setattr(window, "load_path", lambda path: opened.append(path))
    actions[0].trigger()
    assert opened == [window.source_path.resolve()]
    actions[-1].trigger()
    assert window.recent_files == []
    assert not window.settings.contains("files/recent")
    window._remember_recent_file(window.source_path)
    window._set_recent_files_enabled(False)
    assert window.recent_files_button.isHidden()
    assert window.recent_files == []
    assert not window.settings.contains("files/recent")


def test_invert_selection_complements_without_changing_geometry_and_can_undo(gui):
    app, window, _ = gui
    points, faces = window.source_points.copy(), window.source_faces.copy()
    mask = np.zeros(len(faces), dtype=bool)
    mask[:19] = True
    window._present_selection(*window._poly_from_face_mask(mask), mask)
    window._apply_selection_action("invert")
    wait(app, lambda: not window.busy)
    assert np.array_equal(window.crop_candidate[3], ~mask)
    assert np.array_equal(window.source_points, points)
    assert np.array_equal(window.source_faces, faces)
    window._undo_selection()
    assert np.array_equal(window.selected_face_mask, mask)
    full = np.ones(len(faces), dtype=bool)
    window._present_selection(*window._poly_from_face_mask(full), full)
    window._apply_selection_action("invert")
    wait(app, lambda: not window.busy)
    assert not window.crop_candidate[3].any()
    assert len(window.crop_candidate[2]) == 0
    window._apply_selection_action("invert")
    wait(app, lambda: not window.busy)
    assert window.crop_candidate[3].all()


def test_quality_and_algorithm_dropdowns_recalculate_slider(gui, monkeypatch):
    _, window, _ = gui
    targets = {"Draft": 1200, "Balanced": 1800, "Fine": 2400}
    monkeypatch.setattr(mm, "smart_target", lambda points, faces, quality, *args: targets[quality])
    for quality in ("Draft", "Fine", "Balanced"):
        window.slider.setValue(100)
        window.smart_quality.setCurrentText(quality)
        assert window.target.value() == targets[quality]
        position = window.slider.value()
        window._sync_slider_from_target()
        assert window.slider.value() == position
    for algorithm in ("Density balanced", "Fast QEM"):
        window.slider.setValue(100)
        window.algorithm.setCurrentText(algorithm)
        assert window.target.value() == targets[window.smart_quality.currentText()]
        assert window.slider.value() != 100
    mask = np.zeros(len(window.source_faces), dtype=bool)
    mask[:4000] = True
    window._present_selection(*window._poly_from_face_mask(mask), mask)
    window.slider.setValue(100)
    window.smart_quality.setCurrentText("Fine")
    assert window.selection_target == targets["Fine"]
    window.slider.setValue(100)
    window.algorithm.setCurrentText("Density balanced")
    assert window.selection_target == targets["Fine"]
    assert window.slider.value() != 100


def test_association_preference_only_opens_windows_settings(gui, monkeypatch):
    _, window, _ = gui
    opened = []
    monkeypatch.setattr(mm.platform, "system", lambda: "Windows")
    monkeypatch.setattr(mm.sys, "frozen", False, raising=False)
    monkeypatch.setattr(mm.QDesktopServices, "openUrl", lambda url: opened.append(url.toString()) or True)
    window._open_stl_association_settings()
    assert opened == ["ms-settings:defaultapps"]


def test_installer_advertises_stl_without_replacing_default():
    installer = (mm.Path(__file__).resolve().parents[1] / "installer" / "MeshMill.iss").read_text()
    registry = installer.split("[Registry]", 1)[1].split("[Run]", 1)[0]
    assert "OpenWithProgids" in registry
    assert "RegisteredApplications" in registry
    assert "UserChoice" not in registry
    assert 'Subkey: "Software\\Classes\\.stl";' not in registry


def test_opening_camera_establishes_fixed_front(gui):
    _, window, _ = gui
    assert window.view_buttons["front"].isChecked()
    camera = window.renderer.GetActiveCamera()
    opening = np.asarray(camera.GetPosition()) - np.asarray(camera.GetFocalPoint())
    opening /= np.linalg.norm(opening)
    assert np.allclose(window.saved_standard_views["front"][0], opening)
    reference = {key: (view.copy(), up.copy())
                 for key, (view, up) in window.saved_standard_views.items()}
    for direction in ("left", "right", "top", "bottom", "back", "front"):
        window._set_standard_view(direction)
        assert window.view_buttons[direction].isChecked()
        view, up = window.saved_standard_views[direction]
        assert np.isclose(np.linalg.norm(view), 1)
        assert np.isclose(np.dot(view, up), 0)
    camera.Azimuth(17)
    for key, (view, up) in reference.items():
        assert np.array_equal(window.saved_standard_views[key][0], view)
        assert np.array_equal(window.saved_standard_views[key][1], up)


def test_standard_views_fit_bounds_without_cropping(gui):
    _, window, _ = gui
    bounds = window.active_actor.GetBounds()
    width, height = window.vtk_widget.GetRenderWindow().GetSize()
    for parallel in (False, True):
        window.renderer.GetActiveCamera().SetParallelProjection(parallel)
        for direction in mm.STANDARD_VIEWS:
            window._set_standard_view(direction)
            projected = []
            for x in bounds[:2]:
                for y in bounds[2:4]:
                    for z in bounds[4:6]:
                        window.renderer.SetWorldPoint(x, y, z, 1)
                        window.renderer.WorldToDisplay()
                        px, py, _ = window.renderer.GetDisplayPoint()
                        projected.append((px / width, py / height))
            projected = np.asarray(projected)
            assert projected.min() >= .0299, (parallel, direction)
            assert projected.max() <= .9701, (parallel, direction)
            assert np.ptp(projected, axis=0).max() >= .939, (parallel, direction)


def test_reload_original_restores_opening_rotation_pan_and_zoom(gui, monkeypatch):
    app, window, _ = gui
    camera = window.renderer.GetActiveCamera()
    original = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp(),
                camera.GetParallelScale(), camera.GetViewAngle())
    window._set_standard_view("left")
    camera.Azimuth(23)
    camera.Zoom(2)
    camera.SetFocalPoint(3, 4, 5)
    window.mesh_modified = True
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    window._reload_original()
    wait(app, lambda: settled(window))
    restored = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp(),
                camera.GetParallelScale(), camera.GetViewAngle())
    for actual, expected in zip(restored, original):
        assert np.allclose(actual, expected)
    assert window.view_buttons["front"].isChecked()
    assert not window.mesh_modified


def test_all_display_modes_preserve_camera_and_initial_framing(gui):
    app, window, _ = gui
    camera = window.renderer.GetActiveCamera()

    def frame():
        return np.array([*camera.GetPosition(), *camera.GetFocalPoint(),
                         *camera.GetViewUp(), camera.GetViewAngle(), camera.GetParallelScale()])

    def mode(value):
        window.display_type.setCurrentIndex(window.display_type.findData(value))
        wait(app, lambda: settled(window))

    opening = frame()
    for value in mm.DISPLAY_MODES:
        mode(value)
        assert np.allclose(frame(), opening), value
    camera.Azimuth(21)
    camera.Zoom(1.7)
    moved = frame()
    window.density_cache.clear()  # Exercise asynchronous, uncached switching again.
    for value in ("Wireframe", "Density", "Vertices", "Shaded"):
        mode(value)
        assert np.allclose(frame(), moved), value
    # Fresh-window camera orientation and lens, identical across load defaults.
    for value in mm.DISPLAY_MODES:
        mode(value)
        camera.SetPosition(0, 0, 1)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        camera.SetViewAngle(30)
        camera.SetParallelScale(1)
        window.density_cache.clear()
        window.load_path(window.source_path)
        wait(app, lambda: settled(window))
        assert np.allclose(frame(), opening), value
        assert window.view_buttons["front"].isChecked()


def test_loading_becomes_interactive_before_gpu_warmup_finishes(gui, monkeypatch):
    app, window, _ = gui
    points, faces = window.source_points.copy(), window.source_faces.copy()
    warmup = Future()
    original_submit = window.optimization_executor.submit

    def submit(function, *args, **kwargs):
        if function is mm.prepare_gpu_backend:
            return warmup
        return original_submit(function, *args, **kwargs)

    monkeypatch.setattr(window.optimization_executor, "submit", submit)
    monkeypatch.setattr(mm, "gpu_backend_supported", lambda *_: True)
    window.load_path(window.source_path)
    try:
        wait(app, lambda: settled(window), timeout=5)
        assert not warmup.done()
        assert window.preview_button.isEnabled()
        assert not window.loading_frame.isVisible()
        assert np.array_equal(window.source_points, points)
        assert np.array_equal(window.source_faces, faces)
        window._set_standard_view("right")
        assert window.view_buttons["right"].isChecked()
        assert window._cached_source_measurements(window.source_points, window.source_faces) is not None
    finally:
        warmup.set_result({"backend": "test"})


def test_idle_probe_cache_is_readonly_and_rejects_changed_geometry(gui, monkeypatch):
    app, window, _ = gui
    monkeypatch.setattr(mm, "is_gpu_guided_algorithm", lambda _: True)
    points, faces = window.source_points, window.source_faces
    before_points, before_faces = points.copy(), faces.copy()
    window._precache_source_analysis()
    wait(app, lambda: window._cached_probe_statistics(points, faces) is not None)
    job = window._probe_statistics_job
    window._precache_source_analysis()
    assert window._probe_statistics_job is job
    assert window._cached_probe_statistics(points.copy(), faces) is None
    assert window._cached_probe_statistics(points, faces.copy()) is None
    assert np.array_equal(points, before_points)
    assert np.array_equal(faces, before_faces)


@pytest.mark.parametrize("label, expected", [
    ("Install now", "install"),
    ("Remind me next time", "remind"),
    ("Skip this version", "skip"),
])
def test_update_prompt_buttons(gui, label, expected):
    app, window, _ = gui
    window.latest_release_version = "99.0.0"

    def choose():
        prompt = app.activeModalWidget()
        assert isinstance(prompt, QMessageBox)
        assert prompt.defaultButton().text() == "Remind me next time"
        next(button for button in prompt.buttons() if button.text() == label).click()

    QTimer.singleShot(30, choose)
    assert window._confirm_update_install(True) == expected


@pytest.fixture
def gui(tmp_path, monkeypatch, request):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    settings.setValue("optimization/last_algorithm", "Fast QEM")
    settings.setValue("optimization/startup_algorithm", "last")
    settings.setValue("display/type", "Shaded")
    if hasattr(request, "param"):
        settings.setValue("shortcuts/toggle_fullscreen", request.param)
    monkeypatch.setattr(mm, "QSettings", lambda *_: settings)
    monkeypatch.setattr(mm, "cuda_devices", list)
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


def test_repeated_arrow_rotation_keeps_object_center_and_zoom(gui, monkeypatch):
    _app, window, _ = gui
    camera = window.renderer.GetActiveCamera()
    center = window._keyboard_object_pivot()
    monkeypatch.setattr(window, "_keyboard_view_pivot",
                        lambda: pytest.fail("Rotation should not pick a screen surface"))
    camera.SetFocalPoint(*(center + np.array([.1, .2, .3])))
    camera.OrthogonalizeViewUp()
    for key, control, shift in (("right", False, False), ("up", False, False),
                                ("left", True, True), ("right", True, True)):
        position = np.asarray(camera.GetPosition())
        focal = np.asarray(camera.GetFocalPoint())
        radius = np.linalg.norm(position-center)
        focal_radius = np.linalg.norm(focal-center)
        distance = camera.GetDistance()
        scale, angle = camera.GetParallelScale(), camera.GetViewAngle()
        for _ in range(72):
            assert window._keyboard_navigate(key, control, shift)
            assert np.linalg.norm(np.asarray(camera.GetPosition())-center) == pytest.approx(radius)
            assert np.linalg.norm(np.asarray(camera.GetFocalPoint())-center) == pytest.approx(focal_radius)
            assert camera.GetDistance() == pytest.approx(distance)
            assert camera.GetParallelScale() == scale
            assert camera.GetViewAngle() == angle
        assert np.allclose(camera.GetPosition(), position, atol=1e-8)
        assert np.allclose(camera.GetFocalPoint(), focal, atol=1e-8)


def test_navigation_smoothing_is_bounded_and_stops(gui, monkeypatch):
    app, window, _ = gui
    steps = []
    monkeypatch.setattr(window, "_keyboard_navigate",
                        lambda *args, **kwargs: steps.append((args, kwargs["fraction"])))
    window._queue_keyboard_navigation("right", False, False)
    wait(app, lambda: not window.navigation_timer.isActive())
    assert len(steps) == 4
    assert sum(fraction for _, fraction in steps) == pytest.approx(.2)
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
                if button.isHidden():
                    continue
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
            recent_files = dialog.findChild(QCheckBox, "recentFilesEnabled")
            assert not recent_files.isChecked()
            auto_install = dialog.findChild(QCheckBox, "automaticUpdateInstall")
            checking = next(box for box in dialog.findChildren(QCheckBox)
                            if box.text() == "Check for updates when MeshMill opens")
            assert not auto_install.isChecked()
            assert not auto_install.isEnabled()
            scope = dialog.findChild(QComboBox, "automaticUpdateScope")
            assert scope.currentData() == "maintenance"
            assert not scope.isEnabled()
            checking.setChecked(True)
            assert auto_install.isEnabled()
            auto_install.setChecked(True)
            assert scope.isEnabled()
            checking.setChecked(False)
            assert not auto_install.isEnabled()
            assert not auto_install.isChecked()
            assert not scope.isEnabled()
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
            viewport = next(combo for combo in dialog.findChildren(QComboBox)
                            if combo.findData("custom") >= 0 and combo.findData("match") >= 0)
            for mode in ("custom", "match", "custom", "MeshMill Blue"):
                index = viewport.findData(mode)
                if index < 0:
                    index = 1
                viewport.setCurrentIndex(index)
                search.setText("Density gradient")
                assert label("Density gradient").isHidden() == (viewport.currentData() != "custom")
                search.clear()
                assert label("Density gradient").isHidden() == (viewport.currentData() != "custom")
            assert label("Tools pane").isVisible()
            assert label("Geometry processes").isVisible()
            assert not dialog.findChild(QLabel, "preferencesNoMatches").isVisible()
        except Exception as error:  # noqa: BLE001 - collect Qt timer assertions for the test thread
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
    _app, window, _ = gui
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


def test_held_rotation_accelerates_and_resets(gui, monkeypatch):
    _, window, _ = gui
    clock = [100.]
    steps = []
    monkeypatch.setattr(mm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(window, "_keyboard_navigate",
                        lambda *args, **kwargs: steps.append(kwargs["fraction"]))
    for command in (("right", False, False), ("up", False, False), ("left", True, True)):
        window._stop_keyboard_navigation()
        window._held_navigation_key = Qt.Key.Key_Right
        first = len(steps)
        window._queue_keyboard_navigation(*command)
        assert steps[-1] * 5 == pytest.approx(1.)
        window.navigation_timer.stop()
        start = clock[0]
        for delay in (.016, .10, .20, .29):
            clock[0] = start + delay
            window._advance_navigation()
            assert len(steps) == first + 1
        for index, delay in enumerate((.31, .46), start=2):
            clock[0] = start + delay
            window._advance_navigation()
            assert len(steps) == first + index
            assert steps[-1] * 5 == pytest.approx(1.)
        clock[0] = start + 1.2
        window._navigation_last_tick = clock[0] - .016
        window._advance_navigation()
        assert steps[-1] == pytest.approx(.016 * 16)
    window._stop_keyboard_navigation()
    assert window._navigation_hold_started is None
    assert window._navigation_precision_steps == 0


def test_arrow_tap_rotates_exactly_one_degree(gui):
    app, window, _ = gui
    camera = window.renderer.GetActiveCamera()
    window.vtk_widget.setFocus()
    app.processEvents()
    for modifiers, key in ((Qt.KeyboardModifier.NoModifier, Qt.Key.Key_Right),
                           (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
                            Qt.Key.Key_Left)):
        before_direction = np.asarray(camera.GetDirectionOfProjection())
        before_up = np.asarray(camera.GetViewUp())
        QTest.keyPress(window.vtk_widget, key, modifiers)
        QTest.qWait(100)
        QTest.keyRelease(window.vtk_widget, key, modifiers)
        QTest.qWait(50)
        if modifiers == Qt.KeyboardModifier.NoModifier:
            after = np.asarray(camera.GetDirectionOfProjection())
            before = before_direction
        else:
            after = np.asarray(camera.GetViewUp())
            before = before_up
        angle = np.degrees(np.arccos(np.clip(before @ after, -1., 1.)))
        assert angle == pytest.approx(1., abs=.01)
        assert not window.navigation_timer.isActive()


def test_keyboard_pan_starts_slow_and_accelerates_in_each_direction(gui, monkeypatch):
    _, window, _ = gui
    clock = [100.]
    steps = []
    monkeypatch.setattr(mm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(window, "_keyboard_navigate",
                        lambda *args, **kwargs: steps.append(kwargs["fraction"]))
    for key in ("left", "right", "up", "down"):
        window._stop_keyboard_navigation()
        window._held_navigation_key = Qt.Key.Key_Right
        window._queue_keyboard_navigation(key, True, False)
        window.navigation_timer.stop()
        assert steps[-1] == pytest.approx(.016 * 16 * .12)
        start = clock[0]
        speeds = []
        for delay in (.016, .2, .4, .6, .8, 1.):
            clock[0] = start + delay
            window._navigation_last_tick = clock[0] - .016
            window._advance_navigation()
            speeds.append(steps[-1])
        assert speeds == sorted(speeds)
        assert speeds[-1] == pytest.approx(.016 * 16)
        window._stop_keyboard_navigation()
        assert window._navigation_hold_started is None


def test_keyboard_zoom_accelerates_smoothly_and_stops(gui, monkeypatch):
    _, window, _ = gui
    clock = [100.]
    steps = []
    monkeypatch.setattr(mm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(window, "_keyboard_navigate",
                        lambda *args, **kwargs: steps.append(kwargs["fraction"]))
    for key in ("up", "down"):
        window._stop_keyboard_navigation()
        window._held_navigation_key = Qt.Key.Key_Up
        window._queue_keyboard_navigation(key, True, True)
        window.navigation_timer.stop()
        assert steps[-1] == pytest.approx(.016 * 16 * .12)
        start = clock[0]
        speeds = []
        for tick in range(1, 66):
            clock[0] = start + tick * .016
            window._navigation_last_tick = clock[0] - .016
            window._advance_navigation()
            speeds.append(steps[-1])
        assert speeds == sorted(speeds)
        assert max(np.diff(speeds)) < .01
        assert speeds[-1] == pytest.approx(.016 * 16)
        window._stop_keyboard_navigation()
        before = len(steps)
        window._advance_navigation()
        assert len(steps) == before


def test_lathe_probe_drag_spin_stop_and_reset_preserve_mesh(gui, monkeypatch):
    _app, window, _ = gui
    lathe = window.lathe
    original = window.source_points.copy()
    original_faces = window.source_faces.copy()
    lathe.button.setChecked(True)
    lathe.pick.setChecked(True)
    assert lathe.controls.isVisible()
    assert not lathe.timer.isActive()
    lathe.set_spin(True)
    assert lathe.timer.isActive()
    assert lathe.pick.isChecked()
    window._position_view_pad()
    for button in (window.view_buttons["right"],window.view_buttons["bottom"]):
        center = button.mapTo(window.view_pad,button.rect().center())
        assert window.view_pad.mask().contains(center)
    monkeypatch.setattr(window, "_pick_measure_point", lambda position: np.array((*position,0.)))
    monkeypatch.setattr(window, "_world_to_display", lambda point: tuple(point[:2]))
    assert lathe.interaction("press", (100,100))
    assert lathe.interaction("move", (110,105))
    assert lathe.interaction("release", (110,105))
    assert lathe.timer.isActive()
    assert len(lathe.actors) == 2
    assert lathe.position == pytest.approx((110,105))
    assert lathe.interaction("press", (110,105))
    assert not lathe.timer.isActive()
    assert lathe.interaction("move", (120,115))
    assert lathe.interaction("release", (120,115))
    assert lathe.position == pytest.approx((120,115))
    assert lathe.interaction("press", (180,180))
    assert lathe.interaction("release", (180,180))
    assert len(lathe.probes) == 2
    assert lathe.timer.isActive()
    assert set(lathe.sample_groups) == {0,1}
    lathe.button.setChecked(False)
    assert not lathe.timer.isActive()
    assert not lathe.actors
    assert lathe.rotation_frame() is not None
    lathe.reset()
    assert lathe.rotation_frame() is None
    assert np.array_equal(window.source_points, original)
    assert np.array_equal(window.source_faces, original_faces)


def test_lathe_hardware_probe_centers_a_real_rotating_cylinder(gui, monkeypatch):
    app, window, _ = gui
    angles = np.linspace(0,2*np.pi,128,endpoint=False)
    points = np.vstack([np.column_stack((np.cos(angles),np.full(128,h),np.sin(angles)))
                        for h in (-2.,2.)])
    faces = np.array([(i,(i+1)%128,i+128) for i in range(128)]
                     + [((i+1)%128,(i+1)%128+128,i+128) for i in range(128)])
    poly = mm.arrays_polydata(points,faces)
    window._finish_load(window.load_generation, window.source_path, poly, points, faces, None, None)
    wait(app, lambda: settled(window))
    camera = window.renderer.GetActiveCamera()
    camera.SetPosition(0,0,10);camera.SetFocalPoint(0,0,0);camera.SetViewUp(0,1,0)
    camera.ParallelProjectionOn();camera.SetParallelScale(3)
    window.renderer.ResetCameraClippingRange();window.vtk_widget.GetRenderWindow().Render()
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.pick.setChecked(True)
    lathe.rig[:3,3] = (.2,0.,0.)
    lathe.world_to_rig = np.linalg.inv(lathe.rig)
    lathe.model_matrix = lathe.world_to_rig.copy()
    lathe._camera_signature = None
    lathe.remember_camera()
    target = window._world_to_display(np.array((0.,0.,1.)))
    assert lathe.interaction("press", target)
    assert lathe.interaction("release", target)
    lathe.timer.stop()
    clock = [100.]
    lathe.last_tick = clock[0]
    monkeypatch.setattr(mm.time, "monotonic", lambda: clock[0])
    for _ in range(220):
        clock[0] += .05
        lathe.tick()
        app.processEvents()
        QTest.qWait(1)
    lathe.set_spin(False)
    assert lathe.last_result is not None
    assert np.linalg.norm(lathe.center[[0,2]]) < .12
    assert "Wobble" in lathe.status.text()
    assert np.array_equal(window.source_points,points)


def test_lathe_initial_spin_matches_arrow_rotation_and_wraps_at_360(gui, monkeypatch):
    _, window, _ = gui
    camera = window.renderer.GetActiveCamera()
    camera.SetPosition(0,-10,0)
    camera.SetFocalPoint(0,0,0)
    camera.SetViewUp(0,0,1)
    window.keyboard_horizontal = False
    start = (camera.GetPosition(),camera.GetFocalPoint(),camera.GetViewUp())
    window._keyboard_navigate("left",False,False,fraction=1.8/5)
    expected = (camera.GetPosition(),camera.GetFocalPoint(),camera.GetViewUp())
    camera.SetPosition(*start[0]);camera.SetFocalPoint(*start[1]);camera.SetViewUp(*start[2])
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.pick.setChecked(True)
    lathe.timer.stop()
    assert lathe.axis == pytest.approx((0,0,1))
    monkeypatch.setattr(mm.time,"monotonic",lambda: 100.)
    lathe.last_tick = 99.95
    lathe.sweep = 0.
    lathe.tick()
    actual = (camera.GetPosition(),camera.GetFocalPoint(),camera.GetViewUp())
    def camera_frame(state):
        position, focal, up = map(np.asarray, state)
        direction = position - focal
        return direction / np.linalg.norm(direction), up / np.linalg.norm(up)
    assert np.asarray(camera_frame(actual)) == pytest.approx(np.asarray(camera_frame(expected)))
    lathe.sweep = 359.5
    lathe.last_tick = 99.95
    lathe.tick()
    assert lathe.sweep == pytest.approx(1.3)
    assert lathe.status.text().startswith("1")
    assert not lathe.probes


def test_lathe_fingers_follow_surface_and_resize_and_doubleclick_delete(gui):
    app, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.pick.setChecked(True)
    lathe.timer.stop()
    width,height = window.vtk_widget.GetRenderWindow().GetSize()
    for position in ((width*.5,height*.5),(width*.5+50,height*.5)):
        assert lathe.interaction("press",position)
        assert lathe.interaction("release",position)
        lathe.timer.stop()
    wait(app,lathe.ensure_contact_index)
    lathe.refresh_probe_positions()
    anchors = [point.copy() for point in lathe.probe_worlds]
    frames = np.asarray(lathe.probe_frames).copy()
    assert len(anchors) == 2
    lathe.sweep += 35
    lathe.apply_camera()
    lathe.refresh_probe_positions()
    rotated = [point.copy() for point in lathe.probe_worlds]
    assert not np.allclose(rotated,anchors)
    assert np.asarray(lathe.probe_frames) == pytest.approx(frames)
    window.resize(1100,750)
    app.processEvents()
    lathe.refresh_probe_positions()
    for point,contact,screen in zip(lathe.probe_worlds,rotated,lathe.probes):
        assert point == pytest.approx(contact)
        assert np.linalg.norm(screen-np.asarray(window._world_to_display(contact)))<2.
    tip = lathe.probes[0].copy()
    render_width,render_height = window.vtk_widget.GetRenderWindow().GetSize()
    x = tip[0]*window.vtk_widget.width()/render_width
    y = window.vtk_widget.height()-1-tip[1]*window.vtk_widget.height()/render_height
    QTest.mouseDClick(window.vtk_widget,Qt.MouseButton.LeftButton,pos=QPoint(round(x),round(y)))
    assert len(lathe.probes) == 1
    assert lathe.probe_worlds[0] == pytest.approx(rotated[1])


def test_lathe_converged_alignment_automatically_sets_cube_y_zero(gui,monkeypatch):
    _,window,_ = gui
    from lathe_spindle import SpindleFit
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.pick.setChecked(True)
    lathe.timer.stop()
    lathe.probes = [np.array((100.,100.)),np.array((150.,150.))]
    lathe.probe_worlds = [np.array((1.,0.,0.)),np.array((1.,1.,0.))]
    lathe.probe_frames = [(-1.,1.,0.,1.),(1.,1.,0.,1.)]
    lathe.position = lathe.probes[0].copy()
    fitted = SpindleFit(np.zeros(4),np.zeros(2),np.zeros(2),np.zeros(2),4,False)
    clock = [100.]
    monkeypatch.setattr(mm.time,"monotonic",lambda: clock[0])
    for _ in range(3):
        clock[0] += .05
        lathe.last_tick = clock[0]-.05
        lathe.future = Future();lathe.future.set_result(fitted)
        lathe.future_generation = lathe.generation
        lathe.tick()
    assert lathe.calibrated_revision == lathe.revision
    assert lathe.reference_basis[:,2] == pytest.approx(lathe.axis)
    camera = window.renderer.GetActiveCamera()
    for _ in range(20):
        window._keyboard_rotate_about_object(lathe.axis,17.)
        assert lathe.display_orientation(camera)[1] == pytest.approx(0.,abs=1e-8)
    window._update_orientation_indicator()
    assert window.orientation_indicator._orientation[1] == pytest.approx(0.,abs=1e-8)


def test_mass_center_uses_face_area_not_bounding_box_or_vertex_density(gui):
    _,window,_ = gui
    points=np.array(((0.,0.,0.),(2.,0.,0.),(0.,2.,0.),(10.,0.,0.),(11.,0.,0.),(10.,1.,0.)))
    faces=np.array(((0,1,2),(3,4,5)))
    poly=mm.arrays_polydata(points,faces)
    mapper=mm.vtkPolyDataMapper();mapper.SetInputData(poly)
    actor=mm.vtkActor();actor.SetMapper(mapper)
    window.active_actor=actor
    expected=(2*np.mean(points[:3],axis=0)+.5*np.mean(points[3:],axis=0))/2.5
    assert window._keyboard_object_pivot()==pytest.approx(expected)
    assert not np.allclose(expected,(points.min(axis=0)+points.max(axis=0))/2)


def test_unaccepted_lathe_pose_does_not_override_manual_arrow_axis(gui):
    _,window,_ = gui
    camera=window.renderer.GetActiveCamera()
    lathe=window.lathe
    lathe.button.setChecked(True);lathe.set_spin(False)
    lathe.parameters=np.array((.3,-.2,.15,-.2))
    lathe.target_parameters=lathe.parameters.copy()
    lathe.rotation_frame()
    assert lathe.navigation_frame() is None
    camera.SetViewUp(0,1,0);camera.OrthogonalizeViewUp()
    before=camera.NewInstance();before.DeepCopy(camera)
    pivot=window._keyboard_object_pivot()
    assert not np.allclose(pivot,lathe.center)
    window._keyboard_navigate("right",False,False,fraction=.2)
    actual=(camera.GetDirectionOfProjection(),camera.GetViewUp())
    camera.DeepCopy(before)
    window._keyboard_navigate("right",False,False,fraction=.2)
    expected=(camera.GetDirectionOfProjection(),camera.GetViewUp())
    assert np.asarray(actual)==pytest.approx(np.asarray(expected))


def test_accepted_lathe_axis_is_stable_and_manual_reorientation_updates_it(gui):
    _,window,_ = gui
    lathe=window.lathe
    lathe.button.setChecked(True);lathe.set_spin(False)
    lathe.parameters=np.array((.04,-.03,.03,-.02))
    lathe.target_parameters=lathe.parameters.copy()
    lathe.apply_camera()
    lathe.calibrate_orientation()
    assert lathe.navigation_frame() is not None
    camera=window.renderer.GetActiveCamera()
    before=(camera.GetPosition(),camera.GetFocalPoint(),camera.GetViewUp())
    for _ in range(12):
        window._keyboard_navigate("right",False,False,fraction=6.)
        assert lathe.navigation_frame() is not None
    assert np.asarray((camera.GetPosition(),camera.GetFocalPoint(),camera.GetViewUp()))==pytest.approx(np.asarray(before))
    old_axis = lathe.navigation_frame()[1].copy()
    window._keyboard_navigate("up",False,False,fraction=.2)
    assert lathe.navigation_frame() is not None
    assert not np.allclose(lathe.navigation_frame()[1], old_axis)
    assert lathe.navigation_frame()[1] == pytest.approx(camera.GetViewUp(), abs=1e-6)


def test_alignment_respects_user_orientation_and_rebuilds_all_six_views(gui):
    _,window,_=gui
    camera=window.renderer.GetActiveCamera()
    camera.SetPosition(4.,-8.,3.);camera.SetFocalPoint(0.,0.,0.);camera.SetViewUp(.2,.8,.5)
    camera.OrthogonalizeViewUp()
    user_up=np.asarray(camera.GetViewUp()).copy()
    old_views={key:(view.copy(),up.copy()) for key,(view,up) in window.saved_standard_views.items()}
    lathe=window.lathe
    lathe.button.setChecked(True);lathe.set_spin(False)
    assert lathe.axis==pytest.approx(user_up)
    for key,(view,up) in old_views.items():
        assert window.saved_standard_views[key][0]==pytest.approx(view)
        assert window.saved_standard_views[key][1]==pytest.approx(up)
    lathe.parameters=np.array((.03,-.04,.04,-.025))
    lathe.target_parameters=lathe.parameters.copy();lathe.apply_camera();lathe.calibrate_orientation()
    center=lathe.navigation_origin()
    axis=lathe.axis.copy()
    for name in ("front","right","top","bottom","left","back"):
        window._set_standard_view(name)
        assert camera.GetFocalPoint()==pytest.approx(center)
        direction=np.asarray(camera.GetPosition())-np.asarray(camera.GetFocalPoint())
        direction/=np.linalg.norm(direction)
        if name=="top":assert direction==pytest.approx(axis)
        elif name=="bottom":assert direction==pytest.approx(-axis)
        else:assert camera.GetViewUp()==pytest.approx(axis)


def test_probe_placement_holds_pose_and_failed_drop_preserves_other_probes(gui,monkeypatch):
    _app,window,_=gui
    lathe=window.lathe
    lathe.button.setChecked(True)
    lathe.pick.setChecked(True)
    lathe.set_spin(True)
    lathe.set_spin(True)
    window.vtk_widget.GetRenderWindow().Render()
    width,height=window.vtk_widget.GetRenderWindow().GetSize()
    target=(width*.5,height*.5)
    assert lathe.interaction("press",target)
    assert not lathe.timer.isActive()
    camera=window.renderer.GetActiveCamera()
    before=np.array((camera.GetPosition(),camera.GetFocalPoint(),camera.GetViewUp()))
    QTest.qWait(120)
    assert np.array((camera.GetPosition(),camera.GetFocalPoint(),camera.GetViewUp()))==pytest.approx(before)
    assert lathe.interaction("release",target)
    assert lathe.timer.isActive()
    frame=lathe.probe_frames[0]
    lathe.refresh_probe_positions()
    tip=lathe.probes[0].copy()
    assert lathe.interaction("press",tip)
    monkeypatch.setattr(window,"_pick_measure_point",lambda position:None)
    assert lathe.interaction("release",(0,0))
    assert lathe.probe_frames[0]==frame
    assert lathe.timer.isActive()
    assert lathe.delete_probe(lathe.probes[0])
    assert lathe.timer.isActive()


def test_alignment_probe_mount_does_not_jump_during_spin_or_pose_refinement(gui):
    _,window,_=gui
    lathe=window.lathe
    lathe.button.setChecked(True);lathe.set_spin(False)
    lathe.manual_zoom = True
    datum=np.array((.25,-.3,.1))
    lathe.parameters=np.array((.02,-.03,.025,-.015))
    expected=None
    for phase in (0.,30.,90.,180.,270.,359.):
        lathe.sweep=phase
        lathe.parameters=np.array((.02,-.03,.025,-.015))
        lathe.apply_camera()
        world=lathe.spindle_to_original(datum)
        screen=np.asarray(window._world_to_display(world))
        if expected is None:expected=screen
        assert screen==pytest.approx(expected,abs=1e-6)



def test_viewport_axis_projection_has_original_offsets_and_unlabelled_cross(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    for row, (check, axes) in enumerate(lathe.axis_overlays):
        check.setChecked(True)
        assert len(axes) == 3
        for i, (points, actor, label) in enumerate(axes):
            assert bool(actor.GetVisibility()) == (row == 0 or i < 2)
            assert bool(label.GetVisibility()) == (row == 0)
            if row == 0:
                assert label.GetInput().startswith('XYZ'[i] + ':')
            if not actor.GetVisibility():
                continue
            projected = np.asarray(window._world_to_display(lathe.rig[:3, 3]))
            first, second = (np.asarray(points.GetPoint(j)[:2] if row == 0 else
                              window._world_to_display(points.GetPoint(j))) for j in (0, 1))
            direction = second-first
            if np.linalg.norm(direction) > 1e-4:
                distance = abs(direction[0]*(projected-first)[1]-direction[1]*(projected-first)[0])/np.linalg.norm(direction)
                assert distance < .01
    lathe.button.setChecked(False)
    assert all(not actor.GetVisibility() for _, axes in lathe.axis_overlays for _, actor, _ in axes)


def test_alignment_controls_reflect_explicit_modes(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    assert not lathe.timer.isActive()
    assert not lathe.spin.isChecked()
    assert not lathe.pick.isVisible()
    lathe.spin.click()
    assert lathe.timer.isActive()
    assert lathe.spin.isChecked()
    lathe.spin.click()
    assert not lathe.timer.isActive()
    assert not lathe.spin.isChecked()


def test_finalized_axis_becomes_export_frame_without_spinning_mesh(gui):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    from lathe_spindle import pose_matrix
    from mesh_export import calibrated_export_mesh
    _, window, _ = gui
    lathe = window.lathe
    original = window.source_points.copy()
    lathe.button.setChecked(True)
    lathe.set_spin(False)
    lathe.parameters = np.array((.02, -.03, .01, -.02))
    lathe.target_parameters = lathe.parameters.copy()
    expected = pose_matrix(lathe.parameters) @ np.linalg.inv(lathe.rig)
    lathe.sweep = 137.
    lathe.finish.setEnabled(True)
    lathe.finish_calibration()
    assert window.export_calibration_transform == pytest.approx(expected)
    assert window.mesh_modified and window.save_button.isEnabled()
    assert window.generate_script_button.isEnabled()
    assert np.array_equal(window.source_points, original)
    center = np.linalg.inv(expected)[:3, 3]
    assert (expected @ np.r_[center, 1.]) == pytest.approx((0, 0, 0, 1))
    result = calibrated_export_mesh(window.source_poly, expected)
    points = vtk_to_numpy(result.GetPoints().GetData())
    assert points == pytest.approx(original @ expected[:3, :3].T + expected[:3, 3])
    assert not lathe.button.isChecked()
    assert window.optimization_toggle.isChecked()


def test_axis_overlay_changes_rendered_pixels_over_model(gui):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.set_spin(False)
    render_window = window.vtk_widget.GetRenderWindow()
    def pixels():
        render_window.Render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(render_window)
        capture.ReadFrontBufferOff()
        capture.Update()
        width, height, _ = capture.GetOutput().GetDimensions()
        return vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(height, width, -1).copy()
    for check, _ in lathe.axis_overlays:
        check.setChecked(False)
    before = pixels()
    lathe.axis_overlays[0][0].setChecked(True)
    after = pixels()
    changed = np.any(before != after, axis=2)
    assert np.count_nonzero(changed) > 100
    center = np.array(window._world_to_display(lathe.rig[:3, 3])).astype(int)
    x, y = center
    assert np.count_nonzero(changed[max(0,y-35):y+36, max(0,x-35):x+36]) > 20





def test_rotate_captures_viewport_z_after_section_opened(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.axis_choice.setCurrentIndex(lathe.axis_choice.findData("depth"))
    lathe.button.setChecked(True)
    camera = window.renderer.GetActiveCamera()
    camera.SetPosition(4, -8, 3)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(.7, .2, .6)
    camera.OrthogonalizeViewUp()
    expected = -np.array(camera.GetDirectionOfProjection())
    lathe.spin.click()
    assert lathe.timer.isActive()
    assert lathe.spin.isChecked()
    assert lathe.rig[:3, 2] == pytest.approx(expected)
    lathe.set_spin(False)
    direction = np.array(camera.GetDirectionOfProjection())
    lathe.sweep = 30.
    lathe.apply_camera()
    assert camera.GetDirectionOfProjection() == pytest.approx(direction)
    assert lathe.axis == pytest.approx(expected)



def test_alignment_viewport_axis_choices_and_persistence(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    camera = window.renderer.GetActiveCamera()
    camera.SetPosition(4, -8, 3)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(.7, .2, .6)
    camera.OrthogonalizeViewUp()
    depth = -np.array(camera.GetDirectionOfProjection())
    up = np.array(camera.GetViewUp())
    horizontal = np.cross(up, depth)
    for mode, expected in (("vertical", up), ("horizontal", horizontal), ("depth", depth)):
        lathe.axis_choice.setCurrentIndex(lathe.axis_choice.findData(mode))
        assert not lathe.timer.isActive()
        lathe.spin.click()
        assert lathe.rig[:3, 2] == pytest.approx(expected)
        assert np.linalg.det(lathe.rig[:3, :3]) == pytest.approx(1.)
        lathe.spin.click()
        assert window.settings.value("alignment/viewport_axis", "vertical") == mode



def test_open_alignment_fits_model_without_reorienting(gui):
    _, window, _ = gui
    camera = window.renderer.GetActiveCamera()
    camera.SetPosition(4, -8, 3)
    camera.SetFocalPoint(.2, .1, -.1)
    camera.SetViewUp(.7, .2, .6)
    camera.OrthogonalizeViewUp()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(100.)
    direction, up = np.array(camera.GetDirectionOfProjection()), np.array(camera.GetViewUp())
    lathe = window.lathe
    lathe.button.setChecked(True)
    assert camera.GetDirectionOfProjection() == pytest.approx(direction)
    assert camera.GetViewUp() == pytest.approx(up)
    assert camera.GetParallelScale() < 100.
    assert not lathe.timer.isActive()
    lathe.spin.click()
    expected_axis, _ = lathe.viewport_frame()
    assert lathe.rig[:3, 2] == pytest.approx(expected_axis)
    assert camera.GetDirectionOfProjection() == pytest.approx(direction)
    assert camera.GetViewUp() == pytest.approx(up)
    bounds = window.active_actor.GetBounds()
    width, height = window.vtk_widget.GetRenderWindow().GetSize()
    for x in bounds[:2]:
        for y in bounds[2:4]:
            for z in bounds[4:6]:
                px, py = window._world_to_display(np.array((x, y, z)))
                assert 0 <= px <= width
                assert 0 <= py <= height



def test_alignment_default_turntable_matches_left_right_rotation(gui):
    _, window, _ = gui
    lathe = window.lathe
    assert lathe.axis_choice.currentData() == "vertical"
    lathe.button.setChecked(True)
    camera = window.renderer.GetActiveCamera()
    before = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())
    window._keyboard_navigate("left", False, False, fraction=1.8/5)
    expected = np.array((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()))
    camera.SetPosition(*before[0])
    camera.SetFocalPoint(*before[1])
    camera.SetViewUp(*before[2])
    lathe.spin.click()
    lathe.timer.stop()
    lathe.sweep = 1.8
    lathe.apply_camera()
    assert np.array((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())) == pytest.approx(expected)



def test_original_axes_are_model_reference_not_rotation_frame(gui):
    _, window, _ = gui
    camera = window.renderer.GetActiveCamera()
    camera.SetPosition(4, -8, 3)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(.7, .2, .6)
    camera.OrthogonalizeViewUp()
    lathe = window.lathe
    lathe.button.setChecked(True)
    rig = lathe.rig.copy()
    check, axes = lathe.axis_overlays[0]
    check.setChecked(True)
    center = window._keyboard_object_pivot()
    matrix = window.active_actor.GetMatrix()
    transform = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
    for i, (points, _, _) in enumerate(axes):
        from lathe_control import viewport_axis_ends
        first = window._world_to_display(center - transform[:3, i] * lathe.extent * .3)
        second = window._world_to_display(center + transform[:3, i] * lathe.extent * .3)
        expected = viewport_axis_ends(first, second, *window.renderer.GetSize())
        assert points.GetPoint(1)[:2] == pytest.approx(second if expected is None else expected[1], abs=1e-3)
    assert lathe.rig == pytest.approx(rig)
    lathe.spin.click()
    assert lathe.rig[:3, 2] == pytest.approx(camera.GetViewUp())





def test_original_axis_reference_stays_fixed_during_correction(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.axis_overlays[0][0].setChecked(True)
    original = lathe.original_reference_frame.copy()
    lathe.parameters = np.array((.03, -.04, .04, -.025))
    lathe.target_parameters = lathe.parameters.copy()
    lathe.apply_camera()
    lathe.calibrate_orientation()
    lathe.update_axes()
    assert lathe.original_reference_frame == pytest.approx(original)
    for i, (points, _, _) in enumerate(lathe.axis_overlays[0][1]):
        from lathe_control import viewport_axis_ends
        first = window._world_to_display(original[:3, 3] - original[:3, i] * lathe.extent * .3)
        second = window._world_to_display(original[:3, 3] + original[:3, i] * lathe.extent * .3)
        expected = viewport_axis_ends(first, second, *window.renderer.GetSize())
        assert points.GetPoint(1)[:2] == pytest.approx(second if expected is None else expected[1], abs=1e-3)




def test_alignment_centers_projected_bounds_after_layout(gui):
    app, window, _ = gui
    camera = window.renderer.GetActiveCamera()
    camera.SetFocalPoint(20, 30, 40)
    camera.SetPosition(24, 22, 43)
    camera.SetViewUp(.7, .2, .6)
    camera.OrthogonalizeViewUp()
    direction = np.array(camera.GetDirectionOfProjection())
    up = np.array(camera.GetViewUp())
    window.lathe.button.setChecked(True)
    app.processEvents()
    bounds = np.array(window.active_actor.GetBounds()).reshape(3, 2)
    assert camera.GetFocalPoint() == pytest.approx(bounds.mean(axis=1))
    assert camera.GetDirectionOfProjection() == pytest.approx(direction)
    assert camera.GetViewUp() == pytest.approx(up)
    assert not window.lathe.timer.isActive()



def test_alignment_axis_overlays_default_on_and_change_readout(gui):
    _, window, _ = gui
    lathe = window.lathe
    assert all(check.isChecked() for check, _ in lathe.axis_overlays)
    lathe.button.setChecked(True)
    lathe.parameters = np.array((.03, -.04, .04, -.025))
    lathe.update_axes()
    assert "Center shift" in lathe.fit_metrics.text()
    assert "Tilt" in lathe.fit_metrics.text()
    assert all(bool(actor.GetVisibility()) == (row == 0 or i < 2) for row, (_, axes) in enumerate(lathe.axis_overlays) for i, (_, actor, _) in enumerate(axes))



def test_probe_correction_refits_center_zoom_and_preserves_rotation(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.spin.click()
    camera = window.renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(100.)
    lathe._camera_signature = None
    lathe.remember_camera()
    lathe.set_spin(True)
    lathe.parameters = np.array((.03, -.04, .04, -.025))
    lathe.target_parameters = lathe.parameters.copy()
    lathe.apply_camera()
    bounds = np.array(window.active_actor.GetBounds()).reshape(3, 2)
    assert camera.GetFocalPoint() == pytest.approx(bounds.mean(axis=1))
    assert camera.GetParallelScale() < 100.
    assert lathe.timer.isActive()
    before = np.array((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()))
    scale = camera.GetParallelScale()
    lathe.apply_camera()
    assert np.array((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())) == pytest.approx(before)
    assert camera.GetParallelScale() == pytest.approx(scale)



def test_accepted_probe_fit_updates_coordinate_frame_before_zero_wobble(gui, monkeypatch):
    from types import SimpleNamespace
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    original = lathe.original_reference_frame.copy()
    lathe.probes = [np.zeros(2)]
    lathe.parameters = np.array((.02, -.03, .01, -.02))
    lathe.target_parameters = lathe.parameters.copy()
    lathe.last_result = SimpleNamespace(rank=4, accepted=True, after=np.array([.1]), total_runout=np.array([.2]))
    lathe.solve_complete = True
    monkeypatch.setattr(lathe, "ensure_contact_index", lambda: True)
    monkeypatch.setattr(lathe, "draw_probe", lambda **kwargs: None)
    monkeypatch.setattr(lathe, "finger_contact", lambda index: np.array((1.,0.,0.)))
    lathe.tick()
    assert lathe.stable_fits == 0
    assert lathe.reference_basis is not None
    assert window.orientation_reference_basis == pytest.approx(lathe.reference_basis)
    assert lathe.accepted_frame is not None
    assert lathe.original_reference_frame == pytest.approx(original)



def test_probe_calibration_has_signed_per_model_axis_values_independent_of_spin(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.parameters = np.array((.03, -.04, .04, -.025))
    angles, shift = lathe.probe_calibration_values()
    assert np.linalg.norm(angles) == pytest.approx(np.degrees(np.linalg.norm(lathe.parameters[2:])))
    assert np.linalg.norm(shift) == pytest.approx(.05)
    lathe.sweep = 137.
    lathe.update_axes()
    after_angles, after_shift = lathe.probe_calibration_values()
    assert after_angles == pytest.approx(angles)
    assert after_shift == pytest.approx(shift)
    text = lathe.calibration_values.text()
    for axis in "XYZ":
        assert text.count(axis + " ") == 1
        assert axis in lathe.calibration_progress.rotation_text
    assert "\N{DEGREE SIGN}" in lathe.calibration_progress.rotation_text
    assert "Center (" in text



def test_manual_refinement_keeps_rotation_and_updates_fitted_axis(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.spin.click()
    before_axis = lathe.rotation_frame()[1].copy()
    points = window.source_points.copy()
    window._keyboard_rotate_about_object(np.array((1., 0., 0.)), 1.)
    window._update_orientation_indicator()
    lathe.apply_manual_camera()
    assert lathe.timer.isActive()
    assert lathe.spin.isChecked()
    assert not np.allclose(lathe.rotation_frame()[1], before_axis)
    camera = window.renderer.GetActiveCamera()
    before = np.array((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()))
    lathe.apply_camera()
    assert np.array((camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp())) == pytest.approx(before)
    angles, _ = lathe.probe_calibration_values()
    assert np.linalg.norm(angles) == pytest.approx(1.)
    assert np.array_equal(window.source_points, points)
    assert "AXIAL CALIBRATION" in window._toolbox_sections



def test_calibration_stability_requires_full_probe_coverage_and_finish_locks(gui, monkeypatch):
    from types import SimpleNamespace
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.probes = [np.zeros(2), np.zeros(2)]
    lathe.last_result = SimpleNamespace(rank=4)
    lathe.stable_fits = 2
    lathe.solve_complete = True
    lathe.invalidate_coverage()
    lathe.solve_complete = True
    monkeypatch.setattr(lathe, "finger_contact", lambda index: np.zeros(3))
    for phase in range(0, 355, 5):
        lathe.sweep = phase
        lathe.update_calibration_state(True)
    assert not lathe.finish.isEnabled()
    lathe.sweep = 355
    lathe.update_calibration_state(True)
    assert lathe.finish.isEnabled()
    assert lathe.calibration_progress.format() == "Stable"
    lathe.finish.click()
    assert lathe.calibration_locked
    assert not lathe.timer.isActive()
    assert lathe.accepted_frame is not None
    assert not lathe.probes
    assert not lathe.actors
    assert not lathe.button.isChecked()
    assert window.optimization_toggle.isChecked()
    assert all(not actor.GetVisibility() for _, axes in lathe.axis_overlays for _, actor, _ in axes)
    assert not lathe.interaction("press", (0, 0))
    lathe.reset()
    assert not lathe.calibration_locked
    assert lathe.spin.isEnabled()
    assert lathe.calibration_progress.value() == 0


def test_missing_probe_contact_prevents_calibration_finish(gui, monkeypatch):
    from types import SimpleNamespace
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.probes = [np.zeros(2), np.zeros(2)]
    lathe.last_result = SimpleNamespace(rank=4)
    lathe.stable_fits = 2
    monkeypatch.setattr(lathe, "finger_contact", lambda index: None if index else np.zeros(3))
    for phase in range(0, 360, 5):
        lathe.sweep = phase
        lathe.update_calibration_state(True)
    assert not lathe.finish.isEnabled()
    assert lathe.calibration_progress.value() == 50



def test_full_turn_no_remaining_improvement_clears_probes_and_keeps_rotating(gui):
    from types import SimpleNamespace
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.spin.click()
    lathe.probes = [np.zeros(2), np.zeros(2)]
    lathe.coverage = np.ones((2, 72), dtype=bool)
    lathe.last_result = SimpleNamespace(rank=4)
    lathe.stable_fits = 0
    lathe.solve_complete = True
    lathe.future = None
    lathe.update_calibration_state(True)
    assert lathe.finish.isEnabled()
    assert lathe.timer.isActive()
    assert lathe.spin.isChecked()
    assert lathe.calibration_progress.format() == "Stable"

    assert lathe.detection_complete
    assert not lathe.probes
    assert not lathe.actors


def test_full_turn_waits_for_pending_fit_before_stopping(gui):
    from types import SimpleNamespace
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.spin.click()
    lathe.probes = [np.zeros(2)]
    lathe.coverage = np.ones((1, 72), dtype=bool)
    lathe.last_result = SimpleNamespace(rank=4)
    lathe.solve_complete = True
    lathe.future = Future()
    lathe.update_calibration_state(True)
    assert not lathe.finish.isEnabled()
    assert lathe.timer.isActive()
    lathe.future = None



def test_ready_calibration_rotates_without_restarting_detection_until_finalized(gui):
    from types import SimpleNamespace
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.spin.click()
    lathe.probes = [np.zeros(2)]
    lathe.coverage = np.ones((1, 72), dtype=bool)
    lathe.last_result = SimpleNamespace(rank=4)
    lathe.solve_complete = True
    lathe.update_calibration_state(True)
    assert lathe.detection_complete
    original_rig = lathe.rig.copy()
    lathe.set_spin(False)
    lathe.set_spin(True)
    assert lathe.rig == pytest.approx(original_rig)
    phase = lathe.sweep
    lathe.last_tick = time.monotonic() - .02
    lathe.tick()
    assert lathe.sweep != phase
    assert lathe.future is None
    assert lathe.finish.isEnabled()
    assert all(bool(actor.GetVisibility()) == (row == 0 or i < 2) for row, (_, axes) in enumerate(lathe.axis_overlays) for i, (_, actor, _) in enumerate(axes))
    lathe.finish.click()
    assert lathe.calibration_locked
    assert not lathe.timer.isActive()
    assert window.optimization_toggle.isChecked()
    assert not lathe.actors


def test_clear_probes_preserves_alignment_rotation_and_cancels_detection(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.spin.click()
    lathe.parameters = np.array((.03, -.04, .04, -.025))
    lathe.target_parameters = lathe.parameters.copy()
    lathe.probes = [np.zeros(2)]
    lathe.probe_frames = [(0.,1.,0.,1.)]
    lathe.future = Future()
    future = lathe.future
    expected = lathe.parameters.copy()
    lathe.clear_probes_button.click()
    assert not lathe.probes
    assert not lathe.probe_frames
    assert not lathe.actors
    assert future.cancelled()
    assert lathe.future is None
    assert lathe.timer.isActive()
    assert lathe.parameters == pytest.approx(expected)
    assert lathe.target_parameters == pytest.approx(expected)
    assert not lathe.finish.isEnabled()


def test_completed_calibration_rotates_without_surface_analysis(gui, monkeypatch):
    from types import SimpleNamespace
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.spin.click()
    lathe.probes = [np.zeros(2)]
    lathe.coverage = np.ones((1,72), dtype=bool)
    lathe.last_result = SimpleNamespace(rank=4)
    lathe.solve_complete = True
    lathe.update_calibration_state(True)
    assert lathe.detection_complete
    assert lathe.timer.interval() == 8
    assert window.navigation_style._calibrated_inspection
    def forbidden(*args, **kwargs):
        pytest.fail("Analysis called during inspection rotation")
    monkeypatch.setattr(lathe, "finger_contact", forbidden)
    monkeypatch.setattr(lathe, "ensure_contact_index", forbidden)
    monkeypatch.setattr(window.executor, "submit", forbidden)
    for _ in range(5):
        lathe.last_tick = time.monotonic() - .01
        lathe.tick()
    assert lathe.timer.isActive()


def test_calibrated_close_zoom_uses_fine_steps_without_geometry_change(gui):
    _, window, _ = gui
    style = window.navigation_style
    camera = window.renderer.GetActiveCamera()
    camera.ParallelProjectionOff()
    points = window.source_points.copy()
    width, height = window.vtk_widget.GetRenderWindow().GetSize()
    style.GetInteractor().SetEventPosition(width//2,height//2)
    bounds = window.active_actor.GetBounds()
    top = bounds[5]
    moves = []
    for inspection in (False, True):
        camera.SetPosition(0,0,top+1e-4)
        camera.SetFocalPoint(0,0,0)
        camera.SetViewUp(0,1,0)
        window.renderer.ResetCameraClippingRange()
        start = np.array(camera.GetPosition())
        style._calibrated_inspection = inspection
        style._zoom_at_cursor(1.000001)
        moves.append(np.linalg.norm(np.array(camera.GetPosition())-start))
    assert 0 < moves[1] < moves[0]/100
    assert np.array_equal(points,window.source_points)



def test_manual_rotation_refreshes_calibrated_frame_without_spin_tick(gui):
    from lathe_spindle import pose_matrix
    app, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    app.processEvents()
    lathe.calibrate_orientation()
    before = window.orientation_reference_basis.copy()
    camera = window.renderer.GetActiveCamera()
    window._keyboard_rotate_about_object(np.array((1.,0.,0.)),1.)
    window._update_orientation_indicator()
    app.processEvents()
    assert not lathe.timer.isActive()
    assert not np.allclose(window.orientation_reference_basis, before)
    expected = lathe.rig @ np.linalg.inv(pose_matrix(lathe.parameters))
    assert window.orientation_reference_basis == pytest.approx(expected[:3,:3])
    for i, (points, _, _) in enumerate(lathe.axis_overlays[1][1][:2]):
        up = np.asarray(camera.GetViewUp())
        right = np.cross(up, -np.asarray(camera.GetDirectionOfProjection()))
        direction = (right, up)[i]
        actual = np.asarray(points.GetPoint(1)) - expected[:3, 3]
        assert np.linalg.norm(np.cross(actual, direction)) < 1e-5
        assert float(actual @ direction) > 0
    assert camera.GetViewUp() is not None



def test_calibration_buttons_only_show_applicable_actions(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    assert not lathe.spin.isHidden()
    assert lathe.reverse_button.isHidden()
    assert lathe.clear_probes_button.isHidden()
    assert lathe.finish.isHidden()
    assert lathe.reset_button.isHidden()
    lathe.spin.click()
    assert not lathe.reverse_button.isHidden()
    lathe.spin.click()
    assert lathe.reverse_button.isHidden()
    lathe.probes = [np.zeros(2)]
    lathe.sync_controls()
    assert not lathe.clear_probes_button.isHidden()
    assert not lathe.reset_button.isHidden()
    lathe.clear_probes_button.click()
    assert lathe.clear_probes_button.isHidden()
    assert lathe.finish.isHidden()
    lathe.finish.setEnabled(True)
    lathe.detection_complete = True
    lathe.sync_controls()
    assert not lathe.finish.isHidden()
    lathe.finish.click()
    assert lathe.finish.isHidden()
    assert lathe.spin.isHidden()



def test_calibrated_overlay_is_actual_fitted_rotation_frame(gui):
    from lathe_spindle import pose_matrix
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    initial = lathe.original_reference_frame.copy()
    lathe.parameters = np.array((.08, -.06, .12, -.09))
    lathe.target_parameters = lathe.parameters.copy()
    lathe.apply_camera()
    calibrated = lathe.rig @ np.linalg.inv(pose_matrix(lathe.parameters))
    differences = []
    for i, (points, _, label) in enumerate(lathe.axis_overlays[1][1][:2]):
        camera = window.renderer.GetActiveCamera()
        up = np.asarray(camera.GetViewUp())
        right = np.cross(up, -np.asarray(camera.GetDirectionOfProjection()))
        expected = calibrated[:3,3]+(right, up)[i]*lathe.extent*.3
        actual = np.asarray(points.GetPoint(1)) - calibrated[:3, 3]
        direction = (right, up)[i]
        assert np.linalg.norm(np.cross(actual, direction)) < 1e-5
        assert float(actual @ direction) > 0
        old = window._world_to_display(initial[:3,3]+initial[:3,i]*lathe.extent*.3)
        differences.append(np.linalg.norm(np.array(window._world_to_display(expected))-old))
        assert label.GetInput() == "XYZ"[i]
        assert "°" in lathe.axis_overlays[0][1][i][2].GetInput()
    assert max(differences) > 1.
    assert lathe.original_reference_frame == pytest.approx(initial)



def test_calibrated_z_line_tracks_the_rotation_center_and_axis(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.parameters = np.array((.08,-.06,.12,-.09))
    lathe.target_parameters = lathe.parameters.copy()
    lathe.sweep = 47.
    lathe.apply_camera()
    center, axis = lathe.rotation_frame()
    expected_start = window._world_to_display(center-axis*lathe.extent*.3)
    expected_end = window._world_to_display(center+axis*lathe.extent*.3)
    assert not lathe.axis_overlays[1][1][2][1].GetVisibility()
    direction = np.array(expected_end)-expected_start
    assert abs(direction[0]) < abs(direction[1]) * 1e-5



def test_stable_calibration_can_add_constraints_without_resetting_pose(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.spin.click()
    lathe.detection_complete = True
    lathe.has_calibration = True
    lathe.retained_probe_frames = [(-.1,1.,0.,.3),(.1,1.,0.,.3)]
    before = lathe.rig.copy()
    width,height = window.vtk_widget.GetRenderWindow().GetSize()
    position = (width*.5,height*.5)
    assert lathe.interaction("press",position)
    assert len(lathe.probes) == 3
    assert not lathe.detection_complete
    assert lathe.interaction("release",position)
    assert lathe.timer.isActive()
    assert lathe.rig == pytest.approx(before)
    assert not lathe.finish.isEnabled()


def test_overlay_is_observed_until_actual_correction(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    assert lathe.axis_overlays[1][0].text() == "Alignment axes"
    assert [label.GetInput() for _,_,label in lathe.axis_overlays[1][1][:2]] == ["X", "Y"]
    window._keyboard_rotate_about_object(np.array((1.,0.,0.)),1.)
    window._update_orientation_indicator()
    lathe.apply_manual_camera()
    assert lathe.axis_overlays[1][0].text() == "Alignment axes"



def test_space_toggles_calibration_rotation_without_interfering_with_text(gui):
    app, window, _ = gui
    window.activateWindow()
    app.processEvents()
    lathe = window.lathe
    lathe.button.setChecked(True)
    window.vtk_widget.setFocus()
    QTest.keyClick(window.vtk_widget, Qt.Key.Key_Space)
    assert lathe.timer.isActive()
    QTest.keyClick(window.vtk_widget, Qt.Key.Key_Space)
    assert not lathe.timer.isActive()
    editor = QLineEdit(window)
    editor.show()
    editor.setFocus()
    QTest.keyClick(editor, Qt.Key.Key_Space)
    assert editor.text() == " "
    assert not lathe.timer.isActive()
    lathe.button.setChecked(False)
    window.vtk_widget.setFocus()
    QTest.keyClick(window.vtk_widget, Qt.Key.Key_Space)
    assert not lathe.timer.isActive()



def test_alignment_depth_controls_translate_axis_without_zoom_or_mesh_changes(gui):
    app, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    app.processEvents()
    camera = window.renderer.GetActiveCamera()
    camera_before = (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp(), camera.GetParallelScale())
    initial = lathe.rig.copy()
    points, faces = mm.polydata_arrays(window.source_poly)
    points, faces = points.copy(), faces.copy()
    depth = np.asarray(camera.GetDirectionOfProjection())
    window._keyboard_navigate('up', True, True, fraction=.25)
    assert lathe.rig[:3, 3] == pytest.approx(initial[:3, 3] + depth * lathe.extent * .005 * .25)
    assert lathe.rig[:3, :3] == pytest.approx(initial[:3, :3])
    assert (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp(), camera.GetParallelScale()) == camera_before
    window._keyboard_navigate('down', True, True, fraction=.25)
    assert lathe.rig == pytest.approx(initial)
    actual_points, actual_faces = mm.polydata_arrays(window.source_poly)
    assert np.array_equal(actual_points, points) and np.array_equal(actual_faces, faces)
    lathe.button.setChecked(False)
    distance = camera.GetDistance()
    window._keyboard_navigate('up', True, True, fraction=.25)
    assert camera.GetDistance() < distance


def test_calibrated_axis_is_world_geometry_with_natural_depth(gui):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersSources import vtkCubeSource
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkWindowToImageFilter,
    )

    _, window, _ = gui
    points, axis, _ = window.lathe.axis_overlays[1][1][0]
    assert isinstance(axis, vtkActor)
    assert isinstance(axis.GetMapper(), vtkPolyDataMapper)
    points.SetPoint(0, -2., 0., 0.)
    points.SetPoint(1, 2., 0., 0.)
    points.Modified()
    axis.SetVisibility(True)
    axis.GetProperty().SetColor(1., 0., 0.)
    cube = vtkCubeSource()
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(cube.GetOutputPort())
    solid = vtkActor()
    solid.SetMapper(mapper)
    solid.GetProperty().SetColor(0., 1., 0.)
    solid.GetProperty().LightingOff()
    renderer = vtkRenderer()
    renderer.AddActor(solid)
    render = vtkRenderWindow()
    render.SetOffScreenRendering(True)
    render.SetSize(160, 160)
    render.AddRenderer(renderer)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0., 0., 5.)
    camera.SetFocalPoint(0., 0., 0.)
    camera.SetViewUp(0., 1., 0.)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(2.5)
    renderer.ResetCameraClippingRange()
    def image():
        render.Render()
        capture = vtkWindowToImageFilter()
        capture.SetInput(render)
        capture.ReadFrontBufferOff()
        capture.Update()
        return vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(160, 160, -1).copy()
    try:
        without_axis = image()
        renderer.AddActor(axis)
        with_axis = image()
        assert np.array_equal(with_axis[76:84, 76:84], without_axis[76:84, 76:84])
        assert np.count_nonzero(with_axis != without_axis) > 0
    finally:
        renderer.RemoveActor(axis)
        render.Finalize()


def test_alignment_keyboard_movement_renders_once_per_frame(gui):
    app, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    app.processEvents()
    render = window.vtk_widget.GetRenderWindow()
    frames = []
    observer = render.AddObserver('StartEvent', lambda *_: frames.append(1))
    try:
        for command in (('up', True, True), ('down', True, True),
                        ('left', False, False), ('up', False, False),
                        ('left', True, True), ('right', True, False)):
            frames.clear()
            window._keyboard_navigate(*command, fraction=.04)
            assert len(frames) == 1, (command, len(frames))
            assert not lathe._pending_manual_camera
    finally:
        render.RemoveObserver(observer)


@pytest.mark.parametrize('mode', ['vertical', 'horizontal', 'depth'])
def test_alignment_cross_is_viewport_xy_in_every_mode(gui, mode):
    app, window, _ = gui
    lathe = window.lathe
    lathe.axis_choice.setCurrentIndex(lathe.axis_choice.findData(mode))
    lathe.button.setChecked(True)
    app.processEvents()
    lathe.update_axes()
    axes = lathe.axis_overlays[1][1]
    for i, letter in enumerate('XY'):
        points, actor, label = axes[i]
        assert actor.GetVisibility() and label.GetInput() == letter
        start = np.asarray(window._world_to_display(points.GetPoint(0)))
        end = np.asarray(window._world_to_display(points.GetPoint(1)))
        delta = end-start
        assert abs(delta[1-i]) < 1e-3
        assert delta[i] > 0
        original_label = lathe.axis_overlays[0][1][i][2].GetInput()
        assert original_label.startswith(letter + ':') and original_label.endswith('\N{DEGREE SIGN}')
    assert not axes[2][1].GetVisibility()


def test_all_original_offsets_visible_with_only_viewport_xy_references(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    lathe.update_axes()
    for i, (_, actor, label) in enumerate(lathe.axis_overlays[0][1]):
        assert actor.GetVisibility() and label.GetVisibility()
        assert label.GetInput().startswith('XYZ'[i] + ':')
        assert label.GetInput().endswith('\N{DEGREE SIGN}')
    assert [bool(actor.GetVisibility()) for _, actor, _ in lathe.axis_overlays[1][1]] == [True, True, False]


def test_offset_labels_remain_inside_viewport_when_zoomed(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    camera = window.renderer.GetActiveCamera()
    camera.Zoom(12.)
    lathe.update_axes()
    width, height = window.renderer.GetSize()
    for _, _, label in lathe.axis_overlays[0][1]:
        size = [0., 0.]
        label.GetSize(window.renderer, size)
        x, y = label.GetPosition()
        assert x >= 8 and y >= 8
        assert x + size[0] <= width - 8 + 1e-6
        assert y + size[1] <= height - 8 + 1e-6


def test_vertical_rotation_and_pan_directions_swapped_without_affecting_depth(gui, monkeypatch):
    _, window, _ = gui
    window.keyboard_vertical = False
    camera = window.renderer.GetActiveCamera()
    rotations = []
    monkeypatch.setattr(window, '_keyboard_rotate_about_object', lambda axis, angle: rotations.append(angle))
    window._keyboard_navigate('up', False, False)
    window._keyboard_navigate('down', False, False)
    assert rotations == [-5., 5.]
    for key, sign in (('up', -1.), ('down', 1.)):
        start = np.asarray(camera.GetPosition())
        up = np.asarray(camera.GetViewUp())
        window._keyboard_navigate(key, True, False)
        assert float((np.asarray(camera.GetPosition()) - start) @ up) * sign > 0
    zooms = []
    monkeypatch.setattr(window, '_keyboard_zoom_at_view_center', zooms.append)
    window._keyboard_navigate('up', True, True)
    window._keyboard_navigate('down', True, True)
    assert zooms == pytest.approx([1.28, 1/1.28])


def test_double_click_visible_probe_removes_it_through_viewport(gui):
    app, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    window.vtk_widget.GetRenderWindow().Render()
    width, height = window.vtk_widget.GetRenderWindow().GetSize()
    target = (width*.5, height*.5)
    assert lathe.interaction('press', target)
    assert lathe.interaction('release', target)
    lathe.refresh_probe_positions()
    assert len(lathe.probes) == 1
    tip = lathe.probes[0].copy()
    position = QPoint(round(tip[0]*window.vtk_widget.width()/width),
                      round(window.vtk_widget.height()-1-tip[1]*window.vtk_widget.height()/height))
    QTest.mouseDClick(window.vtk_widget, Qt.MouseButton.LeftButton, pos=position)
    app.processEvents()
    assert not lathe.probes


def test_probe_click_preserves_contact_and_double_click_follows_motion(gui, monkeypatch):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    window.vtk_widget.GetRenderWindow().Render()
    width, height = window.vtk_widget.GetRenderWindow().GetSize()
    target = (width*.5, height*.5)
    lathe.interaction('press', target)
    lathe.interaction('release', target)
    lathe.refresh_probe_positions()
    tip = lathe.probes[0].copy()
    world, frame = lathe.probe_worlds[0].copy(), lathe.probe_frames[0]
    assert lathe.interaction('press', tip)
    monkeypatch.setattr(window, '_pick_measure_point', lambda _: world + 1.)
    assert lathe.interaction('release', tip)
    assert lathe.probe_worlds[0] == pytest.approx(world)
    assert lathe.probe_frames[0] == frame
    # Rotation can move the visible finger between first and second clicks.
    lathe.probes[0] = tip + np.array((40., 0.))
    assert lathe.delete_probe(tip)
    assert not lathe.probes and not lathe.probe_frames


def test_full_viewport_axes_have_matching_labels_at_both_ends(gui):
    _, window, _ = gui
    lathe = window.lathe
    camera = window.renderer.GetActiveCamera()
    camera.Azimuth(15.)
    camera.Elevation(10.)
    lathe.button.setChecked(True)
    lathe.update_axes()
    width, height = window.renderer.GetSize()
    for row, (_, axes) in enumerate(lathe.axis_overlays):
        for i, (points, actor, label) in enumerate(axes):
            opposite = lathe.axis_end_labels[(row, i)]
            assert bool(opposite.GetVisibility()) == (bool(actor.GetVisibility()) and row == 0)
            if not actor.GetVisibility():
                continue
            assert opposite.GetInput() == label.GetInput()
            for point_index in (0, 1):
                point = points.GetPoint(point_index)
                xy = np.asarray(point[:2] if row == 0 else window._world_to_display(point))
                assert min(abs(xy[0]), abs(xy[0] - (width-1)), abs(xy[1]), abs(xy[1] - (height-1))) < .02
            for text in (label, opposite):
                size = [0., 0.]
                text.GetSize(window.renderer, size)
                x, y = text.GetPosition()
                assert 8 <= x <= width-size[0]-8+1e-5
                assert 8 <= y <= height-size[1]-8+1e-5


def test_alignment_rotation_fits_until_manual_zoom(gui, monkeypatch):
    app, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    app.processEvents()
    fits = []
    original_fit = window._fit_view_bounds
    def fit(*args, **kwargs):
        fits.append(1)
        original_fit(*args, **kwargs)
    monkeypatch.setattr(window, '_fit_view_bounds', fit)
    for phase in (10., 60., 120.):
        lathe.sweep = phase
        lathe.apply_camera()
    assert len(fits) == 3
    window._keyboard_zoom_at_view_center(1.2)
    lathe.remember_camera()
    lathe.apply_manual_camera(render=False)
    assert lathe.manual_zoom
    fits.clear()
    lathe.sweep = 180.
    lathe.apply_camera()
    assert not fits
    lathe.button.setChecked(False)
    lathe.button.setChecked(True)
    app.processEvents()
    assert not lathe.manual_zoom


def test_viewport_reference_lines_are_subtle_and_unlabelled(gui):
    _, window, _ = gui
    lathe = window.lathe
    lathe.button.setChecked(True)
    for i, (_, actor, label) in enumerate(lathe.axis_overlays[1][1]):
        assert bool(actor.GetVisibility()) == (i < 2)
        assert actor.GetProperty().GetLineWidth() <= 1.5
        assert actor.GetProperty().GetOpacity() <= .4
        assert not label.GetVisibility()
        assert not lathe.axis_end_labels[(1, i)].GetVisibility()
    assert all(label.GetVisibility() for _, _, label in lathe.axis_overlays[0][1])


def test_horizontal_rotation_and_pan_directions_swapped_without_changing_roll(gui, monkeypatch):
    _, window, _ = gui
    window.keyboard_horizontal = False
    camera = window.renderer.GetActiveCamera()
    rotations = []
    monkeypatch.setattr(window, '_keyboard_rotate_about_object', lambda axis, angle: rotations.append(angle))
    window._keyboard_navigate('left', False, False)
    window._keyboard_navigate('right', False, False)
    assert rotations == [5., -5.]
    for key, sign in (('left', 1.), ('right', -1.)):
        start = np.asarray(camera.GetPosition())
        forward = np.asarray(camera.GetDirectionOfProjection())
        up = np.asarray(camera.GetViewUp())
        right = np.cross(forward, up)
        window._keyboard_navigate(key, True, False)
        assert float((np.asarray(camera.GetPosition())-start) @ right) * sign > 0
    rotations.clear()
    window._keyboard_navigate('left', True, True)
    window._keyboard_navigate('right', True, True)
    assert rotations == [5., -5.]


def test_quality_presets_are_spread_across_slider_without_changing_targets(gui):
    _, window, _ = gui
    window.source_faces = np.tile(window.source_faces, (200, 1))
    window.source_measurement_cache = None
    window.target.setRange(1000, len(window.source_faces))
    positions = []
    for quality, position in (('Draft', 300), ('Balanced', 600), ('Fine', 900)):
        window.smart_quality.setCurrentText(quality)
        window._apply_smart_target()
        expected = window.target.value()
        assert window.slider.value() == position
        window._slider_changed(position)
        assert window.target.value() == expected
        positions.append(window.slider.value())
    assert positions == [300, 600, 900]
    values = []
    for position in range(0, 1001, 25):
        window._slider_changed(position)
        values.append(window.target.value())
    assert values == sorted(values)


def test_preferences_direction_diagrams_follow_each_theme(gui, monkeypatch):
    app, window, _ = gui
    from mesh_export_ui import readable_theme_color

    for name in mm.APPEARANCE_THEMES:
        window.appearance_theme_name = name
        window._apply_appearance()
        theme = mm.appearance_theme(name)
        accent = readable_theme_color(
            theme["accent"], [theme["window"], theme["panel"]], theme["text"]
        )

        def inspect_dialog(dialog, theme=theme, accent=accent):
            diagrams = [label for label in dialog.findChildren(QLabel)
                        if label.text().startswith(("Drag ", "Wheel ", "Press "))]
            assert len(diagrams) == 6
            for label in diagrams:
                check = label.parentWidget().findChild(QCheckBox)
                check.setChecked(False)
                assert theme["text"] in label.styleSheet()
                check.setChecked(True)
                assert accent in label.styleSheet()
                check.setChecked(False)
                assert theme["text"] in label.styleSheet()
            return mm.QDialog.DialogCode.Rejected

        monkeypatch.setattr(mm.QDialog, "exec", inspect_dialog)
        window._show_settings()
        app.processEvents()


def test_settings_save_preserves_loaded_mesh_and_dynamic_labels_in_all_themes(gui, monkeypatch):
    app, window, _ = gui
    path = window.source_path
    points, faces = window.source_points.copy(), window.source_faces.copy()
    window.status.setText("Current operation status")
    window.open_button.setToolTip("Current open action")
    for name in mm.APPEARANCE_THEMES:
        filename = window.file_label.text()
        quality = {key: label.text() for key, label in window.quality_metric_labels.items()}
        camera = window.renderer.GetActiveCamera()
        pose = camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()

        def save_dialog(dialog, name=name):
            theme_combo = next(combo for combo in dialog.findChildren(QComboBox)
                               if combo.findText("Dark Focus") >= 0)
            theme_combo.setCurrentIndex(theme_combo.findText(name))
            return mm.QDialog.DialogCode.Accepted

        monkeypatch.setattr(mm.QDialog, "exec", save_dialog)
        window._show_settings()
        app.processEvents()
        assert window.appearance_theme_name == name
        assert window.source_path == path
        assert window.file_label.text() == filename
        assert window.file_label.toolTip() == str(path)
        assert window.status.text() == "Current operation status"
        assert window.open_button.toolTip() == "Current open action"
        assert {key: label.text() for key, label in window.quality_metric_labels.items()} == quality
        assert np.array_equal(window.source_points, points)
        assert np.array_equal(window.source_faces, faces)
        assert (camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()) == pose


def test_label_translation_preserves_runtime_changes_and_retranslates_static_text(monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = mm.QWidget()
    label = QLabel("Static label", root)
    button = mm.QPushButton("Open", root)
    root.setWindowTitle("Initial title")
    label.setToolTip("Initial tooltip")
    monkeypatch.setattr(mm, "tr", lambda text: {"Static label": "Translated label"}.get(text, text))
    mm.localize_widget_tree(root)
    assert label.text() == "Translated label"
    monkeypatch.setattr(mm, "tr", lambda text: text)
    mm.localize_widget_tree(root)
    assert label.text() == "Static label"
    label.setText("321,090")
    label.setToolTip("Current tooltip")
    root.setWindowTitle("Current title")
    button.setText("Save current mesh")
    mm.localize_widget_tree(root)
    mm.localize_widget_tree(root)
    assert label.text() == "321,090"
    assert label.toolTip() == "Current tooltip"
    assert root.windowTitle() == "Current title"
    assert button.text() == "Save current mesh"
    root.close()
    app.processEvents()


def test_owned_dialog_captions_follow_theme_including_nested_and_message_boxes(gui):
    import ctypes
    import os

    from PySide6.QtGui import QColor

    if os.name != "nt":
        pytest.skip("Windows native caption attributes")
    app, window, _ = gui
    settings = mm.QDialog(window)
    nested = mm.QDialog(settings)
    message = QMessageBox(window)
    export = mm.QDialog(window)
    export.setObjectName("meshExportDialog")
    compact = mm.QDialog(window)
    compact.setObjectName("compactMeshExportDialog")
    dialogs = [settings, nested, message, export, compact]
    for dialog in dialogs:
        if dialog.layout() is None:
            mm.QVBoxLayout(dialog).addWidget(QLabel("Dialog content", dialog))
    dwm = ctypes.WinDLL("dwmapi")
    for name in mm.APPEARANCE_THEMES:
        window.appearance_theme_name = name
        window._apply_appearance()
        theme = mm.appearance_theme(name)
        for dialog in dialogs:
            dialog.show()
            app.processEvents()
            assert dialog.windowFlags() & Qt.WindowType.FramelessWindowHint
            header = dialog.findChild(mm.WindowTitleBar, "dialogTitleBar")
            assert header is not None
            assert theme["window"] in header.styleSheet()
            assert theme["text"] in header.styleSheet()
            assert header.findChild(mm.QPushButton, "dialogCaptionClose").icon().isNull() is False
            dark = ctypes.c_int()
            assert dwm.DwmGetWindowAttribute(
                ctypes.c_void_p(int(dialog.winId())), 20,
                ctypes.byref(dark), ctypes.sizeof(dark)
            ) == 0
            assert bool(dark.value) == (name == "Dark Focus")
            for attribute, key in ((35, "window"), (36, "text"), (34, "border")):
                value = ctypes.c_uint()
                result = dwm.DwmGetWindowAttribute(
                    ctypes.c_void_p(int(dialog.winId())), attribute,
                    ctypes.byref(value), ctypes.sizeof(value)
                )
                if result == 0:
                    color = QColor(theme[key])
                    assert value.value == color.red() | (color.green() << 8) | (color.blue() << 16)
            dialog.hide()
    for dialog in reversed(dialogs):
        dialog.close()
        dialog.deleteLater()
    app.processEvents()


def test_caption_activity_has_distinct_theme_shade_and_updates_on_focus(gui):
    from PySide6.QtGui import QColor

    app, window, _ = gui
    dialog = mm.QDialog(window)
    mm.QVBoxLayout(dialog).addWidget(QLabel("Dialog content", dialog))
    dialog.show()
    app.processEvents()
    for name in mm.APPEARANCE_THEMES:
        window.appearance_theme_name = name
        window._apply_appearance()
        theme = mm.appearance_theme(name)
        inactive = mm.inactive_caption_color(theme)
        assert QColor(inactive) != QColor(theme["window"])
        for target, header in ((window, window.app_header),
                               (dialog, dialog.findChild(mm.WindowTitleBar, "dialogTitleBar"))):
            for event_type, expected in ((QEvent.Type.WindowDeactivate, False),
                                         (QEvent.Type.WindowActivate, True)):
                app.sendEvent(target, QEvent(event_type))
                assert header.property("windowActive") == expected
            styles = header.styleSheet() if target is dialog else app.styleSheet()
            assert inactive in styles
            assert theme["muted"] in styles
    dialog.close()
    dialog.deleteLater()
    app.processEvents()
