from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import meshmill as mm


def test_toolbox_content_stays_inside_scroll_viewport(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = mm.MainWindow(settings=QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat))
    window.resize(1920, 1040)
    window.show()
    app.processEvents()

    viewport_width = window.controls_widget.viewport().width()
    assert window.controls_content.minimumWidth() == 0
    assert window.controls_content.width() <= viewport_width

    # These are the widest repeated rows and previously pushed their final
    # controls behind the right edge of the toolbox.
    window.cancel_preview_button.setEnabled(True)
    window.previous_mesh_button.setEnabled(True)
    window._update_action_row_visibility()
    app.processEvents()
    for widget in (
        window.target_display,
        window.smart_quality,
        window.redo_button,
        window.previous_mesh_button,
        window.repository_button,
    ):
        if widget.isHidden():
            continue
        right_edge = widget.mapTo(window.controls_content, widget.rect().topRight()).x()
        assert right_edge <= viewport_width, (widget.text(), right_edge, viewport_width)

    from vtkmodules.vtkFiltersSources import vtkSphereSource
    source = vtkSphereSource()
    source.Update()
    window.source_poly = source.GetOutput()
    window.source_points, window.source_faces = mm.polydata_arrays(window.source_poly)
    window.mesh_modified = True
    window._sync_modified_mesh_actions()
    assert window.save_button.isEnabled() and window.save_button.isVisible()
    assert window.reload_button.isEnabled() and window.reload_button.isVisible()

    window.mesh_modified = False
    window._sync_modified_mesh_actions()
    assert not window.save_button.isEnabled() and not window.save_button.isVisible()
    assert not window.reload_button.isEnabled() and not window.reload_button.isVisible()

    window.close()


def test_tool_switches_preserve_scroll_range_and_collapse_other_tools(tmp_path):
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QStyle, QStyleOptionSlider
    from vtkmodules.vtkFiltersSources import vtkSphereSource

    app = QApplication.instance() or QApplication([])
    window = mm.MainWindow(settings=QSettings(str(tmp_path / "scroll.ini"), QSettings.Format.IniFormat))
    window.resize(1000, 650)
    window.show()
    sphere = vtkSphereSource()
    sphere.Update()
    window.source_poly = sphere.GetOutput()
    window.source_points, window.source_faces = mm.polydata_arrays(window.source_poly)
    actor = mm.vtkActor()
    mapper = mm.vtkPolyDataMapper()
    mapper.SetInputData(window.source_poly)
    actor.SetMapper(mapper)
    window.active_actor = actor
    window.renderer.AddActor(actor)
    window.alignment_toggle.setEnabled(True)
    window.solidify_toggle.setEnabled(True)
    # The asynchronous solid analysis is independent of toolbox navigation.
    window._refresh_solidify_info = lambda: None
    tools = ((window.optimization_toggle, window.optimization_body),
             (window.solidify_toggle, window.solidify_body),
             (window.imprint_section_toggle, window.imprint_body),
             (window.alignment_toggle, window.alignment_body))
    try:
        for _ in range(2):
            for toggle, body in tools:
                window.controls_widget.verticalScrollBar().setValue(200)
                toggle.setChecked(True)
                for _ in range(10):
                    app.processEvents()
                assert toggle.isChecked() and not body.isHidden()
                assert all(other is toggle or not other.isChecked() for other, _ in tools)
                assert all(other is body or other.isHidden() for _, other in tools)
                bar = window.controls_widget.verticalScrollBar()
                assert bar.isEnabled() and bar.maximum() > 0
                assert bar.value() == 0
                viewport = window.controls_widget.viewport()
                event = QWheelEvent(QPointF(10, 10), QPointF(viewport.mapToGlobal(QPoint(10, 10))),
                                    QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton,
                                    Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
                assert event.type() == QEvent.Type.Wheel
                QApplication.sendEvent(viewport, event)
                assert bar.value() > 0
                # Exercise native scrollbar input, not just setValue().
                bar.setValue(0)
                option = QStyleOptionSlider()
                bar.initStyleOption(option)
                handle = bar.style().subControlRect(QStyle.ComplexControl.CC_ScrollBar, option,
                                                   QStyle.SubControl.SC_ScrollBarSlider, bar)
                QTest.mousePress(bar, Qt.MouseButton.LeftButton, pos=handle.center())
                QTest.mouseMove(bar, handle.center() + QPoint(0, 60), delay=20)
                QTest.mouseRelease(bar, Qt.MouseButton.LeftButton,
                                   pos=handle.center() + QPoint(0, 60))
                assert bar.value() > 0
                bar.setValue(bar.maximum())
                app.processEvents()
                assert bar.value() == bar.maximum()
        # Scrolling remains available while geometry actions are disabled.
        window._set_controls_busy(True)
        bar = window.controls_widget.verticalScrollBar()
        bar.setValue(0)
        QApplication.sendEvent(window.controls_widget.viewport(), event)
        assert bar.value() > 0 and bar.isEnabled()
        window._set_controls_busy(False)
        window.alignment_toggle.setChecked(False)
        app.processEvents()
        assert window.optimization_toggle.isChecked()
        assert not window.optimization_body.isHidden()
    finally:
        window.mesh_modified = False
        window.close()
