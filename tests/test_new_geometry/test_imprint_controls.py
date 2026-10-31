import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtWidgets import QApplication

from clamshell_mold import mold_sampling_plan
from geometry_dimension_control import GeometryDimensionControl


def test_auto_values_slider_manual_and_reset():
    app = QApplication.instance() or QApplication([])
    control = GeometryDimensionControl("wall")
    control.set_geometry(200)
    assert control.value() == 5
    assert "5.000" in control.label.text()
    control.slider.setValue(500)
    assert control.mode.currentText() == "Manual"
    assert .2 < control.value() < 50
    control.reset_auto()
    assert control.value() == 5
    assert app is not None


def test_manual_zero_gap_and_clearance_default():
    app = QApplication.instance() or QApplication([])
    gap = GeometryDimensionControl("gap")
    gap.setValue(0)
    plan = mold_sampling_plan(np.array([[0,100],[0,100],[0,100]]), bridge_mm=gap.value())
    assert plan["bridge_mm"] == 0
    assert GeometryDimensionControl("offset").value() == .5
    assert app is not None
