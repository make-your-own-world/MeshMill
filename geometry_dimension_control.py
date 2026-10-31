"""Geometry-scaled millimetre controls for Cast."""
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QWidget


class GeometryDimensionControl(QWidget):
    valueChanged = Signal(float)

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self._lower, self._upper, self._automatic = 0., 10., .5
        self._manual = .5
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.mode = QComboBox()
        self.mode.addItems(["Auto", "Manual"])
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.label = QLabel()
        self.label.setMinimumWidth(64)
        layout.addWidget(self.mode)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.label)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.slider.valueChanged.connect(self._dragged)
        self.set_geometry(100.)

    def value(self):
        return self._automatic if self.mode.currentIndex() == 0 else self._manual

    def reset_auto(self):
        with QSignalBlocker(self.mode):
            self.mode.setCurrentIndex(0)
        self._refresh()
        self.valueChanged.emit(self.value())

    def setValue(self, value):
        self._manual = min(self._upper, max(self._lower, float(value)))
        with QSignalBlocker(self.mode):
            self.mode.setCurrentIndex(1)
        self._refresh()
        self.valueChanged.emit(self.value())

    def set_geometry(self, scale):
        scale = max(float(scale), .001)
        self._lower = scale * .001 if self.kind == "wall" else 0.
        self._upper = max(.5, scale * .1) if self.kind == "offset" else scale * (.25 if self.kind == "wall" else .1)
        if self.kind == "solid-gap":
            self._upper = scale * .5
        self._automatic = .5 if self.kind == "offset" else scale * (.025 if self.kind == "wall" else .015)
        self._manual = min(self._upper, max(self._lower, self._manual))
        self._refresh()

    def _refresh(self):
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(1000 * (self.value() - self._lower) / (self._upper - self._lower)))
        self.label.setText(f"{self.value():.3f} mm")
        self.slider.setToolTip(f"{self._lower:.3f}?{self._upper:.3f} mm. Drag to set a manual value.")

    def _mode_changed(self):
        self._refresh()
        self.valueChanged.emit(self.value())

    def _dragged(self, position):
        self._manual = self._lower + position / 1000 * (self._upper - self._lower)
        with QSignalBlocker(self.mode):
            self.mode.setCurrentIndex(1)
        self._refresh()
        self.valueChanged.emit(self.value())
