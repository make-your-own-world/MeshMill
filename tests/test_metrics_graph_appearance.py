import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

import meshmill as mm


def contrast(a, b):
    def luminance(color):
        channels = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
                    for value in color.getRgbF()[:3]]
        return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), channels))
    low, high = sorted((luminance(a), luminance(b)))
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("theme", list(mm.APPEARANCE_THEMES))
def test_metrics_graph_panel_match_contrast_and_theme_switch(theme):
    app = QApplication.instance() or QApplication([])
    graph = mm.GeometryActivityGraph()
    graph.samples = [(20.0, 70.0, False), (60.0, 30.0, True)]
    graph.resize(480, 112)
    original = graph.samples.copy()
    for name in ("Dark Focus", "Earth Tone", theme):
        graph.apply_appearance(name)
        colors = graph.plot_colors
        background = colors["background"]
        assert background == QColor(mm.appearance_theme(name)["panel"])
        for key in ("text", "legend", "cpu", "gpu"):
            assert contrast(colors[key], background) >= 4.5, (name, key)
        assert contrast(colors["grid"], background) >= 1.5
        # Exercise actual painting, including the header and footer background.
        image = graph.grab().toImage()
        assert image.pixelColor(2, 2) == background
        assert image.pixelColor(2, graph.height() - 2) == background
        assert graph.samples == original
    graph.close()
    app.processEvents()
