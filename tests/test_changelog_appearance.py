import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

import meshmill as mm
from test_metrics_graph_appearance import contrast


@pytest.mark.parametrize("theme", list(mm.APPEARANCE_THEMES))
@pytest.mark.parametrize("current_only", [True, False])
def test_changelog_scrolling_and_document_contrast(theme, current_only):
    app = QApplication.instance() or QApplication([])
    text = mm.resource_path("CHANGELOG.md").read_text(encoding="utf-8")
    if current_only:
        start = text.find(f"## {mm.APP_VERSION}")
        end = text.find("\n## ", start + 1)
        text = text[start:] if end < 0 else text[start:end]
    text += "\n\n[Reference](https://example.com/reference) and **bold** with `inline code`.\n"
    viewer = mm.create_changelog_viewer(text, theme)
    viewer.resize(650, 300)
    viewer.show()
    app.processEvents()
    background = QColor(mm.appearance_theme(theme)["field"])
    assert viewer.palette().color(QPalette.ColorRole.Base) == background
    assert viewer.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    assert viewer.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert viewer.verticalScrollBar().isVisible()
    assert viewer.verticalScrollBar().maximum() > 0
    viewer.verticalScrollBar().setValue(viewer.verticalScrollBar().maximum())
    assert viewer.verticalScrollBar().value() > 0
    assert viewer.openExternalLinks()
    links = 0
    block = viewer.document().begin()
    while block.isValid():
        iterator = block.begin()
        while not iterator.atEnd():
            fragment = iterator.fragment()
            if fragment.isValid() and fragment.text().strip():
                formatting = fragment.charFormat()
                assert contrast(formatting.foreground().color(), background) >= 4.5
                if formatting.isAnchor():
                    links += 1
                    assert formatting.fontUnderline()
                    assert formatting.anchorHref().startswith("https://")
            iterator += 1
        block = block.next()
    assert links > 0
    # Tables wider than the viewport remain reachable horizontally.
    wide_table = ("|" + " Heading |" * 35 + "\n|" + " --- |" * 35
                  + "\n|" + " Cell |" * 35)
    wide_viewer = mm.create_changelog_viewer(wide_table, theme)
    wide_viewer.resize(650, 300)
    wide_viewer.show()
    app.processEvents()
    assert wide_viewer.horizontalScrollBar().maximum() > 0
    assert wide_viewer.horizontalScrollBar().isVisible()
    wide_viewer.close()
    viewer.close()
