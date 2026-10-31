
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from mesh_export_ui import (
    CompactMeshExportDialog,
    MeshExportDialog,
    choose_mesh_export,
    export_preferences,
)


@pytest.fixture
def parent(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = QWidget()
    widget.settings = QSettings(str(tmp_path / 'preferences.ini'), QSettings.Format.IniFormat)
    widget.appearance_theme_name = 'MeshMill Blue'
    widget.source_path = tmp_path / 'source.stl'
    widget._font_pixel_size = lambda: 12
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_remembered_options_survive_settings_reload(parent, tmp_path):
    assert not export_preferences(parent)['compact']
    dialog = MeshExportDialog(parent, 'Save', tmp_path / 'mesh.stl')
    dialog.destination.setCurrentText('Scan processing')
    dialog.select_format('.obj')
    dialog.merge.setChecked(False)
    dialog.remember_preferences()
    dialog.reject()
    parent.settings = QSettings(str(tmp_path / 'preferences.ini'), QSettings.Format.IniFormat)
    assert export_preferences(parent) == {'format': '.obj', 'merge': False,
                                          'software': 'Scan processing', 'compact': True}
    compact = CompactMeshExportDialog(parent, 'Save', tmp_path / 'next.stl')
    assert compact.filename.text() == 'next.obj'
    assert not compact.merge.isChecked()
    assert compact.executor is None
    compact.reject()


@pytest.mark.parametrize('suffix', ['.stl', '.glb', '.obj', '.dae', '.ply'])
def test_compact_options_and_confirmation(parent, tmp_path, suffix):
    options = {'format': suffix, 'merge': True, 'software': 'SketchUp'}
    compact = CompactMeshExportDialog(parent, 'Save', tmp_path / 'mesh.stl', options)
    assert compact.merge.isEnabled() == (suffix not in ('.stl', '.glb'))
    compact.confirm()
    assert compact.result_choice == (tmp_path / ('mesh' + suffix), suffix not in ('.stl', '.glb'))
    assert not export_preferences(parent)['compact']  # A normal save does not opt in.


def test_source_and_existing_output_protection(parent, tmp_path, monkeypatch):
    options = {'format': '.stl', 'merge': True, 'software': 'SketchUp'}
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    compact = CompactMeshExportDialog(parent, 'Save', parent.source_path, options)
    compact.confirm()
    assert compact.result_choice is None and warnings
    compact.filename.setText('existing.stl')
    (tmp_path / 'existing.stl').write_bytes(b'unchanged')
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.No)
    compact.confirm()
    assert compact.result_choice is None
    assert (tmp_path / 'existing.stl').read_bytes() == b'unchanged'
    compact.reject()


def test_comparison_switch_preserves_one_off_choices(parent, tmp_path, monkeypatch):
    parent.settings.setValue('export/compact_save', True)
    observed = []

    def compact_exec(dialog):
        dialog.filename.setText('another-name.stl')
        dialog.format.setCurrentIndex(dialog.format.findData('.ply'))
        dialog.merge.setChecked(False)
        dialog.show_comparison()

    def full_exec(dialog):
        observed.append((dialog.selected_format, dialog.filename.text(), dialog.merge.isChecked()))
        dialog.confirm()

    monkeypatch.setattr(CompactMeshExportDialog, 'exec', compact_exec)
    monkeypatch.setattr(MeshExportDialog, 'exec', full_exec)
    result = choose_mesh_export(parent, 'Save', tmp_path / 'mesh.stl')
    assert observed == [('.ply', 'another-name.ply', False)]
    assert result == (tmp_path / 'another-name.ply', False)
    assert export_preferences(parent)['format'] == '.dae'  # Not remembered implicitly.


def test_full_dialog_can_be_restored_as_default(parent, tmp_path, monkeypatch):
    parent.settings.setValue('export/compact_save', True)
    dialog = CompactMeshExportDialog(parent, 'Save', tmp_path / 'mesh.stl')
    dialog.show_comparison()
    assert not export_preferences(parent)['compact']
    visited = []
    monkeypatch.setattr(MeshExportDialog, 'exec', lambda dialog: (visited.append('full'), dialog.reject()))
    monkeypatch.setattr(CompactMeshExportDialog, 'exec', lambda dialog: pytest.fail('Unexpected compact dialog'))
    assert choose_mesh_export(parent, 'Save', tmp_path / 'mesh.stl') is None
    assert visited == ['full']


def test_saving_preferences_switches_immediately_to_compact(parent, tmp_path, monkeypatch):
    visited = []

    def full_exec(dialog):
        visited.append('full')
        dialog.select_format('.ply')
        dialog.merge.setChecked(False)
        dialog.filename.setText('chosen.ply')
        dialog.remember_preferences()

    def compact_exec(dialog):
        visited.append('compact')
        assert dialog.filename.text() == 'chosen.ply'
        assert not dialog.merge.isChecked()
        dialog.confirm()

    monkeypatch.setattr(MeshExportDialog, 'exec', full_exec)
    monkeypatch.setattr(CompactMeshExportDialog, 'exec', compact_exec)
    assert choose_mesh_export(parent, 'Save', tmp_path / 'mesh.stl') == (tmp_path / 'chosen.ply', False)
    assert visited == ['full', 'compact']
    visited.clear()
    choose_mesh_export(parent, 'Save', tmp_path / 'chosen.stl')
    assert visited == ['compact']


def test_compact_stats_update_for_format_and_merge(parent, tmp_path):
    from vtkmodules.vtkFiltersCore import vtkTriangleFilter
    from vtkmodules.vtkFiltersSources import vtkCubeSource

    cube = vtkCubeSource()
    triangles = vtkTriangleFilter()
    triangles.SetInputConnection(cube.GetOutputPort())
    triangles.Update()
    # Weld the cube's per-face vertices, as loaded application geometry does.
    from vtkmodules.vtkFiltersCore import vtkCleanPolyData
    clean = vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()
    options = {'format': '.ply', 'merge': True, 'software': 'Scan processing'}
    dialog = CompactMeshExportDialog(parent, 'Save', tmp_path / 'mesh.ply', options, clean.GetOutput())

    assert dialog.future is None and dialog.executor is None
    assert dialog.stats.text().startswith('12 triangles')
    assert 'calculated on save' in dialog.stats.text()
    dialog.merge.setChecked(False)
    assert dialog.future is None and dialog.executor is None
    assert dialog.stats.text().startswith('12 triangles')
    dialog.format.setCurrentIndex(dialog.format.findData('.stl'))
    assert dialog.stats.text() == '12 faces · 684 B'
    dialog.reject()


def test_save_dialog_does_not_merge_before_save(parent, tmp_path, monkeypatch):
    from vtkmodules.vtkFiltersSources import vtkSphereSource

    import mesh_export_ui

    def unexpected_analysis(*args, **kwargs):
        pytest.fail('Save dialog performed export geometry processing')

    monkeypatch.setattr(mesh_export_ui, 'estimate_geometry', unexpected_analysis)
    sphere = vtkSphereSource()
    sphere.Update()
    assert not export_preferences(parent)['merge']
    for dialog_type in (MeshExportDialog, CompactMeshExportDialog):
        dialog = dialog_type(parent, 'Save', tmp_path / 'mesh.dae', poly=sphere.GetOutput())
        assert not dialog.merge.isChecked()
        dialog.merge.setChecked(True)
        QApplication.processEvents()
        assert dialog.future is None and dialog.executor is None
        dialog.confirm()
        assert dialog.result_choice == (tmp_path / 'mesh.dae', True)
