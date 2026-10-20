"""Сквозной тест интерфейса (offscreen): загрузка → разметка → группировка → реконструкция → экспорт → проект."""
import os
import time

import pytest
from PIL import Image

from synthetic import make_composite


@pytest.fixture()
def window(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QStandardPaths
    from PySide6.QtWidgets import QMessageBox
    QStandardPaths.setTestModeEnabled(True)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    from ebsd_pf.gui.main_window import MainWindow
    w = MainWindow(auto_start=False)
    w.show()
    yield w
    w.state.mark_dirty(False)
    w.state.runner.wait(10000)
    w._remove_recovery()
    w.close()
    # окно удаляется до завершения интерпретатора (иначе порядок разрушения Qt-объектов не определён)
    from PySide6.QtCore import QCoreApplication, QEvent
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del w


def pump(app, w, sec=0.4, timeout=60):
    from PySide6.QtCore import QEventLoop
    t0 = time.time()
    while time.time() - t0 < sec or w.state.runner._tasks:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(0.01)
        if time.time() - t0 > timeout:
            raise TimeoutError
    app.processEvents()


def test_full_workflow(qapp, window, tmp_path):
    w = window
    s = make_composite(seed=7, jpeg_quality=92)
    img_path = str(tmp_path / "composite.jpg")
    Image.fromarray(s.rgb).save(img_path, quality=95)

    # 1. загрузка + автоматический поиск
    w.open_paths([img_path])
    pump(qapp, w)
    pr = w.state.project
    assert len(pr.images) == 1 and len(pr.figures) == len(s.figures)
    assert w.stack.currentIndex() == 1

    # 2. массовое назначение
    mp = w.markup_page
    mp.view.scene().clearSelection()
    mp.ed_sample.setEditText("S1")
    mp.spin_per_grain.setValue(4)
    mp.bulk_grains()
    mp.bulk_planes()
    assert sorted({f.grain for f in pr.figures}) == ["1", "2"]
    assert [f.plane for f in mp.figures()][:4] == ["100", "110", "111", "112"]
    # одиночное назначение с переходом к следующей
    first = mp.figures()[0]
    mp.items[first.id].setSelected(True)
    mp.ed_plane.setEditText("{1 0 0}")
    mp.apply_attributes()
    assert first.plane == "100" and mp.selected_ids() != [first.id]
    # ручное добавление (с привязкой к линии) и удаление
    n0 = len(pr.figures)
    mp._on_circle_drawn(first.cx + 3, first.cy - 2, first.r * 1.05)
    new = pr.figures[-1]
    assert len(pr.figures) == n0 + 1 and abs(new.r - first.r) < 1.0
    mp.view.scene().clearSelection()
    mp.items[new.id].setSelected(True)
    mp.delete_selected()
    assert len(pr.figures) == n0
    # перемещение окружности
    it = mp.items[first.id]
    it.setPos(first.cx + 5, first.cy)
    mp._on_geometry_edited(first.id)
    assert abs(first.cx - (it.pos().x())) < 1e-9
    mp.nudge(0, 0, 0)

    # 3. группировка
    w.goto(2)
    pump(qapp, w)
    gp = w.group_page
    g1 = pr.grain_figures("S1", "1")
    gp.move_figures([g1[0].id], "S1", "2", 0)
    assert len(pr.grain_figures("S1", "2")) == 5 and pr.grain_figures("S1", "2")[0].id == g1[0].id
    gp.move_figures([g1[0].id], "S1", "1", 0)
    gp.sort_by_plane()
    assert [f.plane for f in pr.grain_figures("S1", "1")] == ["100", "110", "111", "112"]
    gp.move_grains([("S1", "2")], "S2")
    assert pr.samples() == ["S1", "S2"]
    assert gp.tree.topLevelItemCount() == 2

    # 4. реконструкция: общие и индивидуальные параметры, правка маски
    w.goto(3)
    pump(qapp, w, 0.8)
    rp = w.recon_page
    assert rp.fig_id is not None and "Цветных пикселей" in rp.stats.text()
    rp.rb_enh.setChecked(True)
    rp.widgets["dot_size"].spin.setValue(2.5)
    pump(qapp, w, 0.5)
    assert pr.recon.mode == "enhanced" and pr.recon.dot_size == 2.5
    rp.rb_one.setChecked(True)
    rp.widgets["saturation"].spin.setValue(1.7)
    f = rp.fig()
    assert f.params is not None and f.params.saturation == 1.7 and pr.recon.saturation == 1.0

    from ebsd_pf.gui.mask_editor import MaskEditorDialog
    dlg = MaskEditorDialog(w.state, f, rp.current_params().copy())
    before = dlg.seg.n_blobs
    dlg.rb_rem.setChecked(True)
    dlg.brush.setValue(int(f.r * 3))
    from PySide6.QtCore import QPointF
    dlg.begin_stroke()
    dlg._paint_at(QPointF(dlg.crop.cx, dlg.crop.cy))
    dlg.resegment()
    assert dlg.seg.n_blobs < before
    dlg.undo()
    assert dlg.seg.n_blobs == before
    dlg.reject()
    from PySide6.QtCore import QCoreApplication, QEvent
    dlg.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del dlg

    # 5. экспорт
    w.goto(4)
    pump(qapp, w)
    ep = w.export_page
    ep.ed_dir.setText(str(tmp_path / "export"))
    for cb in ep.fmt_checks.values():
        cb.setChecked(True)
    ep.rb_all.setChecked(True)
    ep.do_export()
    pump(qapp, w, 0.5)
    out = [os.path.join(dp, fn) for dp, _, fns in os.walk(tmp_path / "export") for fn in fns]
    assert len(out) == 2 * 4

    # 6. сохранение и повторное открытие
    proj = str(tmp_path / "test.ebsdproj")
    assert w._save_to(proj)
    assert not w.state.dirty
    w.open_project(proj)
    pump(qapp, w)
    p2 = w.state.project
    assert len(p2.figures) == len(s.figures)
    assert all(w.state.image_loaded(i.id) for i in p2.images)
    assert p2.figure(f.id).params.saturation == 1.7
    assert w.stack.currentIndex() == 4

    # автосохранение для восстановления после сбоя
    w.state.mark_dirty()
    w.autosave()
    assert os.path.exists(w.recovery_path)
