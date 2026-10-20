"""Экран экспорта: фигура, зерно, образец, все образцы; PNG/TIFF/SVG/PDF."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox,
                               QProgressBar, QPushButton, QRadioButton, QSplitter, QVBoxLayout, QWidget)

from ..core import exporter
from ..core.exporter import ExportJob, grain_title, sample_title
from ..core.models import format_plane
from .image_view import ImageView
from .preview import render_async
from .qt_utils import LabeledSlider, hint_label
from .state import AppState

FORMATS = [("png", "PNG"), ("tiff", "TIFF"), ("svg", "SVG (вектор)"), ("pdf", "PDF (вектор)")]


class ExportPage(QWidget):
    backRequested = Signal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._task = None
        self._req = 0
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        gb = QGroupBox("Что экспортировать")
        g = QFormLayout(gb)
        self.rb_fig = QRadioButton("Отдельная полюсная фигура")
        self.rb_grain = QRadioButton("Комплект фигур одного зерна")
        self.rb_sample = QRadioButton("Все зерна образца")
        self.rb_all = QRadioButton("Все образцы (пакетный экспорт)")
        self.rb_grain.setChecked(True)
        self.cmb_fig = QComboBox()
        self.cmb_grain = QComboBox()
        self.cmb_sample = QComboBox()
        bg = QButtonGroup(self)
        for rb, cmb in ((self.rb_fig, self.cmb_fig), (self.rb_grain, self.cmb_grain),
                        (self.rb_sample, self.cmb_sample), (self.rb_all, None)):
            bg.addButton(rb)
            if cmb is not None:
                g.addRow(rb, cmb)
            else:
                g.addRow(rb)
        self.chk_combine = QCheckBox("Дополнительно: все зерна образца на одном листе")
        g.addRow(self.chk_combine)
        ll.addWidget(gb)

        gbf = QGroupBox("Форматы и качество")
        f = QFormLayout(gbf)
        frow = QHBoxLayout()
        self.fmt_checks = {}
        for key, text in FORMATS:
            cb = QCheckBox(text)
            self.fmt_checks[key] = cb
            frow.addWidget(cb)
        f.addRow("Форматы:", frow)
        self.cmb_dpi = QComboBox()
        self.cmb_dpi.setEditable(True)
        self.cmb_dpi.addItems(["300", "600", "1200"])
        f.addRow("Разрешение PNG/TIFF, DPI:", self.cmb_dpi)
        self.sl_diam = LabeledSlider(10, 150, 1, 40, 0, " мм")
        f.addRow("Диаметр окружностей:", self.sl_diam)
        self.sl_spacing = LabeledSlider(0, 40, 0.5, 5, 1, " мм")
        f.addRow("Расстояние между окружностями:", self.sl_spacing)
        self.chk_titles = QCheckBox("Подписи образца и зерна на рисунке зерна")
        f.addRow(self.chk_titles)
        self.cmb_points = QComboBox()
        self.cmb_points.addItem("Встроенное растровое изображение (точно)", False)
        self.cmb_points.addItem("Векторные контуры кластеров", True)
        f.addRow("Точки в SVG/PDF:", self.cmb_points)
        self.cmb_layer_dpi = QComboBox()
        self.cmb_layer_dpi.addItems(["300", "600", "1200"])
        f.addRow("Разрешение растрового слоя SVG/PDF:", self.cmb_layer_dpi)
        self.size_info = QLabel("")
        f.addRow(self.size_info)
        f.addRow(hint_label("Окружности, оси и подписи в SVG/PDF всегда векторные. Векторные контуры "
                            "точно повторяют пиксели кластеров; при более чем 12 цветах в кластере "
                            "используется его медианный цвет."))
        ll.addWidget(gbf)

        gbo = QGroupBox("Папка назначения")
        o = QHBoxLayout(gbo)
        self.ed_dir = QLineEdit()
        self.btn_dir = QPushButton("Обзор…")
        o.addWidget(self.ed_dir, 1)
        o.addWidget(self.btn_dir)
        ll.addWidget(gbo)

        erow = QHBoxLayout()
        self.btn_export = QPushButton("Экспортировать")
        self.btn_export.setMinimumHeight(38)
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.setEnabled(False)
        self.btn_open = QPushButton("Открыть папку")
        erow.addWidget(self.btn_export, 1)
        erow.addWidget(self.btn_cancel)
        erow.addWidget(self.btn_open)
        ll.addLayout(erow)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        ll.addWidget(self.progress)
        self.log = QListWidget()
        ll.addWidget(self.log, 1)
        self.btn_back = QPushButton("← Реконструкция")
        ll.addWidget(self.btn_back)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 0, 0)
        rl.addWidget(QLabel("<b>Предпросмотр экспортируемого рисунка</b>"))
        self.preview = ImageView()
        self.preview.setBackgroundBrush(QBrush(QColor(200, 200, 200)))
        rl.addWidget(self.preview, 1)
        split.addWidget(right)
        split.setSizes([480, 700])
        root.addWidget(split)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.update_preview)

        for rb in (self.rb_fig, self.rb_grain, self.rb_sample, self.rb_all):
            rb.toggled.connect(self._on_change)
        for cmb in (self.cmb_fig, self.cmb_grain, self.cmb_sample, self.cmb_dpi, self.cmb_points,
                    self.cmb_layer_dpi):
            cmb.currentIndexChanged.connect(self._on_change)
        self.cmb_dpi.editTextChanged.connect(self._on_change)
        for cb in list(self.fmt_checks.values()) + [self.chk_combine, self.chk_titles]:
            cb.toggled.connect(self._on_change)
        self.sl_diam.valueChanged.connect(self._on_change)
        self.sl_spacing.valueChanged.connect(self._on_change)
        self.ed_dir.editingFinished.connect(self._on_change)
        self.btn_dir.clicked.connect(self._choose_dir)
        self.btn_export.clicked.connect(self.do_export)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_open.clicked.connect(self._open_dir)
        self.btn_back.clicked.connect(self.backRequested)
        self._loading = False

    @property
    def project(self):
        return self.state.project

    # ------------------------------------------------------------------
    def on_enter(self) -> None:
        self._loading = True
        try:
            pr = self.project
            es = pr.export
            for cmb in (self.cmb_fig, self.cmb_grain, self.cmb_sample):
                cmb.blockSignals(True)
            cur_fig, cur_grain, cur_sample = (self.cmb_fig.currentData(), self.cmb_grain.currentData(),
                                              self.cmb_sample.currentData())
            self.cmb_fig.clear()
            self.cmb_grain.clear()
            self.cmb_sample.clear()
            for s, grains in pr.grouped():
                self.cmb_sample.addItem(sample_title(s), s)
                for gname, figs in grains:
                    self.cmb_grain.addItem(f"{sample_title(s)} / {grain_title(gname).lower()}", (s, gname))
                    for fg in figs:
                        self.cmb_fig.addItem(f"{sample_title(s)} / {grain_title(gname).lower()} / "
                                             f"{format_plane(fg.plane) or '{?}'}", fg.id)
            for cmb, cur in ((self.cmb_fig, cur_fig), (self.cmb_grain, cur_grain), (self.cmb_sample, cur_sample)):
                i = cmb.findData(cur) if cur is not None else -1
                cmb.setCurrentIndex(max(i, 0))
                cmb.blockSignals(False)
            {"figure": self.rb_fig, "grain": self.rb_grain, "sample": self.rb_sample,
             "all": self.rb_all}.get(es.scope, self.rb_grain).setChecked(True)
            for k, cb in self.fmt_checks.items():
                cb.setChecked(k in es.formats)
            self.cmb_dpi.setEditText(str(es.dpi))
            self.sl_diam.setValue(pr.layout.diameter_mm)
            self.sl_spacing.setValue(pr.layout.spacing_mm)
            self.chk_titles.setChecked(pr.layout.show_group_title)
            self.chk_combine.setChecked(es.combine_sample)
            self.cmb_points.setCurrentIndex(1 if es.vector_points else 0)
            self.cmb_layer_dpi.setCurrentText(str(es.raster_layer_dpi))
            if not es.output_dir and self.state.path:
                es.output_dir = os.path.join(os.path.dirname(self.state.path), "export")
            self.ed_dir.setText(es.output_dir)
        finally:
            self._loading = False
        self._update_info()
        self._timer.start()

    def _dpi(self) -> int:
        try:
            return max(50, min(4800, int(float(self.cmb_dpi.currentText()))))
        except ValueError:
            return 600

    def _on_change(self, *_):
        if self._loading:
            return
        pr = self.project
        es = pr.export
        es.scope = ("figure" if self.rb_fig.isChecked() else "grain" if self.rb_grain.isChecked()
                    else "sample" if self.rb_sample.isChecked() else "all")
        es.formats = [k for k, cb in self.fmt_checks.items() if cb.isChecked()]
        es.dpi = self._dpi()
        es.combine_sample = self.chk_combine.isChecked()
        es.vector_points = bool(self.cmb_points.currentData())
        es.raster_layer_dpi = int(self.cmb_layer_dpi.currentText())
        es.output_dir = self.ed_dir.text().strip()
        changed_layout = (pr.layout.diameter_mm != self.sl_diam.value()
                          or pr.layout.spacing_mm != self.sl_spacing.value()
                          or pr.layout.show_group_title != self.chk_titles.isChecked())
        pr.layout.diameter_mm = self.sl_diam.value()
        pr.layout.spacing_mm = self.sl_spacing.value()
        pr.layout.show_group_title = self.chk_titles.isChecked()
        self.state.mark_dirty()
        if changed_layout:
            self.state.paramsChanged.emit()
        self._update_info()
        self._timer.start()

    def job(self) -> ExportJob:
        if self.rb_fig.isChecked():
            return ExportJob("figure", fig_id=self.cmb_fig.currentData() or "")
        if self.rb_grain.isChecked():
            s, g = self.cmb_grain.currentData() or ("", "")
            return ExportJob("grain", sample=s, grain=g)
        if self.rb_sample.isChecked():
            return ExportJob("sample", sample=self.cmb_sample.currentData() or "")
        return ExportJob("all")

    def _preview_sheet(self):
        job = self.job()
        try:
            items = exporter.plan(self.project, job, self.project.export)
        except exporter.ExportError:
            return None, 0
        return (items[0][1] if items else None), len(items)

    def _update_info(self) -> None:
        sheet, n = self._preview_sheet()
        if sheet is None:
            self.size_info.setText("Нет фигур для экспорта")
            return
        w, h = exporter.raster_size(sheet, self._dpi())
        self.size_info.setText(f"Рисунков: {n}. Первый: {sheet.width:.1f}×{sheet.height:.1f} мм, "
                               f"{w}×{h} пикс. при {self._dpi()} DPI")

    def update_preview(self) -> None:
        if not self.isVisible():
            return
        sheet, _ = self._preview_sheet()
        if sheet is None:
            self.preview.set_pixmap(None)
            return
        self._req += 1
        req = self._req

        def done(img, req=req):
            if req == self._req:
                self.preview.set_pixmap(QPixmap.fromImage(img), keep_view=True)
                QTimer.singleShot(0, self.preview.fit)
        render_async(self.state, sheet, 8.0, done)

    def _choose_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Папка для экспорта", self.ed_dir.text() or os.path.expanduser("~"))
        if d:
            self.ed_dir.setText(d)
            self._on_change()

    def _open_dir(self) -> None:
        d = self.ed_dir.text().strip()
        if d and os.path.isdir(d):
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    # ------------------------------------------------------------------ экспорт
    def do_export(self) -> None:
        self._on_change()
        es = self.project.export
        if not es.formats:
            QMessageBox.warning(self, "Экспорт", "Выберите хотя бы один формат.")
            return
        if not es.output_dir:
            self._choose_dir()
            es = self.project.export
            if not es.output_dir:
                return
        job = self.job()
        try:
            items = exporter.plan(self.project, job, es)
        except exporter.ExportError as e:
            QMessageBox.warning(self, "Экспорт", str(e))
            return
        missing = {self.project.image(self.project.figure(pf.fig_id).image_id).name
                   for _, sh in items for pf in sh.figures
                   if not self.state.image_loaded(self.project.figure(pf.fig_id).image_id)}
        if missing:
            QMessageBox.warning(self, "Экспорт", "Не загружены исходные изображения:\n" + "\n".join(sorted(missing)))
            return
        ext = {"png": ".png", "tiff": ".tif", "svg": ".svg", "pdf": ".pdf"}
        exists = [os.path.join(es.output_dir, rel) + ext[f] for rel, _ in items for f in es.formats
                  if os.path.exists(os.path.join(es.output_dir, rel) + ext[f])]
        if exists and QMessageBox.question(
                self, "Экспорт", f"{len(exists)} файл(ов) уже существует и будет перезаписано. Продолжить?"
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            os.makedirs(es.output_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Экспорт", f"Не удалось создать папку: {e}")
            return
        from dataclasses import replace
        settings = replace(es, formats=list(es.formats))
        self.btn_export.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.log.addItem(f"Экспорт начат: {len(items)} рисунков × {len(settings.formats)} форматов")

        def done(paths):
            self.btn_export.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.progress.setValue(100)
            for p in paths:
                self.log.addItem(p)
            self.log.addItem(f"Готово: записано файлов {len(paths)}")
            self.log.scrollToBottom()
            self.state.statusMessage.emit(f"Экспорт завершён: {len(paths)} файлов")

        def err(msg, tb):
            self.btn_export.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.log.addItem(f"Ошибка: {msg}")
            self.state.runner.default_error("Экспорт", msg, tb)

        def cancelled_or_progress(v):
            self.progress.setValue(v)

        import copy
        snapshot = copy.deepcopy(self.project)   # GUI может менять проект во время экспорта
        self._task = self.state.runner.run("Экспорт", exporter.run_export, snapshot, self.state.pipeline,
                                           job, settings, with_progress=True, on_done=done, on_error=err,
                                           on_progress=cancelled_or_progress)

    def _cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self.log.addItem("Экспорт отменён пользователем")
            self.btn_export.setEnabled(True)
            self.btn_cancel.setEnabled(False)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._timer.start()
