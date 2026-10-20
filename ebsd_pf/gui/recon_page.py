"""Экран реконструкции: параметры, предпросмотр, сравнение с оригиналом, ручная правка маски."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from PySide6.QtCore import QRectF, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton, QRadioButton,
                               QScrollArea, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from ..core.exporter import grain_sheet, grain_title, sample_title
from ..core.layer import layer_blobs
from ..core.models import COLOR_MODES, LABEL_POSITIONS, ReconParams, format_plane
from ..core.render import layout_grain, render_sheet_image
from ..core.segmentation import overlay_image
from .image_view import ImageView, sync_views
from .mask_editor import MaskEditorDialog
from .preview import render_async
from .qt_utils import LabeledSlider, hint_label, rgb_to_pixmap
from .state import AppState

ROLE_ID = Qt.ItemDataRole.UserRole

# поле, подпись, тип, аргументы
SEG_SPEC = [
    ("color_threshold", "Порог выделения цвета", "slider", (5, 150, 1, 0, "")),
    ("min_object_size", "Мин. размер объекта (шум)", "slider", (1, 200, 1, 0, " пикс.")),
    ("edge_margin", "Отступ от окружности", "slider", (0, 10, 0.5, 1, " пикс.")),
    ("protect_saturated", "Не удалять мелкие насыщенные/тёмные точки", "check", None),
    ("include_dark_points", "Учитывать серые/чёрные (ахроматические) точки", "check", None),
    ("restore_under_axes", "Восстанавливать части кластеров под осями", "check", None),
]
ENH_SPEC = [
    ("dot_size", "Размер цветных пикселей", "slider", (1, 8, 0.5, 1, " пикс.")),
    ("cluster_scale", "Увеличение кластеров", "slider", (1, 4, 0.1, 1, " ×")),
    ("contrast", "Контрастность", "slider", (0.5, 3, 0.05, 2, "")),
    ("saturation", "Насыщенность", "slider", (0, 3, 0.05, 2, "")),
    ("jpeg_cleanup", "Убрать артефакты сжатия (ореолы JPEG)", "check", None),
    ("color_mode", "Цвет кластеров", "combo", COLOR_MODES),
]
DRAW_SPEC = [
    ("draw_axes", "Рисовать оси", "check", None),
    ("circle_width_pt", "Толщина окружности", "slider", (0.1, 4, 0.05, 2, " пт")),
    ("axis_width_pt", "Толщина осей", "slider", (0.05, 3, 0.05, 2, " пт")),
    ("smooth_scaling", "Сглаживание при масштабировании точек", "check", None),
    ("show_label", "Подпись индекса плоскости", "check", None),
    ("label_size_pt", "Размер подписи", "slider", (4, 36, 0.5, 1, " пт")),
    ("label_position", "Положение подписи", "combo", LABEL_POSITIONS),
    ("label_dx_mm", "Сдвиг подписи по X", "slider", (-20, 20, 0.1, 1, " мм")),
    ("label_dy_mm", "Сдвиг подписи по Y", "slider", (-20, 20, 0.1, 1, " мм")),
]


def compute_preview(pipeline, fig, params: ReconParams, sheet, px_per_mm: float, show_mask: bool,
                    show_removed: bool):
    seg = pipeline.segmentation(fig, params)
    if seg is None:
        return None
    overlay = overlay_image(seg, show_mask=show_mask, show_removed=show_removed, show_restored=show_mask)

    def provider(fid, p, radius_px):
        return pipeline.layer(fig, p, radius_px)
    img = render_sheet_image(sheet, px_per_mm, provider)
    pf = sheet.figures[0]
    lay = pipeline.layer(fig, params, pf.r * px_per_mm)
    n_out = layer_blobs(lay)[0] - 1 if lay is not None else 0
    return dict(seg=seg, overlay=overlay, image=img, n_out=n_out)


class ReconPage(QWidget):
    nextRequested = Signal()
    backRequested = Signal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.fig_id: Optional[str] = None
        self.widgets: dict = {}
        self._req = 0
        self._last_fig_shown = None
        self._loading = False

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Orientation.Horizontal)

        # --- слева: список фигур
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Фигуры"])
        ll.addWidget(self.tree, 1)
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Пред.")
        self.btn_nextfig = QPushButton("След. ▶")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_nextfig)
        ll.addLayout(nav)
        ll.addWidget(hint_label("★ — индивидуальные настройки, ✎ — ручная правка маски."))
        split.addWidget(left)

        # --- центр: сравнение
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(4, 0, 4, 0)
        top = QHBoxLayout()
        self.chk_mask = QCheckBox("Контур маски")
        self.chk_mask.setChecked(True)
        self.chk_removed = QCheckBox("Удалённые объекты")
        self.chk_removed.setChecked(True)
        self.chk_grain = QCheckBox("Показать зерно целиком")
        self.btn_mask = QPushButton("Править маску…")
        self.btn_fit = QPushButton("Вписать")
        for w in (self.chk_mask, self.chk_removed, self.chk_grain):
            top.addWidget(w)
        top.addStretch(1)
        top.addWidget(self.btn_mask)
        top.addWidget(self.btn_fit)
        cl.addLayout(top)
        vs = QSplitter(Qt.Orientation.Horizontal)
        wo = QWidget()
        lo = QVBoxLayout(wo)
        lo.setContentsMargins(0, 0, 0, 0)
        lab = QLabel("<b>Оригинал</b> (голубой — контур маски, пурпурный — удалено как шум, "
                     "зелёный — восстановлено под осью)")
        lab.setWordWrap(True)
        lo.addWidget(lab)
        self.view_orig = ImageView()
        lo.addWidget(self.view_orig)
        wr = QWidget()
        lr = QVBoxLayout(wr)
        lr.setContentsMargins(0, 0, 0, 0)
        self.recon_title = QLabel("<b>Реконструкция</b>")
        lr.addWidget(self.recon_title)
        self.view_recon = ImageView()
        self.view_recon.setBackgroundBrush(QBrush(QColor(200, 200, 200)))
        lr.addWidget(self.view_recon)
        vs.addWidget(wo)
        vs.addWidget(wr)
        vs.setStretchFactor(0, 1)
        vs.setStretchFactor(1, 1)
        vs.setSizes([500, 500])
        cl.addWidget(vs, 1)
        self.orig_panel = wo
        sync_views(self.view_orig, self.view_recon)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cl.addWidget(self.stats)
        split.addWidget(center)

        # --- справа: параметры
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        pl = QVBoxLayout(panel)

        gb_scope = QGroupBox("Область применения настроек")
        sl = QVBoxLayout(gb_scope)
        self.rb_all = QRadioButton("Общие настройки для всех фигур")
        self.rb_one = QRadioButton("Индивидуальные настройки этой фигуры")
        self.rb_all.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_all)
        grp.addButton(self.rb_one)
        sl.addWidget(self.rb_all)
        sl.addWidget(self.rb_one)
        brow = QHBoxLayout()
        self.btn_apply_all = QPushButton("Применить ко всем фигурам")
        self.btn_apply_grain = QPushButton("Применить к зерну")
        brow.addWidget(self.btn_apply_all)
        brow.addWidget(self.btn_apply_grain)
        sl.addLayout(brow)
        pl.addWidget(gb_scope)

        gb_mode = QGroupBox("Режим реконструкции")
        ml = QVBoxLayout(gb_mode)
        self.rb_exact = QRadioButton("Точный (геометрия и размеры как в оригинале)")
        self.rb_enh = QRadioButton("Повышение читаемости")
        for rb in (self.rb_exact, self.rb_enh):
            rb.setStyleSheet("QRadioButton { padding: 2px; }")
            ml.addWidget(rb)
        mg = QButtonGroup(self)
        mg.addButton(self.rb_exact)
        mg.addButton(self.rb_enh)
        pl.addWidget(gb_mode)

        self.gb_seg = self._build_group("Выделение цветных пикселей", SEG_SPEC)
        pl.addWidget(self.gb_seg)
        self.gb_enh = self._build_group("Улучшение читаемости", ENH_SPEC)
        self.gb_enh.layout().addRow(hint_label(
            "Увеличение выполняется вокруг центра каждого кластера — центры не смещаются. "
            "Новые точки не создаются."))
        pl.addWidget(self.gb_enh)
        self.gb_draw = self._build_group("Оформление", DRAW_SPEC)
        self.sl_diam = LabeledSlider(10, 150, 1, 40, 0, " мм")
        self.gb_draw.layout().insertRow(0, "Диаметр окружностей (все):", self.sl_diam)
        pl.addWidget(self.gb_draw)
        pl.addStretch(1)
        navb = QHBoxLayout()
        self.btn_back = QPushButton("← Группировка")
        self.btn_next = QPushButton("Далее: экспорт →")
        self.btn_next.setMinimumHeight(34)
        navb.addWidget(self.btn_back)
        navb.addWidget(self.btn_next)
        pl.addLayout(navb)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(450)
        split.addWidget(scroll)
        split.setSizes([200, 900, 470])
        root.addWidget(split)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.update_preview)

        self.tree.currentItemChanged.connect(self._on_tree_current)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_nextfig.clicked.connect(lambda: self._step(1))
        for w in (self.chk_mask, self.chk_removed, self.chk_grain):
            w.toggled.connect(self.schedule)
        self.chk_grain.toggled.connect(lambda on: self.orig_panel.setVisible(not on))
        self.btn_fit.clicked.connect(lambda: (self.view_recon.fit(), self.view_orig.fit()))
        self.btn_mask.clicked.connect(self.edit_mask)
        self.rb_one.toggled.connect(self._on_scope)
        self.rb_enh.toggled.connect(self._on_mode)
        self.btn_apply_all.clicked.connect(self.apply_to_all)
        self.btn_apply_grain.clicked.connect(self.apply_to_grain)
        self.sl_diam.valueChanged.connect(self._on_diam)
        self.btn_back.clicked.connect(self.backRequested)
        self.btn_next.clicked.connect(self.nextRequested)
        state.figuresChanged.connect(self._on_external)
        state.projectReplaced.connect(self._on_external)

    # ------------------------------------------------------------------ построение панели
    def _build_group(self, title: str, spec) -> QGroupBox:
        gb = QGroupBox(title)
        form = QFormLayout(gb)
        for field, label, kind, args in spec:
            if kind == "slider":
                lo, hi, step, dec, suf = args
                w = LabeledSlider(lo, hi, step, lo, dec, suf)
                w.valueChanged.connect(lambda v, f=field: self._on_param(f, v))
                form.addRow(label + ":", w)
            elif kind == "check":
                w = QCheckBox(label)
                w.toggled.connect(lambda v, f=field: self._on_param(f, v))
                form.addRow(w)
            else:
                w = QComboBox()
                for key, text in args.items():
                    w.addItem(text, key)
                w.currentIndexChanged.connect(lambda i, f=field, w=w: self._on_param(f, w.itemData(i)))
                form.addRow(label + ":", w)
            self.widgets[field] = w
        return gb

    @property
    def project(self):
        return self.state.project

    def fig(self):
        return self.project.figure(self.fig_id) if self.fig_id else None

    def current_params(self) -> ReconParams:
        f = self.fig()
        return self.project.params_for(f) if f else self.project.recon

    def _load_params(self) -> None:
        self._loading = True
        try:
            p = self.current_params()
            for field, w in self.widgets.items():
                v = getattr(p, field)
                if isinstance(w, LabeledSlider):
                    w.setValue(float(v))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                else:
                    idx = w.findData(v)
                    w.setCurrentIndex(max(idx, 0))
            (self.rb_enh if p.mode == "enhanced" else self.rb_exact).setChecked(True)
            self.gb_enh.setEnabled(p.mode == "enhanced")
            f = self.fig()
            (self.rb_one if (f is not None and f.params is not None) else self.rb_all).setChecked(True)
            self.rb_one.setEnabled(f is not None)
            self.sl_diam.setValue(self.project.layout.diameter_mm)
        finally:
            self._loading = False

    def _on_param(self, field: str, value) -> None:
        if self._loading:
            return
        p = self.current_params()
        old = getattr(p, field)
        if isinstance(old, bool):
            value = bool(value)
        elif isinstance(old, int) and not isinstance(old, bool):
            value = int(round(value))
        elif isinstance(old, float):
            value = float(value)
        setattr(p, field, value)
        self.state.mark_dirty()
        self.schedule()

    def _on_mode(self, enhanced: bool) -> None:
        self.gb_enh.setEnabled(enhanced)
        if not self._loading:
            self._on_param("mode", "enhanced" if enhanced else "exact")

    def _on_diam(self, v: float) -> None:
        if self._loading:
            return
        self.project.layout.diameter_mm = v
        self.state.notify_params()
        self.schedule()

    def _on_scope(self, individual: bool) -> None:
        if self._loading:
            return
        f = self.fig()
        if f is None:
            return
        if individual and f.params is None:
            f.params = self.project.recon.copy()
        elif not individual and f.params is not None:
            if QMessageBox.question(self, "Настройки",
                                    "Удалить индивидуальные настройки фигуры и использовать общие?"
                                    ) != QMessageBox.StandardButton.Yes:
                self._loading = True
                self.rb_one.setChecked(True)
                self._loading = False
                return
            f.params = None
        self.state.mark_dirty()
        self._load_params()
        self._refresh_tree_marks()
        self.schedule()

    def apply_to_all(self) -> None:
        n_ind = sum(1 for f in self.project.figures if f.params is not None)
        msg = "Применить текущие настройки ко всем фигурам проекта?"
        if n_ind:
            msg += f"\nИндивидуальные настройки ({n_ind} фиг.) будут удалены."
        if QMessageBox.question(self, "Применить ко всем", msg) != QMessageBox.StandardButton.Yes:
            return
        self.project.recon = self.current_params().copy()
        for f in self.project.figures:
            f.params = None
        self.state.notify_params()
        self._load_params()
        self._refresh_tree_marks()
        self.schedule()

    def apply_to_grain(self) -> None:
        f = self.fig()
        if f is None:
            return
        p = self.current_params()
        figs = self.project.grain_figures(f.sample_key, f.grain_key)
        for g in figs:
            g.params = p.copy()
        self.state.notify_params()
        self._load_params()
        self._refresh_tree_marks()
        self.state.statusMessage.emit(f"Настройки применены к {len(figs)} фигурам зерна")
        self.schedule()

    # ------------------------------------------------------------------ список фигур
    def on_enter(self) -> None:
        self.rebuild_tree()
        self._load_params()
        self.schedule()

    def _on_external(self) -> None:
        if self.isVisible():
            self.rebuild_tree()
            self._load_params()
            self.schedule()

    def rebuild_tree(self) -> None:
        bold = QFont()
        bold.setBold(True)
        with QSignalBlocker(self.tree):
            self.tree.clear()
            current = None
            first = None
            for sample, grains in self.project.grouped():
                si = QTreeWidgetItem([sample_title(sample)])
                si.setFont(0, bold)
                self.tree.addTopLevelItem(si)
                for grain, figs in grains:
                    gi = QTreeWidgetItem([grain_title(grain)])
                    si.addChild(gi)
                    for f in figs:
                        fi = QTreeWidgetItem([self._fig_text(f)])
                        fi.setData(0, ROLE_ID, f.id)
                        gi.addChild(fi)
                        first = first or fi
                        if f.id == self.fig_id:
                            current = fi
            self.tree.expandAll()
            if current is None and first is not None:
                current = first
                self.fig_id = first.data(0, ROLE_ID)
            if current is not None:
                self.tree.setCurrentItem(current)
            elif first is None:
                self.fig_id = None

    def _fig_text(self, f) -> str:
        marks = ("★" if f.params is not None else "") + ("✎" if f.mask_edits is not None and not f.mask_edits.is_empty() else "")
        return f"{format_plane(f.plane) or '{?}'} {marks}"

    def _refresh_tree_marks(self) -> None:
        def walk(item):
            for i in range(item.childCount()):
                ch = item.child(i)
                fid = ch.data(0, ROLE_ID)
                if fid:
                    f = self.project.figure(fid)
                    if f:
                        ch.setText(0, self._fig_text(f))
                walk(ch)
        walk(self.tree.invisibleRootItem())

    def _fig_items(self) -> list:
        out = []

        def walk(item):
            for i in range(item.childCount()):
                ch = item.child(i)
                if ch.data(0, ROLE_ID):
                    out.append(ch)
                walk(ch)
        walk(self.tree.invisibleRootItem())
        return out

    def _step(self, d: int) -> None:
        items = self._fig_items()
        ids = [it.data(0, ROLE_ID) for it in items]
        if not ids:
            return
        i = ids.index(self.fig_id) if self.fig_id in ids else 0
        j = max(0, min(len(ids) - 1, i + d))
        self.tree.setCurrentItem(items[j])

    def _on_tree_current(self, cur, _prev) -> None:
        if cur is None:
            return
        fid = cur.data(0, ROLE_ID)
        if not fid:
            return
        self.fig_id = fid
        self._load_params()
        self.schedule()

    def select_figure(self, fig_id: str) -> None:
        for it in self._fig_items():
            if it.data(0, ROLE_ID) == fig_id:
                self.tree.setCurrentItem(it)
                return

    # ------------------------------------------------------------------ предпросмотр
    def schedule(self, *_):
        self._timer.start()

    def update_preview(self) -> None:
        if not self.isVisible():
            return
        f = self.fig()
        if f is None:
            self.view_orig.set_pixmap(None)
            self.view_recon.set_pixmap(None)
            self.stats.setText("Нет полюсных фигур. Вернитесь на экран разметки.")
            return
        if not self.state.image_loaded(f.image_id):
            self.stats.setText("Исходное изображение этой фигуры не загружено.")
            return
        self._req += 1
        req = self._req
        if self.chk_grain.isChecked():
            sheet = grain_sheet(self.project, f.sample_key, f.grain_key)
            self.recon_title.setText(f"<b>Реконструкция зерна</b> — {sample_title(f.sample_key)}, "
                                     f"{grain_title(f.grain_key).lower()}")

            def done_grain(img, req=req):
                if req == self._req:
                    keep = self._last_fig_shown == ("grain", f.id)
                    self.view_recon.set_pixmap(QPixmap.fromImage(img), keep_view=True)
                    if not keep:
                        QTimer.singleShot(0, self.view_recon.fit)
                    self._last_fig_shown = ("grain", f.id)
            render_async(self.state, sheet, 10.0, done_grain)
            return
        self.recon_title.setText("<b>Реконструкция</b>")
        params = self.current_params().copy()
        ls = replace(self.project.layout, show_group_title=False, columns=1)
        sheet = layout_grain([(f.id, f.plane_label, params)], ls)
        k = max(1, min(6, math.ceil(520 / (2 * f.r))))
        ppm = 2 * f.r * k / ls.diameter_mm
        fig_copy = replace(f, mask_edits=f.mask_edits.copy() if f.mask_edits else None)
        show_mask, show_removed = self.chk_mask.isChecked(), self.chk_removed.isChecked()

        def done(res, req=req, k=k, ppm=ppm, sheet=sheet, fid=f.id):
            if req != self._req or res is None:
                return
            keep = self._last_fig_shown == ("fig", fid)
            self._last_fig_shown = ("fig", fid)
            seg = res["seg"]
            pf = sheet.figures[0]
            cxp, cyp = pf.cx * ppm, pf.cy * ppm
            rect = QRectF(0, 0, sheet.width * ppm, sheet.height * ppm)
            self.view_recon.set_pixmap(QPixmap.fromImage(res["image"]), keep_view=True, scene_rect=rect)
            self.view_orig.set_pixmap(rgb_to_pixmap(res["overlay"]), keep_view=True, smooth=False,
                                      offset=(cxp - seg.crop.cx * k, cyp - seg.crop.cy * k), scale=k,
                                      scene_rect=rect)
            if not keep:
                QTimer.singleShot(0, self.view_recon.fit)
            text = seg.stats_text().replace("\n", "  •  ")
            if res["n_out"] < seg.n_blobs:
                text += (f"  •  <span style='color:#b00'>При увеличении слилось кластеров: "
                         f"{seg.n_blobs - res['n_out']} (уменьшите размер точек)</span>")
            if seg.removed:
                text += "  •  <span style='color:#a0a'>Удалённые объекты обведены пурпурным на оригинале</span>"
            self.stats.setText(text)

        self.state.runner.run("Предпросмотр", compute_preview, self.state.pipeline, fig_copy, params,
                              sheet, ppm, show_mask, show_removed, on_done=done, silent=True)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self.schedule()

    # ------------------------------------------------------------------ маска
    def edit_mask(self) -> None:
        f = self.fig()
        if f is None or not self.state.image_loaded(f.image_id):
            return
        dlg = MaskEditorDialog(self.state, f, self.current_params().copy(), self)
        try:
            if dlg.exec():
                edits = dlg.result_edits()
                f.mask_edits = None if edits is None or edits.is_empty() else edits
                f.mask_version += 1
                self.state.notify_geometry(f.id)
                self._refresh_tree_marks()
                self.schedule()
        finally:
            dlg.deleteLater()   # иначе диалог остаётся дочерним объектом страницы до закрытия программы
