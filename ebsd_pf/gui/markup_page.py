"""Экран разметки: поиск окружностей, ручная корректировка и назначение атрибутов."""
from __future__ import annotations

import logging
import re
from typing import Optional

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from ..core.detection import detect_circles, refine_circle
from ..core.geometry import cluster_by_proximity, reading_order
from ..core.models import PLANE_PRESETS, PoleFigure, format_plane, natural_key, new_id, normalize_plane, plane_sort_key
from .circle_items import CircleItem
from .detect_dialog import DetectDialog
from .image_view import ImageView
from .qt_utils import hint_label, rgb_to_pixmap
from .state import AppState

log = logging.getLogger(__name__)


def parse_plane_list(text: str) -> list:
    parts = [normalize_plane(p) for p in re.split(r"[,;]+|\s{2,}", text) if p.strip()]
    if len(parts) == 1 and " " in text.strip():
        parts = [normalize_plane(p) for p in text.split()]
    return [p for p in parts if p]


class MarkupPage(QWidget):
    nextRequested = Signal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self.image_id: Optional[str] = None
        self.items: dict[str, CircleItem] = {}
        self._sync = False
        self._fitted_once = False

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        tb = QHBoxLayout()
        tb.addWidget(QLabel("Изображение:"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(220)
        tb.addWidget(self.image_combo)

        def tbtn(text, tip, checkable=False):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            tb.addWidget(b)
            return b

        self.btn_detect = tbtn("Найти окружности", "Автоматический поиск полюсных фигур на текущем изображении")
        self.btn_detect_all = tbtn("Найти на всех", "Поиск на всех изображениях проекта")
        self.btn_detect_params = tbtn("Параметры поиска…", "Настройка автоматического поиска")
        tb.addSpacing(10)
        self.btn_add = tbtn("➕ Добавить окружность",
                            "Щелчок — центр, второй щелчок — точка на окружности. Esc — выход из режима", True)
        self.chk_snap = QCheckBox("Привязка к линии")
        self.chk_snap.setToolTip("Уточнять центр и радиус новой окружности по линии на изображении")
        self.chk_snap.setChecked(True)
        tb.addWidget(self.chk_snap)
        self.btn_refine = tbtn("Уточнить", "Подогнать выбранные окружности к линии на изображении")
        self.btn_delete = tbtn("Удалить", "Удалить выбранные окружности (Delete)")
        self.btn_clear = tbtn("Удалить все", "Удалить все окружности на текущем изображении")
        tb.addSpacing(10)
        self.btn_fit = tbtn("Вписать", "Показать изображение целиком")
        self.btn_100 = tbtn("1:1", "Масштаб 100%")
        self.zoom_label = QLabel("")
        tb.addWidget(self.zoom_label)
        tb.addStretch(1)
        root.addLayout(tb)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.view = ImageView()
        split.addWidget(self.view)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(4, 0, 0, 0)
        pl.addWidget(QLabel("<b>Полюсные фигуры на изображении</b>"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["№", "Образец", "Зерно", "Индекс"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150)
        pl.addWidget(self.table, 2)

        gb = QGroupBox("Атрибуты выбранных фигур")
        form = QFormLayout(gb)
        self.sel_label = QLabel("Ничего не выбрано")
        form.addRow(self.sel_label)
        self.ed_sample = QComboBox()
        self.ed_sample.setEditable(True)
        self.ed_grain = QComboBox()
        self.ed_grain.setEditable(True)
        self.ed_plane = QComboBox()
        self.ed_plane.setEditable(True)
        self.ed_plane.addItems(PLANE_PRESETS)
        self.plane_preview = QLabel("")
        prow = QHBoxLayout()
        prow.addWidget(self.ed_plane, 1)
        prow.addWidget(self.plane_preview)
        form.addRow("Образец:", self.ed_sample)
        form.addRow("Зерно:", self.ed_grain)
        form.addRow("Индекс плоскости:", prow)
        arow = QHBoxLayout()
        self.btn_apply = QPushButton("Применить (Enter)")
        self.btn_next_grain = QPushButton("Зерно +1")
        self.btn_next_grain.setToolTip("Увеличить номер зерна в поле на 1")
        arow.addWidget(self.btn_apply, 1)
        arow.addWidget(self.btn_next_grain)
        form.addRow(arow)
        self.chk_advance = QCheckBox("После применения выделять следующую фигуру")
        self.chk_advance.setChecked(True)
        form.addRow(self.chk_advance)
        form.addRow(hint_label("Пустое поле не изменяет значение. Индекс вводится без скобок: 111 → {111}."))
        pl.addWidget(gb)

        gb2 = QGroupBox("Массовое назначение")
        f2 = QFormLayout(gb2)
        f2.addRow(hint_label("Применяется к выбранным фигурам, а если ничего не выбрано — ко всем "
                             "на изображении. Образец берётся из поля «Образец»."))
        self.ed_plane_list = QLineEdit("100, 110, 111, 112")
        f2.addRow("Индексы по порядку:", self.ed_plane_list)
        self.btn_bulk_planes = QPushButton("Назначить индексы по порядку (слева направо, сверху вниз)")
        f2.addRow(self.btn_bulk_planes)
        row = QHBoxLayout()
        self.spin_per_grain = QSpinBox()
        self.spin_per_grain.setRange(1, 64)
        self.spin_per_grain.setValue(4)
        self.spin_first_grain = QSpinBox()
        self.spin_first_grain.setRange(0, 100000)
        self.spin_first_grain.setValue(1)
        row.addWidget(QLabel("Фигур в зерне:"))
        row.addWidget(self.spin_per_grain)
        row.addWidget(QLabel("Первое зерно:"))
        row.addWidget(self.spin_first_grain)
        f2.addRow(row)
        self.btn_bulk_grains = QPushButton("Нумеровать зёрна по порядку")
        self.btn_auto_group = QPushButton("Автогруппировка по расположению (зерна + индексы)")
        self.btn_auto_group.setToolTip("Соседние окружности объединяются в зерно; внутри зерна индексы "
                                       "назначаются по порядку из списка выше; образец — из поля «Образец»")
        f2.addRow(self.btn_bulk_grains)
        f2.addRow(self.btn_auto_group)
        pl.addWidget(gb2)
        pl.addWidget(hint_label(
            "Мышь: колесо — масштаб, средняя кнопка или Пробел+перетаскивание — перемещение вида, "
            "перетаскивание окружности — сдвиг, квадратный маркер выбранной окружности — радиус, "
            "рамка — выделение нескольких. Клавиши: стрелки — сдвиг на 1 пикс. (Shift — 10), "
            "+/− — радиус, Delete — удалить."))
        pl.addStretch(1)
        self.btn_next = QPushButton("Далее: группировка →")
        self.btn_next.setMinimumHeight(34)
        pl.addWidget(self.btn_next)
        panel.setMinimumWidth(360)
        split.addWidget(panel)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([900, 400])
        root.addWidget(split, 1)

        # --- сигналы
        self.image_combo.currentIndexChanged.connect(self._on_image_combo)
        self.btn_detect.clicked.connect(lambda: self.detect([self.image_id] if self.image_id else []))
        self.btn_detect_all.clicked.connect(lambda: self.detect([i.id for i in self.state.project.images]))
        self.btn_detect_params.clicked.connect(self._detect_params)
        self.btn_add.toggled.connect(self.view.set_add_mode)
        self.view.addModeCancelled.connect(lambda: self.btn_add.setChecked(False))
        self.view.circleDrawn.connect(self._on_circle_drawn)
        self.btn_refine.clicked.connect(self.refine_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_clear.clicked.connect(self.clear_image)
        self.btn_fit.clicked.connect(self.view.fit)
        self.btn_100.clicked.connect(lambda: self.view.set_zoom(1.0))
        self.view.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.view.scene().selectionChanged.connect(self._on_scene_selection)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.itemDoubleClicked.connect(lambda _: self._center_on_selected())
        self.btn_apply.clicked.connect(self.apply_attributes)
        for cb in (self.ed_sample, self.ed_grain, self.ed_plane):
            cb.lineEdit().returnPressed.connect(self.apply_attributes)
        self.ed_plane.editTextChanged.connect(lambda t: self.plane_preview.setText(format_plane(t)))
        self.btn_next_grain.clicked.connect(self._grain_plus)
        self.btn_bulk_planes.clicked.connect(self.bulk_planes)
        self.btn_bulk_grains.clicked.connect(self.bulk_grains)
        self.btn_auto_group.clicked.connect(self.auto_group)
        self.btn_next.clicked.connect(self.nextRequested)

        for key, fn in [(QKeySequence.StandardKey.Delete, self.delete_selected),
                        (QKeySequence(Qt.Key.Key_Left), lambda: self.nudge(-1, 0, 0)),
                        (QKeySequence(Qt.Key.Key_Right), lambda: self.nudge(1, 0, 0)),
                        (QKeySequence(Qt.Key.Key_Up), lambda: self.nudge(0, -1, 0)),
                        (QKeySequence(Qt.Key.Key_Down), lambda: self.nudge(0, 1, 0)),
                        (QKeySequence("Shift+Left"), lambda: self.nudge(-10, 0, 0)),
                        (QKeySequence("Shift+Right"), lambda: self.nudge(10, 0, 0)),
                        (QKeySequence("Shift+Up"), lambda: self.nudge(0, -10, 0)),
                        (QKeySequence("Shift+Down"), lambda: self.nudge(0, 10, 0)),
                        (QKeySequence(Qt.Key.Key_Plus), lambda: self.nudge(0, 0, 1)),
                        (QKeySequence(Qt.Key.Key_Equal), lambda: self.nudge(0, 0, 1)),
                        (QKeySequence(Qt.Key.Key_Minus), lambda: self.nudge(0, 0, -1))]:
            sc = QShortcut(key, self.view)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

        state.projectReplaced.connect(self._on_project_replaced)
        state.imagesChanged.connect(self.refresh_images)
        state.figuresChanged.connect(self._on_figures_changed)

    # ------------------------------------------------------------------ данные
    @property
    def project(self):
        return self.state.project

    def figures(self) -> list:
        if not self.image_id:
            return []
        figs = self.project.figures_on(self.image_id)
        order = reading_order([(f.cx, f.cy, f.r) for f in figs])
        return [figs[i] for i in order]

    def selected_ids(self) -> list:
        return [it.fig_id for it in self.view.scene().selectedItems() if isinstance(it, CircleItem)]

    def selected_figs(self) -> list:
        ids = set(self.selected_ids())
        return [f for f in self.figures() if f.id in ids]

    def targets(self) -> list:
        """Выбранные фигуры, а если ничего не выбрано — все на изображении."""
        return self.selected_figs() or self.figures()

    # ------------------------------------------------------------------ обновление
    def on_enter(self) -> None:
        self.refresh_images()
        if not self._fitted_once and self.view.pixmap_item is not None:
            self._fitted_once = True
            QTimer.singleShot(0, self.view.fit)

    def _on_project_replaced(self) -> None:
        self.image_id = None
        self.refresh_images()

    def refresh_images(self) -> None:
        with QSignalBlocker(self.image_combo):
            self.image_combo.clear()
            for img in self.project.images:
                self.image_combo.addItem(img.name, img.id)
            ids = [i.id for i in self.project.images]
            if self.image_id not in ids:
                self.image_id = ids[0] if ids else None
                self._show_image()
            idx = ids.index(self.image_id) if self.image_id in ids else -1
            self.image_combo.setCurrentIndex(idx)
        self._refresh_suggestions()

    def select_image(self, image_id: str) -> None:
        ids = [i.id for i in self.project.images]
        if image_id in ids:
            self.image_combo.setCurrentIndex(ids.index(image_id))

    def _on_image_combo(self, idx: int) -> None:
        iid = self.image_combo.itemData(idx)
        if iid and iid != self.image_id:
            self.image_id = iid
            self._show_image()

    def _show_image(self) -> None:
        self.btn_add.setChecked(False)
        arr = self.state.store.get(self.image_id) if self.image_id else None
        self.view.set_pixmap(rgb_to_pixmap(arr) if arr is not None else None)
        self.rebuild_items()
        if arr is not None:
            QTimer.singleShot(0, self.view.fit)   # после того как вид получит свой размер

    def rebuild_items(self) -> None:
        sc = self.view.scene()
        sel = set(self.selected_ids())
        for it in self.items.values():
            sc.removeItem(it)
        self.items.clear()
        for f in self.figures():
            it = CircleItem(f.id, f.cx, f.cy, f.r)
            it.geometryEdited.connect(self._on_geometry_edited)
            sc.addItem(it)
            self.items[f.id] = it
            self._update_item_text(f)
            if f.id in sel:
                it.setSelected(True)
        self.refresh_table()

    def _update_item_text(self, f: PoleFigure) -> None:
        it = self.items.get(f.id)
        if it is None:
            return
        plane = format_plane(f.plane) or "{?}"
        parts = [plane]
        if f.sample or f.grain:
            parts.append(f"{f.sample or '?'} / з.{f.grain or '?'}")
        it.set_text("  ".join(parts), bool(f.sample and f.grain and f.plane))

    def refresh_table(self) -> None:
        self._sync = True
        try:
            figs = self.figures()
            sel = set(self.selected_ids())
            self.table.setRowCount(len(figs))
            for row, f in enumerate(figs):
                vals = [str(row + 1), f.sample, f.grain, format_plane(f.plane)]
                for c, v in enumerate(vals):
                    it = QTableWidgetItem(v)
                    it.setData(Qt.ItemDataRole.UserRole, f.id)
                    self.table.setItem(row, c, it)
            self.table.clearSelection()
            for row, f in enumerate(figs):
                if f.id in sel:
                    self.table.selectRow(row)
        finally:
            self._sync = False
        self._load_attr_editor()

    def _on_figures_changed(self) -> None:
        existing = set(self.items)
        current = {f.id for f in self.figures()}
        if existing != current:
            self.rebuild_items()
        else:
            for f in self.figures():
                it = self.items[f.id]
                if (it.pos().x(), it.pos().y(), it.r) != (f.cx, f.cy, f.r):
                    it.setPos(f.cx, f.cy)
                    it.set_radius(f.r)
                self._update_item_text(f)
            self.refresh_table()
        self._refresh_suggestions()

    def _refresh_suggestions(self) -> None:
        samples = sorted({f.sample for f in self.project.figures if f.sample}, key=natural_key)
        grains = sorted({f.grain for f in self.project.figures if f.grain}, key=natural_key)
        for cb, vals in ((self.ed_sample, samples), (self.ed_grain, grains)):
            text = cb.currentText()
            with QSignalBlocker(cb):
                cb.clear()
                cb.addItems(vals)
                cb.setEditText(text)

    # ------------------------------------------------------------------ выделение
    def _on_scene_selection(self) -> None:
        if self._sync:
            return
        self._sync = True
        try:
            sel = set(self.selected_ids())
            self.table.clearSelection()
            for row in range(self.table.rowCount()):
                it = self.table.item(row, 0)
                if it and it.data(Qt.ItemDataRole.UserRole) in sel:
                    self.table.selectRow(row)
        finally:
            self._sync = False
        self._load_attr_editor()

    def _on_table_selection(self) -> None:
        if self._sync:
            return
        self._sync = True
        try:
            ids = {self.table.item(i.row(), 0).data(Qt.ItemDataRole.UserRole)
                   for i in self.table.selectionModel().selectedRows()}
            for fid, it in self.items.items():
                it.setSelected(fid in ids)
        finally:
            self._sync = False
        self._load_attr_editor()

    def _center_on_selected(self) -> None:
        figs = self.selected_figs()
        if len(figs) == 1:
            f = figs[0]
            it = self.items[f.id]
            self.view.center_on_item_rect(it.sceneBoundingRect(), 1.6)

    def _load_attr_editor(self) -> None:
        figs = self.selected_figs()
        n = len(figs)
        self.sel_label.setText("Ничего не выбрано" if n == 0 else
                               ("Выбрана 1 фигура" if n == 1 else f"Выбрано фигур: {n}"))
        for cb, attr in ((self.ed_sample, "sample"), (self.ed_grain, "grain"), (self.ed_plane, "plane")):
            vals = {getattr(f, attr) for f in figs}
            with QSignalBlocker(cb):
                if len(vals) == 1:
                    cb.setEditText(vals.pop())
                    cb.lineEdit().setPlaceholderText("")
                elif len(vals) > 1:
                    cb.setEditText("")
                    cb.lineEdit().setPlaceholderText("(разные)")
                else:
                    cb.lineEdit().setPlaceholderText("")
        self.plane_preview.setText(format_plane(self.ed_plane.currentText()))

    # ------------------------------------------------------------------ атрибуты
    def apply_attributes(self) -> None:
        figs = self.selected_figs()
        if not figs:
            self.state.statusMessage.emit("Выберите одну или несколько окружностей")
            return
        sample = self.ed_sample.currentText().strip()
        grain = self.ed_grain.currentText().strip()
        plane = normalize_plane(self.ed_plane.currentText())
        for f in figs:
            if sample:
                f.sample = sample
            if grain:
                f.grain = grain
            if plane:
                f.plane = plane
                f.order = plane_sort_key(plane)[1] if plane in PLANE_PRESETS else 10
        self.state.notify_figures()
        if self.chk_advance.isChecked() and len(figs) == 1:
            all_f = self.figures()
            idx = [f.id for f in all_f].index(figs[0].id)
            if idx + 1 < len(all_f):
                nxt = all_f[idx + 1]
                self.view.scene().clearSelection()
                self.items[nxt.id].setSelected(True)   # редактор загружает значения следующей фигуры
                # подсказки для пустых полей следующей фигуры: тот же образец, следующий индекс
                planes = parse_plane_list(self.ed_plane_list.text())
                wrapped = bool(plane and plane in planes and planes.index(plane) == len(planes) - 1)
                if not nxt.sample and sample:
                    self.ed_sample.setEditText(sample)
                if not nxt.grain and grain:
                    self.ed_grain.setEditText(grain)
                    if wrapped:
                        self._grain_plus()
                if not nxt.plane and plane and plane in planes:
                    self.ed_plane.setEditText(planes[(planes.index(plane) + 1) % len(planes)])
                self.ed_plane.setFocus()
                self.ed_plane.lineEdit().selectAll()

    def _grain_plus(self) -> None:
        g = self.ed_grain.currentText().strip()
        m = re.match(r"^(.*?)(\d+)$", g)
        self.ed_grain.setEditText(f"{m.group(1)}{int(m.group(2)) + 1}" if m else "1")

    def bulk_planes(self) -> None:
        planes = parse_plane_list(self.ed_plane_list.text())
        figs = self.targets()
        if not planes or not figs:
            return
        for i, f in enumerate(figs):
            f.plane = planes[i % len(planes)]
            f.order = i % len(planes)
        self.state.notify_figures()
        self.state.statusMessage.emit(f"Индексы назначены {len(figs)} фигурам")

    def bulk_grains(self) -> None:
        figs = self.targets()
        if not figs:
            return
        n = self.spin_per_grain.value()
        g0 = self.spin_first_grain.value()
        sample = self.ed_sample.currentText().strip()
        for i, f in enumerate(figs):
            f.grain = str(g0 + i // n)
            f.order = i % n
            if sample:
                f.sample = sample
        self.state.notify_figures()
        self.state.statusMessage.emit(f"Зёрна {g0}…{g0 + (len(figs) - 1) // n} назначены")

    def auto_group(self) -> None:
        figs = self.targets()
        if not figs:
            return
        groups = cluster_by_proximity([(f.cx, f.cy, f.r) for f in figs])
        planes = parse_plane_list(self.ed_plane_list.text())
        sample = self.ed_sample.currentText().strip()
        g0 = self.spin_first_grain.value()
        for gi, grp in enumerate(groups):
            for k, idx in enumerate(grp):
                f = figs[idx]
                f.grain = str(g0 + gi)
                f.order = k
                if sample:
                    f.sample = sample
                if planes:
                    f.plane = planes[k % len(planes)]
        self.state.notify_figures()
        sizes = sorted({len(g) for g in groups})
        self.state.statusMessage.emit(
            f"Найдено групп (зёрен): {len(groups)}; фигур в группе: {', '.join(map(str, sizes))}")

    # ------------------------------------------------------------------ геометрия
    def _on_geometry_edited(self, _fid: str) -> None:
        # при групповом перемещении сдвигаются все выбранные элементы
        for f in self.figures():
            it = self.items.get(f.id)
            if it is None:
                continue
            nx, ny, nr = it.pos().x(), it.pos().y(), it.r
            if abs(nx - f.cx) > 1e-6 or abs(ny - f.cy) > 1e-6 or abs(nr - f.r) > 1e-6:
                f.cx, f.cy, f.r = nx, ny, nr
                f.auto = False
                self.state.notify_geometry(f.id)

    def nudge(self, dx: float, dy: float, dr: float) -> None:
        figs = self.selected_figs()
        for f in figs:
            f.cx += dx
            f.cy += dy
            f.r = max(3.0, f.r + dr)
            it = self.items[f.id]
            it.setPos(f.cx, f.cy)
            it.set_radius(f.r)
            self.state.notify_geometry(f.id)

    def _on_circle_drawn(self, cx: float, cy: float, r: float) -> None:
        if not self.image_id:
            return
        arr = self.state.store.get(self.image_id)
        if arr is not None and self.chk_snap.isChecked():
            ref = refine_circle(arr, cx, cy, r, self.project.detect.dark_threshold)
            if ref is not None:
                rcx, rcy, rr, rms, cov, _ = ref
                if abs(rr - r) < 0.15 * r and ((rcx - cx) ** 2 + (rcy - cy) ** 2) ** 0.5 < 0.15 * r and cov > 0.5:
                    cx, cy, r = rcx, rcy, rr
        f = PoleFigure(new_id(), self.image_id, cx, cy, r)
        self.project.figures.append(f)
        self.state.notify_figures()
        self.view.scene().clearSelection()
        if f.id in self.items:
            self.items[f.id].setSelected(True)
        self.ed_sample.setFocus()

    def refine_selected(self) -> None:
        arr = self.state.store.get(self.image_id) if self.image_id else None
        figs = self.selected_figs()
        if arr is None or not figs:
            self.state.statusMessage.emit("Выберите окружности для уточнения")
            return
        ok = 0
        for f in figs:
            ref = refine_circle(arr, f.cx, f.cy, f.r, self.project.detect.dark_threshold)
            if ref is not None and ref[4] > 0.5:
                f.cx, f.cy, f.r = ref[0], ref[1], ref[2]
                self.state.notify_geometry(f.id)
                ok += 1
        self.state.figuresChanged.emit()
        self.state.statusMessage.emit(f"Уточнено окружностей: {ok} из {len(figs)}")

    def delete_selected(self) -> None:
        ids = set(self.selected_ids())
        if not ids:
            return
        if len(ids) > 3 and QMessageBox.question(
                self, "Удаление", f"Удалить выбранные окружности ({len(ids)})?") != QMessageBox.StandardButton.Yes:
            return
        self.project.figures = [f for f in self.project.figures if f.id not in ids]
        for fid in ids:
            self.state.pipeline.invalidate(fid)
        self.state.notify_figures()

    def clear_image(self) -> None:
        figs = self.figures()
        if not figs:
            return
        if QMessageBox.question(self, "Удаление",
                                f"Удалить все окружности на этом изображении ({len(figs)})?"
                                ) != QMessageBox.StandardButton.Yes:
            return
        ids = {f.id for f in figs}
        self.project.figures = [f for f in self.project.figures if f.id not in ids]
        self.state.notify_figures()

    # ------------------------------------------------------------------ поиск
    def _detect_params(self) -> None:
        dlg = DetectDialog(self.project.detect, self)
        if dlg.exec():
            self.project.detect = dlg.params()
            self.state.pipeline.dark_threshold = self.project.detect.dark_threshold
            self.state.pipeline.invalidate()
            self.state.mark_dirty()
        dlg.deleteLater()

    def detect(self, image_ids: list, silent_if_none: bool = False) -> None:
        ids = [i for i in image_ids if i and self.state.image_loaded(i)]
        if not ids:
            if not silent_if_none:
                self.state.statusMessage.emit("Нет загруженных изображений для поиска")
            return
        self.btn_detect.setEnabled(False)
        self.btn_detect_all.setEnabled(False)
        params = self.project.detect
        summary = {"found": 0, "added": 0}

        def run_next(queue):
            if not queue:
                self.btn_detect.setEnabled(True)
                self.btn_detect_all.setEnabled(True)
                self.state.statusMessage.emit(
                    f"Поиск завершён: найдено окружностей {summary['found']}, добавлено новых {summary['added']}")
                if summary["found"] == 0:
                    QMessageBox.information(
                        self, "Поиск окружностей",
                        "Полюсные фигуры не найдены.\n\nПопробуйте изменить параметры поиска "
                        "(минимальный радиус, требование осей) или добавьте окружности вручную.")
                return
            iid = queue[0]
            arr = self.state.store.get(iid)
            name = self.project.image(iid).name if self.project.image(iid) else ""

            def done(circles, iid=iid):
                summary["found"] += len(circles)
                summary["added"] += self._add_detected(iid, circles)
                run_next(queue[1:])

            def err(msg, tb):
                self.btn_detect.setEnabled(True)
                self.btn_detect_all.setEnabled(True)
                self.state.runner.default_error("Поиск окружностей", msg, tb)

            self.state.runner.run(f"Поиск окружностей: {name}", detect_circles, arr, params,
                                  with_progress=True, on_done=done, on_error=err)

        run_next(ids)

    def _add_detected(self, image_id: str, circles: list) -> int:
        existing = self.project.figures_on(image_id)
        added = 0
        for c in circles:
            if any(((c.cx - f.cx) ** 2 + (c.cy - f.cy) ** 2) ** 0.5 < 0.3 * max(c.r, f.r) for f in existing):
                continue
            f = PoleFigure(new_id(), image_id, c.cx, c.cy, c.r, auto=True, score=c.score)
            self.project.figures.append(f)
            existing.append(f)
            added += 1
        if added:
            self.state.notify_figures()
        log.info("Изображение %s: найдено %d, добавлено %d", image_id, len(circles), added)
        return added
