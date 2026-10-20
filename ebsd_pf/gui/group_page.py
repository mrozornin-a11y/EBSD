"""Экран группировки: образцы → зерна → полюсные фигуры; порядок, перенос, индексы, предпросмотр."""
from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
                               QLabel, QMessageBox, QPushButton, QSpinBox, QSplitter, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core.exporter import grain_sheet, grain_title, sample_sheet, sample_title
from ..core.models import (NO_GRAIN, NO_SAMPLE, format_plane, natural_key,
                           normalize_plane, plane_sort_key)
from .image_view import ImageView
from .preview import render_async
from .qt_utils import LabeledSlider, hint_label
from .state import AppState

ROLE_KIND = Qt.ItemDataRole.UserRole
ROLE_KEY = Qt.ItemDataRole.UserRole + 1
ROLE_SAMPLE = Qt.ItemDataRole.UserRole + 2
ROLE_GRAIN = Qt.ItemDataRole.UserRole + 3


class GroupTree(QTreeWidget):
    figuresDropped = Signal(list, str, str, int)   # ids, sample, grain ('' = новое), позиция (-1 = в конец)
    grainsDropped = Signal(list, str)              # [(sample, grain)], целевой образец

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["Образец / зерно / фигура", "Индекс"])
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                             | QAbstractItemView.EditTrigger.EditKeyPressed)

    def dropEvent(self, event) -> None:
        target = self.itemAt(event.position().toPoint())
        dip = self.dropIndicatorPosition()
        dragged = self.selectedItems()
        figs = [it.data(0, ROLE_KEY) for it in dragged if it.data(0, ROLE_KIND) == "fig"]
        grains = [(it.data(0, ROLE_SAMPLE), it.data(0, ROLE_KEY)) for it in dragged
                  if it.data(0, ROLE_KIND) == "grain"]
        event.setDropAction(Qt.DropAction.CopyAction)   # исходные элементы не удаляются виджетом
        event.accept()
        if target is None:
            return
        kind = target.data(0, ROLE_KIND)
        if figs:
            if kind == "fig":
                parent = target.parent()
                idx = parent.indexOfChild(target)
                if dip == QAbstractItemView.DropIndicatorPosition.BelowItem:
                    idx += 1
                args = (figs, parent.data(0, ROLE_SAMPLE), parent.data(0, ROLE_KEY), idx)
            elif kind == "grain":
                args = (figs, target.data(0, ROLE_SAMPLE), target.data(0, ROLE_KEY), -1)
            else:
                args = (figs, target.data(0, ROLE_KEY), "", -1)
            QTimer.singleShot(0, lambda: self.figuresDropped.emit(*args))
        elif grains:
            if kind == "sample":
                s = target.data(0, ROLE_KEY)
            else:
                s = target.data(0, ROLE_SAMPLE)
            QTimer.singleShot(0, lambda: self.grainsDropped.emit(grains, s))


class MoveDialog(QDialog):
    def __init__(self, samples, grains, sample="", grain="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Переместить фигуры")
        form = QFormLayout(self)
        self.sample = QComboBox()
        self.sample.setEditable(True)
        self.sample.addItems(samples)
        self.sample.setEditText(sample)
        self.grain = QComboBox()
        self.grain.setEditable(True)
        self.grain.addItems(grains)
        self.grain.setEditText(grain)
        form.addRow("Образец:", self.sample)
        form.addRow("Зерно:", self.grain)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)


class GroupPage(QWidget):
    nextRequested = Signal()
    backRequested = Signal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        self._preview_req = 0
        self._preview_target = None
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        self.tree = GroupTree()
        ll.addWidget(self.tree, 1)
        r1 = QHBoxLayout()
        self.btn_up = QPushButton("↑ Выше")
        self.btn_down = QPushButton("↓ Ниже")
        self.btn_sort = QPushButton("Упорядочить по индексам")
        for b in (self.btn_up, self.btn_down, self.btn_sort):
            r1.addWidget(b)
        ll.addLayout(r1)
        r2 = QHBoxLayout()
        self.btn_move = QPushButton("Переместить…")
        self.btn_rename = QPushButton("Переименовать…")
        self.btn_delete = QPushButton("Удалить фигуры")
        for b in (self.btn_move, self.btn_rename, self.btn_delete):
            r2.addWidget(b)
        ll.addLayout(r2)
        ll.addWidget(hint_label(
            "Перетаскивайте фигуры мышью между зернами и образцами (на зерно — в конец, между фигурами — "
            "в указанное место, на образец — в новое зерно). Зерно можно перетащить на другой образец. "
            "Индекс плоскости редактируется двойным щелчком во втором столбце."))
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 0, 0)
        gb = QGroupBox("Компоновка")
        form = QFormLayout(gb)
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(0, 12)
        self.spin_cols.setSpecialValueText("авто")
        self.sl_spacing = LabeledSlider(0, 40, 0.5, 5, 1, " мм")
        self.sl_diam = LabeledSlider(10, 150, 1, 40, 0, " мм")
        self.chk_titles = QCheckBox("Подписи образца и зерна на рисунке зерна")
        self.cmb_preview = QComboBox()
        self.cmb_preview.addItems(["Выбранное зерно", "Все зерна образца"])
        form.addRow("Столбцов в сетке:", self.spin_cols)
        form.addRow("Расстояние между окружностями:", self.sl_spacing)
        form.addRow("Диаметр окружностей:", self.sl_diam)
        form.addRow(self.chk_titles)
        form.addRow("Предпросмотр:", self.cmb_preview)
        rl.addWidget(gb)
        self.preview = ImageView()
        self.preview.setBackgroundBrush(QBrush(QColor(200, 200, 200)))
        rl.addWidget(self.preview, 1)
        self.preview_info = QLabel("")
        rl.addWidget(self.preview_info)
        nav = QHBoxLayout()
        self.btn_back = QPushButton("← Разметка")
        self.btn_next = QPushButton("Далее: реконструкция →")
        self.btn_next.setMinimumHeight(34)
        nav.addWidget(self.btn_back)
        nav.addStretch(1)
        nav.addWidget(self.btn_next)
        rl.addLayout(nav)
        split.addWidget(right)
        split.setSizes([420, 800])
        root.addWidget(split, 1)

        self.tree.itemSelectionChanged.connect(self._schedule_preview)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.figuresDropped.connect(self.move_figures)
        self.tree.grainsDropped.connect(self.move_grains)
        self.btn_up.clicked.connect(lambda: self.shift(-1))
        self.btn_down.clicked.connect(lambda: self.shift(1))
        self.btn_sort.clicked.connect(self.sort_by_plane)
        self.btn_move.clicked.connect(self.move_dialog)
        self.btn_rename.clicked.connect(self.rename)
        self.btn_delete.clicked.connect(self.delete_figs)
        self.btn_back.clicked.connect(self.backRequested)
        self.btn_next.clicked.connect(self.nextRequested)
        self.spin_cols.valueChanged.connect(self._layout_changed)
        self.sl_spacing.valueChanged.connect(self._layout_changed)
        self.sl_diam.valueChanged.connect(self._layout_changed)
        self.chk_titles.toggled.connect(self._layout_changed)
        self.cmb_preview.currentIndexChanged.connect(self._schedule_preview)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.update_preview)

        state.figuresChanged.connect(self._on_external_change)
        state.projectReplaced.connect(self._on_external_change)
        state.paramsChanged.connect(self._load_layout)

    @property
    def project(self):
        return self.state.project

    # ------------------------------------------------------------------ дерево
    def on_enter(self) -> None:
        self._load_layout()
        self.rebuild()

    def _on_external_change(self) -> None:
        if self.isVisible():
            self.rebuild()

    def _load_layout(self) -> None:
        ls = self.project.layout
        for w in (self.spin_cols, self.sl_spacing, self.sl_diam, self.chk_titles):
            w.blockSignals(True)
        self.spin_cols.setValue(ls.columns)
        self.sl_spacing.setValue(ls.spacing_mm)
        self.sl_diam.setValue(ls.diameter_mm)
        self.chk_titles.setChecked(ls.show_group_title)
        for w in (self.spin_cols, self.sl_spacing, self.sl_diam, self.chk_titles):
            w.blockSignals(False)
        self._schedule_preview()

    def _layout_changed(self, *_):
        ls = self.project.layout
        ls.columns = self.spin_cols.value()
        ls.spacing_mm = self.sl_spacing.value()
        ls.diameter_mm = self.sl_diam.value()
        ls.show_group_title = self.chk_titles.isChecked()
        self.state.mark_dirty()
        self._schedule_preview()

    def _selection_key(self):
        it = self.tree.currentItem()
        if it is None:
            return None
        return (it.data(0, ROLE_KIND), it.data(0, ROLE_KEY), it.data(0, ROLE_SAMPLE))

    def rebuild(self) -> None:
        sel_figs = {it.data(0, ROLE_KEY) for it in self.tree.selectedItems() if it.data(0, ROLE_KIND) == "fig"}
        cur = self._selection_key()
        img_names = {i.id: i.name for i in self.project.images}
        bold = QFont()
        bold.setBold(True)
        with QSignalBlocker(self.tree):
            self.tree.clear()
            current_item = None
            for sample, grains in self.project.grouped():
                si = QTreeWidgetItem([sample_title(sample), ""])
                si.setData(0, ROLE_KIND, "sample")
                si.setData(0, ROLE_KEY, sample)
                si.setFont(0, bold)
                si.setFlags((si.flags() | Qt.ItemFlag.ItemIsDropEnabled) & ~Qt.ItemFlag.ItemIsDragEnabled)
                self.tree.addTopLevelItem(si)
                if cur == ("sample", sample, None):
                    current_item = si
                for grain, figs in grains:
                    n = len(figs)
                    gi = QTreeWidgetItem([f"{grain_title(grain)}   ({n} фиг.)", ""])
                    gi.setData(0, ROLE_KIND, "grain")
                    gi.setData(0, ROLE_KEY, grain)
                    gi.setData(0, ROLE_SAMPLE, sample)
                    gi.setFlags(gi.flags() | Qt.ItemFlag.ItemIsDropEnabled | Qt.ItemFlag.ItemIsDragEnabled)
                    si.addChild(gi)
                    if cur and cur[0] == "grain" and cur[1] == grain and cur[2] == sample:
                        current_item = gi
                    for k, f in enumerate(figs):
                        planes = [normalize_plane(x.plane) for x in figs]
                        dup = planes.count(normalize_plane(f.plane)) > 1 and f.plane
                        fi = QTreeWidgetItem([f"{k + 1}. {format_plane(f.plane) or '{?}'}   "
                                              f"[{img_names.get(f.image_id, '?')}]", f.plane])
                        fi.setData(0, ROLE_KIND, "fig")
                        fi.setData(0, ROLE_KEY, f.id)
                        fi.setData(0, ROLE_SAMPLE, sample)
                        fi.setFlags(fi.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled)
                        if not f.plane or dup:
                            fi.setForeground(0, QBrush(QColor(200, 0, 0)))
                            fi.setToolTip(0, "Индекс не задан" if not f.plane else "Индекс повторяется в зерне")
                        gi.addChild(fi)
                        if f.id in sel_figs:
                            fi.setSelected(True)
                        if cur and cur[0] == "fig" and cur[1] == f.id:
                            current_item = fi
            self.tree.expandAll()
            self.tree.resizeColumnToContents(0)
            if current_item is not None:
                self.tree.setCurrentItem(current_item)
            elif self.tree.topLevelItemCount():
                first = self.tree.topLevelItem(0)
                self.tree.setCurrentItem(first.child(0) if first.childCount() else first)
        self._schedule_preview()

    def _on_item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        if item.data(0, ROLE_KIND) != "fig" or col != 1:
            return
        f = self.project.figure(item.data(0, ROLE_KEY))
        if f is None:
            return
        f.plane = normalize_plane(item.text(1))
        self.state.notify_figures()

    def selected_fig_ids(self) -> list:
        ids = []
        for it in self.tree.selectedItems():
            kind = it.data(0, ROLE_KIND)
            if kind == "fig":
                ids.append(it.data(0, ROLE_KEY))
            elif kind == "grain":
                ids += [f.id for f in self.project.grain_figures(it.data(0, ROLE_SAMPLE), it.data(0, ROLE_KEY))]
        return list(dict.fromkeys(ids))

    # ------------------------------------------------------------------ операции
    @staticmethod
    def _raw(key: str, none_key: str) -> str:
        return "" if key == none_key else key

    def _next_grain(self, sample: str) -> str:
        nums = [int(g) for g in self.project.grains(sample) if g.isdigit()]
        return str(max(nums) + 1 if nums else 1)

    def move_figures(self, ids: list, sample: str, grain: str, index: int) -> None:
        if not ids:
            return
        if not grain:
            grain = self._next_grain(sample)
        moving = [self.project.figure(i) for i in ids if self.project.figure(i)]
        target = [f for f in self.project.grain_figures(sample, grain) if f.id not in ids]
        index = len(target) if index < 0 else min(index, len(target))
        new_order = target[:index] + moving + target[index:]
        for f in moving:
            f.sample = self._raw(sample, NO_SAMPLE)
            f.grain = self._raw(grain, NO_GRAIN)
        for k, f in enumerate(new_order):
            f.order = k
        self.project.renumber_orders()
        self.state.notify_figures()
        self.rebuild()

    def move_grains(self, grains: list, sample: str) -> None:
        for s, g in grains:
            if s == sample:
                continue
            new_g = g
            if g in self.project.grains(sample):
                new_g = self._next_grain(sample)
            for f in self.project.grain_figures(s, g):
                f.sample = self._raw(sample, NO_SAMPLE)
                f.grain = self._raw(new_g, NO_GRAIN)
        self.state.notify_figures()
        self.rebuild()

    def shift(self, delta: int) -> None:
        it = self.tree.currentItem()
        if it is None or it.data(0, ROLE_KIND) != "fig":
            return
        gi = it.parent()
        figs = self.project.grain_figures(gi.data(0, ROLE_SAMPLE), gi.data(0, ROLE_KEY))
        ids = [f.id for f in figs]
        i = ids.index(it.data(0, ROLE_KEY))
        j = i + delta
        if not 0 <= j < len(figs):
            return
        figs[i], figs[j] = figs[j], figs[i]
        for k, f in enumerate(figs):
            f.order = k
        self.state.notify_figures()
        self.rebuild()

    def sort_by_plane(self) -> None:
        it = self.tree.currentItem()
        groups = []
        if it is not None and it.data(0, ROLE_KIND) in ("fig", "grain"):
            gi = it if it.data(0, ROLE_KIND) == "grain" else it.parent()
            groups = [(gi.data(0, ROLE_SAMPLE), gi.data(0, ROLE_KEY))]
        else:
            groups = [(s, g) for s in self.project.samples() for g in self.project.grains(s)]
        for s, g in groups:
            figs = sorted(self.project.grain_figures(s, g), key=lambda f: plane_sort_key(f.plane))
            for k, f in enumerate(figs):
                f.order = k
        self.state.notify_figures()
        self.rebuild()

    def move_dialog(self) -> None:
        ids = self.selected_fig_ids()
        if not ids:
            QMessageBox.information(self, "Перемещение", "Выберите фигуры или зерно.")
            return
        f0 = self.project.figure(ids[0])
        samples = [s for s in self.project.samples() if s != NO_SAMPLE]
        grains = sorted({f.grain for f in self.project.figures if f.grain}, key=natural_key)
        dlg = MoveDialog(samples, grains, f0.sample, f0.grain, self)
        accepted = dlg.exec()
        s = dlg.sample.currentText().strip() or NO_SAMPLE
        g = dlg.grain.currentText().strip()
        dlg.deleteLater()
        if accepted:
            self.move_figures(ids, s, g or NO_GRAIN, -1)

    def rename(self) -> None:
        it = self.tree.currentItem()
        if it is None:
            return
        kind = it.data(0, ROLE_KIND)
        if kind == "sample":
            old = it.data(0, ROLE_KEY)
            new, ok = QInputDialog.getText(self, "Переименовать образец", "Название образца:",
                                           text=self._raw(old, NO_SAMPLE))
            if ok:
                for f in self.project.figures:
                    if f.sample_key == old:
                        f.sample = new.strip()
        elif kind == "grain":
            s, old = it.data(0, ROLE_SAMPLE), it.data(0, ROLE_KEY)
            new, ok = QInputDialog.getText(self, "Переименовать зерно", "Номер зерна:",
                                           text=self._raw(old, NO_GRAIN))
            if ok:
                new = new.strip()
                clash = new and new != old and new in self.project.grains(s)
                if clash and QMessageBox.question(
                        self, "Объединение", f"Зерно {new} уже есть в этом образце. Объединить?"
                ) != QMessageBox.StandardButton.Yes:
                    return
                for f in self.project.grain_figures(s, old):
                    f.grain = new
        elif kind == "fig":
            self.tree.editItem(it, 1)
            return
        else:
            return
        self.state.notify_figures()
        self.rebuild()

    def delete_figs(self) -> None:
        ids = self.selected_fig_ids()
        if not ids:
            return
        if QMessageBox.question(self, "Удаление", f"Удалить фигуры из проекта ({len(ids)})?"
                                ) != QMessageBox.StandardButton.Yes:
            return
        s = set(ids)
        self.project.figures = [f for f in self.project.figures if f.id not in s]
        self.state.notify_figures()
        self.rebuild()

    # ------------------------------------------------------------------ предпросмотр
    def _schedule_preview(self, *_):
        self._timer.start()

    def _current_target(self):
        it = self.tree.currentItem()
        if it is None:
            return None
        kind = it.data(0, ROLE_KIND)
        if kind == "sample":
            return ("sample", it.data(0, ROLE_KEY), None)
        if kind == "grain":
            s, g = it.data(0, ROLE_SAMPLE), it.data(0, ROLE_KEY)
        else:
            s, g = it.parent().data(0, ROLE_SAMPLE), it.parent().data(0, ROLE_KEY)
        if self.cmb_preview.currentIndex() == 1:
            return ("sample", s, None)
        return ("grain", s, g)

    def update_preview(self) -> None:
        if not self.isVisible():
            return
        tgt = self._current_target()
        if tgt is None:
            self.preview.set_pixmap(None)
            self.preview_info.setText("")
            return
        kind, s, g = tgt
        sheet = sample_sheet(self.project, s) if kind == "sample" else grain_sheet(self.project, s, g)
        if not sheet.figures:
            return
        missing = [pf for pf in sheet.figures if not self.state.image_loaded(self.project.figure(pf.fig_id).image_id)]
        self._preview_req += 1
        req = self._preview_req
        keep = self._preview_target == tgt
        self._preview_target = tgt

        def done(img, req=req, keep=keep):
            if req != self._preview_req:
                return
            self.preview.set_pixmap(QPixmap.fromImage(img), keep_view=True)
            if not keep:
                QTimer.singleShot(0, self.preview.fit)
        self.preview_info.setText(
            f"Размер листа: {sheet.width:.1f} × {sheet.height:.1f} мм; фигур: {len(sheet.figures)}"
            + (f"; не загружены изображения для {len(missing)} фигур" if missing else ""))
        render_async(self.state, sheet, 8.0, done)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._schedule_preview()
