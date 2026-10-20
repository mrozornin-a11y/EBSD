"""Ручная корректировка маски выделения цветных пикселей."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QLabel, QPushButton, QRadioButton, QSpinBox, QVBoxLayout)

from ..core.models import MaskEdits, PoleFigure, ReconParams
from ..core.segmentation import segment
from .image_view import ImageView
from .qt_utils import hint_label, rgb_to_pixmap
from .state import AppState


class MaskEditorDialog(QDialog):
    """Кисть «Добавить» включает пиксели исходного изображения в маску (только не-фоновые),
    «Исключить» — убирает, «Ластик» — снимает ручные правки. Цвета всегда берутся из оригинала."""

    def __init__(self, state: AppState, fig: PoleFigure, params: ReconParams, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Правка маски — {fig.plane_label or 'фигура'}")
        self.resize(1000, 800)
        self.state = state
        self.fig = fig
        self.params = params
        self.crop = state.pipeline.crop(fig)
        n = max(64, int(math.ceil(2 * fig.r)))
        if fig.mask_edits is not None:
            self.edits = fig.mask_edits.copy()
            n = self.edits.add.shape[0]
        else:
            self.edits = MaskEdits.empty(n)
        self.n = n
        self.undo_stack: list = []
        self._painting = False
        self.seg = None

        lay = QVBoxLayout(self)
        tools = QHBoxLayout()
        self.rb_add = QRadioButton("Добавить")
        self.rb_rem = QRadioButton("Исключить")
        self.rb_erase = QRadioButton("Ластик правок")
        self.rb_add.setChecked(True)
        g = QButtonGroup(self)
        for rb in (self.rb_add, self.rb_rem, self.rb_erase):
            g.addButton(rb)
            tools.addWidget(rb)
        tools.addSpacing(12)
        tools.addWidget(QLabel("Кисть:"))
        self.brush = QSpinBox()
        self.brush.setRange(1, 200)
        self.brush.setValue(max(2, int(fig.r / 25)))
        self.brush.setSuffix(" пикс.")
        tools.addWidget(self.brush)
        self.chk_overlay = QCheckBox("Показывать маску")
        self.chk_overlay.setChecked(True)
        tools.addWidget(self.chk_overlay)
        self.btn_undo = QPushButton("Отменить")
        self.btn_reset = QPushButton("Сбросить все правки")
        tools.addWidget(self.btn_undo)
        tools.addWidget(self.btn_reset)
        tools.addStretch(1)
        lay.addLayout(tools)
        self.view = ImageView()
        self.view.setDragMode(ImageView.DragMode.NoDrag)
        lay.addWidget(self.view, 1)
        self.info = QLabel("")
        lay.addWidget(self.info)
        lay.addWidget(hint_label(
            "Левая кнопка — рисовать кистью; колесо — масштаб; средняя кнопка или Пробел — перемещение. "
            "Подсветка: голубая заливка — выделенные пиксели, зелёный контур — добавлено вручную, "
            "красный — исключено вручную, пурпурный — удалено фильтром шума. Добавлять можно только "
            "пиксели, отличающиеся от белого фона: новые точки не создаются."))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.btn_undo.clicked.connect(self.undo)
        self.btn_reset.clicked.connect(self.reset)
        self.chk_overlay.toggled.connect(lambda _: self.redraw())
        self.view.viewport().installEventFilter(self)
        self.resegment()
        self.view.fit()

    # ------------------------------------------------------------------
    def result_edits(self) -> MaskEdits:
        return self.edits

    def resegment(self) -> None:
        self.seg = segment(self.crop, self.params, self.edits, self.state.pipeline.dark_threshold)
        self.info.setText(self.seg.stats_text().replace("\n", "  •  "))
        self.redraw()

    def _edit_on_crop(self, arr: np.ndarray) -> np.ndarray:
        size = self.crop.size
        coords = np.arange(size) + 0.5
        col = np.floor(((coords - self.crop.cx) / self.crop.r + 1) / 2 * self.n).astype(int)
        row = np.floor(((coords - self.crop.cy) / self.crop.r + 1) / 2 * self.n).astype(int)
        valid = ((row >= 0) & (row < self.n))[:, None] & ((col >= 0) & (col < self.n))[None, :]
        return arr[np.ix_(np.clip(row, 0, self.n - 1), np.clip(col, 0, self.n - 1))] & valid

    def redraw(self) -> None:
        img = self.crop.rgb.astype(np.float32)
        if self.chk_overlay.isChecked() and self.seg is not None:
            m = self.seg.mask
            img[~m] = img[~m] * 0.55 + 255 * 0.45            # фон и линии приглушены
            img[m] = img[m] * 0.75 + np.array([0, 200, 255]) * 0.25
            for arr, color in ((self._edit_on_crop(self.edits.add), (0, 200, 0)),
                               (self._edit_on_crop(self.edits.remove), (230, 0, 0)),
                               (self.seg.removed_mask, (255, 0, 255))):
                if arr.any():
                    a8 = arr.astype(np.uint8)
                    edge = cv2.dilate(a8, np.ones((3, 3), np.uint8)).astype(bool) & ~arr
                    img[edge] = color
            # граница рабочей области
            h, w = m.shape
            yy, xx = np.mgrid[0:h, 0:w]
            d = np.hypot(xx + 0.5 - self.crop.cx, yy + 0.5 - self.crop.cy)
            ring = np.abs(d - self.seg.inner_r) < 0.5
            img[ring] = (255, 160, 0)
        self.view.set_pixmap(rgb_to_pixmap(np.clip(img, 0, 255).astype(np.uint8)), keep_view=True, smooth=False)

    # ------------------------------------------------------------------ рисование
    def begin_stroke(self) -> None:
        self.undo_stack.append(self.edits.copy())
        self.undo_stack = self.undo_stack[-50:]

    def _paint_at(self, scene_pos: QPointF) -> None:
        x, y = scene_pos.x(), scene_pos.y()
        scale = self.n / (2 * self.crop.r)
        gx = ((x - self.crop.cx) / self.crop.r + 1) / 2 * self.n
        gy = ((y - self.crop.cy) / self.crop.r + 1) / 2 * self.n
        rad = max(0.5, self.brush.value() / 2 * scale)
        yy, xx = np.ogrid[0:self.n, 0:self.n]
        disk = (xx + 0.5 - gx) ** 2 + (yy + 0.5 - gy) ** 2 <= rad * rad
        if not disk.any():
            ix, iy = int(gx), int(gy)
            if 0 <= ix < self.n and 0 <= iy < self.n:
                disk = np.zeros((self.n, self.n), bool)
                disk[iy, ix] = True
        if self.rb_add.isChecked():
            self.edits.add |= disk
            self.edits.remove &= ~disk
        elif self.rb_rem.isChecked():
            self.edits.remove |= disk
            self.edits.add &= ~disk
        else:
            self.edits.add &= ~disk
            self.edits.remove &= ~disk

    def eventFilter(self, obj, ev):
        if obj is self.view.viewport():
            t = ev.type()
            if t == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton and not self.view._space:
                self.begin_stroke()
                self._painting = True
                self._paint_at(self.view.mapToScene(ev.position().toPoint()))
                self.redraw()
                return True
            if t == QEvent.Type.MouseMove and self._painting:
                self._paint_at(self.view.mapToScene(ev.position().toPoint()))
                self.redraw()
                return True
            if t == QEvent.Type.MouseButtonRelease and self._painting and ev.button() == Qt.MouseButton.LeftButton:
                self._painting = False
                self.resegment()
                return True
        return super().eventFilter(obj, ev)

    def undo(self) -> None:
        if self.undo_stack:
            self.edits = self.undo_stack.pop()
            self.resegment()

    def reset(self) -> None:
        self.undo_stack.append(self.edits.copy())
        self.edits = MaskEdits.empty(self.n)
        self.resegment()
