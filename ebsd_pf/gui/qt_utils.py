"""Вспомогательные функции для Qt-интерфейса."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox, QSlider, QWidget)
from PySide6.QtCore import Signal

from ..core.render import rgb_to_qimage


def rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(rgb_to_qimage(rgb))


def error_box(parent, title: str, text: str, details: str = "") -> None:
    box = QMessageBox(QMessageBox.Icon.Critical, title, text, QMessageBox.StandardButton.Ok, parent)
    if details:
        box.setDetailedText(details)
    box.exec()


def info_box(parent, title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)


def ask_yes_no(parent, title: str, text: str) -> bool:
    return QMessageBox.question(parent, title, text,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                ) == QMessageBox.StandardButton.Yes


class LabeledSlider(QWidget):
    """Ползунок с числовым полем (вещественные значения)."""
    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, step: float, value: float,
                 decimals: int = 1, suffix: str = "", parent=None):
        super().__init__(parent)
        self._min, self._max, self._step = minimum, maximum, step
        self._block = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, int(round((maximum - minimum) / step)))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(84)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.setValue(value)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float) -> None:
        self._block = True
        self.spin.setValue(v)
        self.slider.setValue(int(round((v - self._min) / self._step)))
        self._block = False

    def _on_slider(self, i: int) -> None:
        if self._block:
            return
        v = self._min + i * self._step
        self._block = True
        self.spin.setValue(v)
        self._block = False
        self.valueChanged.emit(float(self.spin.value()))

    def _on_spin(self, v: float) -> None:
        if self._block:
            return
        self._block = True
        self.slider.setValue(int(round((v - self._min) / self._step)))
        self._block = False
        self.valueChanged.emit(float(v))


def hint_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: #555; font-size: 11px;")
    return lab
