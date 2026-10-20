"""Диалог параметров автоматического поиска окружностей."""
from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QSpinBox, QVBoxLayout)

from ..core.models import DetectParams
from .qt_utils import hint_label


class DetectDialog(QDialog):
    def __init__(self, params: DetectParams, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры поиска окружностей")
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.min_r = QSpinBox()
        self.min_r.setRange(5, 5000)
        self.min_r.setSuffix(" пикс.")
        self.min_r.setValue(params.min_radius)
        self.max_r = QSpinBox()
        self.max_r.setRange(0, 20000)
        self.max_r.setSpecialValueText("авто")
        self.max_r.setSuffix(" пикс.")
        self.max_r.setValue(params.max_radius)
        self.dark = QSpinBox()
        self.dark.setRange(30, 250)
        self.dark.setValue(params.dark_threshold)
        self.cov = QDoubleSpinBox()
        self.cov.setRange(0.3, 1.0)
        self.cov.setSingleStep(0.05)
        self.cov.setValue(params.ring_coverage)
        self.bg = QDoubleSpinBox()
        self.bg.setRange(0.0, 1.0)
        self.bg.setSingleStep(0.05)
        self.bg.setValue(params.min_inside_background)
        self.axes = QCheckBox("Требовать горизонтальную и вертикальную оси")
        self.axes.setChecked(params.require_axes)
        self.work = QSpinBox()
        self.work.setRange(400, 8000)
        self.work.setSingleStep(200)
        self.work.setSuffix(" пикс.")
        self.work.setValue(params.max_work_dim)
        form.addRow("Минимальный радиус:", self.min_r)
        form.addRow("Максимальный радиус:", self.max_r)
        form.addRow("Порог яркости линий:", self.dark)
        form.addRow("Замкнутость окружности (доля):", self.cov)
        form.addRow("Доля фона внутри (мин.):", self.bg)
        form.addRow("", self.axes)
        form.addRow("Размер рабочей копии:", self.work)
        lay.addLayout(form)
        lay.addWidget(hint_label(
            "Окружность принимается за полюсную фигуру, только если её линия почти замкнута, тонкая, "
            "внутри преобладает фон и (по умолчанию) присутствуют осевые линии. Это отсекает буквы, "
            "маркеры, залитые круги и прочие случайные округлые элементы. Если фигуры не находятся, "
            "уменьшите минимальный радиус или отключите требование осей."))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                              | QDialogButtonBox.StandardButton.RestoreDefaults)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText("По умолчанию")
        bb.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(self._defaults)
        lay.addWidget(bb)

    def _defaults(self) -> None:
        d = DetectParams()
        self.min_r.setValue(d.min_radius)
        self.max_r.setValue(d.max_radius)
        self.dark.setValue(d.dark_threshold)
        self.cov.setValue(d.ring_coverage)
        self.bg.setValue(d.min_inside_background)
        self.axes.setChecked(d.require_axes)
        self.work.setValue(d.max_work_dim)

    def params(self) -> DetectParams:
        return replace(DetectParams(), min_radius=self.min_r.value(), max_radius=self.max_r.value(),
                       dark_threshold=self.dark.value(), ring_coverage=self.cov.value(),
                       min_inside_background=self.bg.value(), require_axes=self.axes.isChecked(),
                       max_work_dim=self.work.value())
