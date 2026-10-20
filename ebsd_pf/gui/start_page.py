"""Стартовый экран: загрузка изображений и проектов."""
from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QVBoxLayout, QWidget)

from .. import APP_NAME_RU, __version__
from .qt_utils import hint_label, rgb_to_pixmap
from .state import AppState


class StartPage(QWidget):
    openImagesRequested = Signal()
    openProjectRequested = Signal()
    openRecentRequested = Signal(str)
    newProjectRequested = Signal()
    removeImageRequested = Signal(str)
    nextRequested = Signal()

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 20, 30, 20)
        title = QLabel(APP_NAME_RU)
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        title.setFont(f)
        root.addWidget(title)
        root.addWidget(hint_label(
            f"Версия {__version__}. Извлечение полюсных фигур из растровых изображений (JPG, PNG, TIF, TIFF), "
            "очистка и улучшение читаемости с сохранением исходного положения и цвета точек. "
            "Новые экспериментальные точки не создаются."))

        btns = QHBoxLayout()
        self.btn_images = QPushButton("Открыть изображения…")
        self.btn_project = QPushButton("Открыть проект…")
        self.btn_new = QPushButton("Новый проект")
        for b in (self.btn_images, self.btn_project, self.btn_new):
            b.setMinimumHeight(40)
            btns.addWidget(b)
        btns.addStretch(1)
        root.addLayout(btns)
        self.btn_images.clicked.connect(self.openImagesRequested)
        self.btn_project.clicked.connect(self.openProjectRequested)
        self.btn_new.clicked.connect(self.newProjectRequested)

        mid = QHBoxLayout()
        gb_img = QGroupBox("Изображения проекта")
        li = QVBoxLayout(gb_img)
        self.images = QListWidget()
        self.images.setViewMode(QListWidget.ViewMode.IconMode)
        self.images.setIconSize(QSize(160, 120))
        self.images.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.images.setSpacing(8)
        self.images.setWordWrap(True)
        self.images.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        li.addWidget(self.images)
        row = QHBoxLayout()
        self.btn_add = QPushButton("Добавить изображения…")
        self.btn_remove = QPushButton("Удалить выбранные")
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_remove)
        row.addStretch(1)
        li.addLayout(row)
        li.addWidget(hint_label("Изображения можно также перетащить в окно программы. "
                                "В один проект можно загрузить несколько изображений."))
        self.btn_add.clicked.connect(self.openImagesRequested)
        self.btn_remove.clicked.connect(self._remove)
        mid.addWidget(gb_img, 3)

        gb_rec = QGroupBox("Недавние проекты")
        lr = QVBoxLayout(gb_rec)
        self.recent = QListWidget()
        self.recent.itemDoubleClicked.connect(lambda it: self.openRecentRequested.emit(it.data(Qt.ItemDataRole.UserRole)))
        lr.addWidget(self.recent)
        lr.addWidget(hint_label("Двойной щелчок — открыть проект."))
        mid.addWidget(gb_rec, 1)
        root.addLayout(mid, 1)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_next = QPushButton("Далее: разметка окружностей →")
        self.btn_next.setMinimumHeight(36)
        self.btn_next.clicked.connect(self.nextRequested)
        bottom.addWidget(self.btn_next)
        root.addLayout(bottom)

        state.imagesChanged.connect(self.refresh)
        state.projectReplaced.connect(self.refresh)
        state.figuresChanged.connect(self.refresh)

    def set_recent(self, paths: list) -> None:
        self.recent.clear()
        for p in paths:
            it = QListWidgetItem(os.path.basename(p))
            it.setToolTip(p)
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.recent.addItem(it)

    def refresh(self) -> None:
        self.images.clear()
        pr = self.state.project
        for img in pr.images:
            n = len(pr.figures_on(img.id))
            it = QListWidgetItem(f"{img.name}\n{img.width}×{img.height}, фигур: {n}")
            it.setData(Qt.ItemDataRole.UserRole, img.id)
            it.setToolTip(img.path)
            arr = self.state.store.get(img.id)
            if arr is not None:
                pm = rgb_to_pixmap(arr[::max(1, arr.shape[0] // 240), ::max(1, arr.shape[1] // 320)])
                it.setIcon(QIcon(pm))
            else:
                it.setIcon(QIcon(QPixmap(160, 120)))
                it.setText(it.text() + "\n(не загружено)")
            self.images.addItem(it)
        self.btn_next.setEnabled(bool(pr.images))

    def _remove(self) -> None:
        for it in self.images.selectedItems():
            self.removeImageRequested.emit(it.data(Qt.ItemDataRole.UserRole))
