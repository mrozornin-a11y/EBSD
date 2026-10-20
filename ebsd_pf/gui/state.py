"""Общее состояние приложения: проект, изображения, конвейер обработки, признак изменений."""
from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QObject, Signal

from ..core.models import Project
from ..core.pipeline import ImageStore, Pipeline
from .workers import TaskRunner


class AppState(QObject):
    dirtyChanged = Signal(bool)
    projectReplaced = Signal()
    imagesChanged = Signal()
    figuresChanged = Signal()          # состав фигур или их атрибуты
    geometryChanged = Signal(str)      # изменена геометрия фигуры (id)
    paramsChanged = Signal()           # параметры реконструкции/компоновки
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = Project()
        self.store = ImageStore()
        self.pipeline = Pipeline(self.store)
        self.runner = TaskRunner(self)
        self.path: Optional[str] = None
        self.dirty = False

    # ----------------------------------------------------------------
    def mark_dirty(self, value: bool = True) -> None:
        if self.dirty != value:
            self.dirty = value
            self.dirtyChanged.emit(value)

    def set_project(self, project: Project, path: Optional[str]) -> None:
        self.store.clear()
        self.pipeline.invalidate()
        self.project = project
        self.pipeline.dark_threshold = project.detect.dark_threshold
        self.path = path
        self.dirty = False
        self.dirtyChanged.emit(False)
        self.projectReplaced.emit()

    def project_name(self) -> str:
        return os.path.basename(self.path) if self.path else "Новый проект"

    def notify_figures(self) -> None:
        self.mark_dirty()
        self.figuresChanged.emit()

    def notify_geometry(self, fig_id: str) -> None:
        self.pipeline.invalidate(fig_id)
        self.mark_dirty()
        self.geometryChanged.emit(fig_id)

    def notify_params(self) -> None:
        self.mark_dirty()
        self.paramsChanged.emit()

    def image_loaded(self, image_id: str) -> bool:
        return image_id in self.store
