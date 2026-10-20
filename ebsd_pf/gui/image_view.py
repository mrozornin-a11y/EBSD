"""Просмотр изображения с масштабированием колесом мыши и панорамированием."""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem,
                               QGraphicsScene, QGraphicsView)


class ImageView(QGraphicsView):
    """Колесо — масштаб (относительно курсора); средняя кнопка или пробел+ЛКМ — перемещение.

    Режим добавления окружности: первый щелчок — центр, второй — точка на окружности."""
    zoomChanged = Signal(float)
    viewChanged = Signal()
    circleDrawn = Signal(float, float, float)     # cx, cy, r (координаты сцены)
    addModeCancelled = Signal()

    MIN_ZOOM, MAX_ZOOM = 0.02, 64.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setBackgroundBrush(QBrush(QColor(110, 110, 110)))
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._panning = False
        self._pan_start = QPointF()
        self._space = False
        self._add_mode = False
        self._add_center: Optional[QPointF] = None
        self._preview_circle: Optional[QGraphicsEllipseItem] = None
        self._preview_line: Optional[QGraphicsLineItem] = None
        self.horizontalScrollBar().valueChanged.connect(lambda _: self.viewChanged.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _: self.viewChanged.emit())

    # ---------------------------------------------------------------- изображение
    def set_pixmap(self, pixmap: Optional[QPixmap], keep_view: bool = False, smooth: bool = True,
                   offset: tuple = (0.0, 0.0), scale: float = 1.0, scene_rect: Optional[QRectF] = None) -> None:
        if pixmap is None:
            if self.pixmap_item is not None:
                self.scene().removeItem(self.pixmap_item)
                self.pixmap_item = None
            return
        if self.pixmap_item is None:
            self.pixmap_item = QGraphicsPixmapItem()
            self.pixmap_item.setZValue(-10)
            self.scene().addItem(self.pixmap_item)
        self.pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation if smooth else Qt.TransformationMode.FastTransformation)
        self.pixmap_item.setPixmap(pixmap)
        self.pixmap_item.setPos(offset[0], offset[1])
        self.pixmap_item.setScale(scale)
        rect = scene_rect if scene_rect is not None else self.pixmap_item.sceneBoundingRect()
        self.scene().setSceneRect(QRectF(rect).adjusted(-50, -50, 50, 50))
        if not keep_view:
            self.fit()

    def fit(self) -> None:
        if self.pixmap_item is not None:
            self.fitInView(self.pixmap_item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        elif not self.scene().itemsBoundingRect().isEmpty():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.zoom())

    def zoom(self) -> float:
        return self.transform().m11()

    def set_zoom(self, z: float) -> None:
        z = max(self.MIN_ZOOM, min(self.MAX_ZOOM, z))
        self.setTransform(QTransform.fromScale(z, z))
        self.zoomChanged.emit(z)

    def zoom_by(self, factor: float) -> None:
        z = self.zoom()
        nz = max(self.MIN_ZOOM, min(self.MAX_ZOOM, z * factor))
        if nz != z:
            self.scale(nz / z, nz / z)
            self.zoomChanged.emit(nz)
            self.viewChanged.emit()

    def center_on_item_rect(self, rect: QRectF, margin: float = 1.4) -> None:
        r = QRectF(rect)
        c = r.center()
        r.setWidth(r.width() * margin)
        r.setHeight(r.height() * margin)
        r.moveCenter(c)
        self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
        self.zoomChanged.emit(self.zoom())

    # ---------------------------------------------------------------- режим добавления
    def set_add_mode(self, on: bool) -> None:
        self._add_mode = on
        self._clear_preview()
        self.setDragMode(QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.RubberBandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def add_mode(self) -> bool:
        return self._add_mode

    def _clear_preview(self) -> None:
        for it in (self._preview_circle, self._preview_line):
            if it is not None and it.scene() is self.scene():
                self.scene().removeItem(it)
        self._preview_circle = None
        self._preview_line = None
        self._add_center = None

    # ---------------------------------------------------------------- события
    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_by(math.pow(1.0015, delta))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = True
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if event.key() == Qt.Key.Key_Escape and self._add_mode:
            if self._add_center is not None:
                self._clear_preview()
            else:
                self.addModeCancelled.emit()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space = False
            self.viewport().setCursor(Qt.CursorShape.CrossCursor if self._add_mode else Qt.CursorShape.ArrowCursor)
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or (
                event.button() == Qt.MouseButton.LeftButton and self._space):
            self._panning = True
            self._pan_start = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self._add_mode and event.button() == Qt.MouseButton.LeftButton:
            p = self.mapToScene(event.position().toPoint())
            if self._add_center is None:
                self._add_center = p
                pen = QPen(QColor(255, 140, 0), 2)
                pen.setCosmetic(True)
                pen.setStyle(Qt.PenStyle.DashLine)
                self._preview_circle = self.scene().addEllipse(QRectF(p, p), pen)
                self._preview_line = self.scene().addLine(p.x(), p.y(), p.x(), p.y(), pen)
                self._preview_circle.setZValue(100)
                self._preview_line.setZValue(100)
            else:
                c = self._add_center
                r = math.hypot(p.x() - c.x(), p.y() - c.y())
                self._clear_preview()
                if r >= 3:
                    self.circleDrawn.emit(c.x(), c.y(), r)
            event.accept()
            return
        if self._add_mode and event.button() == Qt.MouseButton.RightButton:
            self._clear_preview()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            d = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            event.accept()
            return
        if self._add_mode and self._add_center is not None and self._preview_circle is not None:
            p = self.mapToScene(event.position().toPoint())
            c = self._add_center
            r = math.hypot(p.x() - c.x(), p.y() - c.y())
            self._preview_circle.setRect(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))
            self._preview_line.setLine(c.x(), c.y(), p.x(), p.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = False
            self.viewport().setCursor(
                Qt.CursorShape.OpenHandCursor if self._space else
                (Qt.CursorShape.CrossCursor if self._add_mode else Qt.CursorShape.ArrowCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)


def sync_views(a: ImageView, b: ImageView) -> None:
    """Синхронизирует масштаб и положение двух видов с одинаковыми координатами сцены."""
    state = {"busy": False}

    def copy(src: ImageView, dst: ImageView):
        if state["busy"]:
            return
        state["busy"] = True
        try:
            dst.setTransform(src.transform())
            center = src.mapToScene(src.viewport().rect().center())
            dst.centerOn(center)
        finally:
            state["busy"] = False

    a.viewChanged.connect(lambda: copy(a, b))
    b.viewChanged.connect(lambda: copy(b, a))
    a.zoomChanged.connect(lambda _: copy(a, b))
    b.zoomChanged.connect(lambda _: copy(b, a))
