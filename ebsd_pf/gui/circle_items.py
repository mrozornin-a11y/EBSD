"""Графический элемент окружности полюсной фигуры: перемещение, изменение радиуса, подпись."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
from PySide6.QtGui import QFontMetricsF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject


class LabelItem(QGraphicsItem):
    """Подпись с белой подложкой, не масштабируется вместе с видом."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.font = QFont()
        self.font.setPointSize(9)
        self.font.setBold(True)
        self._text = ""
        self._rect = QRectF()

    def setText(self, text: str) -> None:
        self.prepareGeometryChange()
        self._text = text
        fm = QFontMetricsF(self.font)
        self._rect = QRectF(0, -fm.height() - 4, fm.horizontalAdvance(text) + 8, fm.height() + 4)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._rect

    def paint(self, painter, option, widget=None) -> None:
        if not self._text:
            return
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255, 225)))
        painter.drawRoundedRect(self._rect, 3, 3)
        painter.setFont(self.font)
        painter.setPen(QColor(10, 10, 10))
        painter.drawText(self._rect, Qt.AlignmentFlag.AlignCenter, self._text)

COLOR_NEW = QColor(0, 160, 255)
COLOR_ASSIGNED = QColor(0, 180, 60)
COLOR_SELECTED = QColor(255, 120, 0)
HANDLE_PX = 9


class CircleItem(QGraphicsObject):
    geometryEdited = Signal(str)       # по окончании перемещения/изменения радиуса
    geometryLive = Signal(str)

    def __init__(self, fig_id: str, cx: float, cy: float, r: float):
        super().__init__()
        self.fig_id = fig_id
        self._r = float(r)
        self._resizing = False
        self._moved = False
        self._assigned = False
        self.setPos(cx, cy)
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self.label = LabelItem(self)
        self._place_label()

    # ------------------------------------------------------------------
    @property
    def r(self) -> float:
        return self._r

    def set_radius(self, r: float) -> None:
        self.prepareGeometryChange()
        self._r = max(2.0, float(r))
        self._place_label()
        self.update()

    def set_text(self, text: str, assigned: bool) -> None:
        self.label.setText(text)
        self._assigned = assigned
        self._place_label()
        self.update()

    def _place_label(self) -> None:
        c = math.sqrt(0.5)
        self.label.setPos(self._r * c, -self._r * c)

    def _view_scale(self) -> float:
        sc = self.scene()
        if sc and sc.views():
            return max(sc.views()[0].transform().m11(), 1e-6)
        return 1.0

    def _handle_rect(self) -> QRectF:
        h = HANDLE_PX / self._view_scale()
        return QRectF(self._r - h / 2, -h / 2, h, h)

    def boundingRect(self) -> QRectF:
        m = HANDLE_PX / self._view_scale() + 2
        return QRectF(-self._r - m, -self._r - m, 2 * (self._r + m), 2 * (self._r + m))

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(QPointF(0, 0), self._r + 3 / self._view_scale(), self._r + 3 / self._view_scale())
        if self.isSelected():
            path.addRect(self._handle_rect())
        return path

    def paint(self, painter, option, widget=None) -> None:
        sel = self.isSelected()
        color = COLOR_SELECTED if sel else (COLOR_ASSIGNED if self._assigned else COLOR_NEW)
        pen = QPen(color, 3 if sel else 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 25 if sel else 0)))
        painter.drawEllipse(QPointF(0, 0), self._r, self._r)
        s = 6 / self._view_scale()
        painter.drawLine(QPointF(-s, 0), QPointF(s, 0))
        painter.drawLine(QPointF(0, -s), QPointF(0, s))
        if sel:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            hp = QPen(color, 2)
            hp.setCosmetic(True)
            painter.setPen(hp)
            painter.drawRect(self._handle_rect())

    # ------------------------------------------------------------------
    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._moved = True
            self.geometryLive.emit(self.fig_id)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.prepareGeometryChange()
        return super().itemChange(change, value)

    def hoverMoveEvent(self, event) -> None:
        if self.isSelected() and self._handle_rect().adjusted(-2, -2, 2, 2).contains(event.pos()):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if (event.button() == Qt.MouseButton.LeftButton and self.isSelected()
                and self._handle_rect().adjusted(-2, -2, 2, 2).contains(event.pos())):
            self._resizing = True
            event.accept()
            return
        self._moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._resizing:
            p = event.pos()
            self.set_radius(math.hypot(p.x(), p.y()))
            self.geometryLive.emit(self.fig_id)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._resizing:
            self._resizing = False
            self.geometryEdited.emit(self.fig_id)
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self._moved:
            self._moved = False
            self.geometryEdited.emit(self.fig_id)
