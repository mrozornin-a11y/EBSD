"""Компоновка и отрисовка полюсных фигур через QPainter.

Один и тот же код рисует предпросмотр на экране и экспорт в PNG/TIFF/SVG/PDF,
поэтому результат экспорта совпадает с предпросмотром. Все размеры компоновки
задаются в миллиметрах; толщины линий и шрифты — в пунктах.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath,
                           QPen)

from .models import LayoutSettings, ReconParams

PT_TO_MM = 25.4 / 72.0
FONT_PX = 100.0
FONT_FAMILIES = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

# layer_provider(fig_id, params, radius_px) -> RGBA ndarray | None
LayerProvider = Callable[[str, ReconParams, float], Optional[np.ndarray]]


def make_font(bold: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setStyleHint(QFont.StyleHint.SansSerif)
    f.setPixelSize(int(FONT_PX))
    f.setBold(bold)
    f.setKerning(True)
    return f


def text_box_mm(text: str, size_pt: float, bold: bool = False) -> QRectF:
    """Плотный прямоугольник текста в мм относительно точки базовой линии."""
    fm = QFontMetricsF(make_font(bold))
    br = fm.tightBoundingRect(text)
    adv = fm.horizontalAdvance(text)
    s = size_pt * PT_TO_MM / FONT_PX
    width = max(br.width(), adv - max(br.x(), 0))
    return QRectF(br.x() * s, br.y() * s, width * s, br.height() * s)


def label_geometry(r: float, text: str, p: ReconParams) -> tuple[QRectF, QPointF]:
    """Положение подписи относительно центра окружности (мм).

    Подпись всегда вне круга: прямоугольник подписи не пересекает диск радиуса
    r + половина толщины линии + зазор, поэтому не может перекрыть точки."""
    box = text_box_mm(text, p.label_size_pt)
    w, h = box.width(), box.height()
    lw = p.circle_width_pt * PT_TO_MM
    off = 0.8 + lw / 2.0
    c = math.sqrt(0.5)
    if p.label_position == "top_left":
        right = -(r + off) * c
        left = right - w
        bottom = -(r + off) * c
    elif p.label_position == "top_center":
        left = -w / 2.0
        bottom = -(r + off)
    else:  # top_right
        left = (r + off) * c
        bottom = -(r + off) * c
    rect = QRectF(left + p.label_dx_mm, bottom - h + p.label_dy_mm, w, h)
    need = r + off
    for _ in range(60):
        nx = min(max(0.0, rect.left()), rect.right())
        ny = min(max(0.0, rect.top()), rect.bottom())
        d = math.hypot(nx, ny)
        if d >= need - 1e-6:
            break
        rect.translate(0, -(need - d) - 0.05)
    origin = QPointF(rect.left() - box.x(), rect.top() - box.y())
    return rect, origin


@dataclass
class PlacedFigure:
    fig_id: str
    cx: float
    cy: float
    r: float
    label: str
    params: ReconParams
    label_rect: Optional[QRectF] = None
    label_origin: Optional[QPointF] = None


@dataclass
class PlacedText:
    text: str
    origin: QPointF
    size_pt: float
    bold: bool = False


@dataclass
class Sheet:
    width: float
    height: float
    figures: list = field(default_factory=list)
    texts: list = field(default_factory=list)


# entry: (fig_id, label_text, params)
def _grid_block(entries: list, ls: LayoutSettings, x0: float, y0: float,
                pitch: Optional[tuple] = None) -> tuple[list, float, float]:
    n = len(entries)
    if n == 0:
        return [], 0.0, 0.0
    cols = ls.columns if ls.columns > 0 else int(math.ceil(math.sqrt(n)))
    cols = max(1, min(cols, n))
    rows = int(math.ceil(n / cols))
    r = ls.diameter_mm / 2.0
    ext_t = ext_l = ext_r = 0.0
    geo = []
    for fid, text, p in entries:
        lw = p.circle_width_pt * PT_TO_MM / 2.0
        ext_t, ext_l, ext_r = max(ext_t, lw), max(ext_l, lw), max(ext_r, lw)
        if p.show_label and text:
            rect, origin = label_geometry(r, text, p)
            ext_t = max(ext_t, -rect.top() - r)
            ext_r = max(ext_r, rect.right() - r)
            ext_l = max(ext_l, -rect.left() - r)
            geo.append((rect, origin))
        else:
            geo.append((None, None))
    ext_b = max(p.circle_width_pt * PT_TO_MM / 2.0 for _, _, p in entries)
    px = 2 * r + ls.spacing_mm + ext_l + ext_r
    py = 2 * r + ls.spacing_mm + ext_t + ext_b
    if pitch:
        px, py = max(px, pitch[0]), max(py, pitch[1])
    placed = []
    for i, ((fid, text, p), (rect, origin)) in enumerate(zip(entries, geo)):
        row, col = divmod(i, cols)
        cx = x0 + ext_l + r + col * px
        cy = y0 + ext_t + r + row * py
        placed.append(PlacedFigure(fid, cx, cy, r, text, p, rect, origin))
    width = ext_l + 2 * r + (cols - 1) * px + ext_r
    height = ext_t + 2 * r + (rows - 1) * py + ext_b
    return placed, width, height


def layout_grain(entries: list, ls: LayoutSettings, title: str = "") -> Sheet:
    m = ls.margin_mm
    y = m
    texts = []
    if title and ls.show_group_title:
        box = text_box_mm(title, ls.title_size_pt, bold=True)
        texts.append(PlacedText(title, QPointF(m - box.x(), y - box.y()), ls.title_size_pt, True))
        y += box.height() + 2.0
    placed, w, h = _grid_block(entries, ls, m, y)
    width = max(w, max((text_box_mm(t.text, t.size_pt, t.bold).width() for t in texts), default=0))
    return Sheet(width + 2 * m, y + h + m, placed, texts)


def layout_sample(blocks: list, ls: LayoutSettings, title: str = "") -> Sheet:
    """blocks: [(grain_title, entries)] — зерна образца друг под другом с подписями."""
    m = ls.margin_mm
    y = m
    texts = []
    figs = []
    width = 0.0
    if title:
        box = text_box_mm(title, ls.title_size_pt + 1, bold=True)
        texts.append(PlacedText(title, QPointF(m - box.x(), y - box.y()), ls.title_size_pt + 1, True))
        y += box.height() + 3.0
        width = box.width()
    for k, (gtitle, entries) in enumerate(blocks):
        if gtitle:
            box = text_box_mm(gtitle, ls.title_size_pt)
            texts.append(PlacedText(gtitle, QPointF(m - box.x(), y - box.y()), ls.title_size_pt))
            y += box.height() + 1.5
            width = max(width, box.width())
        placed, w, h = _grid_block(entries, ls, m, y)
        figs.extend(placed)
        width = max(width, w)
        y += h + (ls.spacing_mm * 1.5 if k < len(blocks) - 1 else 0)
    return Sheet(width + 2 * m, y + m, figs, texts)


# ---------------------------------------------------------------------------------------
# Отрисовка
# ---------------------------------------------------------------------------------------
def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    a = np.ascontiguousarray(rgba)
    h, w = a.shape[:2]
    img = QImage(a.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return img.copy()


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    a = np.ascontiguousarray(rgb)
    h, w = a.shape[:2]
    img = QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()


def qimage_to_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8, count=img.bytesPerLine() * h).reshape(h, img.bytesPerLine())
    return arr[:, :w * 3].reshape(h, w, 3).copy()


def layer_to_paths(rgba: np.ndarray, max_colors: int = 12) -> list:
    """Векторизация слоя: [(QColor, QPainterPath в пикселях слоя)].

    Каждый кластер — объединение пиксельных квадратов (контур точно повторяет пиксели).
    Если в кластере больше max_colors различных цветов, используется медианный цвет."""
    A = rgba[..., 3] >= 128
    n, labels, stats, _ = cv2.connectedComponentsWithStats(A.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        sub = labels[y:y + h, x:x + w] == i
        cols = rgba[y:y + h, x:x + w, :3]
        packed = (cols[..., 0].astype(np.int64) << 16) | (cols[..., 1].astype(np.int64) << 8) | cols[..., 2]
        uniq = np.unique(packed[sub])
        if len(uniq) > max_colors:
            med = np.median(cols[sub], axis=0).astype(int)
            groups = [((int(med[0]), int(med[1]), int(med[2])), sub)]
        else:
            groups = [(((int(u) >> 16) & 255, (int(u) >> 8) & 255, int(u) & 255), sub & (packed == u))
                      for u in uniq]
        for color, m in groups:
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.WindingFill)
            for j in range(h):
                row = m[j]
                if not row.any():
                    continue
                d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
                starts = np.nonzero(d == 1)[0]
                ends = np.nonzero(d == -1)[0]
                for a, b in zip(starts, ends):
                    path.addRect(QRectF(x + a, y + j, b - a, 1))
            out.append((QColor(*color), path.simplified()))
    return out


def paint_figure(painter: QPainter, pf: PlacedFigure, layer_provider: Optional[LayerProvider],
                 layer_px_per_mm: float, vector_points: bool = False) -> None:
    p = pf.params
    cx, cy, r = pf.cx, pf.cy, pf.r
    black = QColor(0, 0, 0)
    if p.draw_axes and p.axis_width_pt > 0:
        pen = QPen(black, p.axis_width_pt * PT_TO_MM)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
        painter.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))

    if layer_provider is not None:
        rgba = layer_provider(pf.fig_id, p, r * layer_px_per_mm)
        if rgba is not None and rgba.shape[0] > 0:
            S = rgba.shape[0]
            size_mm = S / layer_px_per_mm
            x0, y0 = cx - size_mm / 2.0, cy - size_mm / 2.0
            if vector_points:
                painter.save()
                painter.translate(x0, y0)
                painter.scale(1.0 / layer_px_per_mm, 1.0 / layer_px_per_mm)
                painter.setPen(Qt.PenStyle.NoPen)
                for color, path in layer_to_paths(rgba):
                    painter.setBrush(QBrush(color))
                    painter.drawPath(path)
                painter.restore()
            else:
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, bool(p.smooth_scaling))
                painter.drawImage(QRectF(x0, y0, size_mm, size_mm), rgba_to_qimage(rgba))
                painter.restore()

    if p.circle_width_pt > 0:
        pen = QPen(black, p.circle_width_pt * PT_TO_MM)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    if p.show_label and pf.label and pf.label_origin is not None:
        _draw_text(painter, pf.label, QPointF(cx + pf.label_origin.x(), cy + pf.label_origin.y()),
                   p.label_size_pt)


def _draw_text(painter: QPainter, text: str, origin: QPointF, size_pt: float, bold: bool = False):
    painter.save()
    painter.translate(origin)
    s = size_pt * PT_TO_MM / FONT_PX
    painter.scale(s, s)
    painter.setFont(make_font(bold))
    painter.setPen(QColor(0, 0, 0))
    painter.drawText(QPointF(0, 0), text)
    painter.restore()


def paint_sheet(painter: QPainter, sheet: Sheet, unit_per_mm: float,
                layer_provider: Optional[LayerProvider], layer_px_per_mm: Optional[float] = None,
                vector_points: bool = False, background: bool = True) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.scale(unit_per_mm, unit_per_mm)
    if background:
        painter.fillRect(QRectF(0, 0, sheet.width, sheet.height), QColor(255, 255, 255))
    lppm = layer_px_per_mm or unit_per_mm
    for pf in sheet.figures:
        paint_figure(painter, pf, layer_provider, lppm, vector_points)
    for t in sheet.texts:
        _draw_text(painter, t.text, t.origin, t.size_pt, t.bold)
    painter.restore()


def render_sheet_image(sheet: Sheet, px_per_mm: float, layer_provider: Optional[LayerProvider]) -> QImage:
    w = max(1, int(math.ceil(sheet.width * px_per_mm)))
    h = max(1, int(math.ceil(sheet.height * px_per_mm)))
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(255, 255, 255))
    painter = QPainter(img)
    try:
        paint_sheet(painter, sheet, px_per_mm, layer_provider)
    finally:
        painter.end()
    return img
