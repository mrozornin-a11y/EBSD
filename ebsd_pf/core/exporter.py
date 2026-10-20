"""Экспорт полюсных фигур: отдельная фигура, зерно, образец, все образцы. PNG/TIFF/SVG/PDF."""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QMarginsF, QRectF, QSize, QSizeF
from PySide6.QtGui import QColor, QImage, QPageLayout, QPageSize, QPainter, QPdfWriter
from PySide6.QtSvg import QSvgGenerator

from .. import APP_NAME, __version__
from .imageio_utils import save_rgb
from .models import ExportSettings, NO_GRAIN, NO_SAMPLE, PoleFigure, Project, normalize_plane
from .pipeline import Pipeline
from .render import Sheet, layout_grain, layout_sample, paint_sheet, qimage_to_rgb

log = logging.getLogger(__name__)

MAX_RASTER_PIXELS = 400_000_000
SVG_DPI = 96.0


class ExportError(Exception):
    pass


def safe_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(s)).strip(" .")
    return s or "_"


def sample_title(sample: str) -> str:
    return sample if sample == NO_SAMPLE else f"Образец {sample}"


def grain_title(grain: str) -> str:
    return "Зерно —" if grain == NO_GRAIN else f"Зерно {grain}"


def entries(project: Project, figs: list) -> list:
    return [(f.id, f.plane_label, project.params_for(f)) for f in figs]


def grain_sheet(project: Project, sample: str, grain: str) -> Sheet:
    figs = project.grain_figures(sample, grain)
    return layout_grain(entries(project, figs), project.layout,
                        f"{sample_title(sample)}, {grain_title(grain).lower()}")


def figure_sheet(project: Project, fig: PoleFigure) -> Sheet:
    ls = project.layout
    from dataclasses import replace
    return layout_grain(entries(project, [fig]), replace(ls, show_group_title=False))


def sample_sheet(project: Project, sample: str) -> Sheet:
    blocks = [(grain_title(g), entries(project, project.grain_figures(sample, g)))
              for g in project.grains(sample)]
    return layout_sample(blocks, project.layout, sample_title(sample))


@dataclass
class ExportJob:
    kind: str                 # figure | grain | sample | all
    sample: str = ""
    grain: str = ""
    fig_id: str = ""


def _grain_base(sample: str, grain: str) -> str:
    g = "x" if grain == NO_GRAIN else grain
    return f"{safe_name(sample)}_grain{safe_name(g)}"


def plan(project: Project, job: ExportJob, settings: ExportSettings) -> list:
    """Список (относительный путь без расширения, Sheet)."""
    items = []
    if job.kind == "figure":
        fig = project.figure(job.fig_id)
        if fig is None:
            raise ExportError("Фигура не найдена")
        plane = normalize_plane(fig.plane) or fig.id[:6]
        items.append((f"{_grain_base(fig.sample_key, fig.grain_key)}_{safe_name(plane)}",
                      figure_sheet(project, fig)))
    elif job.kind == "grain":
        if not project.grain_figures(job.sample, job.grain):
            raise ExportError("В зерне нет фигур")
        items.append((_grain_base(job.sample, job.grain), grain_sheet(project, job.sample, job.grain)))
    elif job.kind in ("sample", "all"):
        samples = [job.sample] if job.kind == "sample" else project.samples()
        for s in samples:
            folder = safe_name(s) if job.kind == "all" or settings.combine_sample else ""
            for g in project.grains(s):
                base = _grain_base(s, g)
                items.append((os.path.join(folder, base) if folder else base, grain_sheet(project, s, g)))
            if settings.combine_sample:
                base = f"{safe_name(s)}_all_grains"
                items.append((os.path.join(folder, base) if folder else base, sample_sheet(project, s)))
    else:
        raise ExportError(f"Неизвестный тип экспорта: {job.kind}")
    return items


def raster_size(sheet: Sheet, dpi: float) -> tuple[int, int]:
    return (max(1, int(math.ceil(sheet.width * dpi / 25.4))),
            max(1, int(math.ceil(sheet.height * dpi / 25.4))))


def export_sheet(sheet: Sheet, base_path: str, fmt: str, settings: ExportSettings,
                 layer_provider) -> str:
    fmt = fmt.lower()
    ext = {"png": ".png", "tiff": ".tif", "svg": ".svg", "pdf": ".pdf"}[fmt]
    path = base_path + ext
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt in ("png", "tiff"):
        dpi = float(settings.dpi)
        w, h = raster_size(sheet, dpi)
        if w * h > MAX_RASTER_PIXELS:
            raise ExportError(f"Слишком большое изображение ({w}×{h} пикс.). "
                              f"Уменьшите DPI или диаметр фигур.")
        img = QImage(w, h, QImage.Format.Format_RGB32)
        if img.isNull():
            raise ExportError(f"Недостаточно памяти для изображения {w}×{h} пикс.")
        img.fill(QColor(255, 255, 255))
        painter = QPainter(img)
        try:
            paint_sheet(painter, sheet, dpi / 25.4, layer_provider, dpi / 25.4)
        finally:
            painter.end()
        save_rgb(path, qimage_to_rgb(img), int(dpi))
    elif fmt == "svg":
        upm = SVG_DPI / 25.4
        gen = QSvgGenerator()
        gen.setFileName(path)
        w, h = sheet.width * upm, sheet.height * upm
        gen.setSize(QSize(int(math.ceil(w)), int(math.ceil(h))))
        gen.setViewBox(QRectF(0, 0, w, h))
        gen.setResolution(int(SVG_DPI))
        gen.setTitle("Полюсные фигуры")
        gen.setDescription(f"{APP_NAME} {__version__}")
        painter = QPainter()
        if not painter.begin(gen):
            raise ExportError(f"Не удалось создать файл SVG: {path}")
        try:
            paint_sheet(painter, sheet, upm, layer_provider, settings.raster_layer_dpi / 25.4,
                        vector_points=settings.vector_points)
        finally:
            painter.end()
        _fix_svg_size(path, sheet.width, sheet.height)
    elif fmt == "pdf":
        writer = QPdfWriter(path)
        writer.setResolution(1200)
        writer.setCreator(f"{APP_NAME} {__version__}")
        writer.setTitle("Полюсные фигуры")
        page = QPageSize(QSizeF(sheet.width, sheet.height), QPageSize.Unit.Millimeter, "Custom",
                         QPageSize.SizeMatchPolicy.ExactMatch)
        writer.setPageLayout(QPageLayout(page, QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0),
                                         QPageLayout.Unit.Millimeter))
        painter = QPainter()
        if not painter.begin(writer):
            raise ExportError(f"Не удалось создать файл PDF: {path}")
        try:
            paint_sheet(painter, sheet, writer.resolution() / 25.4, layer_provider,
                        settings.raster_layer_dpi / 25.4, vector_points=settings.vector_points)
        finally:
            painter.end()
    else:
        raise ExportError(f"Неподдерживаемый формат: {fmt}")
    log.info("Экспортировано: %s", path)
    return path


def _fix_svg_size(path: str, width_mm: float, height_mm: float) -> None:
    """QSvgGenerator округляет физический размер до целых пикселей; задаём точный размер в мм,
    чтобы масштаб по осям совпадал с viewBox (без анизотропного искажения)."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r'<svg width="[^"]*" height="[^"]*"',
                  f'<svg width="{width_mm:.4f}mm" height="{height_mm:.4f}mm"', text, count=1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def make_layer_provider(project: Project, pipeline: Pipeline):
    def provider(fig_id, params, radius_px):
        fig = project.figure(fig_id)
        if fig is None:
            return None
        return pipeline.layer(fig, params, radius_px)
    return provider


def run_export(project: Project, pipeline: Pipeline, job: ExportJob, settings: ExportSettings,
               progress: Optional[Callable[[int], None]] = None,
               is_cancelled: Optional[Callable[[], bool]] = None) -> list:
    if not settings.output_dir:
        raise ExportError("Не выбрана папка для экспорта")
    if not settings.formats:
        raise ExportError("Не выбран ни один формат")
    items = plan(project, job, settings)
    provider = make_layer_provider(project, pipeline)
    total = len(items) * len(settings.formats)
    done = 0
    written = []
    for rel, sheet in items:
        for fmt in settings.formats:
            if is_cancelled and is_cancelled():
                return written
            written.append(export_sheet(sheet, os.path.join(settings.output_dir, rel), fmt, settings,
                                        provider))
            done += 1
            if progress:
                progress(int(100 * done / max(total, 1)))
    return written
