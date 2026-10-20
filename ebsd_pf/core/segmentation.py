"""Выделение цветных пикселей и кластеров внутри полюсной фигуры.

Принципы:
* Новые точки не создаются: в маску попадают только пиксели исходного изображения.
  Единственное исключение — опциональное восстановление участков кластеров, перекрытых
  осевой линией (заполняется только узкая полоса оси между цветными пикселями одного
  цвета по обе стороны линии); такие пиксели учитываются и показываются отдельно.
* Удаление шума никогда не происходит незаметно: удалённые объекты сохраняются
  в отдельной маске и списке и показываются пользователю.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from .geometry import Crop, chroma_value, pixel_grid

SAME_CLUSTER_DELTA_E = 25.0   # порог CIEDE2000: цвета по обе стороны оси принадлежат одному кластеру
from .models import MaskEdits, ReconParams


@dataclass
class LineInfo:
    ring_found: bool = False
    ring_r: float = 0.0          # радиус средней линии окружности (пикс. вырезки)
    ring_inner: float = 0.0      # внутренний край линии окружности
    ring_width: float = 0.0
    h_axis: Optional[tuple[int, int]] = None   # строки полосы горизонтальной оси [y0, y1)
    v_axis: Optional[tuple[int, int]] = None   # столбцы полосы вертикальной оси [x0, x1)


@dataclass
class SegResult:
    crop: Crop
    inner_r: float
    mask: np.ndarray                 # bool HxW — итоговые цветные пиксели
    colors: np.ndarray               # uint8 HxWx3 — цвета (значимы внутри маски)
    removed_mask: np.ndarray         # bool — удалено как шум
    removed: list = field(default_factory=list)   # [(x, y, area)] в координатах вырезки
    restored_mask: Optional[np.ndarray] = None   # восстановлено под осями
    trimmed_mask: Optional[np.ndarray] = None    # ореолы JPEG, обрезанные по краям кластеров
    manual_add: int = 0
    manual_remove: int = 0
    n_achromatic: int = 0                        # объекты без выраженного цвета
    lines: LineInfo = field(default_factory=LineInfo)
    labels: Optional[np.ndarray] = None
    n_blobs: int = 0
    centroids: Optional[np.ndarray] = None       # (n,2) непрерывные координаты вырезки
    areas: Optional[np.ndarray] = None
    bboxes: Optional[np.ndarray] = None          # (n,4) x, y, w, h

    @property
    def n_pixels(self) -> int:
        return int(self.mask.sum())

    def stats_text(self) -> str:
        lines = [
            f"Цветных пикселей: {self.n_pixels}",
            f"Кластеров (связных объектов): {self.n_blobs}",
        ]
        if self.removed:
            lines.append(f"Удалено как шум: {len(self.removed)} объектов, "
                         f"{int(self.removed_mask.sum())} пикс.")
        else:
            lines.append("Удалено как шум: нет")
        if self.restored_mask is not None and self.restored_mask.any():
            lines.append(f"Восстановлено под осями: {int(self.restored_mask.sum())} пикс.")
        if self.trimmed_mask is not None and self.trimmed_mask.any():
            lines.append(f"Обрезано ореолов JPEG: {int(self.trimmed_mask.sum())} пикс.")
        if self.n_achromatic:
            lines.append(f"Ахроматических (серых) объектов: {self.n_achromatic} — цвет мог быть "
                         f"утрачен при сжатии JPEG")
        if self.manual_add or self.manual_remove:
            lines.append(f"Ручная правка: +{self.manual_add} / −{self.manual_remove} пикс.")
        li = self.lines
        lines.append("Окружность: " + (f"найдена, толщина {li.ring_width:.1f} пикс." if li.ring_found
                                       else "линия не выражена"))
        ax = []
        if li.h_axis:
            ax.append("горизонтальная")
        if li.v_axis:
            ax.append("вертикальная")
        lines.append("Оси: " + (", ".join(ax) if ax else "не найдены"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------
def analyze_lines(dark: np.ndarray, cx: float, cy: float, r: float) -> LineInfo:
    """Определяет положение и толщину линии окружности и осей по маске тёмных пикселей."""
    h, w = dark.shape
    X, Y = pixel_grid(h, w, cx, cy)
    D = np.hypot(X, Y)
    info = LineInfo(ring_r=r, ring_inner=r * 0.98)
    lo = 0.85 * r
    ann = (D >= lo) & (D < 1.15 * r)
    step = 0.5
    idx = ((D[ann] - lo) / step).astype(np.int64)
    if idx.size:
        nb = int(idx.max()) + 1
        tot = np.bincount(idx, minlength=nb).astype(np.float64)
        drk = np.bincount(idx, weights=dark[ann].astype(np.float64), minlength=nb)
        frac = np.where(tot > 0, drk / np.maximum(tot, 1), 0)
        p = int(np.argmax(frac))
        pv = frac[p]
        if pv >= 0.3:
            thr = max(0.25 * pv, 0.12)
            a = p
            while a > 0 and frac[a - 1] >= thr:
                a -= 1
            b = p
            while b < nb - 1 and frac[b + 1] >= thr:
                b += 1
            info.ring_found = True
            info.ring_inner = lo + a * step
            outer = lo + (b + 1) * step
            info.ring_width = outer - info.ring_inner
            info.ring_r = (info.ring_inner + outer) / 2.0
    inner = info.ring_inner
    span = 0.9 * inner
    need = 0.35 * 2 * span

    def find_band(horizontal: bool):
        tol = max(4, int(0.08 * r))
        if horizontal:
            c0, c1 = int(max(0, math.floor(cx - span))), int(min(w, math.ceil(cx + span)))
            r0, r1 = int(max(0, math.floor(cy) - tol)), int(min(h, math.floor(cy) + tol + 1))
            prof = dark[r0:r1, c0:c1].sum(axis=1)
        else:
            c0, c1 = int(max(0, math.floor(cy - span))), int(min(h, math.ceil(cy + span)))
            r0, r1 = int(max(0, math.floor(cx) - tol)), int(min(w, math.floor(cx) + tol + 1))
            prof = dark[c0:c1, r0:r1].sum(axis=0)
        if prof.size == 0:
            return None
        p = int(np.argmax(prof))
        if prof[p] < need:
            return None
        t = 0.35 * prof[p]
        a = p
        while a > 0 and prof[a - 1] >= t:
            a -= 1
        b = p
        while b < len(prof) - 1 and prof[b + 1] >= t:
            b += 1
        if b - a + 1 > max(8, 0.08 * r):   # слишком широкая «ось» — это не линия
            return None
        return (r0 + a, r0 + b + 1)

    info.h_axis = find_band(True)
    info.v_axis = find_band(False)
    return info


def _edits_on_crop(edits: MaskEdits, crop: Crop) -> tuple[np.ndarray, np.ndarray]:
    n = edits.add.shape[0]
    size = crop.size
    coords = np.arange(size, dtype=np.float64) + 0.5
    col = np.floor(((coords - crop.cx) / crop.r + 1.0) / 2.0 * n).astype(np.int64)
    row = np.floor(((coords - crop.cy) / crop.r + 1.0) / 2.0 * n).astype(np.int64)
    cv = (col >= 0) & (col < n)
    rv = (row >= 0) & (row < n)
    colc = np.clip(col, 0, n - 1)
    rowc = np.clip(row, 0, n - 1)
    valid = rv[:, None] & cv[None, :]
    add = edits.add[np.ix_(rowc, colc)] & valid
    rem = edits.remove[np.ix_(rowc, colc)] & valid
    return add, rem


def crop_to_edit_index(crop: Crop, n: int, x: np.ndarray, y: np.ndarray):
    """Пиксели вырезки -> индексы сетки правок (для редактора маски)."""
    col = np.floor(((x + 0.5 - crop.cx) / crop.r + 1.0) / 2.0 * n).astype(np.int64)
    row = np.floor(((y + 0.5 - crop.cy) / crop.r + 1.0) / 2.0 * n).astype(np.int64)
    return row, col


def _restore_under_axes(colored, colors, rgb, lines: LineInfo, disk, r):
    """Заполняет полосу оси там, где кластер одного цвета виден по обе стороны линии."""
    restored = np.zeros_like(colored)
    max_band = max(6, int(0.05 * r))
    h, w = colored.shape

    def process(band, transpose):
        if band is None:
            return
        y0, y1 = band
        if y1 - y0 > max_band:
            return
        C = colored.T if transpose else colored
        K = colors.transpose(1, 0, 2) if transpose else colors
        Dk = disk.T if transpose else disk
        R = restored.T if transpose else restored
        H = C.shape[0]
        if y0 - 2 < 0 or y1 + 2 > H:
            return
        above = C[y0 - 1] | C[y0 - 2]
        below = C[y1] | C[y1 + 1]
        ca = np.where(C[y0 - 1][:, None], K[y0 - 1], K[y0 - 2]).astype(np.int16)
        cb = np.where(C[y1][:, None], K[y1], K[y1 + 1]).astype(np.int16)
        cand = above & below & Dk[y0:y1].all(axis=0)
        similar = np.zeros_like(cand)
        if cand.any():
            # один и тот же кластер: перцептивная разница цветов CIEDE2000 по обе стороны линии
            la = rgb2lab(ca[cand][None].astype(np.float64) / 255.0)[0]
            lb = rgb2lab(cb[cand][None].astype(np.float64) / 255.0)[0]
            similar[cand] = deltaE_ciede2000(la, lb) <= SAME_CLUSTER_DELTA_E
        cols = np.nonzero(cand & similar)[0]
        n = y1 - y0
        for x in cols:
            for k, yy in enumerate(range(y0, y1)):
                if C[yy, x]:
                    continue
                t = (k + 1) / (n + 1)
                K[yy, x] = np.clip(ca[x] * (1 - t) + cb[x] * t, 0, 255).astype(np.uint8)
                C[yy, x] = True
                R[yy, x] = True

    process(lines.h_axis, False)
    process(lines.v_axis, True)
    return restored


def _masked_median(colors: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Медиана 3×3 по цветам только среди пикселей маски (без примеси фона)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return colors
    pad_m = np.pad(mask, 1)
    pad_c = np.pad(colors, ((1, 1), (1, 1), (0, 0)))
    neigh = []
    valid = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neigh.append(pad_c[ys + 1 + dy, xs + 1 + dx].astype(np.float32))
            valid.append(pad_m[ys + 1 + dy, xs + 1 + dx])
    N = np.stack(neigh, axis=1)           # (n, 9, 3)
    V = np.stack(valid, axis=1)           # (n, 9)
    N[~V] = np.nan
    med = np.nanmedian(N, axis=1)
    out = colors.copy()
    out[ys, xs] = np.clip(med + 0.5, 0, 255).astype(np.uint8)
    return out


def _cluster_median(colors, labels, n):
    out = colors.copy()
    flat_l = labels.ravel()
    sel = flat_l > 0
    if not sel.any():
        return out
    idx = np.nonzero(sel)[0]
    order = idx[np.argsort(flat_l[idx], kind="stable")]
    lab_sorted = flat_l[order]
    splits = np.nonzero(np.diff(lab_sorted))[0] + 1
    flat_c = out.reshape(-1, 3)
    for grp in np.split(order, splits):
        flat_c[grp] = np.median(flat_c[grp], axis=0).astype(np.uint8)
    return out


def segment(crop: Crop, params: ReconParams, edits: Optional[MaskEdits] = None,
            dark_threshold: int = 140) -> SegResult:
    p = params.effective()
    rgb = crop.rgb
    h, w = rgb.shape[:2]
    chroma, value = chroma_value(rgb)
    dark = (value < dark_threshold) & (chroma < 90)
    lines = analyze_lines(dark, crop.cx, crop.cy, crop.r)

    X, Y = pixel_grid(h, w, crop.cx, crop.cy)
    D = np.hypot(X, Y)
    inner_r = max(1.0, lines.ring_inner - float(p.edge_margin))
    disk = D <= inner_r

    thr = int(p.color_threshold)
    # уровень фона (бумага может быть не идеально белой)
    inside_vals = value[disk]
    bgv = float(np.percentile(inside_vals, 90)) if inside_vals.size else 255.0
    bgv = max(bgv, 180.0)
    darkness = bgv - rgb.min(axis=2).astype(np.float32)

    # зона линий: окрестность осей и внутренний край окружности. Здесь принимаются
    # только явно цветные пиксели (бахрома и «звон» JPEG у чёрных линий отбрасываются).
    zone = np.zeros((h, w), dtype=bool)
    if lines.h_axis:
        pad = max(2, (lines.h_axis[1] - lines.h_axis[0]) // 2)
        zone[max(0, lines.h_axis[0] - pad):lines.h_axis[1] + pad, :] = True
    if lines.v_axis:
        pad = max(2, (lines.v_axis[1] - lines.v_axis[0]) // 2)
        zone[:, max(0, lines.v_axis[0] - pad):lines.v_axis[1] + pad] = True
    zone |= D >= inner_r - 2.0
    strong = chroma >= int(thr * 1.6)

    colored = chroma >= thr
    achromatic = np.zeros((h, w), dtype=bool)
    if p.include_dark_points:
        # ахроматические точки: заметно темнее фона (в т.ч. цветные точки, обесцвеченные JPEG)
        achromatic = (darkness >= thr + 15) & ~colored
        colored |= achromatic
    colored &= disk
    colored &= ~zone | strong
    achromatic &= colored

    colors = rgb.copy()
    restored = None
    if p.restore_under_axes:
        restored = _restore_under_axes(colored, colors, rgb, lines, disk, crop.r)

    # --- удаление шума (с учётом и отчётом)
    removed_mask = np.zeros((h, w), dtype=bool)
    removed: list = []
    if p.min_object_size > 1 and colored.any():
        n, labels, stats, cents = cv2.connectedComponentsWithStats(colored.astype(np.uint8), connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA]
        small = areas < int(p.min_object_size)
        small[0] = False
        if p.protect_saturated and small.any():
            # «сильные» объекты (насыщенные или заметно темнее фона) не удаляются никогда
            strength = np.maximum(chroma, darkness.astype(np.int16)).astype(np.int32)
            maxst = np.zeros(n, dtype=np.int32)
            np.maximum.at(maxst, labels[colored], strength[colored])
            protected = maxst >= max(2 * thr, 80)
            small &= ~protected
        if small.any():
            removed_mask = small[labels]
            colored &= ~removed_mask
            for i in np.nonzero(small)[0]:
                removed.append((float(cents[i][0] + 0.5), float(cents[i][1] + 0.5), int(areas[i])))

    # --- обрезка ореолов JPEG по краю кластеров (только режим улучшения)
    trimmed = None
    if p.jpeg_cleanup and colored.any():
        n, labels = cv2.connectedComponents(colored.astype(np.uint8), connectivity=8)
        maxch = np.zeros(n, dtype=np.int32)
        np.maximum.at(maxch, labels[colored], chroma[colored].astype(np.int32))
        eroded = cv2.erode(colored.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        edge = colored & ~eroded
        weak = edge & (chroma < 0.45 * maxch[labels])
        # объект целиком не удаляется: пиксель максимальной хроматичности сохраняется
        keep_px = colored & (chroma >= maxch[labels])
        weak &= ~keep_px
        trimmed = weak
        colored &= ~weak

    # --- ручные правки (имеют приоритет над автоматикой)
    manual_add = manual_remove = 0
    if edits is not None and not edits.is_empty():
        add, rem = _edits_on_crop(edits, crop)
        nonwhite = ~((value >= 235) & (chroma < 12))
        add &= disk & nonwhite
        before = colored.copy()
        colored |= add
        colored &= ~rem
        manual_add = int((colored & ~before).sum())
        manual_remove = int((before & ~colored).sum())
        if removed:
            removed_mask &= ~add
            removed = [t for t in removed if not add[min(int(t[1]), h - 1), min(int(t[0]), w - 1)]]

    # --- цвета
    if p.color_mode == "smooth":
        colors = _masked_median(colors, colored)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(colored.astype(np.uint8), connectivity=8)
    if p.color_mode == "cluster":
        colors = _cluster_median(colors, labels, n)

    n_achrom = 0
    if achromatic.any() and n > 1:
        has_col = np.zeros(n, dtype=bool)
        has_col[np.unique(labels[colored & ~achromatic])] = True
        has_any = np.zeros(n, dtype=bool)
        has_any[np.unique(labels[colored])] = True
        n_achrom = int((has_any & ~has_col)[1:].sum())
    res = SegResult(crop=crop, inner_r=inner_r, mask=colored, colors=colors, n_achromatic=n_achrom,
                    removed_mask=removed_mask, removed=removed, restored_mask=restored,
                    trimmed_mask=trimmed, manual_add=manual_add, manual_remove=manual_remove,
                    lines=lines, labels=labels, n_blobs=int(n - 1),
                    centroids=(cents[1:] + 0.5) if n > 1 else np.zeros((0, 2)),
                    areas=stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.zeros(0, dtype=np.int64),
                    bboxes=stats[1:, :4] if n > 1 else np.zeros((0, 4), dtype=np.int64))
    return res


def overlay_image(seg: SegResult, show_mask=True, show_removed=True, show_restored=True) -> np.ndarray:
    """Исходная вырезка с подсветкой маски (для сравнения и контроля)."""
    img = seg.crop.rgb.astype(np.float32)
    if show_mask:
        m = seg.mask
        # контур маски — голубой
        er = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        edge = m & ~er
        img[edge] = img[edge] * 0.3 + np.array([0, 200, 255]) * 0.7
    if show_restored and seg.restored_mask is not None:
        img[seg.restored_mask] = np.array([0, 255, 0])
    if show_removed and seg.removed_mask.any():
        rm = seg.removed_mask.astype(np.uint8)
        ring = cv2.dilate(rm, np.ones((5, 5), np.uint8)).astype(bool) & ~seg.removed_mask
        img[ring] = np.array([255, 0, 255])
    if seg.trimmed_mask is not None and show_removed:
        img[seg.trimmed_mask] = img[seg.trimmed_mask] * 0.4 + np.array([255, 120, 0]) * 0.6
    return np.clip(img, 0, 255).astype(np.uint8)
