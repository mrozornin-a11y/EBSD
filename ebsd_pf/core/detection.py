"""Автоматический поиск окружностей полюсных фигур.

Алгоритм:
1. Маска тёмных линий (низкая яркость и низкая хроматичность).
2. Кандидаты: контуры маски на уменьшенной копии + преобразование Хафа (OpenCV).
3. Уточнение каждого кандидата на исходном разрешении: по радиальным лучам ищутся
   пересечения с тёмной линией, по ним робастно подгоняется окружность.
4. Проверка признаков полюсной фигуры: окружность почти замкнута, тонкая, внутри
   преобладает фон, есть горизонтальная и вертикальная оси (необязательно).
5. Подавление дублей и вложенных окружностей.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .geometry import reading_order, robust_fit_circle
from .models import DetectParams

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


@dataclass
class DetectedCircle:
    cx: float
    cy: float
    r: float
    score: float
    ring_cov: float = 0.0
    axes_cov: float = 0.0
    inside_bg: float = 0.0
    thickness: float = 1.0


# ---------------------------------------------------------------------------------------
# Классификация пикселей в произвольных точках
# ---------------------------------------------------------------------------------------
def _sample_rgb(rgb: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (rgb в точках int16 (...,3), признак попадания в изображение)."""
    h, w = rgb.shape[:2]
    ix = np.floor(xs).astype(np.int64)
    iy = np.floor(ys).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ixc = np.clip(ix, 0, w - 1)
    iyc = np.clip(iy, 0, h - 1)
    vals = rgb[iyc, ixc].astype(np.int16)
    vals[~inside] = 255  # за пределами — белый фон
    return vals, inside


def _classify(rgb, xs, ys, dark_threshold):
    vals, _ = _sample_rgb(rgb, xs, ys)
    mx = vals.max(axis=-1)
    mn = vals.min(axis=-1)
    ch = mx - mn
    dark = (mx < dark_threshold) & (ch < 90)
    bg = (mx >= 200) & (ch <= 30)
    return dark, bg


def _sample_mask(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ix = np.floor(xs).astype(np.int64)
    iy = np.floor(ys).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = mask[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)] > 0
    return out & inside


def _ring_prescreen(mask: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Быстрая оценка доли окружности, покрытой тёмной линией (на уменьшенной маске)."""
    ang = np.linspace(0, 2 * math.pi, 120, endpoint=False)
    tol = max(1.5, 0.05 * r)
    offs = np.arange(-tol, tol + 0.01, 0.75)
    xs = cx + (r + offs[None, :]) * np.cos(ang)[:, None]
    ys = cy + (r + offs[None, :]) * np.sin(ang)[:, None]
    return float(_sample_mask(mask, xs, ys).any(axis=1).mean())


def _dark_mask_chunked(rgb: np.ndarray, thr: int, chunk: int = 1024) -> np.ndarray:
    h = rgb.shape[0]
    out = np.empty(rgb.shape[:2], dtype=np.uint8)
    for y in range(0, h, chunk):
        part = rgb[y:y + chunk].astype(np.int16)
        mx = part.max(axis=2)
        ch = mx - part.min(axis=2)
        out[y:y + chunk] = ((mx < thr) & (ch < 90)).astype(np.uint8) * 255
    return out


# ---------------------------------------------------------------------------------------
# Уточнение окружности по радиальным лучам
# ---------------------------------------------------------------------------------------
def _ray_edges(rgb, cx, cy, r, dark_threshold, n_angles=None, lo=0.8, hi=1.2, dark_mask=None):
    """Точки пересечения лучей с тёмной линией вблизи радиуса r.

    Возвращает (xs, ys, run_lengths, angle_mask)."""
    if n_angles is None:
        n_angles = int(np.clip(2 * math.pi * r / 2.0, 90, 720))
    step = 0.5
    radii = np.arange(lo * r, hi * r + step, step)
    n = len(radii)
    ang = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
    xs = cx + radii[None, :] * ca
    ys = cy + radii[None, :] * sa
    if dark_mask is not None:
        dark = _sample_mask(dark_mask, xs, ys)
    else:
        dark, _ = _classify(rgb, xs, ys, dark_threshold)
    if not dark.any():
        return np.empty(0), np.empty(0), np.empty(0), np.zeros(n_angles, bool)
    idx = np.arange(n)[None, :].repeat(n_angles, axis=0)
    mid = int(np.argmin(np.abs(radii - r)))
    # ближайший к r тёмный отсчёт на каждом луче
    dist = np.where(dark, np.abs(idx - mid), 10 ** 9)
    j = np.argmin(dist, axis=1)
    has = dist[np.arange(n_angles), j] < 10 ** 9
    # границы тёмного отрезка, содержащего j
    nd_idx = np.where(~dark, idx, -1)
    left_nd = np.maximum.accumulate(nd_idx, axis=1)            # последний светлый <= i
    nd_idx2 = np.where(~dark, idx, n)
    right_nd = np.minimum.accumulate(nd_idx2[:, ::-1], axis=1)[:, ::-1]  # первый светлый >= i
    rows = np.arange(n_angles)
    lft = left_nd[rows, j] + 1
    rgt = right_nd[rows, j] - 1
    run_len = (rgt - lft + 1) * step
    center_idx = (lft + rgt) / 2.0
    rad = lo * r + center_idx * step
    ok = has & (run_len <= max(0.12 * r, 4.0)) & (lft > 0) & (rgt < n - 1)
    ex = cx + rad * np.cos(ang)
    ey = cy + rad * np.sin(ang)
    return ex[ok], ey[ok], run_len[ok], ok


def refine_circle(rgb: np.ndarray, cx: float, cy: float, r: float, dark_threshold: int = 140,
                  iterations: int = 3, dark_mask: np.ndarray | None = None
                  ) -> Optional[tuple[float, float, float, float, float, float]]:
    """Уточняет окружность по изображению.

    Возвращает (cx, cy, r, rms, coverage, thickness) или None."""
    best = None
    for it in range(iterations):
        lo, hi = (0.75, 1.25) if it == 0 else (0.88, 1.12)
        ex, ey, runs, ok = _ray_edges(rgb, cx, cy, r, dark_threshold, lo=lo, hi=hi, dark_mask=dark_mask)
        if len(ex) < 12:
            return best
        ncx, ncy, nr, rms, keep = robust_fit_circle(ex, ey, min_tol=max(1.0, 0.01 * r))
        if not np.isfinite(rms) or nr <= 0:
            return best
        if abs(nr - r) > 0.3 * r or math.hypot(ncx - cx, ncy - cy) > 0.3 * r:
            return best
        coverage = float(keep.sum()) / float(len(ok))
        thick = float(np.median(runs[keep])) if keep.any() else 1.0
        cx, cy, r = ncx, ncy, nr
        best = (cx, cy, r, rms, coverage, thick)
    return best


# ---------------------------------------------------------------------------------------
# Признаки полюсной фигуры
# ---------------------------------------------------------------------------------------
def _features(rgb, cx, cy, r, thick, thr):
    n_ang = 360
    ang = np.linspace(0, 2 * math.pi, n_ang, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    tol = max(1.5, 0.5 * thick + 1.0, 0.015 * r)
    offs = np.arange(-tol, tol + 0.01, 0.5)
    xs = cx + (r + offs[None, :]) * ca[:, None]
    ys = cy + (r + offs[None, :]) * sa[:, None]
    dark, _ = _classify(rgb, xs, ys, thr)
    ring_cov = float(dark.any(axis=1).mean())

    gap = max(2.0 * thick + 1.0, 0.06 * r)
    d_in, _ = _classify(rgb, cx + (r - gap) * ca, cy + (r - gap) * sa, thr)
    d_out, _ = _classify(rgb, cx + (r + gap) * ca, cy + (r + gap) * sa, thr)
    inner_clear = 1.0 - float(d_in.mean())
    outer_clear = 1.0 - float(d_out.mean())

    # фон внутри: сетка точек по кольцам
    rin = max(r - gap, 1.0)
    rr = np.linspace(0.12, 0.88, 8) * rin
    aa = np.linspace(0, 2 * math.pi, 48, endpoint=False) + 0.05
    gx = cx + rr[:, None] * np.cos(aa)[None, :]
    gy = cy + rr[:, None] * np.sin(aa)[None, :]
    _, bg = _classify(rgb, gx, gy, thr)
    inside_bg = float(bg.mean())

    # оси: доля позиций вдоль диаметра, где вблизи есть не-фон, и доля тёмных
    t = np.linspace(-0.9, 0.9, 181) * rin
    atol = max(2.0, 0.03 * r)
    ao = np.arange(-atol, atol + 0.01, 0.5)

    def axis_cov(horizontal: bool):
        if horizontal:
            px = cx + t[:, None] + 0 * ao[None, :]
            py = cy + ao[None, :] + 0 * t[:, None]
        else:
            px = cx + ao[None, :] + 0 * t[:, None]
            py = cy + t[:, None] + 0 * ao[None, :]
        d, b = _classify(rgb, px, py, thr)
        # лучшая линия смещения (ось может быть чуть сдвинута)
        dark_line = d.mean(axis=0)
        best = int(np.argmax(dark_line))
        lo_i, hi_i = max(best - 2, 0), min(best + 3, len(ao))
        cover = (~b[:, lo_i:hi_i]).any(axis=1).mean()
        darkf = d[:, lo_i:hi_i].any(axis=1).mean()
        return float(cover), float(darkf)

    hc, hd = axis_cov(True)
    vc, vd = axis_cov(False)
    axes_cov = min(hc, vc) if min(hd, vd) >= 0.3 else min(hc, vc) * min(hd, vd) / 0.3
    return dict(ring_cov=ring_cov, inner_clear=inner_clear, outer_clear=outer_clear,
                inside_bg=inside_bg, axes_cov=float(axes_cov))


def evaluate_circle(rgb, cx, cy, r, params: DetectParams, thickness: float = 1.0) -> tuple[bool, dict]:
    f = _features(rgb, cx, cy, r, thickness, params.dark_threshold)
    ok = (f["ring_cov"] >= params.ring_coverage
          and f["inner_clear"] >= 0.5
          and f["outer_clear"] >= 0.5
          and f["inside_bg"] >= params.min_inside_background)
    if params.require_axes:
        ok = ok and f["axes_cov"] >= 0.6
    f["score"] = f["ring_cov"] + 0.5 * f["axes_cov"] + 0.25 * f["inside_bg"]
    return ok, f


# ---------------------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------------------
def _radius_bands(min_r: float, max_r: float) -> list:
    """Октавные полосы радиусов с перекрытием."""
    bands = []
    lo = float(min_r)
    while lo < max_r:
        hi = min(float(max_r), lo * 2.2)
        bands.append((lo, hi))
        if hi >= max_r:
            break
        lo = hi / 1.15
    return bands


def _band_candidates(dark_full, rgb, lo, hi, params: DetectParams, max_side: int):
    """Кандидаты в полосе радиусов [lo, hi] на копии, уменьшенной так, что радиус ≈ 12 пикс."""
    h, w = dark_full.shape
    s = min(1.0, 12.0 / lo, max_side / float(max(h, w)))
    s = max(s, min(1.0, 6.0 / lo))
    if s < 1.0:
        sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        small_dark = cv2.resize(dark_full, (sw, sh), interpolation=cv2.INTER_AREA)
        small_dark = (small_dark > 40).astype(np.uint8) * 255
        small_rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small_dark, small_rgb = dark_full, rgb
    rlo = max(3, int(math.floor(lo * s)))
    rhi = max(rlo + 2, int(math.ceil(hi * s)))
    cands: list[tuple[float, float, float]] = []

    # A) контуры тёмной маски
    closed = cv2.morphologyEx(small_dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_len = 2 * math.pi * rlo * 0.5
    max_len = 2 * math.pi * rhi * 3.0
    for c in contours:
        if len(c) < 12:
            continue
        ln = cv2.arcLength(c, True)
        if ln < min_len or ln > max_len * 4:
            continue
        (x, y), rr = cv2.minEnclosingCircle(c)
        if rlo * 0.8 <= rr <= rhi * 1.2:
            cands.append((x / s, y / s, rr / s))
        pts = c[:, 0, :].astype(np.float64)
        if len(pts) >= 20:
            fx, fy, fr, rms, _ = robust_fit_circle(pts[:, 0], pts[:, 1])
            if np.isfinite(rms) and rlo * 0.8 <= fr <= rhi * 1.2 and rms < 0.15 * fr:
                cands.append((fx / s, fy / s, fr / s))

    # B) преобразование Хафа
    gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
    try:
        hc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=max(rlo, 6),
                              param1=120, param2=max(10, int(rlo * 0.9)),
                              minRadius=rlo, maxRadius=rhi)
    except cv2.error as e:
        log.warning("HoughCircles: %s", e)
        hc = None
    if hc is not None:
        for x, y, rr in hc[0][:800]:
            cands.append((float(x) / s, float(y) / s, float(rr) / s))

    # быстрая проверка замкнутости по уменьшенной маске
    pre_thr = max(0.3, params.ring_coverage - 0.3)
    dil = cv2.dilate(small_dark, np.ones((3, 3), np.uint8))
    return [c for c in cands if _ring_prescreen(dil, c[0] * s, c[1] * s, c[2] * s) >= pre_thr]


def _candidates(rgb, params: DetectParams, max_r: float, is_cancelled=None):
    min_r = max(5, int(params.min_radius))
    dark_full = _dark_mask_chunked(rgb, params.dark_threshold)
    max_side = max(800, int(params.max_work_dim) * 2)
    cands: list[tuple[float, float, float]] = []
    for lo, hi in _radius_bands(min_r, max_r):
        if is_cancelled and is_cancelled():
            raise Cancelled()
        cands += _band_candidates(dark_full, rgb, lo, hi, params, max_side)
    # грубое удаление совпадающих кандидатов
    uniq: list[tuple[float, float, float]] = []
    for c in sorted(cands, key=lambda t: -t[2]):
        if not any(abs(c[2] - u[2]) < 0.08 * u[2] and math.hypot(c[0] - u[0], c[1] - u[1]) < 0.08 * u[2]
                   for u in uniq):
            uniq.append(c)
    return uniq, dark_full


def detect_circles(rgb: np.ndarray, params: DetectParams | None = None,
                   progress: Callable[[int], None] | None = None,
                   is_cancelled: Callable[[], bool] | None = None) -> list[DetectedCircle]:
    params = params or DetectParams()
    h, w = rgb.shape[:2]
    max_r = params.max_radius if params.max_radius > 0 else min(h, w) / 2.0
    min_r = max(5, params.min_radius)
    if progress:
        progress(5)
    cands, dark_full = _candidates(rgb, params, max_r, is_cancelled)
    log.info("Кандидатов окружностей: %d", len(cands))
    if progress:
        progress(25)

    accepted: list[DetectedCircle] = []
    n = max(1, len(cands))
    for k, (cx, cy, r) in enumerate(cands):
        if is_cancelled and is_cancelled():
            raise Cancelled()
        if progress and k % 10 == 0:
            progress(25 + int(70 * k / n))
        # кандидат совпадает с уже принятой (уточнённой) окружностью — пропускаем
        if any(abs(r - a.r) < 0.1 * a.r and math.hypot(cx - a.cx, cy - a.cy) < 0.1 * a.r for a in accepted):
            continue
        # полюсные фигуры не вложены друг в друга: кандидат внутри принятой фигуры с осями не проверяем
        # (кандидаты обрабатываются от больших радиусов к малым)
        if any(a.axes_cov >= 0.6 and math.hypot(cx - a.cx, cy - a.cy) + r <= a.r * 1.02 for a in accepted):
            continue
        ref = refine_circle(rgb, cx, cy, r, params.dark_threshold, dark_mask=dark_full)
        if ref is None:
            continue
        rcx, rcy, rr, rms, cov, thick = ref
        if not (min_r <= rr <= max_r * 1.05):
            continue
        if rms > max(1.5, 0.025 * rr) or cov < params.ring_coverage * 0.8:
            continue
        ok, f = evaluate_circle(rgb, rcx, rcy, rr, params, thick)
        if not ok:
            continue
        score = f["score"] - min(rms / rr, 0.2)
        accepted.append(DetectedCircle(rcx, rcy, rr, score, f["ring_cov"], f["axes_cov"],
                                       f["inside_bg"], thick))

    result = suppress(accepted)
    order = reading_order([(c.cx, c.cy, c.r) for c in result])
    if progress:
        progress(100)
    return [result[i] for i in order]


def suppress(circles: list[DetectedCircle]) -> list[DetectedCircle]:
    """Подавление дублей и вложенных окружностей."""
    kept: list[DetectedCircle] = []
    for c in sorted(circles, key=lambda c: -c.score):
        dup = False
        for k in kept:
            d = math.hypot(c.cx - k.cx, c.cy - k.cy)
            if d < 0.3 * max(c.r, k.r) and 0.75 < c.r / k.r < 1.33:
                dup = True
                break
        if not dup:
            kept.append(c)
    # вложенность: полюсные фигуры не содержат друг друга
    result = list(kept)
    changed = True
    while changed:
        changed = False
        for a in result:
            for b in result:
                if a is b:
                    continue
                outer, inner = (a, b) if a.r >= b.r else (b, a)
                d = math.hypot(a.cx - b.cx, a.cy - b.cy)
                if d + inner.r <= outer.r * 1.02:
                    if d < 0.1 * outer.r:
                        drop = inner          # концентрические — оставляем внешнюю
                    else:
                        drop = inner if inner.score <= outer.score else outer
                    result.remove(drop)
                    changed = True
                    break
            if changed:
                break
    return result
