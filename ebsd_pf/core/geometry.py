"""Геометрические утилиты: подгонка окружности, вырезка области, классификация пикселей."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def chroma_value(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Хроматичность (max-min) и яркость (max) в шкале 0..255, int16."""
    a = rgb.astype(np.int16)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    return mx - mn, mx


def dark_mask(rgb: np.ndarray, dark_threshold: int = 140, chroma_limit: int = 90) -> np.ndarray:
    """Пиксели тёмных (чёрных/серых) линий."""
    ch, val = chroma_value(rgb)
    return (val < dark_threshold) & (ch < chroma_limit)


def background_mask(rgb: np.ndarray, value_min: int = 200, chroma_max: int = 30) -> np.ndarray:
    ch, val = chroma_value(rgb)
    return (val >= value_min) & (ch <= chroma_max)


def fit_circle(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Алгебраическая подгонка окружности (метод Kåsa, МНК)."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mx, my = x.mean(), y.mean()
    u, v = x - mx, y - my
    A = np.column_stack([u, v, np.ones_like(u)])
    b = u * u + v * v
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    return float(cx + mx), float(cy + my), float(r)


def robust_fit_circle(xs, ys, iterations: int = 4, k: float = 2.5, min_tol: float = 0.75):
    """Подгонка окружности с итеративным отбрасыванием выбросов. Возвращает (cx, cy, r, rms, inliers)."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    keep = np.ones(len(x), dtype=bool)
    cx = cy = r = 0.0
    for _ in range(iterations):
        if keep.sum() < 5:
            break
        cx, cy, r = fit_circle(x[keep], y[keep])
        res = np.abs(np.hypot(x - cx, y - cy) - r)
        sd = float(np.sqrt(np.mean(res[keep] ** 2)))
        new_keep = res <= max(k * sd, min_tol)
        if new_keep.sum() < 5 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 5 else keep
            break
        keep = new_keep
    if keep.sum() >= 3:
        cx, cy, r = fit_circle(x[keep], y[keep])
        res = np.abs(np.hypot(x[keep] - cx, y[keep] - cy) - r)
        rms = float(np.sqrt(np.mean(res ** 2)))
    else:
        rms = float("inf")
    return cx, cy, r, rms, keep


@dataclass
class Crop:
    """Квадратная вырезка вокруг окружности.

    Координаты непрерывные: пиксель (col, row) занимает [col, col+1) × [row, row+1).
    cx, cy — центр окружности в координатах вырезки.
    """
    rgb: np.ndarray
    x0: int
    y0: int
    cx: float
    cy: float
    r: float

    @property
    def size(self) -> int:
        return self.rgb.shape[0]


CROP_EXTENT = 1.12  # половина стороны вырезки в радиусах


def crop_circle(image: np.ndarray, cx: float, cy: float, r: float, extent: float = CROP_EXTENT) -> Crop:
    """Вырезает квадрат вокруг окружности; за пределами изображения — белый фон."""
    h, w = image.shape[:2]
    half = int(math.ceil(r * extent)) + 2
    x0 = int(math.floor(cx)) - half
    y0 = int(math.floor(cy)) - half
    size = 2 * half + 1
    out = np.full((size, size, 3), 255, dtype=np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, w), min(y0 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return Crop(rgb=out, x0=x0, y0=y0, cx=cx - x0, cy=cy - y0, r=r)


def pixel_grid(h: int, w: int, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Смещения центров пикселей от (cx, cy)."""
    ys = np.arange(h, dtype=np.float64) + 0.5 - cy
    xs = np.arange(w, dtype=np.float64) + 0.5 - cx
    return np.meshgrid(xs, ys)


def sample_bool(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, default: bool = False) -> np.ndarray:
    """Значения маски в точках с непрерывными координатами (ближайший пиксель)."""
    h, w = mask.shape
    ix = np.floor(xs).astype(np.int64)
    iy = np.floor(ys).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.full(ix.shape, default, dtype=bool)
    out[inside] = mask[iy[inside], ix[inside]]
    return out


def reading_order(points: list[tuple[float, float, float]]) -> list[int]:
    """Порядок «слева направо, сверху вниз» для окружностей (x, y, r) с допуском по строкам."""
    if not points:
        return []
    idx = sorted(range(len(points)), key=lambda i: points[i][1])
    rows: list[list[int]] = []
    for i in idx:
        x, y, r = points[i]
        if rows:
            last = rows[-1]
            ry = np.mean([points[j][1] for j in last])
            rr = np.mean([points[j][2] for j in last])
            if abs(y - ry) <= 0.5 * max(rr, r):
                last.append(i)
                continue
        rows.append([i])
    order: list[int] = []
    for row in rows:
        order.extend(sorted(row, key=lambda i: points[i][0]))
    return order


def cluster_by_proximity(points: list[tuple[float, float, float]], factor: float = 1.35) -> list[list[int]]:
    """Объединяет соседние окружности в группы (зерна): связь, если расстояние между центрами
    меньше factor·(r1 + r2). Возвращает группы в порядке чтения."""
    n = len(points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = points[i]
            xj, yj, rj = points[j]
            if math.hypot(xi - xj, yi - yj) < factor * (ri + rj):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    glist = list(groups.values())
    # порядок групп — по их центрам
    centers = []
    for g in glist:
        xs = [points[i][0] for i in g]
        ys = [points[i][1] for i in g]
        rs = [points[i][2] for i in g]
        centers.append((float(np.mean(xs)), float(min(ys)), float(np.mean(rs)) * 2))
    order = reading_order(centers)
    result = []
    for gi in order:
        g = glist[gi]
        sub = reading_order([points[i] for i in g])
        result.append([g[k] for k in sub])
    return result
