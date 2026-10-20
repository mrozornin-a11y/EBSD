"""Генератор синтетических изображений полюсных фигур с известной «истиной» для тестов."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

PALETTE = [
    (230, 20, 20), (20, 180, 30), (30, 60, 230), (240, 140, 0), (200, 0, 200),
    (0, 190, 200), (250, 220, 0), (140, 70, 20),
]


@dataclass
class Cluster:
    x: float          # центроид в координатах изображения (непрерывных)
    y: float
    color: tuple
    radius: float     # 0 = одиночный пиксель
    fig_index: int


@dataclass
class SyntheticPF:
    cx: float
    cy: float
    r: float
    clusters: list = field(default_factory=list)


@dataclass
class SyntheticImage:
    rgb: np.ndarray
    figures: list
    distractors: list


def _draw_pf(img, cx, cy, r, thick, rng, n_clusters, fig_index, axes_on_top):
    clusters = []
    sh = 4  # субпиксельная точность cv2 (shift)
    s = 1 << sh

    def axes():
        cv2.line(img, (int((cx - r) * s), int(cy * s)), (int((cx + r) * s), int(cy * s)),
                 (0, 0, 0), max(1, thick // 2), cv2.LINE_AA, sh)
        cv2.line(img, (int(cx * s), int((cy - r) * s)), (int(cx * s), int((cy + r) * s)),
                 (0, 0, 0), max(1, thick // 2), cv2.LINE_AA, sh)

    if not axes_on_top:
        axes()
    placed = []
    tries = 0
    while len(clusters) < n_clusters and tries < 5000:
        tries += 1
        rad = rng.choice([0, 0, 1, 1.5, 2, 3, 4])
        a = rng.uniform(0, 2 * math.pi)
        d = math.sqrt(rng.uniform(0, 1)) * (r - thick * 2 - rad - 4)
        x = cx + d * math.cos(a)
        y = cy + d * math.sin(a)
        if rad == 0:
            x = math.floor(x) + 0.5
            y = math.floor(y) + 0.5
        # не перекрываться с другими кластерами
        if any(math.hypot(x - px, y - py) < pr + rad + 4 for px, py, pr in placed):
            continue
        # кластеры не ставим строго на оси (проверяется отдельно)
        if abs(x - cx) < rad + 3 or abs(y - cy) < rad + 3:
            continue
        color = PALETTE[rng.integers(len(PALETTE))]
        if rad == 0:
            img[int(y), int(x)] = color
        else:
            cv2.circle(img, (int(round(x * s)), int(round(y * s))), int(rad * s), color, -1, cv2.LINE_8, sh)
        placed.append((x, y, rad))
        # cv2 рисует с центром пикселя в целых координатах -> непрерывные координаты +0.5
        tx, ty = (x, y) if rad == 0 else (x + 0.5, y + 0.5)
        clusters.append(Cluster(tx, ty, tuple(int(c) for c in color), rad, fig_index))
    if axes_on_top:
        axes()
    cv2.circle(img, (int(round(cx * s)), int(round(cy * s))), int(round(r * s)), (0, 0, 0), thick, cv2.LINE_AA, sh)
    # пересчитываем фактические центроиды отрисованных кластеров по пикселям
    return clusters


def make_composite(seed: int = 0, jpeg_quality: int | None = None, n_rows: int = 2, n_cols: int = 4,
                   r: int = 70, thick: int = 2, n_clusters: int = 25, distractors: bool = True,
                   gap: int = 30) -> SyntheticImage:
    rng = np.random.default_rng(seed)
    margin = 60
    W = margin * 2 + n_cols * (2 * r + gap) + 260
    H = margin * 2 + n_rows * (2 * r + gap + 20)
    img = np.full((H, W, 3), 255, np.uint8)
    figs = []
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            cx = margin + r + j * (2 * r + gap) + rng.uniform(-3, 3)
            cy = margin + r + i * (2 * r + gap + 20) + rng.uniform(-3, 3)
            cl = _draw_pf(img, cx, cy, r, thick, rng, n_clusters, k, axes_on_top=(k % 2 == 1))
            figs.append(SyntheticPF(cx + 0.5, cy + 0.5, r, cl))
            k += 1
    dis = []
    if distractors:
        x0 = margin + n_cols * (2 * r + gap) + 20
        # 1) чёрный залитый круг
        cv2.circle(img, (x0 + 50, 80), 35, (0, 0, 0), -1, cv2.LINE_AA)
        dis.append(("filled", x0 + 50, 80, 35))
        # 2) окружность без осей
        cv2.circle(img, (x0 + 150, 90), 45, (0, 0, 0), 2, cv2.LINE_AA)
        dis.append(("no_axes", x0 + 150, 90, 45))
        # 3) крупная буква O
        cv2.putText(img, "O", (x0 + 10, 230), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 4, cv2.LINE_AA)
        dis.append(("letter", x0 + 45, 205, 30))
        # 4) цветное кольцо
        cv2.circle(img, (x0 + 170, 220), 40, (220, 30, 30), 3, cv2.LINE_AA)
        dis.append(("colored_ring", x0 + 170, 220, 40))
        # 5) эллипс с осями
        cv2.ellipse(img, (x0 + 120, 330), (60, 30), 0, 0, 360, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img, (x0 + 60, 330), (x0 + 180, 330), (0, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, (x0 + 120, 300), (x0 + 120, 360), (0, 0, 0), 1, cv2.LINE_AA)
        dis.append(("ellipse", x0 + 120, 330, 60))
        # 6) дуга (незамкнутая окружность)
        cv2.ellipse(img, (x0 + 120, 450), (50, 50), 0, 0, 200, (0, 0, 0), 2, cv2.LINE_AA)
        dis.append(("arc", x0 + 120, 450, 50))
        # подпись
        cv2.putText(img, "Fig. 3 (a) sample 1", (margin, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 1, cv2.LINE_AA)
    if jpeg_quality:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=jpeg_quality)
        img = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")).copy()
    return SyntheticImage(img, figs, dis)
