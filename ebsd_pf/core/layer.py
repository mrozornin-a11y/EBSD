"""Перенос цветных пикселей и кластеров на новую фигуру (растровый слой RGBA).

Координаты сохраняются в нормированной системе окружности: (x − cx)/r.
Увеличение кластеров выполняется масштабированием каждого кластера вокруг его
собственного центроида; увеличение «пикселей» — симметричной дилатацией каждого
пикселя. Оба преобразования не смещают центры кластеров.
"""
from __future__ import annotations

import cv2
import numpy as np

from .models import ReconParams
from .segmentation import SegResult


def _premultiplied_rgba(seg: SegResult) -> np.ndarray:
    a = seg.mask.astype(np.float32)
    rgba = np.zeros(seg.mask.shape + (4,), dtype=np.float32)
    rgba[..., :3] = seg.colors.astype(np.float32) * a[..., None]
    rgba[..., 3] = a * 255.0
    return rgba


def _unpremultiply(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4]
    rgb = np.where(a > 0, rgba[..., :3] * 255.0 / np.maximum(a, 1e-6), 0)
    out = np.empty(rgba.shape, dtype=np.uint8)
    out[..., :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(a[..., 0] + 0.5, 0, 255).astype(np.uint8)
    return out


def _interp(scale: float, smooth: bool) -> int:
    if not smooth:
        return cv2.INTER_NEAREST
    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR


def adjust_colors(rgb: np.ndarray, contrast: float, saturation: float) -> np.ndarray:
    """Насыщенность (HSV) и контраст относительно белого фона."""
    out = rgb
    if abs(saturation - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL).reshape(rgb.shape)
    if abs(contrast - 1.0) > 1e-3:
        f = out.astype(np.float32)
        out = np.clip(255.0 - (255.0 - f) * contrast + 0.5, 0, 255).astype(np.uint8)
    return out


def _forward_splat(seg: SegResult, f: float, k: float, half: float, S: int) -> np.ndarray:
    """Уменьшение без потери точек: каждый исходный пиксель переносится в ближайший
    выходной пиксель (обратная выборка при уменьшении могла бы пропустить одиночные точки)."""
    crop = seg.crop
    ys, xs = np.nonzero(seg.mask)
    xc = xs + 0.5
    yc = ys + 0.5
    if k != 1.0:
        lab = seg.labels[ys, xs] - 1
        cen = seg.centroids[lab]
        xc = (xc - cen[:, 0]) * k + cen[:, 0]
        yc = (yc - cen[:, 1]) * k + cen[:, 1]
        order = np.argsort(-seg.areas[lab], kind="stable")   # мелкие кластеры поверх
    else:
        order = np.arange(len(xs))
    X = np.floor((xc - crop.cx) * f + half).astype(np.int64)[order]
    Y = np.floor((yc - crop.cy) * f + half).astype(np.int64)[order]
    ok = (X >= 0) & (X < S) & (Y >= 0) & (Y < S)
    layer = np.zeros((S, S, 4), dtype=np.float32)
    cols = seg.colors[ys, xs][order].astype(np.float32)
    layer[Y[ok], X[ok], :3] = cols[ok]
    layer[Y[ok], X[ok], 3] = 255.0
    return layer


def render_layer(seg: SegResult, params: ReconParams, out_radius_px: float) -> np.ndarray:
    """Возвращает RGBA uint8 размера S×S (S≈2R), центр окружности — центр массива."""
    p = params.effective()
    R = float(out_radius_px)
    S = max(2, int(round(2 * R)))
    half = S / 2.0
    crop = seg.crop
    f = R / crop.r
    layer = np.zeros((S, S, 4), dtype=np.float32)
    if seg.mask.any():
        src = _premultiplied_rgba(seg)
        k = max(1.0, float(p.cluster_scale))
        interp = _interp(f * k, p.smooth_scaling)
        if f * k < 1.0 and not p.smooth_scaling:
            layer = _forward_splat(seg, f, k, half, S)
        elif k == 1.0:
            # dst_idx = f*src_idx + f*(0.5 - cx) + half - 0.5
            M = np.array([[f, 0, f * (0.5 - crop.cx) + half - 0.5],
                          [0, f, f * (0.5 - crop.cy) + half - 0.5]], dtype=np.float64)
            layer = cv2.warpAffine(src, M, (S, S), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
        else:
            fk = f * k
            order = np.argsort(-seg.areas)  # крупные снизу, мелкие поверх
            for i in order:
                x, y, w, h = (int(v) for v in seg.bboxes[i])
                cxb, cyb = seg.centroids[i]
                lab = seg.labels[y:y + h, x:x + w] == (i + 1)
                patch = src[y:y + h, x:x + w] * lab[..., None]
                # непрерывные координаты: X = fk*x_c + T
                Tx = (cxb * (1 - k) - crop.cx) * f + half
                Ty = (cyb * (1 - k) - crop.cy) * f + half
                X0 = int(np.floor(fk * x + Tx)) - 1
                Y0 = int(np.floor(fk * y + Ty)) - 1
                X1 = int(np.ceil(fk * (x + w) + Tx)) + 1
                Y1 = int(np.ceil(fk * (y + h) + Ty)) + 1
                cx0, cy0 = max(X0, 0), max(Y0, 0)
                cx1, cy1 = min(X1, S), min(Y1, S)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                M = np.array([[fk, 0, fk * (x + 0.5) + Tx - 0.5 - cx0],
                              [0, fk, fk * (y + 0.5) + Ty - 0.5 - cy0]], dtype=np.float64)
                warped = cv2.warpAffine(patch, M, (cx1 - cx0, cy1 - cy0), flags=interp,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                dst = layer[cy0:cy1, cx0:cx1]
                a = warped[..., 3:4] / 255.0
                dst[...] = warped + dst * (1.0 - a)   # «поверх», премультиплицированные цвета
    out = _unpremultiply(layer)

    # --- увеличение размера пикселей (симметричная дилатация)
    grow = (max(1.0, float(p.dot_size)) - 1.0) * f / 2.0
    if grow >= 0.5 and (out[..., 3] > 0).any():
        g = int(round(grow))
        A = out[..., 3] >= 128
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        Ad = cv2.dilate(A.astype(np.uint8), se).astype(bool)
        new = Ad & ~A
        if new.any():
            # цвет нового пикселя — цвет ближайшего исходного пикселя
            _, lbl = cv2.distanceTransformWithLabels((~A).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            src_idx = np.flatnonzero(A.ravel())
            flat = out.reshape(-1, 4)
            take = src_idx[lbl[new] - 1]
            newflat = np.flatnonzero(new.ravel())
            flat[newflat, :3] = flat[take, :3]
            flat[newflat, 3] = 255
        out[A, 3] = 255

    # --- цвет: насыщенность и контраст
    if abs(p.saturation - 1) > 1e-3 or abs(p.contrast - 1) > 1e-3:
        sel = out[..., 3] > 0
        if sel.any():
            out[sel, :3] = adjust_colors(out[sel, :3][None, ...], p.contrast, p.saturation)[0]

    # --- ограничение диском
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    d = np.hypot(xx + 0.5 - half, yy + 0.5 - half)
    out[d > R] = 0
    return out


def layer_blobs(rgba: np.ndarray):
    """Связные объекты слоя: (n, labels, stats, centroids) — для контроля и векторизации."""
    A = (rgba[..., 3] >= 128).astype(np.uint8)
    return cv2.connectedComponentsWithStats(A, connectivity=8)
