"""Выделение цветных кластеров: полнота, отсутствие ложных точек, сохранение координат, отчётность."""
import math

import cv2
import numpy as np
import pytest

from ebsd_pf.core.geometry import crop_circle
from ebsd_pf.core.models import MaskEdits, ReconParams
from ebsd_pf.core.segmentation import crop_to_edit_index, segment
from synthetic import make_composite


def _evaluate(s, params, tol=1.5):
    miss = false = 0
    errs = []
    for pf in s.figures:
        crop = crop_circle(s.rgb, pf.cx, pf.cy, pf.r)
        seg = segment(crop, params)
        cents = seg.centroids + [crop.x0, crop.y0]
        used = set()
        for cl in pf.clusters:
            d = np.hypot(cents[:, 0] - cl.x, cents[:, 1] - cl.y) if len(cents) else np.array([1e9])
            j = int(np.argmin(d))
            if d[j] < tol:
                used.add(j)
                errs.append(d[j])
            else:
                miss += 1
        false += len(cents) - len(used)
    return miss, false, errs


def test_clean_image_exact_recovery():
    s = make_composite(seed=2)
    miss, false, errs = _evaluate(s, ReconParams())
    assert miss == 0 and false == 0
    assert max(errs) < 0.8          # центроиды кластеров — субпиксельная точность
    assert np.mean(errs) < 0.3


def test_jpeg_q90_no_false_points():
    s = make_composite(seed=2, jpeg_quality=90)
    miss, false, _ = _evaluate(s, ReconParams())
    total = sum(len(f.clusters) for f in s.figures)
    assert false == 0
    assert miss <= 0.03 * total     # теряются лишь точки, которые JPEG сделал почти белыми


def test_colors_are_taken_from_source_in_exact_mode():
    s = make_composite(seed=5)
    pf = s.figures[0]
    crop = crop_circle(s.rgb, pf.cx, pf.cy, pf.r)
    seg = segment(crop, ReconParams())
    assert np.array_equal(seg.colors[seg.mask], crop.rgb[seg.mask])


def test_axes_and_circle_not_detected_as_points():
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(img, (150, 150), 120, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.line(img, (30, 150), (270, 150), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, (150, 30), (150, 270), (0, 0, 0), 1, cv2.LINE_AA)
    crop = crop_circle(img, 150.5, 150.5, 120)
    seg = segment(crop, ReconParams())
    assert seg.n_pixels == 0
    assert seg.lines.h_axis is not None and seg.lines.v_axis is not None and seg.lines.ring_found


def test_restore_cluster_hidden_under_axis():
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(img, (200, 150), 6, (220, 30, 30), -1)          # кластер на оси
    cv2.circle(img, (150, 150), 120, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (30, 150), (270, 150), (0, 0, 0), 2)          # ось поверх кластера
    cv2.line(img, (150, 30), (150, 270), (0, 0, 0), 2)
    crop = crop_circle(img, 150.5, 150.5, 120)
    on = segment(crop, ReconParams(restore_under_axes=True))
    off = segment(crop, ReconParams(restore_under_axes=False))
    assert off.n_blobs == 2 and on.n_blobs == 1
    assert on.restored_mask.sum() > 0
    cx, cy = on.centroids[0] + [crop.x0, crop.y0]
    assert math.hypot(cx - 200.5, cy - 150.5) < 0.6
    # восстановленные пиксели имеют цвет кластера
    assert (on.colors[on.restored_mask][:, 0] > 150).all()


def test_different_clusters_across_axis_not_merged():
    img = np.full((300, 300, 3), 255, np.uint8)
    img[140:150, 195:206] = (220, 30, 30)       # красный кластер над осью
    img[151:161, 195:206] = (30, 40, 220)       # синий кластер под осью
    cv2.circle(img, (150, 150), 120, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (30, 150), (270, 150), (0, 0, 0), 2)
    cv2.line(img, (150, 30), (150, 270), (0, 0, 0), 2)
    crop = crop_circle(img, 150.5, 150.5, 120)
    seg = segment(crop, ReconParams(restore_under_axes=True))
    assert seg.n_blobs == 2 and not seg.restored_mask.any()


def test_noise_removal_is_reported_and_protects_strong_points():
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.circle(img, (150, 150), 120, (0, 0, 0), 2, cv2.LINE_AA)
    img[100, 100] = (200, 180, 180)      # бледный одиночный пиксель (шум)
    img[200, 120] = (230, 10, 10)        # насыщенная одиночная точка (данные)
    cv2.circle(img, (110, 190), 4, (20, 160, 20), -1)
    crop = crop_circle(img, 150.5, 150.5, 120)
    seg = segment(crop, ReconParams(min_object_size=3, protect_saturated=True))
    assert len(seg.removed) == 1                      # удалён только бледный пиксель
    rx, ry, area = seg.removed[0]
    assert (int(rx + crop.x0), int(ry + crop.y0)) == (100, 100) and area == 1
    assert seg.removed_mask.sum() == 1
    assert seg.n_blobs == 2
    seg2 = segment(crop, ReconParams(min_object_size=3, protect_saturated=False))
    assert len(seg2.removed) == 2 and "Удалено как шум: 2" in seg2.stats_text()


def test_manual_mask_edits():
    s = make_composite(seed=3, n_rows=1, n_cols=1, distractors=False, n_clusters=10)
    pf = s.figures[0]
    crop = crop_circle(s.rgb, pf.cx, pf.cy, pf.r)
    base = segment(crop, ReconParams())
    n = 300
    edits = MaskEdits.empty(n)
    # исключаем кластер, ближайший к первому истинному, — ровно его пиксели
    cl = pf.clusters[0]
    cents = base.centroids + [crop.x0, crop.y0]
    j = int(np.argmin(np.hypot(cents[:, 0] - cl.x, cents[:, 1] - cl.y)))
    ys, xs = np.nonzero(base.labels == j + 1)
    row, col = crop_to_edit_index(crop, n, xs, ys)
    edits.remove[row, col] = True
    # «добавление» на белом фоне не создаёт точек
    edits.add[10:40, 90:110] = True
    seg = segment(crop, ReconParams(), edits)
    assert seg.n_blobs == base.n_blobs - 1
    assert seg.manual_remove > 0 and seg.manual_add == 0
    # правки переживают ресемплинг
    seg_r = segment(crop, ReconParams(), edits.resampled(600))
    assert seg_r.n_blobs == base.n_blobs - 1


@pytest.mark.parametrize("mode", ["original", "smooth", "cluster"])
def test_color_modes_keep_mask(mode):
    s = make_composite(seed=4, n_rows=1, n_cols=1, distractors=False, jpeg_quality=85)
    pf = s.figures[0]
    crop = crop_circle(s.rgb, pf.cx, pf.cy, pf.r)
    a = segment(crop, ReconParams(mode="enhanced", color_mode="original"))
    b = segment(crop, ReconParams(mode="enhanced", color_mode=mode))
    assert np.array_equal(a.mask, b.mask)        # режим цвета не меняет состав точек
