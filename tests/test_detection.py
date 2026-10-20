"""Поиск окружностей: точность, отбраковка посторонних округлых элементов, сжатие JPEG."""
import math

import numpy as np
import pytest

from ebsd_pf.core.detection import detect_circles, refine_circle
from ebsd_pf.core.models import DetectParams
from synthetic import make_composite


def _match(found, truth, tol):
    errs = []
    for f in truth:
        d = [math.hypot(c.cx - f.cx, c.cy - f.cy) for c in found]
        j = int(np.argmin(d))
        errs.append((d[j], abs(found[j].r - f.r)))
    return errs


@pytest.mark.parametrize("quality", [None, 85, 60])
def test_finds_all_and_rejects_distractors(quality):
    s = make_composite(seed=1, jpeg_quality=quality)
    found = detect_circles(s.rgb, DetectParams())
    assert len(found) == len(s.figures)
    for dc, dr in _match(found, s.figures, 1.0):
        assert dc < 0.5 and dr < 0.5
    # ни одна найденная окружность не совпадает с отвлекающим элементом
    for kind, x, y, r in s.distractors:
        assert all(math.hypot(c.cx - x, c.cy - y) > 0.5 * r for c in found), kind


def test_without_axes_requirement_accepts_plain_circle():
    s = make_composite(seed=2)
    found = detect_circles(s.rgb, DetectParams(require_axes=False))
    plain = [d for d in s.distractors if d[0] == "no_axes"][0]
    assert any(math.hypot(c.cx - plain[1], c.cy - plain[2]) < 3 for c in found)
    # залитый круг, буква, эллипс и дуга всё равно отбрасываются
    for kind, x, y, r in s.distractors:
        if kind in ("filled", "letter", "ellipse", "arc", "colored_ring"):
            assert all(math.hypot(c.cx - x, c.cy - y) > 0.5 * r for c in found), kind


def test_small_circles_in_large_image():
    s = make_composite(seed=12, n_rows=2, n_cols=6, r=28, thick=1, n_clusters=6, gap=14, distractors=False)
    big = np.full((4000, 6000, 3), 255, np.uint8)
    big[100:100 + s.rgb.shape[0], 200:200 + s.rgb.shape[1]] = s.rgb
    found = detect_circles(big)
    assert len(found) == len(s.figures)


def test_refine_from_rough_manual_circle():
    s = make_composite(seed=3, distractors=False)
    f = s.figures[0]
    ref = refine_circle(s.rgb, f.cx + 4, f.cy - 3, f.r * 1.07)
    assert ref is not None
    cx, cy, r, rms, cov, thick = ref
    assert math.hypot(cx - f.cx, cy - f.cy) < 0.5 and abs(r - f.r) < 0.5 and cov > 0.8


def test_empty_image_returns_nothing():
    img = np.full((400, 600, 3), 255, np.uint8)
    assert detect_circles(img) == []
