"""Перенос точек на новую фигуру: нормированные координаты, центры кластеров, отсутствие потерь."""
import numpy as np
import pytest

from ebsd_pf.core.geometry import crop_circle
from ebsd_pf.core.layer import layer_blobs, render_layer
from ebsd_pf.core.models import ReconParams
from ebsd_pf.core.segmentation import segment
from synthetic import make_composite


@pytest.fixture(scope="module")
def seg_sparse():
    s = make_composite(seed=3, n_clusters=8, n_rows=1, n_cols=1, distractors=False)
    pf = s.figures[0]
    return segment(crop_circle(s.rgb, pf.cx, pf.cy, pf.r), ReconParams())


def _norm_errors(seg, layer, R):
    n, lab, st, ce = layer_blobs(layer)
    src = (seg.centroids - [seg.crop.cx, seg.crop.cy]) / seg.crop.r
    dst = (ce[1:] + 0.5 - layer.shape[0] / 2) / R
    return n - 1, [float(np.min(np.hypot(*(dst - s_).T))) for s_ in src]


@pytest.mark.parametrize("mode,dot,scale,R", [
    ("exact", 1, 1, 400), ("exact", 1, 1, 50), ("enhanced", 3, 1, 400),
    ("enhanced", 1, 2.5, 400), ("enhanced", 2, 2, 300),
])
def test_cluster_centres_preserved(seg_sparse, mode, dot, scale, R):
    p = ReconParams(mode=mode, dot_size=dot, cluster_scale=scale)
    seg = segment(seg_sparse.crop, p)
    layer = render_layer(seg, p, R)
    n_out, errs = _norm_errors(seg, layer, R)
    assert n_out == seg.n_blobs                 # ни один кластер не потерян и не добавлен
    assert max(errs) * R < 0.75                 # смещение центра < 0.75 пикселя результата


def test_downscaling_never_loses_points():
    s = make_composite(seed=3, n_clusters=30, n_rows=1, n_cols=1, distractors=False)
    pf = s.figures[0]
    seg = segment(crop_circle(s.rgb, pf.cx, pf.cy, pf.r), ReconParams())
    for R in (35, 50, 69.5):
        layer = render_layer(seg, ReconParams(), R)
        n, lab, st, ce = layer_blobs(layer)
        dst = (ce[1:] + 0.5 - layer.shape[0] / 2) / R
        src = (seg.centroids - [seg.crop.cx, seg.crop.cy]) / seg.crop.r
        for s_ in src:
            assert np.min(np.hypot(*(dst - s_).T)) * R < 1.5


def test_exact_layer_uses_only_source_colors(seg_sparse):
    layer = render_layer(seg_sparse, ReconParams(), 333)
    out = {tuple(c) for c in layer[layer[..., 3] > 0][:, :3]}
    src = {tuple(c) for c in seg_sparse.colors[seg_sparse.mask]}
    assert out <= src


def test_dot_growth_uses_nearest_source_color(seg_sparse):
    p = ReconParams(mode="enhanced", dot_size=4)
    layer = render_layer(seg_sparse, p, 300)
    out = {tuple(c) for c in layer[layer[..., 3] > 0][:, :3]}
    src = {tuple(c) for c in seg_sparse.colors[seg_sparse.mask]}
    assert out <= src                           # увеличение не порождает новых цветов


def test_layer_clipped_to_disk(seg_sparse):
    p = ReconParams(mode="enhanced", dot_size=8, cluster_scale=4)
    R = 200
    layer = render_layer(seg_sparse, p, R)
    S = layer.shape[0]
    yy, xx = np.mgrid[0:S, 0:S]
    d = np.hypot(xx + 0.5 - S / 2, yy + 0.5 - S / 2)
    assert not (layer[..., 3][d > R] > 0).any()


def test_saturation_and_contrast_change_colors_not_positions(seg_sparse):
    p0 = ReconParams(mode="enhanced")
    p1 = ReconParams(mode="enhanced", saturation=2.0, contrast=1.5)
    a = render_layer(seg_sparse, p0, 250)
    b = render_layer(seg_sparse, p1, 250)
    assert np.array_equal(a[..., 3] > 0, b[..., 3] > 0)
    assert not np.array_equal(a[..., :3], b[..., :3])
