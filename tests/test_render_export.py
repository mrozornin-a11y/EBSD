"""Компоновка, подписи и экспорт PNG/TIFF/SVG/PDF."""
import math
import os
import re

import pytest
from PIL import Image

from ebsd_pf.core import exporter
from ebsd_pf.core.detection import detect_circles
from ebsd_pf.core.models import (ExportSettings, LayoutSettings, PoleFigure, Project, ReconParams,
                                 SourceImage, format_plane, new_id, normalize_plane)
from ebsd_pf.core.pipeline import ImageStore, Pipeline
from ebsd_pf.core.render import label_geometry, layout_grain
from synthetic import make_composite


@pytest.fixture()
def project(qapp):
    s = make_composite(seed=4, jpeg_quality=90)
    store = ImageStore()
    store.set("img", s.rgb)
    pr = Project()
    pr.images.append(SourceImage("img", "x.png", "x.png", s.rgb.shape[1], s.rgb.shape[0]))
    planes = ["100", "110", "111", "112"]
    for i, c in enumerate(detect_circles(s.rgb)):
        pr.figures.append(PoleFigure(new_id(), "img", c.cx, c.cy, c.r, sample="S1",
                                     grain=str(1 + i // 4), plane=planes[i % 4], order=i % 4))
    return pr, Pipeline(store)


def test_plane_formatting():
    assert format_plane("111") == "{111}"
    assert format_plane("{110}") == "{110}"
    assert normalize_plane(" {1 1 2} ") == "112"
    assert format_plane("") == ""
    assert format_plane("-1 1 0") == "{-110}"


@pytest.mark.parametrize("pos", ["top_right", "top_center", "top_left"])
@pytest.mark.parametrize("dy", [0.0, 15.0])
def test_label_never_overlaps_disk(qapp, pos, dy):
    p = ReconParams(label_position=pos, label_size_pt=14, label_dy_mm=dy)
    r = 20.0
    rect, origin = label_geometry(r, "{111}", p)
    nx = min(max(0.0, rect.left()), rect.right())
    ny = min(max(0.0, rect.top()), rect.bottom())
    assert math.hypot(nx, ny) >= r + p.circle_width_pt * 25.4 / 72 / 2 - 1e-6
    assert rect.bottom() < 0          # подпись над центром (в верхней части)


def test_grid_equal_spacing_and_diameter(qapp):
    ls = LayoutSettings(diameter_mm=30, spacing_mm=4, columns=2)
    entries = [(str(i), "{%d}" % i, ReconParams()) for i in range(4)]
    sheet = layout_grain(entries, ls)
    xs = sorted({round(f.cx, 6) for f in sheet.figures})
    ys = sorted({round(f.cy, 6) for f in sheet.figures})
    assert len(xs) == 2 and len(ys) == 2
    assert all(f.r == 15 for f in sheet.figures)
    gap_x = xs[1] - xs[0] - 30
    gap_y = ys[1] - ys[0] - 30
    assert gap_x >= 4 and gap_y >= 4
    # подписи не выходят за лист
    for f in sheet.figures:
        assert f.cx + f.label_rect.right() <= sheet.width + 1e-6
        assert f.cy + f.label_rect.top() >= 0


@pytest.mark.parametrize("n", [1, 3, 5, 6])
def test_non_standard_group_sizes(qapp, n):
    sheet = layout_grain([(str(i), "{1}", ReconParams()) for i in range(n)], LayoutSettings(columns=0))
    assert len(sheet.figures) == n
    assert len({(round(f.cx, 3), round(f.cy, 3)) for f in sheet.figures}) == n


def test_export_all_formats(project, tmp_path):
    pr, pl = project
    es = ExportSettings(output_dir=str(tmp_path), formats=["png", "tiff", "svg", "pdf"], dpi=300)
    paths = exporter.run_export(pr, pl, exporter.ExportJob("all"), es)
    assert len(paths) == 2 * 4
    for p in paths:
        assert os.path.getsize(p) > 1000
    png = [p for p in paths if p.endswith(".png")][0]
    with Image.open(png) as im:
        assert round(im.info["dpi"][0]) == 300 and im.mode == "RGB"
        assert im.getpixel((2, 2)) == (255, 255, 255)
    tif = [p for p in paths if p.endswith(".tif")][0]
    with Image.open(tif) as im:
        assert round(float(im.info["dpi"][0])) == 300
    svg = open([p for p in paths if p.endswith(".svg")][0], encoding="utf-8").read()
    assert "{100}" in svg and "{112}" in svg and "<image" in svg
    m = re.search(r'<svg width="([\d.]+)mm" height="([\d.]+)mm"\s+viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    w_mm, h_mm, vw, vh = map(float, m.groups())
    assert abs(vw / w_mm - vh / h_mm) < 1e-3          # одинаковый масштаб по осям
    pdf = open([p for p in paths if p.endswith(".pdf")][0], "rb").read()
    assert pdf.startswith(b"%PDF")


@pytest.mark.parametrize("dpi", [300, 600, 1200])
def test_raster_size_matches_dpi(project, tmp_path, dpi):
    pr, pl = project
    es = ExportSettings(output_dir=str(tmp_path), formats=["png"], dpi=dpi)
    f = pr.figures[0]
    paths = exporter.run_export(pr, pl, exporter.ExportJob("figure", fig_id=f.id), es)
    sheet = exporter.figure_sheet(pr, f)
    with Image.open(paths[0]) as im:
        assert im.size == exporter.raster_size(sheet, dpi)


def test_vector_points_and_sample_sheet(project, tmp_path):
    pr, pl = project
    es = ExportSettings(output_dir=str(tmp_path), formats=["svg", "pdf"], vector_points=True,
                        combine_sample=True)
    paths = exporter.run_export(pr, pl, exporter.ExportJob("sample", sample="S1"), es)
    names = {os.path.basename(p) for p in paths}
    assert "S1_all_grains.svg" in names and "S1_grain1.pdf" in names
    svg = open([p for p in paths if p.endswith("S1_grain1.svg")][0], encoding="utf-8").read()
    assert "<image" not in svg and "<path" in svg      # точки — векторные контуры
