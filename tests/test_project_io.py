"""Сохранение и загрузка проекта."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from ebsd_pf.core.imageio_utils import load_image, load_image_b64
from ebsd_pf.core.models import MaskEdits, PoleFigure, Project, ReconParams, SourceImage
from ebsd_pf.core.pipeline import ImageStore
from ebsd_pf.core.project_io import ProjectError, load_project, save_project


def _project(tmp_path):
    rgb = np.full((60, 80, 3), 255, np.uint8)
    rgb[10:20, 10:20] = (200, 30, 30)
    img_path = tmp_path / "images" / "рисунок 1.png"
    img_path.parent.mkdir()
    Image.fromarray(rgb).save(img_path)
    pr = Project()
    pr.images.append(SourceImage("i1", str(img_path), img_path.name, 80, 60))
    edits = MaskEdits.empty(32)
    edits.add[3:6, 4:9] = True
    edits.remove[20, 20] = True
    f1 = PoleFigure("f1", "i1", 30.5, 25.25, 20.0, sample="S1", grain="2", plane="111", order=2,
                    params=ReconParams(mode="enhanced", dot_size=3.5), mask_edits=edits)
    f2 = PoleFigure("f2", "i1", 60, 30, 15, sample="S1", grain="2", plane="100")
    pr.figures += [f1, f2]
    pr.recon.color_threshold = 50
    pr.layout.spacing_mm = 7.5
    pr.export.formats = ["svg", "pdf"]
    pr.export.dpi = 1200
    pr.detect.require_axes = False
    pr.ui["page"] = 3
    return pr, rgb


def test_roundtrip(tmp_path):
    pr, _ = _project(tmp_path)
    path = str(tmp_path / "proj.ebsdproj")
    save_project(pr, path)
    p2 = load_project(path)
    assert [f.id for f in p2.figures] == ["f1", "f2"]
    f1 = p2.figure("f1")
    assert (f1.cx, f1.cy, f1.r, f1.sample, f1.grain, f1.plane, f1.order) == (30.5, 25.25, 20.0, "S1", "2", "111", 2)
    assert f1.params.mode == "enhanced" and f1.params.dot_size == 3.5
    assert np.array_equal(f1.mask_edits.add, pr.figures[0].mask_edits.add)
    assert np.array_equal(f1.mask_edits.remove, pr.figures[0].mask_edits.remove)
    assert p2.figure("f2").params is None
    assert p2.recon.color_threshold == 50 and p2.layout.spacing_mm == 7.5
    assert p2.export.formats == ["svg", "pdf"] and p2.export.dpi == 1200
    assert p2.detect.require_axes is False and p2.ui["page"] == 3


def test_relative_path_after_moving_project(tmp_path):
    pr, rgb = _project(tmp_path)
    path = tmp_path / "proj.ebsdproj"
    save_project(pr, str(path))
    moved = tmp_path / "moved"
    os.rename(tmp_path / "images", tmp_path / "images_tmp")
    moved.mkdir()
    os.rename(tmp_path / "images_tmp", moved / "images")
    os.rename(path, moved / "proj.ebsdproj")
    p2 = load_project(str(moved / "proj.ebsdproj"))
    assert os.path.isfile(p2.images[0].path)
    assert np.array_equal(load_image(p2.images[0].path), rgb)


def test_embedded_images(tmp_path):
    pr, rgb = _project(tmp_path)
    pr.embed_images = True
    store = ImageStore()
    store.set("i1", rgb)
    path = str(tmp_path / "emb.ebsdproj")
    save_project(pr, path, store)
    data = json.load(open(path, encoding="utf-8"))
    assert data["images"][0]["embedded_png"]
    p2 = load_project(path)
    assert np.array_equal(load_image_b64(p2.images[0].embedded_png), rgb)


def test_backup_and_bad_file(tmp_path):
    pr, _ = _project(tmp_path)
    path = str(tmp_path / "p.ebsdproj")
    save_project(pr, path)
    save_project(pr, path)
    assert os.path.exists(path + ".bak")
    bad = tmp_path / "bad.ebsdproj"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProjectError):
        load_project(str(bad))
    other = tmp_path / "other.ebsdproj"
    other.write_text('{"format": "something"}', encoding="utf-8")
    with pytest.raises(ProjectError):
        load_project(str(other))


def test_image_formats(tmp_path):
    rgb = (np.random.default_rng(0).random((40, 50, 3)) * 255).astype(np.uint8)
    for ext in ("png", "jpg", "tif", "tiff"):
        p = tmp_path / f"a.{ext}"
        Image.fromarray(rgb).save(p, quality=100) if ext == "jpg" else Image.fromarray(rgb).save(p)
        out = load_image(str(p))
        assert out.shape == (40, 50, 3) and out.dtype == np.uint8
    # 16-битный TIFF и RGBA
    g16 = (np.arange(40 * 50).reshape(40, 50) * 30).astype(np.uint16)
    Image.fromarray(g16).save(tmp_path / "g16.tif")
    assert load_image(str(tmp_path / "g16.tif")).shape == (40, 50, 3)
    rgba = np.zeros((10, 10, 4), np.uint8)
    Image.fromarray(rgba, "RGBA").save(tmp_path / "t.png")
    assert (load_image(str(tmp_path / "t.png")) == 255).all()     # прозрачность -> белый фон
