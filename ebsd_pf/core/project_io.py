"""Сохранение и загрузка проекта (собственный формат JSON, расширение .ebsdproj)."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil

from .. import __version__
from .imageio_utils import encode_png_b64
from .models import (DetectParams, ExportSettings, LayoutSettings, PoleFigure, Project, ReconParams,
                     SourceImage)

log = logging.getLogger(__name__)

FORMAT_ID = "EBSD-PoleFigure-Reconstructor-Project"
FORMAT_VERSION = 1


class ProjectError(Exception):
    pass


def project_to_dict(project: Project, project_path: str | None, store=None) -> dict:
    base = os.path.dirname(os.path.abspath(project_path)) if project_path else None
    images = []
    for img in project.images:
        d = img.to_dict()
        d["path"] = os.path.abspath(img.path) if img.path else ""
        d["rel_path"] = None
        if base and img.path:
            try:
                d["rel_path"] = os.path.relpath(os.path.abspath(img.path), base)
            except ValueError:  # другой диск в Windows
                d["rel_path"] = None
        if project.embed_images:
            if not d.get("embedded_png") and store is not None and store.get(img.id) is not None:
                d["embedded_png"] = encode_png_b64(store.get(img.id))
        else:
            d["embedded_png"] = None
        images.append(d)
    return {
        "format": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "app_version": __version__,
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "embed_images": project.embed_images,
        "images": images,
        "figures": [f.to_dict() for f in project.figures],
        "recon_defaults": project.recon.to_dict(),
        "layout": project.layout.to_dict(),
        "export": project.export.to_dict(),
        "detect": project.detect.__dict__.copy(),
        "ui": project.ui,
    }


def save_project(project: Project, path: str, store=None, keep_backup: bool = True) -> None:
    data = project_to_dict(project, path, store)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        if keep_backup and os.path.exists(path):
            try:
                shutil.copy2(path, path + ".bak")
            except OSError:
                pass
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise ProjectError(f"Не удалось сохранить проект: {e}") from e
    # встроенные данные сохраняем в модели, чтобы не кодировать повторно
    if project.embed_images:
        for img, d in zip(project.images, data["images"]):
            img.embedded_png = d.get("embedded_png")


def project_from_dict(data: dict, project_path: str | None) -> Project:
    if data.get("format") != FORMAT_ID:
        raise ProjectError("Файл не является проектом программы реконструкции полюсных фигур")
    if int(data.get("format_version", 1)) > FORMAT_VERSION:
        raise ProjectError("Проект создан более новой версией программы")
    base = os.path.dirname(os.path.abspath(project_path)) if project_path else None
    p = Project()
    for d in data.get("images", []):
        img = SourceImage.from_dict(d)
        img.path = resolve_image_path(d, base) or d.get("path", "")
        p.images.append(img)
    ids = {i.id for i in p.images}
    for d in data.get("figures", []):
        try:
            fig = PoleFigure.from_dict(d)
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Пропущена повреждённая запись фигуры: %s", e)
            continue
        if fig.image_id in ids:
            p.figures.append(fig)
    p.recon = ReconParams.from_dict(data.get("recon_defaults"))
    p.layout = LayoutSettings.from_dict(data.get("layout"))
    p.export = ExportSettings.from_dict(data.get("export"))
    p.detect = DetectParams.from_dict(data.get("detect"))
    p.embed_images = bool(data.get("embed_images", False))
    p.ui = dict(data.get("ui") or {})
    return p


def load_project(path: str) -> Project:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        raise ProjectError(f"Файл проекта повреждён: {e}") from e
    except OSError as e:
        raise ProjectError(f"Не удалось открыть проект: {e}") from e
    return project_from_dict(data, path)


def resolve_image_path(d: dict, base: str | None) -> str | None:
    """Ищет исходное изображение: относительный путь, абсолютный, то же имя рядом с проектом."""
    cands = []
    if base and d.get("rel_path"):
        cands.append(os.path.normpath(os.path.join(base, d["rel_path"])))
    if d.get("path"):
        cands.append(d["path"])
        if base:
            cands.append(os.path.join(base, os.path.basename(d["path"])))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None
