"""Загрузка изображений (JPG/PNG/TIF/TIFF) в RGB uint8 с поддержкой путей Unicode."""
from __future__ import annotations

import base64
import io
import logging
import os

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # большие составные рисунки статей

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
IMAGE_FILTER = "Изображения (*.jpg *.jpeg *.png *.tif *.tiff *.JPG *.JPEG *.PNG *.TIF *.TIFF)"


class ImageLoadError(Exception):
    pass


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Приводит массив любого типа/числа каналов к RGB uint8."""
    a = np.asarray(arr)
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    if a.dtype != np.uint8:
        src_dtype = a.dtype
        a = np.nan_to_num(a.astype(np.float64))
        hi_v = float(a.max()) if a.size else 0.0
        lo, hi = 0.0, 255.0
        if np.issubdtype(src_dtype, np.integer):
            # 8-, 12- или 16-битные данные: масштаб по разрядности, без растяжения контраста
            if hi_v > 4095:
                hi = float(np.iinfo(src_dtype).max)
            elif hi_v > 255:
                hi = 4095.0
        else:
            if hi_v <= 1.0:
                hi = 1.0
            else:
                lo, hi = float(a.min()), max(hi_v, float(a.min()) + 1.0)
        a = np.clip((a - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3:
        if a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        elif a.shape[2] == 2:  # серый + альфа
            g = a[..., 0].astype(np.float32)
            al = a[..., 1].astype(np.float32) / 255.0
            g = g * al + 255 * (1 - al)
            a = np.repeat(g.astype(np.uint8)[..., None], 3, axis=2)
        elif a.shape[2] >= 4:
            rgb = a[..., :3].astype(np.float32)
            al = a[..., 3:4].astype(np.float32) / 255.0
            a = (rgb * al + 255.0 * (1 - al) + 0.5).astype(np.uint8)  # композиция на белом
    else:
        raise ImageLoadError("Неподдерживаемая размерность изображения")
    return np.ascontiguousarray(a[..., :3])


def _pil_to_rgb(img: Image.Image) -> np.ndarray:
    try:
        img.seek(0)  # многостраничные TIFF: первая страница
    except Exception:
        pass
    if img.mode in ("RGB", "L", "RGBA", "LA"):
        return _to_uint8_rgb(np.asarray(img))
    if img.mode in ("P", "PA", "CMYK", "YCbCr", "LAB", "HSV", "1"):
        conv = "RGBA" if ("A" in img.mode or "transparency" in img.info) else "RGB"
        return _to_uint8_rgb(np.asarray(img.convert(conv)))
    # I;16, I, F и прочие
    return _to_uint8_rgb(np.asarray(img))


def load_image(path: str) -> np.ndarray:
    """Загружает изображение как RGB uint8 (H, W, 3)."""
    if not os.path.isfile(path):
        raise ImageLoadError(f"Файл не найден: {path}")
    try:
        with Image.open(path) as img:
            img.load()
            return _pil_to_rgb(img)
    except Exception as e_pil:
        log.warning("Pillow не смог открыть %s: %s; пробуем OpenCV", path, e_pil)
        try:
            import cv2
            data = np.fromfile(path, dtype=np.uint8)
            arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ImageLoadError("OpenCV не распознал формат")
            if arr.ndim == 3:
                if arr.shape[2] == 3:
                    arr = arr[..., ::-1]
                elif arr.shape[2] == 4:
                    arr = arr[..., [2, 1, 0, 3]]
            return _to_uint8_rgb(arr)
        except ImageLoadError:
            raise
        except Exception as e_cv:
            raise ImageLoadError(f"Не удалось открыть изображение «{os.path.basename(path)}»: {e_pil}; {e_cv}")


def load_image_b64(data: str) -> np.ndarray:
    with Image.open(io.BytesIO(base64.b64decode(data))) as img:
        img.load()
        return _pil_to_rgb(img)


def encode_png_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=False, compress_level=6)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_rgb(path: str, rgb: np.ndarray, dpi: int | None = None) -> None:
    """Сохраняет RGB-массив в PNG/TIFF/JPG с метаданными DPI (пути Unicode допустимы)."""
    ext = os.path.splitext(path)[1].lower()
    img = Image.fromarray(rgb)
    kw: dict = {}
    if dpi:
        kw["dpi"] = (int(dpi), int(dpi))
    if ext in (".tif", ".tiff"):
        kw["compression"] = "tiff_lzw"
        fmt = "TIFF"
    elif ext in (".jpg", ".jpeg"):
        kw["quality"] = 95
        fmt = "JPEG"
    else:
        kw["optimize"] = False
        fmt = "PNG"
    tmp = path + ".tmp"
    img.save(tmp, format=fmt, **kw)
    os.replace(tmp, path)
