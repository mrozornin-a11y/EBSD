"""Хранилище изображений и конвейер обработки фигур с кэшированием (потокобезопасно)."""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

from .geometry import Crop, crop_circle
from .layer import render_layer
from .models import PoleFigure, ReconParams
from .segmentation import SegResult, segment


class ImageStore:
    """Пиксельные данные исходных изображений (RGB uint8), доступные из рабочих потоков."""

    def __init__(self):
        self._images: dict[str, np.ndarray] = {}
        self._versions: dict[str, int] = {}
        self._lock = threading.Lock()

    def set(self, image_id: str, rgb: np.ndarray) -> None:
        rgb.setflags(write=False)
        with self._lock:
            self._images[image_id] = rgb
            self._versions[image_id] = self._versions.get(image_id, 0) + 1

    def get(self, image_id: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._images.get(image_id)

    def version(self, image_id: str) -> int:
        with self._lock:
            return self._versions.get(image_id, 0)

    def remove(self, image_id: str) -> None:
        with self._lock:
            self._images.pop(image_id, None)

    def clear(self) -> None:
        with self._lock:
            self._images.clear()

    def __contains__(self, image_id: str) -> bool:
        with self._lock:
            return image_id in self._images


class _LRU:
    def __init__(self, max_items: int, max_bytes: int):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _size(v) -> int:
        if isinstance(v, np.ndarray):
            return v.nbytes
        if isinstance(v, SegResult):
            return v.crop.rgb.nbytes * 4
        return 1024

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, value):
        with self._lock:
            if key in self._d:
                self._bytes -= self._size(self._d.pop(key))
            self._d[key] = value
            self._bytes += self._size(value)
            while self._d and (len(self._d) > self.max_items or self._bytes > self.max_bytes):
                _, v = self._d.popitem(last=False)
                self._bytes -= self._size(v)

    def drop(self, pred) -> None:
        with self._lock:
            for k in [k for k in self._d if pred(k)]:
                self._bytes -= self._size(self._d.pop(k))

    def clear(self):
        with self._lock:
            self._d.clear()
            self._bytes = 0


class Pipeline:
    def __init__(self, store: ImageStore):
        self.store = store
        self.dark_threshold = 140
        self._seg = _LRU(256, 600 * 1024 * 1024)
        self._layers = _LRU(128, 400 * 1024 * 1024)

    def crop(self, fig: PoleFigure) -> Optional[Crop]:
        img = self.store.get(fig.image_id)
        if img is None:
            return None
        return crop_circle(img, fig.cx, fig.cy, fig.r)

    def _seg_key(self, fig: PoleFigure, params: ReconParams):
        return (fig.id, fig.image_id, self.store.version(fig.image_id), fig.geometry(),
                params.seg_key(), fig.mask_version, self.dark_threshold)

    def segmentation(self, fig: PoleFigure, params: ReconParams) -> Optional[SegResult]:
        key = self._seg_key(fig, params)
        res = self._seg.get(key)
        if res is not None:
            return res
        crop = self.crop(fig)
        if crop is None:
            return None
        edits = fig.mask_edits.copy() if fig.mask_edits is not None else None
        res = segment(crop, params, edits, self.dark_threshold)
        self._seg.put(key, res)
        return res

    def layer(self, fig: PoleFigure, params: ReconParams, radius_px: float) -> Optional[np.ndarray]:
        e = params.effective()
        key = (self._seg_key(fig, params), e.dot_size, e.cluster_scale, e.contrast, e.saturation,
               e.smooth_scaling, round(float(radius_px), 2))
        lay = self._layers.get(key)
        if lay is not None:
            return lay
        seg = self.segmentation(fig, params)
        if seg is None:
            return None
        lay = render_layer(seg, params, radius_px)
        lay.setflags(write=False)
        self._layers.put(key, lay)
        return lay

    def invalidate(self, fig_id: str | None = None) -> None:
        if fig_id is None:
            self._seg.clear()
            self._layers.clear()
        else:
            self._seg.drop(lambda k: k[0] == fig_id)
            self._layers.drop(lambda k: k[0][0] == fig_id)
