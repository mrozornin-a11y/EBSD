"""Модель данных проекта: изображения, полюсные фигуры, параметры реконструкции и экспорта."""
from __future__ import annotations

import base64
import io
import re
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Optional

import numpy as np

PLANE_PRESETS = ["100", "110", "111", "112"]
NO_SAMPLE = "Без образца"
NO_GRAIN = "—"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def normalize_plane(text: str | None) -> str:
    """Убирает скобки и пробелы: '{1 1 1}' -> '111', '(110)' -> '110'."""
    if not text:
        return ""
    s = str(text).strip()
    s = s.strip("{}()[]<> \t")
    s = re.sub(r"\s+", " ", s)
    # "1 1 1" -> "111" только если все компоненты однозначные
    parts = s.split(" ")
    if len(parts) > 1 and all(re.fullmatch(r"-?\d", p) for p in parts):
        s = "".join(parts)
    return s


def format_plane(text: str | None) -> str:
    """Оформление индекса плоскости: '111' -> '{111}'."""
    p = normalize_plane(text)
    return "{" + p + "}" if p else ""


def plane_sort_key(plane: str) -> tuple:
    p = normalize_plane(plane)
    if p in PLANE_PRESETS:
        return (0, PLANE_PRESETS.index(p), p)
    return (1, 0, p)


def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _filter_kwargs(cls, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


# --------------------------------------------------------------------------------------
# Параметры
# --------------------------------------------------------------------------------------
@dataclass
class DetectParams:
    min_radius: int = 20            # px исходного изображения
    max_radius: int = 0             # 0 = авто (половина меньшей стороны)
    dark_threshold: int = 140       # яркость, ниже которой пиксель считается линией
    ring_coverage: float = 0.85     # доля окружности, покрытая линией
    require_axes: bool = True       # требовать горизонтальную и вертикальную оси
    min_inside_background: float = 0.30  # доля фона внутри окружности
    max_work_dim: int = 1800        # размер уменьшенной копии для поиска кандидатов

    @classmethod
    def from_dict(cls, d: dict) -> "DetectParams":
        return cls(**_filter_kwargs(cls, d or {}))


RECON_MODES = {"exact": "Точный", "enhanced": "Повышение читаемости"}
COLOR_MODES = {
    "original": "Исходные цвета пикселей",
    "smooth": "Сглаживание внутри кластера",
    "cluster": "Единый цвет кластера (медиана)",
}
LABEL_POSITIONS = {
    "top_right": "Сверху справа",
    "top_center": "Сверху по центру",
    "top_left": "Сверху слева",
}


@dataclass
class ReconParams:
    mode: str = "exact"
    # --- сегментация
    color_threshold: int = 35       # порог отличия от фона / хроматичности, 0..255
    min_object_size: int = 1        # объекты меньше (пикс.) считаются шумом
    protect_saturated: bool = True  # не удалять мелкие, но насыщенные объекты
    include_dark_points: bool = True  # учитывать ахроматические (серые/чёрные) точки
    restore_under_axes: bool = True
    edge_margin: float = 1.0        # отступ внутрь от окружности, пикс.
    # --- улучшение читаемости (только режим enhanced)
    dot_size: float = 1.0           # размер цветного пикселя, в пикселях исходника
    cluster_scale: float = 1.0      # увеличение кластеров вокруг их центров
    contrast: float = 1.0
    saturation: float = 1.0
    jpeg_cleanup: bool = False
    color_mode: str = "original"
    # --- оформление
    draw_axes: bool = True
    circle_width_pt: float = 1.0
    axis_width_pt: float = 0.5
    show_label: bool = True
    label_size_pt: float = 10.0
    label_position: str = "top_right"
    label_dx_mm: float = 0.0
    label_dy_mm: float = 0.0
    smooth_scaling: bool = False    # интерполяция при масштабировании точного слоя

    SEG_FIELDS = (
        "mode", "color_threshold", "min_object_size", "protect_saturated",
        "include_dark_points", "restore_under_axes", "edge_margin", "jpeg_cleanup", "color_mode",
    )

    def effective(self) -> "ReconParams":
        """Параметры с учётом режима: в точном режиме улучшения отключены."""
        if self.mode == "exact":
            return replace(self, dot_size=1.0, cluster_scale=1.0, contrast=1.0, saturation=1.0,
                           jpeg_cleanup=False, color_mode="original")
        return replace(self)

    def seg_key(self) -> tuple:
        e = self.effective()
        return tuple(getattr(e, n) for n in self.SEG_FIELDS)

    def copy(self) -> "ReconParams":
        return replace(self)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ReconParams":
        return cls(**_filter_kwargs(cls, d or {}))


@dataclass
class LayoutSettings:
    diameter_mm: float = 40.0
    spacing_mm: float = 5.0
    columns: int = 2                # 0 = авто
    margin_mm: float = 3.0
    show_group_title: bool = False
    title_size_pt: float = 11.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "LayoutSettings":
        return cls(**_filter_kwargs(cls, d or {}))


@dataclass
class ExportSettings:
    output_dir: str = ""
    formats: list = field(default_factory=lambda: ["png"])
    dpi: int = 600
    vector_points: bool = False     # векторизовать кластеры в SVG/PDF
    raster_layer_dpi: int = 600     # разрешение встроенного слоя точек в SVG/PDF
    combine_sample: bool = False    # все зерна образца на одном листе
    scope: str = "grain"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "ExportSettings":
        obj = cls(**_filter_kwargs(cls, d or {}))
        obj.formats = [f for f in obj.formats if f in ("png", "tiff", "svg", "pdf")] or ["png"]
        return obj


# --------------------------------------------------------------------------------------
# Ручная правка маски
# --------------------------------------------------------------------------------------
def _png_b64_encode(arr: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((arr > 0).astype(np.uint8) * 255).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_b64_decode(s: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(io.BytesIO(base64.b64decode(s))).convert("L")
    return np.asarray(img) > 127


# Область маски в нормированных координатах: [-MASK_EXTENT, MASK_EXTENT]
MASK_EXTENT = 1.0


@dataclass
class MaskEdits:
    """Ручные правки маски в нормированной системе координат окружности.

    Массивы квадратные (N×N), покрывают квадрат [-1, 1]² относительно центра,
    единица = радиус окружности. При изменении геометрии окружности правки
    пересчитываются (ресемплинг ближайшим соседом).
    """
    add: np.ndarray
    remove: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "MaskEdits":
        z = np.zeros((n, n), dtype=bool)
        return cls(add=z.copy(), remove=z.copy())

    def is_empty(self) -> bool:
        return not (self.add.any() or self.remove.any())

    def resampled(self, n: int) -> "MaskEdits":
        if self.add.shape[0] == n:
            return self
        import cv2
        a = cv2.resize(self.add.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0
        r = cv2.resize(self.remove.astype(np.uint8), (n, n), interpolation=cv2.INTER_NEAREST) > 0
        return MaskEdits(a, r)

    def copy(self) -> "MaskEdits":
        return MaskEdits(self.add.copy(), self.remove.copy())

    def to_dict(self) -> dict:
        return {"n": int(self.add.shape[0]), "add": _png_b64_encode(self.add),
                "remove": _png_b64_encode(self.remove)}

    @classmethod
    def from_dict(cls, d: dict | None) -> Optional["MaskEdits"]:
        if not d:
            return None
        try:
            a = _png_b64_decode(d["add"])
            r = _png_b64_decode(d["remove"])
            if a.shape != r.shape:
                return None
            return cls(a, r)
        except Exception:
            return None


# --------------------------------------------------------------------------------------
# Объекты проекта
# --------------------------------------------------------------------------------------
@dataclass
class SourceImage:
    id: str
    path: str
    name: str
    width: int = 0
    height: int = 0
    embedded_png: Optional[str] = None   # base64 при встраивании в проект

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SourceImage":
        return cls(**_filter_kwargs(cls, d))


@dataclass
class PoleFigure:
    id: str
    image_id: str
    cx: float
    cy: float
    r: float
    sample: str = ""
    grain: str = ""
    plane: str = ""
    order: int = 0
    auto: bool = False
    score: float = 0.0
    params: Optional[ReconParams] = None
    mask_edits: Optional[MaskEdits] = None
    mask_version: int = 0

    @property
    def plane_label(self) -> str:
        return format_plane(self.plane)

    @property
    def sample_key(self) -> str:
        return self.sample.strip() or NO_SAMPLE

    @property
    def grain_key(self) -> str:
        return self.grain.strip() or NO_GRAIN

    def geometry(self) -> tuple:
        return (round(self.cx, 3), round(self.cy, 3), round(self.r, 3))

    def to_dict(self) -> dict:
        return {
            "id": self.id, "image_id": self.image_id,
            "cx": float(self.cx), "cy": float(self.cy), "r": float(self.r),
            "sample": self.sample, "grain": self.grain, "plane": self.plane,
            "order": int(self.order), "auto": bool(self.auto), "score": float(self.score),
            "params": self.params.to_dict() if self.params else None,
            "mask_edits": self.mask_edits.to_dict() if self.mask_edits else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PoleFigure":
        fig = cls(
            id=d.get("id") or new_id(), image_id=d["image_id"],
            cx=float(d["cx"]), cy=float(d["cy"]), r=float(d["r"]),
            sample=str(d.get("sample", "")), grain=str(d.get("grain", "")),
            plane=normalize_plane(d.get("plane", "")), order=int(d.get("order", 0)),
            auto=bool(d.get("auto", False)), score=float(d.get("score", 0.0)),
        )
        if d.get("params"):
            fig.params = ReconParams.from_dict(d["params"])
        fig.mask_edits = MaskEdits.from_dict(d.get("mask_edits"))
        return fig


@dataclass
class Project:
    images: list = field(default_factory=list)       # list[SourceImage]
    figures: list = field(default_factory=list)      # list[PoleFigure]
    recon: ReconParams = field(default_factory=ReconParams)
    layout: LayoutSettings = field(default_factory=LayoutSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    detect: DetectParams = field(default_factory=DetectParams)
    embed_images: bool = False
    ui: dict = field(default_factory=dict)

    # --- доступ
    def image(self, image_id: str) -> Optional[SourceImage]:
        return next((i for i in self.images if i.id == image_id), None)

    def figure(self, fig_id: str) -> Optional[PoleFigure]:
        return next((f for f in self.figures if f.id == fig_id), None)

    def figures_on(self, image_id: str) -> list:
        return [f for f in self.figures if f.image_id == image_id]

    def params_for(self, fig: PoleFigure) -> ReconParams:
        return fig.params if fig.params is not None else self.recon

    # --- группировка
    def samples(self) -> list:
        keys = {f.sample_key for f in self.figures}
        return sorted(keys, key=lambda s: (s == NO_SAMPLE, natural_key(s)))

    def grains(self, sample: str) -> list:
        keys = {f.grain_key for f in self.figures if f.sample_key == sample}
        return sorted(keys, key=lambda g: (g == NO_GRAIN, natural_key(g)))

    def grain_figures(self, sample: str, grain: str) -> list:
        figs = [f for f in self.figures if f.sample_key == sample and f.grain_key == grain]
        return sorted(figs, key=lambda f: (f.order, plane_sort_key(f.plane), f.id))

    def grouped(self) -> list:
        """[(sample, [(grain, [figures])])]"""
        return [(s, [(g, self.grain_figures(s, g)) for g in self.grains(s)]) for s in self.samples()]

    def renumber_orders(self) -> None:
        for s in self.samples():
            for g in self.grains(s):
                for i, f in enumerate(self.grain_figures(s, g)):
                    f.order = i
