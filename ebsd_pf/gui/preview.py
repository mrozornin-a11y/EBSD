"""Фоновая отрисовка предпросмотра компоновок (данные копируются, чтобы не зависеть от правок в GUI)."""
from __future__ import annotations

from dataclasses import replace

from ..core.render import Sheet, render_sheet_image
from .state import AppState


def snapshot_provider(state: AppState, fig_ids):
    figs = {}
    for fid in fig_ids:
        f = state.project.figure(fid)
        if f is not None:
            figs[fid] = replace(f, mask_edits=f.mask_edits.copy() if f.mask_edits else None)
    pipeline = state.pipeline

    def provider(fig_id, params, radius_px):
        f = figs.get(fig_id)
        if f is None:
            return None
        return pipeline.layer(f, params, radius_px)
    return provider


def freeze_sheet(sheet: Sheet) -> Sheet:
    """Копия компоновки с независимыми копиями параметров."""
    figs = [replace(pf, params=pf.params.copy()) for pf in sheet.figures]
    return Sheet(sheet.width, sheet.height, figs, list(sheet.texts))


def render_async(state: AppState, sheet: Sheet, px_per_mm: float, on_done, name="Предпросмотр",
                 max_px: int = 3200):
    sheet = freeze_sheet(sheet)
    longest = max(sheet.width, sheet.height) * px_per_mm
    if longest > max_px:
        px_per_mm *= max_px / longest
    provider = snapshot_provider(state, [pf.fig_id for pf in sheet.figures])
    return state.runner.run(name, render_sheet_image, sheet, px_per_mm, provider,
                            on_done=on_done, silent=True)
