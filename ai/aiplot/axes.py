from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frame import PlotBox


@dataclass
class TickSet:
    positions: list[float]
    step_px: float
    regular: bool

    def as_dict(self) -> dict:
        return {
            "positions": [round(p, 2) for p in self.positions],
            "step_px": round(self.step_px, 3),
            "regular": self.regular,
        }


def _runs(flags: np.ndarray, gap: int = 1) -> list[tuple[int, int]]:
    idx = np.where(flags)[0]
    if idx.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - prev <= gap + 1:
            prev = v
            continue
        out.append((start, prev))
        start = prev = v
    out.append((start, prev))
    return out


def _scan_band(band: np.ndarray, min_ink: int) -> list[float]:
    counts = band.sum(axis=1)
    flags = counts >= min_ink
    return [round((a + b) / 2.0, 2) for a, b in _runs(flags)]


def _regularise(positions: list[float]) -> TickSet:
    if len(positions) < 2:
        return TickSet(positions, 0.0, False)
    diffs = np.diff(positions)
    step = float(np.median(diffs))
    if step <= 0:
        return TickSet(positions, 0.0, False)
    kept = [positions[0]]
    for p in positions[1:]:
        if abs((p - kept[-1]) - step) <= 0.35 * step:
            kept.append(p)
    while kept and abs((kept[-1] - kept[0]) / step - round((kept[-1] - kept[0]) / step)) > 0.3:
        kept.pop()
    regular = len(kept) >= 3
    return TickSet(kept, step, regular)


def _vertical_ticks(ink: np.ndarray, x: int, y0: int, y1: int, window: int, outward: bool):
    if outward:
        lo, hi = max(0, x - window), max(0, x - 1)
    else:
        lo, hi = x + 2, x + 2 + window
    band = ink[y0 : y1 + 1, lo:hi]
    if band.size == 0:
        return []
    min_ink = max(2, int(0.3 * band.shape[1]))
    return [y0 + p for p in _scan_band(band, min_ink)]


def _horizontal_ticks(ink: np.ndarray, y: int, x0: int, x1: int, window: int, outward: bool):
    if outward:
        lo, hi = max(0, y - window), max(0, y - 1)
    else:
        lo, hi = y + 2, y + 2 + window
    band = ink[lo:hi, x0 : x1 + 1]
    if band.size == 0:
        return []
    min_ink = max(2, int(0.3 * band.shape[0]))
    return [x0 + p for p in _scan_band(band.T, min_ink)]


def find_ticks(ink: np.ndarray, box: PlotBox, window: int = 14) -> tuple[TickSet, TickSet]:
    candidates_y = [
        _vertical_ticks(ink, box.x0, box.y0, box.y1, window, outward=True),
        _vertical_ticks(ink, box.x0, box.y0, box.y1, window, outward=False),
    ]
    candidates_x = [
        _horizontal_ticks(ink, box.y1, box.x0, box.x1, window, outward=True),
        _horizontal_ticks(ink, box.y1, box.x0, box.x1, window, outward=False),
    ]
    y_ticks = max(candidates_y, key=lambda c: (len(c), c[-1] if c else 0))
    x_ticks = max(candidates_x, key=lambda c: (len(c), c[-1] if c else 0))
    return _regularise(x_ticks), _regularise(y_ticks)


def guess_scale(values: list[float | None]) -> str:
    numeric = [v for v in values if v is not None and v > 0]
    if len(numeric) < 3:
        return "linear"
    ratios = [b / a for a, b in zip(numeric, numeric[1:]) if a > 0 and b > 0]
    if not ratios:
        return "linear"
    spread = max(ratios) - min(ratios)
    if spread < 0.25 * float(np.median(ratios)) and abs(float(np.median(ratios)) - 1.0) > 0.2:
        return "log"
    return "linear"