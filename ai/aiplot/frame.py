from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class PlotBox:
    x0: int
    y0: int
    x1: int
    y1: int
    has_top: bool = True
    has_right: bool = True
    confidence: float = 0.0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_dict(self) -> dict:
        return asdict(self)

    def contains(self, x: float, y: float, margin: int = 0) -> bool:
        return (
            self.x0 - margin <= x <= self.x1 + margin
            and self.y0 - margin <= y <= self.y1 + margin
        )


def _long_lines(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = ink.shape
    u8 = ink.astype(np.uint8) * 255
    kv = max(15, int(0.18 * h))
    kh = max(15, int(0.18 * w))
    vert = cv2.morphologyEx(
        u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
    )
    horz = cv2.morphologyEx(
        u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
    )
    return vert > 0, horz > 0


def _group_positions(positions: list[int], gap: int = 3) -> list[tuple[int, int]]:
    if not positions:
        return []
    groups: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [(int(round(sum(g) / len(g))), len(g)) for g in groups]


def _line_candidates(mask: np.ndarray, axis: int) -> list[tuple[int, int]]:
    counts = mask.sum(axis=axis)
    if counts.size == 0:
        return []
    peak = int(counts.max())
    if peak < 10:
        return []
    threshold = max(int(0.35 * peak), 10)
    idx = np.where(counts >= threshold)[0]
    groups = _group_positions([int(v) for v in idx])
    span = mask.shape[0] if axis == 0 else mask.shape[1]
    out: list[tuple[int, int]] = []
    for center, thickness in groups:
        col = counts[max(0, center - thickness // 2) : center + thickness // 2 + 1]
        strength = int(col.max()) if col.size else 0
        if strength >= 0.25 * span:
            out.append((center, strength))
    return out


def _extent_v(vert: np.ndarray, x: int) -> tuple[int, int]:
    ys = np.where(vert[:, x])[0]
    return int(ys.min()), int(ys.max())


def _extent_h(horz: np.ndarray, y: int) -> tuple[int, int]:
    xs = np.where(horz[y, :])[0]
    return int(xs.min()), int(xs.max())


def find_plot_box(ink: np.ndarray) -> PlotBox:
    h, w = ink.shape
    vert, horz = _long_lines(ink)
    vlines = _line_candidates(vert, 0)
    hlines = _line_candidates(horz, 1)

    best: PlotBox | None = None
    best_score = 0.0
    for i, (left, ls) in enumerate(vlines):
        for right, rs in vlines[i + 1 :]:
            for j, (top, ts) in enumerate(hlines):
                for bottom, bs in hlines[j + 1 :]:
                    width = right - left
                    height = bottom - top
                    if width < 0.2 * w or height < 0.2 * h:
                        continue
                    if width < 20 or height < 20:
                        continue
                    score = width * height * (ls + rs + ts + bs)
                    if score > best_score:
                        best_score = float(score)
                        best = PlotBox(left, top, right, bottom, True, True, 1.0)

    if best is not None:
        return best

    if hlines and vlines:
        left_candidates = [c for c, _ in vlines if c < 0.5 * w] or [vlines[0][0]]
        bottom_candidates = [c for c, _ in hlines if c > 0.5 * h] or [hlines[-1][0]]
        left = min(left_candidates)
        bottom = max(bottom_candidates)
        y0, y1 = _extent_v(vert, left)
        x0, x1 = _extent_h(horz, bottom)
        top = max(y0, 0)
        right = min(x1, w - 1)
        if right - left > 20 and bottom - top > 20:
            return PlotBox(left, top, right, bottom, False, False, 0.6)

    ys, xs = np.where(ink)
    if xs.size == 0:
        raise ValueError("Image contains no ink")
    x0, x1 = int(np.percentile(xs, 2)), int(np.percentile(xs, 98))
    y0, y1 = int(np.percentile(ys, 2)), int(np.percentile(ys, 98))
    if x1 - x0 < 20 or y1 - y0 < 20:
        raise ValueError("Could not locate a plot area")
    return PlotBox(x0, y0, x1, y1, False, False, 0.3)