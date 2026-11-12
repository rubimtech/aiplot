from __future__ import annotations

import cv2
import numpy as np

from .recognize import Recognition

_PALETTE = [
    (60, 60, 230),
    (230, 120, 40),
    (40, 180, 60),
    (200, 60, 200),
    (30, 160, 220),
    (120, 90, 20),
    (20, 20, 20),
    (180, 180, 40),
]


def draw_overlay(bgr: np.ndarray, result: Recognition, *, thickness: int = 3) -> np.ndarray:
    out = bgr.copy()
    box = result.box
    cv2.rectangle(out, (box.x0, box.y0), (box.x1, box.y1), (0, 165, 255), 3)

    for x in result.x_ticks.positions:
        cv2.line(out, (int(round(x)), box.y1), (int(round(x)), box.y1 + 12), (0, 165, 255), 2)
    for y in result.y_ticks.positions:
        cv2.line(out, (box.x0 - 12, int(round(y))), (box.x0, int(round(y))), (0, 165, 255), 2)

    for i, curve in enumerate(result.curves):
        color = _PALETTE[i % len(_PALETTE)]
        pts = np.array([[int(round(px)), int(round(py))] for px, py in curve.points], np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
        for px, py in curve.points:
            cv2.circle(out, (int(round(px)), int(round(py))), 4, color, -1, cv2.LINE_AA)
        if len(pts) >= 1:
            anchor = tuple(int(v) for v in pts[0])
            cv2.putText(
                out,
                f"{i + 1}",
                (anchor[0] + 6, anchor[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
    return out