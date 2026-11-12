from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ocr
from .axes import TickSet, find_ticks, guess_scale
from .calibration import axis_from_ref_points, build_calibration
from .curves import CurveCandidate, extract_curves
from .frame import PlotBox, find_plot_box
from .imaging import decode_image, ink_mask, to_gray


def _anchor_ticks(ticks: TickSet, lo: float, hi: float) -> TickSet:
    span = float(hi - lo)
    if span <= 0:
        return ticks
    if (
        len(ticks.positions) >= 3
        and ticks.regular
        and ticks.positions[0] <= lo + 0.08 * span
        and ticks.positions[-1] >= hi - 0.08 * span
    ):
        return ticks
    return TickSet([float(lo), float(hi)], span, True)


def _ascending(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    return points if points[0][0] <= points[-1][0] else list(reversed(points))


@dataclass
class Recognition:
    width: int
    height: int
    box: PlotBox
    x_ticks: TickSet
    y_ticks: TickSet
    x_values: list[float | None] = field(default_factory=list)
    y_values: list[float | None] = field(default_factory=list)
    x_scale: str = "linear"
    y_scale: str = "linear"
    curves: list[CurveCandidate] = field(default_factory=list)
    calibration: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "image": {"width": self.width, "height": self.height},
            "plot_box": self.box.as_dict(),
            "axes": {
                "x": {
                    **self.x_ticks.as_dict(),
                    "values": self.x_values,
                    "scale": self.x_scale,

                },
                "y": {
                    **self.y_ticks.as_dict(),
                    "values": self.y_values,
                    "scale": self.y_scale,

                },
            },
            "calibration": self.calibration,
            "curves": [c.as_dict() for c in self.curves],
            "warnings": self.warnings,
        }

    def base_curves(self) -> list[dict]:
        out = []
        for curve in self.curves:
            if len(curve.points) < 2 or curve.kind == "annotation":
                continue
            out.append(
                {
                    "label": curve.label,
                    "color": curve.color,
                    "trace_color": curve.color,
                    "style": "solid",
                    "visible": True,
                    "connect_as": "line",
                    "target_point_count": max(9, min(120, len(curve.points))),
                    "points": [
                        {"pixel": [round(x, 3), round(y, 3)], "origin": "ai"}
                        for x, y in _ascending(curve.points)
                    ],
                }
            )
        return out

    def data_points(self) -> list[dict]:
        if not self.calibration:
            return []
        x_axis = axis_from_ref_points(self.calibration["x"], 0)
        y_axis = axis_from_ref_points(self.calibration["y"], 1)
        if x_axis is None or y_axis is None:
            return []
        out = []
        for curve in self.curves:
            if len(curve.points) < 2 or curve.kind == "annotation":
                continue
            out.append(
                {
                    "label": curve.label,
                    "color": curve.color,
                    "points": [
                        {"x": x_axis.to_value(px), "y": y_axis.to_value(py)}
                        for px, py in _ascending(curve.points)
                    ],
                }
            )
        return out


def _values_from_range(
    ticks: TickSet,
    span: tuple[float, float] | None,
    scale: str,
    *,
    descending: bool = False,
) -> list[float | None]:
    if not ticks.positions:
        return []
    if span is None:
        return [None] * len(ticks.positions)
    lo, hi = span
    n = len(ticks.positions)
    if n == 1:
        return [lo]
    if scale == "log" and lo > 0 and hi > 0:
        values = list(np.geomspace(lo, hi, n))
    else:
        values = list(np.linspace(lo, hi, n))
    if descending:
        values.reverse()
    return values


def _exponent_at_origin(
    grid, pairs: list[tuple[float, float]]
) -> int | None:
    """Infer the power of ten sitting at the first detected decade line."""
    if not pairs or grid.step_px <= 0:
        return None
    for pixel, value in pairs:
        if value <= 0:
            continue
        t = (pixel - grid.positions[0]) / grid.step_px
        return int(round(math.log10(value) - t))
    return None


def _labelled_pairs(
    ticks: TickSet, values: list[float | None]
) -> list[tuple[float, float]]:
    if not values or len(values) != len(ticks.positions):
        return []
    return [
        (float(p), float(v))
        for p, v in zip(ticks.positions, values)
        if v is not None and v > 0
    ]


def _log_range(
    grid, exponent0: int, pixel_a: float, pixel_b: float
) -> tuple[float, float]:
    t_a = (pixel_a - grid.positions[0]) / grid.step_px
    t_b = (pixel_b - grid.positions[0]) / grid.step_px
    lo = 10.0 ** (exponent0 + t_a)
    hi = 10.0 ** (exponent0 + t_b)
    return (min(lo, hi), max(lo, hi))


def recognize(
    data: bytes,
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    x_scale: str | None = None,
    y_scale: str | None = None,
    use_ocr: bool = True,
    max_curves: int = 8,
    max_points: int = 120,
    mode: str = "auto",
) -> Recognition:
    bgr = decode_image(data)
    gray = to_gray(bgr)
    ink = ink_mask(gray)
    box = find_plot_box(ink)

    x_ticks, y_ticks = find_ticks(ink, box)
    x_anchor = _anchor_ticks(x_ticks, box.x0, box.x1)
    y_anchor = _anchor_ticks(y_ticks, box.y0, box.y1)
    warnings: list[str] = []
    if len(x_ticks.positions) < 2 and x_range is None:
        warnings.append("x axis ticks not found; calibrate X manually")
    if len(y_ticks.positions) < 2 and y_range is None:
        warnings.append("y axis ticks not found; calibrate Y manually")
    if x_range is None and y_range is None and not (x_ticks.regular and y_ticks.regular):
        warnings.append("tick spacing is irregular; verify the suggested range")

    x_values: list[float | None] = []
    y_values: list[float | None] = []
    if use_ocr and ocr.available():
        if x_ticks.positions and x_anchor.positions == x_ticks.positions:
            x_values = ocr.read_x_labels(gray, box.x0, box.x1, box.y1, x_ticks.positions)
        if y_ticks.positions and y_anchor.positions == y_ticks.positions:
            y_values = ocr.read_y_labels(gray, box.x0, y_ticks.positions)
        if all(v is None for v in x_values) and x_range is None:
            warnings.append("x tick labels unreadable; pass --x-range")
        if all(v is None for v in y_values) and y_range is None:
            warnings.append("y tick labels unreadable; pass --y-range")

    x_scale = x_scale or guess_scale(x_values)
    y_scale = y_scale or guess_scale(y_values)

    if x_range is not None:
        x_values = _values_from_range(x_anchor, x_range, x_scale)
    elif all(v is None for v in x_values):
        x_values = _values_from_range(x_anchor, None, x_scale)
    if y_range is not None:
        y_values = _values_from_range(y_anchor, y_range, y_scale, descending=True)
    elif all(v is None for v in y_values):
        y_values = _values_from_range(y_anchor, None, y_scale)

    calibration = None
    has_x = any(v is not None for v in x_values)
    has_y = any(v is not None for v in y_values)
    if has_x and has_y and len(x_anchor.positions) >= 2 and len(y_anchor.positions) >= 2:
        try:
            calibration = build_calibration(
                box,
                x_anchor,
                y_anchor,
                x_values=x_values,
                y_values=y_values,
                x_scale=x_scale,
                y_scale=y_scale,
            )
        except ValueError as exc:
            warnings.append(f"calibration not built: {exc}")
    elif not has_x or not has_y:
        warnings.append("axis values unknown; curves are pushed as pixel points only")

    curves = extract_curves(
        bgr, ink, box, max_curves=max_curves, max_points=max_points, mode=mode
    )
    if not curves:
        warnings.append("no curves extracted; use the editor tools")
    if box.confidence < 0.6:
        warnings.append("plot frame detected with low confidence; verify the plot box")

    return Recognition(
        width=bgr.shape[1],
        height=bgr.shape[0],
        box=box,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_values=x_values,
        y_values=y_values,
        x_scale=x_scale,
        y_scale=y_scale,
        curves=curves,
        calibration=calibration,
        warnings=warnings,
    )