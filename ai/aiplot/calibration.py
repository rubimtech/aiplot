from __future__ import annotations

import math

from .axes import TickSet
from .frame import PlotBox


def _log10(value: float) -> float:
    return math.log10(value)


def _finite_pairs(
    ticks: TickSet, values: list[float | None] | None
) -> list[tuple[float, float]]:
    if values is None or len(values) != len(ticks.positions):
        return []
    pairs = [
        (float(p), float(v))
        for p, v in zip(ticks.positions, values)
        if v is not None and math.isfinite(float(v))
    ]
    return pairs


def _axis_payload(
    pairs: list[tuple[float, float]],
    axis: str,
    fixed_pixel: float,
    scale: str,
) -> dict:
    if len(pairs) < 2:
        raise ValueError(f"{axis} axis needs two labelled ticks")
    out = []
    for pos, value in (pairs[0], pairs[-1]):
        pixel = [pos, fixed_pixel] if axis == "x" else [fixed_pixel, pos]
        out.append({"pixel": pixel, "value": value})
    return {"scale": scale, "ref_points": out}


def build_calibration(
    box: PlotBox,
    x_ticks: TickSet,
    y_ticks: TickSet,
    *,
    x_values: list[float | None] | None = None,
    y_values: list[float | None] | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    name: str = "Axes",
) -> dict:
    x_pairs = _finite_pairs(x_ticks, x_values)
    y_pairs = _finite_pairs(y_ticks, y_values)
    return {
        "name": name,
        "x": _axis_payload(x_pairs, "x", float(box.y1), x_scale),
        "y": _axis_payload(y_pairs, "y", float(box.x0), y_scale),
        "source": "manual",
        "coords_type": "cartesian",
        "model": "auto",
        "axis_points": [],
        "theta_units": "degrees",
        "origin_radius": 0.0,
        "scale_bar": None,
        "bar_horizontal": False,
    }


class LinearAxis:
    def __init__(self, pixel_a: float, value_a: float, pixel_b: float, value_b: float):
        if abs(pixel_b - pixel_a) < 1e-9:
            raise ValueError("degenerate axis")
        self.pa, self.va = pixel_a, value_a
        self.pb, self.vb = pixel_b, value_b

    def to_value(self, pixel: float) -> float:
        t = (pixel - self.pa) / (self.pb - self.pa)
        return self.va + t * (self.vb - self.va)

    def to_pixel(self, value: float) -> float:
        t = (value - self.va) / (self.vb - self.va)
        return self.pa + t * (self.pb - self.pa)


class LogAxis(LinearAxis):
    def to_value(self, pixel: float) -> float:
        t = (pixel - self.pa) / (self.pb - self.pa)
        return 10 ** (_log10(self.va) + t * (_log10(self.vb) - _log10(self.va)))

    def to_pixel(self, value: float) -> float:
        t = (_log10(value) - _log10(self.va)) / (_log10(self.vb) - _log10(self.va))
        return self.pa + t * (self.pb - self.pa)


def axis_from_ref_points(payload: dict, coord: int) -> LinearAxis | None:
    refs = payload.get("ref_points") or []
    if len(refs) < 2:
        return None
    pairs = [(float(r["pixel"][coord]), float(r["value"])) for r in refs]
    (pa, va), (pb, vb) = pairs[0], pairs[-1]
    scale = payload.get("scale", "linear")
    if pa == pb:
        return None
    if scale == "log" and va > 0 and vb > 0:
        return LogAxis(pa, va, pb, vb)
    return LinearAxis(pa, va, pb, vb)


def axis_from_ticks(ticks: TickSet, values: list[float | None], scale: str) -> LinearAxis | None:
    pairs = _finite_pairs(ticks, values)
    if len(pairs) < 2:
        return None
    (pa, va), (pb, vb) = pairs[0], pairs[-1]
    if scale == "log" and va > 0 and vb > 0:
        return LogAxis(pa, va, pb, vb)
    return LinearAxis(pa, va, pb, vb)