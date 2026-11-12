from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .frame import PlotBox


@dataclass
class CurveCandidate:
    label: str
    color: str
    points: list[tuple[float, float]] = field(default_factory=list)
    kind: str = "color"
    score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    angle: float | None = None
    rho: float | None = None
    slope: float | None = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "color": self.color,
            "kind": self.kind,
            "score": round(self.score, 4),
            "points_px": [(round(x, 2), round(y, 2)) for x, y in self.points],
            "warnings": list(self.warnings),
        }


def _bgr_to_hex(bgr: tuple[float, float, float]) -> str:
    b, g, r = (int(max(0, min(255, round(c)))) for c in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"


def _simplify(points: list[tuple[float, float]], eps: float, max_points: int) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, eps, False).reshape(-1, 2)
    out = [(float(x), float(y)) for x, y in approx]
    if len(out) > max_points:
        idx = np.linspace(0, len(out) - 1, max_points).round().astype(int)
        out = [out[i] for i in idx]
    return out


def _track_component(mask: np.ndarray, x: int, y: int, w: int, h: int) -> list[tuple[float, float]]:
    sub = mask[y : y + h, x : x + w]
    points: list[tuple[float, float]] = []
    for cx in range(w):
        col = np.flatnonzero(sub[:, cx])
        if col.size == 0:
            continue
        points.append((x + cx + 0.5, y + float(np.median(col)) + 0.5))
    return points


def _components(
    mask: np.ndarray,
    box: PlotBox,
    *,
    min_area: int,
    max_points: int,
    label_prefix: str,
    bgr: np.ndarray | None = None,
    fallback_color: str = "#2563eb",
) -> list[CurveCandidate]:
    boxes = np.zeros_like(mask, dtype=np.uint8)
    boxes[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1] = 1
    mask = (mask & (boxes > 0)).astype(np.uint8)
    if not mask.any():
        return []
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    linked = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
    out: list[CurveCandidate] = []
    for idx in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[idx])
        if area < min_area:
            continue
        if w < max(8, int(0.03 * box.width)) and h < max(8, int(0.03 * box.height)):
            continue
        comp = labels == idx
        pts = _track_component(comp, x, y, w, h)
        if len(pts) < 3:
            continue
        if bgr is not None:
            color = _bgr_to_hex(tuple(float(v) for v in bgr[comp].mean(axis=0)))
        else:
            color = fallback_color
        score = area / float(max(1, w * h))
        out.append(
            CurveCandidate(
                label=f"{label_prefix} {len(out) + 1}",
                color=color,
                points=_simplify(pts, 1.5, max_points),
                kind="color" if label_prefix.startswith("curve") else "ink",
                score=score,
                bbox=(x, y, w, h),
            )
        )
    out.sort(key=lambda c: -len(c.points))
    return out


def _merge_close(
    curves: list[CurveCandidate], gap_x: int = 12, gap_y: int = 16
) -> list[CurveCandidate]:
    ordered = sorted(curves, key=lambda c: c.bbox[0] if c.bbox else 0)
    merged: list[CurveCandidate] = []
    for curve in ordered:
        if curve.bbox is None:
            merged.append(curve)
            continue
        cx0, cy0, cw, ch = curve.bbox
        cx1, cy1 = cx0 + cw, cy0 + ch
        joined = None
        for target in merged:
            if target.bbox is None:
                continue
            mx0, my0, mw, mh = target.bbox
            mx1, my1 = mx0 + mw, my0 + mh
            if cx0 <= mx1 + gap_x and cy0 <= my1 + gap_y and cy1 >= my0 - gap_y:
                joined = target
                break
        if joined is None:
            merged.append(curve)
            continue
        joined.points = sorted(joined.points + curve.points)
        mx0, my0, mw, mh = joined.bbox
        nx0, ny0 = min(mx0, cx0), min(my0, cy0)
        nx1, ny1 = max(mx0 + mw, cx1), max(my0 + mh, cy1)
        joined.bbox = (nx0, ny0, nx1 - nx0, ny1 - ny0)
        joined.warnings.extend(w for w in curve.warnings if w not in joined.warnings)
    return merged


def _hue_peaks(hsv: np.ndarray, box: PlotBox, limit: int) -> list[int]:
    region = hsv[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
    colorful = (region[:, :, 1] >= 70) & (region[:, :, 2] >= 40)
    if colorful.sum() < 50:
        return []
    hist = cv2.calcHist([region], [0], colorful.astype(np.uint8) * 255, [180], [0, 180]).ravel()
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(np.concatenate([hist[-5:], hist, hist[:5]]), kernel, mode="same")[5:-5]
    peak = float(smooth.max())
    if peak <= 0:
        return []
    threshold = max(peak * 0.18, colorful.sum() / 180.0)
    candidates: list[tuple[float, int]] = []
    for h in range(180):
        left = smooth[(h - 1) % 180]
        right = smooth[(h + 1) % 180]
        if smooth[h] >= threshold and smooth[h] >= left and smooth[h] >= right:
            candidates.append((float(smooth[h]), h))
    candidates.sort(reverse=True)
    chosen: list[int] = []
    for _, h in candidates:
        if all(min(abs(h - c), 180 - abs(h - c)) > 10 for c in chosen):
            chosen.append(h)
        if len(chosen) >= limit:
            break
    return chosen


def _hue_mask(hsv: np.ndarray, hue: int, tol: int = 8) -> np.ndarray:
    lower = (hue - tol) % 180
    upper = (hue + tol) % 180
    spans = [(lower, upper)] if lower <= upper else [(0, upper), (lower, 179)]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in spans:
        part = cv2.inRange(
            hsv,
            np.array([lo, 60, 40], dtype=np.uint8),
            np.array([hi, 255, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, part)
    return mask > 0


def extract_color_curves(
    bgr: np.ndarray,
    box: PlotBox,
    *,
    max_curves: int = 8,
    min_area: int = 120,
    max_points: int = 120,
) -> list[CurveCandidate]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    peaks = _hue_peaks(hsv, box, max_curves)
    curves: list[CurveCandidate] = []
    for hue in peaks:
        found = _components(
            _hue_mask(hsv, hue),
            box,
            min_area=min_area,
            max_points=max_points,
            label_prefix="curve",
            bgr=bgr,
        )
        curves.extend(_merge_close(found))
    curves.sort(key=lambda c: -len(c.points))
    for i, curve in enumerate(curves):
        curve.label = f"curve {i + 1}"
        if _looks_like_annotation(curve):
            curve.kind = "annotation"
            curve.warnings.append("axis-aligned marker, not a curve")
    return curves[: max_curves * 2]


def _looks_like_annotation(curve: CurveCandidate) -> bool:
    pts = curve.points
    if len(pts) < 2:
        return False
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    if width < 1e-6 and height < 1e-6:
        return False
    if height < 2.5 and width > 30:
        return True
    if width < 2.5 and height > 30:
        return True
    diagonal = math.hypot(width, height)
    if diagonal < 30:
        return False
    path = sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )
    return path > 1.15 * diagonal


def _strip_long_lines(mask: np.ndarray, box: PlotBox) -> np.ndarray:
    u8 = mask.astype(np.uint8) * 255
    kv = max(10, int(0.25 * box.height))
    kh = max(10, int(0.25 * box.width))
    vert = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv)))
    horz = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)))
    lines = cv2.bitwise_or(vert, horz)
    cleaned = cv2.subtract(u8, lines)
    return cleaned > 0


def extract_ink_curves(
    ink: np.ndarray,
    box: PlotBox,
    *,
    max_curves: int = 8,
    min_area: int = 150,
    max_points: int = 120,
) -> list[CurveCandidate]:
    inner = np.zeros_like(ink)
    margin = 2
    inner[box.y0 + margin : box.y1 - margin, box.x0 + margin : box.x1 - margin] = True
    masked = ink & inner
    if not masked.any():
        return []
    cleaned = _strip_long_lines(masked, box)
    curves = _components(
        cleaned, box, min_area=min_area, max_points=max_points, label_prefix="ink"
    )
    for i, curve in enumerate(curves):
        curve.label = f"ink {i + 1}"
        if curve.score > 0.75 and len(curve.points) < 0.2 * box.width:
            curve.warnings.append("possible text or marker cluster")
    return curves[:max_curves]


def _straightness(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Return (slope, rms residual in px) of a least-squares straight fit."""
    if len(points) < 3:
        return math.inf, math.inf
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if xs.max() - xs.min() < 1e-6:
        return math.inf, math.inf
    slope, intercept = np.polyfit(xs, ys, 1)
    residual = ys - (slope * xs + intercept)
    return float(slope), float(np.sqrt(np.mean(residual**2)))


def _slope_of(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.inf
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if xs.max() - xs.min() < 1e-6:
        return math.inf
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def _mean_distance(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    if not a or not b:
        return math.inf
    b_arr = np.array(b, dtype=float)
    best_per_point = []
    for x, y in a:
        d = np.hypot(b_arr[:, 0] - x, b_arr[:, 1] - y)
        best_per_point.append(float(d.min()))
    return float(np.median(best_per_point))


def _is_duplicate(
    curve: CurveCandidate, existing: list[CurveCandidate], tolerance: float
) -> bool:
    for other in existing:
        if _mean_distance(curve.points, other.points) <= tolerance:
            return True
    return False


def _row_runs(mask: np.ndarray, y: int, x0: int, x1: int, max_width: int) -> list[tuple[float, float]]:
    row = mask[y, x0:x1]
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    runs: list[tuple[float, float]] = []
    start = prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v - prev <= 2:
            prev = v
            continue
        width = prev - start + 1
        if width <= max_width:
            runs.append((x0 + (start + prev) / 2.0, float(width)))
        start = prev = v
    width = prev - start + 1
    if width <= max_width:
        runs.append((x0 + (start + prev) / 2.0, float(width)))
    return runs


def _enforce_monotonic(
    points: list[tuple[float, float]], tolerance: float = 2.0
) -> list[tuple[float, float]]:
    """Drop backtracking points caused by grid crossings."""
    if len(points) < 3:
        return points
    direction = 1.0 if points[-1][0] >= points[0][0] else -1.0
    out = [points[0]]
    for x, y in points[1:]:
        prev_x = out[-1][0]
        if (x - prev_x) * direction < 0:
            if abs(x - prev_x) <= tolerance:
                continue
            if len(out) >= 2 and abs(x - out[-2][0]) * direction > 0:
                out.pop()
                if (x - out[-1][0]) * direction < 0:
                    continue
            else:
                continue
        out.append((x, y))
    return out


def trace_curves_by_rows(
    mask: np.ndarray,
    box: PlotBox,
    *,
    max_curves: int = 12,
    max_points: int = 60,
    max_width: int = 16,
    max_jump: float = 12.0,
    min_slope: float = 0.25,
    max_slope: float = 60.0,
    max_residual_px: float = 3.5,
    step: int = 2,
    min_height_frac: float = 0.5,
) -> list[CurveCandidate]:
    x0, x1 = box.x0 + 3, box.x1 - 2
    ys = list(range(box.y0 + 3, box.y1 - 2, step))
    if not ys:
        return []

    runs_by_row: dict[int, list[tuple[float, float]]] = {}
    for y in ys:
        runs = _row_runs(mask, y, x0, x1, max_width)
        if runs:
            runs_by_row[y] = runs
    if len(runs_by_row) < 0.25 * len(ys):
        return []

    active: list[dict] = []
    finished: list[dict] = []
    for y in ys:
        runs = runs_by_row.get(y, [])
        used: set[int] = set()
        next_active: list[dict] = []
        for track in active:
            predicted = track["x"] + track["slope"] * step
            best_i = -1
            best_d = max_jump + 1.0
            for i, (cx, _w) in enumerate(runs):
                if i in used:
                    continue
                d = abs(cx - predicted)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                used.add(best_i)
                cx = runs[best_i][0]
                new_slope = track["slope"] * 0.7 + (cx - track["x"]) / step * 0.3
                track["slope"] = float(np.clip(new_slope, -max_slope, max_slope))
                track["x"] = cx
                track["points"].append((cx, y + 0.5))
                track["misses"] = 0
                next_active.append(track)
            else:
                track["misses"] += 1
                if track["misses"] <= 2:
                    track["x"] = predicted
                    next_active.append(track)
                else:
                    finished.append(track)
        for i, (cx, _w) in enumerate(runs):
            if i in used:
                continue
            next_active.append(
                {"x": cx, "slope": 0.0, "points": [(cx, y + 0.5)], "misses": 0}
            )
        active = next_active

    finished.extend(active)

    candidates: list[CurveCandidate] = []
    min_points = max(12, int(0.35 * len(ys)))
    for track in finished:
        points = track["points"]
        if len(points) < min_points:
            continue
        height = points[-1][1] - points[0][1]
        if height < min_height_frac * box.height:
            continue
        points = _enforce_monotonic(points)
        if len(points) < min_points:
            continue
        slope, residual = _straightness(points)
        if not math.isfinite(slope):
            continue
        if abs(slope) < min_slope or abs(slope) > max_slope:
            continue
        if residual > max_residual_px:
            continue
        candidates.extend(
            [
                CurveCandidate(
                    label="line",
                    color="#111827",
                    points=_simplify(points, 1.0, max_points),
                    kind="line",
                    score=min(1.0, height / float(box.height)),
                    slope=slope,
                )
            ]
        )

    kept: list[CurveCandidate] = []
    for curve in sorted(candidates, key=lambda c: -len(c.points)):
        if _is_duplicate(curve, kept, tolerance=0.01 * box.width):
            continue
        kept.append(curve)
        if len(kept) >= max_curves:
            break

    kept.sort(key=lambda c: min(p[0] for p in c.points))
    for i, curve in enumerate(kept):
        curve.label = f"line {i + 1}"
    return kept


def extract_line_curves(
    ink: np.ndarray,
    box: PlotBox,
    *,
    max_curves: int = 12,
    max_points: int = 60,
    min_coverage: float = 0.55,
    max_abs_slope: float = 45.0,
    min_abs_slope: float = 0.2,
    max_residual_px: float = 6.0,
) -> list[CurveCandidate]:
    inner = np.zeros_like(ink)
    margin = 3
    inner[box.y0 + margin : box.y1 - margin, box.x0 + margin : box.x1 - margin] = True
    masked = ink & inner
    if masked.sum() < 200:
        return []
    cleaned = _strip_long_lines(masked, box)
    if cleaned.sum() < 200:
        return []
    u8 = (cleaned.astype(np.uint8)) * 255
    u8 = cv2.dilate(u8, np.ones((3, 3), np.uint8), iterations=1)
    raw = cv2.HoughLines(u8, 1, np.pi / 1440, threshold=120)
    if raw is None:
        return []

    found: list[CurveCandidate] = []
    for line in raw[:, 0]:
        rho, theta = float(line[0]), float(line[1])
        curve = _track_along_line(masked, box, rho, theta, max_points=max_points)
        if curve is None or curve.score < min_coverage:
            continue
        slope, residual = _straightness(curve.points)
        if abs(slope) > max_abs_slope or abs(slope) < min_abs_slope:
            continue
        if residual > max_residual_px:
            continue
        span_x = max(p[0] for p in curve.points) - min(p[0] for p in curve.points)
        if span_x < 0.25 * box.width:
            continue
        if _is_duplicate(curve, found, tolerance=0.012 * box.width):
            continue
        curve.slope = slope
        curve.angle = math.degrees(theta) % 180.0
        curve.rho = rho
        found.append(curve)
        if len(found) >= max_curves:
            break

    found.sort(key=lambda c: min(p[0] for p in c.points))
    for i, curve in enumerate(found):
        curve.label = f"line {i + 1}"
        curve.kind = "line"
    return found

def _track_along_line(
    mask: np.ndarray,
    box: PlotBox,
    rho: float,
    theta: float,
    *,
    max_points: int,
    window: int = 6,
) -> CurveCandidate | None:
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    x_anchor = rho * cos_t
    y_anchor = rho * sin_t
    if abs(cos_t) < 1e-6:
        return None
    slope = -cos_t / sin_t if abs(sin_t) > 1e-6 else math.inf

    rows, cols = mask.shape
    points: list[tuple[float, float]] = []
    hits = 0
    total = 0

    steep = abs(slope) >= 1.0
    if steep:
        inv = 1.0 / slope
        for y in range(box.y0 + 2, box.y1 - 1, 2):
            x_pred = x_anchor + (y - y_anchor) * inv
            if not (box.x0 + 1 < x_pred < box.x1 - 1):
                continue
            total += 1
            lo = max(0, int(x_pred) - window)
            hi = min(cols, int(x_pred) + window + 1)
            row = mask[y, lo:hi]
            idx = np.flatnonzero(row)
            if idx.size == 0:
                continue
            hits += 1
            points.append((lo + float(np.median(idx)) + 0.5, y + 0.5))
    else:
        for x in range(box.x0 + 2, box.x1 - 1, 2):
            y_pred = y_anchor + slope * (x - x_anchor)
            if not (box.y0 + 1 < y_pred < box.y1 - 1):
                continue
            total += 1
            lo = max(0, int(y_pred) - window)
            hi = min(rows, int(y_pred) + window + 1)
            column = mask[lo:hi, x]
            idx = np.flatnonzero(column)
            if idx.size == 0:
                continue
            hits += 1
            points.append((x + 0.5, lo + float(np.median(idx)) + 0.5))
    if total < 20 or len(points) < 6:
        return None
    coverage = hits / float(total)
    points.sort(key=lambda p: p[0])
    return CurveCandidate(
        label="line",
        color="#111827",
        points=_simplify(points, 1.2, max_points),
        kind="line",
        score=coverage,
        angle=math.degrees(theta) % 180.0,
        rho=rho,
    )


def extract_curves(
    bgr: np.ndarray,
    ink: np.ndarray,
    box: PlotBox,
    *,
    max_curves: int = 8,
    max_points: int = 120,
    mode: str = "auto",
) -> list[CurveCandidate]:
    color_curves: list[CurveCandidate] = []
    if mode in ("auto", "color"):
        color_curves = extract_color_curves(bgr, box, max_curves=max_curves, max_points=max_points)
    if mode == "color":
        return color_curves
    if mode == "auto" and len(color_curves) >= 2:
        return color_curves

    inner = np.zeros_like(ink)
    margin = 3
    inner[box.y0 + margin : box.y1 - margin, box.x0 + margin : box.x1 - margin] = True
    degridded = _strip_long_lines(ink & inner, box)
    traced = trace_curves_by_rows(
        degridded, box, max_curves=max_curves, max_points=max_points
    )
    line_curves = traced or extract_line_curves(
        ink, box, max_curves=max_curves, max_points=max_points
    )
    if line_curves:
        return color_curves + line_curves
    if mode == "lines":
        return []
    if color_curves:
        return color_curves
    return extract_ink_curves(ink, box, max_curves=max_curves, max_points=max_points)


def curve_rmse(points: list[tuple[float, float]], truth: list[tuple[float, float]]) -> float:
    if not points or not truth:
        return math.inf
    truth_arr = np.array(truth, dtype=float)
    errors = []
    for x, y in points:
        i = int(np.argmin(np.abs(truth_arr[:, 0] - x)))
        errors.append((truth_arr[i, 1] - y) ** 2)
    return float(math.sqrt(float(np.mean(errors))))