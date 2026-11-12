from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from aiplot.curves import _looks_like_annotation, trace_curves_by_rows
from aiplot.frame import PlotBox
from aiplot.recognize import recognize

VALVE = Path(__file__).resolve().parents[2] / "in" / "valve.png"

pytestmark = pytest.mark.skipif(not VALVE.is_file(), reason="valve.png not available")


def _data():
    return VALVE.read_bytes()


@pytest.fixture(scope="module")
def valve():
    return recognize(
        _data(),
        use_ocr=False,
        x_range=(3.0, 3000.0),
        y_range=(10.0, 1000.0),
        x_scale="log",
        y_scale="log",
    )


def test_valve_frame_is_exact():
    result = recognize(_data(), use_ocr=False)
    assert (result.box.x0, result.box.y0, result.box.x1, result.box.y1) == (156, 131, 944, 1056)


def test_valve_seven_log_curves(valve):
    assert len(valve.curves) == 8
    lines = [c for c in valve.curves if c.kind == "line"]
    assert len(lines) == 7


def test_valve_curves_are_ordered_top_to_bottom(valve):
    for curve in valve.curves:
        if curve.kind != "line":
            continue
        ys = [p[1] for p in curve.points]
        assert ys[0] < ys[-1]


def test_exported_points_ascend_in_x(valve):
    for entry in valve.base_curves():
        xs = [p["pixel"][0] for p in entry["points"]]
        assert xs == sorted(xs)
    for entry in valve.data_points():
        xs = [p["x"] for p in entry["points"]]
        assert xs == sorted(xs)


def test_valve_curves_span_the_plot(valve):
    for curve in valve.curves:
        if curve.kind != "line":
            continue
        ys = [p[1] for p in curve.points]
        assert max(ys) - min(ys) > 0.5 * valve.box.height


def test_valve_calibration_matches_printed_example(valve):
    """Orange example markers: q=50 kg/h and dp=6000 Pa must land on their pixels."""
    x_axis_refs = valve.calibration["x"]["ref_points"]
    y_axis_refs = valve.calibration["y"]["ref_points"]
    assert x_axis_refs[0]["value"] == pytest.approx(3.0)
    assert x_axis_refs[1]["value"] == pytest.approx(3000.0)
    assert y_axis_refs[0]["value"] == pytest.approx(1000.0)
    assert y_axis_refs[1]["value"] == pytest.approx(10.0)

    from aiplot.calibration import axis_from_ref_points

    x_axis = axis_from_ref_points(valve.calibration["x"], 0)
    y_axis = axis_from_ref_points(valve.calibration["y"], 1)
    assert x_axis is not None and y_axis is not None

    assert x_axis.to_value(477.0) == pytest.approx(50.0, rel=0.03)
    assert y_axis.to_value(696.0) == pytest.approx(60.0, rel=0.03)


def test_valve_curves_are_monotonic_in_data_space(valve):
    curves = valve.data_points()
    assert len(curves) == 7
    for entry in curves:
        xs = [p["x"] for p in entry["points"]]
        normalized = [math.copysign(1.0, xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
        flips = sum(1 for a, b in zip(normalized, normalized[1:]) if a != b)
        assert flips <= 2, f"{entry['label']} is not monotonic"


def test_lines_are_not_duplicates(valve):
    lines = [c for c in valve.curves if c.kind == "line"]
    for i, a in enumerate(lines):
        for b in lines[i + 1 :]:
            ax = min(p[0] for p in a.points)
            bx = min(p[0] for p in b.points)
            assert abs(ax - bx) > 0.02 * valve.box.width


def test_annotation_detector_flags_axis_aligned_marker():
    from aiplot.curves import CurveCandidate

    horizontal = CurveCandidate(
        label="h",
        color="#f00",
        points=[(10.0, 50.0), (200.0, 50.0), (400.0, 50.0)],
    )
    assert _looks_like_annotation(horizontal)

    legit = CurveCandidate(
        label="l",
        color="#000",
        points=[(10.0, 400.0), (60.0, 300.0), (110.0, 200.0), (160.0, 100.0)],
    )
    assert not _looks_like_annotation(legit)


def test_row_tracing_needs_clean_mask():
    """A grid-lined mask must not produce spurious straight lines."""
    mask = np.zeros((300, 400), dtype=bool)
    for x in range(10, 390, 20):
        mask[:, x] = True
    for y in range(10, 290, 20):
        mask[y, :] = True
    box = PlotBox(5, 5, 395, 295)
    assert trace_curves_by_rows(mask, box, max_residual_px=1.0) == []