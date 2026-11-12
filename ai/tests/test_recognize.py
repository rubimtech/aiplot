from __future__ import annotations

import cv2

from aiplot.curves import curve_rmse
from aiplot.recognize import recognize

from make_plot import render_plot


def _png_bytes(image) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def test_plot_box_detected():
    plot = render_plot()
    result = recognize(_png_bytes(plot.image), use_ocr=False)
    x0, y0, x1, y1 = plot.box
    assert abs(result.box.x0 - x0) <= 4
    assert abs(result.box.y0 - y0) <= 4
    assert abs(result.box.x1 - x1) <= 4
    assert abs(result.box.y1 - y1) <= 4
    assert result.box.has_top and result.box.has_right


def test_ticks_detected():
    plot = render_plot()
    result = recognize(_png_bytes(plot.image), use_ocr=False)
    assert len(result.x_ticks.positions) == 11
    assert len(result.y_ticks.positions) == 6
    x_step = plot.box[2] - plot.box[0]
    assert abs(result.x_ticks.step_px - x_step / 10) <= 2.0
    assert result.x_ticks.regular


def test_coloured_curves_extracted():
    plot = render_plot()
    result = recognize(_png_bytes(plot.image), use_ocr=False)
    assert len(result.curves) == 2
    truth = {name: pts for name, pts in plot.truth.items()}
    for curve in result.curves:
        errors = [curve_rmse(curve.points, pts) for pts in truth.values()]
        assert min(errors) < 3.0


def test_calibration_from_ranges_roundtrip():
    plot = render_plot()
    result = recognize(
        _png_bytes(plot.image),
        use_ocr=False,
        x_range=plot.x_range,
        y_range=plot.y_range,
    )
    assert result.calibration is not None
    assert len(result.calibration["x"]["ref_points"]) == 2

    data_curves = result.data_points()
    assert len(data_curves) == 2
    for entry in data_curves:
        pts = entry["points"]
        xs = [p["x"] for p in pts]
        assert min(xs) > -0.2 and max(xs) < 10.2
        assert all(-0.2 < p["y"] < 5.2 for p in pts)


def test_calibration_is_linear_on_linear_axes():
    plot = render_plot()
    result = recognize(
        _png_bytes(plot.image),
        use_ocr=False,
        x_range=plot.x_range,
        y_range=plot.y_range,
    )
    x_refs = result.calibration["x"]["ref_points"]
    assert x_refs[0]["value"] == 0.0 and x_refs[1]["value"] == 10.0
    y_refs = result.calibration["y"]["ref_points"]
    assert y_refs[0]["value"] == 5.0 and y_refs[1]["value"] == 0.0


def test_base_curve_payload_matches_editor_schema():
    plot = render_plot()
    result = recognize(_png_bytes(plot.image), use_ocr=False)
    payload = result.base_curves()
    assert payload
    for curve in payload:
        assert set(curve) >= {"label", "color", "points", "connect_as", "target_point_count"}
        assert curve["target_point_count"] >= 9
        for point in curve["points"]:
            assert point["origin"] == "ai"
            px, py = point["pixel"]
            assert 0 <= px <= plot.image.shape[1]
            assert 0 <= py <= plot.image.shape[0]


def test_noisy_image_still_detects_box():
    plot = render_plot(noise=6.0)
    result = recognize(_png_bytes(plot.image), use_ocr=False)
    x0, y0, x1, y1 = plot.box
    assert abs(result.box.x0 - x0) <= 6
    assert abs(result.box.y1 - y1) <= 6