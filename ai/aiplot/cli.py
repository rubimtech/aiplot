from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from .push import BaseError, DigitizerClient, push_recognition
from .recognize import recognize


def _parse_range(values: list[float] | None) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) != 2:
        raise SystemExit("range needs exactly two numbers: MIN MAX")
    return float(values[0]), float(values[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiplot",
        description="Recognise curves in a plot image and hand them to the digitizer editor.",
    )
    parser.add_argument("image", help="path to the plot image")
    parser.add_argument("--json", dest="json_path", help="write the full recognition JSON here")
    parser.add_argument("--x-range", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--x-scale", choices=("linear", "log"), default=None)
    parser.add_argument("--y-scale", choices=("linear", "log"), default=None)
    parser.add_argument("--mode", choices=("auto", "color", "lines", "ink"), default="auto")
    parser.add_argument("--overlay", help="write a PNG with the detected box and curves")
    parser.add_argument("--no-ocr", action="store_true", help="skip reading tick labels")
    parser.add_argument("--max-curves", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=120)
    parser.add_argument("--push", action="store_true", help="create a session in the editor")
    parser.add_argument(
        "--base",
        default=os.environ.get("AIPLOT_BASE", "http://127.0.0.1:8000"),
        help="digitizer backend origin (default http://127.0.0.1:8000)",
    )
    parser.add_argument("--list-session", action="store_true", help="print last editor session id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.image)
    if not path.is_file():
        raise SystemExit(f"image not found: {path}")
    data = path.read_bytes()

    result = recognize(
        data,
        x_range=_parse_range(args.x_range),
        y_range=_parse_range(args.y_range),
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        mode=args.mode,
        use_ocr=not args.no_ocr,
        max_curves=args.max_curves,
        max_points=args.max_points,
    )
    payload = result.as_dict()
    payload["data_curves"] = result.data_points()

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.overlay:
        from .render import draw_overlay

        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(args.overlay, draw_overlay(bgr, result))

    summary = {
        "plot_box": payload["plot_box"],
        "x_ticks": len(result.x_ticks.positions),
        "y_ticks": len(result.y_ticks.positions),
        "calibration": "ready" if result.calibration else "pixel-only",
        "curves": [
            {"label": c.label, "points": len(c.points), "color": c.color} for c in result.curves
        ],
        "warnings": result.warnings,
    }
    print(json.dumps(summary, indent=2))

    if args.push:
        try:
            pushed = push_recognition(args.base, path.name, data, result)
        except BaseError as exc:
            print(f"push failed: {exc}\n{exc.body or ''}", file=sys.stderr)
            return 2
        print(json.dumps(pushed, indent=2))
    elif args.list_session:
        with DigitizerClient(args.base) as client:
            last = client.last_session()
            app_url = client.app_url()
        print(json.dumps({"session_id": last.get("id"), "app_url": app_url}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())