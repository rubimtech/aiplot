from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PLOT_BOX = (130, 70, 900, 640)
X_RANGE = (0.0, 10.0)
Y_RANGE = (0.0, 5.0)


@dataclass
class SyntheticPlot:
    image: np.ndarray
    box: tuple[int, int, int, int]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    truth: dict[str, list[tuple[float, float]]]

    def pixel_of(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        px = x0 + (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (x1 - x0)
        py = y1 - (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (y1 - y0)
        return float(px), float(py)

    def data_of(self, px: float, py: float) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        x = self.x_range[0] + (px - x0) / (x1 - x0) * (self.x_range[1] - self.x_range[0])
        y = self.y_range[0] + (y1 - py) / (y1 - y0) * (self.y_range[1] - self.y_range[0])
        return float(x), float(y)


def _curve_a(x: float) -> float:
    return 2.5 + 2.0 * float(np.exp(-x / 3.0))


def _curve_b(x: float) -> float:
    return 0.5 + 0.45 * x


def render_plot(noise: float = 0.0, grid: bool = True, size: tuple[int, int] = (1000, 760)) -> SyntheticPlot:
    width, height = size
    img = np.full((height, width, 3), 255, np.uint8)
    x0, y0, x1, y1 = PLOT_BOX

    def px_of(x: float) -> float:
        return x0 + (x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * (x1 - x0)

    def py_of(y: float) -> float:
        return y1 - (y - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0]) * (y1 - y0)

    if grid:
        for i in range(11):
            x = int(round(px_of(X_RANGE[0] + i)))
            cv2.line(img, (x, y0), (x, y1), (210, 210, 210), 1, cv2.LINE_AA)
        for i in range(6):
            y = int(round(py_of(Y_RANGE[0] + i)))
            cv2.line(img, (x0, y), (x1, y), (210, 210, 210), 1, cv2.LINE_AA)

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2, cv2.LINE_AA)
    for i in range(11):
        x = int(round(px_of(X_RANGE[0] + i)))
        cv2.line(img, (x, y1), (x, y1 + 10), (0, 0, 0), 2, cv2.LINE_AA)
    for i in range(6):
        y = int(round(py_of(Y_RANGE[0] + i)))
        cv2.line(img, (x0 - 10, y), (x0, y), (0, 0, 0), 2, cv2.LINE_AA)

    xs = np.linspace(X_RANGE[0], X_RANGE[1], 400)
    truth: dict[str, list[tuple[float, float]]] = {"a": [], "b": []}
    for name, fn, color in (("a", _curve_a, (0, 0, 230)), ("b", _curve_b, (230, 60, 0))):
        pts = np.array(
            [[int(round(px_of(float(x)))), int(round(py_of(fn(float(x)))))] for x in xs],
            np.int32,
        )
        cv2.polylines(img, [pts], False, color, 3, cv2.LINE_AA)
        for x in xs:
            truth[name].append((px_of(float(x)), py_of(fn(float(x)))))

    if noise > 0:
        rng = np.random.default_rng(7)
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    return SyntheticPlot(img, PLOT_BOX, X_RANGE, Y_RANGE, truth)


def main() -> None:
    out = Path(__file__).with_name("sample_plot.png")
    plot = render_plot()
    cv2.imwrite(str(out), plot.image)
    print(out)


if __name__ == "__main__":
    main()