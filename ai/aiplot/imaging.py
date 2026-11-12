from __future__ import annotations

import numpy as np
import cv2


def decode_image(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image data")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = otsu == 0
    if ink.mean() > 0.5:
        ink = ~ink
    return ink


def median_thickness(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dist[dist > 0]
    return float(np.median(ridge)) * 2.0