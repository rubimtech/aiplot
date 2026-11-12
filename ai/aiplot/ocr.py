from __future__ import annotations

import os
import re
import shutil

import cv2
import numpy as np

_NUM = re.compile(r"^-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?$")
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Public\Tesseract-OCR\tesseract.exe",
)


def _binary() -> str | None:
    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _WINDOWS_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def _configure() -> str | None:
    path = _binary()
    if not path:
        return None
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = path
        pytesseract.get_tesseract_version()
        return path
    except Exception:
        return None


def available() -> bool:
    return _configure() is not None


def _read(crop: np.ndarray, invert_for_dark_text: bool = True) -> str:
    if _configure() is None or crop.size == 0:
        return ""
    import pytesseract
    h, w = crop.shape[:2]
    scale = max(1, int(round(48.0 / max(1, min(h, w)))))
    scale = min(scale, 6)
    up = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (3, 3), 0)
    _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert_for_dark_text and np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789.,-+eE"
    text = pytesseract.image_to_string(binary, config=config).strip()
    return text.replace(" ", "")


def parse_number(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    if not text or not _NUM.match(text):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_x_labels(gray: np.ndarray, x0: int, x1: int, y_axis: int, positions: list[float]) -> list[float | None]:
    out: list[float | None] = []
    for pos in positions:
        left = int(max(0, pos - max(18, (x1 - x0) / max(1, len(positions)) / 2)))
        right = int(min(gray.shape[1], pos + max(18, (x1 - x0) / max(1, len(positions)) / 2)))
        top = y_axis + 2
        bottom = min(gray.shape[0], y_axis + 42)
        out.append(parse_number(_read(gray[top:bottom, left:right])))
    return out


def read_y_labels(gray: np.ndarray, x_axis: int, positions: list[float]) -> list[float | None]:
    out: list[float | None] = []
    for pos in positions:
        top = int(max(0, pos - 14))
        bottom = int(min(gray.shape[0], pos + 14))
        left = max(0, x_axis - 70)
        right = max(1, x_axis - 3)
        out.append(parse_number(_read(gray[top:bottom, left:right])))
    return out