"""RapidOCR wrapper.

The engine is expensive to construct, so it is created lazily and reused.
All coordinates returned by :meth:`read` are in the coordinate space of the
image that was passed in (the internal upscale is undone).
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from .model import TextLine

logging.getLogger("RapidOCR").setLevel(logging.ERROR)
logging.getLogger("RapidOCR").propagate = False

_DEFAULT_TARGET_WIDTH = 1400


class Ocr:
    _engine = None
    _lock = threading.Lock()

    @classmethod
    def engine(cls):
        with cls._lock:
            if cls._engine is None:
                from rapidocr import RapidOCR

                cls._engine = RapidOCR()
            return cls._engine

    def __init__(self, target_width: int = _DEFAULT_TARGET_WIDTH) -> None:
        self.target_width = target_width

    @staticmethod
    def choose_scale(width: int, target_width: int = _DEFAULT_TARGET_WIDTH) -> float:
        return max(2.0, min(4.0, target_width / max(width, 1)))

    def read(self, img: np.ndarray, scale: float | None = None) -> list[TextLine]:
        """OCR an image region and return text lines in input coordinates."""
        if img is None or img.size == 0:
            return []
        h, w = img.shape[:2]
        if scale is None:
            scale = self.choose_scale(w, self.target_width)
        if scale != 1.0:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
        res = self.engine()(img)
        boxes = getattr(res, "boxes", None)
        texts = getattr(res, "txts", None)
        scores = getattr(res, "scores", None)
        if boxes is None or texts is None:
            return []
        out: list[TextLine] = []
        for box, text, conf in zip(boxes, texts, scores if scores is not None else [1.0] * len(texts)):
            xs = [float(p[0]) / scale for p in box]
            ys = [float(p[1]) / scale for p in box]
            txt = str(text).strip()
            if not txt:
                continue
            out.append(
                TextLine(
                    text=txt,
                    x1=int(round(min(xs))),
                    y1=int(round(min(ys))),
                    x2=int(round(max(xs))),
                    y2=int(round(max(ys))),
                    conf=float(conf),
                )
            )
        out.sort(key=lambda t: (t.cy, t.cx))
        return out


_ocr: Ocr | None = None


def get_ocr() -> Ocr:
    global _ocr
    if _ocr is None:
        _ocr = Ocr()
    return _ocr
