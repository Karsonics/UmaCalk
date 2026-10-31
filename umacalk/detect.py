"""Visual detection: popup panels and Score Info rows.

Everything here works in *image* coordinates. UI metrics are expressed as
fractions of the detected panel box, which makes the same code work at any
window size (verified: row pitch is 0.1156 x panel height in both 1920x1080
fullscreen and a ~1287x697 window).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .model import Rect

# --- panel detection -------------------------------------------------------

_PANEL_FILL_MIN = 0.93
_PANEL_ASPECT = (1.05, 2.80)      # h / w
_PANEL_AREA_FRAC = (0.03, 0.75)   # of the frame


def _panel_candidates(img: np.ndarray, thr: int) -> list[tuple[Rect, float]]:
    b, g, r = cv2.split(img)
    mask = ((b > thr) & (g > thr) & (r > thr)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    frame_area = float(w * h)
    out: list[tuple[Rect, float]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.02 * frame_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 40 or ch < 40:
            continue
        fill = area / float(cw * ch)
        if fill < _PANEL_FILL_MIN:
            continue
        aspect = ch / float(cw)
        if not (_PANEL_ASPECT[0] <= aspect <= _PANEL_ASPECT[1]):
            continue
        if not (_PANEL_AREA_FRAC[0] <= area / frame_area <= _PANEL_AREA_FRAC[1]):
            continue
        out.append((Rect(x, y, cw, ch), fill))
    out.sort(key=lambda t: t[0].area, reverse=True)
    return out


def _iou(a: Rect, b: Rect) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float((x2 - x1) * (y2 - y1))
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def find_panels(img: np.ndarray, limit: int = 8) -> list[Rect]:
    """Every plausible panel box, biggest first.

    A popup drawn over another light panel (the Score Info list, for instance)
    produces several candidates, so callers must verify which one is wanted
    instead of trusting the largest.
    """
    out: list[Rect] = []
    for thr in (246, 238, 230, 220, 205, 190):
        for rect, _fill in _panel_candidates(img, thr):
            if any(_iou(rect, seen) > 0.55 for seen in out):
                continue
            out.append(rect)
        if len(out) >= limit:
            break
    out.sort(key=lambda r: r.area, reverse=True)
    return out[:limit]


def find_panel(img: np.ndarray) -> Rect | None:
    """Locate the light rounded popup panel (largest candidate)."""
    panels = find_panels(img, limit=1)
    return panels[0] if panels else None


# --- Score Info rows -------------------------------------------------------

@dataclass
class Bar:
    rect: Rect

    @property
    def y0(self) -> int:
        return self.rect.y

    @property
    def y1(self) -> int:
        return self.rect.y2


def _green_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h > 35) & (h < 90) & (s > 80) & (v > 80)).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def find_bars(img: np.ndarray, panel: Rect) -> list[Bar]:
    """Find the bright green progress bars that anchor every list row."""
    x, y, w, h = panel.as_tuple()
    sub = img[y:y + h, x:x + w]
    mask = _green_mask(sub)
    rowsum = (mask > 0).sum(axis=1)
    min_px = max(12, int(0.12 * w))

    bands: list[tuple[int, int]] = []
    start = None
    for yy, count in enumerate(rowsum):
        if count >= min_px and start is None:
            start = yy
        elif count < min_px and start is not None:
            bands.append((start, yy - 1))
            start = None
    if start is not None:
        bands.append((start, len(rowsum) - 1))

    min_h = max(3, int(0.004 * h))
    max_h = max(min_h + 2, int(0.030 * h))
    bars: list[Bar] = []
    for y0, y1 in bands:
        bh = y1 - y0 + 1
        if bh < min_h or bh > max_h:
            continue
        seg = mask[y0:y1 + 1, :]
        cols = np.where(seg.max(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        bx, bx2 = int(cols.min()), int(cols.max())
        if bx2 - bx < 0.18 * w:
            continue
        bars.append(Bar(Rect(bx, y0, bx2 - bx + 1, bh)))
    bars.sort(key=lambda b: b.rect.y)
    return bars  # panel-local coordinates


def row_pitch(bars: list[Bar]) -> float | None:
    if len(bars) < 2:
        return None
    ys = [b.rect.cy for b in bars]
    diffs = sorted(b2 - b1 for b1, b2 in zip(ys, ys[1:]) if b2 - b1 > 4)
    if not diffs:
        return None
    mid = diffs[len(diffs) // 2]
    # keep diffs close to the median so merged/split detections do not skew it
    close = [d for d in diffs if 0.6 * mid <= d <= 1.6 * mid]
    return float(np.median(close)) if close else float(mid)


def horizontal_separators(img: np.ndarray, panel: Rect) -> list[int]:
    """Rows (panel-local y) that look like card separators inside a panel."""
    x, y, w, h = panel.as_tuple()
    sub = img[y:y + h, x:x + w]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    dark = (gray < 235).astype(np.uint8)
    frac = dark.mean(axis=1)
    ys = np.where(frac > 0.55)[0]
    if ys.size == 0:
        return []
    groups: list[list[int]] = [[int(ys[0])]]
    for v in ys[1:]:
        if v - groups[-1][-1] <= 2:
            groups[-1].append(int(v))
        else:
            groups.append([int(v)])
    return [int(np.mean(g)) for g in groups]
