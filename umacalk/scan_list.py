"""Score Info list parsing and the scroll-to-collect loop."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from . import detect
from .model import Rect, TextLine, UmaRow
from .names import NameResolver
from .ocr import Ocr
from .parse import match_distance, parse_int

# Field geometry as fractions of the panel box (measured on real captures and
# identical across windowed / borderless 1920x1080).
NAME_X_MAX = 0.62
SCORE_X_MIN = 0.62
PILL_X_MAX = 0.34
NAME_GAP_MAX = 0.26      # x row pitch, distance from name baseline to bar top


class Driver(Protocol):
    def grab(self) -> np.ndarray: ...
    def scroll_at(self, x: int, y: int, notches: int) -> None: ...
    def click_at(self, x: int, y: int) -> None: ...
    def key(self, key: str) -> None: ...


@dataclass
class ListFrame:
    panel: Rect
    bars: list[detect.Bar]
    pitch: float | None
    rows: list[UmaRow]
    lines: list[TextLine]
    warnings: list[str] = field(default_factory=list)


def _pick_name(cands: list[TextLine], bar: detect.Bar, pitch: float) -> TextLine | None:
    best: tuple[float, TextLine] | None = None
    for ln in cands:
        gap = bar.rect.y - ln.y2
        if -0.05 * pitch <= gap <= NAME_GAP_MAX * pitch:
            score = abs(gap) + 0.001 * ln.h
            if best is None or score < best[0]:
                best = (score, ln)
    return best[1] if best else None


def parse_frame(
    img: np.ndarray,
    ocr: Ocr,
    resolver: NameResolver,
    taken: set[str] | None = None,
) -> ListFrame:
    panel = detect.find_panel(img)
    if panel is None:
        raise RuntimeError("popup panel not found - is the Score Info screen open?")
    crop = img[panel.y:panel.y2, panel.x:panel.x2]
    bars = detect.find_bars(img, panel)  # panel-local coords
    pitch = detect.row_pitch(bars)
    lines = ocr.read(crop)
    warnings: list[str] = []
    if pitch is None:
        return ListFrame(panel, bars, None, [], lines, ["no row bars detected"])

    w, h = panel.w, panel.h
    rows: list[UmaRow] = []
    for bar in bars:
        band_top = bar.rect.y - 0.36 * pitch
        band_bot = bar.rect.y2 + 0.32 * pitch
        in_band = [ln for ln in lines if band_top <= ln.cy <= band_bot]

        score_line = None
        for ln in in_band:
            if ln.x1 < SCORE_X_MIN * w:
                continue
            val = parse_int(ln.text)
            if val is None or not (500 <= val <= 200_000):
                continue
            if score_line is None or ln.x2 > score_line.x2:
                score_line = ln

        # name: big text sitting just above the bar, left of the score column
        name_cands = [
            ln for ln in in_band
            if ln.x2 < NAME_X_MAX * w
            and ln.x1 < 0.40 * w
            and ln.h >= 0.10 * pitch
            and not ln.text.upper().startswith(("MVP", "RANK"))
            and match_distance(ln.text) is None
        ]
        name_line = _pick_name(name_cands, bar, pitch)

        pill_line = None
        for ln in in_band:
            if ln.y1 < bar.rect.y2 - 0.08 * pitch:
                continue
            if ln.x2 > PILL_X_MAX * w:
                continue
            if match_distance(ln.text):
                pill_line = ln
                break

        mvp = any("MVP" in ln.text.upper() for ln in in_band)
        rank = None
        for ln in in_band:
            t = re.sub(r"\s+", " ", ln.text.strip().upper())
            if t.endswith("RANK"):
                t = t[:-4].strip()
            if t in ("S", "A", "B", "C", "D"):
                rank = t
                break

        name = None
        conf = 0.0
        if name_line:
            m = resolver.resolve(name_line.text, taken)
            if m:
                name = m.name
                conf = m.score / 100.0 * min(1.0, name_line.conf)
            else:
                name = name_line.text
                conf = 0.4
                warnings.append(f"unknown name: {name_line.text!r}")
        score = parse_int(score_line.text) if score_line else None
        if score is None:
            warnings.append(f"no score for {name or '?'}")

        rows.append(
            UmaRow(
                name=name or "?",
                score=score,
                distance=match_distance(pill_line.text) if pill_line else None,
                mvp=mvp,
                rank=rank,
                y=bar.rect.cy,
                confidence=round(conf, 3),
                raw={
                    "name": name_line.text if name_line else None,
                    "score": score_line.text if score_line else None,
                    "pill": pill_line.text if pill_line else None,
                },
            )
        )
    return ListFrame(panel, bars, pitch, rows, lines, warnings)


@dataclass
class ScanResult:
    rows: list[UmaRow]
    panel: Rect | None
    warnings: list[str] = field(default_factory=list)
    frames: int = 0
    scrolled_rows: float = 0.0


def scan(
    driver: Driver,
    ocr: Ocr,
    resolver: NameResolver,
    *,
    log: Callable[[str], None] | None = None,
    max_frames: int = 60,
    on_frame: Callable[[ListFrame], None] | None = None,
) -> ScanResult:
    say = log or (lambda _m: None)

    head = parse_frame(driver.grab(), ocr, resolver)
    if head.panel is None:
        return ScanResult(
            [], None,
            ["Score Info screen not found - open Team Trials > Race History > Score Info first"],
        )
    cx = head.panel.x + head.panel.w // 2
    cy = head.panel.y + head.panel.h // 2
    driver.scroll_at(cx, cy, -15)          # rewind to the top
    frame = parse_frame(driver.grab(), ocr, resolver)

    rows: dict[str, UmaRow] = {}
    warnings: list[str] = []
    frames = 0
    stalls = 0
    prev_sig: tuple | None = None

    while frames < max_frames:
        frames += 1
        fresh = 0
        for row in frame.rows:
            if row.name == "?":
                continue
            if row.name not in rows:
                rows[row.name] = row
                fresh += 1
            elif row.score and rows[row.name].score and row.score != rows[row.name].score:
                warnings.append(
                    f"{row.name}: conflicting scores {rows[row.name].score} vs {row.score}")
        warnings.extend(frame.warnings)
        if on_frame:
            on_frame(frame)
        say(f"frame {frames}: {len(frame.rows)} rows in view, {fresh} new, total {len(rows)}")

        sig = tuple(round(b.rect.cy) for b in frame.bars)
        if sig and sig == prev_sig:
            stalls += 1
        else:
            stalls = 0
        prev_sig = sig
        if stalls >= 2:
            break

        driver.scroll_at(cx, cy, 1)
        frame = parse_frame(driver.grab(), ocr, resolver)

    ordered = sorted(rows.values(), key=lambda r: -(r.score or 0))
    return ScanResult(ordered, frame.panel, sorted(set(warnings)), frames)
