"""Score Info list parsing and the scroll-to-collect loop."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from . import detect
from .model import DISTANCES, Rect, TextLine, UmaRow
from .names import NameResolver
from .ocr import Ocr
from .parse import match_distance, parse_int

from rapidfuzz import fuzz

# Field geometry as fractions of the panel box (measured on real captures and
# identical across windowed / borderless 1920x1080).
NAME_X_MAX = 0.62
SCORE_X_MIN = 0.62
PILL_X_MAX = 0.34
NAME_GAP_MAX = 0.26      # x row pitch, distance from name baseline to bar top
EDGE_INSET = 0.025       # rows clipped by the panel edge are not trustworthy
MAX_UMAS = 15            # 5 distances x 3 slots
RAW_SIMILAR = 88.0       # same raw text read twice under different matches



class Driver(Protocol):
    def grab(self) -> np.ndarray: ...
    def scroll_at(self, x: int, y: int, notches: int) -> None: ...
    def click_at(self, x: int, y: int) -> None: ...
    def key(self, key: str) -> None: ...


class PanelGone(RuntimeError):
    """The Score Info panel was not visible in a captured frame."""


def _grab_retry(driver: Driver, ocr: Ocr, resolver: NameResolver,
                tries: int = 3) -> ListFrame | None:
    """Retry only transient panel-detection misses, not OCR or driver failures."""
    for attempt in range(max(1, tries)):
        try:
            return parse_frame(driver.grab(), ocr, resolver)
        except PanelGone:
            if attempt + 1 < max(1, tries):
                time.sleep(0.12)
    return None


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


def _twin(rows: dict[str, UmaRow], row: UmaRow) -> UmaRow | None:
    """An already-accepted row whose raw OCR text is the same string.

    A clipped or half-scrolled row can OCR into something that fuzzy-matches a
    *different* character, so the same physical row gets stored twice. Comparing
    the raw text catches that; comparing resolved names never would.
    """
    raw = (row.raw or {}).get("name")
    if not raw:
        return None
    for other in rows.values():
        other_raw = (other.raw or {}).get("name")
        if other_raw and fuzz.WRatio(raw, other_raw) >= RAW_SIMILAR:
            return other
    return None


def parse_frame(
    img: np.ndarray,
    ocr: Ocr,
    resolver: NameResolver,
    taken: set[str] | None = None,
) -> ListFrame:
    panel = detect.find_panel(img)
    if panel is None:
        raise PanelGone("popup panel not found - is the Score Info screen open?")
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
        # A row whose bar touches the panel edge is clipped by the viewport: its
        # name/score text is half cut and OCRs into a different (wrong) uma.
        if bar.rect.y < EDGE_INSET * h or bar.rect.y2 > (1 - EDGE_INSET) * h:
            continue
        band_top = bar.rect.y - 0.36 * pitch
        band_bot = bar.rect.y2 + 0.32 * pitch
        in_band = [ln for ln in lines if band_top <= ln.cy <= band_bot]

        score_line = None
        for ln in in_band:
            if ln.x1 < SCORE_X_MIN * w:
                continue
            if ln.y1 <= 1 or ln.y2 >= h - 1:
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
            and ln.y1 > 1 and ln.y2 < h - 1
            and len(ln.text.strip()) >= 4
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

    head = _grab_retry(driver, ocr, resolver)
    if head is None:
        return ScanResult(
            [], None,
            ["Score Info screen not found - open Team Trials > Race History > Score Info first"],
        )
    cx = head.panel.x + head.panel.w // 2
    cy = head.panel.y + head.panel.h // 2
    driver.scroll_at(cx, cy, 30)           # rewind: positive == up
    frame = _grab_retry(driver, ocr, resolver)
    if frame is None:
        return ScanResult([], head.panel, ["Score Info panel not visible after resetting the list"])

    rows: dict[str, UmaRow] = {}
    warnings: list[str] = []
    frames = 0
    quiet = 0            # consecutive frames that neither moved nor added rows
    missing = 0          # consecutive frames with no panel at all
    prev_nav: tuple = ()

    while frames < max_frames:
        frames += 1
        if frame.panel is None:
            missing += 1
            if missing >= 2:
                warnings.append("Score Info panel disappeared - scan stopped")
                break
            frame = _grab_retry(driver, ocr, resolver)
            if frame is None:
                warnings.append("Score Info panel disappeared - scan stopped")
                break
            continue
        missing = 0

        fresh = 0
        for row in frame.rows:
            if row.name == "?":
                continue
            existing = rows.get(row.name)
            if existing is None:
                twin = _twin(rows, row)
                if twin is not None:
                    if row.score and twin.score and row.score != twin.score:
                        warnings.append(
                            f"{twin.name}: conflicting scores {twin.score} vs {row.score}")
                    if row.confidence > twin.confidence:
                        twin.score = row.score or twin.score
                        twin.confidence = row.confidence
                        twin.distance = row.distance or twin.distance
                        twin.raw = row.raw
                    continue
                rows[row.name] = row
                fresh += 1
                say(f"  + {row.name} (raw={row.raw.get('name')!r}, conf={row.confidence:.2f})")
            elif row.score and existing.score and row.score != existing.score:
                warnings.append(
                    f"{row.name}: conflicting scores {existing.score} vs {row.score}")
        warnings.extend(frame.warnings)
        if on_frame:
            on_frame(frame)
        say(f"frame {frames}: {len(frame.rows)} rows in view, {fresh} new, total {len(rows)}")

        # bars within 3px count as the same position
        nav = tuple(round(b.rect.cy / 3) for b in frame.bars)
        if fresh == 0 and nav == prev_nav and frame.rows:
            quiet += 1
        else:
            quiet = 0
        prev_nav = nav
        if quiet >= 2:
            break
        if len(rows) >= MAX_UMAS and fresh == 0:
            break

        driver.scroll_at(cx, cy, -1)       # down one notch
        frame = _grab_retry(driver, ocr, resolver)
        if frame is None:
            warnings.append("Score Info panel disappeared after scrolling - scan stopped")
            break

    if len(rows) < 3:
        warnings.append(f"only {len(rows)} umas read - the list may not have scrolled")
    if len(rows) > MAX_UMAS:
        warnings.append(
            f"{len(rows)} umas read but at most {MAX_UMAS} can race - "
            f"kept the {MAX_UMAS} most confident readings")
        keep = sorted(rows.values(), key=lambda r: -r.confidence)[:MAX_UMAS]
        rows = {r.name: r for r in keep}

    # the game allows three umas per distance, so a fourth reading in one band
    # is a duplicate of a row read twice, not a real uma
    for dist in DISTANCES:
        group = [r for r in rows.values() if r.distance == dist]
        if len(group) > 3:
            group.sort(key=lambda r: -r.confidence)
            for extra in group[3:]:
                warnings.append(
                    f"{extra.name}: {len(group)} umas read for {dist}, keeping 3 - dropped")
                rows.pop(extra.name, None)

    ordered = sorted(rows.values(), key=lambda r: -(r.score or 0))
    return ScanResult(ordered, frame.panel, sorted(set(warnings)), frames)
