"""Score Details panel parsing and the scroll-to-collect loop.

A card is only trusted when its displayed sub-rows sum exactly to its
displayed total; that identity doubles as the completeness test, because a
card clipped by the viewport edge loses a row and fails the sum.

Ownership (which uma a card belongs to) is state carried across frames: when
scrolling, the uma header is usually off-screen, so a per-frame parse could
not attribute anything.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import detect
from .model import DetailCard, RaceDetail, Rect, ScoreLine, TextLine, UmaDetail
from .ocr import Ocr
from .parse import card_category, clean, match_subrow, parse_int, parse_race_header

VALUE_X_MIN = 0.82      # right-aligned value column
LABEL_X_MAX = 0.72
TEAM_OWNER = "__team__"
HEADER_BAND = 0.18      # top fraction of a panel that carries "Class N, Race N"


class PanelGone(RuntimeError):
    """No Score Details popup could be identified in the frame."""


@dataclass
class Pair:
    label: TextLine
    value: TextLine
    value_int: int
    kind: str          # "card" | "row"
    subrow: str | None = None


@dataclass
class UmaHeader:
    name: str
    gained: int | None
    placement: str | None
    y: float


@dataclass
class FrameParse:
    panel: Rect
    lines: list[TextLine]
    header_label: str | None = None
    class_no: int | None = None
    race_no: int | None = None
    total: int | None = None
    cards: list[DetailCard] = field(default_factory=list)
    umas: list[UmaHeader] = field(default_factory=list)
    seq: list[tuple[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def signature(self) -> tuple:
        return tuple((c.title, c.total) for c in self.cards)


def _pair_rows(lines: list[TextLine], w: int) -> list[Pair]:
    values = [
        ln for ln in lines
        if ln.x2 > VALUE_X_MIN * w and parse_int(ln.text, signed=True) is not None
    ]
    labels = [ln for ln in lines if ln not in values and ln.x2 < LABEL_X_MAX * w]
    used: set[int] = set()
    pairs: list[Pair] = []
    for v in values:
        best: tuple[float, int] | None = None
        for idx, ln in enumerate(labels):
            if idx in used:
                continue
            dy = abs(ln.cy - v.cy)
            if dy <= 0.65 * max(ln.h, v.h):
                if best is None or dy < best[0]:
                    best = (dy, idx)
        if best is None:
            continue
        idx = best[1]
        used.add(idx)
        label = labels[idx]
        val = parse_int(v.text, signed=True)
        if val is None:
            continue
        sub = match_subrow(label.text)
        pairs.append(
            Pair(
                label=label,
                value=v,
                value_int=val,
                kind="row" if sub else "card",
                subrow=sub,
            )
        )
    pairs.sort(key=lambda p: p.label.cy)
    return pairs


def _find_uma_headers(lines: list[TextLine], w: int) -> list[UmaHeader]:
    headers: list[UmaHeader] = []
    for ln in lines:
        if "gained score" not in clean(ln.text).lower():
            continue
        gained = None
        for cand in lines:
            if cand is ln:
                continue
            if 0 <= cand.y1 - ln.y2 <= 1.8 * ln.h and cand.x1 > 0.3 * w:
                gained = parse_int(cand.text)
                if gained is not None:
                    break
        name = None
        bestx = 1e9
        for cand in lines:
            if cand is ln:
                continue
            if not (ln.cy - 4.0 * ln.h <= cand.cy <= ln.cy - 0.4 * ln.h):
                continue
            if cand.x1 > 0.16 * w:
                continue
            txt = clean(cand.text)
            if not txt or parse_int(txt) is not None or match_subrow(txt):
                continue
            if cand.x1 < bestx:
                bestx = cand.x1
                name = txt
        placement = None
        for cand in lines:
            if ln.cy - 4.0 * ln.h <= cand.cy <= ln.cy + 1.0 * ln.h and cand.x1 < 0.45 * w:
                t = clean(cand.text)
                if t[:1].isdigit() and t[-2:].lower() in ("st", "nd", "rd", "th"):
                    placement = t
                    break
        if name:
            headers.append(UmaHeader(name, gained, placement, ln.cy))
    return headers


def _locate(img: np.ndarray, ocr: Ocr) -> tuple[Rect, list[TextLine]] | None:
    """Pick the panel that is really the Score Details popup.

    The popup floats over other light UI, so the largest near-white rectangle
    is often the list behind it. A candidate is accepted only when its header
    band carries a "Class N, Race N" label.
    """
    candidates = detect.find_panels(img)
    fallback: Rect | None = None
    for panel in candidates:
        crop = img[panel.y:panel.y2, panel.x:panel.x2]
        band = crop[: max(8, int(HEADER_BAND * panel.h)), :]
        head = ocr.read(band, scale=3.0)
        if any(parse_race_header(ln.text)[1] is not None for ln in head):
            return panel, ocr.read(crop)
        if fallback is None:
            fallback = panel
    if fallback is None:
        return None
    # no race header anywhere: accept the largest panel only if it looks like
    # a score breakdown (covers screens whose header OCRs poorly)
    crop = img[fallback.y:fallback.y2, fallback.x:fallback.x2]
    lines = ocr.read(crop)
    text = " ".join(clean(ln.text).lower() for ln in lines)
    if "gained score" in text or "team bonus" in text:
        return fallback, lines
    return None


def parse_frame(img: np.ndarray, ocr: Ocr) -> FrameParse:
    found = _locate(img, ocr)
    if found is None:
        raise PanelGone("popup panel not found - is a Score Details screen open?")
    panel, lines = found
    w, h = panel.w, panel.h
    fp = FrameParse(panel=panel, lines=lines)

    for ln in lines:
        if ln.y1 < 0.16 * h:
            cls, race = parse_race_header(ln.text)
            if race is not None:
                fp.class_no, fp.race_no = cls, race
                fp.header_label = clean(ln.text)
    if fp.total is None:
        for ln in lines:
            if ln.y1 >= 0.16 * h or ln.h <= 0.035 * h:
                continue
            val = parse_int(ln.text)
            if val and val > 1000:
                fp.total = val
                break

    headers = _find_uma_headers(lines, w)
    pairs = _pair_rows(lines, w)

    events: list[tuple[float, str, object]] = []
    for hd in headers:
        events.append((hd.y, "uma", hd))
    for ln in lines:
        if "team bonus" in clean(ln.text).lower():
            events.append((ln.cy, "team", ln))
    for p in pairs:
        events.append((p.label.cy, "pair", p))
    events.sort(key=lambda e: e[0])

    current: DetailCard | None = None
    owner_is_team = True
    for _y, kind, obj in events:
        if kind == "uma":
            hd = obj  # type: ignore[assignment]
            fp.umas.append(hd)
            fp.seq.append(("uma", hd))
            owner_is_team = False
            current = None
            continue
        if kind == "team":
            fp.seq.append(("team", obj))
            owner_is_team = True
            current = None
            continue
        p: Pair = obj  # type: ignore[assignment]
        if p.kind == "card":
            card = DetailCard(
                title=clean(p.label.text),
                total=p.value_int,
                category=card_category(p.label.text),
                y=int(p.label.cy),
            )
            fp.cards.append(card)
            fp.seq.append(("card", card))
            current = card
            continue
        if current is not None and p.subrow:
            current.rows.append(
                ScoreLine(label=p.subrow, label_raw=clean(p.label.text), value=p.value_int)
            )

    for card in fp.cards:
        if not card.balanced:
            fp.warnings.append(
                f"incomplete card {card.title!r}: rows {card.row_sum} != total {card.total}"
            )
    return fp


@dataclass
class DetailScan:
    race: RaceDetail | None
    frames: int = 0
    warnings: list[str] = field(default_factory=list)


def _grab(driver, ocr: Ocr, tries: int = 3) -> tuple[FrameParse, np.ndarray] | None:
    """Grab and parse a frame, tolerating a popup that is mid-animation."""
    for attempt in range(max(1, tries)):
        img = driver.grab()
        try:
            return parse_frame(img, ocr), img
        except PanelGone:
            if attempt + 1 >= max(1, tries):
                return None
            time.sleep(0.12)
    return None


def _dump(dump_dir: str | None, race_no: int | None, index: int,
          img: np.ndarray, frame: FrameParse) -> None:
    if not dump_dir:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        stem = f"race{race_no or 0}_{index:02d}"
        cv2.imwrite(os.path.join(dump_dir, stem + ".png"), img)
        payload = {
            "panel": frame.panel.as_tuple(),
            "race": frame.race_no,
            "class": frame.class_no,
            "total": frame.total,
            "cards": [
                {
                    "title": c.title,
                    "total": c.total,
                    "balanced": c.balanced,
                    "rows": [
                        {"label": r.label, "raw": r.label_raw, "value": r.value}
                        for r in c.rows
                    ],
                }
                for c in frame.cards
            ],
            "lines": [
                {"text": ln.text, "x1": ln.x1, "y1": ln.y1,
                 "x2": ln.x2, "y2": ln.y2, "conf": ln.conf}
                for ln in frame.lines
            ],
        }
        with open(os.path.join(dump_dir, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
    except Exception:  # diagnostics must never break a scan
        pass


def scan_panel(
    driver,
    ocr: Ocr,
    *,
    expect_race: int | None = None,
    log: Callable[[str], None] | None = None,
    max_frames: int = 60,
    on_frame: Callable[[FrameParse], None] | None = None,
    dump_dir: str | None = None,
) -> DetailScan:
    say = log or (lambda _m: None)
    got = _grab(driver, ocr)
    if got is None:
        return DetailScan(
            None,
            warnings=["Score Details screen not found - open it before scanning"],
        )
    frame, img = got
    cx = frame.panel.x + frame.panel.w // 2
    cy = frame.panel.y + frame.panel.h // 2
    driver.scroll_at(cx, cy, 30)           # rewind: positive == up
    got = _grab(driver, ocr)
    if got is None:
        return DetailScan(
            None,
            warnings=["Score Details popup vanished before it could be read"],
        )
    frame, img = got

    if expect_race is not None and frame.race_no is not None and frame.race_no != expect_race:
        return DetailScan(
            None,
            warnings=[f"expected race {expect_race}, panel shows race {frame.race_no}"],
        )

    race = RaceDetail(race_no=frame.race_no, class_no=frame.class_no, total=frame.total)
    cards: dict[tuple[str, str, int | None], DetailCard] = {}
    order: list[str] = []
    gained: dict[str, int | None] = {}
    placement: dict[str, str | None] = {}
    team_bonus: DetailCard | None = None
    owner = TEAM_OWNER
    frames = 0
    empty = 0

    while frames < max_frames:
        frames += 1
        if on_frame:
            on_frame(frame)
        _dump(dump_dir, race.race_no, frames, img, frame)

        # walk this frame's ordered events, carrying ownership forward
        fresh = 0
        for kind, obj in frame.seq:
            if kind == "uma":
                hd: UmaHeader = obj  # type: ignore[assignment]
                owner = hd.name
                gained[hd.name] = hd.gained
                placement[hd.name] = hd.placement
                if hd.name not in order:
                    order.append(hd.name)
            elif kind == "team":
                owner = TEAM_OWNER
            else:
                card: DetailCard = obj  # type: ignore[assignment]
                if not card.balanced:
                    continue
                key = (owner, card.title, card.total)
                if key in cards:
                    continue
                cards[key] = card
                fresh += 1
                if owner == TEAM_OWNER:
                    team_bonus = card
        for hd in frame.umas:
            gained[hd.name] = hd.gained
            placement[hd.name] = hd.placement
            if hd.name not in order:
                order.append(hd.name)

        say(f"frame {frames}: {len(frame.cards)} cards, {fresh} new, total {len(cards)}")

        if fresh == 0:
            empty += 1
        else:
            empty = 0
        if empty >= 2:
            say("no new cards after two scrolls - stopping")
            break

        driver.scroll_at(cx, cy, -1)       # down one notch
        got = _grab(driver, ocr)
        if got is None:
            race.warnings.append("Score Details popup closed during the scan")
            say("popup closed mid-scan - stopping")
            break
        frame, img = got
        if frame.race_no is not None and race.race_no is not None and frame.race_no != race.race_no:
            race.warnings.append(f"panel changed to race {frame.race_no} mid-scan")
            break

    by_owner: dict[str, list[DetailCard]] = {}
    for (own, _t, _tot), card in cards.items():
        by_owner.setdefault(own, []).append(card)
    for name in order:
        race.umas.append(
            UmaDetail(
                name=name,
                placement=placement.get(name),
                gained=gained.get(name),
                cards=sorted(by_owner.get(name, []), key=lambda c: c.y),
            )
        )
    race.team_bonus = team_bonus

    for u in race.umas:
        if u.gained is not None and u.card_sum != u.gained:
            race.warnings.append(f"{u.name}: cards {u.card_sum} != gained {u.gained}")
    if race.total is not None:
        parts = (team_bonus.total if team_bonus else 0) + sum(u.gained or 0 for u in race.umas)
        if parts != race.total:
            race.warnings.append(f"race {race.race_no}: parts {parts} != total {race.total}")

    return DetailScan(race, frames, race.warnings)
