"""Score Details panel parsing and the scroll-to-collect loop.

A card is only trusted when its displayed sub-rows sum exactly to its
displayed total; that identity doubles as the completeness test, because a
card clipped by the viewport edge loses a row and fails the sum.

Ownership (which uma a card belongs to) is state carried across frames: when
scrolling, the uma header is usually off-screen, so a per-frame parse could
not attribute anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import detect
from .model import DetailCard, RaceDetail, Rect, ScoreLine, TextLine, UmaDetail
from .ocr import Ocr
from .parse import card_category, clean, match_subrow, parse_int, parse_race_header

VALUE_X_MIN = 0.82      # right-aligned value column
LABEL_X_MAX = 0.72
TEAM_OWNER = "__team__"


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


def parse_frame(img: np.ndarray, ocr: Ocr) -> FrameParse:
    panel = detect.find_panel(img)
    if panel is None:
        raise RuntimeError("popup panel not found - is a Score Details screen open?")
    crop = img[panel.y:panel.y2, panel.x:panel.x2]
    lines = ocr.read(crop)
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


def scan_panel(
    driver,
    ocr: Ocr,
    *,
    expect_race: int | None = None,
    log: Callable[[str], None] | None = None,
    max_frames: int = 60,
    on_frame: Callable[[FrameParse], None] | None = None,
) -> DetailScan:
    say = log or (lambda _m: None)
    frame = parse_frame(driver.grab(), ocr)
    if frame.panel is None:
        return DetailScan(
            None,
            warnings=["Score Details screen not found - open it before scanning"],
        )
    cx = frame.panel.x + frame.panel.w // 2
    cy = frame.panel.y + frame.panel.h // 2
    driver.scroll_at(cx, cy, -15)
    frame = parse_frame(driver.grab(), ocr)

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
    stalls = 0
    prev_sig: tuple | None = None

    while frames < max_frames:
        frames += 1
        if on_frame:
            on_frame(frame)

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

        sig = frame.signature()
        if sig and sig == prev_sig:
            stalls += 1
        else:
            stalls = 0
        prev_sig = sig
        if stalls >= 2:
            break

        driver.scroll_at(cx, cy, 1)
        frame = parse_frame(driver.grab(), ocr)
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
