"""Self-contained regression tests (no pytest needed).

    py -3.13 tests/run_tests.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umacalk.app import find_race_entries  # noqa: E402
from umacalk.ocr import Ocr  # noqa: E402
from umacalk.parse import match_subrow, parse_int, parse_race_header  # noqa: E402
from umacalk.scan_detail import scan_panel  # noqa: E402
from umacalk.model import UmaRow  # noqa: E402
from umacalk.scan_list import parse_frame, _twin  # noqa: E402
from umacalk.driver import ReplayDriver  # noqa: E402
from umacalk.names import NameResolver  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "Samples")
CROP = (279, 178, 1564, 875)  # windowed client area

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def load(name: str, crop: tuple[int, int, int, int] | None = CROP) -> np.ndarray:
    path = os.path.join(SAMPLES, name)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"missing sample: {path}")
    if crop:
        x0, y0, x1, y1 = crop
        img = img[y0:y1, x0:x1]
    return img


def test_dedupe() -> None:
    print("dedupe")
    gold = UmaRow(name="Gold Ship", score=61272, distance="Long",
                  raw={"name": "Gold Ship"})
    rows = {"Gold Ship": gold}
    # same physical row, clipped text, resolves to a different character
    twin = UmaRow(name="Gold City", score=61272, distance="Long",
                  raw={"name": "Gold Shlp"})
    check("clipped row detected as twin", _twin(rows, twin) is gold)
    other = UmaRow(name="Vodka", score=56201, distance="Mile",
                   raw={"name": "Vodka"})
    check("different uma is not a twin", _twin(rows, other) is None)
    check("missing raw is never a twin",
          _twin(rows, UmaRow(name="X", score=None, distance=None)) is None)


def test_parse() -> None:
    print("parse")
    check("plain", parse_int("61,272 pts") == 61272)
    check("no space", parse_int("144,764pts") == 144764)
    check("ocr letters", parse_int("6l,27Z pts") == 61272)
    check("negative", parse_int("-500 pts") == -500)
    check("race header", parse_race_header("Class 3, Race 4") == (3, 4))
    check("subrow ace", match_subrow("Ace Bonus") == "ace")
    check("subrow opponent", match_subrow("Opponent Rating Bonus") == "opponent")
    check("place is not ace", match_subrow("1st place") is None)
    check("team title not ace", match_subrow("1st, 2nd & 3rd Place") is None)
    check("streak", match_subrow("2 Streak Bonus") == "streak")


def test_list(ocr: Ocr, resolver: NameResolver) -> None:
    print("list parse")
    expected = {
        "Gold Ship": (61272, "Long"),
        "Vodka": (56201, "Mile"),
        "Fuji Kiseki": (54261, "Sprint"),
        "Daiwa Scarlet": (52520, "Mile"),
        "Grass Wonder": (52340, "Long"),
        "Agnes Tachyon": (47267, "Medium"),
        "Sakura Bakushin O": (46018, "Sprint"),
    }
    frame = parse_frame(load("Screenshot 2026-10-30 172608.png"), ocr, resolver)
    got = {r.name: (r.score, r.distance) for r in frame.rows}
    check("windowed row count", len(frame.rows) == 7, str(len(frame.rows)))
    for name, want in expected.items():
        check(f"windowed {name}", got.get(name) == want, str(got.get(name)))
    check("windowed no warnings", not frame.warnings, str(frame.warnings))

    full = parse_frame(load("Screenshot 2026-10-30 172652.png", None), ocr, resolver)
    fgot = {r.name: r.score for r in full.rows}
    check("fullscreen row count", len(full.rows) == 7, str(len(full.rows)))
    check("fullscreen mvp", any(r.mvp for r in full.rows))
    check("fullscreen Sakura", fgot.get("Sakura Bakushin O") == 59363, str(fgot.get("Sakura Bakushin O")))
    check("fullscreen Oguri", fgot.get("Oguri Cap") == 54547, str(fgot.get("Oguri Cap")))


def test_detail(ocr: Ocr) -> None:
    print("detail parse")
    frames = [
        load("Screenshot 2026-10-30 173803.png"),
        load("Screenshot 2026-10-30 174149.png"),
        load("Screenshot 2026-10-30 174230.png"),
        load("Screenshot 2026-10-30 174247.png"),
    ]
    drv = ReplayDriver(frames)
    res = scan_panel(drv, ocr, expect_race=4, log=None)
    check("race number", res.race is not None and res.race.race_no == 4)
    if res.race is None:
        return
    race = res.race
    check("race total", race.total == 144764, str(race.total))
    check("team bonus", race.team_bonus is not None and race.team_bonus.total == 12078,
          str(race.team_bonus.total if race.team_bonus else None))
    names = [u.name for u in race.umas]
    check("one uma found", names == ["Gold Ship"], str(names))
    if race.umas:
        u = race.umas[0]
        check("gained score", u.gained == 61637, str(u.gained))
        check("placement", u.placement == "1st", str(u.placement))
        check("seven complete cards", len(u.cards) == 7, str(len(u.cards)))
        check("all cards balance", all(c.balanced for c in u.cards))
        titles = {c.title: c.total for c in u.cards}
        check("1st place card", titles.get("1st place") == 25156, str(titles.get("1st place")))
        check("long shot card", titles.get("Long Shot") == 10063, str(titles.get("Long Shot")))
        check("positioning cards", titles.get("Good Positioning (Mid-Race)") == 2517
              and titles.get("Good Positioning (Late-Race)") == 2517)


def test_race_entries(ocr: Ocr) -> None:
    print("race list")
    img = load("Screenshot 2026-10-30 172557.png")
    entries = find_race_entries(img, ocr)
    check("five races found", [e[0] for e in entries] == [1, 2, 3, 4, 5], str(entries))
    check("entries ordered top-down", all(
        entries[i][2] < entries[i + 1][2] for i in range(len(entries) - 1)))


def main() -> int:
    ocr = Ocr(1400)
    resolver = NameResolver()
    test_parse()
    test_dedupe()
    test_list(ocr, resolver)
    test_detail(ocr)
    test_race_entries(ocr)
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
