"""Self-contained regression tests (no pytest needed).

    py -3.13 tests/run_tests.py
"""
from __future__ import annotations

import os
import json
import stat
import sys
import tempfile
import threading
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umacalk.app import find_race_entries  # noqa: E402
from umacalk.ocr import Ocr  # noqa: E402
from umacalk.parse import match_subrow, parse_int, parse_race_header  # noqa: E402
from umacalk.scan_detail import scan_panel  # noqa: E402
from umacalk.model import UmaRow, Run  # noqa: E402
from umacalk.scan_list import parse_frame, scan as scan_list, _twin, PanelGone, ListFrame, _grab_retry  # noqa: E402
from umacalk.driver import ReplayDriver  # noqa: E402
from umacalk.names import NameResolver  # noqa: E402
from umacalk.config import Config  # noqa: E402
from umacalk.capture import CaptureError, Capturer  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

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


def test_config_path_no_traversal() -> None:
    print("config path safety")
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
        cfg = Config(base_dir=root)
        result = cfg.path("subdir")
        check("normal path works", result == os.path.join(os.path.realpath(root), "subdir"))
        try:
            cfg.path(os.path.join("..", os.path.basename(outside), "file.txt"))
            check("relative traversal blocked", False)
        except ValueError:
            check("relative traversal blocked", True)
        try:
            cfg.path(outside)
            check("absolute path blocked", False)
        except ValueError:
            check("absolute path blocked", True)

        # A path component inside the base may itself point outside through a symlink.
        link = os.path.join(root, "linked")
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            check("symlink path check unavailable", True, "platform does not permit symlink creation")
        else:
            try:
                Config(base_dir=root).path(os.path.join("linked", "file.txt"))
                check("symlink escape blocked", False)
            except ValueError:
                check("symlink escape blocked", True)
        if os.name == "nt":
            try:
                cfg.path("Z:\\outside\\file.txt")
                check("different drive blocked", False)
            except ValueError:
                check("different drive blocked", True)


def test_history_none_races() -> None:
    print("history None races handling")
    from umacalk.history import append_run
    from openpyxl import Workbook
    # First create a valid history file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    wb = Workbook()
    ws = wb.active
    ws.title = "Runs"
    ws.append(["Run", "Date", "Distance", "Uma", "Score", "Race total"])
    wb.save(path)

    # Test with None races (simulates capture failure)
    run = Run(
        run_id="test_none",
        captured_at="2026-01-01 00:00",
        rows=[UmaRow(name="Test", score=100, distance="Sprint")],
        races=None,  # Explicitly None
    )
    try:
        append_run(path, run)
        saved = load_workbook(path, data_only=True)
        row = list(saved["Runs"].iter_rows(min_row=2, values_only=True))[0]
        saved.close()
        check("None races handled without crash", row[5] is None)
    except Exception as e:
        check("None races handled without crash", False, str(e))
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_history_corrupted_excel() -> None:
    print("history corrupted file handling")
    from umacalk.history import append_run
    from openpyxl import Workbook

    run = Run(
        run_id="test_corrupt",
        captured_at="2026-01-01 00:00",
        rows=[UmaRow(name="Test", score=100, distance="Sprint")],
        races=[],
    )
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "history.xlsx")
        damaged = b"This is not an XLSX workbook\x00"
        with open(path, "wb") as fh:
            fh.write(damaged)
        try:
            append_run(path, run)
        except ValueError:
            with open(path, "rb") as fh:
                check("damaged workbook rejected without overwrite", fh.read() == damaged)
        else:
            check("damaged workbook rejected without overwrite", False, "expected ValueError")

        missing_sheet = os.path.join(root, "missing-runs.xlsx")
        wb = Workbook()
        wb.active.title = "Other"
        wb.save(missing_sheet)
        wb.close()
        try:
            append_run(missing_sheet, run)
        except ValueError:
            check("workbook without Runs sheet rejected", True)
        else:
            check("workbook without Runs sheet rejected", False, "expected ValueError")

        invalid_header = os.path.join(root, "invalid-header.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "Runs"
        ws.append(["Run", "Date", "Uma"])
        wb.save(invalid_header)
        wb.close()
        with open(invalid_header, "rb") as fh:
            before = fh.read()
        try:
            append_run(invalid_header, run)
        except ValueError:
            with open(invalid_header, "rb") as fh:
                check("partial workbook rejected without overwrite", fh.read() == before)
        else:
            check("partial workbook rejected without overwrite", False, "expected ValueError")

        from unittest.mock import patch
        valid = os.path.join(root, "valid.xlsx")
        wb = Workbook()
        ws = wb.active
        ws.title = "Runs"
        ws.append(["Run", "Date", "Distance", "Uma", "Score", "Race total"])
        wb.save(valid)
        wb.close()
        with open(valid, "rb") as fh:
            before = fh.read()
        original_close = Workbook.close
        closed = []

        def tracked_close(workbook):
            closed.append(True)
            return original_close(workbook)

        with (
            patch("umacalk.history.Workbook.save", side_effect=OSError("disk full")),
            patch("umacalk.history.Workbook.close", new=tracked_close),
        ):
            try:
                append_run(valid, run)
            except OSError:
                with open(valid, "rb") as fh:
                    intact = fh.read() == before
                try:
                    saved = load_workbook(valid, read_only=True, data_only=True)
                    header = tuple(cell.value for cell in next(saved["Runs"].iter_rows(max_row=1)))
                    saved.close()
                    readable = header == ("Run", "Date", "Distance", "Uma", "Score", "Race total")
                except Exception:
                    readable = False
                leftovers = [n for n in os.listdir(root) if n.startswith(".umacalk-history-")]
                check("failed history save preserves readable original", intact and readable)
                check("temporary workbook removed after save failure", not leftovers)
                check("workbook closed after save failure", bool(closed))
            else:
                check("failed history save preserves readable original", False, "expected OSError")
                check("temporary workbook removed after save failure", False)
                check("workbook closed after save failure", bool(closed))

        with open(valid, "rb") as fh:
            before_replace = fh.read()
        with patch("umacalk.history.os.replace", side_effect=OSError("replace failed")):
            try:
                append_run(valid, run)
            except OSError:
                with open(valid, "rb") as fh:
                    intact = fh.read() == before_replace
                leftovers = [n for n in os.listdir(root) if n.startswith(".umacalk-history-")]
                check("failed replacement preserves original", intact)
                check("temporary workbook removed after replacement failure", not leftovers)
            else:
                check("failed replacement preserves original", False, "expected OSError")
                check("temporary workbook removed after replacement failure", False)

        if os.name != "nt":
            os.chmod(valid, 0o640)
            append_run(valid, run)
            check("atomic history save preserves mode bits",
                  stat.S_IMODE(os.stat(valid).st_mode) == 0o640)


def test_optional_input_robustness() -> None:
    print("optional input robustness")
    with tempfile.TemporaryDirectory() as root:
        learned = os.path.join(root, "learned.json")
        with open(learned, "w", encoding="utf-8") as fh:
            fh.write('{"unexpected": "object"}')
        resolver = NameResolver(learned_path=learned)
        check("non-list learned names ignored", "Silence Suzuka" in resolver.names)
        with open(learned, "w", encoding="utf-8") as fh:
            fh.write('["Custom Uma", " custom uma ", "Custom Uma", 42, null]')
        resolver = NameResolver(learned_path=learned)
        matching_names = [name for name in resolver.names if name.casefold() == "custom uma"]
        check("duplicates and non-string learned names normalized",
              len(matching_names) == 1 and resolver.resolve("Custom Uma") is not None)
        resolver.learn("Another Uma")
        with open(learned, encoding="utf-8") as fh:
            saved_names = json.load(fh)
        check("learned names saved without duplicates",
              sum(name.casefold() == "custom uma" for name in saved_names) == 1)

        invalid = os.path.join(root, "invalid.json")
        invalid_bytes = b"[\xff"
        with open(invalid, "wb") as fh:
            fh.write(invalid_bytes)
        NameResolver(learned_path=invalid).learn("Do not overwrite")
        with open(invalid, "rb") as fh:
            check("invalid encoding preserved", fh.read() == invalid_bytes)
        malformed = os.path.join(root, "malformed.json")
        malformed_bytes = b'["unfinished"'
        with open(malformed, "wb") as fh:
            fh.write(malformed_bytes)
        malformed_resolver = NameResolver(learned_path=malformed)
        check("malformed JSON loads with roster fallback",
              malformed_resolver.resolve("Silence Suzuka") is not None)
        malformed_resolver.learn("Do not overwrite")
        with open(malformed, "rb") as fh:
            check("malformed JSON preserved", fh.read() == malformed_bytes)
        with open(invalid, "w", encoding="utf-8") as fh:
            fh.write('{"wrong": "shape"}')
        NameResolver(learned_path=invalid).learn("Do not overwrite")
        with open(invalid, encoding="utf-8") as fh:
            check("unexpected JSON structure preserved", fh.read() == '{"wrong": "shape"}')
    check("black capture detected", Capturer.looks_black(np.zeros((4, 4, 3), dtype=np.uint8)))
    check("malformed grayscale capture rejected", Capturer.looks_black(np.zeros((4, 4), dtype=np.uint8)))
    check("lit capture accepted", not Capturer.looks_black(np.full((4, 4, 3), 200, dtype=np.uint8)))


def test_list_panel_retry() -> None:
    print("list panel retry")
    from unittest.mock import patch
    from umacalk.model import Rect

    resolver = NameResolver()
    frame = ListFrame(Rect(0, 0, 100, 100), [], None, [], [])
    class Driver:
        def __init__(self):
            self.grabs = 0

        def grab(self):
            self.grabs += 1
            return np.zeros((2, 2, 3), dtype=np.uint8)

    with patch("umacalk.scan_list.parse_frame", side_effect=[PanelGone(), frame]) as parse:
        result = _grab_retry(Driver(), Ocr(), resolver)
    check("transient panel miss retried", result is frame and parse.call_count == 2)

    driver = Driver()
    with patch("umacalk.scan_list.parse_frame", side_effect=PanelGone("still gone")) as parse:
        result = _grab_retry(driver, Ocr(), resolver, tries=3)
    check("persistent panel miss stops at retry limit",
          result is None and parse.call_count == driver.grabs == 3)
    with patch("umacalk.scan_list.parse_frame", side_effect=PanelGone("still gone")) as parse:
        result = scan_list(driver, Ocr(), resolver)
    check("scan reports persistent initial panel failure",
          not result.rows and any("Score Info screen not found" in w for w in result.warnings)
          and parse.call_count == 3)
    with patch("umacalk.scan_list.parse_frame", side_effect=ValueError("OCR failed")) as parse:
        try:
            _grab_retry(Driver(), Ocr(), resolver, tries=3)
        except ValueError:
            check("non-panel failures are not retried", parse.call_count == 1)
        else:
            check("non-panel failures are not retried", False, "expected ValueError")


def test_capture_cleanup() -> None:
    print("capture cleanup")

    class StartFailureCamera:
        stopped = False

        def start(self, **_kwargs):
            raise RuntimeError("initialization failed")

        def stop(self):
            self.stopped = True

    camera = StartFailureCamera()
    module = types.ModuleType("bettercam")
    module.create = lambda **_kwargs: camera
    from unittest.mock import patch
    with patch.dict(sys.modules, {"bettercam": module}):
        try:
            Capturer((0, 0, 10, 10), backend="bettercam")
        except CaptureError:
            check("partially initialized camera stopped", camera.stopped)
        else:
            check("partially initialized camera stopped", False, "expected CaptureError")

    class DelayedCamera:
        attempts = 0
        stopped = False

        def start(self, **_kwargs):
            return None

        def get_latest_frame(self):
            self.attempts += 1
            return None if self.attempts < 3 else np.full((2, 2, 3), 200, dtype=np.uint8)

        def stop(self):
            self.stopped = True

    camera = DelayedCamera()
    module = types.ModuleType("bettercam")
    module.create = lambda **_kwargs: camera
    with patch.dict(sys.modules, {"bettercam": module}):
        capturer = Capturer((0, 0, 10, 10), backend="bettercam")
        check("startup frame warm-up retries", camera.attempts == 3 and capturer.name == "bettercam")
        capturer.close()
        check("warmed-up camera closes", camera.stopped)

    class EmptyCamera:
        stopped = False

        def get_latest_frame(self):
            return None

        def stop(self):
            self.stopped = True

    camera = EmptyCamera()
    capturer = Capturer.__new__(Capturer)
    capturer.rect = (0, 0, 10, 10)
    capturer.backend = "bettercam"
    capturer._impl = camera
    capturer._mss = None
    capturer.name = "bettercam"
    capturer._tid = threading.get_ident()
    capturer._opening = False
    try:
        capturer.grab()
    except CaptureError:
        check("frame acquisition failure releases camera",
              camera.stopped and capturer._impl is None and not capturer.name)
    else:
        check("frame acquisition failure releases camera", False, "expected CaptureError")


def main() -> int:
    ocr = Ocr(1400)
    resolver = NameResolver()
    test_parse()
    test_dedupe()
    test_list(ocr, resolver)
    test_detail(ocr)
    test_race_entries(ocr)
    test_config_path_no_traversal()
    test_history_none_races()
    test_history_corrupted_excel()
    test_optional_input_robustness()
    test_list_panel_retry()
    test_capture_cleanup()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
