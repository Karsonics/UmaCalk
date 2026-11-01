"""Orchestration: a scan session and the two capture steps.

Step 1 - Score Info list:  every uma with distance, score, MVP flag.
Step 2 - Race Results:     click each of the five races, read its breakdown.

Both steps are independent so a failed step never invalidates the other.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
from rapidfuzz import fuzz

from . import probe as probe_mod
from .capture import Capturer
from .config import Config
from .driver import LiveDriver, ReplayDriver
from .export import write_txt, write_xlsx
from .history import append_run
from .model import RaceDetail, Run
from .names import NameResolver
from .ocr import Ocr
from .parse import clean
from .reconcile import Issue, check
from .scan_detail import scan_panel
from .scan_list import scan as scan_list

Logger = Callable[[str], None]


@dataclass
class Session:
    cfg: Config
    driver: Any
    ocr: Ocr
    resolver: NameResolver
    run: Run
    log: Logger
    profile: probe_mod.DisplayProfile | None = None
    warnings: list[str] = field(default_factory=list)


def _new_run() -> Run:
    now = datetime.now()
    return Run(
        run_id=now.strftime("%Y%m%d_%H%M%S"),
        captured_at=now.strftime("%Y-%m-%d %H:%M"),
    )


def make_resolver(cfg: Config) -> NameResolver:
    from . import names as names_mod

    extra: list[str] = []
    names_file = cfg.path(cfg.names_file)
    if os.path.exists(names_file):
        with open(names_file, encoding="utf-8") as fh:
            extra = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    return NameResolver(
        threshold=cfg.name_threshold,
        learned_path=cfg.path(cfg.learned_names),
        extra=extra,
    )


def open_live(cfg: Config, log: Logger, abort=None) -> Session:
    prof = probe_mod.probe(cfg.window_title)
    if not prof.hwnd:
        raise RuntimeError(f"window {cfg.window_title!r} not found")
    if cfg.monitor_device:
        for m in prof.monitors:
            if m.device.upper() == cfg.monitor_device.upper():
                prof.monitor = m
    log(prof.describe())
    cap = Capturer(prof.client, cfg.capture_backend)
    log(f"capture     {cap.name}  {prof.client[2]}x{prof.client[3]}")
    drv = LiveDriver(prof.hwnd, prof.client, cap,
                     settle=cfg.scroll_settle, abort=abort)
    drv.focus()
    return Session(cfg, drv, Ocr(cfg.ocr_target_width), make_resolver(cfg),
                   _new_run(), log, prof)


def open_replay(cfg: Config, frames: list[np.ndarray], log: Logger) -> Session:
    run = _new_run()
    return Session(cfg, ReplayDriver(frames), Ocr(cfg.ocr_target_width),
                   make_resolver(cfg), run, log, None)


# ----------------------------------------------------------------- step 1

def _ensure_live(session: Session) -> None:
    """Re-probe before every scan: the window may have moved, the capture backend
    is thread-affine, and we must never send input to another application."""
    if not isinstance(session.driver, LiveDriver):
        return
    prof = probe_mod.probe(session.cfg.window_title)
    if not prof.hwnd:
        raise RuntimeError(f"window {session.cfg.window_title!r} not found")
    if prof.hwnd != session.driver.hwnd:
        raise RuntimeError("game window changed - restart umaCalk")
    rect = tuple(prof.client)
    if rect != tuple(session.driver.rect):
        session.log(f"window moved or resized to {rect[2]}x{rect[3]} - recalibrating")
        session.driver.rect = rect
        session.driver.capturer.rect = rect
    session.driver.focus()
    import time

    import win32gui

    for _ in range(3):
        if win32gui.GetForegroundWindow() == session.driver.hwnd:
            return
        time.sleep(0.15)
        session.driver.focus()
    raise RuntimeError("game window is not focused - click the game, then try again")


def step_list(session: Session) -> Run:
    _ensure_live(session)
    session.log("reading Score Info ...")
    result = scan_list(
        session.driver, session.ocr, session.resolver,
        log=session.log, max_frames=session.cfg.max_frames,
    )
    session.run.rows = result.rows
    session.warnings.extend(result.warnings)
    if not result.rows:
        for warning in result.warnings:
            session.log(f"  {warning}")
        return session.run
    session.log(f"{len(result.rows)} umas read")
    return session.run


# ----------------------------------------------------------------- step 2

def find_race_entries(img: np.ndarray, ocr: Ocr) -> list[tuple[int, int, int]]:
    """Locate the clickable race rows on the Race Results popup."""
    h, w = img.shape[:2]
    crop = img[:, : int(0.75 * w)]
    lines = ocr.read(crop, scale=1.6)
    found: list[tuple[int, int, int]] = []
    for ln in lines:
        m = re.search(r"race\s*(\d+)", clean(ln.text), re.I)
        if m:
            found.append((int(m.group(1)), ln.cx + 0, ln.cy))
    found.sort(key=lambda t: t[2])
    seen: set[int] = set()
    out: list[tuple[int, int, int]] = []
    for n, x, y in found:
        if n in seen:
            continue
        seen.add(n)
        out.append((n, x, y))
    return out


def _close_panel(session: Session) -> None:
    from . import detect

    img = session.driver.grab()
    for panel in detect.find_panels(img):
        crop = img[panel.y:panel.y2, panel.x:panel.x2]
        for ln in session.ocr.read(crop):
            if clean(ln.text).lower().startswith("close"):
                session.driver.click_at(panel.x + ln.cx, panel.y + ln.cy)
                return
    session.driver.key("escape")


def _save_diag(dump_dir: str | None, name: str, img, ocr: Ocr) -> None:
    """Write the raw frame and its OCR text so a failure can be replayed."""
    if not dump_dir:
        return
    import cv2

    try:
        os.makedirs(dump_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dump_dir, name + ".png"), img)
        lines = ocr.read(img, scale=1.4)
        with open(os.path.join(dump_dir, name + ".txt"), "w", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(f"{ln.text}\t{ln.x1},{ln.y1},{ln.x2},{ln.y2}\n")
    except Exception:
        pass


def _info_point(img, panel, bar_y: float, pitch: float) -> tuple[int, int]:
    """Centre of the circular (i) button on a Score Info row.

    Measured from the list layout: ~0.92 of the panel width horizontally and
    a quarter of a row above the green bar, level with the name/score line.
    A small round blob near that point is used to snap the click exactly.
    """
    import cv2

    by = panel.y + bar_y
    ex = int(round(panel.x + 0.92 * panel.w))
    ey = int(round(by - 0.25 * pitch))
    half = 14
    x0, x1 = max(0, ex - half), min(img.shape[1], ex + half)
    y0, y1 = max(0, ey - half), min(img.shape[0], ey + half)
    if x1 <= x0 or y1 <= y0:
        return ex, ey
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mask = (gray < 160).astype(np.uint8)
    count, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (30 <= area <= 900 and 9 <= bw <= 26 and 9 <= bh <= 26):
            continue
        dist = abs(cents[i][0] - half) + abs(cents[i][1] - half)
        if best is None or dist < best[0]:
            best = (dist, cents[i])
    if best is not None and best[0] <= 10:
        return int(round(x0 + best[1][0])), int(round(y0 + best[1][1]))
    return ex, ey


def _find_uma_row(session: Session, name: str, tries: int = 40):
    from .scan_list import parse_frame as parse_list_frame

    drv = session.driver
    img = drv.grab()
    frame = parse_list_frame(img, session.ocr, session.resolver)
    if frame.panel is None:
        return None, None, None
    drv.scroll_at(frame.panel.cx, frame.panel.cy, 30)      # back to the top
    for _ in range(tries):
        img = drv.grab()
        frame = parse_list_frame(img, session.ocr, session.resolver)
        if frame.panel is None:
            return None, None, None
        for row in frame.rows:
            if row.name == name:
                return frame, row, img
        drv.scroll_at(frame.panel.cx, frame.panel.cy, -1)
    return None, None, None


def step_races(session: Session, expect: int = 5) -> list[RaceDetail]:
    """Open Score Details from Score Info and collect each race once.

    A single (i) popup contains all three umas in that race. Keep walking
    rows in Score Info display order; after a complete roster is extracted,
    the other two rows are implicitly done.
    """
    cfg = session.cfg
    _ensure_live(session)
    diag = None
    if cfg.save_frames:
        diag = os.path.join(cfg.output_path, "diag", session.run.run_id)
        session.log(f"diagnostic frames -> {diag}")

    if not session.run.rows:
        raise RuntimeError("scan the uma list first (Score Info), then the races")

    targets = [r for r in session.run.rows if r.score]
    pending = {r.name for r in targets}
    races: list[RaceDetail] = []
    for target in targets:
        if target.name not in pending:
            continue
        frame, row, img = _find_uma_row(session, target.name)
        if row is None:
            session.log(f"{target.name}: not found in the list; leaving it pending")
            continue
        x, y = _info_point(img, frame.panel, row.y, frame.pitch)
        session.log(f"{target.distance}: opening Score Details via {target.name}")
        session.driver.click_at(x, y, pause=cfg.click_pause)
        if diag:
            _save_diag(diag, f"{target.distance.lower()}_after_click",
                       session.driver.grab(), session.ocr)
        try:
            res = scan_panel(
                session.driver, session.ocr,
                log=session.log, max_frames=cfg.max_frames,
                dump_dir=os.path.join(diag, target.name.lower().replace(" ", "_")) if diag else None,
            )
        except Exception as exc:  # one broken race must not abort the rest
            session.log(f"{target.name}: scan failed - {exc}")
            session.warnings.append(f"{target.name}: {exc}")
            _close_panel(session)
            continue
        if res.race is None:
            session.warnings.extend(res.warnings)
            session.log(f"{target.name}: " + "; ".join(res.warnings))
            _close_panel(session)
            continue
        session.warnings.extend(res.race.warnings)
        extracted = {u.name: u for u in res.race.umas}
        # OCR can vary a character in a header, so match the roster fuzzily.
        # Teammates may have different distance labels. Match the popup roster
        # against the full Score Info list and only complete a three-uma race
        # when each teammate's card totals balance to their gained score.
        matched: dict[str, Any] = {}
        for found_name, uma in extracted.items():
            candidate = max(pending, key=lambda name: fuzz.WRatio(name, found_name), default=None)
            if candidate is not None and fuzz.WRatio(candidate, found_name) >= 88:
                matched[candidate] = uma
        incomplete = [name for name, uma in matched.items()
                      if uma.gained is None or not uma.cards or uma.card_sum != uma.gained]
        if target.name not in matched or len(matched) != 3 or len(extracted) != 3 or incomplete:
            details = sorted(set(incomplete + [n for n in extracted if not any(
                fuzz.WRatio(row_name, n) >= 88 for row_name in pending)]))
            session.log(f"{target.name}: incomplete race extraction ({details}); will retry")
        else:
            races.append(res.race)
            pending.difference_update(matched)
            session.log(
                f"{target.distance}: race {res.race.race_no}, {len(res.race.umas)} umas, "
                f"{sum(len(u.cards) for u in res.race.umas)} cards")
        _close_panel(session)
    session.run.races = races
    session.run.match_total = None
    return races


# --------------------------------------------------------------- finishing

@dataclass
class Result:
    run: Run
    issues: list[Issue]
    json_path: str | None = None
    txt_path: str | None = None
    xlsx_path: str | None = None
    history_path: str | None = None


def finish(session: Session, *, write_history: bool = True) -> Result:
    run = session.run
    issues = check(run)
    if not run.rows and not run.races:
        session.log("nothing captured - no files written")
        return Result(run, issues)
    out_dir = session.cfg.output_path
    os.makedirs(out_dir, exist_ok=True)
    stem = run.run_id

    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(run.to_json())

    txt_path = write_txt(run, os.path.join(out_dir, f"{stem}.txt"), issues)
    xlsx_path = write_xlsx(run, os.path.join(out_dir, f"{stem}.xlsx"), issues)
    hist = None
    if write_history:
        hist = append_run(session.cfg.history_path, run)
    session.log(f"wrote {json_path}")
    session.log(f"wrote {txt_path}")
    session.log(f"wrote {xlsx_path}")
    if hist:
        session.log(f"updated {hist}")
    for issue in issues:
        if issue.level in ("error", "warn"):
            session.log(f"  {issue}")
    return Result(run, issues, json_path, txt_path, xlsx_path, hist)
