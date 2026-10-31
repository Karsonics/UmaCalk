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
    panel = detect.find_panel(img)
    if panel is not None:
        crop = img[panel.y:panel.y2, panel.x:panel.x2]
        for ln in session.ocr.read(crop):
            if clean(ln.text).lower().startswith("close"):
                session.driver.click_at(panel.x + ln.cx, panel.y + ln.cy)
                return
    session.driver.key("escape")


def step_races(session: Session, expect: int = 5) -> list[RaceDetail]:
    cfg = session.cfg
    _ensure_live(session)
    races: list[RaceDetail] = []
    for n in range(1, expect + 1):
        img = session.driver.grab()
        entries = find_race_entries(img, session.ocr)
        if len(entries) < 3:
            raise RuntimeError(
                "race list not found - open Team Trials > See All Race Results first")
        target = next((e for e in entries if e[0] == n), None)
        if target is None:
            session.log(f"race {n} not on screen, skipping")
            continue
        _, x, y = target
        session.log(f"race {n}: clicking ({x},{y})")
        session.driver.click_at(x, y, pause=cfg.click_pause)
        res = scan_panel(
            session.driver, session.ocr, expect_race=n,
            log=session.log, max_frames=cfg.max_frames,
        )
        if res.race is None:
            session.warnings.extend(res.warnings)
            session.log(f"race {n}: " + "; ".join(res.warnings))
            _close_panel(session)
            continue
        races.append(res.race)
        session.warnings.extend(res.race.warnings)
        session.log(
            f"race {n}: {len(res.race.umas)} umas, {sum(len(u.cards) for u in res.race.umas)} cards")
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
