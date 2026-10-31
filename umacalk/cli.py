"""Command line entry point.

DPI awareness is enabled before anything else touches a window.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from . import __version__, config, probe, app
from .ocr import Ocr


def _crop(img: np.ndarray, spec: str | None) -> np.ndarray:
    if not spec:
        return img
    x0, y0, x1, y1 = (int(v) for v in spec.split(","))
    return img[y0:y1, x0:x1].copy()


def _load(path: str, crop: str | None) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image: {path}")
    return _crop(img, crop)


def _say(msg: str) -> None:
    print(msg, flush=True)


def cmd_probe(args) -> int:
    prof = probe.probe(args.window)
    print(prof.describe())
    for m in prof.monitors:
        print(f"  monitor {m.index}: {m.device} {m.width}x{m.height} @ {m.scale * 100:.0f}% "
              f"origin ({m.left},{m.top}){' primary' if m.primary else ''}")
    return 0


def cmd_list(args) -> int:
    cfg = config.load(args.config) if args.config else config.Config()
    cfg.base_dir = args.base or cfg.base_dir
    sess = app.open_replay(cfg, [_load(p, args.crop) for p in args.images], _say)
    app.step_list(sess)
    result = app.finish(sess, write_history=False)
    for issue in result.issues:
        _say(str(issue))
    return 0


def cmd_detail(args) -> int:
    cfg = config.load(args.config) if args.config else config.Config()
    frames = [_load(p, args.crop) for p in args.images]
    sess = app.open_replay(cfg, frames, _say)
    from .scan_detail import scan_panel

    res = scan_panel(sess.driver, sess.ocr, expect_race=args.race,
                     log=_say, max_frames=cfg.max_frames)
    if res.race is None:
        _say("; ".join(res.warnings))
        return 1
    race = res.race
    _say(f"race {race.race_no} class {race.class_no}: total {race.total} "
         f"team {race.team_bonus.total if race.team_bonus else None}")
    for u in race.umas:
        _say(f"  {u.name}: {u.gained} ({len(u.cards)} cards, sum {u.card_sum})"
             f"{'' if u.balanced else '  MISMATCH'}")
        for c in u.cards:
            vals = {r.label: r.value for r in c.rows}
            _say(f"    {c.title:<38} {c.total:>8}  " +
                 " ".join(f"{k}={v}" for k, v in vals.items()))
    for w in res.warnings:
        _say(f"! {w}")
    return 0


def cmd_run(args) -> int:
    cfg = config.load(args.config)
    sess = app.open_live(cfg, _say)
    _say("")
    _say("Step 1: open Team Trials > Race History > Score Info, then press Enter.")
    input()
    app.step_list(sess)
    if cfg.capture_details:
        _say("")
        _say("Step 2: go back and open See All Race Results, then press Enter.")
        input()
        app.step_races(sess)
    app.finish(sess)
    return 0


def cmd_init(args) -> int:
    config.write_default(args.config)
    _say(f"wrote {args.config}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="umacalk", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("probe", help="show window / monitor / DPI information")
    sp.add_argument("--window", default="umamusume")
    sp.set_defaults(func=cmd_probe)

    sl = sub.add_parser("list", help="parse Score Info screenshots")
    sl.add_argument("images", nargs="+")
    sl.add_argument("--crop", help="x0,y0,x1,y1 to cut the game client area")
    sl.add_argument("--config")
    sl.add_argument("--base")
    sl.set_defaults(func=cmd_list)

    sd = sub.add_parser("detail", help="parse Score Details screenshots")
    sd.add_argument("images", nargs="+")
    sd.add_argument("--crop")
    sd.add_argument("--config")
    sd.add_argument("--race", type=int, help="expected race number, warns on mismatch")
    sd.set_defaults(func=cmd_detail)

    sr = sub.add_parser("run", help="live capture")
    sr.add_argument("--config", default="config.toml")
    sr.set_defaults(func=cmd_run)

    si = sub.add_parser("init", help="write a default config.toml")
    si.add_argument("--config", default="config.toml")
    si.set_defaults(func=cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    probe.enable_dpi_awareness()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
