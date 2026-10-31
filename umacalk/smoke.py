"""Live smoke test: capture one frame from the running game and report.

Does not click anything. Proves the window lookup, client rect, capture
backend and panel detection all work against the real client.
"""
from __future__ import annotations

import sys

import cv2

from . import config, detect, probe


def main() -> int:
    probe.enable_dpi_awareness()
    cfg = config.load("config.toml")
    prof = probe.probe(cfg.window_title)
    print(prof.describe())
    for m in prof.monitors:
        print(f"  monitor {m.index}: {m.device} {m.width}x{m.height} "
              f"@{m.scale * 100:.0f}% origin ({m.left},{m.top})"
              f"{' primary' if m.primary else ''}")
    if not prof.hwnd:
        print("\nGame window not found. Start Umamusume, or set window_title in config.toml.")
        return 1

    from .capture import Capturer

    cap = Capturer(prof.client, cfg.capture_backend)
    img = cap.grab()
    h, w = img.shape[:2]
    mean = float(img.mean())
    print(f"\ncapture     {cap.name}  {w}x{h}  mean {mean:.1f}")
    if mean < 5:
        print("frame looks black - capture backend not seeing the game")
        return 2

    panel = detect.find_panel(img)
    if panel is None:
        print("no Score Info / Score Details panel on screen "
              "(open one of those screens, then re-run)")
    else:
        print(f"panel       x={panel.x} y={panel.y} {panel.w}x{panel.h}")
        bars = detect.find_bars(img, panel)
        if bars:
            print(f"rows        {len(bars)} progress bars, pitch {detect.row_pitch(bars):.1f}px")

    shot = "runs/live_smoke.png"
    cv2.imwrite(shot, img)
    print(f"saved       {shot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
