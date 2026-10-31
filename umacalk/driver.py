"""Input drivers: live (pyautogui) and replay (screenshots)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .capture import Capturer


class Aborted(RuntimeError):
    pass


class LiveDriver:
    """Drives the real game window: screenshots in, mouse/wheel out."""

    def __init__(self, hwnd: int, rect: tuple[int, int, int, int],
                 capturer: Capturer, *, settle: float = 0.35,
                 abort: threading.Event | None = None) -> None:
        self.hwnd = hwnd
        self.rect = rect
        self.capturer = capturer
        self.settle = settle
        self.abort = abort or threading.Event()

    # -- geometry -----------------------------------------------------------
    @property
    def origin(self) -> tuple[int, int]:
        return (self.rect[0], self.rect[1])

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        ox, oy = self.origin
        return (ox + int(x), oy + int(y))

    # -- io -----------------------------------------------------------------
    def _check(self) -> None:
        if self.abort.is_set():
            raise Aborted("aborted by user")

    def grab(self) -> np.ndarray:
        self._check()
        return self.capturer.grab()

    def focus(self) -> None:
        import win32con
        import win32gui

        try:
            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
        except Exception:  # noqa: BLE001 - focus is best effort
            pass

    def click_at(self, x: int, y: int, *, pause: float = 0.5) -> None:
        import pyautogui

        self._check()
        sx, sy = self.to_screen(x, y)
        pyautogui.moveTo(sx, sy, duration=0.12)
        pyautogui.click()
        time.sleep(pause)

    def scroll_at(self, x: int, y: int, notches: int) -> None:
        import pyautogui

        self._check()
        sx, sy = self.to_screen(x, y)
        pyautogui.moveTo(sx, sy, duration=0.05)
        pyautogui.scroll(int(notches) * 120)
        time.sleep(self.settle)

    def key(self, name: str) -> None:
        import pyautogui

        self._check()
        pyautogui.press(name)
        time.sleep(0.25)


@dataclass
class ReplayDriver:
    """Feeds a fixed list of frames; scrolling moves one frame at a time."""

    frames: list[np.ndarray]
    index: int = 0
    clicks: list[tuple[int, int]] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)

    def grab(self) -> np.ndarray:
        if not self.frames:
            raise Aborted("no replay frames")
        return self.frames[min(self.index, len(self.frames) - 1)]

    def scroll_at(self, x: int, y: int, notches: int) -> None:
        if notches < 0:
            self.index = 0
        elif notches > 0:
            self.index = min(len(self.frames) - 1, self.index + 1)

    def click_at(self, x: int, y: int, *, pause: float = 0.0) -> None:
        self.clicks.append((x, y))

    def key(self, name: str) -> None:
        self.keys.append(name)

    def focus(self) -> None:
        return None
