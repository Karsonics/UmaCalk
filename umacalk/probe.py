"""Windows probing: DPI awareness, monitors, game window, display profile.

The DPI awareness call must happen before any window is created, otherwise
Windows virtualises coordinates and every click lands at the wrong place on a
scaled display.
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass, field

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
try:
    shcore = ctypes.windll.shcore
except OSError:  # pragma: no cover
    shcore = None


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


@dataclass
class Monitor:
    index: int
    device: str
    left: int
    top: int
    width: int
    height: int
    dpi: int = 96
    primary: bool = False

    @property
    def scale(self) -> float:
        return self.dpi / 96.0

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


@dataclass
class DisplayProfile:
    hwnd: int = 0
    title: str = ""
    dpi_mode: str = "none"
    client: tuple[int, int, int, int] = (0, 0, 0, 0)   # screen x, y, w, h
    monitor: Monitor | None = None
    mode: str = "unknown"                              # windowed | borderless | exclusive
    window_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    monitors: list[Monitor] = field(default_factory=list)

    def describe(self) -> str:
        m = self.monitor
        cx, cy, cw, ch = self.client
        mon = (f"{m.device} {m.width}x{m.height} @ {m.scale * 100:.0f}% "
               f"origin ({m.left},{m.top})") if m else "unknown"
        return (
            f"window      {self.title!r}  (hwnd 0x{self.hwnd:08X})\n"
            f"mode        {self.mode}\n"
            f"monitor     {mon}\n"
            f"client      {cw}x{ch} at ({cx},{cy})\n"
            f"dpi         {self.dpi_mode}"
        )


def enable_dpi_awareness() -> str:
    """Declare per-monitor v2 awareness. Returns the mode actually achieved."""
    try:
        # -4 == DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except (AttributeError, OSError):
        pass
    if shcore is not None:
        try:
            shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            return "per-monitor"
        except OSError:
            pass
    try:
        user32.SetProcessDPIAware()
        return "system"
    except (AttributeError, OSError):
        return "none"


def list_monitors() -> list[Monitor]:
    monitors: list[Monitor] = []

    def cb(hmon, hdc, lprc, data):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            dpi = 96
            if shcore is not None:
                try:
                    dx = wintypes.UINT()
                    dy = wintypes.UINT()
                    if shcore.GetDpiForMonitor(hmon, 0, ctypes.byref(dx), ctypes.byref(dy)) == 0:
                        dpi = int(dx.value)
                except OSError:
                    pass
            r = info.rcMonitor
            monitors.append(
                Monitor(
                    index=len(monitors),
                    device=info.szDevice,
                    left=r.left,
                    top=r.top,
                    width=r.right - r.left,
                    height=r.bottom - r.top,
                    dpi=dpi,
                    primary=bool(info.dwFlags & 1),
                )
            )
        return True

    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                              ctypes.POINTER(RECT), wintypes.LPARAM)(cb)
    user32.EnumDisplayMonitors(None, None, proc, 0)
    return monitors


def _window_text(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def find_window(title_part: str) -> int:
    """Largest visible top-level window whose title contains title_part."""
    needle = title_part.lower()
    found: list[tuple[int, int]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _l):
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _window_text(hwnd)
        if needle and needle not in title.lower():
            return True
        r = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(r))
        area = max(0, r.right - r.left) * max(0, r.bottom - r.top)
        if area > 200 * 200:
            found.append((area, hwnd))
        return True

    user32.EnumWindows(cb, 0)
    if not found:
        return 0
    found.sort(reverse=True)
    return found[0][1]


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return (r.left, r.top, r.right - r.left, r.bottom - r.top)


def client_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    point = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(point))
    return (point.x, point.y, r.right - r.left, r.bottom - r.top)


def window_dpi(hwnd: int) -> int:
    try:
        return int(user32.GetDpiForWindow(hwnd))
    except (AttributeError, OSError):
        return 96


def _style(hwnd: int) -> int:
    return user32.GetWindowLongW(hwnd, -16) & 0xFFFFFFFF


def classify_mode(hwnd: int, client: tuple[int, int, int, int],
                  monitors: list[Monitor]) -> str:
    if not monitors:
        return "unknown"
    cx, cy, cw, ch = client
    mon = pick_monitor(client, monitors)
    if mon is None:
        return "unknown"
    if abs(cw - mon.width) <= 2 and abs(ch - mon.height) <= 2 and abs(cx - mon.left) <= 2:
        style = _style(hwnd)
        has_caption = bool(style & 0x00C00000)   # WS_CAPTION
        return "borderless" if has_caption or (style & 0x00040000) else "exclusive"
    return "windowed"


def pick_monitor(rect: tuple[int, int, int, int], monitors: list[Monitor]) -> Monitor | None:
    """Monitor with the largest overlap with the given screen rect."""
    x, y, w, h = rect
    best: tuple[int, Monitor] | None = None
    for m in monitors:
        ox = max(0, min(x + w, m.left + m.width) - max(x, m.left))
        oy = max(0, min(y + h, m.top + m.height) - max(y, m.top))
        area = ox * oy
        if area and (best is None or area > best[0]):
            best = (area, m)
    return best[1] if best else None


def probe(title_part: str = "umamusume", dpi_mode: str | None = None) -> DisplayProfile:
    monitors = list_monitors()
    hwnd = find_window(title_part)
    prof = DisplayProfile(
        hwnd=hwnd,
        title=_window_text(hwnd) if hwnd else "",
        dpi_mode=dpi_mode or enable_dpi_awareness(),
        monitors=monitors,
    )
    if hwnd:
        prof.client = client_rect(hwnd)
        prof.window_rect = window_rect(hwnd)
        prof.monitor = pick_monitor(prof.client, monitors)
        prof.mode = classify_mode(hwnd, prof.client, monitors)
    return prof


def wait_for_window(title_part: str = "umamusume", timeout: float = 20.0) -> int:
    end = time.time() + timeout
    while time.time() < end:
        hwnd = find_window(title_part)
        if hwnd:
            return hwnd
        time.sleep(0.25)
    return 0
