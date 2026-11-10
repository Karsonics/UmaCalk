"""Screen capture with a backend fallback chain.

bettercam (DXGI Desktop Duplication) is the only backend that reliably grabs
exclusive-fullscreen DirectX output; mss and PIL.ImageGrab are the fallbacks
for windowed and borderless modes.
"""
from __future__ import annotations

import threading

import numpy as np


class CaptureError(RuntimeError):
    pass


class Capturer:
    def __init__(self, rect: tuple[int, int, int, int], backend: str = "auto") -> None:
        self.rect = rect
        self.backend = backend
        self._impl = None
        self._mss = None
        self.name = ""
        self._tid = threading.get_ident()
        self._opening = False
        self._open()

    def _open(self) -> None:
        order = [self.backend] if self.backend != "auto" else ["bettercam", "mss", "pil"]
        errors: list[str] = []
        self._opening = True
        try:
            for name in order:
                try:
                    self._impl = self._make(name)
                    self.name = name
                    if self.looks_black(self._first_frame()):
                        raise CaptureError("backend returned a black frame")
                    return
                except Exception as exc:  # noqa: BLE001 - fall through
                    errors.append(f"{name}: {exc}")
                    if self.name == name:
                        self.close()
                    self._impl = None
                    self._mss = None
                    self.name = ""
            raise CaptureError("no working capture backend (" + "; ".join(errors) + ")")
        finally:
            self._opening = False

    def _ensure_thread(self) -> None:
        """mss and bettercam hold thread-affine handles; rebuild on a new thread."""
        tid = threading.get_ident()
        if tid == self._tid:
            return
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
        self._impl = None
        self._mss = None
        self.name = ""
        self._tid = tid
        self._open()

    def _first_frame(self) -> np.ndarray:
        import time

        attempts = 20 if self.name == "bettercam" else 1
        for _ in range(attempts):
            try:
                frame = self.grab()
            except Exception:  # noqa: BLE001
                frame = None
            if frame is not None and getattr(frame, "size", 0):
                return frame
            time.sleep(0.05)
        raise CaptureError("no frame from backend")

    def _make(self, name: str):
        x, y, w, h = self.rect
        if name == "bettercam":
            import bettercam  # type: ignore

            cam = bettercam.create(output_color="BGR")
            if cam is None:
                raise RuntimeError("bettercam.create() returned None")
            try:
                cam.start(region=(x, y, x + w, y + h), target_fps=30, video_mode=False)
            except Exception:
                try:
                    cam.stop()
                except Exception:  # noqa: BLE001 - cleanup is best effort
                    pass
                raise
            return cam
        if name == "mss":
            import mss

            sct = mss.mss()
            self._mss = sct
            try:
                sct.grab({"left": x, "top": y, "width": w, "height": h})
            except Exception:
                sct.close()
                self._mss = None
                raise
            return sct
        if name == "pil":
            from PIL import ImageGrab

            return ImageGrab
        raise ValueError(f"unknown backend {name}")

    def grab(self) -> np.ndarray:
        if not self._opening:
            self._ensure_thread()
        x, y, w, h = self.rect
        try:
            if self.name == "bettercam":
                frame = self._impl.get_latest_frame()
                if frame is None:
                    raise CaptureError("bettercam returned no frame")
                return frame
            if self.name == "mss":
                shot = self._mss.grab({"left": x, "top": y, "width": w, "height": h})
                return np.asarray(shot)[:, :, :3].copy()
            if self.name != "pil" or self._impl is None:
                raise CaptureError("capture backend is not available")
            img = self._impl.grab(bbox=(x, y, x + w, y + h), all_screens=True)
            return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        except Exception as exc:
            # During BetterCam startup, a few empty frames are expected while
            # Desktop Duplication initializes; _first_frame owns that retry.
            if not self._opening:
                self.close()
            if isinstance(exc, CaptureError):
                raise
            raise CaptureError(f"{self.backend} frame acquisition failed: {exc}") from exc

    def close(self) -> None:
        impl, mss_obj, name = self._impl, self._mss, self.name
        self._impl = None
        self._mss = None
        self.name = ""
        if name == "bettercam" and impl is not None:
            try:
                impl.stop()
            except Exception:  # noqa: BLE001
                pass
        if mss_obj is not None:
            try:
                mss_obj.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def looks_black(img: np.ndarray, threshold: float = 0.985) -> bool:
        if img is None or img.size == 0 or img.ndim != 3 or img.shape[2] < 3:
            return True
        return float((img.max(axis=2) < 8).mean()) > threshold
