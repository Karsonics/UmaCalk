"""Screen capture with a backend fallback chain.

bettercam (DXGI Desktop Duplication) is the only backend that reliably grabs
exclusive-fullscreen DirectX output; mss and PIL.ImageGrab are the fallbacks
for windowed and borderless modes.
"""
from __future__ import annotations

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
        self._open()

    def _open(self) -> None:
        order = [self.backend] if self.backend != "auto" else ["bettercam", "mss", "pil"]
        errors: list[str] = []
        for name in order:
            try:
                self._impl = self._make(name)
                self.name = name
                if self.looks_black(self._first_frame()):
                    raise CaptureError("backend returned a black frame")
                return
            except Exception as exc:  # noqa: BLE001 - fall through
                errors.append(f"{name}: {exc}")
        raise CaptureError("no working capture backend (" + "; ".join(errors) + ")")

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
            cam.start(region=(x, y, x + w, y + h), target_fps=30, video_mode=False)
            return cam
        if name == "mss":
            import mss

            sct = mss.mss()
            sct.grab({"left": x, "top": y, "width": w, "height": h})
            self._mss = sct
            return sct
        if name == "pil":
            from PIL import ImageGrab

            return ImageGrab
        raise ValueError(f"unknown backend {name}")

    def grab(self) -> np.ndarray:
        x, y, w, h = self.rect
        if self.name == "bettercam":
            frame = self._impl.get_latest_frame()
            if frame is None:
                raise CaptureError("bettercam returned no frame")
            return frame
        if self.name == "mss":
            shot = self._mss.grab({"left": x, "top": y, "width": w, "height": h})
            return np.asarray(shot)[:, :, :3].copy()
        from PIL import Image

        img = self._impl.grab(bbox=(x, y, x + w, y + h), all_screens=True)
        return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()

    def close(self) -> None:
        if self.name == "bettercam" and self._impl is not None:
            try:
                self._impl.stop()
            except Exception:  # noqa: BLE001
                pass
        if self.name == "mss" and self._mss is not None:
            try:
                self._mss.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def looks_black(img: np.ndarray, threshold: float = 0.985) -> bool:
        if img is None or img.size == 0:
            return True
        return float((img.max(axis=2) < 8).mean()) > threshold
