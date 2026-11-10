"""Configuration loading/saving (TOML)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, fields

import tomli_w


@dataclass
class Config:
    window_title: str = "umamusume"
    output_dir: str = "runs"
    history_file: str = "history.xlsx"
    names_file: str = "data/names.txt"
    learned_names: str = "data/learned_names.json"
    monitor_device: str = ""
    capture_backend: str = "auto"
    scroll_settle: float = 0.35
    click_pause: float = 0.6
    capture_details: bool = True
    auto_click_races: bool = True
    auto_save: bool = True
    save_frames: bool = True
    ocr_target_width: int = 1400
    name_threshold: float = 82.0
    max_frames: int = 60
    base_dir: str = "."

    # -- helpers ------------------------------------------------------------
    def path(self, name: str) -> str:
        """Resolve a configured path and reject escapes, including symlinks."""
        base = os.path.realpath(os.path.abspath(self.base_dir))
        resolved = os.path.realpath(os.path.abspath(os.path.join(base, name)))
        try:
            common = os.path.commonpath((base, resolved))
        except ValueError:
            # Different Windows drives cannot share a common path.
            common = ""
        if os.path.normcase(common) != os.path.normcase(base):
            raise ValueError(f"Invalid path: {name!r} escapes base directory")
        return resolved

    @property
    def output_path(self) -> str:
        return self.path(self.output_dir)

    @property
    def history_path(self) -> str:
        return self.path(self.history_file)


def load(path: str) -> Config:
    cfg = Config(base_dir=os.path.dirname(os.path.abspath(path)) or ".")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        known = {f.name for f in fields(Config)}
        for key, value in data.items():
            if key in known and key != "base_dir":
                setattr(cfg, key, value)
    return cfg


def save(cfg: Config, path: str) -> None:
    data = {k: v for k, v in asdict(cfg).items() if k != "base_dir"}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        tomli_w.dump(data, fh)


DEFAULT_TOML = '''# umaCalk configuration
window_title = "umamusume"
output_dir = "runs"
history_file = "history.xlsx"
# leave empty to auto-pick the monitor holding the game window
monitor_device = ""
# auto | bettercam | mss | pil
capture_backend = "auto"

# time (seconds) to wait after each scroll and click
scroll_settle = 0.35
click_pause = 0.6

# capture the per-race score breakdown as well as the uma list
capture_details = true
# click through the five races on the Race Results screen
auto_click_races = true
# save every captured frame to runs/diag/ for troubleshooting
save_frames = true
# export automatically once the race details finish
auto_save = true

# OCR upscale target: 1400 works for windowed and fullscreen
ocr_target_width = 1400
# fuzzy name match threshold (0-100)
name_threshold = 82.0
max_frames = 60
'''


def write_default(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(DEFAULT_TOML)
