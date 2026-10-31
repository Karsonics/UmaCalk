"""Text parsing: numbers, distances, card labels."""
from __future__ import annotations

import re
import unicodedata

from .model import SUBROW_LABELS

_NUM = re.compile(r"[-+]?\d[\d,.\sOoQDlI|!SBZg]*")
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0",
                            "l": "1", "I": "1", "|": "1", "!": "1",
                            "S": "5", "B": "8", "Z": "2", "g": "9"})


def clean(text: str) -> str:
    """Normalise unicode noise from OCR output."""
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u00a0", " ").strip()


def parse_int(text: str, *, signed: bool = True) -> int | None:
    """Extract an integer from an OCR string such as '+12,924 pts' or '1O,0OO'."""
    if not text:
        return None
    t = clean(text)
    t = re.sub(r"(?i)\bpts?\.?\b", "", t).strip()
    m = _NUM.search(t)
    if not m:
        return None
    raw = m.group(0)
    digits = re.sub(r"[^\d,.\-+]", "", raw)
    if re.search(r"[A-Za-z]", raw):
        digits = raw.translate(_DIGIT_FIX)
        digits = re.sub(r"[^\d,.\-+]", "", digits)
    # Thousands separators: drop commas and dots that are followed by exactly 3 digits.
    digits = re.sub(r"(?<=\d)[,.](\d{3})(?=\D|$)", r"\1", digits)
    digits = digits.replace(",", "").replace(".", "")
    if digits in ("", "-", "+"):
        return None
    try:
        val = int(digits)
    except ValueError:
        return None
    if not signed and val < 0:
        val = abs(val)
    return val


def is_negative(text: str) -> bool:
    return "-" in text


_DIST_ALIASES = {
    "sprint": "Sprint",
    "mile": "Mile",
    "medium": "Medium",
    "middle": "Medium",
    "long": "Long",
    "dirt": "Dirt",
}


def match_distance(text: str) -> str | None:
    t = clean(text).lower().strip(" .:")
    for key, val in _DIST_ALIASES.items():
        if t == key:
            return val
        if len(t) >= 4 and key.startswith(t[:4]):
            return val
    return None


_SUB_PREFIXES = (
    ("base", ("base point", "base score", "base")),
    ("ace", ("ace bonus",)),
    ("opponent", ("opponent rating", "opponent bonus", "opponent")),
    ("support", ("support bonus", "support")),
)


def match_subrow(text: str) -> str | None:
    """Classify a sub-row label.

    Prefix matching only - plain substring search matched "ace" inside
    "place", which silently turned every card title into an Ace bonus row.
    """
    t = clean(text).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    for key, prefixes in _SUB_PREFIXES:
        if any(t.startswith(p) for p in prefixes):
            return key
    if t.endswith("streak bonus") or t == "streak bonus":
        return "streak"
    return None


_CATEGORY_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(\d+)(st|nd|rd|th)\s+place", re.I), "placement"),
    (re.compile(r"^good positioning.*early", re.I), "position_early"),
    (re.compile(r"^good positioning.*mid", re.I), "position_mid"),
    (re.compile(r"^good positioning.*late", re.I), "position_late"),
    (re.compile(r"^good positioning", re.I), "position_other"),
    (re.compile(r"^unique skill", re.I), "skill_unique"),
    (re.compile(r"^rare skill", re.I), "skill_rare"),
    (re.compile(r"^skill activated|^skill\b", re.I), "skill_normal"),
    (re.compile(r"long shot", re.I), "long_shot"),
    (re.compile(r"strong start", re.I), "strong_start"),
    (re.compile(r"^victory", re.I), "victory"),
    (re.compile(r"target time|time bonus", re.I), "target_time"),
    (re.compile(r"rushed", re.I), "rushed"),
    (re.compile(r"length|nose|neck|head|distance", re.I), "margin"),
    (re.compile(r"1st, 2nd & 3rd|trio", re.I), "team_trio"),
    (re.compile(r"all placed|all top", re.I), "team_all_placed"),
]


def card_category(title: str) -> str:
    t = clean(title)
    for rx, cat in _CATEGORY_RULES:
        if rx.search(t):
            return cat
    return "other"


_RACE_HEADER = re.compile(r"(?:class\s*)?(\d+)\s*[,,]?\s*race\s*(\d+)", re.I)


def parse_race_header(text: str) -> tuple[int | None, int | None]:
    """'Class 3, Race 4' -> (3, 4)."""
    m = _RACE_HEADER.search(clean(text))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def subrow_label(key: str) -> str:
    return SUBROW_LABELS.get(key, key)
