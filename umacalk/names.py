"""Character name resolution.

OCR strings are matched against a roster with a strict fuzzy threshold.  Name
matching is the one field where a single wrong character matters, so a match
must also be unique within a run (the game forbids duplicate characters).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from .parse import clean

# Playable characters on the Global client (extendable via data/names.txt).
ROSTER: tuple[str, ...] = (
    "Special Week", "Silence Suzuka", "Tokai Teio", "Maruzensky",
    "Fuji Kiseki", "Oguri Cap", "Gold Ship", "Vodka", "Daiwa Scarlet",
    "Taiki Shuttle", "Grass Wonder", "Hishi Amazon", "Mejiro McQueen",
    "El Condor Pasa", "T.M. Opera O", "Narita Brian", "Symboli Rudolf",
    "Air Groove", "Agnes Tachyon", "Sakura Bakushin O", "Seeking the Pearl",
    "Super Creek", "Inari One", "Winning Ticket", "Air Shakur",
    "Eishin Flash", "Curren Chan", "Manhattan Cafe", "Fine Motion",
    "Mejiro Dober", "Matikanefukukitaru", "Nice Nature", "King Halo",
    "Machikane Tannhauser", "Hishi Akebono", "Bamboo Memory",
    "Ikuno Dictus", "Daitaku Helios", "Shinko Windy", "Kitasan Black",
    "Satono Diamond", "Satono Crown", "Mejiro Ryan", "Mejiro Palmer",
    "Mejiro Ardan", "Mihono Bourbon", "Rice Shower", "Haru Urara",
    "Mr. C.B.", "Twin Turbo", "Marvelous Sunday", "Sakura Chiyono O",
    "Hokko Tarumae", "Zenno Rob Roy", "Aston Machan", "Sweep Tosho",
    "Narita Taishin", "Admire Vega", "Irisviel", "Super Express",
    "Tsurumaru Tsuyoshi", "Yukino Bijin", "Smart Falcon", "Mayano Top Gun",
    "Nishino Flower", "Hishi Miroku", "Tanino Gimlet", "Katsuragi Ace",
    "Symboli Kris S", "Tanino Gimlet", "Chrono Genesis", "Cheval Grand",
    "Verxina", "Vivlos", "Copano Rickey", "Wonder Acute", "Hokko Brave",
    "Meisho Doto", "Kawakami Princess", "Sweep Tosho", "K.S. Miracle",
    "Furioso", "Nakayama Festa", "Rulership", "Sounds of Earth",
    "Kitasan Black", "Satono Diamond", "Lucky Lilac", "Gran Alegria",
    "Loves Only You", "Efforia", "Shahryar", "Titleholder", "Do Deuce",
    "Mejiro Bright", "Mejiro Ramonu", "Mejiro Lourdes",
)


@dataclass
class Match:
    name: str
    score: float
    exact: bool = False


@dataclass
class NameResolver:
    roster: tuple[str, ...] = ROSTER
    threshold: float = 82.0
    learned_path: str | None = None
    extra: list[str] = field(default_factory=list)

    @staticmethod
    def _read_names(path: str) -> list[str] | None:
        """Read and normalize a valid learned-name list; None means preserve it."""
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError, UnicodeError, RecursionError):
            return None
        if not isinstance(data, list):
            return None
        normalized: dict[str, str] = {}
        for name in data:
            if isinstance(name, str) and name.strip():
                clean_name = name.strip()
                normalized.setdefault(clean_name.casefold(), clean_name)
        return list(normalized.values())

    def __post_init__(self) -> None:
        names = set(self.roster) | set(self.extra)
        if self.learned_path and os.path.exists(self.learned_path):
            learned = self._read_names(self.learned_path)
            if learned is not None:
                names |= set(learned)
        normalized: dict[str, str] = {}
        for name in names:
            if isinstance(name, str) and name.strip():
                clean_name = name.strip()
                normalized.setdefault(clean_name.casefold(), clean_name)
        self._names = sorted(normalized.values(), key=str.casefold)
        self._lower = {n.lower(): n for n in self._names}

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def resolve(self, raw: str, taken: set[str] | None = None) -> Match | None:
        text = clean(raw)
        if not text:
            return None
        if text in self._lower.values():
            return Match(text, 100.0, True)
        pool = [n for n in self._names if not taken or n not in taken] or self._names
        hit = process.extractOne(text, pool, scorer=fuzz.WRatio, score_cutoff=self.threshold)
        if not hit:
            # second chance: strip punctuation / spacing noise
            squashed = "".join(ch for ch in text.lower() if ch.isalnum())
            best = None
            for n in pool:
                ns = "".join(ch for ch in n.lower() if ch.isalnum())
                s = fuzz.ratio(squashed, ns)
                if s >= self.threshold and (best is None or s > best[1]):
                    best = (n, s)
            if not best:
                return None
            return Match(best[0], float(best[1]))
        name, score, _ = hit
        return Match(name, float(score))

    def learn(self, name: str) -> None:
        """Persist a manually corrected name so it matches exactly next time."""
        if not self.learned_path or not isinstance(name, str) or not name.strip():
            return
        name = name.strip()
        if os.path.exists(self.learned_path):
            data = self._read_names(self.learned_path)
            # Do not replace an unreadable or structurally invalid user file.
            if data is None:
                return
        else:
            data = []
        if name.casefold() not in {entry.casefold() for entry in data}:
            data.append(name)
            parent = os.path.dirname(os.path.abspath(self.learned_path))
            os.makedirs(parent, exist_ok=True)
            with open(self.learned_path, "w", encoding="utf-8") as fh:
                json.dump(sorted(data), fh, indent=2)
            self.__post_init__()
