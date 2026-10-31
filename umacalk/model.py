"""Core data types for umaCalk."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1

DISTANCES = ("Sprint", "Mile", "Medium", "Long", "Dirt")

# Canonical sub-row keys used on the Score Details cards.
SUBROW_LABELS = {
    "base": "Base Points",
    "ace": "Ace Bonus",
    "opponent": "Opponent Rating Bonus",
    "support": "Support Bonus",
    "streak": "Streak Bonus",
}


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class TextLine:
    text: str
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def w(self) -> int:
        return self.x2 - self.x1


@dataclass
class UmaRow:
    """One row of the Score Info list."""

    name: str
    score: int | None
    distance: str | None = None
    title: str | None = None
    rank: str | None = None
    mvp: bool = False
    y: int = 0
    confidence: float = 1.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreLine:
    label: str
    label_raw: str
    value: int


@dataclass
class DetailCard:
    title: str
    total: int | None
    rows: list[ScoreLine] = field(default_factory=list)
    category: str = "other"
    y: int = 0

    @property
    def row_sum(self) -> int:
        return sum(r.value for r in self.rows)

    @property
    def balanced(self) -> bool:
        return self.total is not None and self.row_sum == self.total


@dataclass
class UmaDetail:
    name: str
    placement: str | None = None
    gained: int | None = None
    cards: list[DetailCard] = field(default_factory=list)

    @property
    def card_sum(self) -> int:
        return sum(c.total or 0 for c in self.cards)

    @property
    def balanced(self) -> bool:
        return self.gained is not None and self.card_sum == self.gained


@dataclass
class RaceDetail:
    race_no: int | None = None
    class_no: int | None = None
    total: int | None = None
    team_bonus: DetailCard | None = None
    umas: list[UmaDetail] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Run:
    run_id: str
    captured_at: str
    rows: list[UmaRow] = field(default_factory=list)
    races: list[RaceDetail] = field(default_factory=list)
    match_total: int | None = None
    display: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema: int = SCHEMA_VERSION

    @property
    def computed_total(self) -> int:
        return sum(r.score or 0 for r in self.rows)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
