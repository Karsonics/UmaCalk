"""Cross-checks between the list scan, the detail scans and the game's own totals."""
from __future__ import annotations

from dataclasses import dataclass

from .model import DISTANCES, Run


@dataclass
class Issue:
    level: str      # "error" | "warn" | "info"
    scope: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.scope}: {self.message}"


def check(run: Run, expect_races: int = 5) -> list[Issue]:
    issues: list[Issue] = []
    rows = run.rows

    if not rows:
        issues.append(Issue("error", "list", "no umas parsed"))
        return issues

    seen: dict[str, int] = {}
    for r in rows:
        seen[r.name] = seen.get(r.name, 0) + 1
    for name, count in seen.items():
        if count > 1:
            issues.append(Issue("error", name, f"appears {count} times"))
    if len(rows) % 5:
        issues.append(Issue("warn", "list", f"{len(rows)} umas is not a multiple of 5"))
    if len(rows) > 15:
        issues.append(Issue("error", "list", f"{len(rows)} umas (max 15)"))

    for r in rows:
        if r.score is None:
            issues.append(Issue("error", r.name, "missing score"))
        elif not (1000 <= r.score <= 99999):
            issues.append(Issue("warn", r.name, f"implausible score {r.score}"))
        if r.distance is None:
            issues.append(Issue("warn", r.name, "distance not read"))
        elif r.distance not in DISTANCES:
            issues.append(Issue("warn", r.name, f"bad distance {r.distance!r}"))
        if r.confidence < 0.85:
            issues.append(Issue("warn", r.name, f"low OCR confidence {r.confidence:.2f}"))

    per_distance: dict[str, int] = {}
    for r in rows:
        if r.distance:
            per_distance[r.distance] = per_distance.get(r.distance, 0) + 1
    for d, n in per_distance.items():
        if n > 3:
            issues.append(Issue("warn", d, f"{n} umas (max 3)"))

    if run.match_total is not None and rows and run.computed_total != run.match_total:
        issues.append(Issue(
            "warn", "totals",
            f"sum of umas {run.computed_total:,} != match total {run.match_total:,}"))

    for race in run.races:
        scope = f"race {race.race_no}"
        for u in race.umas:
            if u.gained is not None and u.cards:
                if u.card_sum != u.gained:
                    issues.append(Issue(
                        "error", scope, f"{u.name}: cards {u.card_sum:,} != gained {u.gained:,}"))
            elif u.gained is not None and not u.cards:
                issues.append(Issue("warn", scope, f"{u.name}: no cards captured"))
        if race.total is not None and race.umas:
            parts = (race.team_bonus.total if race.team_bonus else 0) + sum(
                u.gained or 0 for u in race.umas)
            if len(race.umas) >= 3 and parts != race.total:
                issues.append(Issue(
                    "error", scope, f"parts {parts:,} != race total {race.total:,}"))
        for u in race.umas:
            known = {r.name for r in rows}
            if u.name not in known:
                issues.append(Issue("warn", scope, f"{u.name} not in the Score Info list"))

    if run.races and len(run.races) < expect_races:
        issues.append(Issue("warn", "detail", f"only {len(run.races)}/{expect_races} races captured"))
    if not run.races:
        issues.append(Issue("info", "detail", "no per-race breakdown captured"))
    return issues
