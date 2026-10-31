"""Text and Excel exports for one run."""
from __future__ import annotations

import os
from datetime import datetime

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .model import Run
from .parse import subrow_label
from .reconcile import Issue

HEAD_FILL = PatternFill("solid", fgColor="1F4E3D")
HEAD_FONT = Font(bold=True, color="FFFFFF")
WARN_FILL = PatternFill("solid", fgColor="FDE2E1")


def _fmt(n: int | None) -> str:
    return f"{n:,}" if isinstance(n, int) else "-"


# --------------------------------------------------------------------------- txt

def write_txt(run: Run, path: str, issues: list[Issue] | None = None) -> str:
    rows = sorted(run.rows, key=lambda r: -(r.score or 0))
    total = run.computed_total
    lines: list[str] = []
    lines.append(f"Team Trials - {run.captured_at}")
    lines.append(f"Run {run.run_id}")
    lines.append(f"Total {total:,} pts" + (
        f"   (in-game total {run.match_total:,})" if run.match_total else ""))
    lines.append("")
    lines.append(f"  {'#':>2}  {'Distance':<8} {'Uma':<22} {'Score':>7}  {'MVP':<3} {'Rank':<4}")
    lines.append("  " + "-" * 52)
    for i, r in enumerate(rows, 1):
        share = (r.score or 0) / total * 100 if total else 0
        lines.append(
            f"  {i:>2}  {(r.distance or '?'):<8} {r.name:<22} {_fmt(r.score):>7}  "
            f"{'*' if r.mvp else '':<3} {(r.rank or ''):<4} {share:5.1f}%")
    lines.append("")
    lines.append("By distance")
    for d in ("Sprint", "Mile", "Medium", "Long", "Dirt"):
        group = [r for r in rows if r.distance == d]
        if not group:
            continue
        sub = sum(r.score or 0 for r in group)
        lines.append(f"  {d:<8} {len(group)} uma(s)  {sub:>8,}  avg {sub // len(group):>7,}")

    for race in run.races:
        lines.append("")
        head = f"Race {race.race_no}"
        if race.class_no is not None:
            head += f" (Class {race.class_no})"
        if race.total:
            head += f"  total {race.total:,}"
        lines.append(head)
        if race.team_bonus and race.team_bonus.total:
            tb = race.team_bonus
            lines.append(f"  Team bonus   {tb.title:<28} {tb.total:>8,}")
        for u in race.umas:
            lines.append(f"  {u.name}  {u.gained or 0:>8,}"
                         + (f"  ({u.placement})" if u.placement else ""))
            for card in u.cards:
                lines.append(f"    {card.title:<34} {card.total or 0:>8,}")
                for row in card.rows:
                    lines.append(f"        {subrow_label(row.label):<26} {row.value:>8,}")
        for w in race.warnings:
            lines.append(f"  ! {w}")

    if issues:
        lines.append("")
        lines.append("Checks")
        for issue in issues:
            lines.append(f"  {issue}")

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# -------------------------------------------------------------------------- xlsx

def _style_header(ws, ncols: int, row: int = 1) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def _autosize(ws, widths: dict[int, int]) -> None:
    for idx, width in widths.items():
        ws.column_dimensions[get_column_letter(idx)].width = width


def write_xlsx(run: Run, path: str, issues: list[Issue] | None = None) -> str:
    wb = Workbook()
    total = run.computed_total

    ws = wb.active
    ws.title = "Run"
    headers = ["#", "Distance", "Uma", "Score", "% of match", "MVP", "Rank", "OCR confidence"]
    ws.append(headers)
    for i, r in enumerate(sorted(run.rows, key=lambda r: -(r.score or 0)), 1):
        ws.append([
            i, r.distance or "", r.name, r.score,
            round((r.score or 0) / total, 4) if total else None,
            "yes" if r.mvp else "", r.rank or "", r.confidence,
        ])
    for row in ws.iter_rows(min_row=2, min_col=4, max_col=4):
        for cell in row:
            cell.number_format = "#,##0"
    for row in ws.iter_rows(min_row=2, min_col=5, max_col=5):
        for cell in row:
            cell.number_format = "0.0%"
    for row in ws.iter_rows(min_row=2, min_col=8, max_col=8):
        for cell in row:
            cell.number_format = "0.00"
            if (cell.value or 0) < 0.85:
                cell.fill = WARN_FILL
    _style_header(ws, len(headers))
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{ws.max_row}"
    _autosize(ws, {1: 4, 2: 10, 3: 24, 4: 10, 5: 12, 6: 6, 7: 6, 8: 15})

    wd = wb.create_sheet("Itemized")
    heads = ["Race", "Class", "Uma", "Place", "Card", "Category",
             "Total", "Base", "Ace", "Opponent", "Support", "Streak"]
    wd.append(heads)
    for race in run.races:
        for u in race.umas:
            for card in u.cards:
                vals = {r.label: r.value for r in card.rows}
                wd.append([
                    race.race_no, race.class_no, u.name, u.placement, card.title,
                    card.category, card.total, vals.get("base"), vals.get("ace"),
                    vals.get("opponent"), vals.get("support"), vals.get("streak"),
                ])
        if race.team_bonus:
            tb = race.team_bonus
            vals = {r.label: r.value for r in tb.rows}
            wd.append([
                race.race_no, race.class_no, "(team bonus)", "", tb.title,
                tb.category, tb.total, vals.get("base"), vals.get("ace"),
                vals.get("opponent"), vals.get("support"), vals.get("streak"),
            ])
    for row in wd.iter_rows(min_row=2, min_col=7, max_col=12):
        for cell in row:
            if isinstance(cell.value, int):
                cell.number_format = "#,##0"
    _style_header(wd, len(heads))
    wd.auto_filter.ref = f"A1:{get_column_letter(len(heads))}{wd.max_row}"
    _autosize(wd, {1: 6, 2: 6, 3: 22, 4: 7, 5: 36, 6: 16, 7: 9, 8: 8, 9: 8, 10: 10, 11: 9, 12: 8})

    wr = wb.create_sheet("Races")
    heads = ["Race", "Class", "Race total", "Team bonus", "Team card",
             "Own uma scores", "Check"]
    wr.append(heads)
    for race in run.races:
        gained = sum(u.gained or 0 for u in race.umas)
        ok = race.total is None or (race.team_bonus.total if race.team_bonus else 0) + gained == race.total
        wr.append([
            race.race_no, race.class_no, race.total,
            race.team_bonus.total if race.team_bonus else None,
            race.team_bonus.title if race.team_bonus else None,
            gained, "ok" if ok else "mismatch",
        ])
    for row in wr.iter_rows(min_row=2, min_col=3, max_col=6):
        for cell in row:
            if isinstance(cell.value, int):
                cell.number_format = "#,##0"
    _style_header(wr, len(heads))
    _autosize(wr, {1: 6, 2: 6, 3: 12, 4: 12, 5: 30, 6: 16, 7: 10})

    wn = wb.create_sheet("Checks")
    wn.append(["Level", "Scope", "Message"])
    for issue in issues or []:
        wn.append([issue.level, issue.scope, issue.message])
        if issue.level == "error":
            for cell in wn[wn.max_row]:
                cell.fill = WARN_FILL
    _style_header(wn, 3)
    _autosize(wn, {1: 8, 2: 14, 3: 70})

    # summary block on the Run sheet
    col = get_column_letter(len(headers) + 2)
    ws[f"{col}1"] = "Summary"
    ws[f"{col}1"].font = Font(bold=True)
    ws[f"{col}2"] = "Total"
    ws[f"{col}3"] = total
    ws[f"{col}3"].number_format = "#,##0"
    ws[f"{col}4"] = "Umas"
    ws[f"{col}5"] = len(run.rows)
    ws[f"{col}6"] = "Captured"
    ws[f"{col}7"] = run.captured_at

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wb.save(path)
    return path


def run_stem(run: Run) -> str:
    stamp = run.captured_at or datetime.now().isoformat(timespec="minutes")
    safe = run.run_id or stamp.replace(":", "").replace("-", "").replace("T", "_")[:15]
    return safe
