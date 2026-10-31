"""Append runs to a long-lived history workbook with per-uma averages."""
from __future__ import annotations

import os
from statistics import mean, pstdev

from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .model import Run

HEAD_FILL = PatternFill("solid", fgColor="1F4E3D")
HEAD_FONT = Font(bold=True, color="FFFFFF")


def _header(ws) -> None:
    for cell in ws[1]:
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"


def _open(path: str) -> Workbook:
    if os.path.exists(path):
        return load_workbook(path)
    wb = Workbook()
    ws = wb.active
    ws.title = "Runs"
    ws.append(["Run", "Date", "Distance", "Uma", "Score", "Race total"])
    _header(ws)
    return wb


def append_run(history_path: str, run: Run) -> str:
    wb = _open(history_path)
    ws = wb["Runs"]
    race_total = next((r.total for r in run.races if r.total), None)
    for row in run.rows:
        ws.append([run.run_id, run.captured_at, row.distance, row.name,
                   row.score, race_total])

    if "Averages" in wb.sheetnames:
        del wb["Averages"]
    wa = wb.create_sheet("Averages")
    wa.append(["Uma", "Runs", "Average", "Best", "Worst", "Std dev", "Consistency"])
    _header(wa)

    data: dict[str, list[int]] = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        name, score = r[3], r[4]
        if isinstance(name, str) and isinstance(score, int):
            data.setdefault(name, []).append(score)

    for name in sorted(data, key=lambda n: -mean(data[n])):
        scores = data[name]
        avg = mean(scores)
        sd = pstdev(scores) if len(scores) > 1 else 0.0
        wa.append([name, len(scores), round(avg), max(scores), min(scores),
                   round(sd), round(1 - sd / avg, 3) if avg else 0.0])

    for row in wa.iter_rows(min_row=2, min_col=3, max_col=6):
        for cell in row:
            if isinstance(cell.value, (int, float)):
                cell.number_format = "#,##0"
    for row in wa.iter_rows(min_row=2, min_col=7, max_col=7):
        for cell in row:
            cell.number_format = "0.0%"
    for idx, width in {1: 24, 2: 7, 3: 11, 4: 10, 5: 10, 6: 10, 7: 12}.items():
        wa.column_dimensions[get_column_letter(idx)].width = width

    if wa.max_row > 1:
        chart = BarChart()
        chart.title = "Average score per uma"
        chart.y_axis.title = "pts"
        chart.height = 9
        chart.width = 18
        ref = Reference(wa, min_col=3, min_row=1, max_row=wa.max_row)
        cats = Reference(wa, min_col=1, min_row=2, max_row=wa.max_row)
        chart.add_data(ref, titles_from_data=True)
        chart.set_categories(cats)
        wa.add_chart(chart, f"I{2}")

    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    wb.save(history_path)
    return history_path
