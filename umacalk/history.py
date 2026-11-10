"""Append runs to a long-lived history workbook with per-uma averages."""
from __future__ import annotations

import os
import stat
import tempfile
import zipfile
from xml.etree.ElementTree import ParseError
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
        try:
            wb = load_workbook(path)
        except (OSError, ValueError, KeyError, IndexError, EOFError,
                zipfile.BadZipFile, ParseError) as exc:
            raise ValueError(f"Cannot read history workbook {path!r}: {exc}") from exc
        if "Runs" not in wb.sheetnames:
            wb.close()
            raise ValueError(f"History workbook {path!r} has no 'Runs' sheet")
        expected = ("run", "date", "distance", "uma", "score", "race total")
        header = tuple(
            str(cell.value).strip().casefold() if cell.value is not None else ""
            for cell in wb["Runs"][1][:len(expected)]
        )
        if header != expected:
            wb.close()
            raise ValueError(f"History workbook {path!r} has an invalid 'Runs' header")
        return wb
    wb = Workbook()
    ws = wb.active
    ws.title = "Runs"
    ws.append(["Run", "Date", "Distance", "Uma", "Score", "Race total"])
    _header(ws)
    return wb


def _save_workbook_atomic(wb: Workbook, history_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(history_path))
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        original_mode = (
            stat.S_IMODE(os.stat(history_path).st_mode)
            if os.path.exists(history_path) else None
        )
        fd, temp_path = tempfile.mkstemp(
            prefix=".umacalk-history-", suffix=".xlsx", dir=directory)
        os.close(fd)
        wb.save(temp_path)
        if original_mode is not None:
            # Preserve POSIX mode bits and Windows' read-only flag where supported.
            os.chmod(temp_path, original_mode)
        os.replace(temp_path, history_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                # The saved mode may be read-only; make the temporary file
                # removable before cleaning it up after a failed replace.
                os.chmod(temp_path, stat.S_IREAD | stat.S_IWRITE)
            except OSError:
                pass
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def _append_run_to_workbook(wb: Workbook, history_path: str, run: Run) -> str:
    ws = wb["Runs"]
    # Handle case where races is None (capture failure)
    race_total = None
    if run.races:
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

    _save_workbook_atomic(wb, history_path)
    return history_path


def append_run(history_path: str, run: Run) -> str:
    wb = _open(history_path)
    try:
        return _append_run_to_workbook(wb, history_path, run)
    finally:
        wb.close()
