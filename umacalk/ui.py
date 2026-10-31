"""Tkinter front end: tray icon, global hotkeys, results table.

Hotkeys (global, work while the game has focus):
    F8  scan the Score Info list      (open it first)
    F7  scan all five race breakdowns (open Race Results first)
    F6  export the current run
    F9  abort a running scan
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
import traceback
from tkinter import messagebox, ttk

import pystray
from PIL import Image, ImageDraw
from pynput import keyboard

from . import app, config, probe
from .driver import Aborted

HINTS = "F8 scan uma list   |   F7 scan race details   |   F6 export   |   F9 abort"


class Gui:
    def __init__(self, cfg: config.Config, cfg_path: str) -> None:
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.session: app.Session | None = None
        self.abort = threading.Event()
        self.busy = False
        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.row_ids: list[int] = []
        self.last_result: app.Result | None = None

        self.root = tk.Tk()
        self.root.title("umaCalk")
        self.root.geometry("760x560")
        self.root.minsize(640, 480)
        self._build()
        self._start_hotkeys()
        self._start_tray()
        self.root.after(120, self._drain)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)

    # ------------------------------------------------------------------ ui

    def _build(self) -> None:
        root = self.root
        top = ttk.Frame(root, padding=(10, 8))
        top.pack(fill="x")

        ttk.Label(top, text="umaCalk", font=("Segoe UI", 15, "bold")).pack(side="left")
        self.status = ttk.Label(top, text="idle", foreground="#555")
        self.status.pack(side="right")

        bar = ttk.Frame(root, padding=(10, 0))
        bar.pack(fill="x")
        self.btn_list = ttk.Button(bar, text="Scan uma list (F8)", command=self.scan_list)
        self.btn_list.pack(side="left", padx=(0, 6))
        self.btn_races = ttk.Button(bar, text="Scan races (F7)", command=self.scan_races)
        self.btn_races.pack(side="left", padx=(0, 6))
        self.btn_export = ttk.Button(bar, text="Export (F6)", command=self.export)
        self.btn_export.pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Open folder", command=self.open_folder).pack(side="left")
        ttk.Button(bar, text="Settings", command=self.open_settings).pack(side="right")

        cols = ("n", "distance", "uma", "score", "conf")
        heads = {"n": "#", "distance": "Distance", "uma": "Uma", "score": "Score", "conf": "Conf"}
        widths = {"n": 40, "distance": 90, "uma": 260, "score": 100, "conf": 60}
        frame = ttk.Frame(root, padding=(10, 8, 10, 0))
        frame.pack(fill="both", expand=False)
        self.tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree.heading(c, text=heads[c])
            self.tree.column(c, width=widths[c], anchor="w" if c == "uma" else "center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", self._edit)
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        logf = ttk.Frame(root, padding=(10, 8))
        logf.pack(fill="both", expand=True)
        self.logbox = tk.Text(logf, height=8, wrap="word", state="disabled",
                              font=("Consolas", 9), background="#fbfbfb")
        self.logbox.pack(side="left", fill="both", expand=True)
        lsb = ttk.Scrollbar(logf, orient="vertical", command=self.logbox.yview)
        lsb.pack(side="right", fill="y")
        self.logbox.configure(yscrollcommand=lsb.set)

        ttk.Label(root, text=HINTS, foreground="#666", padding=(10, 0, 10, 8)).pack(fill="x")

    # -------------------------------------------------------------- helpers

    def log(self, msg: str) -> None:
        self.queue.put(("log", msg))

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.logbox.configure(state="normal")
                    self.logbox.insert("end", payload + "\n")
                    self.logbox.see("end")
                    self.logbox.configure(state="disabled")
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "rows":
                    self._fill_rows()
                elif kind == "busy":
                    self._set_busy(payload)
                elif kind == "error":
                    messagebox.showerror("umaCalk", payload)
        except queue.Empty:
            pass
        self.root.after(120, self._drain)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.btn_list, self.btn_races, self.btn_export):
            b.configure(state=state)

    def _fill_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.row_ids = []
        if not self.session:
            return
        rows = sorted(self.session.run.rows, key=lambda r: -(r.score or 0))
        for i, r in enumerate(rows):
            self.tree.insert("", "end", values=(
                i + 1, r.distance or "?", r.name, f"{r.score:,}" if r.score else "-",
                f"{r.confidence:.2f}"))
            self.row_ids.append(id(r))

    def _background(self, fn, status: str) -> None:
        if self.busy:
            return
        self.abort.clear()
        self.queue.put(("busy", True))
        self.queue.put(("status", status))

        def runner() -> None:
            try:
                fn()
            except Aborted:
                self.log("aborted")
            except Exception as exc:  # noqa: BLE001 - surface anything to the user
                self.log(traceback.format_exc())
                self.queue.put(("error", str(exc)))
            finally:
                self.queue.put(("busy", False))
                self.queue.put(("status", "idle"))

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    # --------------------------------------------------------------- actions

    def scan_list(self) -> None:
        def work() -> None:
            if self.session is None:
                self.session = app.open_live(self.cfg, self.log, self.abort)
            app.step_list(self.session)
            self.queue.put(("rows", None))
            self.log(f"list: {len(self.session.run.rows)} umas")
        self._background(work, "scanning uma list")

    def scan_races(self) -> None:
        def work() -> None:
            if self.session is None:
                self.session = app.open_live(self.cfg, self.log, self.abort)
            app.step_races(self.session)
            self.queue.put(("rows", None))
            if self.cfg.auto_save:
                self._export_now()
        self._background(work, "scanning race details")

    def export(self) -> None:
        if self.session is None:
            messagebox.showinfo("umaCalk", "Nothing captured yet.")
            return
        self._background(self._export_now, "exporting")

    def _export_now(self) -> None:
        assert self.session is not None
        self.last_result = app.finish(self.session)
        self.log("saved to " + self.cfg.output_path)

    def open_folder(self) -> None:
        path = self.cfg.output_path
        os.makedirs(path, exist_ok=True)
        os.startfile(path)  # type: ignore[attr-defined]

    def open_settings(self) -> None:
        config.write_default(self.cfg_path)
        os.startfile(self.cfg_path)  # type: ignore[attr-defined]

    def hide(self) -> None:
        self.root.withdraw()

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()

    def quit(self) -> None:
        try:
            self.listener.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self.icon.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
        self.root.destroy()

    # ------------------------------------------------------------ edit rows

    def _edit(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel or not self.session:
            return
        vals = self.tree.item(sel[0], "values")
        rows = sorted(self.session.run.rows, key=lambda r: -(r.score or 0))
        idx = int(vals[0]) - 1
        if idx < 0 or idx >= len(rows):
            return
        row = rows[idx]
        dlg = tk.Toplevel(self.root)
        dlg.title("Edit uma")
        dlg.transient(self.root)
        dlg.grab_set()
        fields = [("Name", row.name), ("Distance", row.distance or ""),
                  ("Score", str(row.score or ""))]
        entries = []
        for i, (label, value) in enumerate(fields):
            ttk.Label(dlg, text=label).grid(row=i, column=0, sticky="w", padx=8, pady=4)
            e = ttk.Entry(dlg, width=28)
            e.insert(0, value)
            e.grid(row=i, column=1, padx=8, pady=4)
            entries.append(e)

        def ok() -> None:
            row.name = entries[0].get().strip() or row.name
            dist = entries[1].get().strip()
            row.distance = dist or None
            try:
                row.score = int(entries[2].get().replace(",", "").strip())
            except ValueError:
                pass
            dlg.destroy()
            self._fill_rows()

        ttk.Button(dlg, text="Save", command=ok).grid(row=3, column=1, sticky="e", padx=8, pady=8)

    # ------------------------------------------------------------ tray + keys

    def _start_hotkeys(self) -> None:
        self.listener = keyboard.GlobalHotKeys({
            "<f8>": lambda: self.root.after(0, self.scan_list),
            "<f7>": lambda: self.root.after(0, self.scan_races),
            "<f6>": lambda: self.root.after(0, self.export),
            "<f9>": lambda: self._abort(),
        })
        self.listener.start()

    def _abort(self) -> None:
        self.abort.set()
        self.log("abort requested")

    def _start_tray(self) -> None:
        img = Image.new("RGB", (64, 64), (31, 78, 61))
        d = ImageDraw.Draw(img)
        d.text((22, 18), "U", fill=(255, 255, 255))
        menu = pystray.Menu(
            pystray.MenuItem("Show", lambda: self.root.after(0, self.show), default=True),
            pystray.MenuItem("Scan uma list", lambda: self.root.after(0, self.scan_list)),
            pystray.MenuItem("Scan races", lambda: self.root.after(0, self.scan_races)),
            pystray.MenuItem("Export", lambda: self.root.after(0, self.export)),
            pystray.MenuItem("Quit", lambda: self.root.after(0, self.quit)),
        )
        self.icon = pystray.Icon("umaCalk", img, "umaCalk", menu)

        def run() -> None:
            try:
                self.icon.run()
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()


def main() -> int:
    cfg_path = os.environ.get("UMACALK_CONFIG", os.path.join(os.getcwd(), "config.toml"))
    probe.enable_dpi_awareness()
    config.write_default(cfg_path)
    cfg = config.load(cfg_path)
    gui = Gui(cfg, cfg_path)
    gui.log(f"config {cfg_path}")
    gui.root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
