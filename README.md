# umaCalk

Team Trials result capture for **Umamusume: Pretty Derby** (Steam).

Press two keys after a Team Trials run and get a `.txt` report, a detailed `.xlsx`,
and a running history workbook that tracks every uma's average score over time.

No injection, no memory reading, no traffic interception — it reads the screen and
simulates clicks, exactly as you would yourself.

(BIG TLDR ITS STILL IN REALLY EARLY DEVOLOPMENT SO ITS NOT EVEN CLOSE TO BEING DONE)

---

## What it captures

**From Team Trials → Race History → Score Info** (press `F8` with that screen open)

| Field | Example |
|---|---|
| Distance | Sprint / Mile / Medium / Long / Dirt |
| Uma | Silence Suzuka |
| Score | 77,405 |
| MVP / rank badge | `★`, `S` |

**From Team Trials → See All Race Results** (press `F7` with that screen open)

The tool clicks through all five races by itself and reads each breakdown:

- race number, class, race total
- team bonus card (`1st, 2nd & 3rd Place`, etc.)
- per uma: placement, gained score, and every card
  (`1st place`, `3/4 Length`, `Long Shot`, `Rare Skill Activated x2`,
  `Good Positioning (Mid-Race)`, …) with its Base / Ace / Opponent Rating /
  Support / Streak split

Every card is validated with `sum(rows) == total`, every uma with
`sum(cards) == gained score`, and the race with
`team bonus + uma scores == race total`. Misreads are flagged instead of exported
silently.

---

## Install

```
py -3.13 -m pip install -r requirements.txt
```

Python 3.10+ is fine; 3.13 is what this was developed against.

Then, from the project folder:

```
py -3.13 -m umacalk.ui
```

The first launch writes `config.toml` next to the current directory.

---

## Use

1. Finish your Team Trials run.
2. Open **Team Trials → Race History → Score Info**, press **F8**.
   The list scrolls, every uma is read, and the table fills in.
3. Go back and open **See All Race Results**, press **F7**.
   It clicks each of the five races, reads the breakdown, and closes them again.
4. Files appear in `runs/`, and `history.xlsx` is updated.

| Key / button | Action |
|---|---|
| `F8` | scan the uma list |
| `F7` | scan the five race breakdowns |
| `F6` | export again (after manual edits) |
| `F9` | abort a scan in progress |

Double-click a row in the table to correct a name, distance or score before
exporting.

Works in **windowed**, **borderless** and **exclusive fullscreen**. Window size and
monitor don't matter — the panel is located in the frame and used as the reference
for every crop and click.

---

## Output

```
runs/
  20261030_180740.json    raw capture, re-exportable
  20261030_180740.txt     readable report
  20261030_180740.xlsx    Run / Itemized / Races / Checks sheets
history.xlsx               all runs + per-uma averages chart
```

`history.xlsx` is the long game: it shows each uma's average, best, worst and
consistency across runs, which is how you decide who should be your Ace.

---

## Configuration (`config.toml`)

| Key | Default | Meaning |
|---|---|---|
| `window_title` | `umamusume` | window title to attach to |
| `output_dir` | `runs` | where per-run files go |
| `history_file` | `history.xlsx` | running workbook |
| `monitor_device` | `""` | pin capture to `\\.\DISPLAYn`; empty = auto |
| `capture_backend` | `auto` | `auto` / `bettercam` / `mss` / `pil` |
| `scroll_settle` | `0.35` | seconds to wait after a scroll |
| `click_pause` | `0.6` | seconds to wait after a click |
| `capture_details` | `true` | read race breakdowns too |
| `auto_click_races` | `true` | click through the race list |
| `auto_save` | `true` | export when the detail scan finishes |
| `ocr_target_width` | `1400` | upscale target before OCR |
| `name_threshold` | `82` | fuzzy name match cut-off |

If the wrong monitor is picked, run `py -3.13 -m umacalk probe` to list monitors and
set `monitor_device`.

---

## Dev

```
py -3.13 tests\run_tests.py                      # offline regression suite
py -3.13 -m umacalk.smoke                        # live check: capture one frame, click nothing
py -3.13 -m umacalk list  Samples\*.png --crop 279,178,1564,875
py -3.13 -m umacalk detail Samples\*.png --crop 279,178,1564,875
py -3.13 -m umacalk probe
```

The tests run entirely on the PNGs in `Samples\`; no game required. `Samples\` is
git-ignored because the screenshots contain your account name — drop your own PNGs
in to run the suite.

Layout:

```
umacalk/
  probe.py       window + monitor + DPI discovery
  capture.py     DXGI / mss / GDI frame sources
  driver.py      live and replay input drivers
  ocr.py         RapidOCR wrapper
  detect.py      white-panel and progress-bar detection
  scan_list.py   Score Info scroll + stitching
  scan_detail.py Score Details scroll + card parsing
  parse.py       number / label repair
  names.py       fuzzy uma name resolution
  reconcile.py   cross-checks
  export.py      txt + xlsx
  history.py     running workbook
  app.py         session orchestration
  ui.py          tray app, hotkeys, table
  cli.py         headless commands
```

---

## If a scan goes wrong

| Symptom | Cause |
|---|---|
| "game window is not focused" | click the game window, then press the hotkey again |
| "Score Info panel disappeared" | the popup was closed or another screen was opened mid-scan |
| scan stops after a few rows | the list stopped moving — raise `scroll_settle` to `0.5` |
| a card is missing from a race | it never fully entered the viewport; rescan that race |
| names look wrong | double-click the row and correct it, then `F6` |

The tool scrolls **one wheel notch per step** and refuses to send input unless the
game window is in the foreground, so it cannot run away scrolling another window.

---

## Notes

- Automation of the game is against Cygames' terms of service. This tool only reads
  the screen and moves the mouse; it never touches the game's process, files or
  network traffic. Use at your own risk.
- Everything stays on your machine. No network calls, no telemetry.
- Screenshots and run JSON are kept in `runs/`; delete them freely.
