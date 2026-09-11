# Sheet-roll tracking and handwriting OCR, two cameras

Detects fibreglass sheet rolls as an operator carries them past two overlapping
cameras, tracks each roll, reads the red handwriting on it (ply number on top,
`start - end` lengths below), and reports one stable ID per roll across both
views. Runs on a Raspberry Pi 5 (2 GB); tested offline against recordings from
the cameras that will be deployed.

```
python run.py --cam1 videos/cam0_test1.mp4 --cam2 videos/cam2_test1.mp4   # offline
python run.py --cam1 /dev/video0 --cam2 /dev/video2 --no-display          # on the Pi
python -m pytest tests/ -q                                                # 46 tests
```

---

## What the footage taught us

Three findings shaped the design more than any design decision did. They are
worth reading before changing anything.

### 1. The ink is pale magenta, not red

Through the glossy wrap, a red pen stroke measures about **H=170, S=40-80**,
with a red-excess (`R - max(G,B)`) of only **25-45**. A conventional "red" HSV
gate — hue 0-12 with decent saturation — finds almost none of it.

Worse, that conventional gate locks onto **human skin**, which sits on the
*opposite* side of the hue wheel (H≈11, orange-red) at *higher* saturation than
the ink. The first working version of the detector confidently cropped the
operator's face on every frame.

So detection gates the **magenta side only** (`hue_hi_min = 158`). The orange
band is kept available but narrow and strict (`hue_lo_max = 6`), for deep-red
ink under different lighting. If you change the pen or the lighting, this is
the first thing to re-check — `tools/calibrate_roi.py` plus a few saved crops
will tell you quickly.

### 2. The separator between the two lengths is not consistent

The operator writes the lengths with whatever mark comes to hand. Sometimes a
dash (`54.7-9.2`), very often just another dot:

```
48·3·65·433     means  start 48.3   end 65.433
```

The digit string alone is ambiguous. Three constraints resolve it:

- each side carries at most one decimal point,
- both values are under 100 m (the list tops out at 65.433),
- the end reading normally exceeds the start.

Against the supplied master list this leaves **exactly one legal reading for 53
of the 54 rows**, and never discards the true one. The last row (`5.9`/`16`,
which could also read `5`/`9.16`) is settled by the ply number's own master
entry. `tests/test_parse.py` pins all 54.

The ordering rule is a **preference, not a filter**. A roll clipped by the frame
edge loses digits, and a reading that looks out of order is usually a truncated
read of a real value — throwing it away discards the only evidence there is.

### 3. PP-OCR misreads this handwriting, consistently

This is the honest limitation of the system, and the most important thing in
this document.

The recognition model is trained on printed text. On this handwriting it makes
*systematic* errors that no amount of preprocessing fixes. The roll marked

```
99
4·7 - 17·5
```

comes back as **`19`** and **`4.7-135`** — a looped 9 read as 1, a crossed 7 read
as 3, and the faint decimal point dropped. That result was identical across
every combination tried: raw crops, proportional ink darkening, hard
binarisation, CLAHE, unsharp masking, and scale factors from 1.5× to 4×. Two
things did help and are in the pipeline — cropping tight to the ink rather than
to a dilated blob, and darkening strokes *proportionally to redness* instead of
thresholding — but neither corrects the misreads above.

**What the system does about it.** The master list is a closed set of 54 rolls,
which turns an open-ended handwriting problem into a 54-way choice. The digits
that were read are scored against every row, and the best match above a
threshold identifies the roll. `19` + `4.7-135` matches ply 99 at 0.80 and the
roll is correctly identified; the genuinely-absent test roll `54.7-9.2` matches
nothing and is reported as unidentified rather than forced onto a row.

**What it deliberately does not do.** The list decides *which roll this is*. It
never decides *what is written on it*. Every result carries both:

| column | meaning |
|---|---|
| `read_ply`, `read_start`, `read_end` | what the camera actually read |
| `matched_ply`, `expected_start`, `expected_end` | what the master list says |
| `ply_source` | `ocr_exact`, `values_exact`, or `fuzzy` |
| `status` | `match`, `value_mismatch`, or `unidentified` |

A `value_mismatch` row is the normal outcome for a fuzzy match, and it is
supposed to be visible. Nothing is silently corrected into a plausible number.

**If you need the read values themselves to be right** — as opposed to just
identifying the roll — the answer is a recogniser fine-tuned on this plant's
own handwriting. A few hundred labelled crops of real rolls would do it, and
the pipeline already saves suitable crops (`ocr.save_debug_crops`). That is a
data-collection task, not a code change.

---

## How it works

```
  camera ─┬─► detect (red ink) ─► track (IoU) ─┬─► OCR queue ─► PP-OCR/ONNX
          │                                     │                    │
          └─────────────── annotate ◄───────────┴──── vote ◄─────────┘
                                                       │
                                              master list: identify
                                                       │
                                            JSONL events + CSV summary
```

| module | responsibility |
|---|---|
| `sources.py` | USB cameras (newest frame wins) or video files (every frame) |
| `detect.py` | red-ink mask → stroke grouping → tight writing box + roll body |
| `track.py` | greedy IoU/centroid tracking, and multi-frame vote accumulation |
| `ocr.py` | PP-OCR on ONNX Runtime; one engine, one worker thread |
| `parse.py` | OCR text → `(ply, start, end)`, including the dotted-separator split |
| `master.py` | the list: exact lookup, value lookup, fuzzy match, validation |
| `identity.py` | one global ID per roll, across both cameras |
| `pipeline.py` | when it is worth spending an OCR call |
| `logio.py` | JSONL audit trail + CSV summary, both UTF-8 |

### Identity across the two cameras

Resolved in order of how much the evidence is worth:

1. **Ply number read off the roll** — unique in the list, decides outright.
2. **Exact length pair** — also unique per row, so it recovers a roll whose ply
   line was clipped off the top of frame.
3. **Fuzzy match** against the list, for the systematic misreads above.
4. **Co-occurrence in the overlap zone** — two tracks in both cameras at the
   same moment are the same roll, so the ID is stable from the first frame
   rather than appearing only once the writing is read.
5. **Twin merge** — an unidentified roll the other camera read the same way
   recently is the same roll. Its window (`twin_window_s`, 30 s) is separate
   from the co-occurrence window on purpose: in the test recordings the
   operator presented a roll to one camera and then walked to the other, 14 s
   apart, so a window sized for simultaneity split one roll into two records.

An ID moves in one direction only: unidentified → identified → never again. An
ID that has settled is never swapped because a later frame read the lengths
slightly differently. That rule exists because without it a single roll became
three rows in the results (`ROLL-U5`, `ROLL-547`, `ROLL-U11`) as its reading
drifted between frames.

### Staying inside a 2 GB Pi

- **One OCR engine, one worker thread**, shared by both cameras. Two sessions
  would double the largest allocation in the process for no throughput gain on
  four cores.
- **Bounded OCR queue.** When reads fall behind, new requests are dropped
  rather than queued — memory must not grow behind a roll that has already
  left the frame.
- **Detection on a downscaled copy**, every Nth frame; tracking carries the
  boxes in between. Detection measures ~3 ms per frame on a laptop.
- **OCR is gated hard**: only tracks that are established, moving, sharp, big
  enough in pixels, not already read, and not read within the last
  `min_interval_s`. Candidates are ranked so a stray red mark on the floor
  cannot starve the roll that matters.
- **Motion filter** rejects the rolls stacked on background racks: they never
  move, so they never earn an OCR call.

Offline runs use an inline OCR path instead of the worker thread, so a tuning
run processes every submitted crop and gives the same answer every time.

Measure the real numbers on the hardware:

```bash
python tools/bench_ocr.py --video videos/cam0_test2.mp4 --frames 150
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`data/master_list.csv` needs the columns `ply_no, start, end, length_m,
no_of_ply, item_number, item_description, packing_list`. Ply numbers and
`(start, end)` pairs are both unique in the supplied file, and `end - start`
equals `length_m` for every row — the pipeline relies on all three.

### On the Pi

```bash
python run.py --cam1 /dev/video0 --cam2 /dev/video2 --config config.pi.json --no-display
```

`config.pi.json` lowers the detection scale, detects every third frame, caps
OCR to one call per frame and two threads, and turns the display off. Confirm
which devices are which with `v4l2-ctl --list-devices`.

### Calibrating the zones

Worth doing once on site — it is what makes the cross-camera link work before
the writing has been read, and stops background racks being considered at all:

```bash
python tools/calibrate_roi.py --cam1 /dev/video0 --cam2 /dev/video2 --out config.json
python run.py --cam1 /dev/video0 --cam2 /dev/video2 --config config.json
```

### Options

| flag | effect |
|---|---|
| `--config FILE` | JSON overriding any field in `rollocr/config.py` |
| `--no-display` | headless (use on the Pi) |
| `--record` | write an annotated video of both views |
| `--realtime` | pace video files at their own frame rate, as a live rehearsal |
| `--max-seconds N` | stop after N seconds |
| `--quiet` | print only the first confirmation per roll |

### Output

`output/sightings.jsonl` — every read and every confirmation, in order,
including readings that disagreed with the list. `output/rolls.csv` — one row
per roll, rewritten as results firm up.

On screen, a strip across the top leads with the **pipeline frame rate** — how
fast the loop is actually getting through frames, which is the number that says
whether the Pi is keeping up — followed by each camera's own capture rate, the
roll count, and OCR calls with drops. The rate is colour-coded: green at 12 fps
and above, amber from 6, red below that, where reads start being dropped. Each
camera panel repeats its own rate beneath the strip.

Track boxes are grey while reading, green when the read matches the list, amber
when it disagrees (with the list values shown beneath), blue when nothing in the
list resembles it.

---

## Current results on the supplied recordings

| pair | rolls reported | outcome |
|---|---|---|
| `cam0_test3` / `cam2_test3` | 1 | correctly identified as **ply 99**, one ID across both cameras |
| `cam0_test1` / `cam2_test1` | 1 | reads `54.7-9.2`, which is in no master row — correctly reported as unidentified rather than forced onto a row, and still one ID across both cameras |
| `cam0_test2` / `cam2_test2` | 2 | ply 99 identified from camera 1; camera 2's partial read (`4 - 1.135`) scored too low to match and is reported separately as unidentified |

Two of the three passes give exactly one record for the one roll that went by.
The pair-2 split is the expected shape of a bad read: rather than guess, the
system reports what it saw and marks it unidentified.

The pair-1 roll appears to be a test roll genuinely absent from the list. If it
should be there, adding the row will let the system identify it.

## Known limitations

- **Read values are frequently wrong even when the roll is identified** — see
  finding 3. Identification is reliable; transcription is not. Treat
  `value_mismatch` rows as "identified, values unverified".
- **A roll clipped by the frame edge often loses its ply line**, which is why
  the length pair is used as a second route in.
- **A poor enough read is reported as its own unidentified roll** rather than
  merged into the right one (pair 2 above). Raising `ocr.min_votes` trades
  recall for fewer of these; lowering the fuzzy-match threshold in
  `master.best_match` does the opposite and risks false identifications.
- **The overlap zones are unset by default**, so step 4 above (linking before
  the writing is read) is inactive until you calibrate them; linking then falls
  back to step 5, which only fires once a roll has been read. Calibrate on site
  with `tools/calibrate_roi.py`. The default is deliberately off rather than
  "whole frame" — treating the whole frame as overlap linked unrelated tracks
  and ended up labelling desks and chair legs with the roll's ID.
- Tuned on one pen, one lighting setup and one camera model. Changing any of
  them means re-checking the ink gates in finding 1.
