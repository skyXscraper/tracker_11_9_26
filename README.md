# Sheet-roll tracking and handwriting OCR, two cameras

Detects fibreglass sheet rolls as an operator carries them past two overlapping
cameras, tracks each roll, reads the red handwriting on it (ply number on top,
`start - end` lengths below), and reports one stable ID per roll across both
views. Runs on a Raspberry Pi 5 (2 GB); tested offline against recordings from
the cameras that will be deployed.

```
python run.py --cam1 videos/cam0_test1.mp4 --cam2 videos/cam2_test1.mp4   # offline
python run.py --cam1 /dev/video0 --cam2 /dev/video2 --no-display          # on the Pi
python -m pytest tests/ -q                                                # 60 tests
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

Measured against a real packing list of 54 rows, this leaves **exactly one
legal reading for 53 of them**, and never discards the true one. The last
(`5.9`/`16`, equally readable as `5`/`9.16`) stays genuinely ambiguous and the
best-ranked candidate is taken — the list is not consulted to break the tie.
`tests/test_parse.py` pins all 54, feeding the parser only the digits.

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

**What the system does about it: nothing.** It reports the reading. An earlier
version scored the OCR'd digits against all 54 packing-list rows and adopted
the best-matching row's ply number, so `19` + `4.7-135` was reported as ply 99.
That was removed deliberately. The station exists to verify what is handwritten
on each roll, and a pipeline that resolves its answer against the expected
answer verifies nothing -- a mislabelled roll would quietly display the
expected numbers and pass.

So the packing list is not consulted at runtime at all: not for values, not for
identity, not to break a tie between two possible splits. Output is exactly
what was read, in the format the marking is written in:

| column | meaning |
|---|---|
| `ply_no` | the ply number as read, blank if the line was not legible |
| `range` | the lengths as read, `start-end` |
| `start`, `end`, `span_m` | the same values, split out |
| `status` | `read` (both lines) or `partial` (lengths only) |

`tools/compare_to_master.py` compares a finished run against the list
afterwards, with both columns side by side and the difference obvious.

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
                                         ply + start-end, as read
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
| `master.py` | the packing list, for after-the-fact comparison only |
| `identity.py` | one global ID per roll, across both cameras |
| `pipeline.py` | when it is worth spending an OCR call |
| `logio.py` | JSONL audit trail + CSV summary, both UTF-8 |
| `hailo_ocr.py` | optional PP-OCRv5 recognition on a Hailo NPU |

### Identity across the two cameras

From the reading and from tracking — never from a lookup:

1. **Ply number read off the roll.** Two tracks that read the same ply are the
   same roll, in either camera.
2. **Co-occurrence in the overlap zone** — two tracks in both cameras at the
   same moment are the same roll, so the ID is stable from the first frame
   rather than appearing only once the writing is read.
3. **Twin merge** — an unidentified roll the other camera read the same way
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

## Optional: PP-OCRv5 recognition on a Hailo NPU

The Pi AI Kit carries a **Hailo-8L**, and the Hailo Model Zoo publishes
pre-compiled PaddleOCR-v5 HEFs for that architecture. Setting
`ocr.backend = "hailo"` runs recognition on the NPU instead of the CPU.

```bash
mkdir -p ~/hailo_models && cd ~/hailo_models
wget https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/paddle_ocr_v5_mobile_recognition.hef
wget -O ppocrv5_dict.txt https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt

cd ~/tracker_11_9_26
python tools/test_hailo.py --image tests/fixtures/roll_ply99.jpg
```

`tools/test_hailo.py` runs the NPU and the CPU over the same crop and prints
both readings with timings. The fixtures carry known ground truth
(`roll_ply99.jpg` is ply 99, 4.7-17.5), so a correct result is recognisable
without a camera. Only switch the pipeline over once that looks right:

```bash
python run.py --cam1 /dev/video0 --cam2 /dev/video2 \
  --config config.pi.json --no-display   # add "ocr": {"backend": "hailo"} to the config
```

If the device or `hailo_platform` is missing the backend logs a warning and
falls back to the CPU path, so the same tree still runs on a laptop.

### What the compiled model forces

The HEF reports `input 48x320x3 UINT8` and `output 1x40x18385`, and three
consequences follow:

* **Fixed 48x320 input.** Lines are resized to 48 tall keeping aspect and
  padded to 320 wide. The pad value is **128, not 0** — PP-OCR normalises
  `(x/255 - 0.5)/0.5` inside the HEF, so the trained model's "zero padding" is
  mid-grey going in. Padding black feeds it a bar it never saw in training.
* **Softmax is already applied on-device**, over 40 CTC timesteps and 18385
  classes, so the host only does a greedy collapse against `ppocrv5_dict.txt`.
* **It recognises one line at a time.** A roll carries two, so the crop is
  split on the ink mask's horizontal projection first. That is far cheaper than
  running the 544x960 detection HEF to locate lines the colour detector has
  already found, and it preserves each line's vertical position — which is what
  tells ply from lengths in `parse.py`.

Handwriting is sparse, so a raw row projection breaks a single line into
fragments wherever the pen lifted; the mask is closed horizontally, the
projection smoothed, and bands closer together than half a line height merged.
`tests/test_hailo_ocr.py` pins that at two bands for a two-line crop.

### What to expect from it

Speed, and possibly a little accuracy. The NPU removes the ~200 ms CPU
recognition cost, which lets you relax the per-track OCR throttle and read a
roll on many more of the frames it is visible for — more votes, steadier
results. And because this is PP-OCR**v5** against the v4 weights RapidOCR ships,
the misreads in finding 3 may improve.

They may equally not. Run `tools/test_hailo.py` on `roll_ply99.jpg` and look at
what comes back: if v5 still reads `17.5` as `135`, the conclusion in finding 3
stands and no amount of hardware changes it — the fix is a recogniser fine-tuned
on this plant's handwriting.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### On the Pi, the venv must see the system packages

`sudo apt install hailo-all` installs `hailo_platform` into the *system* Python
(`/usr/lib/python3/dist-packages`), which a plain `python -m venv` cannot see.
Because the Hailo backend falls back to the CPU when that import fails, an
isolated venv leaves the pipeline running on the CPU while appearing to use the
NPU. Create it so it can see them:

```bash
sudo apt install -y python3-opencv python3-numpy
python -m venv --system-site-packages venv
source venv/bin/activate
python -c "import hailo_platform, cv2, numpy; print('ok')"
```

Raspberry Pi OS Trixie ships **Python 3.13**, where `rapidocr-onnxruntime` will
not install at all — only the newer `rapidocr` package supports it, and
`requirements.txt` selects between them with an environment marker. Running the
NPU backend needs neither: the system OpenCV and NumPy are enough.

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

Ground truth for pairs 2 and 3 is ply **99**, **4.7-17.5**; the pair-1 roll is
not in the packing list at all.

| pair | rolls | read as |
|---|---:|---|
| `cam0_test1` / `cam2_test1` | 1 | `ply 547` `54.7-9.21` — one ID across both cameras |
| `cam0_test2` / `cam2_test2` | 1 | `ply 19` `4.7-135` — one ID across both cameras |
| `cam0_test3` / `cam2_test3` | 2 | `ply 19` `4.1-15` and `ply ?` `4.7-135` — the two cameras read the roll differently, so they stay separate records |

The digits are wrong — `99` reads as `19`, `17.5` as `135` — and they are
reported wrong, which is the point. The earlier version matched these against
the packing list and printed ply 99, which looked far better and told you
nothing about what the camera could actually see.

Pair 3 shows the cost of the change honestly: without a list to force the two
readings together, two disagreeing reads of one roll stay two records. The
fixes for that are better recognition or a calibrated overlap zone, not a
lookup.

## Known limitations

- **Read values are frequently wrong** — see finding 3. The output is what the
  camera read, so a wrong read is reported as a wrong read rather than
  smoothed over. `tools/compare_to_master.py` will tell you how often, after
  the fact.
- **A roll clipped by the frame edge often loses its ply line**, which is why
  the length pair is used as a second route in.
- **Two disagreeing reads of one roll stay two records** (pair 3 above), since
  nothing outside the image is consulted to reconcile them. Raising
  `ocr.min_votes` trades recall for steadier readings; calibrating the overlap
  zones links the two views before either has been read.
- **The overlap zones are unset by default**, so step 4 above (linking before
  the writing is read) is inactive until you calibrate them; linking then falls
  back to step 5, which only fires once a roll has been read. Calibrate on site
  with `tools/calibrate_roi.py`. The default is deliberately off rather than
  "whole frame" — treating the whole frame as overlap linked unrelated tracks
  and ended up labelling desks and chair legs with the roll's ID.
- Tuned on one pen, one lighting setup and one camera model. Changing any of
  them means re-checking the ink gates in finding 1.
