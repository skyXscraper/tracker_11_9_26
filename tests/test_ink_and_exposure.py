"""Tests for ink of any colour, and for exposure correction.

Synthetic strokes are drawn here rather than loaded, so each pen colour is
exactly controlled and the test says unambiguously which colours are found.
The real-footage behaviour -- that wrap crinkle is not mistaken for ink -- is
pinned against the committed crops of genuine handwriting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import Config                            # noqa: E402
from rollocr.detect import any_ink_mask, ink_mask, red_ink_mask  # noqa: E402
from rollocr.exposure import clipped_fraction, correct_exposure  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

PENS = {                     # BGR
    "black": (30, 30, 30),
    "blue": (160, 60, 20),
    "red": (60, 50, 200),
    "green": (40, 130, 30),
}


def wrap_with_writing(ink_bgr, size=(160, 420)):
    """A bright wrap with the lengths written on it in one pen colour."""
    height, width = size
    canvas = np.full((height, width, 3), 232, np.uint8)
    cv2.putText(canvas, "4.7-17.5", (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, ink_bgr, 5, cv2.LINE_AA)
    return canvas


def ink_pixels(image, cfg, for_crop=True):
    return int((ink_mask(image, cfg, for_crop=for_crop) > 0).sum())


# -- any colour of ink --------------------------------------------------------

@pytest.mark.parametrize("pen", list(PENS))
def test_any_colour_mode_finds_every_pen(pen):
    cfg = Config().detect
    assert cfg.ink_mode == "any", "all ink colours must be the default"
    found = ink_pixels(wrap_with_writing(PENS[pen]), cfg)
    assert found > 400, f"{pen} ink was not detected ({found} px)"


def test_red_mode_is_genuinely_red_only():
    """The legacy detector must still ignore a black pen -- that selectivity is
    the whole reason to keep it for red-only sites."""
    cfg = Config().detect
    cfg.ink_mode = "red"
    assert ink_pixels(wrap_with_writing(PENS["red"]), cfg) > 400
    assert ink_pixels(wrap_with_writing(PENS["black"]), cfg) < 50


def test_mode_switch_dispatches_to_the_right_detector():
    image = wrap_with_writing(PENS["black"])
    cfg = Config().detect
    cfg.ink_mode = "red"
    assert np.array_equal(ink_mask(image, cfg), red_ink_mask(image, cfg))
    cfg.ink_mode = "any"
    assert np.array_equal(ink_mask(image, cfg, for_crop=True),
                          any_ink_mask(image, cfg, for_crop=True))


def test_a_plain_wrap_is_not_ink():
    cfg = Config().detect
    blank = np.full((160, 420, 3), 232, np.uint8)
    assert ink_pixels(blank, cfg) == 0


@pytest.mark.parametrize("name", ["roll_54p7.jpg", "roll_ply99.jpg"])
def test_wrap_crinkle_is_not_mistaken_for_ink(name):
    """Glossy creases are thin, dark and neutral -- the same signature as black
    ink. At first the any-colour detector found 3,633 px of "ink" on a crop
    whose real writing is 752 px, almost all of it crinkle. It must now stay in
    the same range as the red detector, which only ever sees the pen strokes."""
    crop = cv2.imread(str(FIXTURES / name))
    cfg = Config().detect
    cfg.ink_mode = "red"
    real = ink_pixels(crop, cfg)
    cfg.ink_mode = "any"
    found = ink_pixels(crop, cfg)
    assert found <= real * 1.5, f"crinkle leaking in: {found} px vs {real} px of real ink"


def test_a_large_dark_region_is_not_a_stroke():
    """A forearm or the floor is dark but not thin; only strokes should count."""
    cfg = Config().detect
    image = np.full((200, 420, 3), 232, np.uint8)
    cv2.rectangle(image, (60, 40), (360, 170), (40, 40, 40), -1)
    inside = ink_mask(image, cfg, for_crop=True)[70:140, 100:320]
    assert (inside > 0).mean() < 0.05


# -- exposure -----------------------------------------------------------------

def overexposed(image):
    return np.clip(image.astype(np.float32) * 1.4 + 55, 0, 255).astype(np.uint8)


def test_correction_restores_contrast_to_a_washed_out_roll():
    cfg = Config().exposure
    washed = overexposed(wrap_with_writing(PENS["blue"]))
    fixed = correct_exposure(washed, cfg)
    assert cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY).std() > \
        cv2.cvtColor(washed, cv2.COLOR_BGR2GRAY).std()


def test_correction_keeps_the_pen_colour():
    """Only lightness is adjusted, so the ink detector still sees colour."""
    cfg = Config().exposure
    image = wrap_with_writing(PENS["red"])
    fixed = correct_exposure(image, cfg)

    def hue_of_ink(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        stroke = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 170
        return float(np.median(hsv[:, :, 0][stroke]))

    # Hue is circular and red sits on the wrap: 178 and 0 are both red, two
    # degrees apart, not 178.
    difference = abs(hue_of_ink(image) - hue_of_ink(fixed))
    assert min(difference, 180 - difference) < 8


def test_correction_can_be_switched_off():
    cfg = Config().exposure
    cfg.correct = False
    image = overexposed(wrap_with_writing(PENS["black"]))
    assert np.array_equal(correct_exposure(image, cfg), image)


def test_clipped_fraction_measures_blow_out():
    assert clipped_fraction(np.full((10, 10, 3), 255, np.uint8)) == 1.0
    assert clipped_fraction(np.full((10, 10, 3), 120, np.uint8)) == 0.0


def test_a_fully_clipped_roll_cannot_be_rescued_in_software():
    """Documents a limit rather than a feature: once the wrap has clipped to
    white the stroke is gone, which is why exposure is lowered at the camera."""
    cfg = Config()
    gone = np.full((160, 420, 3), 255, np.uint8)
    assert ink_pixels(correct_exposure(gone, cfg.exposure), cfg.detect) == 0


# -- camera settings ----------------------------------------------------------

def test_camera_exposure_is_left_alone_unless_configured():
    """An unconfigured camera must keep its own exposure; only an explicit
    setting should change what the sensor does."""
    capture = Config().capture
    assert capture.auto_exposure is None
    assert capture.exposure is None


def test_camera_exposure_is_applied_when_configured():
    from rollocr.sources import CameraSource

    class FakeCapture:
        def __init__(self):
            self.props = {}

        def set(self, prop, value):
            self.props[prop] = value
            return True

        def get(self, prop):
            return self.props.get(prop, 0)

    cfg = Config().capture
    cfg.auto_exposure = False
    cfg.exposure = 120
    capture = FakeCapture()
    CameraSource._apply_exposure(capture, cfg)

    assert capture.props[cv2.CAP_PROP_AUTO_EXPOSURE] == 1, "manual mode first"
    assert capture.props[cv2.CAP_PROP_EXPOSURE] == 120.0


# -- a marking is compact ------------------------------------------------------

def test_scattered_clutter_is_not_grouped_into_one_giant_marking():
    """Regression: the grouping step joined strokes scattered across the whole
    frame into single "markings", and OCR was handed crops averaging 442,000 px
    -- 37x a real marking -- so every read was slow and every result garbage."""
    from rollocr.detect import RollDetector

    cfg = Config()
    frame = np.full((720, 1280, 3), 232, np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(140):                                # sparse strokes everywhere
        x, y = int(rng.integers(20, 1240)), int(rng.integers(20, 690))
        cv2.line(frame, (x, y), (x + 14, y + 3), (35, 35, 35), 3)

    detector = RollDetector(cfg.detect, (1280, 720), cfg.exposure)
    frame_area = 1280 * 720
    for detection in detector.detect(frame):
        x0, y0, x1, y1 = detection.writing_bbox
        assert (x1 - x0) * (y1 - y0) / frame_area <= cfg.detect.max_writing_fraction * 1.5


def test_a_compact_marking_still_passes_the_size_gate():
    from rollocr.detect import RollDetector

    cfg = Config()
    frame = np.full((720, 1280, 3), 232, np.uint8)
    cv2.putText(frame, "88", (560, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (30, 30, 30), 6)
    cv2.putText(frame, "48.3-65.4", (470, 380), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (30, 30, 30), 6)

    detections = RollDetector(cfg.detect, (1280, 720), cfg.exposure).detect(frame)
    assert detections, "a real, compact marking must not be rejected as clutter"
