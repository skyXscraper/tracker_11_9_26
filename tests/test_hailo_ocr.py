"""Tests for the Hailo recognition path that do not need the NPU.

Line splitting, input preparation and CTC decoding are plain array work, so
they can be pinned here on a laptop. Only the inference call itself needs the
device, and that is what `tools/test_hailo.py` is for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollocr.config import Config                       # noqa: E402
from rollocr.hailo_ocr import (                         # noqa: E402
    PAD_VALUE, REC_HEIGHT, REC_WIDTH,
    ctc_greedy_decode, preprocess_line, split_text_lines,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CFG = Config()


def load(name: str):
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None, f"missing fixture {name}"
    return image


# -- splitting the two written lines ----------------------------------------

def test_two_line_crop_splits_into_exactly_two_lines():
    """Regression: sparse handwriting fragmented this crop into five bands,
    because a raw row projection breaks wherever the pen lifted."""
    bands = split_text_lines(load("roll_ply99.jpg"), CFG.detect)
    assert len(bands) == 2, f"expected ply line + length line, got {bands}"

    (top_a, bottom_a), (top_b, bottom_b) = bands
    assert bottom_a < top_b, "bands must be disjoint and ordered top to bottom"
    assert (bottom_a - top_a) >= 8 and (bottom_b - top_b) >= 8


def test_clipped_crop_yields_a_single_line():
    """When the frame edge takes the ply line, only the lengths remain."""
    bands = split_text_lines(load("roll_54p7.jpg"), CFG.detect)
    assert len(bands) == 1


def test_split_never_returns_more_than_max_lines():
    bands = split_text_lines(load("roll_ply99.jpg"), CFG.detect, max_lines=1)
    assert len(bands) == 1


def test_split_on_blank_input_is_empty():
    blank = np.full((80, 200, 3), 240, dtype=np.uint8)
    assert split_text_lines(blank, CFG.detect) == []


# -- preparing the fixed 48x320 input ---------------------------------------

def test_preprocess_matches_the_compiled_input_shape():
    line = load("roll_54p7.jpg")[60:100]
    prepared = preprocess_line(line)
    assert prepared.shape == (REC_HEIGHT, REC_WIDTH, 3)
    assert prepared.dtype == np.uint8


def test_preprocess_pads_with_mid_grey_not_black():
    """PP-OCR normalises (x/255 - 0.5)/0.5 inside the HEF, so the trained
    model's zero padding is 128 going in. Padding with 0 would feed it a black
    bar it never saw in training."""
    narrow = np.full((40, 40, 3), 200, dtype=np.uint8)
    prepared = preprocess_line(narrow)
    assert prepared[:, -1, :].tolist() == [[PAD_VALUE] * 3] * REC_HEIGHT


def test_preprocess_preserves_aspect_ratio():
    line = np.full((48, 96, 3), 200, dtype=np.uint8)
    line[:, :48] = 10                       # left half dark
    prepared = preprocess_line(line)
    # 48x96 scales to 48x96, so the dark half should still end at x=48.
    assert prepared[24, 10, 0] < 100
    assert prepared[24, 80, 0] > 100


def test_preprocess_handles_an_over_wide_line():
    wide = np.full((20, 4000, 3), 200, dtype=np.uint8)
    assert preprocess_line(wide).shape == (REC_HEIGHT, REC_WIDTH, 3)


def test_preprocess_handles_an_empty_line():
    assert preprocess_line(np.zeros((0, 10, 3), np.uint8)).shape == (REC_HEIGHT, REC_WIDTH, 3)


# -- CTC decoding ------------------------------------------------------------

CHARSET = ["<blank>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]


def logits_for(indices, classes=len(CHARSET), confidence=0.9):
    out = np.full((len(indices), classes), (1.0 - confidence) / (classes - 1), np.float32)
    for step, index in enumerate(indices):
        out[step, index] = confidence
    return out


def test_ctc_collapses_repeats_and_drops_blanks():
    # "4", "7", blank, "1", "1" -> 47.11 becomes "4711" with the repeat kept
    # only because a blank separates the two ones.
    text, _ = ctc_greedy_decode(logits_for([5, 5, 0, 8, 0, 2, 0, 2]), CHARSET)
    assert text == "4711"


def test_ctc_merges_adjacent_duplicates():
    text, _ = ctc_greedy_decode(logits_for([2, 2, 2]), CHARSET)
    assert text == "1", "an unbroken run is one character"


def test_ctc_decodes_a_full_reading():
    # 4 . 7 - 1 7 . 5  (blank-separated, as the network emits it)
    indices = [5, 0, 11, 0, 8, 0, 12, 0, 2, 0, 8, 0, 11, 0, 6]
    text, confidence = ctc_greedy_decode(logits_for(indices), CHARSET)
    assert text == "4.7-17.5"
    assert 0.85 <= confidence <= 1.0


def test_ctc_all_blank_gives_nothing():
    text, confidence = ctc_greedy_decode(logits_for([0, 0, 0]), CHARSET)
    assert text == ""
    assert confidence == 0.0


def test_ctc_ignores_indices_beyond_the_charset():
    """A charset that disagrees with the model's class count must not crash."""
    text, _ = ctc_greedy_decode(logits_for([5, 0, 40], classes=64), CHARSET)
    assert text == "4"


@pytest.mark.parametrize("shape", [(1, 40, len(CHARSET)), (40, len(CHARSET))])
def test_ctc_accepts_batched_or_flat_output(shape):
    """The device returns 1x40x18385; a squeezed array must decode the same."""
    data = np.zeros(shape, np.float32)
    data.reshape(-1, len(CHARSET))[:, 0] = 1.0
    assert ctc_greedy_decode(data, CHARSET)[0] == ""
