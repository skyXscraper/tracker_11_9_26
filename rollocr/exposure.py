"""Taming an overexposed roll, in the camera and after it.

The wrap is white and glossy, so under strong light it blows out: the surface
clips to pure white and thin pen strokes are washed out with it. Two things
follow, and both are measured rather than assumed.

**The real fix is at the camera.** Once a pixel has clipped to 255 its detail is
gone -- nothing downstream can bring it back. On test frames lifted until ~80%
of the image was clipped white, every enhancement tried (percentile stretch,
CLAHE, gamma, stroke darkening, each alone and combined) still produced garbage
readings. So `sources.py` lowers the camera's exposure *before* the highlights
clip, using the `capture.*exposure*` settings.

**Software correction still earns its place** on frames that are bright but not
clipped, and on recordings where the camera setting can no longer be changed.
It stretches the luminance between the darkest ink and the roll surface, then
restores local contrast with CLAHE. Colour is left alone -- only lightness is
adjusted -- so the ink detector still sees each pen's colour.

Mono conversion is done here too. The recogniser does not need colour to read
digits, and a single well-contrasted channel removes any dependence on the pen.
"""

from __future__ import annotations

import cv2
import numpy as np


def correct_exposure(bgr: np.ndarray, cfg) -> np.ndarray:
    """Bring a too-bright image back into a readable range.

    Works on lightness only, in LAB, so hue and saturation -- which the ink
    detector relies on for coloured pens -- are preserved.
    """
    if bgr is None or bgr.size == 0 or not cfg.correct:
        return bgr

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]

    # Stretch between the darkest ink and the roll surface. Percentiles, not
    # min/max, so a few specular glints or deep shadows cannot flatten it.
    low = float(np.percentile(lightness, cfg.low_percentile))
    high = float(np.percentile(lightness, cfg.high_percentile))
    if high - low > cfg.min_range:
        lightness = np.clip((lightness.astype(np.float32) - low) * 255.0 / (high - low),
                            0, 255).astype(np.uint8)

    # Overall exposure pull-down: values above 1 compress the highlights, which
    # is exactly where faint ink on a bright roll lives.
    if cfg.gamma and abs(cfg.gamma - 1.0) > 1e-3:
        table = (np.linspace(0, 1, 256) ** cfg.gamma * 255).astype(np.uint8)
        lightness = cv2.LUT(lightness, table)

    if cfg.clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                                tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        lightness = clahe.apply(lightness)

    lab[:, :, 0] = lightness
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clipped_fraction(bgr: np.ndarray, level: int = 250) -> float:
    """Share of pixels blown out to white. Above ~0.3, fix it at the camera."""
    if bgr is None or bgr.size == 0:
        return 0.0
    return float((bgr >= level).all(axis=2).mean())


def to_mono(bgr: np.ndarray) -> np.ndarray:
    """Single-channel lightness, returned as three channels for the recogniser."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
