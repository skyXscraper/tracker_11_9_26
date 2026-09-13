"""Finding rolls by their handwritten marking, in any colour of ink.

Two detectors live here, chosen by `detect.ink_mode`.

**"any" (the default)** finds pen strokes whatever colour the pen was. A marking
on a white wrap stands out in one of two ways, and the detector accepts either:

  * it is *darker* than the wrap -- black, dark blue, most pens. Found with a
    morphological black-hat on lightness, which responds to thin dark features
    and ignores large dark regions such as a forearm or the floor.
  * it is *more colourful* than the wrap -- red, green, pale blue. Pale ink can
    be barely darker than glossy wrap, so lightness alone misses it; chroma (the
    distance from neutral grey) catches it. Passed through a top-hat, so again
    only thin coloured features count, not a whole hand.

Both are then restricted to strokes sitting on a bright, near-neutral surface,
which is what rejects cables, machinery and clothing.

Measured on real frames against the red-only detector it replaced: both found
every real marking (3/3), but "any" produced roughly twice the false candidates
on factory clutter (12 against 5 over 8 frames). Red is rare in a factory, so a
red-only gate is naturally more selective. Most of those extras are discarded
downstream -- by the tracker's hit count, the motion filter, OCR ranking and
voting -- but they are real, which is why the red detector is kept.

**"red"** is the original detector, for sites that only ever use a red pen and
want its extra selectivity. Tuned on real footage: through the glossy wrap red
ink photographs *pale magenta* (H~170, S 40-80), while skin -- the dangerous
false positive -- sits on the orange side of the hue wheel at higher saturation,
so gating on the magenta side separates the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]          # roll body, full-resolution xyxy
    writing_bbox: tuple[int, int, int, int]  # just the marking, full-resolution xyxy
    ink_area: int
    strokes: int
    sharpness: float

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def writing_height(self) -> int:
        return self.writing_bbox[3] - self.writing_bbox[1]


def ink_mask(bgr: np.ndarray, cfg, for_crop: bool = False) -> np.ndarray:
    """Binary mask of pen strokes, in whichever mode the site is configured for.

    ``for_crop`` matters because stroke width does: detection runs on a
    downscaled whole frame, where a pen stroke is a couple of pixels wide, while
    OCR works on full-resolution crops where it is several times wider.
    """
    if getattr(cfg, "ink_mode", "any") == "red":
        return red_ink_mask(bgr, cfg)
    return any_ink_mask(bgr, cfg, for_crop)


def ink_weight(bgr: np.ndarray, cfg, for_crop: bool = True) -> np.ndarray:
    """Continuous ink strength in [0, 1], for darkening strokes before OCR.

    The mask says where ink is; this says how strongly, so enhancement can
    deepen a faint stroke more than the surface around it without the ragged
    edges a hard threshold would leave.
    """
    if getattr(cfg, "ink_mode", "any") == "red":
        blue, green, red = (bgr[:, :, i].astype(np.float32) for i in range(3))
        return np.clip((red - np.maximum(green, blue)) / 45.0, 0.0, 1.0)

    dark, colour = _stroke_responses(bgr, cfg, for_crop)
    dark_w = np.clip(dark / max(cfg.dark_ink_min * 2.0, 1.0), 0.0, 1.0)
    colour_w = np.clip(colour / max(cfg.colour_ink_min * 2.0, 1.0), 0.0, 1.0)
    return np.maximum(dark_w, colour_w)


def _stroke_responses(bgr: np.ndarray, cfg, for_crop: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """How strongly each pixel reads as a thin dark stroke, and as a thin coloured one."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lightness = lab[:, :, 0]
    chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)

    # The kernel must be wider than a pen stroke and narrower than a hand.
    divisor = cfg.crop_stroke_kernel_div if for_crop else cfg.stroke_kernel_div
    size = max(5, (min(lightness.shape[:2]) // divisor) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dark = cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, kernel)
    colour = cv2.morphologyEx(chroma, cv2.MORPH_TOPHAT, kernel)
    return dark, colour


def any_ink_mask(bgr: np.ndarray, cfg, for_crop: bool = False) -> np.ndarray:
    """Pen strokes of any colour: thin and either darker or more colourful than
    the wrap, and sitting on a bright near-neutral surface."""
    dark, colour = _stroke_responses(bgr, cfg, for_crop)
    strokes = (dark > cfg.dark_ink_min) | (colour > cfg.colour_ink_min)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    surface = cv2.inRange(hsv, (0, 0, cfg.surface_val_min), (180, cfg.surface_sat_max, 255))
    surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # Grown so strokes near the edge of the wrap, which break its brightness,
    # still count as being on it.
    grow = max(3, cfg.surface_grow)
    surface = cv2.dilate(surface, np.ones((grow, grow), np.uint8))

    return (strokes & (surface > 0)).astype(np.uint8) * 255


def red_ink_mask(bgr: np.ndarray, cfg) -> np.ndarray:
    """Binary mask of red pen strokes, built to exclude skin."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Magenta side of red: where this pen's ink actually lands.
    mask = cv2.inRange(hsv, (cfg.hue_hi_min, cfg.sat_min, cfg.val_min), (180, 255, 255))

    # Orange side, kept deliberately narrow and strict.  Deep red ink under
    # different lighting shows up here, but so does skin, so the gates are hard.
    if cfg.hue_lo_max > 0:
        low = cv2.inRange(hsv, (0, max(cfg.sat_min, 90), cfg.val_min), (cfg.hue_lo_max, 255, 255))
        mask = mask | low

    blue, green, red = (bgr[:, :, i].astype(np.int16) for i in range(3))
    excess = (red - np.maximum(green, blue)) > cfg.red_excess_min
    return mask & (excess.astype(np.uint8) * 255)


def _tight_bounds(strokes: np.ndarray, group) -> tuple[int, int, int, int] | None:
    """Bounding box of the actual ink inside a grouped region."""
    gx, gy, gw, gh = group
    patch = strokes[gy:gy + gh, gx:gx + gw]
    points = cv2.findNonZero(patch)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return (gx + x, gy + y, w, h)


def _bright_body(bgr: np.ndarray) -> np.ndarray:
    """The roll itself: bright, near-neutral wrap."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 150), (180, 90, 255))


def sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian -- used to skip motion-blurred frames."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class RollDetector:
    def __init__(self, cfg, frame_size: tuple[int, int], exposure_cfg=None) -> None:
        self.cfg = cfg
        self.exposure_cfg = exposure_cfg
        self.frame_w, self.frame_h = frame_size
        # Anisotropic on purpose: writing runs horizontally, and the two
        # written lines sit close together vertically.  A square kernel big
        # enough to bridge the gap either side of the dash also swallows
        # unrelated red marks metres away in the background.
        self._group_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.group_gap_x, cfg.group_gap_y))
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(cfg.close_kernel))
        self._roi_px = self._roi_pixels()

    def _roi_pixels(self) -> tuple[int, int, int, int] | None:
        if not self.cfg.roi:
            return None
        x0, y0, x1, y1 = self.cfg.roi
        return (int(x0 * self.frame_w), int(y0 * self.frame_h),
                int(x1 * self.frame_w), int(y1 * self.frame_h))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        cfg = self.cfg
        scale = cfg.scale
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # On the downscaled copy, so correcting a blown-out frame costs little.
        # Faint ink on an overexposed roll is not found at all without it.
        if self.exposure_cfg is not None and self.exposure_cfg.correct_detection:
            from .exposure import correct_exposure
            small = correct_exposure(small, self.exposure_cfg)

        strokes_mask = ink_mask(small, cfg)
        strokes_mask = cv2.morphologyEx(strokes_mask, cv2.MORPH_CLOSE, self._close_kernel)

        # Join the ply line and the length line into one marking.
        grouped = cv2.morphologyEx(strokes_mask, cv2.MORPH_CLOSE, self._group_kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)

        gray = None
        detections: list[Detection] = []
        for i in range(1, count):
            gx, gy, gw, gh, area = stats[i]
            if area < cfg.min_group_area:
                continue

            patch = (strokes_mask[gy:gy + gh, gx:gx + gw] > 0).astype(np.uint8)
            n_sub, _, sub_stats, _ = cv2.connectedComponentsWithStats(patch, 8)
            strokes = sum(1 for j in range(1, n_sub) if sub_stats[j, 4] >= cfg.min_stroke_area)
            if strokes < cfg.min_strokes:
                continue

            # Tighten onto the ink itself.  The grouped component is a
            # dilated blob; cropping to it hands OCR a picture that is mostly
            # floor and forearm, with the writing too small to resolve.
            tight = _tight_bounds(strokes_mask, (gx, gy, gw, gh))
            if tight is None:
                continue
            tx, ty, tw, th = tight

            # A real marking is two short lines of strokes: compact and dense.
            # Scattered clutter joined by the grouping step is neither, and
            # without these two checks it was handed to OCR as crops averaging
            # 442,000 px -- 37x a real marking -- which made every read slow
            # and every result garbage. Measured on real frames: no real
            # marking exceeded 2.7% of the frame, while clutter groups reached
            # the entire frame.
            frame_fraction = (tw * th) / float(small.shape[0] * small.shape[1])
            if frame_fraction > cfg.max_writing_fraction:
                continue
            density = int((strokes_mask[ty:ty + th, tx:tx + tw] > 0).sum()) / max(tw * th, 1)
            if density < cfg.min_writing_density:
                continue

            # Back to full resolution.
            wx0, wy0 = int(tx / scale), int(ty / scale)
            wx1, wy1 = int((tx + tw) / scale), int((ty + th) / scale)
            if (wy1 - wy0) < cfg.min_text_height_px:
                continue

            body = self._roll_bbox(small, (tx, ty, tw, th), scale)
            if self._roi_px and not self._overlaps_roi(body):
                continue

            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            focus = sharpness(gray[wy0:wy1, wx0:wx1])

            detections.append(Detection(
                bbox=body,
                writing_bbox=(wx0, wy0, wx1, wy1),
                ink_area=int(area / (scale * scale)),
                strokes=strokes,
                sharpness=focus,
            ))

        detections.sort(key=lambda d: -d.ink_area)
        return detections

    def _roll_bbox(self, small: np.ndarray, group, scale: float) -> tuple[int, int, int, int]:
        """Grow the writing box onto the roll body it sits on."""
        gx, gy, gw, gh = group
        pad_x = int(gw * self.cfg.roll_pad_frac)
        pad_y = int(gh * self.cfg.roll_pad_frac)
        sx0, sy0 = max(0, gx - pad_x), max(0, gy - pad_y)
        sx1 = min(small.shape[1], gx + gw + pad_x)
        sy1 = min(small.shape[0], gy + gh + pad_y)

        region = small[sy0:sy1, sx0:sx1]
        body = _bright_body(region)
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 0.05 * region.shape[0] * region.shape[1]:
                bx, by, bw, bh = cv2.boundingRect(largest)
                sx0, sy0 = sx0 + bx, sy0 + by
                sx1, sy1 = sx0 + bw, sy0 + bh

        return (max(0, int(sx0 / scale)), max(0, int(sy0 / scale)),
                min(self.frame_w, int(sx1 / scale)), min(self.frame_h, int(sy1 / scale)))

    def _overlaps_roi(self, bbox) -> bool:
        rx0, ry0, rx1, ry1 = self._roi_px
        x0, y0, x1, y1 = bbox
        return not (x1 < rx0 or x0 > rx1 or y1 < ry0 or y0 > ry1)
