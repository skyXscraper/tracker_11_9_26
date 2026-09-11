"""Finding rolls by the one thing that is unique to them: the red marking.

Two facts from the test footage drive this module.

1.  The ink photographs *pale magenta*, not deep red.  Through the glossy wrap
    a stroke measures roughly H=170, S=40-80, with red-excess of only 25-45 --
    far weaker than a naive "red" gate expects.

2.  Human skin is the dangerous false positive, and it sits on the *opposite*
    side of the hue wheel (H~11, orange-red) with *higher* saturation than the
    ink.  Gating on the magenta side therefore separates ink from operators
    cleanly, which a plain 0-12 hue band does not -- that band locks onto faces.

Everything else (desks, floor stains, cardboard) is rejected by requiring
several ink strokes grouped together over a bright roll body.
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


def ink_mask(bgr: np.ndarray, cfg) -> np.ndarray:
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
    def __init__(self, cfg, frame_size: tuple[int, int]) -> None:
        self.cfg = cfg
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
