"""On-screen annotation for the live view.

Shows the marking in the format it is written in: the ply number, then the
lengths as ``start-end``. Colour carries the state so it reads at a glance
across a room -- grey while still reading, green once both lines have been
read, amber when only the lengths came through and the ply line did not.

Nothing here is checked against the packing list, so nothing on screen is ever
corrected towards an expected value. The numbers drawn are the numbers read.
"""

from __future__ import annotations

import cv2

STATUS_COLOURS = {
    "pending": (170, 170, 170),   # still reading
    "read": (80, 200, 90),        # ply and lengths both read
    "partial": (40, 170, 250),    # lengths read, ply line not legible
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(frame, text, origin, colour, scale=0.5, thickness=1):
    x, y = origin
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(frame, (x, y - th - baseline - 3), (x + tw + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 3, y - baseline), FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_tracks(frame, tracks, registry, camera: str):
    """Boxes, IDs and current readings for every live track in this camera."""
    for track in tracks:
        roll = registry.rolls.get(registry.assigned_id(camera, track.id) or "")
        # Only claim an identified roll's ID for the track that actually earned
        # it in this camera. Otherwise every box that happens to share the
        # record -- including background clutter -- gets labelled as the roll.
        if roll is not None and not roll.provisional:
            owner = roll.cameras.get(camera, {}).get("track_id")
            if owner is not None and owner != track.id:
                roll = None

        status = roll.status if roll and not roll.provisional else "pending"
        colour = STATUS_COLOURS.get(status, (170, 170, 170))

        x0, y0, x1, y1 = track.bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
        wx0, wy0, wx1, wy1 = track.writing_bbox
        cv2.rectangle(frame, (wx0, wy0), (wx1, wy1), (255, 255, 255), 1)

        global_id = roll.global_id if roll else "tracking"
        _label(frame, global_id, (x0, max(16, y0 - 26)), colour, 0.6, 2)

        # The marking as written: ply on top, lengths as start-end below.
        ply = track.confirmed_ply or _leading(track.ply_votes)
        if track.confirmed_range:
            values = "{}-{}".format(*track.confirmed_range)
        else:
            leader = _leading(track.range_votes)
            values = "{}-{}".format(*leader) if leader else "..."
        _label(frame, "ply {}".format(ply or "?"), (x0, max(32, y0 - 6)), colour)
        _label(frame, values, (x0, min(frame.shape[0] - 4, y1 + 18)), colour, 0.55)
    return frame


def _leading(votes):
    return votes.most_common(1)[0][0] if votes else None


def fps_colour(fps: float) -> tuple[int, int, int]:
    """Green when keeping up, amber when slipping, red when it will drop rolls."""
    if fps >= 12.0:
        return (90, 220, 110)
    if fps >= 6.0:
        return (60, 200, 240)
    return (70, 90, 240)


def draw_status(frame, camera: str, fps: float, stats: dict, top: int = 96):
    """Per-camera readout: the frame rate large, the detail underneath.

    ``top`` clears the combined view's header strip, which is painted over the
    stacked canvas afterwards and would otherwise hide this.
    """
    _label(frame, "{}  {:.1f} fps".format(camera, fps), (8, top),
           fps_colour(fps), 0.8, 2)
    detail = "tracks {}   ocr {} (dropped {})".format(
        stats.get("tracks", 0), stats.get("ocr", 0), stats.get("dropped", 0))
    _label(frame, detail, (8, top + 24), (220, 220, 220), 0.45)
    return frame


def draw_header(canvas, pipeline_fps: float, per_camera: dict, stats: dict):
    """A strip across the top of the combined view, led by the frame rate.

    This is the number that says whether the Pi is keeping up, so it gets the
    most prominent place on screen rather than being buried in a corner.
    """
    height = 34
    strip = canvas[:height]
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], height), (18, 18, 18), -1)
    cv2.addWeighted(strip, 0.85, strip, 0.0, 0, strip)

    cv2.putText(canvas, "{:.1f} FPS".format(pipeline_fps), (10, 24),
                FONT, 0.75, fps_colour(pipeline_fps), 2, cv2.LINE_AA)

    cameras = "   ".join("{} {:.1f}".format(name, fps)
                         for name, fps in sorted(per_camera.items()))
    detail = "{}   |   rolls {}   ocr {} ({} dropped)".format(
        cameras, stats.get("rolls", 0), stats.get("ocr", 0), stats.get("dropped", 0))
    cv2.putText(canvas, detail, (130, 23), FONT, 0.48, (225, 225, 225), 1, cv2.LINE_AA)
    return canvas


def draw_roi(frame, roi, colour=(90, 90, 200), label=""):
    if not roi:
        return frame
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
    cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 1)
    if label:
        _label(frame, label, (x0 + 4, y0 + 18), colour, 0.4)
    return frame


def stack(frames, scale: float = 0.5):
    """Put the two camera views side by side for a single window."""
    import numpy as np

    resized = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
               for f in frames if f is not None]
    if not resized:
        return None
    height = max(f.shape[0] for f in resized)
    padded = []
    for frame in resized:
        if frame.shape[0] < height:
            pad = np.zeros((height - frame.shape[0], frame.shape[1], 3), dtype=frame.dtype)
            frame = np.vstack([frame, pad])
        padded.append(frame)
    return np.hstack(padded)
