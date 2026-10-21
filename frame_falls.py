from __future__ import annotations

import math


# 10 s never completed on camera 33 (2026-10-21): its stream reconnects every ~30 s and drops 13-16 s of frames.
# A lie-down there was continuously seen for 6.5 s before a gap.
FALL_DWELL_SECONDS = 6.0
FALL_BOX_RATIO = 1.2
# A lying person's score swings around the person threshold, so sightings drop out for a frame or several.
# At ~2 FPS a 1 s limit restarted the dwell on every miss: a 26 s lie-down on the track never reached 10 s.
FALL_GAP_SECONDS = 5.0


def update_fall_state(box, frame_shape, state, now, dwell_seconds=FALL_DWELL_SECONDS):
    """A stationary person with a wide box is a suspected fall, not a confirmed one."""
    x1, y1, x2, y2 = box
    height, width = frame_shape[:2]
    box_width, box_height = x2 - x1, y2 - y1
    valid = (
        all(math.isfinite(value) for value in box)
        and box_width > 0 and box_height > 0
        and x1 > 2 and y1 > 2 and x2 < width - 2 and y2 < height - 2
    )
    was_active = state.get("active", False)
    gap = now - state.get("lastSeen", now)
    state["lastSeen"] = now
    if not valid:
        state.pop("since", None)
        active = was_active
    else:
        foot = ((x1 + x2) / 2, y2)
        anchor = state.get("anchor", foot)
        moved = math.dist(foot, anchor) > state.get("radius", max(10.0, box_height * 0.5))
        if "since" not in state or gap > FALL_GAP_SECONDS or moved:
            state.update(since=now, anchor=foot, radius=max(10.0, box_height * 0.5))
        duration = max(0.0, now - state["since"])
        # shortcut: bbox geometry cannot distinguish lying, crouching or merged people.
        active = (duration >= dwell_seconds or (was_active and not moved)) and box_width / box_height >= FALL_BOX_RATIO
        state["active"] = active
    event = "ENTER" if active and not was_active else "EXIT" if was_active and not active else "STAY" if active else "NONE"
    return {
        "fallState": "FALL_SUSPECTED" if active else "NORMAL" if valid else "UNKNOWN",
        "fallEvent": event,
        "fallNotify": event in {"ENTER", "EXIT"},
    }
