# -*- coding: utf-8 -*-

"""
Congestion vs Parked (parking lot top-view) — ready to run.

Goal
-----
Detect:
  1) "PARKED" cars inside parking slots (immobile is normal there)
  2) "CONGESTION" only in the central aisle (immobile there is suspicious)

Key idea
--------
We split the top-view into:
  - AISLE segments (C0..C3)  -> used for congestion detection
  - SLOTS (Left 3x4, Right 3x4) -> used for parked detection

Decision logic
--------------
For each zone we compute:
  - occupancy  : fraction of pixels that differ from background (presence)
  - motion     : fraction of pixels changing frame-to-frame (movement)

PARKED(slot)    : occupancy high AND motion low for PARK_SEC
CONGESTED(aisle): occupancy high AND motion low for STOP_SEC
But to avoid false positives while a car is maneuvering into a slot:
  - if a "new slot becomes occupied" recently, we ignore aisle congestion
    for MANEUVER_GRACE_SEC seconds.

Controls
--------
  b : capture / recapture background (do this when scene is "empty enough")
  r : reset all timers/states
  q or ESC : quit

Notes
-----
- This script assumes you already have calib.json with:
    {"M": [[...],[...],[...]], "out_w": int, "out_h": int}
  produced by your homography calibration script.
- It runs on a webcam (CAM_INDEX) or RTSP stream (set RTSP_URL).

Dependencies
------------
  pip install opencv-python numpy requests

"""

import time
import json
import cv2
import numpy as np
from typing import Dict, Tuple


# ======================
# CONFIG (edit here)
# ======================

# --- Camera source ---
USE_RTSP = True
RTSP_URL = "rtsp://10.111.191.109:8554/cam"   # your phone stream
CAM_INDEX = 0                                  # used if USE_RTSP=False

# --- Homography calibration file ---
CALIB_FILE = "calib.json"

# --- Debug windows ---
SHOW = True

# --- Zone layout parameters (relative to top-view width/height) ---
# These ratios were taken from your original script structure:
#   left zone  : ~0%..40% of width
#   center     : ~40%..60% of width (aisle)
#   right zone : ~60%..99% of width
#
# You can adjust these 6 numbers if the green grid doesn't match your drawing.
X_L0 = 0.00
X_L1 = 0.40
X_C0 = 0.40
X_C1 = 0.60
X_R0 = 0.60
X_R1 = 0.99

# Vertical range used for parking + aisle (avoid borders if needed)
Y0 = 0.25
Y1 = 0.87

# Slots layout: 3 rows x 4 columns on each side (as you described)
SLOT_ROWS = 3
SLOT_COLS = 4

# Aisle segmentation (for localization like C0..C3)
AISLE_SEGMENTS = 4

# --- Presence mask method ---
# Background reference is the most stable for "parked" vehicles (no learning).
# Press 'b' to capture background.
USE_MOG2_FOR_PRESENCE = False  # keep False unless you *really* need it
MOG2_HISTORY = 400
MOG2_VARTHR = 16
MOG2_LEARNING_RATE = 0.001     # small => doesn't swallow immobile objects too fast

# --- Thresholds (tune) ---
# Presence (background difference) threshold in grayscale units (0..255)
PRESENCE_DIFF_THR = 20

# Occupancy thresholds
AISLE_OCC_THR = 0.08           # "something is in the aisle segment"
SLOT_OCC_THR  = 0.08           # "something is in a slot"

# Motion (frame-to-frame) thresholds
MOTION_DIFF_THR = 15           # pixel change threshold for motion mask
AISLE_MOTION_THR = 0.010       # below -> considered "not moving"
SLOT_MOTION_THR  = 0.008       # below -> considered "not moving"

# Timers (seconds)
STOP_SEC = 3.0                 # aisle must be "present & not moving" for this long => congestion
PARK_SEC = 6.0                 # slot must be "present & not moving" for this long => parked

# Anti-false-positive grace window after a new slot becomes occupied
MANEUVER_GRACE_SEC = 2.5

# Morphology (noise cleanup)
KERNEL_OPEN = 5                # size for morphological opening


# ======================
# Helpers: geometry / masks
# ======================

def rect_mask(w: int, h: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """
    Create a filled-rectangle mask in an image of size (h, w).
    x0,y0,x1,y1 are in absolute pixels (floats allowed).
    """
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    return m


def build_zones(out_w: int, out_h: int):
    """
    Build:
      - aisle segments masks: C0..C{AISLE_SEGMENTS-1}
      - slot masks: L_r_c and R_r_c with r in [0..SLOT_ROWS-1], c in [0..SLOT_COLS-1]
    Return:
      aisle_masks, slot_masks, boxes (for drawing), area dicts
    """
    # Convert relative ratios -> absolute pixels
    xL0, xL1 = X_L0 * out_w, X_L1 * out_w
    xC0, xC1 = X_C0 * out_w, X_C1 * out_w
    xR0, xR1 = X_R0 * out_w, X_R1 * out_w

    y0, y1 = Y0 * out_h, Y1 * out_h

    aisle_masks: Dict[str, np.ndarray] = {}
    slot_masks: Dict[str, np.ndarray] = {}
    boxes: Dict[str, Tuple[float, float, float, float]] = {}

    # ---- AISLE segments (split vertically into AISLE_SEGMENTS) ----
    for k in range(AISLE_SEGMENTS):
        yy0 = y0 + (y1 - y0) * k / AISLE_SEGMENTS
        yy1 = y0 + (y1 - y0) * (k + 1) / AISLE_SEGMENTS
        name = f"C{k}"
        aisle_masks[name] = rect_mask(out_w, out_h, xC0, yy0, xC1, yy1)
        boxes[name] = (xC0, yy0, xC1, yy1)

    # ---- Slots: Left side 3x4 and Right side 3x4 ----
    # We split each side rectangle into a grid.
    def add_slot_grid(side_prefix: str, x0_: float, x1_: float):
        for r in range(SLOT_ROWS):
            yy0 = y0 + (y1 - y0) * r / SLOT_ROWS
            yy1 = y0 + (y1 - y0) * (r + 1) / SLOT_ROWS
            for c in range(SLOT_COLS):
                xx0 = x0_ + (x1_ - x0_) * c / SLOT_COLS
                xx1 = x0_ + (x1_ - x0_) * (c + 1) / SLOT_COLS
                name = f"{side_prefix}_{r}_{c}"
                slot_masks[name] = rect_mask(out_w, out_h, xx0, yy0, xx1, yy1)
                boxes[name] = (xx0, yy0, xx1, yy1)

    add_slot_grid("L", xL0, xL1)
    add_slot_grid("R", xR0, xR1)

    # Compute mask areas once (number of non-zero pixels)
    aisle_area = {k: float(np.count_nonzero(m)) for k, m in aisle_masks.items()}
    slot_area  = {k: float(np.count_nonzero(m)) for k, m in slot_masks.items()}

    return aisle_masks, slot_masks, boxes, aisle_area, slot_area


# ======================
# Helpers: vision metrics
# ======================

def morph_open(bin_img: np.ndarray, ksize: int) -> np.ndarray:
    """Morphological opening to remove small noise blobs."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)


def compute_occupancy(presence_mask: np.ndarray,
                      masks: Dict[str, np.ndarray],
                      areas: Dict[str, float]) -> Dict[str, float]:
    """Occupancy ratio per zone."""
    out = {}
    for name, zone_mask in masks.items():
        zone = cv2.bitwise_and(presence_mask, presence_mask, mask=zone_mask)
        out[name] = float(np.count_nonzero(zone)) / max(1.0, areas[name])
    return out


def compute_motion(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Motion mask (binary) from absdiff(prev, curr).
    """
    diff = cv2.absdiff(gray, prev_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mm = cv2.threshold(diff, MOTION_DIFF_THR, 255, cv2.THRESH_BINARY)
    mm = morph_open(mm, KERNEL_OPEN)
    return mm


def presence_from_background(bg_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Presence mask = |gray - bg_gray| > threshold.
    Works well for parked objects because background is fixed.
    """
    diff = cv2.absdiff(gray, bg_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, pm = cv2.threshold(diff, PRESENCE_DIFF_THR, 255, cv2.THRESH_BINARY)
    pm = morph_open(pm, KERNEL_OPEN)
    return pm


# ======================
# State machines
# ======================

def reset_states(aisle_masks, slot_masks):
    """
    Initialize state dicts for aisle and slots.
    We keep timers to know how long the condition holds.
    """
    aisle_state = {}
    for k in aisle_masks.keys():
        aisle_state[k] = {
            "occ": 0.0,          # latest occupancy
            "motion": 0.0,       # latest motion
            "t0": None,          # when "stopped & present" started
            "congested": False,  # decision
        }

    slot_state = {}
    for k in slot_masks.keys():
        slot_state[k] = {
            "occ": 0.0,
            "motion": 0.0,
            "t0": None,          # when "parked condition" started
            "occupied": False,   # immediate occupancy flag
            "parked": False,     # decision
        }

    return aisle_state, slot_state


def update_slots(slot_state: Dict, slot_occ: Dict[str, float], slot_mot: Dict[str, float],
                 now: float) -> Tuple[Dict, bool]:
    """
    Update slot states.
    Returns (slot_state, new_occupancy_event_happened).
    """
    new_event = False

    for name, st in slot_state.items():
        occ = slot_occ[name]
        mot = slot_mot[name]

        # Keep latest metrics
        st["occ"] = occ
        st["motion"] = mot

        # Occupied right now?
        was_occupied = st["occupied"]
        st["occupied"] = (occ >= SLOT_OCC_THR)

        # Detect transition free -> occupied (parking event)
        if (not was_occupied) and st["occupied"]:
            new_event = True

        # Parked condition: present AND not moving
        parked_condition = st["occupied"] and (mot <= SLOT_MOTION_THR)

        if parked_condition:
            if st["t0"] is None:
                st["t0"] = now
            # If condition holds long enough -> parked
            st["parked"] = (now - st["t0"]) >= PARK_SEC
        else:
            st["t0"] = None
            st["parked"] = False

    return slot_state, new_event


def update_aisle(aisle_state: Dict, aisle_occ: Dict[str, float], aisle_mot: Dict[str, float],
                 now: float, grace_active: bool) -> Dict:
    """
    Update aisle congestion state.
    If grace_active is True, we suppress congestion to avoid "parking maneuver" false positives.
    """
    for name, st in aisle_state.items():
        occ = aisle_occ[name]
        mot = aisle_mot[name]

        st["occ"] = occ
        st["motion"] = mot

        stopped_and_present = (occ >= AISLE_OCC_THR) and (mot <= AISLE_MOTION_THR)

        # If we are in maneuver grace period: never declare congestion
        if grace_active:
            st["t0"] = None
            st["congested"] = False
            continue

        if stopped_and_present:
            if st["t0"] is None:
                st["t0"] = now
            st["congested"] = (now - st["t0"]) >= STOP_SEC
        else:
            st["t0"] = None
            st["congested"] = False

    return aisle_state


# ======================
# Drawing helpers
# ======================

def draw_boxes(img: np.ndarray, boxes: Dict[str, Tuple[float, float, float, float]]):
    """Draw all zone rectangles."""
    for name, (x0, y0, x1, y1) in boxes.items():
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)


def put_text(img: np.ndarray, x: int, y: int, s: str, scale: float = 0.55, thick: int = 2):
    """Convenience for text overlay."""
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick, cv2.LINE_AA)


# ======================
# Main
# ======================

def main():
    # -------- Load calibration (homography + output size) --------
    with open(CALIB_FILE, "r", encoding="utf-8") as f:
        calib = json.load(f)

    # Homography matrix (3x3) mapping camera image -> top view
    M = np.array(calib["M"], dtype=np.float32)

    # Output top-view size
    out_w = int(calib.get("out_w", 640))
    out_h = int(calib.get("out_h", 480))

    # Build zone masks
    aisle_masks, slot_masks, boxes, aisle_area, slot_area = build_zones(out_w, out_h)

    # Init states
    aisle_state, slot_state = reset_states(aisle_masks, slot_masks)

    # For motion computation
    prev_gray = None

    # Background reference (captured by 'b')
    bg_gray = None

    # Optional MOG2
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=MOG2_VARTHR) \
             if USE_MOG2_FOR_PRESENCE else None

    # Parking maneuver grace timer
    last_parking_event = -1e9  # "very old"

    # -------- Open camera stream --------
    cap = cv2.VideoCapture(RTSP_URL if USE_RTSP else CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source. Check RTSP_URL / CAM_INDEX.")

    print("[INFO] Press 'b' to capture background (recommended).")
    print("[INFO] Press 'r' to reset states, 'q' or ESC to quit.")


    # -------- Main loop --------
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # If stream fails, wait a bit and try again
            time.sleep(0.05)
            continue

        # Warp to top-view (bird-eye)
        top = cv2.warpPerspective(frame, M, (out_w, out_h))

        # Convert to grayscale for presence/motion computations
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

        # Initialize prev_gray on first frame
        if prev_gray is None:
            prev_gray = gray.copy()

        # Compute motion mask (frame-to-frame)
        motion_mask = compute_motion(prev_gray, gray)

        # Compute presence mask
        if USE_MOG2_FOR_PRESENCE:
            # Foreground via MOG2
            fg = bg_sub.apply(top, learningRate=MOG2_LEARNING_RATE)
            fg = morph_open(fg, KERNEL_OPEN)
            presence_mask = fg
        else:
            # Presence via background difference (recommended for parked detection)
            if bg_gray is None:
                # If background not captured yet, we can't compute a reliable presence mask
                presence_mask = np.zeros((out_h, out_w), dtype=np.uint8)
            else:
                presence_mask = presence_from_background(bg_gray, gray)

        # Update prev frame
        prev_gray = gray

        # Compute occupancy & motion ratios per zone
        aisle_occ = compute_occupancy(presence_mask, aisle_masks, aisle_area)
        slot_occ  = compute_occupancy(presence_mask, slot_masks, slot_area)

        aisle_mot = compute_occupancy(motion_mask, aisle_masks, aisle_area)
        slot_mot  = compute_occupancy(motion_mask, slot_masks, slot_area)

        # Update states + timers
        now = time.time()

        slot_state, new_parking_event = update_slots(slot_state, slot_occ, slot_mot, now)
        if new_parking_event:
            last_parking_event = now

        grace_active = (now - last_parking_event) <= MANEUVER_GRACE_SEC
        aisle_state = update_aisle(aisle_state, aisle_occ, aisle_mot, now, grace_active)

        # -------- Overlay / Debug --------
        overlay = top.copy()
        draw_boxes(overlay, boxes)

        # Display summary line
        n_occ_slots = sum(1 for st in slot_state.values() if st["occupied"])
        n_parked    = sum(1 for st in slot_state.values() if st["parked"])
        n_cong      = sum(1 for st in aisle_state.values() if st["congested"])

        put_text(overlay, 10, 20,
                 f"BG={'YES' if bg_gray is not None else 'NO'}  "
                 f"Slots occ={n_occ_slots} parked={n_parked}  "
                 f"Aisle congested segs={n_cong}  "
                 f"Grace={'ON' if grace_active else 'OFF'}",
                 scale=0.6)

        # Show each aisle segment metrics at top of its rectangle
        for name, (x0, y0, x1, y1) in boxes.items():
            if not name.startswith("C"):
                continue
            st = aisle_state[name]
            tag = "CONG" if st["congested"] else "ok"
            put_text(overlay, int(x0) + 5, int(y0) + 18,
                     f"{name} occ={st['occ']*100:4.1f}% mot={st['motion']*100:4.1f}% {tag}",
                     scale=0.55)

        # Optionally, show slots with a tiny status (occupied/parked)
        # We keep this light to avoid clutter.
        for name, (x0, y0, x1, y1) in boxes.items():
            if name.startswith("L_") or name.startswith("R_"):
                st = slot_state[name]
                if st["occupied"]:
                    label = "P" if st["parked"] else "O"
                    put_text(overlay, int(x0) + 4, int(y0) + 16, label, scale=0.5)

        if SHOW:
            cv2.imshow("top (overlay)", overlay)
            cv2.imshow("presence (bgdiff or fg)", presence_mask)
            cv2.imshow("motion (absdiff)", motion_mask)

        # -------- Keyboard controls --------
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break

        if key == ord('b'):
            # Capture background reference from current frame
            bg_gray = gray.copy()
            print("[INFO] Background captured.")

        if key == ord('r'):
            # Reset states/timers
            aisle_state, slot_state = reset_states(aisle_masks, slot_masks)
            last_parking_event = -1e9
            print("[INFO] States reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
