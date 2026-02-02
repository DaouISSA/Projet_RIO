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
import requests
from requests.exceptions import RequestException
import threading  #Post HTTP Threads
import queue
import re




# ======================
# CONFIG (edit here)
# ======================

# --- Camera source ---
USE_RTSP = True
RTSP_URL = "rtsp://137.194.154.170:8554/cam"   # your phone stream
CAM_INDEX = 0                                  # used if USE_RTSP=False

# --- Threads Config --- #
POST_QUEUE_SIZE = 1  # 1 = toujours garder le dernier état

post_queue = queue.Queue(maxsize=POST_QUEUE_SIZE)
stop_event = threading.Event()

# --- Server Config --- 
SERVER_URL = "http://137.194.155.118:8000/"
SEND_EVERY_SEC = 2.5
HTTP_TIMEOUT = 1.0

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
Y0 = 0.23
Y1 = 0.95

# Slots layout: 3 rows x 4 columns on each side (as you described)
SLOT_ROWS = 4
SLOT_COLS = 4
AISLE_SEGMENTS = 4

# Aisle segmentation (for localization like C0..C3)
AISLE_SEGMENTS = 4

# --- Presence mask method ---
# Background reference is the most stable for "parked" vehicles (no learning).
# Press 'b' to capture background.
USE_MOG2_FOR_PRESENCE = False  # improved robustness to lighting changes
MOG2_HISTORY = 400
MOG2_VARTHR = 16
MOG2_LEARNING_RATE = 0.001     # small => doesn't swallow immobile objects too fast

# --- Thresholds (tune) ---
# Presence (background difference) threshold in grayscale units (0..255)
PRESENCE_DIFF_THR = 40  # raise to reduce sensor noise / grain

# Occupancy thresholds
AISLE_OCC_THR = 0.15           # higher threshold to avoid false positives
SLOT_OCC_THR = 0.40           # higher threshold to avoid false positives

# --- Mini-allee Occupancy thresholds + Ratio ---
SLOT_APRON_RATIO = 0.35   # 20% de la hauteur de la case = mini-allée (0.15..0.30)
APRON_OCC_THR = 0.10      # seuil présence pour dire "quelque chose bloque devant"
APRON_MOTION_THR = 0.015  # seuil motion (comme l'allée)
APRON_STOP_SEC = 4.0      # durée d'arrêt pour déclarer congestion sur mini-allée

# Motion (frame-to-frame) thresholds
MOTION_DIFF_THR = 15           # pixel change threshold for motion mask
AISLE_MOTION_THR = 0.010       # below -> considered "not moving"
SLOT_MOTION_THR  = 0.008       # below -> considered "not moving"

# --- Anti-ombres (presence via HSV) ---
USE_HSV_PRESENCE = True
HS_DIFF_THR = 35      # seuil sur diff(H,S), à ajuster 15..40

USE_V_IN_HSV = True
V_DIFF_THR = 20

# Timers (seconds)
STOP_SEC = 4.0                 # aisle must be "present & not moving" for this long => congestion
PARK_SEC = 4.0                 # slot must be "present & not moving" for this long => parked

OCCUPY_DELAY_SEC = 3.0   # délai avant état "occupied"

# Anti-false-positive grace window after a new slot becomes occupied
MANEUVER_GRACE_SEC = 2.5

# Morphology (noise cleanup)
KERNEL_OPEN = 7                # stronger morphology to remove small blobs

BULK_ENDPOINT = "/update/bulk"


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
    apron_masks: Dict[str, np.ndarray] = {}
    boxes: Dict[str, Tuple[float, float, float, float]] = {}

    # ---- AISLE segments (split vertically into AISLE_SEGMENTS) ----
    for k in range(AISLE_SEGMENTS):
        yy0 = y0 + (y1 - y0) * k / AISLE_SEGMENTS
        yy1 = y0 + (y1 - y0) * (k + 1) / AISLE_SEGMENTS
        name = f"PATH_CENTRAL_{k}"
        aisle_masks[name] = rect_mask(out_w, out_h, xC0, yy0, xC1, yy1)
        boxes[name] = (xC0, yy0, xC1, yy1)


    # ---- Slots: Left side 3x4 and Right side 3x4 ----
    # We split each side rectangle into a grid.
    def add_slot_grid(side_prefix: str, x0_: float, x1_: float):
        cell_w = (x1_ - x0_) / SLOT_COLS
        cell_h = (y1 - y0) / SLOT_ROWS

        apron_h = SLOT_APRON_RATIO * cell_h

        for r in range(1, SLOT_ROWS + 1):
            # limites verticales de la cellule r
            yy0 = y0 + (r - 1) * cell_h
            yy1 = y0 + r * cell_h

            # split vertical : [slot_y0..slot_y1] + [ap_y0..ap_y1]
            slot_y0 = yy0
            slot_y1 = yy1 - apron_h  # partie haute
            ap_y0   = yy1 - apron_h  # bande basse
            ap_y1   = yy1

            for c in range(1, SLOT_COLS + 1):
                # limites horizontales de la cellule c
                xx0 = x0_ + (c - 1) * cell_w
                xx1 = x0_+ c * cell_w

                # ---- SLOT (place) ----
                sname = f"{side_prefix}_{r}_{c}"
                slot_masks[sname] = rect_mask(out_w, out_h, xx0, slot_y0, xx1, slot_y1)
                boxes[sname] = (xx0, slot_y0, xx1, slot_y1)

                # ---- APRON (mini-allée sous la place) ----
                aname = f"A_{side_prefix}_{r}_{c}"
                apron_masks[aname] = rect_mask(out_w, out_h, xx0, ap_y0, xx1, ap_y1)
                boxes[aname] = (xx0, ap_y0, xx1, ap_y1)

    add_slot_grid("L", xL0, xL1)
    add_slot_grid("R", xR0, xR1)

    # Compute mask areas once (number of non-zero pixels)
    aisle_area = {k: float(np.count_nonzero(m)) for k, m in aisle_masks.items()}
    slot_area  = {k: float(np.count_nonzero(m)) for k, m in slot_masks.items()}
    apron_area = {k: float(np.count_nonzero(m)) for k, m in apron_masks.items()}


    return aisle_masks, slot_masks, apron_masks, boxes, aisle_area, slot_area, apron_area

APRON_RE_A = re.compile(r"^A_(L|R)_(\d+)_(\d+)$")
APRON_RE_PATH = re.compile(r"^PATH_ROW_(\d+)_(L|R)_SEG_(\d+)$")

def server_id_for_apron(apron_name: str) -> str:
    """
    Mapping IDs caméra -> IDs serveur pour les aprons.

    Caméra (format actuel):
      A_L_1_1  ... A_L_4_4
      A_R_1_1  ... A_R_4_4

    Serveur attendu:
      APRON_L_0_0 ... APRON_L_3_3
      APRON_R_0_0 ... APRON_R_3_3

    IMPORTANT: côté gauche, l'ordre des colonnes est inversé.
      A_L_1_4 -> APRON_L_0_0
    """
    m = APRON_RE_A.match(apron_name)
    if m:
        side = m.group(1)              # L/R
        row_cam = int(m.group(2))      # 1..4
        col_cam = int(m.group(3))      # 1..4

        row_srv = row_cam - 1

        if side == "L":
            col_srv = SLOT_COLS - col_cam
        else:
            col_srv = col_cam - 1

        return f"APRON_{side}_{row_srv}_{col_srv}"

    # (Optionnel) Si un jour tu utilises un autre format d'ID caméra
    m = APRON_RE_PATH.match(apron_name)
    if m:
        row_cam = int(m.group(1))      # 1..4
        side = m.group(2)              # L/R
        seg_cam = int(m.group(3))      # 0..3

        row_srv = row_cam - 1
        if side == "L":
            col_srv = (SLOT_COLS - 1) - seg_cam
        else:
            col_srv = seg_cam

        return f"APRON_{side}_{row_srv}_{col_srv}"

    raise ValueError(f"Bad apron id format: {apron_name}")


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

def presence_from_background_hsv(bg_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    bg_hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    cur_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)

    dh = cv2.absdiff(cur_hsv[:, :, 0], bg_hsv[:, :, 0])
    ds = cv2.absdiff(cur_hsv[:, :, 1], bg_hsv[:, :, 1])
    dv = cv2.absdiff(cur_hsv[:, :, 2], bg_hsv[:, :, 2])

    # flou léger pour réduire le grain
    dh = cv2.GaussianBlur(dh, (5, 5), 0)
    ds = cv2.GaussianBlur(ds, (5, 5), 0)
    dv = cv2.GaussianBlur(dv, (5, 5), 0)

    # 1) masque HS (anti-ombres)
    hs = cv2.addWeighted(dh, 0.7, ds, 0.3, 0)
    _, pm_hs = cv2.threshold(hs, HS_DIFF_THR, 255, cv2.THRESH_BINARY)

    if USE_V_IN_HSV:
        # 2) masque V (capture main/objets même si H/S change peu)
        _, pm_v = cv2.threshold(dv, V_DIFF_THR, 255, cv2.THRESH_BINARY)
        pm = cv2.bitwise_or(pm_hs, pm_v)
    else:
        pm = pm_hs

    pm = morph_open(pm, KERNEL_OPEN)
    return pm

# ======================
# State machines
# ======================

def reset_states(aisle_masks, slot_masks, apron_masks):
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
            "t0": None,            # parked timer
            "occ_t0": None,        # <-- NOUVEAU : timer occupation
            "occupied": False,
            "parked": False,
        }

    apron_state = {}
    for k in apron_masks.keys():
        apron_state[k] = {
            "occ": 0.0,
            "motion": 0.0,
            "t0": None,
            "congested": False,
        }
    return aisle_state, slot_state, apron_state

def update_slots(slot_state, slot_occ, slot_mot, now):
    new_event = False

    for name, st in slot_state.items():
        occ = slot_occ[name]
        mot = slot_mot[name]

        st["occ"] = occ
        st["motion"] = mot

        # --- OCCUPATION AVEC DÉLAI ---
        if occ >= SLOT_OCC_THR:
            if st["occ_t0"] is None:
                st["occ_t0"] = now
            if (now - st["occ_t0"]) >= OCCUPY_DELAY_SEC:
                if not st["occupied"]:
                    new_event = True   # événement "nouvelle occupation"
                st["occupied"] = True
        else:
            st["occ_t0"] = None
            st["occupied"] = False
            st["parked"] = False
            st["t0"] = None
            continue

        # --- PARKED (après occupied) ---
        parked_condition = st["occupied"] and (mot <= SLOT_MOTION_THR)

        if parked_condition:
            if st["t0"] is None:
                st["t0"] = now
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

def update_aprons(apron_state: Dict, apron_occ: Dict[str, float], apron_mot: Dict[str, float],
                  now: float) -> Dict:
    """
    Congestion locale devant les places (mini-allées).
    """
    for name, st in apron_state.items():
        occ = apron_occ[name]
        mot = apron_mot[name]
        st["occ"] = occ
        st["motion"] = mot

        stopped_and_present = (occ >= APRON_OCC_THR) and (mot <= APRON_MOTION_THR)

        if stopped_and_present:
            if st["t0"] is None:
                st["t0"] = now
            st["congested"] = (now - st["t0"]) >= APRON_STOP_SEC
        else:
            st["t0"] = None
            st["congested"] = False

    return apron_state

# ======================
# Congestion closure rules (one-way assumptions)
# ======================

APRON_RE = re.compile(r"^A_(L|R)_(\d+)_(\d+)$")
AISLE_RE = re.compile(r"^PATH_CENTRAL_(\d+)$")

def apply_congestion_closure(aisle_state: Dict, apron_state: Dict) -> None:
    """Apply deterministic congestion propagation rules.

    Rules (one-way):
      - Left aprons: if A_L_r_c congested, then A_L_r_1..c congested.
      - Right aprons: if A_R_r_c congested, then A_R_r_c..SLOT_COLS congested.
      - Central aisle: if PATH_CENTRAL_k congested, then PATH_CENTRAL_{k+1..end} congested (below).
    This overrides occupancy/motion: even if occ becomes low, closure keeps them congested.
    """

    # --- Aprons closure ---
    # Gather current congested positions
    left_max_c_by_row = {}   # row -> max c that is congested on left
    right_min_c_by_row = {}  # row -> min c that is congested on right

    for name, st in apron_state.items():
        if not st.get("congested"):
            continue
        m = APRON_RE.match(name)
        if not m:
            continue
        side, r_s, c_s = m.group(1), m.group(2), m.group(3)
        r = int(r_s)
        c = int(c_s)
        if side == "L":
            # if any apron at column c is congested, everything 1..c is congested
            left_max_c_by_row[r] = max(left_max_c_by_row.get(r, 0), c)
        else:
            # if any apron at column c is congested, everything c..SLOT_COLS is congested
            right_min_c_by_row[r] = min(right_min_c_by_row.get(r, SLOT_COLS + 1), c)

    # Apply closures
    for name, st in apron_state.items():
        m = APRON_RE.match(name)
        if not m:
            continue
        side, r_s, c_s = m.group(1), m.group(2), m.group(3)
        r = int(r_s)
        c = int(c_s)

        if side == "L":
            max_c = left_max_c_by_row.get(r, 0)
            if max_c and c <= max_c:
                st["congested"] = True
        else:
            min_c = right_min_c_by_row.get(r, SLOT_COLS + 1)
            if min_c <= SLOT_COLS and c >= min_c:
                st["congested"] = True

    # --- Central aisle closure (below is also blocked) ---
    congested_indices = []
    for name, st in aisle_state.items():
        if not st.get("congested"):
            continue
        m = AISLE_RE.match(name)
        if m:
            congested_indices.append(int(m.group(1)))

    if congested_indices:
        k0 = min(congested_indices)  # first blocked segment from the top
        for k in range(k0, AISLE_SEGMENTS):
            key = f"PATH_CENTRAL_{k}"
            if key in aisle_state:
                aisle_state[key]["congested"] = True

# ======================
# Payloads for Server
# ======================

def build_zones_payload(slot_state, aisle_state, apron_state, now):
    zones = []

    # --- SLOTS (places) ---
    for slot_name, st in slot_state.items():
        sid = server_id_for_slot(slot_name)

        if not st["occupied"]:
            state = "free"
        elif st["parked"]:
            state = "parked"
        else:
            state = "occupied"

        zones.append({
            "id": sid,
            "type": "slot",
            "state": state
        })

    # --- AISLE (congestion) ---
    for aisle_name, st in aisle_state.items():
        sid = server_id_for_aisle(aisle_name)
        zones.append({
            "id": sid,
            "type": "aisle",
            "state": "congested" if st["congested"] else "free"
        })
        # --- APRONS (mini-allées gauche/droite) ---
    for apron_name, st in apron_state.items():
        sid = server_id_for_apron(apron_name)
        zones.append({
            "id": sid,
            "type": "apron",
            "state": "congested" if st["congested"] else "free"
        })


    # --- SUMMARY ---
    slots_total = len(slot_state)
    slots_occupied = sum(1 for st in slot_state.values() if st["occupied"])
    slots_parked = sum(1 for st in slot_state.values() if st["parked"])
    aisle_cong = sum(1 for st in aisle_state.values() if st["congested"])
    apron_cong = sum(1 for st in apron_state.values() if st["congested"])

    return {
        "timestamp": now,
        "zones": zones,
        "summary": {
            "slots_total": slots_total,
            "slots_occupied": slots_occupied,
            "slots_parked": slots_parked,
            "aisle_congested": aisle_cong,
            "apron_congested": apron_cong
        }
    }


def post_worker(server_url, q, stop_evt):         #post_payload
    print("[POST-THREAD] started")  # <-- AJOUT
    while not stop_evt.is_set():
        try:
            payload = q.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            url = server_url.rstrip("/") + BULK_ENDPOINT
            print("\n===== PAYLOAD SENT TO SERVER =====")
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            print("=================================\n")
            r = requests.post(url, json=payload, timeout=0.5)
            print(f"[POST-THREAD] sent zones={len(payload.get('zones', []))} code={r.status_code}")
        except RequestException as e:
            print(f"[POST-THREAD] ERROR: {e}")  # <-- AJOUT (au moins pendant debug)
        finally:
            q.task_done()

# ======================
# Drawing helpers
# ======================

def draw_boxes(img, boxes, aisle_state, apron_state):
    for name, (x0, y0, x1, y1) in boxes.items():

        # --- Détection du type de zone ---
        if name.startswith("PATH_CENTRAL_") or name.startswith("C"):
            congested = aisle_state.get(name, {}).get("congested", False)
        elif name.startswith("A_"):
            congested = apron_state[name]["congested"]
        else:
            congested = False  # slots normaux

        # --- Couleur ---
        if congested:
            color = (0, 0, 255)      # ROUGE = congestion
            thickness = 3
        else:
            color = (0, 255, 0)      # VERT = OK
            thickness = 2

        cv2.rectangle(img,(int(x0), int(y0)),(int(x1), int(y1)),color,thickness)

def put_text(img: np.ndarray, x: int, y: int, s: str, scale: float = 0.55, thick: int = 2):
    """Convenience for text overlay."""
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thick, cv2.LINE_AA)


def put_text_mini(img: np.ndarray, x: int, y: int, s: str):
    """Smaller text for tiny cells."""
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 0), 1, cv2.LINE_AA)

def box_center(box):
    """Retourne le centre (cx, cy) d'une box (x0,y0,x1,y1) en pixels top-view."""
    x0, y0, x1, y1 = box
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

def parse_slot_name(name: str):
    """
    name format: 'L_r_c' or 'R_r_c'
    Retourne (side, r, c) où side='L' ou 'R'.
    """
    side, r, c = name.split("_")
    return side, int(r), int(c)

def server_id_for_slot(slot_name: str) -> str:
    # slot_name = "L_1_1" (cam)
    side, r, c = parse_slot_name(slot_name)  # r,c = 1..SLOT_ROWS / 1..SLOT_COLS

    # Serveur attend un index 0-based (0..3).
    row_srv = r - 1

    # IMPORTANT: côté gauche, l'ordre des colonnes est inversé.
    # Exemple demandé :
    #   Cam L_1_4 -> Srv L_0_0
    # Donc col_srv = (SLOT_COLS - c)
    if side == "L":
        col_srv = SLOT_COLS - c
    else:
        col_srv = c - 1

    return f"{side}_{row_srv}_{col_srv}"

def server_id_for_aisle(aisle_name: str) -> str:
    # déjà au bon format: PATH_CENTRAL_0..3
    return aisle_name

# ======================
# Main
# ======================

def main():
    # -------- Load calibration (homography + output size) --------
    with open(CALIB_FILE, "r", encoding="utf-8") as f:
        calib = json.load(f)

    # Homography matrix (3x3) mapping camera image -> top view
    M = np.array(calib["M"], dtype=np.float32)

    bg_bgr = None # l’image top-view couleur, pas seulement le gris.

    # Output top-view size
    out_w = int(calib.get("out_w", 640))
    out_h = int(calib.get("out_h", 480))

    # Build zone masks
    aisle_masks, slot_masks, apron_masks, boxes, aisle_area, slot_area, apron_area = build_zones(out_w, out_h)

    # Init states
    aisle_state, slot_state, apron_state = reset_states(aisle_masks, slot_masks, apron_masks)

    # For motion computation
    prev_gray = None

    # Background reference (captured by 'b')
    bg_gray = None

    # Optional MOG2
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=MOG2_VARTHR) \
             if USE_MOG2_FOR_PRESENCE else None

    # Parking maneuver grace timer
    last_parking_event = -1e9  # "very old"

    # timer to send requests to the server every seconds
    last_send = 0.0

    # -------- Open camera stream --------
    cap = cv2.VideoCapture(RTSP_URL if USE_RTSP else CAM_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video source. Check RTSP_URL / CAM_INDEX.")

    print("[INFO] Press 'b' to capture background (recommended).")
    print("[INFO] Press 'r' to reset states, 'q' or ESC to quit.")

    # -------- Thread POST created --------
    post_thread = threading.Thread(
        target=post_worker,
        args=(SERVER_URL, post_queue, stop_event),
        daemon=True
    )
    post_thread.start()


    # -------- Main loop --------
    
    # --- AJOUT: LISSAGE (moyenne glissante) pour stabiliser les %occ/%motion ---
    smoothed_occ = {}
    smoothed_mot = {}
    ALPHA = 0.10  # 0.10 = très stable ; 0.20 = plus réactif

    def smooth_dict(raw: dict, mem: dict, alpha: float):
        for k, v in raw.items():
            mem[k] = v if k not in mem else (alpha * v + (1 - alpha) * mem[k])
        return mem.copy()

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
                if bg_bgr is None:
                    presence_mask = np.zeros((out_h, out_w), dtype=np.uint8)
                else:
                    if USE_HSV_PRESENCE:
                        presence_mask = presence_from_background_hsv(bg_bgr, top)
                    else:
                        presence_mask = presence_from_background(bg_gray, gray)
    
            # Update prev frame
            prev_gray = gray
    
            # Compute occupancy & motion ratios per zone
            aisle_occ = compute_occupancy(presence_mask, aisle_masks, aisle_area)
            slot_occ  = compute_occupancy(presence_mask, slot_masks, slot_area)
            apron_occ = compute_occupancy(presence_mask, apron_masks, apron_area)
    
            aisle_mot = compute_occupancy(motion_mask, aisle_masks, aisle_area)
            slot_mot  = compute_occupancy(motion_mask, slot_masks, slot_area)
            apron_mot = compute_occupancy(motion_mask,  apron_masks, apron_area)
    
    
            # --- AJOUT: appliquer le lissage (évite le clignotement dû au bruit caméra) ---
            slot_occ  = smooth_dict(slot_occ,  smoothed_occ, ALPHA)
            aisle_occ = smooth_dict(aisle_occ, smoothed_occ, ALPHA)
            apron_occ = smooth_dict(apron_occ, smoothed_occ, ALPHA)
    
            slot_mot  = smooth_dict(slot_mot,  smoothed_mot, ALPHA)
            aisle_mot = smooth_dict(aisle_mot, smoothed_mot, ALPHA)
            apron_mot = smooth_dict(apron_mot, smoothed_mot, ALPHA)
    
    
            # Update states + timers
            now = time.time()
    
            slot_state, new_parking_event = update_slots(slot_state, slot_occ, slot_mot, now)
            if new_parking_event:
                last_parking_event = now
    
            grace_active = (now - last_parking_event) <= MANEUVER_GRACE_SEC
            aisle_state = update_aisle(aisle_state, aisle_occ, aisle_mot, now, grace_active)
            apron_state = update_aprons(apron_state, apron_occ, apron_mot, now)

            # --- APPLY ONE-WAY CONGESTION CLOSURE RULES ---
            apply_congestion_closure(aisle_state, apron_state)
    
            # -------- Envoi serveur toutes les SEND_EVERY_SEC secondes --------
            if (now - last_send) >= SEND_EVERY_SEC:
                payload = build_zones_payload(slot_state, aisle_state, apron_state, now)
    
                # push non bloquant : on garde le dernier état seulement
                try:
                    if post_queue.full():
                        post_queue.get_nowait()
                    post_queue.put_nowait(payload)
                except queue.Empty:
                    pass
    
                last_send = now
            # --------------------------------------------------------------------
    
            # -------- Overlay / Debug --------
            overlay = top.copy()
            draw_boxes(overlay, boxes, aisle_state, apron_state)
    
            # Display summary line
            n_occ_slots = sum(1 for st in slot_state.values() if st["occupied"])
            n_parked    = sum(1 for st in slot_state.values() if st["parked"])
            n_cong      = sum(1 for st in aisle_state.values() if st["congested"])
            n_apron_cong = sum(1 for st in apron_state.values() if st["congested"])
    
            put_text(overlay, 10, 20,
                     f"BG={'YES' if bg_gray is not None else 'NO'}  "
                     f"Slots occ={n_occ_slots} parked={n_parked}  "
                     f"Aisle congested segs={n_cong}  Apron cong={n_apron_cong}   "
                     f"Grace={'ON' if grace_active else 'OFF'}",
                     scale=0.6)
    
            # Show each aisle segment metrics at top of its rectangle
            for name, (x0, y0, x1, y1) in boxes.items():
                if not (name.startswith("PATH_CENTRAL_") or name.startswith("C")):
                    continue
                st = aisle_state[name]
                tag = "CONG" if st["congested"] else "ok"
                put_text(overlay, int(x0) + 5, int(y0) + 18,
                         f"{name} occ={st['occ']*100:4.1f}% mot={st['motion']*100:4.1f}% {tag}",
                         scale=0.55)
                
            # --- AJOUT: afficher les mini-allées (aprons) ---
            for name, (x0, y0, x1, y1) in boxes.items():
                if not name.startswith("A_"):
                    continue
                st = apron_state[name]
                tag = "CONG" if st["congested"] else "ok"
                put_text_mini(overlay, int(x0) + 2, int(y0) + 12, name)
                put_text_mini(overlay, int(x0) + 2, int(y0) + 24, f"{st['occ']*100:3.0f}% {tag}")
    
            # --- AJOUT: afficher ID + occ + état (tout petit) pour chaque place ---
            for name, (x0, y0, x1, y1) in boxes.items():
                if name.startswith("L_") or name.startswith("R_"):
                    st = slot_state[name]
                    put_text_mini(overlay, int(x0) + 2, int(y0) + 12, name)
    
                    if not st["occupied"]:
                        # F = free
                        put_text_mini(overlay, int(x0) + 2, int(y0) + 24, f"{st['occ']*100:3.0f}% F")
                    else:
                        # O = occupied, P = parked
                        lab = "P" if st["parked"] else "O"
                        put_text_mini(overlay, int(x0) + 2, int(y0) + 24, f"{st['occ']*100:3.0f}% {lab}")
    
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
                bg_gray = gray.copy()       # tu peux garder si tu veux debug
                bg_bgr = top.copy()         # <-- AJOUT: background couleur pour HSV
                print("[INFO] Background captured.")
    
            if key == ord('r'):
                # Reset states/timers
                aisle_state, slot_state, apron_state = reset_states(aisle_masks, slot_masks, apron_masks)
                last_parking_event = -1e9
                print("[INFO] States reset.")
    
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
