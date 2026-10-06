import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque, Counter

# ============================================================
#  TUNABLE CONFIG  (adjust these if recognition is too strict/loose)
# ============================================================
MODEL_PATH = 'hand_landmarker.task'

FINGER_STRAIGHT_ANGLE = 160.0   # deg at PIP for a "straight" finger
FINGER_REACH_FACTOR   = 1.15    # tip must be this much further from wrist than PIP
FINGER_TIP_MCP_MIN    = 0.60    # tip-to-MCP distance / hand_size for an extended finger

THUMB_STRAIGHT_ANGLE  = 150.0   # deg at thumb MCP for a straight thumb
THUMB_MIN_MCP_DIST    = 0.50    # thumb tip distance from index/pinky MCP to be "out"
PINCH_MAX_DIST        = 0.30    # thumb tip to index tip = touching (for 0)

SMOOTH_N              = 8       # frames used for majority vote
SHOW_DEBUG            = True    # prints raw feature values on screen

# ============================================================
#  LANDMARK INDICES
# ============================================================
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

FINGER_JOINTS = {
    'index':  (INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
    'middle': (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    'ring':   (RING_MCP,   RING_PIP,   RING_TIP),
    'pinky':  (PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
}

# bones to draw: (from, to)
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm edge
]

# ============================================================
#  MODEL SETUP
# ============================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8,
)
detector = vision.HandLandmarker.create_from_options(options)

# ============================================================
#  GEOMETRY HELPERS
# ============================================================
def to_pixels(landmarks, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

def dist(a, b):
    return float(np.linalg.norm(a - b))

def angle_at(a, b, c):
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def analyse_hand(pts):
    hand_size = dist(pts[WRIST], pts[MIDDLE_MCP]) + 1e-6

    fingers = {}
    for name, (mcp, pip, tip) in FINGER_JOINTS.items():
        straightness = angle_at(pts[mcp], pts[pip], pts[tip])
        reach = dist(pts[tip], pts[WRIST]) / (dist(pts[pip], pts[WRIST]) + 1e-6)
        tip_mcp = dist(pts[tip], pts[mcp]) / hand_size
        extended = (straightness > FINGER_STRAIGHT_ANGLE and
                    reach > FINGER_REACH_FACTOR and
                    tip_mcp > FINGER_TIP_MCP_MIN)
        fingers[name] = {
            'angle': straightness,
            'reach': reach,
            'tip_mcp': tip_mcp,
            'extended': extended,
        }

    # Thumb: straightness at MCP + distance from index/pinky MCP
    thumb_angle = angle_at(pts[THUMB_CMC], pts[THUMB_MCP], pts[THUMB_TIP])
    d_index_mcp = dist(pts[THUMB_TIP], pts[INDEX_MCP]) / hand_size
    d_pinky_mcp = dist(pts[THUMB_TIP], pts[PINKY_MCP]) / hand_size
    thumb_straight = thumb_angle > THUMB_STRAIGHT_ANGLE
    thumb_out = thumb_straight and d_index_mcp > THUMB_MIN_MCP_DIST and d_pinky_mcp > THUMB_MIN_MCP_DIST

    pinch = {
        'index':  dist(pts[THUMB_TIP], pts[INDEX_TIP])  / hand_size,
        'middle': dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / hand_size,
        'ring':   dist(pts[THUMB_TIP], pts[RING_TIP])   / hand_size,
        'pinky':  dist(pts[THUMB_TIP], pts[PINKY_TIP])  / hand_size,
    }

    return {
        'size': hand_size,
        'fingers': fingers,
        'thumb': {
            'angle': thumb_angle,
            'straight': thumb_straight,
            'out': thumb_out,
            'd_index_mcp': d_index_mcp,
            'd_pinky_mcp': d_pinky_mcp,
        },
        'pinch': pinch,
    }

# ============================================================
#  CLASSIFIER  (SASL 0-10 rules based on your image)
# ============================================================
def classify_number(f):
    """Return SASL number (0-10) or None."""
    e = {k: v['extended'] for k, v in f['fingers'].items()}
    idx, mid, rng, pky = e['index'], e['middle'], e['ring'], e['pinky']
    thumb_out = f['thumb']['out']
    thumb_straight = f['thumb']['straight']
    p = f['pinch']

    # ---- 1: Only index extended, thumb not out ----
    if idx and not mid and not rng and not pky and not thumb_out:
        return 1

    # ---- 2: Index & middle extended, thumb not out ----
    if idx and mid and not rng and not pky and not thumb_out:
        return 2

    # ---- 3: Index & middle extended, thumb out (straight) ----
    if idx and mid and not rng and not pky and thumb_out:
        return 3

    # ---- 4: Four fingers extended, thumb not out ----
    if idx and mid and rng and pky and not thumb_out:
        return 4

    # ---- 5: All five fingers extended (thumb out) ----
    if idx and mid and rng and pky and thumb_out:
        return 5

    # ---- 0: Thumb & index pinch, other three extended, thumb curved ----
    if (p['index'] < PINCH_MAX_DIST and mid and rng and pky and not thumb_straight):
        return 0

    # ---- 6: Thumb & pinky extended, others curled ----
    if pky and thumb_out and not idx and not mid and not rng:
        return 6

    # ---- 7: Thumb & ring extended, others curled ----
    if rng and thumb_out and not idx and not mid and not pky:
        return 7

    # ---- 8: Thumb & middle extended, others curled ----
    if mid and thumb_out and not idx and not rng and not pky:
        return 8

    # ---- 9: Thumb & index extended, others curled ----
    if idx and thumb_out and not mid and not rng and not pky:
        return 9

    # ---- 10: Thumbs up (fist, thumb straight out) ----
    if thumb_out and not idx and not mid and not rng and not pky:
        return 10

    return None

# ============================================================
#  DRAWING
# ============================================================
def draw_hand(frame, pts):
    for a, b in SKELETON:
        cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                 (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, tuple(p.astype(int)), 5, (0, 0, 0), -1)

def draw_debug(frame, feats, origin):
    x0, y0 = origin
    lines = []
    for name, v in feats['fingers'].items():
        lines.append(f"{name[:2]}: a={v['angle']:3.0f} r={v['reach']:.2f} "
                     f"t={v['tip_mcp']:.2f} {'UP' if v['extended'] else 'dn'}")
    lines.append(f"thumb: a={feats['thumb']['angle']:3.0f} "
                 f"out={'Y' if feats['thumb']['out'] else 'N'} "
                 f"str={'Y' if feats['thumb']['straight'] else 'N'}")
    lines.append("pinch " + " ".join(f"{k[0]}={v:.2f}" for k, v in feats['pinch'].items()))

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

# ============================================================
#  MAIN LOOP
# ============================================================
cap = cv2.VideoCapture(0)
history = deque(maxlen=SMOOTH_N)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    detected_this_frame = []

    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            pts = to_pixels(hand_lms, w, h)
            draw_hand(frame, pts)

            feats = analyse_hand(pts)
            number = classify_number(feats)

            if number is not None:
                detected_this_frame.append(number)
                cv2.putText(frame, str(number),
                            (int(pts[WRIST][0]) - 20, int(pts[WRIST][1]) + 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 4, cv2.LINE_AA)

            if SHOW_DEBUG:
                draw_debug(frame, feats, (int(pts[WRIST][0]) - 20,
                                          int(pts[WRIST][1]) + 100))

    # Temporal smoothing: majority vote over the last N frames
    current = detected_this_frame[0] if detected_this_frame else None
    history.append(current)
    valid = [n for n in history if n is not None]

    stable = None
    if len(valid) >= SMOOTH_N * 0.6:
        stable = Counter(valid).most_common(1)[0][0]

    if stable is not None:
        cv2.rectangle(frame, (10, 10), (150, 110), (0, 0, 0), -1)
        cv2.putText(frame, str(stable), (45, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6, cv2.LINE_AA)

    cv2.imshow("SASL numbers", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()