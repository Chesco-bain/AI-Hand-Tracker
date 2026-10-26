import cv2
import numpy as np
import joblib
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
MODEL_FILE = 'hand_model.pkl'
SMOOTH_N   = 8

WRIST, MIDDLE_MCP = 0, 9

clf = joblib.load(MODEL_FILE)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8,
)
detector = vision.HandLandmarker.create_from_options(options)

def to_pixels(lms, w, h):
    return np.array([[lm.x*w, lm.y*h] for lm in lms], dtype=np.float32)

def extract_features(pts):
    wrist = pts[WRIST]
    mcp   = pts[MIDDLE_MCP]
    size  = np.linalg.norm(mcp - wrist) + 1e-6
    vec   = mcp - wrist
    theta = np.arctan2(vec[1], vec[0])
    c, s  = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return ((pts - wrist) @ R.T / size).flatten()

DISPLAY_MAP = {
    '0': 'O / 0',
    'O': 'O / 0',
    '2': 'V / 2',
    'V': 'V / 2'
}

# ============================================================
# CAMERA SETUP 
# Change this to the number that worked in Step 1! (1, 2, or 3)
CAMERA_INDEX = 1

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
    print("Make sure DroidCam is running and try a different index.")
    exit()

# Read the first frame to verify the camera is actually sending data
ok, frame = cap.read()
if not ok:
    print("ERROR: Camera opened, but cannot read frames. DroidCam might be frozen.")
    exit()
# ============================================================

history = deque(maxlen=SMOOTH_N)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: 
        print("Lost connection to camera.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    predicted = None
    confidence = 0.0

    if results.hand_landmarks:
        pts = to_pixels(results.hand_landmarks[0], w, h)
        feats = extract_features(pts).reshape(1, -1)
        
        probs = clf.predict_proba(feats)[0]
        classes = clf.classes_
        
        # --- Combine probabilities for identical handshapes ---
        idx_0 = np.where(classes == '0')[0]
        idx_O = np.where(classes == 'O')[0]
        idx_2 = np.where(classes == '2')[0]
        idx_V = np.where(classes == 'V')[0]

        if len(idx_0) > 0 and len(idx_O) > 0:
            combined_O0 = probs[idx_0[0]] + probs[idx_O[0]]
            probs[idx_0[0]] = combined_O0
            probs[idx_O[0]] = combined_O0

        if len(idx_2) > 0 and len(idx_V) > 0:
            combined_V2 = probs[idx_2[0]] + probs[idx_V[0]]
            probs[idx_2[0]] = combined_V2
            probs[idx_V[0]] = combined_V2
        # ------------------------------------------------------

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        predicted = clf.classes_[idx]

    history.append(predicted if confidence > 0.5 else None)
    valid = [n for n in history if n is not None]
    stable = Counter(valid).most_common(1)[0][0] if len(valid) >= SMOOTH_N * 0.6 else None

    if stable is not None:
        display_text = DISPLAY_MAP.get(str(stable), str(stable))
        
        cv2.rectangle(frame, (10, 10), (250, 130), (0, 0, 0), -1)
        
        font_scale = 3.0 if len(display_text) == 1 else 1.8
        
        cv2.putText(frame, display_text, (25, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, f"{confidence:.0%}", (180, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('SASL numbers + alphabet (DroidCam)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()