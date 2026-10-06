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
SKELETON = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20), (0,17),
]

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

cap = cv2.VideoCapture(0)
history = deque(maxlen=SMOOTH_N)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    predicted = None
    confidence = 0.0

    if results.hand_landmarks:
        pts = to_pixels(results.hand_landmarks[0], w, h)
        for a, b in SKELETON:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (0,255,0), 2)
        for p in pts:
            cv2.circle(frame, tuple(p.astype(int)), 5, (0,0,0), -1)

        feats = extract_features(pts).reshape(1, -1)
        probs = clf.predict_proba(feats)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        predicted = int(clf.classes_[idx])

    history.append(predicted if confidence > 0.5 else None)
    valid = [n for n in history if n is not None]
    stable = Counter(valid).most_common(1)[0][0] if len(valid) >= SMOOTH_N * 0.6 else None

    if stable is not None:
        cv2.rectangle(frame, (10, 10), (170, 110), (0, 0, 0), -1)
        cv2.putText(frame, str(stable), (55, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, f"{confidence:.0%}", (120, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('SASL numbers (trained)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()