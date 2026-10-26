import cv2
import numpy as np
import joblib
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
MODEL_FILE = 'hand_model.pkl'

SMOOTH_N = 4
WRIST, MIDDLE_MCP = 0, 9

WINDOW_NAME = 'SASL numbers + alphabet (Both Hands)'

clf = joblib.load(MODEL_FILE)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
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
    '0': 'O / 0', 'O': 'O / 0',
    '2': 'V / 2', 'V': 'V / 2',
    '7': 'L / 7', 'L': 'L / 7',
}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
is_fullscreen = False

history = {'L': deque(maxlen=SMOOTH_N), 'R': deque(maxlen=SMOOTH_N)}

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    detected_hands = set()

    if results.hand_landmarks:
        for i, hand_lms in enumerate(results.hand_landmarks):
            handedness = results.handedness[i][0].category_name
            prefix = 'R' if handedness == 'Left' else 'L'   # swapped
            detected_hands.add(prefix)

            pts   = to_pixels(hand_lms, w, h)
            feats = extract_features(pts).reshape(1, -1)

            probs   = clf.predict_proba(feats)[0]
            classes = clf.classes_

            for pair in (('0', 'O'), ('2', 'V'), ('7', 'L')):
                idxs = []
                for ch in pair:
                    full = f"{prefix}_{ch}"
                    if full in classes:
                        idxs.append(int(np.where(classes == full)[0][0]))
                if len(idxs) == 2:
                    combined = probs[idxs[0]] + probs[idxs[1]]
                    probs[idxs[0]] = combined
                    probs[idxs[1]] = combined

            idx        = int(np.argmax(probs))
            confidence = float(probs[idx])
            predicted  = clf.classes_[idx]

            history[prefix].append(predicted if confidence > 0.3 else None)
            valid  = [n for n in history[prefix] if n is not None]
            stable = Counter(valid).most_common(1)[0][0] \
                     if len(valid) >= SMOOTH_N * 0.5 else None

            if stable is not None:
                letter       = stable.split('_')[-1]
                display_text = DISPLAY_MAP.get(letter, letter)

                x_off = 10 if prefix == 'R' else w - 260
                cv2.rectangle(frame, (x_off, 10), (x_off + 240, 130), (0, 0, 0), -1)

                font_scale = 3.0 if len(display_text) == 1 else 1.8
                cv2.putText(frame, display_text, (x_off + 15, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 255, 255), 6, cv2.LINE_AA)
                cv2.putText(frame, f"{confidence:.0%}", (x_off + 175, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, f"{prefix} hand", (x_off + 15, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for hp in ('L', 'R'):
        if hp not in detected_hands:
            history[hp].clear()

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        is_fullscreen = not is_fullscreen
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
        )

cap.release()
cv2.destroyAllWindows()