import cv2
import numpy as np
import csv
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = 'hand_landmarker.task'
DATA_FILE  = 'hand_data.csv'

WRIST, MIDDLE_MCP = 0, 9
SKELETON = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20), (0,17),
]

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
    """42-dim feature vector: translate to wrist, rotate to wrist->middle_mcp,
    scale by hand size. Invariant to position, rotation, and distance."""
    wrist = pts[WRIST]
    mcp   = pts[MIDDLE_MCP]
    size  = np.linalg.norm(mcp - wrist) + 1e-6
    vec   = mcp - wrist
    theta = np.arctan2(vec[1], vec[0])
    c, s  = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    return ((pts - wrist) @ R.T / size).flatten()

# Load existing samples if present
samples = []
if os.path.exists(DATA_FILE):
    with open(DATA_FILE) as f:
        for row in csv.reader(f):
            samples.append((int(row[0]), [float(x) for x in row[1:]]))
    print(f"Loaded {len(samples)} existing samples.")

counts = {i: 0 for i in range(10)}
for lbl, _ in samples:
    counts[lbl] += 1

print("Hold a handshape and press 0-9 to save. Backspace=undo, q=quit&save.")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    feats = None
    if results.hand_landmarks:
        pts = to_pixels(results.hand_landmarks[0], w, h)
        for a, b in SKELETON:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (0,255,0), 2)
        for p in pts:
            cv2.circle(frame, tuple(p.astype(int)), 5, (0,0,0), -1)
        feats = extract_features(pts)

    cv2.putText(frame, f"Total: {sum(counts.values())}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    for i in range(10):
        cv2.putText(frame, f"{i}: {counts[i]}", (10, 65 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow('Collect SASL data', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif ord('0') <= key <= ord('9') and feats is not None:
        lbl = key - ord('0')
        samples.append((lbl, feats))
        counts[lbl] += 1
        print(f"Saved {lbl}  (total for {lbl}: {counts[lbl]})")
    elif key == 8 and samples:
        lbl, _ = samples.pop()
        counts[lbl] -= 1
        print(f"Undid last sample for {lbl}")

cap.release()
cv2.destroyAllWindows()

with open(DATA_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    for lbl, feats in samples:
        writer.writerow([lbl] + list(feats))

print(f"Saved {len(samples)} samples to {DATA_FILE}")