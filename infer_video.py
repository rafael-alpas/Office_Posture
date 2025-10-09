# =========================================================
#  POSTURE ANALYSIS HYBRID MODEL (YOLO + MediaPipe + Classifier)
# =========================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import cv2
import torch
import numpy as np
import joblib
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
from collections import Counter, deque
import math
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
CLASSIFIER_PATH = "runs/train/posture_model/posture_classifier_final.keras"
SCALER_PATH = "runs/train/posture_model/scaler.pkl"
VIDEO_DIR = "test_videos/mark.mp4"
OUTPUT_DIR = "runs/inference_videos"
RESULT_DIR = "infer_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ===============================
# LOAD MODELS
# ===============================
print("[INFO] Loading YOLOv5 (person detection)...")
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
yolo.conf = 0.25
yolo.iou = 0.45
yolo.max_det = 5
yolo.classes = [0]  # Person only

print("[INFO] Loading Posture Classifier...")
classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

print("[INFO] Loading Scaler...")
scaler = joblib.load(SCALER_PATH)

# ===============================
# Initialize MediaPipe Pose
# ===============================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,
                    min_tracking_confidence=0.7, model_complexity=2)

# ===============================
# Helper: Extract landmarks
# ===============================
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)
    if not result.pose_landmarks:
        return None, None
    lm = []
    for landmark in result.pose_landmarks.landmark:
        lm.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return np.array(lm).reshape(1, -1), result.pose_landmarks

# ===============================
# Helper: Draw pose + spine tilt
# ===============================
def draw_pose_overlay(image, landmarks, label):
    h, w, _ = image.shape
    pts = landmarks.landmark

    def point(idx): return (int(pts[idx].x * w), int(pts[idx].y * h))

    for i in [11, 12, 23, 24]:
        cv2.circle(image, point(i), 5, (0, 255, 255), -1)

    cv2.line(image, point(11), point(12), (255, 255, 255), 2)
    cv2.line(image, point(23), point(24), (255, 255, 255), 2)

    mid_shoulder = ((pts[11].x + pts[12].x) / 2, (pts[11].y + pts[12].y) / 2)
    mid_hip = ((pts[23].x + pts[24].x) / 2, (pts[23].y + pts[24].y) / 2)

    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    angle = math.degrees(math.atan2(dy, dx))

    if abs(angle) > 15:
        color = (0, 0, 255)
    elif label == "good_posture":
        color = (0, 255, 0)
    else:
        color = (0, 255, 255)

    direction = "Lean Back" if dx > 0 else "Lean Forward"

    start = (int(mid_hip[0] * w), int(mid_hip[1] * h))
    end = (int(mid_shoulder[0] * w), int(mid_shoulder[1] * h))
    cv2.arrowedLine(image, start, end, color, 3, tipLength=0.3)
    cv2.putText(image, f"Tilt: {angle:.1f}° {direction}", (start[0], start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return angle

# ===============================
# Helper: Draw posture bar
# ===============================
def draw_posture_bar(frame, good_count, bad_count):
    total = good_count + bad_count
    if total == 0:
        good_pct, bad_pct = 0, 0
    else:
        good_pct = int((good_count / total) * 100)
        bad_pct = 100 - good_pct

    bar_x, bar_y = 50, 40
    bar_w, bar_h = 300, 20
    cv2.putText(frame, f"Good: {good_pct}%   Bad: {bad_pct}%", (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * good_pct / 100), bar_y + bar_h), (0, 255, 0), -1)
    cv2.rectangle(frame, (bar_x, bar_y + 25), (bar_x + int(bar_w * bad_pct / 100), bar_y + 25 + bar_h), (0, 0, 255), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, bar_y + 25), (bar_x + bar_w, bar_y + 25 + bar_h), (255, 255, 255), 2)

# ===============================
# Inference Loop
# ===============================
print("\n--- Starting Inference ---\n")
total_predictions = Counter()
recent_preds = deque(maxlen=5)  # smoothing

video_path = Path(VIDEO_DIR)
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = Path(OUTPUT_DIR) / f"{video_path.stem}_annotated.mp4"
writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

preds_video = Counter()
frame_idx = 0
results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    detections = yolo(frame).xyxy[0]
    annotated = frame.copy()

    if len(detections) > 0:
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            person = frame[y1:y2, x1:x2].copy()
            if person.size == 0:
                continue

            lm, raw = extract_landmarks(person)
            if lm is None:
                continue

            # ===== FIX: Scale landmarks =====
            landmarks_scaled = scaler.transform(lm)

            # ===== FIX: Classifier prediction =====
            pred = classifier.predict(landmarks_scaled, verbose=0)
            conf_posture = float(pred[0][0])
            pred_label = "good_posture" if conf_posture > 0.5 else "bad_posture"

            # ---------- SMOOTHING ----------
            recent_preds.append(pred_label)
            label = Counter(recent_preds).most_common(1)[0][0]
            # ---------------------------------

            preds_video[label] += 1
            total_predictions[label] += 1

            color = (0, 255, 0) if label == "good_posture" else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} ({conf_posture:.2f})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            angle = draw_pose_overlay(person, raw, label)
            annotated[y1:y2, x1:x2] = person

            results.append({
                "frame": frame_idx,
                "label": label,
                "confidence": conf_posture,
                "tilt_angle": angle
            })

    draw_posture_bar(annotated, preds_video.get("good_posture", 0), preds_video.get("bad_posture", 0))
    writer.write(annotated)

# ===============================
# Summary Frame & Save CSV + PNG
# ===============================
good = preds_video["good_posture"]
bad = preds_video["bad_posture"]
total = good + bad
good_pct = int((good / total) * 100) if total else 0
bad_pct = 100 - good_pct

summary_csv = Path(RESULT_DIR) / f"{video_path.stem}.csv"
summary_png = Path(RESULT_DIR) / f"{video_path.stem}_summary.png"

pd.DataFrame(results).to_csv(summary_csv, index=False)

plt.figure(figsize=(5, 4))
plt.bar(["Good", "Bad"], [good_pct, bad_pct], color=["green", "red"])
plt.ylabel("Percentage")
plt.title(f"Posture Summary - {video_path.stem}")
plt.savefig(summary_png)
plt.close()

summary = np.zeros((height, width, 3), dtype=np.uint8)
cv2.putText(summary, f"Posture Summary - {video_path.name}", (50, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(summary, f"Frames: {frame_idx}", (50, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
cv2.putText(summary, f"Good: {good} ({good_pct}%)", (50, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.putText(summary, f"Bad:  {bad} ({bad_pct}%)", (50, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

tip = (
    "Poor posture detected often please adjust!" if bad_pct > 50
    else "Mostly good posture!" if bad_pct > 20
    else "Excellent posture maintained!"
)
cv2.putText(summary, tip, (50, 280),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

for _ in range(int(fps * 3)):
    writer.write(summary)

cap.release()
writer.release()

print(f"[DONE] Annotated video saved → {output_path}")
print(f"[DONE] CSV results saved → {summary_csv}")
print(f"[DONE] PNG summary saved → {summary_png}")
print(f"Frames processed: {frame_idx}")
print(f"Posture summary: {preds_video}")
print("\n--- Inference Complete ---\n")
