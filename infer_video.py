import argparse
import json
import warnings
from collections import Counter, deque
from pathlib import Path
import math
import textwrap
from typing import Dict, List, Optional

# Suppress noisy future warnings from dependencies to keep logs readable.
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import joblib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

# Defaults mirror legacy hard-coded values so behaviour stays unchanged.
DEFAULT_CLASSIFIER_PATH = "runs/train/posture_model/posture_classifier_final.keras"
DEFAULT_SCALER_PATH = "runs/train/posture_model/scaler.pkl"
DEFAULT_VIDEO_PATH = "test_videos/mark.mp4"
DEFAULT_OUTPUT_DIR = "runs/inference_videos"
DEFAULT_RESULT_DIR = "infer_results"

# Persistent MediaPipe pose estimator configured for streaming video.
mp_pose = mp.solutions.pose
POSE_STREAM = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=2,
)


# Camera orientation presets adjust tilt interpretation for different viewpoints.
CAMERA_ORIENTATION_CONFIG: Dict[str, Dict[str, float]] = {
    "front": {"offset": 0.0, "sign": 1.0},
    "north": {"offset": 0.0, "sign": 1.0},
    "northwest": {"offset": -15.0, "sign": -1.0},
    "side": {"offset": -90.0, "sign": -1.0},
}

# Preferred landmark order when estimating head direction.
HEAD_REFERENCE_ORDER: Dict[str, tuple[int, ...]] = {
    "front": (7, 8, 0),
    "north": (7, 8, 0),
    "northwest": (7, 0, 8),
    "side": (7, 0, 8),
}

VISIBILITY_MIN = 0.2
COORD_MARGIN = 0.35
SPINE_TILT_NEUTRAL_DEG = 15.0
HEAD_TILT_NEUTRAL_DEG = 10.0
OVERLAY_SMOOTH_ALPHA = 0.75
PIXEL_CATCHUP_DISTANCE = 20.0
ANGLE_MAX_STEP = 8.0
ANGLE_CATCHUP_THRESHOLD = 22.5
OVERLAY_MISSING_FRAMES = 8
MIN_SEGMENT_PIXELS = 6.0
ANGLE_MEDIAN_WINDOW = 5


def normalize_angle(angle: float) -> float:
    """Wrap angle into [-180, 180] for clearer comparisons."""
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def adjust_for_camera(angle: float, orientation: str, offset_override: Optional[float] = None) -> float:
    """Apply camera orientation correction to align tilt direction."""
    config = CAMERA_ORIENTATION_CONFIG.get(orientation, {"offset": 0.0, "sign": 1.0})
    offset = config["offset"] if offset_override is None else offset_override
    signed = (angle + offset) * config.get("sign", 1.0)
    return normalize_angle(signed)


def weighted_point(
    landmarks,
    indices,
    min_vis: float = VISIBILITY_MIN,
    margin: float = COORD_MARGIN,
) -> Optional[np.ndarray]:
    """Return visibility-weighted average of the given landmark indices."""
    coords = []
    weights = []
    for idx in indices:
        lm = landmarks[idx]
        if lm.visibility >= VISIBILITY_MIN:
            if -margin <= lm.x <= 1.0 + margin and -margin <= lm.y <= 1.0 + margin:
                coords.append([
                    float(np.clip(lm.x, 0.0, 1.0)),
                    float(np.clip(lm.y, 0.0, 1.0)),
                    lm.z,
                ])
                weights.append(max(lm.visibility, 1e-3))
    if not coords:
        return None
    coords = np.array(coords)
    weights = np.array(weights)
    weighted = np.average(coords, axis=0, weights=weights)
    return weighted


def estimate_body_yaw(landmarks, min_vis: float = VISIBILITY_MIN) -> Optional[float]:
    """Approximate subject yaw using shoulders first, then hips if shoulders fail."""
    pairs = [(11, 12), (23, 24)]
    for left_idx, right_idx in pairs:
        left = landmarks[left_idx]
        right = landmarks[right_idx]
        if left.visibility >= min_vis and right.visibility >= min_vis:
            dx = right.x - left.x
            dz = right.z - left.z
            if abs(dx) < 1e-5 and abs(dz) < 1e-5:
                continue
            return math.degrees(math.atan2(dz, dx))
    return None


def build_feedback(avg_spine_tilt: Optional[float], avg_head_tilt: Optional[float]) -> List[str]:
    """Generate adaptive guidance lines based on observed tilt severities."""
    tips: List[str] = []
    if avg_head_tilt is not None and avg_head_tilt > HEAD_TILT_NEUTRAL_DEG:
        tips.append("Try lifting your chin and aligning your ears with your shoulders.")
    if avg_spine_tilt is not None and avg_spine_tilt > SPINE_TILT_NEUTRAL_DEG:
        tips.append("Straighten your spine and keep your shoulders back.")
    if not tips:
        tips.append("Posture looks balanced - keep it up!")
    return tips



def create_tracker():
    """Return a CSRT tracker instance when OpenCV supports it."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker is not available in this OpenCV build.")


def extract_landmarks(image):
    """Convert a cropped frame into landmarks + the raw pose structure."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = POSE_STREAM.process(image_rgb)
    if not results.pose_landmarks:
        return None, None
    lm = []
    for landmark in results.pose_landmarks.landmark:
        lm.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return np.array(lm).reshape(1, -1), results.pose_landmarks


def draw_pose_overlay(
    image,
    landmarks,
    label,
    orientation,
    offset_override,
    overlay_state,
    smooth_alpha=OVERLAY_SMOOTH_ALPHA,
):
    """Overlay key joints plus spine and head tilt vectors for visual feedback."""
    h, w, _ = image.shape
    pts = landmarks.landmark

    def point_px(idx: int) -> tuple[int, int]:
        lm = pts[idx]
        return int(lm.x * w), int(lm.y * h)

    for key in (
        "mid_shoulder",
        "mid_hip",
        "ear_mid",
        "spine_start_px",
        "spine_end_px",
        "head_start_px",
        "head_end_px",
    ):
        overlay_state.setdefault(key, None)
    overlay_state.setdefault("spine_missing", OVERLAY_MISSING_FRAMES)
    overlay_state.setdefault("head_missing", OVERLAY_MISSING_FRAMES)
    overlay_state.setdefault("spine_angle", None)
    overlay_state.setdefault("head_angle", None)
    if "spine_angle_history" not in overlay_state:
        overlay_state["spine_angle_history"] = deque(maxlen=ANGLE_MEDIAN_WINDOW)
    if "head_angle_history" not in overlay_state:
        overlay_state["head_angle_history"] = deque(maxlen=ANGLE_MEDIAN_WINDOW)

    for idx in (11, 12, 23, 24, 7, 8):
        lm = pts[idx]
        if lm.visibility >= VISIBILITY_MIN:
            cv2.circle(image, point_px(idx), 5, (0, 255, 255), -1)

    def as_point2(raw):
        if raw is None:
            return None
        return np.array(raw[:2], dtype=np.float32)

    def smooth_point(key, new_point):
        prev = overlay_state.get(key)
        if new_point is None:
            return prev
        new_arr = np.array(new_point, dtype=np.float32)
        if prev is None:
            overlay_state[key] = new_arr
            return new_arr
        dist = np.linalg.norm(new_arr - prev)
        if dist >= PIXEL_CATCHUP_DISTANCE:
            overlay_state[key] = new_arr
            return new_arr
        smoothed = prev * smooth_alpha + new_arr * (1.0 - smooth_alpha)
        overlay_state[key] = smoothed
        return smoothed

    def smooth_px(key, new_point):
        if new_point is None:
            prev = overlay_state.get(key)
            return prev if isinstance(prev, np.ndarray) else None
        new_arr = np.array(new_point, dtype=np.float32)
        prev = overlay_state.get(key)
        if prev is None or not isinstance(prev, np.ndarray):
            overlay_state[key] = new_arr
            return new_arr
        dist = np.linalg.norm(new_arr - prev)
        if dist >= PIXEL_CATCHUP_DISTANCE:
            overlay_state[key] = new_arr
            return new_arr
        smoothed = prev * smooth_alpha + new_arr * (1.0 - smooth_alpha)
        overlay_state[key] = smoothed
        return smoothed

    def to_px(point):
        if point is None:
            return None
        x = np.clip(point[0], 0.0, 1.0) * w
        y = np.clip(point[1], 0.0, 1.0) * h
        return np.array([x, y], dtype=np.float32)

    def to_int_tuple(arr):
        if arr is None:
            return None
        return tuple(np.round(arr).astype(int))

    def contract_segment(start_px, end_px, pad_px, prev_start_key, prev_end_key):
        start_arr = np.array(start_px, dtype=np.float32)
        end_arr = np.array(end_px, dtype=np.float32)
        vec = end_arr - start_arr
        length = np.linalg.norm(vec)
        if length <= MIN_SEGMENT_PIXELS:
            prev_start = overlay_state.get(prev_start_key)
            prev_end = overlay_state.get(prev_end_key)
            if isinstance(prev_start, np.ndarray) and isinstance(prev_end, np.ndarray):
                return prev_start, prev_end
            return None
        direction = vec / max(length, 1e-4)
        adj_start = start_arr + direction * pad_px
        adj_end = end_arr - direction * pad_px
        return adj_start, adj_end

    def smooth_angle_value(key, angle):
        if angle is None:
            return overlay_state.get(key)
        prev = overlay_state.get(key)
        if prev is None:
            overlay_state[key] = angle
            return angle
        delta = normalize_angle(angle - prev)
        if abs(delta) >= ANGLE_CATCHUP_THRESHOLD:
            overlay_state[key] = angle
            return angle
        delta = max(-ANGLE_MAX_STEP, min(ANGLE_MAX_STEP, delta))
        smoothed = normalize_angle(prev + delta)
        overlay_state[key] = smoothed
        return smoothed

    def update_angle_history(key, angle):
        history = overlay_state[key]
        if angle is None:
            return history[-1] if history else None
        history.append(angle)
        return float(np.median(np.array(history, dtype=np.float32)))

    def compute_vertical(a, b):
        if a is None or b is None:
            return None
        return normalize_angle(math.degrees(math.atan2(a[1] - b[1], a[0] - b[0])) + 90.0)

    mid_shoulder_raw = as_point2(weighted_point(pts, [11, 12]))
    mid_hip_raw = as_point2(weighted_point(pts, [23, 24]))

    head_order = HEAD_REFERENCE_ORDER.get(orientation, (7, 8, 0))
    ear_mid_raw = None
    for idx in head_order:
        lm = pts[idx]
        if lm.visibility >= VISIBILITY_MIN and -COORD_MARGIN <= lm.x <= 1.0 + COORD_MARGIN and -COORD_MARGIN <= lm.y <= 1.0 + COORD_MARGIN:
            ear_mid_raw = np.array([np.clip(lm.x, 0.0, 1.0), np.clip(lm.y, 0.0, 1.0)], dtype=np.float32)
            break
    if ear_mid_raw is None:
        ear_mid_raw = as_point2(weighted_point(pts, list(dict.fromkeys(head_order))))

    mid_shoulder = smooth_point("mid_shoulder", mid_shoulder_raw)
    mid_hip = smooth_point("mid_hip", mid_hip_raw)
    ear_mid = smooth_point("ear_mid", ear_mid_raw)

    raw_spine_available = mid_shoulder_raw is not None and mid_hip_raw is not None
    raw_head_available = ear_mid_raw is not None and mid_shoulder_raw is not None

    overlay_state["spine_missing"] = 0 if raw_spine_available else min(OVERLAY_MISSING_FRAMES, overlay_state["spine_missing"] + 1)
    overlay_state["head_missing"] = 0 if raw_head_available else min(OVERLAY_MISSING_FRAMES, overlay_state["head_missing"] + 1)

    if overlay_state["spine_missing"] >= OVERLAY_MISSING_FRAMES:
        overlay_state["spine_start_px"] = None
        overlay_state["spine_end_px"] = None
        overlay_state["spine_angle_history"].clear()
        overlay_state["spine_angle"] = None
    if overlay_state["head_missing"] >= OVERLAY_MISSING_FRAMES:
        overlay_state["head_start_px"] = None
        overlay_state["head_end_px"] = None
        overlay_state["head_angle_history"].clear()
        overlay_state["head_angle"] = None

    yaw_estimate = estimate_body_yaw(pts)

    metrics = {
        "spine_tilt_raw": None,
        "spine_tilt": None,
        "spine_direction": "No data",
        "head_tilt_raw": None,
        "head_tilt": None,
        "head_direction": "No data",
        "yaw": yaw_estimate,
        "spine_visible": mid_shoulder is not None and mid_hip is not None,
        "head_visible": ear_mid is not None and mid_shoulder is not None,
        "spine_measured": raw_spine_available,
        "head_measured": raw_head_available,
        "spine_drawn": False,
        "head_drawn": False,
    }

    def update_segment(raw_start, raw_end, start_key, end_key, pad_px):
        if raw_start is None or raw_end is None:
            return
        start_px = to_px(raw_start)
        end_px = to_px(raw_end)
        if start_px is None or end_px is None:
            return
        contracted = contract_segment(start_px, end_px, pad_px, start_key, end_key)
        if contracted is None:
            return
        overlay_state[start_key] = smooth_px(start_key, contracted[0])
        overlay_state[end_key] = smooth_px(end_key, contracted[1])

    update_segment(mid_hip_raw, mid_shoulder_raw, "spine_start_px", "spine_end_px", pad_px=10.0)
    update_segment(mid_shoulder_raw, ear_mid_raw, "head_start_px", "head_end_px", pad_px=8.0)

    metrics["spine_tilt_raw"] = compute_vertical(mid_shoulder_raw, mid_hip_raw)
    metrics["head_tilt_raw"] = compute_vertical(ear_mid_raw, mid_shoulder_raw)

    if metrics["spine_tilt_raw"] is not None:
        candidate = adjust_for_camera(metrics["spine_tilt_raw"], orientation, offset_override)
        candidate = update_angle_history("spine_angle_history", candidate)
        spine_angle = smooth_angle_value("spine_angle", candidate)
        metrics["spine_tilt"] = spine_angle
        if spine_angle is not None:
            if spine_angle > 5:
                metrics["spine_direction"] = "Lean Forward"
            elif spine_angle < -5:
                metrics["spine_direction"] = "Lean Back"
            else:
                metrics["spine_direction"] = "Neutral"

    if metrics["head_tilt_raw"] is not None:
        candidate = adjust_for_camera(metrics["head_tilt_raw"], orientation, offset_override)
        candidate = update_angle_history("head_angle_history", candidate)
        head_angle = smooth_angle_value("head_angle", candidate)
        metrics["head_tilt"] = head_angle
        if head_angle is not None:
            if head_angle > 5:
                metrics["head_direction"] = "Head Forward"
            elif head_angle < -5:
                metrics["head_direction"] = "Head Back"
            else:
                metrics["head_direction"] = "Neutral"

    spine_start_draw = to_int_tuple(overlay_state["spine_start_px"])
    spine_end_draw = to_int_tuple(overlay_state["spine_end_px"])
    if spine_start_draw and spine_end_draw and metrics["spine_tilt"] is not None:
        metrics["spine_drawn"] = True
        angle = metrics["spine_tilt"]
        if abs(angle) > SPINE_TILT_NEUTRAL_DEG:
            spine_color = (0, 0, 255)
        elif label == "good_posture":
            spine_color = (0, 255, 0)
        else:
            spine_color = (0, 255, 255)
        cv2.arrowedLine(image, spine_start_draw, spine_end_draw, spine_color, 3, tipLength=0.3)
        cv2.putText(
            image,
            f"Spine: {angle:.1f} deg {metrics['spine_direction']}",
            (spine_start_draw[0], max(20, spine_start_draw[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            spine_color,
            2,
        )

    head_start_draw = to_int_tuple(overlay_state["head_start_px"])
    head_end_draw = to_int_tuple(overlay_state["head_end_px"])
    if head_start_draw and head_end_draw and metrics["head_tilt"] is not None:
        metrics["head_drawn"] = True
        angle = metrics["head_tilt"]
        if abs(angle) <= HEAD_TILT_NEUTRAL_DEG:
            head_color = (0, 140, 255)
        else:
            head_color = (0, 69, 255)
        cv2.arrowedLine(image, head_start_draw, head_end_draw, head_color, 2, tipLength=0.25)
        cv2.circle(image, head_end_draw, 4, head_color, -1)
        cv2.putText(
            image,
            f"Head: {angle:.1f} deg {metrics['head_direction']}",
            (head_end_draw[0], max(20, head_end_draw[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            head_color,
            2,
        )

    metrics["tracking_lost"] = not (metrics["spine_drawn"] or metrics["head_drawn"])
    return metrics


def draw_posture_bar(frame, good_count, bad_count):
    """Render running class percentages as stacked bars on the frame."""
    total = good_count + bad_count
    if total == 0:
        good_pct, bad_pct = 0, 0
    else:
        good_pct = int((good_count / total) * 100)
        bad_pct = 100 - good_pct

    bar_x, bar_y = 50, 40
    bar_w, bar_h = 300, 20
    cv2.putText(
        frame,
        f"Good: {good_pct}%   Bad: {bad_pct}%",
        (bar_x, bar_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_w * good_pct / 100), bar_y + bar_h),
        (0, 255, 0),
        -1,
    )
    cv2.rectangle(
        frame,
        (bar_x, bar_y + 25),
        (bar_x + int(bar_w * bad_pct / 100), bar_y + 25 + bar_h),
        (0, 0, 255),
        -1,
    )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
    cv2.rectangle(
        frame,
        (bar_x, bar_y + 25),
        (bar_x + bar_w, bar_y + 25 + bar_h),
        (255, 255, 255),
        2,
    )


def save_summary(result_dir, video_path, frame_idx, preds_video, results, fps):
    """
    Persist per-frame predictions and create a simple bar-chart summary.
    Returns aggregate counts used later for the final overlay.
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    video_stem = video_path.stem

    summary_csv = result_dir / f"{video_stem}.csv"
    summary_png = result_dir / f"{video_stem}_summary.png"

    pd.DataFrame(results).to_csv(summary_csv, index=False)

    good = preds_video["good_posture"]
    bad = preds_video["bad_posture"]
    total = good + bad
    good_pct = int((good / total) * 100) if total else 0
    bad_pct = 100 - good_pct

    plt.figure(figsize=(5, 4))
    plt.bar(["Good", "Bad"], [good_pct, bad_pct], color=["green", "red"])
    plt.ylabel("Percentage")
    plt.title(f"Posture Summary - {video_stem}")
    plt.savefig(summary_png)
    plt.close()

    print(f"[DONE] CSV results saved -> {summary_csv}")
    print(f"[DONE] PNG summary saved -> {summary_png}")

    return good, bad, good_pct, bad_pct


def parse_args():
    """Expose all tunable knobs as CLI flags so experiments stay reproducible."""
    parser = argparse.ArgumentParser(
        description="Run posture inference on a video using YOLOv5 + MediaPipe + classifier."
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--classifier-path", default=DEFAULT_CLASSIFIER_PATH, help="Trained classifier path.")
    parser.add_argument("--scaler-path", default=DEFAULT_SCALER_PATH, help="StandardScaler path.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where annotated videos will be written.",
    )
    parser.add_argument(
        "--result-dir",
        default=DEFAULT_RESULT_DIR,
        help="Directory for CSV summaries and plots.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.1, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--yolo-max-det", type=int, default=1, help="Max detections per frame.")
    parser.add_argument("--box-expansion", type=float, default=0.05, help="Fractional padding around YOLO box.")
    parser.add_argument("--box-min-frac", type=float, default=0.25, help="Minimum fraction of frame covered by box.")
    parser.add_argument("--box-max-frac", type=float, default=0.8, help="Maximum fraction of frame covered by box.")
    parser.add_argument("--ema-alpha", type=float, default=0.9, help="EMA smoothing factor for box center/size.")
    parser.add_argument("--box-clamp", type=float, default=0.1, help="Max fractional box size change per frame.")
    parser.add_argument(
        "--tracker-blend",
        type=float,
        default=0.2,
        help="Blend factor between tracker box and fresh YOLO detection.",
    )
    parser.add_argument(
        "--max-lost",
        type=int,
        default=15,
        help="Number of frames to reuse last box when detections are missing.",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=5,
        help="Length of prediction smoothing deque.",
    )
    parser.add_argument(
        "--use-tracker",
        action="store_true",
        help="Enable CSRT tracker fusion (disabled by default).",
    )
    parser.add_argument(
        "--tracker-reinit-secs",
        type=float,
        default=3.0,
        help="Seconds between tracker reinitialisations when enabled.",
    )
    parser.add_argument(
        "--refine-with-landmarks",
        action="store_true",
        help="Recentre bounding boxes using body landmarks (experimental).",
    )
    parser.add_argument(
        "--landmark-margin",
        type=float,
        default=0.1,
        help="Extra padding around the landmark-derived box when refinement is enabled.",
    )
    parser.add_argument(
        "--camera-angle",
        default="front",
        choices=sorted(CAMERA_ORIENTATION_CONFIG.keys()),
        help="Camera orientation preset used to correct tilt direction.",
    )
    parser.add_argument(
        "--camera-offset",
        type=float,
        default=None,
        help="Manual tilt offset in degrees (applied after the preset).",
    )
    return parser.parse_args()


def run_inference(args):
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    result_dir = Path(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading YOLOv5 (person detection)...")
    yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    yolo.conf = args.yolo_conf
    yolo.iou = args.yolo_iou
    yolo.max_det = args.yolo_max_det
    yolo.classes = [0]

    print("[INFO] Loading Posture Classifier...")
    classifier = tf.keras.models.load_model(args.classifier_path)

    print("[INFO] Loading Scaler...")
    scaler = joblib.load(args.scaler_path)

    orientation = args.camera_angle.lower()
    offset_override = args.camera_offset

    print("\n--- Starting Inference ---\n")
    print(f"[INFO] Camera angle correction mode: {orientation}")
    if offset_override is not None:
        print(f"[INFO] Using manual tilt offset override: {offset_override:.2f} deg")

    recent_preds = deque(maxlen=args.recent_window)
    preds_video = Counter()
    total_predictions = Counter()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_path = output_dir / f"{video_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    tracker = None
    tracking = False
    reinit_interval = int(fps * args.tracker_reinit_secs) if args.tracker_reinit_secs > 0 else 0
    smoothed_box = None
    lost_frames = 0
    frame_idx = 0
    results = []
    overlay_state: Dict[str, Optional[np.ndarray]] = {}

    frame_area = frame_width * frame_height
    frames_with_box = 0
    frames_with_yolo = 0
    box_areas: List[float] = []
    loss_streak = 0
    max_loss_streak = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        detections = yolo(frame).xyxy[0]
        annotated = frame.copy()

        current_box = None
        measurement_box = None

        if args.use_tracker and tracking:
            ok, track_box = tracker.update(frame)
            if ok:
                tx, ty, tw, th = track_box
                current_box = (tx + tw / 2, ty + th / 2, tw, th)
            else:
                tracking = False
                tracker = None

        if len(detections) > 0:
            detections = detections[detections[:, 4].argsort(descending=True)]
            mx1, my1, mx2, my2 = map(float, detections[0][:4])
            mw = mx2 - mx1
            mh = my2 - my1
            mcx = mx1 + mw / 2
            mcy = my1 + mh / 2
            mw *= (1 + 2 * args.box_expansion)
            mh *= (1 + 2 * args.box_expansion)
            min_w = frame_width * args.box_min_frac
            max_w = frame_width * args.box_max_frac
            min_h = frame_height * args.box_min_frac
            max_h = frame_height * args.box_max_frac
            mw = min(max(mw, min_w), max_w)
            mh = min(max(mh, min_h), max_h)
            mcx = min(max(mcx, mw / 2), frame_width - mw / 2)
            mcy = min(max(mcy, mh / 2), frame_height - mh / 2)
            measurement_box = (mcx, mcy, mw, mh)
            frames_with_yolo += 1

        need_reinit = (
            args.use_tracker
            and measurement_box is not None
            and (not tracking or (reinit_interval and frame_idx % reinit_interval == 0))
        )

        if need_reinit:
            cx, cy, mw, mh = measurement_box
            tracker = create_tracker()
            bx = int(round(cx - mw / 2))
            by = int(round(cy - mh / 2))
            bw = max(2, int(round(mw)))
            bh = max(2, int(round(mh)))
            bx = max(0, min(frame_width - bw, bx))
            by = max(0, min(frame_height - bh, by))
            tracker.init(frame, (bx, by, bw, bh))
            tracking = True
            current_box = measurement_box
            lost_frames = 0
        elif current_box is None and measurement_box is not None:
            current_box = measurement_box
            lost_frames = 0
        elif current_box is not None and measurement_box is not None:
            cx, cy, w, h = current_box
            mcx, mcy, mw, mh = measurement_box
            blend = args.tracker_blend
            current_box = (
                cx * (1 - blend) + mcx * blend,
                cy * (1 - blend) + mcy * blend,
                w * (1 - blend) + mw * blend,
                h * (1 - blend) + mh * blend,
            )
            lost_frames = 0

        if current_box is None:
            lost_frames += 1
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
            if smoothed_box is not None and lost_frames <= args.max_lost:
                current_box = smoothed_box
            else:
                smoothed_box = None
                overlay_state.clear()
                recent_preds.clear()
                if args.use_tracker:
                    tracking = False
                    tracker = None
                draw_posture_bar(
                    annotated,
                    preds_video.get("good_posture", 0),
                    preds_video.get("bad_posture", 0),
                )
                writer.write(annotated)
                continue
        else:
            lost_frames = 0
            loss_streak = 0
            frames_with_box += 1

        if smoothed_box is None:
            smoothed_box = current_box
        else:
            prev_cx, prev_cy, prev_w, prev_h = smoothed_box
            cur_cx, cur_cy, cur_w, cur_h = current_box
            clamp = args.box_clamp
            cur_w = max(prev_w * (1 - clamp), min(cur_w, prev_w * (1 + clamp)))
            cur_h = max(prev_h * (1 - clamp), min(cur_h, prev_h * (1 + clamp)))
            alpha = args.ema_alpha
            cx = prev_cx * alpha + cur_cx * (1 - alpha)
            cy = prev_cy * alpha + cur_cy * (1 - alpha)
            box_w = prev_w * alpha + cur_w * (1 - alpha)
            box_h = prev_h * alpha + cur_h * (1 - alpha)
            min_w = frame_width * args.box_min_frac
            max_w = frame_width * args.box_max_frac
            min_h = frame_height * args.box_min_frac
            max_h = frame_height * args.box_max_frac
            box_w = min(max(box_w, min_w), max_w)
            box_h = min(max(box_h, min_h), max_h)
            cx = min(max(cx, box_w / 2), frame_width - box_w / 2)
            cy = min(max(cy, box_h / 2), frame_height - box_h / 2)
            smoothed_box = (cx, cy, box_w, box_h)

        cx, cy, box_w, box_h = smoothed_box
        x1 = int(round(cx - box_w / 2))
        y1 = int(round(cy - box_h / 2))
        x2 = int(round(cx + box_w / 2))
        y2 = int(round(cy + box_h / 2))
        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(x1 + 1, min(frame_width, x2))
        y2 = max(y1 + 1, min(frame_height, y2))

        person = frame[y1:y2, x1:x2].copy()
        if person.size == 0:
            overlay_state.clear()
            draw_posture_bar(
                annotated,
                preds_video.get("good_posture", 0),
                preds_video.get("bad_posture", 0),
            )
            writer.write(annotated)
            continue

        lm, raw_landmarks = extract_landmarks(person)
        if lm is None:
            overlay_state.clear()
            draw_posture_bar(
                annotated,
                preds_video.get("good_posture", 0),
                preds_video.get("bad_posture", 0),
            )
            writer.write(annotated)
            continue

        if args.refine_with_landmarks and raw_landmarks:
            key_indices = [11, 12, 13, 14, 23, 24, 25, 26]
            xs = [raw_landmarks.landmark[i].x for i in key_indices if 0.0 <= raw_landmarks.landmark[i].x <= 1.0]
            ys = [raw_landmarks.landmark[i].y for i in key_indices if 0.0 <= raw_landmarks.landmark[i].y <= 1.0]
            if xs and ys:
                margin = args.landmark_margin
                min_x = max(0.0, min(xs) - margin)
                max_x = min(1.0, max(xs) + margin)
                min_y = max(0.0, min(ys) - margin)
                max_y = min(1.0, max(ys) + margin)
                if max_x > min_x and max_y > min_y:
                    new_x1 = int(round(x1 + min_x * (x2 - x1)))
                    new_x2 = int(round(x1 + max_x * (x2 - x1)))
                    new_y1 = int(round(y1 + min_y * (y2 - y1)))
                    new_y2 = int(round(y1 + max_y * (y2 - y1)))
                    new_x1 = max(0, min(frame_width - 1, new_x1))
                    new_y1 = max(0, min(frame_height - 1, new_y1))
                    new_x2 = max(new_x1 + 1, min(frame_width, new_x2))
                    new_y2 = max(new_y1 + 1, min(frame_height, new_y2))
                    refined = frame[new_y1:new_y2, new_x1:new_x2].copy()
                    if refined.size != 0:
                        lm_refined, raw_refined = extract_landmarks(refined)
                        if lm_refined is not None:
                            center_x = (new_x1 + new_x2) / 2
                            center_y = (new_y1 + new_y2) / 2
                            target_w = min(
                                max(new_x2 - new_x1, frame_width * args.box_min_frac),
                                frame_width * args.box_max_frac,
                            )
                            target_h = min(
                                max(new_y2 - new_y1, frame_height * args.box_min_frac),
                                frame_height * args.box_max_frac,
                            )
                            x1_candidate = int(round(center_x - target_w / 2))
                            y1_candidate = int(round(center_y - target_h / 2))
                            x2_candidate = int(round(center_x + target_w / 2))
                            y2_candidate = int(round(center_y + target_h / 2))
                            x1_candidate = max(0, min(frame_width - 1, x1_candidate))
                            y1_candidate = max(0, min(frame_height - 1, y1_candidate))
                            x2_candidate = max(x1_candidate + 1, min(frame_width, x2_candidate))
                            y2_candidate = max(y1_candidate + 1, min(frame_height, y2_candidate))
                            refined_expanded = frame[y1_candidate:y2_candidate, x1_candidate:x2_candidate].copy()
                            if refined_expanded.size != 0:
                                person = refined_expanded
                                lm = lm_refined
                                raw_landmarks = raw_refined
                                x1, y1, x2, y2 = x1_candidate, y1_candidate, x2_candidate, y2_candidate
                                box_w = x2 - x1
                                box_h = y2 - y1
                                smoothed_box = (
                                    x1 + box_w / 2,
                                    y1 + box_h / 2,
                                    box_w,
                                    box_h,
                                )

        box_areas.append(((x2 - x1) * (y2 - y1)) / frame_area)

        landmarks_scaled = scaler.transform(lm)
        pred = classifier.predict(landmarks_scaled, verbose=0)
        conf_posture = float(pred[0][0])
        pred_label = "good_posture" if conf_posture > 0.5 else "bad_posture"

        recent_preds.append(pred_label)
        label = Counter(recent_preds).most_common(1)[0][0]

        preds_video[label] += 1
        total_predictions[label] += 1

        color = (0, 255, 0) if label == "good_posture" else (0, 0, 255)
        metrics = draw_pose_overlay(
            person,
            raw_landmarks,
            label,
            orientation,
            offset_override,
            overlay_state,
        )
        angle = metrics.get("spine_tilt")
        annotated[y1:y2, x1:x2] = person

        if metrics.get("tracking_lost"):
            cv2.putText(
                annotated,
                "Subject out of view",
                (50, frame_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 215, 255),
                2,
            )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} ({conf_posture:.2f})",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        results.append(
            {
                "frame": frame_idx,
                "label": label,
                "confidence": conf_posture,
                "tilt_angle": angle,
                "tilt_angle_raw": metrics.get("spine_tilt_raw"),
                "head_tilt_angle": metrics.get("head_tilt"),
                "head_tilt_raw": metrics.get("head_tilt_raw"),
                "spine_direction": metrics.get("spine_direction"),
                "head_direction": metrics.get("head_direction"),
                "yaw_estimate": metrics.get("yaw"),
                "tracking_lost": metrics.get("tracking_lost", False),
                "spine_visible": metrics.get("spine_visible", False),
                "head_visible": metrics.get("head_visible", False),
                "spine_measured": metrics.get("spine_measured", False),
                "head_measured": metrics.get("head_measured", False),
                "spine_drawn": metrics.get("spine_drawn", False),
                "head_drawn": metrics.get("head_drawn", False),
            }
        )

        draw_posture_bar(
            annotated,
            preds_video.get("good_posture", 0),
            preds_video.get("bad_posture", 0),
        )
        writer.write(annotated)

    good, bad, good_pct, bad_pct = save_summary(
        result_dir,
        video_path,
        frame_idx,
        preds_video,
        results,
        fps,
    )

    detection_rate_box = frames_with_box / frame_idx if frame_idx else 0.0
    detection_rate_yolo = frames_with_yolo / frame_idx if frame_idx else 0.0
    avg_box_area = float(np.mean(box_areas)) if box_areas else 0.0
    std_box_area = float(np.std(box_areas)) if box_areas else 0.0

    spine_tilts = [r["tilt_angle"] for r in results if r.get("tilt_angle") is not None]
    head_tilts = [r["head_tilt_angle"] for r in results if r.get("head_tilt_angle") is not None]
    spine_tilts_raw = [r["tilt_angle_raw"] for r in results if r.get("tilt_angle_raw") is not None]
    head_tilts_raw = [r["head_tilt_raw"] for r in results if r.get("head_tilt_raw") is not None]
    yaw_values = [r["yaw_estimate"] for r in results if r.get("yaw_estimate") is not None]
    tracking_lost_frames = sum(1 for r in results if r.get("tracking_lost"))
    tracking_lost_rate = tracking_lost_frames / frame_idx if frame_idx else 0.0

    avg_spine_tilt_signed = float(np.mean(spine_tilts)) if spine_tilts else None
    avg_spine_tilt = float(np.mean(np.abs(spine_tilts))) if spine_tilts else None
    avg_head_tilt_signed = float(np.mean(head_tilts)) if head_tilts else None
    avg_head_tilt = float(np.mean(np.abs(head_tilts))) if head_tilts else None
    avg_spine_tilt_raw = float(np.mean(np.abs(spine_tilts_raw))) if spine_tilts_raw else None
    avg_head_tilt_raw = float(np.mean(np.abs(head_tilts_raw))) if head_tilts_raw else None
    avg_yaw = float(np.mean(yaw_values)) if yaw_values else None

    feedback_lines = build_feedback(avg_spine_tilt, avg_head_tilt)
    if bad_pct > 50:
        feedback_lines.insert(0, f"Poor posture detected in {bad_pct}% of frames.")
    elif bad_pct > 20:
        feedback_lines.insert(0, f"Posture inconsistencies noted ({bad_pct}% bad frames).")
    feedback_text = "\n".join(feedback_lines)

    diagnostics = {
        "video": str(video_path),
        "frames_total": frame_idx,
        "frames_with_box": frames_with_box,
        "frames_with_yolo": frames_with_yolo,
        "detection_rate_box": detection_rate_box,
        "detection_rate_yolo": detection_rate_yolo,
        "avg_box_area_fraction": avg_box_area,
        "std_box_area_fraction": std_box_area,
        "max_consecutive_missing": max_loss_streak,
        "frames_tracking_lost": tracking_lost_frames,
        "tracking_lost_rate": tracking_lost_rate,
        "avg_spine_tilt_deg": avg_spine_tilt,
        "avg_spine_tilt_signed_deg": avg_spine_tilt_signed,
        "avg_spine_tilt_raw_deg": avg_spine_tilt_raw,
        "avg_head_tilt_deg": avg_head_tilt,
        "avg_head_tilt_signed_deg": avg_head_tilt_signed,
        "avg_head_tilt_raw_deg": avg_head_tilt_raw,
        "avg_yaw_deg": avg_yaw,
        "feedback": feedback_lines,
        "params": {
            "yolo_conf": args.yolo_conf,
            "yolo_iou": args.yolo_iou,
            "box_expansion": args.box_expansion,
            "box_min_frac": args.box_min_frac,
            "box_max_frac": args.box_max_frac,
            "ema_alpha": args.ema_alpha,
            "box_clamp": args.box_clamp,
            "use_tracker": args.use_tracker,
            "tracker_blend": args.tracker_blend,
            "max_lost": args.max_lost,
            "refine_with_landmarks": args.refine_with_landmarks,
            "landmark_margin": args.landmark_margin,
            "camera_angle": orientation,
            "camera_offset": offset_override,
        },
    }
    diag_path = result_dir / f"{video_path.stem}_diagnostics.json"
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("[INFO] Inference diagnostics:")
    for key, value in diagnostics.items():
        if key == "params":
            continue
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")
    print(f"[INFO] Diagnostics JSON saved at: {diag_path}")

    feedback_path = result_dir / f"{video_path.stem}_feedback.txt"
    with feedback_path.open("w", encoding="utf-8") as f:
        if feedback_text:
            f.write(feedback_text + "\n")
    print(f"[INFO] Adaptive feedback saved at: {feedback_path}")

    summary = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    cv2.putText(
        summary,
        f"Posture Summary - {video_path.name}",
        (50, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        summary,
        f"Frames: {frame_idx}",
        (50, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        summary,
        f"Good: {good} ({good_pct}%)",
        (50, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        summary,
        f"Bad:  {bad} ({bad_pct}%)",
        (50, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
    )

    feedback_y = 260
    for line in feedback_lines:
        for chunk in textwrap.wrap(line, width=48) or [""]:
            cv2.putText(
                summary,
                chunk,
                (50, feedback_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 0),
                2,
            )
            feedback_y += 35

    for _ in range(int(fps * 3)):
        writer.write(summary)

    cap.release()
    writer.release()

    print(f"[DONE] Annotated video saved -> {output_path}")
    print(f"[INFO] Frames processed: {frame_idx}")
    print(f"[INFO] Posture summary: {preds_video}")
    print("\n--- Inference Complete ---\n")


if __name__ == "__main__":
    run_inference(parse_args())
