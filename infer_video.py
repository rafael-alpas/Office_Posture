import argparse
import json
import warnings
from collections import Counter, deque
from pathlib import Path
import math

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


def draw_pose_overlay(image, landmarks, label):
    """Overlay key joints and spine tilt vector for visual feedback."""
    h, w, _ = image.shape
    pts = landmarks.landmark

    def point(idx):
        return (int(pts[idx].x * w), int(pts[idx].y * h))

    for idx in [11, 12, 23, 24]:
        cv2.circle(image, point(idx), 5, (0, 255, 255), -1)

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
    cv2.putText(
        image,
        f"Tilt: {angle:.1f} deg {direction}",
        (start[0], start[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    return angle


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

    print("\n--- Starting Inference ---\n")
    # Keep a short history of predicted labels to debounce frame-by-frame noise.
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

    # Tracker state is optional and only used when --use-tracker is enabled.
    tracker = None
    tracking = False
    reinit_interval = int(fps * args.tracker_reinit_secs) if args.tracker_reinit_secs > 0 else 0
    smoothed_box = None
    lost_frames = 0
    frame_idx = 0
    results = []
    # Diagnostics helpers capture detection coverage and box stability.
    frame_area = frame_width * frame_height
    frames_with_box = 0
    frames_with_yolo = 0
    box_areas = []
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

        # EMA keeps boxes steady while still reacting slowly to new detections.
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
            draw_posture_bar(
                annotated,
                preds_video.get("good_posture", 0),
                preds_video.get("bad_posture", 0),
            )
            writer.write(annotated)
            continue

        lm, raw_landmarks = extract_landmarks(person)
        if lm is None:
            draw_posture_bar(
                annotated,
                preds_video.get("good_posture", 0),
                preds_video.get("bad_posture", 0),
            )
            writer.write(annotated)
            continue

        if args.refine_with_landmarks and raw_landmarks:
            # Use shoulders/hips/knees landmarks to tighten the crop around the torso.
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
        angle = draw_pose_overlay(person, raw_landmarks, label)
        annotated[y1:y2, x1:x2] = person

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

    tip = (
        "Poor posture detected often, please adjust!" if bad_pct > 50
        else "Mostly good posture!" if bad_pct > 20
        else "Excellent posture maintained!"
    )
    cv2.putText(
        summary,
        tip,
        (50, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2,
    )

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
