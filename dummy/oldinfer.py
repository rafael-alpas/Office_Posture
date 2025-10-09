# infer_video.py
import os
import cv2
import torch
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import Counter

# ✅ Load YOLOv5 model (person detection)
yolo_model = torch.hub.load("ultralytics/yolov5", "custom",
                             path="runs/train/posture_model/weights/last.pt")
yolo_model.conf = 0.25  # detection confidence threshold

# ✅ Load TensorFlow posture classifier
posture_model = tf.keras.models.load_model(
    "runs/train/posture_model/posture_classifier_best.h5"
)

# ✅ MediaPipe setup
mp_pose = mp.solutions.pose
# NOTE: min_tracking_confidence is already lowered to 0.3 for better tracking
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image, bbox, padding=50):
    """
    Extract pose landmarks inside a bounding box with added padding 
    and return flat vector along with the final padded coordinates.
    
    Args:
        image (np.ndarray): The full video frame.
        bbox (tuple): (x1, y1, x2, y2) of the detected person (YOLO output).
        padding (int): Pixels to expand the ROI beyond the detected bbox.
        
    Returns:
        tuple: (landmarks_vector, landmark_object, (final_x1, final_y1, final_x2, final_y2)) 
               or (None, None, None).
    """
    
    x1, y1, x2, y2 = map(int, bbox)
    h, w, _ = image.shape
    
    # 1. Apply padding to the raw box coordinates
    x1_padded = x1 - padding
    y1_padded = y1 - padding
    x2_padded = x2 + padding
    y2_padded = y2 + padding
    
    # 2. Clip the padded coordinates to ensure they are within the image dimensions
    final_x1 = max(0, x1_padded)
    final_y1 = max(0, y1_padded)
    final_x2 = min(w, x2_padded)
    final_y2 = min(h, y2_padded)
    
    # 3. Check for invalid slice after clipping (e.g., if the initial bbox was too small/bad)
    if final_x1 >= final_x2 or final_y1 >= final_y2:
        return None, None, None
    
    # Define the ROI using the final, clipped and padded coordinates
    roi = image[final_y1:final_y2, final_x1:final_x2]
    
    # Check for empty ROI
    if roi.size == 0:
        return None, None, None
        
    # --- MediaPipe Processing ---
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_roi)
    
    if results.pose_landmarks:
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        
        # Return the landmarks, landmark object, and the final padded coordinates
        return np.array(row).reshape(1, -1), results.pose_landmarks, (final_x1, final_y1, final_x2, final_y2)
    
    return None, None, None


def run_inference(video_path, output_path="runs/test/test1_posture_inference.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    os.makedirs("runs/test", exist_ok=True)
    
    # Capture video properties for writer and robust summary frame creation
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, frame_fps,
                          (frame_width, frame_height))

    frame_count = 0
    predictions = []
    
    # Initialize a dummy frame to ensure 'frame' exists for post-loop logic
    frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # YOLO person detection
        results = yolo_model(frame)
        detections = results.xyxy[0].cpu().numpy()

        for det in detections:
            x1_yolo, y1_yolo, x2_yolo, y2_yolo, conf, cls = det
            if int(cls) != 0:  # only person
                continue
                
            # Use the original YOLO detection box coordinates as input
            yolo_bbox = (x1_yolo, y1_yolo, x2_yolo, y2_yolo)
            
            # Extract landmarks. Function now returns 3 values (landmarks, obj, final_bbox_coords)
            landmarks, landmark_obj, final_bbox = extract_landmarks(frame, yolo_bbox)
            
            if landmarks is not None and landmark_obj is not None:
                
                # Unpack the final padded and clipped coordinates for drawing
                final_x1, final_y1, final_x2, final_y2 = final_bbox

                # Predict posture
                pred = posture_model.predict(landmarks, verbose=0)
                label_idx = np.argmax(pred)
                label = "Good Posture" if label_idx == 1 else "Bad Posture"
                predictions.append(label)

                # Draw bounding box + label (use the ORIGINAL YOLO box for the visual box)
                color = (0, 255, 0) if label == "Good Posture" else (0, 0, 255)
                # Note: x1_yolo, y1_yolo etc. are floats but cv2.rectangle handles them fine
                cv2.rectangle(frame, (int(x1_yolo), int(y1_yolo)),
                              (int(x2_yolo), int(y2_yolo)), color, 2)
                cv2.putText(frame, f"{label} ({pred.max():.2f})",
                            (int(x1_yolo), int(y1_yolo) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Draw pose landmarks inside the final (padded) ROI
                mp_drawing.draw_landmarks(frame[final_y1:final_y2, final_x1:final_x2],
                                         landmark_obj,
                                         mp_pose.POSE_CONNECTIONS)

        out.write(frame)
    
    # Ensure there was at least one frame to generate a summary
    if frame_count == 0:
        print("❌ No frames were read from the video. Cannot generate summary.")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return

    # ✅ Summary Logic
    summary_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    summary = Counter(predictions)
    
    # Define all expected labels and their colors for guaranteed display
    posture_stats = {
        "Good Posture": (0, 255, 0),  # Green (BGR)
        "Bad Posture": (0, 0, 255)   # Red (BGR)
    }

    # Drawing Summary Text
    cv2.putText(summary_frame, "📊 Posture Analysis Summary", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    y_offset = 200
    
    # Loop through the defined labels and use .get() to safely retrieve the count, defaulting to 0
    for label, color in posture_stats.items():
        count = summary.get(label, 0) 
        
        cv2.putText(summary_frame, f"{label}: {count} frames",
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)
        y_offset += 60
        
    cv2.putText(summary_frame,
                f"Total frames processed: {frame_count}",
                (50, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2)

    # Write summary frame to the output video
    for _ in range(int(frame_fps * 2)):  # show summary for ~2 seconds
        out.write(summary_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n📊 Posture Analysis Summary:")
    # Fix the printout summary for consistency
    for label in posture_stats.keys():
        print(f"{label}: {summary.get(label, 0)} frames")
    print(f"Total frames processed: {frame_count}")
    print(f"✅ Saved output video to: {output_path}")

if __name__ == "__main__":
    run_inference("yolov5/Posture_Dataset/test/test_posture_1 - Trim.mp4")