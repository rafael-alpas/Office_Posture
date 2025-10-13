import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Default locations keep parity with the original workflow.
DEFAULT_DATASET_ROOT = "yolov5/Posture_Dataset"
DEFAULT_OUTPUT = "landmarks_dataset.csv"

# Reuse a single static-image pose estimator to avoid repeated initialisation.
mp_pose = mp.solutions.pose
POSE_STATIC = mp_pose.Pose(static_image_mode=True)


def extract_landmarks(image_path: Path):
    """
    Run MediaPipe Pose on one image and return the flat landmark vector.
    Returns None when the pose is not detected or the file cannot be read.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Skipping unreadable image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = POSE_STATIC.process(image_rgb)

    if not results.pose_landmarks:
        return None

    row = []
    for landmark in results.pose_landmarks.landmark:
        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return row


def process_dataset(dataset_root: Path, output_csv: Path):
    """
    Traverse a YOLO-style dataset and extract pose landmarks for each image.
    """
    # Convert incoming paths to Path objects for convenience.
    dataset_root = Path(dataset_root)
    output_csv = Path(output_csv)

    # Process train/valid splits; test is inference-only.
    splits = ["train", "valid"]
    data = []
    seen_files = set()

    for split in splits:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[WARN] Missing {split} folder structure (expected images/ & labels/). Skipping...")
            continue

        for image_path in tqdm(sorted(img_dir.glob("*")), desc=f"Processing {split} set"):
            # Skip non-image files (e.g., caches).
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            label_path = lbl_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                print(f"[WARN] Missing label for {image_path.name}, skipping...")
                continue

            if image_path.name in seen_files:
                print(f"[WARN] Duplicate file skipped: {image_path.name}")
                continue
            seen_files.add(image_path.name)

            try:
                with label_path.open("r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    print(f"[WARN] Empty label file: {image_path.name}")
                    continue
                class_id = int(lines[0].split()[0])
                if class_id not in (0, 1):
                    print(f"[WARN] Unexpected class ID ({class_id}) in {image_path.name}, skipping...")
                    continue
            except Exception as exc:
                print(f"[WARN] Error reading label for {image_path.name}: {exc}")
                continue

            class_name = "bad_posture" if class_id == 0 else "good_posture"
            landmarks = extract_landmarks(image_path)
            if landmarks:
                data.append([split, class_name, class_id, image_path.name] + landmarks)
            else:
                print(f"[WARN] No landmarks found in {image_path.name}, skipping...")

    # Prepare the header: metadata columns + 33 (x, y, z, visibility) blocks.
    columns = ["split", "class_name", "label", "filename"]
    for idx in range(33):
        columns += [f"x{idx}", f"y{idx}", f"z{idx}", f"v{idx}"]

    df = pd.DataFrame(data, columns=columns)
    if df.empty:
        print("[ERROR] No data extracted! Please check dataset paths and Mediapipe setup.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved landmark dataset to {output_csv}")
    print(f"[INFO] Shape: {df.shape}")
    print("[INFO] Class distribution:")
    print(df["class_name"].value_counts())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose landmarks from a YOLO-format dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing train/valid folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="CSV file path for extracted landmarks (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataset(Path(args.dataset_root), Path(args.output))
