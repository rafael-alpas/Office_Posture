import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_landmarks(image_path):
    """Extracts 33 pose landmarks (x,y,z,visibility) from an image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipping unreadable image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
        return row
    return None


def process_dataset(dataset_root, output_csv):
    """
    Reads YOLO dataset structure:
    dataset_root/
        train/
            images/
            labels/
        valid/
            images/
            labels/
    """
    data = []
    seen_files = set()

    for split in ["train", "valid"]:
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"⚠️ Missing {split} folder structure (expected images/ & labels/). Skipping...")
            continue

        for fname in tqdm(os.listdir(img_dir), desc=f"Processing {split} set"):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(img_dir, fname)
            label_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

            if not os.path.exists(label_path):
                print(f"⚠️ Missing label for {fname}, skipping...")
                continue

            if fname in seen_files:
                print(f"⚠️ Duplicate file skipped: {fname}")
                continue
            seen_files.add(fname)

            try:
                with open(label_path, "r") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                if not lines:
                    print(f"⚠️ Empty label file: {fname}")
                    continue
                class_id = int(lines[0].split()[0])
                if class_id not in [0, 1]:
                    print(f"⚠️ Unexpected class ID ({class_id}) in {fname}, skipping...")
                    continue
            except Exception as e:
                print(f"⚠️ Error reading label for {fname}: {e}")
                continue

            class_name = "bad_posture" if class_id == 0 else "good_posture"
            landmarks = extract_landmarks(image_path)
            if landmarks:
                data.append([split, class_name, class_id, fname] + landmarks)
            else:
                print(f"⚠️ No landmarks found in {fname}, skipping...")

    # Build dataframe
    cols = ["split", "class_name", "label", "filename"]
    for i in range(33):
        cols += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]

    df = pd.DataFrame(data, columns=cols)

    if df.empty:
        print("❌ No data extracted! Please check dataset paths and Mediapipe setup.")
        return

    # Save cleaned dataframe
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved landmark dataset to {output_csv}")
    print(f"📊 Shape: {df.shape}")
    print("🔍 Class distribution:")
    print(df['class_name'].value_counts())


# Example usage
if __name__ == "__main__":
    process_dataset("yolov5/Posture_Dataset", "landmarks_dataset.csv")
