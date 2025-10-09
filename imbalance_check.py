import pandas as pd

# Path to your generated CSV file
csv_path = "landmarks_dataset.csv"

# Load the dataset
df = pd.read_csv(csv_path)

print("\n🔍 Checking dataset balance...")

# Show total number of samples
print(f"Total samples: {len(df)}")

# Count labels
print("\nClass counts:")
print(df["label"].value_counts())

# Check unique labels and class names
print("\nUnique numeric labels:", df["label"].unique())
print("Unique class names:", df["class_name"].unique())

# Verify train/valid split distribution
print("\nSplit distribution:")
print(df["split"].value_counts())

# Warn if one class is missing
if len(df["label"].unique()) < 2:
    print("\n⚠️ WARNING: Only one class found in dataset! Please check your folder structure:")
    print("   - yolov5/Posture_Dataset/train/bad_posture/")
    print("   - yolov5/Posture_Dataset/train/good_posture/")
    print("   - yolov5/Posture_Dataset/valid/bad_posture/")
    print("   - yolov5/Posture_Dataset/valid/good_posture/")
else:
    print("\n✅ Both classes detected. Dataset looks good for training!")
