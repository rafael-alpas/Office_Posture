# train.py
import os
import tensorflow as tf
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

# Train YOLOv5 (person detection only)
os.system(
    "python yolov5/train.py "
    "--img 512 --batch 6 --epochs 100 "
    "--data yolov5/Posture_Dataset/data.yaml "
    "--weights yolov5s.pt "
    "--project runs/train --name posture_model "
    "--nosave"
)

print("\n✅ YOLOv5 training complete! Check runs/train/posture_model/weights/best.pt")

# Train TensorFlow posture classifier using landmark dataset
# Load landmark dataset
df = pd.read_csv("landmarks_dataset.csv")

#   Drop metadata columns
X = df.drop(columns=["split", "class_name", "label", "filename"]).values
y = df["label"].values  # 0 = bad posture, 1 = good posture

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define TensorFlow model
def build_posture_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),  # 132 (33 landmarks × 4 values)
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # ['bad posture', 'good posture']
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# TensorBoard logging
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Checkpoint saving (best model)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "runs/train/posture_classifier_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# Train classifier
posture_model = build_posture_model()
posture_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_cb, checkpoint_cb]
)

# Save final model
os.makedirs("runs/train/posture_model", exist_ok=True)
posture_model.save("runs/train/posture_model/posture_classifier_final.h5")

print("\n✅ TensorFlow posture classifier saved in runs/train/posture_model/")
print("\n✅ YOLOv5 weights available at runs/train/posture_model/weights/best.pt")
