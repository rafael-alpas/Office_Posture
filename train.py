import os # Kept for os.makedirs
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping
from collections import Counter

# --- STAGE 2: TensorFlow Posture Classifier Training ---
print("\n--- Starting TensorFlow Posture Classifier Training & Scaler Generation ---")

# Load the pre-extracted landmark data
try:
    df = pd.read_csv("landmarks_dataset.csv")
except FileNotFoundError:
    print("❌ Error: 'landmarks_dataset.csv' not found. Please run the data extraction step first.")
    exit()

if df.empty:
    print("❌ Error: 'landmarks_dataset.csv' is empty. Data extraction failed.")
    exit()

# Prepare data
# Drop non-feature columns
X = df.drop(columns=["split", "class_name", "label", "filename"]).values
y = df["label"].values  # 0 = bad posture, 1 = good posture

print(f"Data shape: {X.shape}")
print("Label distribution:", Counter(y))

# --- Feature Scaling and Saving Scaler ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler_path = "runs/train/posture_model/scaler.pkl"
os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
joblib.dump(scaler, scaler_path)
print(f"✅ StandardScaler fitted and saved to {scaler_path}")

# --- Data Splitting ---
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# --- Compute Class Weights for Imbalanced Data ---
class_weights = dict(
    zip(np.unique(y), compute_class_weight('balanced', classes=np.unique(y), y=y))
)
print("Class weights for training:", class_weights)

# --- Build Model ---
def build_posture_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Callbacks ---
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "runs/train/posture_model/posture_classifier_best.keras",
    monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# --- Train Model ---
posture_model = build_posture_model(X_train.shape[1])
print("\nStarting Posture Classifier Training...")

history = posture_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb],
    class_weight=class_weights
)

# --- Save final model ---
final_model_path = "runs/train/posture_model/posture_classifier_final.keras"
posture_model.save(final_model_path)

print("\n✅ TensorFlow posture classifier training complete!")
print("Model artifacts saved in runs/train/posture_model/")
print(f"   Final model: {final_model_path}")
print(f"   Best checkpoint: runs/train/posture_model/posture_classifier_best.keras")
print(f"   StandardScaler (for inference) saved at: {scaler_path}")