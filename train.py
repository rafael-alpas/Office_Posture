import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# Default arguments keep behaviour identical to the original scripts.
DEFAULT_DATA_PATH = "landmarks_dataset.csv"
DEFAULT_MODEL_DIR = "runs/train/posture_model"


def build_posture_model(input_shape: int) -> tf.keras.Model:
    """Create the dense neural network used for posture classification."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def parse_args():
    """Collect CLI parameters so we can train on alternate datasets/dirs."""
    parser = argparse.ArgumentParser(
        description="Train the posture classification model on extracted landmarks."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="CSV file produced by landmark_extraction.py (default: %(default)s)",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory where scaler and trained models will be saved (default: %(default)s)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation split (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    # Ensure the target directory exists before we start writing artifacts.
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting TensorFlow Posture Classifier Training & Scaler Generation ---")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[ERROR] '{data_path}' not found. Please run landmark_extraction.py first.")
        return

    if df.empty:
        print(f"[ERROR] '{data_path}' is empty. Data extraction failed.")
        return

    # Drop metadata columns and keep only landmark features for training.
    feature_cols = [col for col in df.columns if col not in {"split", "class_name", "label", "filename"}]
    X = df[feature_cols].values
    y = df["label"].values  # 0 = bad posture, 1 = good posture

    print(f"[INFO] Data shape: {X.shape}")
    print(f"[INFO] Label distribution: {Counter(y)}")

    # Fit a scaler for inference-time normalisation and persist it alongside the model.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[INFO] StandardScaler fitted and saved to {scaler_path}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=args.val_split,
        stratify=y,
        random_state=args.seed,
    )
    print(f"[INFO] Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # Balance the class weights so the network treats both classes evenly.
    class_weights = dict(
        zip(np.unique(y), compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y))
    )
    print(f"[INFO] Class weights for training: {class_weights}")

    # Standard callbacks: TensorBoard logging, best-checkpoint saving, and early stopping.
    log_dir = Path("logs/fit") / datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    checkpoint_path = model_dir / "posture_classifier_best.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)

    posture_model = build_posture_model(X_train.shape[1])
    print("\n[INFO] Starting posture classifier training...")

    posture_model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb],
        class_weight=class_weights,
        verbose=1,
    )

    # Evaluate on the held-out validation split to capture reproducible metrics.
    y_val_probs = posture_model.predict(X_val, verbose=0).ravel()
    y_val_pred = (y_val_probs >= 0.5).astype(int)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="macro", zero_division=0
    )
    precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_val, y_val_pred)
    balanced_acc = balanced_accuracy_score(y_val, y_val_pred)

    metrics_dict = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_positive": float(precision_bin),
        "recall_positive": float(recall_bin),
        "f1_positive": float(f1_bin),
    }

    print("\n[INFO] Validation metrics:")
    for key, value in metrics_dict.items():
        print(f"  - {key}: {value:.4f}")

    final_model_path = model_dir / "posture_classifier_final.keras"
    posture_model.save(final_model_path)

    # Persist metrics artefacts for future comparisons.
    metrics_path = model_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    report_text = classification_report(
        y_val, y_val_pred, target_names=["bad_posture", "good_posture"], zero_division=0
    )
    report_path = model_dir / "classification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_val, y_val_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["bad", "good"])
    ax.set_yticklabels(["bad", "good"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cm_path = model_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    # Append a structured entry to the project changelog for auditability.
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.exists():
        changelog_path.write_text("# Changelog\n\n", encoding="utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    changelog_entry = (
        f"## {timestamp} - Training Run\n"
        f"- data: {data_path}\n"
        f"- change: train.py auto-run\n"
        f"- hypothesis: n/a (auto)\n"
        f"- metrics: accuracy={metrics_dict['accuracy']:.3f}, "
        f"macro_f1={metrics_dict['f1_macro']:.3f}, "
        f"balanced_accuracy={metrics_dict['balanced_accuracy']:.3f}\n"
        f"- decision: keep\n\n"
    )
    with changelog_path.open("a", encoding="utf-8") as f:
        f.write(changelog_entry)

    print("\n[INFO] TensorFlow posture classifier training complete!")
    print(f"[INFO] Model artifacts saved in {model_dir}")
    print(f"[INFO] Final model: {final_model_path}")
    print(f"[INFO] Best checkpoint: {checkpoint_path}")
    print(f"[INFO] StandardScaler saved at: {scaler_path}")
    print(f"[INFO] Metrics JSON saved at: {metrics_path}")
    print(f"[INFO] Classification report saved at: {report_path}")
    print(f"[INFO] Confusion matrix plot saved at: {cm_path}")


if __name__ == "__main__":
    main()
