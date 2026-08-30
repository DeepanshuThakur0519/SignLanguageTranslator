"""
train_model.py
---------------
Trains a Random Forest classifier (per synopsis section 2/3) on the
42-D normalized landmark feature vectors collected via collect_data.py,
and saves the trained model + label encoder for use by the Flask backend.

Usage:
    python train_model.py
    python train_model.py --n-estimators 300 --test-size 0.2
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def main():
    parser = argparse.ArgumentParser(description="Train the gesture classifier.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"No dataset found at {CSV_PATH}. Run collect_data.py for each gesture first."
        )

    df = pd.read_csv(CSV_PATH)
    if df.empty or df["label"].nunique() < 2:
        raise ValueError("Need at least 2 distinct gesture labels with samples to train a classifier.")

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Test accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")
    print(f"[INFO] Label encoder saved to {ENCODER_PATH}")


if __name__ == "__main__":
    main()
