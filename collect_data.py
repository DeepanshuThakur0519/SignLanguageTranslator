"""
collect_data.py
----------------
Run this LOCALLY (on a machine with a webcam) to build your training
dataset. It captures live frames, extracts the 42-D normalized hand-landmark
feature vector for each frame, and appends labeled rows to
backend/data/landmarks.csv.

Usage:
    python collect_data.py --label A --samples 300
    python collect_data.py --label B --samples 300
    ... repeat for every gesture/letter/word you want to recognize ...

Controls while running:
    's'  -> start/pause capturing samples for the current label
    'q'  -> quit
"""

import argparse
import csv
import os
import sys
import time

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from hand_utils import create_hands_detector, landmarks_to_feature  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")


def ensure_csv_header():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(42)]
            writer.writerow(header)


def main():
    parser = argparse.ArgumentParser(description="Collect labeled hand-gesture landmark data.")
    parser.add_argument("--label", required=True, help="Gesture/class label, e.g. A, B, HELLO")
    parser.add_argument("--samples", type=int, default=300, help="Number of samples to capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    args = parser.parse_args()

    ensure_csv_header()

    cap = cv2.VideoCapture(args.camera)
    hands = create_hands_detector(static_image_mode=False, max_num_hands=1)

    collected = 0
    capturing = False
    print(f"[INFO] Ready to collect '{args.label}'. Press 's' to start/pause, 'q' to quit.")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        while cap.isOpened() and collected < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            feature = landmarks_to_feature(results)

            if results.multi_hand_landmarks:
                import mediapipe as mp
                mp_drawing = mp.solutions.drawing_utils
                mp_hands = mp.solutions.hands
                mp_drawing.draw_landmarks(
                    frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

            if capturing and feature is not None:
                writer.writerow([args.label] + feature.tolist())
                collected += 1
                time.sleep(0.03)  # slight delay for pose variety between samples

            status = "CAPTURING" if capturing else "PAUSED"
            cv2.putText(frame, f"Label: {args.label}  [{status}]  {collected}/{args.samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Data Collection - Sign Language Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                capturing = not capturing

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Collected {collected} samples for label '{args.label}'. Saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
