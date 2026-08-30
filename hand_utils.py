"""
hand_utils.py
--------------
Shared utilities for the Sign Language Translator project.

Implements:
  - MediaPipe Hands wrapper for 21-point 3D landmark extraction
  - Wrist-relative, bounding-box-normalized feature vectorization
    (matches the x_i' = (x_i - x_0) / W_bbox  formula from the synopsis)
  - A simple temporal debounce buffer to stabilize predictions across frames
"""

import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# MediaPipe Hands setup
# ---------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

NUM_LANDMARKS = 21          # MediaPipe Hands always returns 21 keypoints
FEATURE_DIM = NUM_LANDMARKS * 2  # 42-D normalized (x, y) vector, per synopsis


def create_hands_detector(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.5):
    """Factory for a configured MediaPipe Hands detector instance."""
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def extract_landmarks(results):
    """
    Given a MediaPipe `results` object (from hands.process(frame)),
    return the first detected hand's 21 landmarks as an (21, 3) array
    of (x, y, z) in normalized image coordinates, or None if no hand found.
    """
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
    return pts


def normalize_landmarks(points_xy):
    """
    Convert raw (21, 2) [or (21,3), z ignored] landmark array into a
    42-D wrist-relative, bounding-box-scaled feature vector.

    x_i' = (x_i - x0) / W_bbox
    y_i' = (y_i - y0) / H_bbox

    where (x0, y0) is the wrist landmark (index 0) and W_bbox/H_bbox are the
    width/height of the hand's bounding box. This makes the feature vector
    invariant to hand position and (roughly) to distance from the camera.
    """
    pts = np.asarray(points_xy, dtype=np.float32)[:, :2]  # drop z if present

    x0, y0 = pts[0]
    xs, ys = pts[:, 0], pts[:, 1]

    w_bbox = max(xs.max() - xs.min(), 1e-6)
    h_bbox = max(ys.max() - ys.min(), 1e-6)

    x_norm = (xs - x0) / w_bbox
    y_norm = (ys - y0) / h_bbox

    feature = np.empty(FEATURE_DIM, dtype=np.float32)
    feature[0::2] = x_norm
    feature[1::2] = y_norm
    return feature


def landmarks_to_feature(results):
    """Convenience: MediaPipe results -> 42-D normalized feature vector, or None."""
    pts = extract_landmarks(results)
    if pts is None:
        return None
    return normalize_landmarks(pts)


class DebounceBuffer:
    """
    Temporal debouncing buffer, matching the synopsis' "Debounce Buffer /
    Temporal Frame Stabilization" pipeline stage.

    Keeps a rolling window of the last `size` predictions and only emits a
    stable output once the same label has appeared at least `min_agree`
    times in that window, with confidence above `min_confidence`. This
    filters out flicker caused by transient/transitional hand poses.
    """

    def __init__(self, size=8, min_agree=5, min_confidence=0.6):
        self.size = size
        self.min_agree = min_agree
        self.min_confidence = min_confidence
        self.buffer = []          # list of (label, confidence)
        self.last_emitted = None

    def push(self, label, confidence):
        self.buffer.append((label, confidence))
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        return self.stable_label()

    def stable_label(self):
        if not self.buffer:
            return None
        labels = [l for l, c in self.buffer if c >= self.min_confidence]
        if not labels:
            return None
        # majority vote
        values, counts = np.unique(labels, return_counts=True)
        best_idx = np.argmax(counts)
        best_label, best_count = values[best_idx], counts[best_idx]
        if best_count >= self.min_agree:
            if best_label != self.last_emitted:
                self.last_emitted = best_label
                return best_label
            return None  # already emitted this label, avoid repeats
        return None

    def reset(self):
        self.buffer.clear()
        self.last_emitted = None
