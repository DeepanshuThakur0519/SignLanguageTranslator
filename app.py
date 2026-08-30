"""
app.py
------
Flask backend for the Sign Language Translator.

Pipeline (matches synopsis Figure 1):
  Webcam frame (base64, from browser)
    -> MediaPipe Extraction (21 hand keypoints)
    -> Feature Vectorization (42-D normalized vector)
    -> ML Classifier Model (Random Forest)
    -> Debounce Buffer (temporal stabilization, per session)
    -> Output (text; audio handled client-side via Web Speech API)

Run:
    python app.py
Then open http://localhost:5000 in a browser.
"""

import base64
import os
import uuid

import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from hand_utils import create_hands_detector, landmarks_to_feature, DebounceBuffer

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# --- Load model (if it exists) -------------------------------------------------
model = None
label_encoder = None
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("[INFO] Loaded trained gesture model.")
else:
    print("[WARN] No trained model found at backend/model/gesture_model.pkl.")
    print("       Run collect_data.py + train_model.py first, or use /predict in demo mode.")

# One MediaPipe Hands detector instance reused across requests (single process/thread).
hands_detector = create_hands_detector(static_image_mode=True, max_num_hands=1)

# Per-session debounce buffers, keyed by a client-provided session id.
session_buffers = {}


def decode_base64_image(data_url):
    """Decode a 'data:image/jpeg;base64,...' string into a BGR numpy image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": list(label_encoder.classes_) if label_encoder is not None else [],
    })


@app.route("/api/session", methods=["POST"])
def new_session():
    session_id = str(uuid.uuid4())
    session_buffers[session_id] = DebounceBuffer(size=8, min_agree=5, min_confidence=0.6)
    return jsonify({"session_id": session_id})


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    image_data = payload.get("image")
    session_id = payload.get("session_id")

    if not image_data:
        return jsonify({"error": "No image provided"}), 400

    try:
        frame = decode_base64_image(image_data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not decode image: {exc}"}), 400

    if frame is None:
        return jsonify({"error": "Empty frame"}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    feature = landmarks_to_feature(results)

    if feature is None:
        return jsonify({"hand_detected": False, "label": None, "confidence": 0.0, "stable_label": None})

    if model is None:
        # Demo mode: no trained model yet, just confirm hand detection.
        return jsonify({
            "hand_detected": True,
            "label": None,
            "confidence": 0.0,
            "stable_label": None,
            "message": "Hand detected, but no trained model is loaded yet.",
        })

    probs = model.predict_proba([feature])[0]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    label = label_encoder.inverse_transform([best_idx])[0]

    stable_label = None
    if session_id and session_id in session_buffers:
        stable_label = session_buffers[session_id].push(label, confidence)

    return jsonify({
        "hand_detected": True,
        "label": str(label),
        "confidence": confidence,
        "stable_label": stable_label,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
