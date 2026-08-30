# 🤟 Sign Language Translator

A real-time sign language translator that converts hand gestures captured from
a webcam into text and synthesized speech — built for the **Mini Project
(BCC 351), B.Tech CSE III Semester (AY 2026-27)**.

This is a working prototype implementing the architecture described in the
project synopsis:

```
Webcam Input → MediaPipe Extraction (21 keypoints) → Feature Vectorization
(42-D normalized vector) → ML Classifier (Random Forest) → Debounce Buffer
(temporal stabilization) → Output Synthesis (text + voice)
```

## Features

- **Real-time hand tracking** using Google MediaPipe Hands (21 landmarks/frame)
- **Scale & position invariant features** — each landmark is normalized
  relative to the wrist and the hand's bounding box, exactly as specified in
  the synopsis:
  `x_i' = (x_i - x0) / W_bbox`, `y_i' = (y_i - y0) / H_bbox`
- **Random Forest classifier** (scikit-learn) trained on your own gesture data
- **Temporal debounce buffer** to prevent flickery / false-positive output
- **Browser-based frontend** — webcam capture, live prediction display,
  sentence builder, and text-to-speech via the Web Speech API
- **Flask REST API backend** connecting the two

## Project Structure

```
sign-language-translator/
├── backend/
│   ├── app.py                # Flask server + /api/predict endpoint
│   ├── hand_utils.py          # MediaPipe wrapper, normalization, debounce buffer
│   ├── data/
│   │   └── landmarks.csv      # your collected training data (generated)
│   └── model/
│       ├── collect_data.py    # webcam script to record labeled gestures
│       ├── train_model.py     # trains + saves the Random Forest model
│       ├── gesture_model.pkl  # trained model (generated)
│       └── label_encoder.pkl  # label encoder (generated)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd sign-language-translator
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+ and a working webcam.

### 2. Collect training data

Run this once **per gesture/letter/word** you want the system to recognize
(e.g. the letters A–Z, or words like HELLO, THANKS, YES, NO):

```bash
cd backend/model
python collect_data.py --label A --samples 300
python collect_data.py --label B --samples 300
python collect_data.py --label HELLO --samples 300
# ...repeat for each gesture
```

Controls: press **`s`** to start/pause capturing, **`q`** to quit. Vary your
hand angle and distance slightly while capturing for a more robust model.
Samples are appended to `backend/data/landmarks.csv`.

### 3. Train the classifier

```bash
python train_model.py
```

This trains a Random Forest on your collected landmarks, prints accuracy /
classification report, and saves `gesture_model.pkl` + `label_encoder.pkl`
into `backend/model/`.

### 4. Run the app

```bash
cd ../..            # back to backend/ from model/, or project root
cd backend
python app.py
```

Open **http://localhost:5000** in your browser, click **Start Camera**, and
allow webcam access. Show a trained gesture — the predicted letter/word
appears live, gets spoken aloud (toggle-able), and accumulates into a
sentence you can edit (space / backspace / clear) and re-speak.

## How it works

1. The browser captures webcam frames and POSTs them as base64 JPEGs to
   `/api/predict`.
2. The backend decodes the frame, runs MediaPipe Hands to extract 21 3D
   landmarks, and normalizes them into a 42-D feature vector relative to the
   wrist and hand bounding box (`hand_utils.py`).
3. The Random Forest classifier predicts a label + confidence.
4. A per-session `DebounceBuffer` requires the same label to appear
   consistently across several consecutive frames above a confidence
   threshold before it's "emitted" — this filters out noisy/transitional
   frames.
5. Emitted labels are appended to the sentence in the UI and spoken via the
   browser's `SpeechSynthesis` API.

## Extending this prototype

- Swap Random Forest for an MLP (`sklearn.neural_network.MLPClassifier`) —
  both were listed as candidate classifiers in the synopsis.
- Add two-hand support (`max_num_hands=2`) and gesture sequences for
  dynamic (motion-based) signs rather than static poses.
- Deploy the Flask backend behind `gunicorn`/`nginx` for production use.
- Add user accounts / history if this becomes a persistent assistive tool.

## Hardware & Software Requirements (per synopsis)

| Category | Requirement |
|---|---|
| CPU | Intel Core i3 / AMD Ryzen 3 or higher (2.0 GHz base clock) |
| RAM | 4 GB (8 GB recommended) |
| Camera | Integrated/external HD USB webcam, 720p @ 30fps |
| OS | Windows 10/11, Linux (Ubuntu 20.04+), or macOS |
| Language | Python 3.9+ |
| CV/AI libs | OpenCV 4.8+, MediaPipe 0.10+, scikit-learn 1.3+ |
| Other | Flask, NumPy, Pandas, Matplotlib |

## References

- Zhang, F., Bazarevsky, V., Vakunov, A., et al. (2020). *MediaPipe Hands:
  On-device Real-time Hand Tracking.* arXiv:2006.10214.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.*
  JMLR, 12, 2825-2830.
- Rastgoo, R., Kiani, K., & Escalera, S. (2021). *Real-time American Sign
  Language Recognition using Deep Learning and Hand Keypoints.*
  Neurocomputing, 421, 234-248.

## Authors

- Deepanshu Singh (2502300100058)
- Bhavishya (2502300100051)

B.Tech CSE, III Semester — Mini Project (BCC 351), AKTU — AY 2026-27
