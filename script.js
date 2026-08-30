const API_BASE = ""; // same origin as Flask backend

const webcamEl = document.getElementById("webcam");
const overlayEl = document.getElementById("overlay");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const speakToggle = document.getElementById("speakToggle");
const statusEl = document.getElementById("status");
const currentLabelEl = document.getElementById("currentLabel");
const confidenceEl = document.getElementById("confidence");
const sentenceEl = document.getElementById("sentence");
const modelInfoEl = document.getElementById("modelInfo");

const spaceBtn = document.getElementById("spaceBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const clearBtn = document.getElementById("clearBtn");
const speakSentenceBtn = document.getElementById("speakSentenceBtn");

let stream = null;
let captureInterval = null;
let sessionId = null;
let sentence = "";
const PREDICT_INTERVAL_MS = 350; // ~3 predictions/sec, matches debounce buffer size

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.model_loaded) {
      modelInfoEl.textContent = `Model loaded. Classes: ${data.classes.join(", ")}`;
    } else {
      modelInfoEl.textContent =
        "No trained model found yet. Run collect_data.py + train_model.py, then restart the backend.";
    }
  } catch (err) {
    modelInfoEl.textContent = "Could not reach backend at /api/health.";
  }
}

async function startSession() {
  try {
    const res = await fetch(`${API_BASE}/api/session`, { method: "POST" });
    const data = await res.json();
    sessionId = data.session_id;
  } catch (err) {
    console.error("Failed to start session", err);
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    webcamEl.srcObject = stream;
    await startSession();

    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = "Camera running — show a hand sign";

    captureInterval = setInterval(captureAndPredict, PREDICT_INTERVAL_MS);
  } catch (err) {
    statusEl.textContent = "Could not access webcam: " + err.message;
  }
}

function stopCamera() {
  if (captureInterval) clearInterval(captureInterval);
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "Camera stopped";
  currentLabelEl.textContent = "–";
  confidenceEl.textContent = "";
}

function captureAndPredict() {
  if (!webcamEl.videoWidth) return;

  captureCanvas.width = webcamEl.videoWidth;
  captureCanvas.height = webcamEl.videoHeight;
  captureCtx.drawImage(webcamEl, 0, 0, captureCanvas.width, captureCanvas.height);
  const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);

  fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, session_id: sessionId }),
  })
    .then((res) => res.json())
    .then(handlePrediction)
    .catch((err) => {
      statusEl.textContent = "Prediction error: " + err.message;
    });
}

function handlePrediction(data) {
  if (!data.hand_detected) {
    statusEl.textContent = "No hand detected — show your hand to the camera";
    currentLabelEl.textContent = "–";
    confidenceEl.textContent = "";
    return;
  }

  if (!data.label) {
    statusEl.textContent = data.message || "Hand detected, waiting for model...";
    currentLabelEl.textContent = "–";
    return;
  }

  statusEl.textContent = "Hand detected";
  currentLabelEl.textContent = data.label;
  confidenceEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  if (data.stable_label) {
    appendToSentence(data.stable_label);
  }
}

function appendToSentence(label) {
  sentence += label;
  sentenceEl.textContent = sentence;
  if (speakToggle.checked) {
    speak(label);
  }
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.0;
  window.speechSynthesis.speak(utter);
}

spaceBtn.addEventListener("click", () => {
  sentence += " ";
  sentenceEl.textContent = sentence;
});

backspaceBtn.addEventListener("click", () => {
  sentence = sentence.slice(0, -1);
  sentenceEl.textContent = sentence;
});

clearBtn.addEventListener("click", () => {
  sentence = "";
  sentenceEl.textContent = sentence;
});

speakSentenceBtn.addEventListener("click", () => {
  if (sentence.trim()) speak(sentence);
});

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

checkHealth();
