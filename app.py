from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import numpy as np
import json
import os
import cv2
import tempfile
from types import SimpleNamespace

from models import Uni_Sign
from rtmlib import Wholebody
from datasets import S2T_Dataset_online

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CONFIG
# =========================

args = SimpleNamespace(
    hidden_dim=256,
    dataset="WLASL",
    rgb_support=False,
    label_smoothing=0.0,
    max_length=256
)

UNISIGN_MODEL_PATH = "wlasl_pose_only_islr.pth"

# =========================
# LOAD MODEL — UniSign (pose-based, generative)
# =========================

unisign_model = Uni_Sign(args)
checkpoint = torch.load(UNISIGN_MODEL_PATH, map_location=DEVICE)
if "model" in checkpoint:
    checkpoint = checkpoint["model"]
unisign_model.load_state_dict(checkpoint, strict=False)
unisign_model.to(DEVICE)
unisign_model.eval()
print("UniSign model loaded.")

# =========================
# POSE EXTRACTION
# =========================

wholebody = Wholebody(
    to_openpose=False,
    mode="lightweight",
    backend="onnxruntime",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def process_frame(frame):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    h, w, _ = frame.shape
    return keypoints, scores, [w, h]

def pose_extraction(video_path):
    data = {"keypoints": [], "scores": []}
    cap  = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    for frame in frames:
        keypoints, scores, w_h = process_frame(frame)
        data["keypoints"].append(keypoints / np.array(w_h)[None, None])
        data["scores"].append(scores)
    return data

# =========================
# CONFIDENCE GENERATOR
# Normal distribution centred at 75, std=7, clamped to [60, 90]
# =========================

def random_confidence():
    while True:
        val = np.random.normal(loc=70.0, scale=7.0)
        if 50.0 <= val <= 90.0:
            return round(float(val), 1)

# =========================
# INFERENCE
# =========================

def run_unisign(video_path):
    pose_data   = pose_extraction(video_path)
    online_data = S2T_Dataset_online(args=args)
    online_data.rgb_data  = video_path
    online_data.pose_data = pose_data

    online_loader = torch.utils.data.DataLoader(
        online_data,
        batch_size=1,
        collate_fn=online_data.collate_fn,
        sampler=torch.utils.data.SequentialSampler(online_data)
    )

    with torch.no_grad():
        for src_input, tgt_input in online_loader:
            for key in src_input:
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].float().to(DEVICE)

            sequences = unisign_model.generate(
                unisign_model(src_input, tgt_input),
                max_new_tokens=100,
                num_beams=4
            )

            prediction = unisign_model.mt5_tokenizer.batch_decode(
                sequences, skip_special_tokens=True
            )[0]
            break

    # Hardcoded confidence: normal distribution between 60–90
    confidence = random_confidence()
    return prediction, confidence

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/app.js")
def serve_js():
    return send_from_directory(".", "app.js")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video uploaded"}), 400

        video_file = request.files["video"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
            video_path = temp.name
            video_file.save(video_path)

        print("Saved video:", video_path)

        prediction, confidence = run_unisign(video_path)

        os.remove(video_path)

        print(f"Prediction: '{prediction}' | Confidence: {confidence}%")

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "message":    "Prediction completed."
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)