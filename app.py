from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import numpy as np
from types import SimpleNamespace

from models import Uni_Sign

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
    label_smoothing=0.0
)

MODEL_PATH = "wlasl_pose_only_islr.pth"

# =========================
# LOAD MODEL
# =========================

model = Uni_Sign(args)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if "model" in checkpoint:
    checkpoint = checkpoint["model"]

model.load_state_dict(checkpoint, strict=False)

model.to(DEVICE)
model.eval()

print("UniSign model loaded successfully.")

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
        data = request.get_json()

        body = torch.tensor(
            np.array(data["body"]),
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        left = torch.tensor(
            np.array(data["left"]),
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        right = torch.tensor(
            np.array(data["right"]),
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        face_all = torch.tensor(
            np.array(data["face_all"]),
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        attention_mask = torch.tensor(
            np.array(data["attention_mask"]),
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)

        src_input = {
            "body": body,
            "left": left,
            "right": right,
            "face_all": face_all,
            "attention_mask": attention_mask
        }

        # dummy target input
        tgt_input = {
            "gt_sentence": [""]
        }

        with torch.no_grad():

            pre_compute = model(src_input, tgt_input)

            output_tokens = model.generate(
                pre_compute,
                max_new_tokens=30,
                num_beams=4
            )

            prediction = model.mt5_tokenizer.batch_decode(
                output_tokens,
                skip_special_tokens=True
            )[0]

        return jsonify({
            "prediction": prediction,
            "confidence": "Generated",
            "message": "UniSign live translation completed."
        })

    except Exception as e:
        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(debug=True)