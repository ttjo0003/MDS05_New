from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import numpy as np
from types import SimpleNamespace

from transformers import data

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

missing, unexpected = model.load_state_dict(checkpoint, strict=False)

print("Missing keys count:", len(missing))
print("Unexpected keys count:", len(unexpected))
print("First missing keys:", missing[:10])
print("First unexpected keys:", unexpected[:10])

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
        print("Received request")
        data = request.get_json()

        print("Keys:", data.keys())
        print("Body shape:", np.array(data["body"]).shape)
        print("Left shape:", np.array(data["left"]).shape)
        print("Right shape:", np.array(data["right"]).shape)
        print("Face shape:", np.array(data["face_all"]).shape)
        print("Mask shape:", np.array(data["attention_mask"]).shape)

        body = torch.tensor(np.array(data["body"]), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        left = torch.tensor(np.array(data["left"]), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        right = torch.tensor(np.array(data["right"]), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        face_all = torch.tensor(np.array(data["face_all"]), dtype=torch.float32).unsqueeze(0).to(DEVICE)

        attention_mask = torch.tensor(
            np.array(data["attention_mask"]),
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)

        def normalize_skeleton(x):
            coord = x[..., :2]
            conf = x[..., 2:]

            valid = (coord.abs().sum(dim=-1, keepdim=True) > 0).float()

            mean = (coord * valid).sum(dim=(1, 2), keepdim=True) / (
                valid.sum(dim=(1, 2), keepdim=True) + 1e-6
            )

            std = torch.sqrt(
                ((coord - mean) ** 2 * valid).sum(dim=(1, 2), keepdim=True) /
                (valid.sum(dim=(1, 2), keepdim=True) + 1e-6)
            )

            coord = (coord - mean) / (std + 1e-6)
            coord = coord * valid

            return torch.cat([coord, conf], dim=-1)

        body = normalize_skeleton(body)
        left = normalize_skeleton(left)
        right = normalize_skeleton(right)
        face_all = normalize_skeleton(face_all)

        print("Body first frame:", body[0, 0, :, :])
        print("Left abs sum:", left.abs().sum().item())
        print("Right abs sum:", right.abs().sum().item())
        print("Face abs sum:", face_all.abs().sum().item())
        print(
            "Input abs total:",
            body.abs().sum().item()
            + left.abs().sum().item()
            + right.abs().sum().item()
            + face_all.abs().sum().item()
        )

        src_input = {
            "body": body,
            "left": left,
            "right": right,
            "face_all": face_all,
            "attention_mask": attention_mask
        }

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

            print("Output tokens:", output_tokens[0].tolist())
            print("Prediction:", prediction)

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