from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, sys, torch

sys.path.insert(0, "./Uni-Sign")

import utils
from models import Uni_Sign
from datasets import S2T_Dataset_online
from torch.utils.data import DataLoader
from demo.online_inference import pose_extraction, inference_and_return
import argparse
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ──────────────────────────────────────
print("Loading model... (this may take a minute)")

parser = argparse.ArgumentParser('Uni-Sign', parents=[utils.get_args_parser()])
args = parser.parse_args([
    "--finetune", "checkpoints/wlasl_rgb_pose_islr.pth",
    "--seed", "42",
    "--rgb",              # enable RGB mode for this model
])

utils.set_seed(args.seed)

model = Uni_Sign(args=args)
model.train()
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

state_dict = torch.load(args.finetune, map_location='cpu')['model']
ret = model.load_state_dict(state_dict, strict=True)
print('Missing keys:', ret.missing_keys)
print('Unexpected keys:', ret.unexpected_keys)

model.eval()
model.to(torch.bfloat16)
print("Model loaded and ready!")

# ── Predict endpoint ────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
        video_file.save(tmp_path)

    try:
        print(f"Running pose extraction on {tmp_path}...")
        pose_data = pose_extraction(tmp_path)

        print("Building dataloader...")
        online_data = S2T_Dataset_online(args=args)
        online_data.rgb_data = tmp_path
        online_data.pose_data = pose_data

        online_sampler = torch.utils.data.SequentialSampler(online_data)
        online_dataloader = DataLoader(
            online_data,
            batch_size=1,
            collate_fn=online_data.collate_fn,
            sampler=online_sampler,
        )

        print("Running inference...")
        result = inference_and_return(online_dataloader, model)
        print(f"Result: {result}")

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.unlink(tmp_path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=5000, debug=False)