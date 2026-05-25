import torch

# ======================
# CONFIG
# ======================

VIDEO_DIR = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\videos"

METADATA_PATH = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\WLASL_v0.3.json"

SAVE_MODEL_PATH = "rgb_mobilenet_lstm.pth"
SAVE_LABEL_PATH = "rgb_label_map.json"

NUM_FRAMES = 30
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

# Optional: limit number of glosses first for testing
MAX_CLASSES = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")