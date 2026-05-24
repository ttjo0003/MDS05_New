import os
import json
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ======================
# CONFIG
# ======================

DATASET_DIR = "rgb_dataset"
SAVE_MODEL_PATH = "rgb_mobilenet_lstm.pth"
SAVE_LABEL_PATH = "rgb_label_map.json"

NUM_FRAMES = 30
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
