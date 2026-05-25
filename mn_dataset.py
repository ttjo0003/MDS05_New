import os
import cv2
import json
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from mn_config import NUM_FRAMES, IMG_SIZE


class WLASLVideoDataset(Dataset):
    def __init__(self, video_dir, metadata_path, split, label2id, transform=None):
        self.video_dir = Path(video_dir)
        self.metadata_path = metadata_path
        self.split = split
        self.label2id = label2id
        self.transform = transform
        self.samples = []

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        for item in metadata:
            gloss = item["gloss"]

            if gloss not in label2id:
                continue

            for inst in item["instances"]:
                if inst.get("split") != split:
                    continue

                video_id = inst.get("video_id")
                video_path = self.find_video_file(video_id)

                if video_path is not None:
                    self.samples.append((str(video_path), label2id[gloss], gloss, video_id))

        print(f"{split} samples loaded:", len(self.samples))

    def find_video_file(self, video_id):
        possible_exts = [".mp4", ".avi", ".mov", ".webm", ".mkv"]

        for ext in possible_exts:
            path = self.video_dir / f"{video_id}{ext}"
            if path.exists():
                return path

        return None

    def __len__(self):
        return len(self.samples)

    def sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]

        if len(frames) >= NUM_FRAMES:
            indices = np.linspace(0, len(frames) - 1, NUM_FRAMES).astype(int)
            frames = [frames[i] for i in indices]
        else:
            last = frames[-1]
            while len(frames) < NUM_FRAMES:
                frames.append(last)

        processed = []

        for frame in frames:
            img = Image.fromarray(frame)

            if self.transform:
                img = self.transform(img)

            processed.append(img)

        return torch.stack(processed)

    def __getitem__(self, idx):
        video_path, label, gloss, video_id = self.samples[idx]
        frames = self.sample_frames(video_path)

        return frames, torch.tensor(label, dtype=torch.long)