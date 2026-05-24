# ======================
# DATASET
# ======================

class VideoDataset(Dataset):
    def __init__(self, root_dir, label2id, transform=None):
        self.root_dir = Path(root_dir)
        self.label2id = label2id
        self.transform = transform
        self.samples = []

        for label in sorted(os.listdir(root_dir)):
            label_path = self.root_dir / label
            if not label_path.is_dir():
                continue

            for file in os.listdir(label_path):
                if file.lower().endswith((".mp4", ".avi", ".mov", ".webm")):
                    self.samples.append((str(label_path / file), label2id[label]))
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
        video_path, label = self.samples[idx]
        frames = self.sample_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long)
