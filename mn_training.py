import os
import json
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from mn_config import (
    VIDEO_DIR,
    METADATA_PATH,
    SAVE_MODEL_PATH,
    SAVE_LABEL_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LR,
    DEVICE,
    MAX_CLASSES
)

from mn_dataset import WLASLVideoDataset
from mn_model import MobileNetLSTM


def build_label_map(metadata_path, max_classes=None):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    labels = sorted([item["gloss"] for item in metadata])

    if max_classes is not None:
        labels = labels[:max_classes]

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label


def train():
    label2id, id2label = build_label_map(METADATA_PATH, MAX_CLASSES)
    num_classes = len(label2id)

    print("Device:", DEVICE)
    print("Number of classes:", num_classes)
    print("Classes:", label2id)

    with open(SAVE_LABEL_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label
            },
            f,
            indent=2
        )

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = WLASLVideoDataset(
        video_dir=VIDEO_DIR,
        metadata_path=METADATA_PATH,
        split="train",
        label2id=label2id,
        transform=train_transform
    )

    val_dataset = WLASLVideoDataset(
        video_dir=VIDEO_DIR,
        metadata_path=METADATA_PATH,
        split="val",
        label2id=label2id,
        transform=val_transform
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    if len(train_dataset) == 0:
        raise ValueError("No training samples found. Check VIDEO_DIR, METADATA_PATH, video_id filenames, and split names.")

    if len(val_dataset) == 0:
        print("Warning: No validation samples found. Val Acc will be 0.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Debug one batch
    frames, labels = next(iter(train_loader))
    print("Debug batch frames shape:", frames.shape)
    print("Debug batch labels shape:", labels.shape)

    model = MobileNetLSTM(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for frames, labels in train_loader:
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(frames)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(frames)
                preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {total_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "label2id": label2id,
                    "id2label": id2label,
                    "num_frames": 30,
                    "img_size": IMG_SIZE
                },
                SAVE_MODEL_PATH
            )

            print("Saved best model.")

    print("Training complete.")
    print("Best Val Acc:", best_val_acc)


if __name__ == "__main__":
    train()