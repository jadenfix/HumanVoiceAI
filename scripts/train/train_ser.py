"""
Training script for the Speech Emotion Recognition (SER) model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
from pathlib import Path

# Add project root to path
import sys

sys.path.append(str(Path(__file__).parent.absolute()))

from src.emotion.ser_model import SerModel
from src.audio.feature_extractor import FeatureExtractor


class SERDataset(Dataset):
    """Dataset for Speech Emotion Recognition."""

    def __init__(self, data_dir, split="train", sample_rate=22050, max_length=10):
        """Initialize the dataset.

        Args:
            data_dir: Path to the dataset directory
            split: 'train', 'val', or 'test'
            sample_rate: Audio sample rate
            max_length: Maximum audio length in seconds
        """
        self.data_dir = Path(data_dir) / split
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)

        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Emotion labels
        self.emotion_labels = ["neutral", "happy", "sad", "angry", "surprised"]
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotion_labels)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get audio path and emotion label
        item = self.metadata[idx]
        audio_path = self.data_dir / "audio" / item["audio_file"]
        emotion = item["emotion"]

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        # Trim or pad audio
        target_length = self.sample_rate * self.max_length
        if waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]
        elif waveform.size(1) < target_length:
            pad_length = target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        # Extract features
        features = self.feature_extractor.extract_from_waveform(
            waveform, self.sample_rate
        )

        # Convert emotion to one-hot
        emotion_idx = self.emotion_to_idx[emotion]

        return {
            "features": features,
            "emotion": emotion_idx,
            "audio_path": str(audio_path),
        }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

    for i, batch in enumerate(progress_bar):
        # Move data to device
        features = batch["features"].to(device)
        emotions = batch["emotion"].to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, emotions)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += emotions.size(0)
        correct += (predicted == emotions).sum().item()
        running_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(
            {"loss": running_loss / (i + 1), "acc": 100 * correct / total}
        )

    # Log metrics
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    if writer is not None:
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, epoch, writer):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]", leave=False)

        for i, batch in enumerate(progress_bar):
            # Move data to device
            features = batch["features"].to(device)
            emotions = batch["emotion"].to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, emotions)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += emotions.size(0)
            correct += (predicted == emotions).sum().item()
            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": running_loss / (i + 1), "acc": 100 * correct / total}
            )

    # Log metrics
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    if writer is not None:
        writer.add_scalar("Loss/val", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", accuracy, epoch)

    return avg_loss, accuracy


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SER model")
    parser.add_argument(
        "--data_dir", type=str, default="data/ser", help="Path to dataset directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/ser",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/ser", help="Directory to save logs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize model
    model = SerModel().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Load checkpoint if resuming
    start_epoch = 0
    best_accuracy = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_accuracy"]
        print(f"Resuming from epoch {start_epoch}")

    # Create datasets and dataloaders
    train_dataset = SERDataset(args.data_dir, split="train")
    val_dataset = SERDataset(args.data_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    print("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )

        # Update learning rate
        scheduler.step(val_acc)

        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save checkpoint
        is_best = val_acc > best_accuracy
        best_accuracy = max(val_acc, best_accuracy)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_accuracy": best_accuracy,
        }

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "latest.pth"))

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    main()
