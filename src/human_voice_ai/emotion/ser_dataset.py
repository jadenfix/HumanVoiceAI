"""
Dataset and data loading utilities for Speech Emotion Recognition (SER).
"""

import os
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from human_voice_ai.base_dataset import BaseDataset
from human_voice_ai.audio.audio_utils import load_audio, normalize_audio
from human_voice_ai.utils.config import Config


class SERDataset(BaseDataset):
    """
    Dataset for Speech Emotion Recognition (SER) tasks.
    Loads audio files and their corresponding emotion labels.
    """

    # Mapping of emotion labels to indices
    EMOTION_MAP = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "disgust": 5,
        "surprise": 6,
        "calm": 7,
    }

    # Reverse mapping for decoding
    INDEX_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}

    def __init__(
        self,
        config: Config,
        data_dir: Union[str, Path],
        split: str = "train",
        max_duration: float = 10.0,
        min_duration: float = 0.5,
        target_sample_rate: int = 22050,
        augment: bool = False,
        **kwargs,
    ):
        """
        Initialize the SER dataset.

        Args:
            config: Configuration object
            data_dir: Root directory containing the dataset
            split: Data split ('train', 'val', 'test')
            max_duration: Maximum duration of audio clips in seconds
            min_duration: Minimum duration of audio clips in seconds
            target_sample_rate: Target sample rate for audio
            augment: Whether to apply data augmentation
            **kwargs: Additional arguments for BaseDataset
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.target_sample_rate = target_sample_rate
        self.augment = augment and (split == "train")

        # Audio processing parameters
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length
        self.n_mels = config.audio.n_mels
        self.f_min = config.audio.f_min
        self.f_max = config.audio.f_max

        # Initialize transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )

        # AmplitudeToDB transform
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Load data samples
        self.samples = self._load_samples()

        # Print dataset statistics
        print(f"Loaded {len(self.samples)} {split} samples from {data_dir}")
        self._print_class_distribution()

    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load dataset samples from the data directory.

        Returns:
            List of sample dictionaries
        """
        samples = []
        metadata_path = self.data_dir / f"{self.split}_metadata.json"

        if metadata_path.exists():
            # Load from metadata file if it exists
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            for item in metadata:
                audio_path = self.data_dir / item["audio_path"]
                if audio_path.exists():
                    samples.append(
                        {
                            "audio_path": str(audio_path),
                            "emotion": item["emotion"],
                            "duration": item.get("duration", 0.0),
                            "speaker_id": item.get("speaker_id", ""),
                        }
                    )
        else:
            # Scan directory for audio files
            audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
            for root, _, files in os.walk(self.data_dir / self.split):
                for file in files:
                    if Path(file).suffix.lower() in audio_exts:
                        audio_path = Path(root) / file
                        # Try to extract emotion from filename or directory structure
                        emotion = self._infer_emotion(audio_path)
                        if emotion is not None:
                            samples.append(
                                {
                                    "audio_path": str(audio_path),
                                    "emotion": emotion,
                                    "duration": 0.0,  # Will be updated in _load_audio
                                    "speaker_id": "",
                                }
                            )

        return samples

    def _infer_emotion(self, audio_path: Path) -> Optional[str]:
        """
        Infer emotion from file path or directory structure.

        Args:
            audio_path: Path to audio file

        Returns:
            Inferred emotion label or None if not found
        """
        # Check if emotion is in the parent directory name
        for emotion in self.EMOTION_MAP:
            if emotion in audio_path.parent.name.lower():
                return emotion

        # Check if emotion is in the filename
        for emotion in self.EMOTION_MAP:
            if emotion in audio_path.stem.lower():
                return emotion

        return None

    def _print_class_distribution(self) -> None:
        """Print the class distribution of the dataset."""
        if not self.samples:
            return

        class_counts = {emotion: 0 for emotion in self.EMOTION_MAP}
        for sample in self.samples:
            if sample["emotion"] in class_counts:
                class_counts[sample["emotion"]] += 1

        print("Class distribution:")
        for emotion, count in class_counts.items():
            if count > 0:
                print(f"  {emotion}: {count} samples")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed audio waveform
        """
        # Load audio file
        waveform = load_audio(
            audio_path,
            sample_rate=self.target_sample_rate,
            mono=True,
            normalize=True,
            device="cpu",  # Move to device later
        )

        # Trim silence
        waveform = self._trim_silence(waveform)

        # Ensure minimum duration
        min_samples = int(self.min_duration * self.target_sample_rate)
        if waveform.size(1) < min_samples:
            # Pad with zeros if too short
            padding = min_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Truncate to max duration
        max_samples = int(self.max_duration * self.target_sample_rate)
        if waveform.size(1) > max_samples:
            # Random crop for training, center crop for validation/test
            if self.split == "train" and self.augment:
                start = random.randint(0, waveform.size(1) - max_samples)
            else:
                start = (waveform.size(1) - max_samples) // 2
            waveform = waveform[:, start : start + max_samples]

        return waveform

    def _trim_silence(
        self, waveform: torch.Tensor, threshold_db: float = 40.0
    ) -> torch.Tensor:
        """
        Trim silence from the beginning and end of an audio signal.

        Args:
            waveform: Input audio waveform
            threshold_db: Threshold in dB below reference to consider as silence

        Returns:
            Trimmed waveform
        """
        if waveform.numel() == 0:
            return waveform

        # Compute short-time energy
        frame_length = 1024
        hop_length = 256

        # Pad the signal if needed
        if waveform.size(1) < frame_length:
            return waveform

        # Compute energy
        energy = torch.stft(
            waveform,
            n_fft=frame_length,
            hop_length=hop_length,
            window=torch.hann_window(frame_length, device=waveform.device),
            return_complex=True,
        ).abs()

        # Convert to dB
        db = 20 * torch.log10(energy + 1e-10)
        max_db = db.max()

        # Find frames above threshold
        mask = db > (max_db - threshold_db)
        non_silent = torch.any(mask, dim=0)

        if not torch.any(non_silent):
            return waveform  # All silent, return as is

        # Find first and last non-silent frame
        first = torch.argmax(non_silent)
        last = len(non_silent) - torch.argmax(non_silent.flip(0)) - 1

        # Convert frame indices to sample indices
        start = first * hop_length
        end = min((last + 1) * hop_length + frame_length, waveform.size(1))

        return waveform[:, start:end]

    def _extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram features from waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Mel-spectrogram features
        """
        # Compute mel-spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Convert to decibels
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

        return mel_spec_db

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - features: Mel-spectrogram features
                - label: Emotion class index
                - audio: Raw audio waveform
        """
        sample = self.samples[idx]

        # Load and preprocess audio
        waveform = self._load_audio(sample["audio_path"])

        # Apply data augmentation
        if self.augment:
            waveform = self._apply_augmentation(waveform)

        # Extract features
        features = self._extract_features(waveform)

        # Get label
        label = self.EMOTION_MAP.get(sample["emotion"].lower(), 0)

        return {
            "features": features.squeeze(0),  # Remove channel dim
            "label": torch.tensor(label, dtype=torch.long),
            "audio": waveform.squeeze(0),  # Remove channel dim
            "filepath": sample["audio_path"],
        }

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Augmented waveform
        """
        # Time stretching
        if random.random() < 0.3:  # 30% chance
            rate = random.uniform(0.9, 1.1)
            waveform = torchaudio.functional.time_stretch(
                waveform.unsqueeze(0), self.target_sample_rate, rate
            ).squeeze(0)

        # Pitch shifting
        if random.random() < 0.3:  # 30% chance
            n_steps = random.randint(-2, 2)
            waveform = torchaudio.functional.pitch_shift(
                waveform.unsqueeze(0), self.target_sample_rate, n_steps
            ).squeeze(0)

        # Add noise
        if random.random() < 0.2:  # 20% chance
            noise = torch.randn_like(waveform) * random.uniform(0.001, 0.01)
            waveform = waveform + noise

        return waveform

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        Pads sequences to the maximum length in the batch.

        Args:
            batch: List of samples

        Returns:
            Dictionary of batched tensors
        """
        # Get maximum length
        max_len = max(sample["features"].size(1) for sample in batch)

        # Initialize output tensors
        batch_size = len(batch)
        n_mels = batch[0]["features"].size(0)

        features = torch.zeros((batch_size, n_mels, max_len))
        labels = torch.zeros(batch_size, dtype=torch.long)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        filepaths = []

        # Fill tensors
        for i, sample in enumerate(batch):
            seq_len = sample["features"].size(1)
            features[i, :, :seq_len] = sample["features"]
            labels[i] = sample["label"]
            audio_lengths[i] = sample["audio"].size(0)
            filepaths.append(sample["filepath"])

        # Stack audio (pad to max length)
        max_audio_len = max(sample["audio"].size(0) for sample in batch)
        audio = torch.zeros((batch_size, max_audio_len))
        for i, sample in enumerate(batch):
            audio[i, : sample["audio"].size(0)] = sample["audio"]

        return {
            "features": features,
            "labels": labels,
            "audio": audio,
            "audio_lengths": audio_lengths,
            "filepaths": filepaths,
        }


def create_ser_data_loaders(
    config: Config,
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        config: Configuration object
        data_dir: Root directory containing the dataset
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        **kwargs: Additional arguments for SERDataset

    Returns:
        Dictionary containing data loaders for 'train', 'val', and 'test' splits
    """
    data_dir = Path(data_dir)

    # Create datasets
    train_dataset = SERDataset(
        config=config,
        data_dir=data_dir / "train",
        split="train",
        augment=True,
        **kwargs,
    )

    val_dataset = SERDataset(
        config=config, data_dir=data_dir / "val", split="val", augment=False, **kwargs
    )

    test_dataset = SERDataset(
        config=config, data_dir=data_dir / "test", split="test", augment=False, **kwargs
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=SERDataset.collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=SERDataset.collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=SERDataset.collate_fn,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": len(SERDataset.EMOTION_MAP),
    }
