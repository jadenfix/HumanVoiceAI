"""
Base dataset class for the VoiceAI project.
Provides common functionality for data loading, preprocessing, and augmentation.
"""

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from human_voice_ai.utils.config import Config


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets in the VoiceAI project.
    Handles common functionality like file I/O, data splitting, and basic transforms.
    """
    
    def __init__(
        self, 
        config: Config,
        data_dir: Union[str, Path],
        split: str = 'train',
        **kwargs
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            config: Configuration object
            data_dir: Root directory containing the dataset
            split: Data split ('train', 'val', 'test')
            **kwargs: Additional dataset-specific arguments
        """
        super().__init__()
        self.config = config
        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.sample_rate = config.audio.sample_rate
        
        # Validate split
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        
        # Load data samples
        self.samples = self._load_samples()
        
        # Initialize transforms
        self._init_transforms()
        
        print(f"Loaded {len(self)} samples from {self.split} split")
    
    @abstractmethod
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load dataset samples.
        
        Returns:
            List of sample dictionaries, where each dictionary contains
            the information needed to load and process a single data sample.
        """
        pass
    
    def _init_transforms(self) -> None:
        """Initialize data transforms for the dataset."""
        self.transforms = []
        
        # Add data augmentation for training
        if self.split == 'train' and hasattr(self.config, 'augmentation'):
            if getattr(self.config.augmentation, 'time_stretch', False):
                self.transforms.append(self._time_stretch)
            if getattr(self.config.augmentation, 'pitch_shift', False):
                self.transforms.append(self._pitch_shift)
            if getattr(self.config.augmentation, 'add_noise', False):
                self.transforms.append(self._add_noise)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing the processed sample data
        """
        pass
    
    def _load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Load an audio file and return it as a PyTorch tensor.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            torch.Tensor: Audio waveform with shape (1, samples)
        """
        import torchaudio
        
        # Load audio file
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {audio_path}: {e}")
        
        # Convert to mono if needed
        if waveform.dim() > 1 and waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        return waveform
    
    def _time_stretch(self, waveform: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        """
        Time-stretch an audio waveform.
        
        Args:
            waveform: Input audio waveform
            rate: Stretch factor (e.g., 0.9 = 10% faster, 1.1 = 10% slower)
            
        Returns:
            Time-stretched waveform
        """
        if rate == 1.0:
            return waveform
            
        # Convert to numpy for processing
        audio = waveform.numpy()[0]
        
        # Calculate new length after time stretch
        orig_length = len(audio)
        new_length = int(orig_length / rate)
        
        # Resample to achieve time stretch
        time_stretched = np.interp(
            np.linspace(0, orig_length - 1, new_length),
            np.arange(orig_length),
            audio
        )
        
        # Convert back to tensor
        return torch.FloatTensor(time_stretched).unsqueeze(0)
    
    def _pitch_shift(self, waveform: torch.Tensor, n_steps: int = 0) -> torch.Tensor:
        """
        Shift the pitch of an audio waveform.
        
        Args:
            waveform: Input audio waveform
            n_steps: Number of semitones to shift (positive = higher pitch)
            
        Returns:
            Pitch-shifted waveform
        """
        if n_steps == 0:
            return waveform
            
        # Simple pitch shift using resampling
        rate = 2.0 ** (-n_steps / 12.0)
        
        # Time-stretch to change duration
        time_stretched = self._time_stretch(waveform, rate)
        
        # Resample to original length
        target_length = waveform.size(1)
        if time_stretched.size(1) > target_length:
            # Truncate if too long
            return time_stretched[:, :target_length]
        else:
            # Pad if too short
            result = torch.zeros_like(waveform)
            result[:, :time_stretched.size(1)] = time_stretched
            return result
    
    def _add_noise(self, waveform: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
        """
        Add random Gaussian noise to an audio waveform.
        
        Args:
            waveform: Input audio waveform
            snr_db: Signal-to-noise ratio in decibels
            
        Returns:
            Noisy waveform
        """
        # Calculate signal power
        signal_power = torch.mean(waveform ** 2)
        
        # Calculate noise power based on desired SNR
        snr = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr
        
        # Generate noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        # Add noise to signal
        return waveform + noise
    
    def _random_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random data augmentation to a waveform.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Augmented waveform
        """
        if not self.transforms or random.random() > 0.5:
            return waveform
            
        # Apply random transforms
        for transform in self.transforms:
            if transform == self._time_stretch:
                rate = random.uniform(0.9, 1.1)
                waveform = transform(waveform, rate)
            elif transform == self._pitch_shift:
                n_steps = random.randint(-2, 2)
                waveform = transform(waveform, n_steps)
            elif transform == self._add_noise:
                snr_db = random.uniform(15.0, 30.0)
                waveform = transform(waveform, snr_db)
                
        return waveform
    
    @classmethod
    def create_splits(
        cls,
        config: Config,
        data_dir: Union[str, Path],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        **kwargs
    ) -> Dict[str, 'BaseDataset']:
        """
        Create train/val/test splits of the dataset.
        
        Args:
            config: Configuration object
            data_dir: Root directory containing the dataset
            train_ratio: Fraction of data to use for training
            val_ratio: Fraction of data to use for validation
            test_ratio: Fraction of data to use for testing
            **kwargs: Additional arguments to pass to the dataset constructor
            
        Returns:
            Dictionary containing 'train', 'val', and 'test' datasets
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        # Load all samples
        temp_ds = cls(config, data_dir, split='train', **kwargs)
        all_samples = temp_ds.samples
        random.shuffle(all_samples)
        
        # Calculate split indices
        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split samples
        train_samples = all_samples[:n_train]
        val_samples = all_samples[n_train:n_train + n_val]
        test_samples = all_samples[n_train + n_val:]
        
        # Create datasets
        train_ds = cls(config, data_dir, split='train', **kwargs)
        val_ds = cls(config, data_dir, split='val', **kwargs)
        test_ds = cls(config, data_dir, split='test', **kwargs)
        
        # Override samples with our split
        train_ds.samples = train_samples
        val_ds.samples = val_samples
        test_ds.samples = test_samples
        
        return {
            'train': train_ds,
            'val': val_ds,
            'test': test_ds
        }
