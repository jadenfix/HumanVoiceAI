"""
Audio feature extraction module for the Human Voice AI project.
Extracts mel-spectrogram, pitch (F0), and energy features from audio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as audio_F
from typing import Optional, Tuple, Union
import numpy as np


class FeatureExtractor(nn.Module):
    """
    Feature extractor for audio processing.
    Extracts mel-spectrogram, pitch (F0), and energy features.

    Args:
        sample_rate: Audio sample rate in Hz
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length between STFT windows
        win_length: Window length for STFT
        f_min: Minimum frequency for mel filterbanks
        f_max: Maximum frequency for mel filterbanks
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = None,
        f_min: float = 0.0,
        f_max: float = None,
        device: str = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.f_min = f_min
        self.f_max = f_max or (sample_rate // 2)
        self._device = device or "cpu"

        # Initialize mel spectrogram transform
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=self.f_max,
            n_stft=n_fft // 2 + 1,
        ).to(self._device)

        # Initialize window function and register as buffer
        self.register_buffer(
            "window", torch.hann_window(self.win_length, device=self._device)
        )

    def forward(self, audio_path: str) -> torch.Tensor:
        """Forward pass for feature extraction from audio file."""
        return self.extract(audio_path)

    def extract(self, audio_path: str) -> torch.Tensor:
        """Extract features from an audio file."""
        # Load audio file using soundfile which is more reliable
        import soundfile as sf

        waveform, sample_rate = sf.read(audio_path, dtype="float32")

        # Convert to tensor and ensure correct shape [1, T] for single channel
        waveform = torch.from_numpy(waveform).unsqueeze(0)

        # Move to the specified device
        waveform = waveform.to(self._device)

        return self.extract_from_waveform(waveform, sample_rate)

    def extract_from_waveform(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """Extract features from a waveform tensor.

        Args:
            waveform: Input audio waveform tensor of shape [channels, time] or [time]
            sample_rate: Sample rate of the input audio

        Returns:
            torch.Tensor: Extracted features of shape [time, n_mels + 2]

        Raises:
            ValueError: If input is invalid or feature extraction fails
        """
        # Input validation
        if waveform.numel() == 0:
            raise ValueError("Input waveform is empty")

        # Ensure waveform is [channels, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, time]
        elif waveform.dim() > 2:
            raise ValueError(f"Expected 1D or 2D input, got {waveform.dim()}D")

        # Move to the correct device
        waveform = waveform.to(self._device)

        # Resample if needed
        if sample_rate != self.sample_rate:
            if sample_rate <= 0 or self.sample_rate <= 0:
                raise ValueError(
                    f"Invalid sample rates: input={sample_rate}, model={self.sample_rate}"
                )

            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            ).to(self._device)
            waveform = resampler(waveform)

        # Extract features
        mel_spec = self._extract_mel_spectrogram(waveform)  # [time, n_mels]
        f0 = self._extract_pitch(waveform)  # [time]
        energy = self._extract_energy(waveform)  # [time]

        # Ensure all features have the same number of time steps
        min_time_steps = min(mel_spec.size(0), f0.size(0), energy.size(0))
        if min_time_steps == 0:
            raise ValueError("One or more feature extractors returned empty output")

        # Slice to common length
        mel_spec = mel_spec[:min_time_steps]  # [min_time_steps, n_mels]
        f0 = f0[:min_time_steps].unsqueeze(1)  # [min_time_steps, 1]
        energy = energy[:min_time_steps].unsqueeze(1)  # [min_time_steps, 1]

        # Combine features
        try:
            features = torch.cat(
                [mel_spec, f0, energy], dim=1
            )  # [min_time_steps, n_mels + 2]
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to concatenate features. Shapes - mel: {mel_spec.shape}, "
                f"f0: {f0.shape}, energy: {energy.shape}"
            ) from e

        # Final validation
        if features.dim() != 2 or features.size(1) != self.n_mels + 2:
            raise RuntimeError(
                f"Unexpected feature shape: {features.shape}. "
                f"Expected [time, {self.n_mels + 2}]"
            )

        return features

    def _extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from waveform.

        Args:
            waveform: Input tensor of shape [channels, time]

        Returns:
            torch.Tensor: Mel spectrogram of shape [time, n_mels]
        """
        # Ensure waveform is 2D [channel, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, time]

        # Calculate expected frames
        seq_len = waveform.size(-1)
        expected_frames = (seq_len - self.win_length) // self.hop_length + 1
        print(f"Mel - Input length: {seq_len}, Expected frames: {expected_frames}")

        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        actual_frames = stft.size(-1)
        print(f"Mel - Actual frames: {actual_frames}")
        if actual_frames != expected_frames:
            print(
                f"  Warning: Frame mismatch! Expected {expected_frames}, got {actual_frames}"
            )

        # Convert to power spectrogram
        # [channel, freq, time, 2] -> [channel, freq, time]
        spec = torch.view_as_real(stft).pow(2).sum(-1)

        # Apply mel filterbank
        mel_spec = self.mel_scale(spec)  # [channel, n_mels, time]

        # Convert to log scale with a small offset to avoid log(0)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))

        # Average over channels if multi-channel
        if mel_spec.size(0) > 1:
            mel_spec = mel_spec.mean(dim=0, keepdim=False)  # [n_mels, time]
        else:
            mel_spec = mel_spec.squeeze(0)  # Remove channel dim

        # Transpose to [time, n_mels]
        mel_spec = mel_spec.transpose(0, 1)

        return mel_spec

    def _extract_pitch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract pitch (F0) using autocorrelation.

        Args:
            waveform: Input audio tensor of shape [channels, time]

        Returns:
            torch.Tensor: Pitch values of shape [time_frames]
        """
        # Ensure waveform is [channels, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = waveform.device

        # Calculate expected frames with center padding (matching STFT behavior)
        seq_len = waveform.size(-1)
        pad = self.n_fft // 2
        expected_frames = ((seq_len + 2 * pad - self.win_length) // self.hop_length) + 1
        print(f"Pitch - Input length: {seq_len}, Expected frames: {expected_frames}")

        # Pad the waveform to match STFT behavior (center=True)
        waveform_padded = F.pad(waveform, (pad, pad), mode="reflect")

        # Log actual frames after padding
        actual_frames = (
            waveform_padded.size(-1) - self.win_length
        ) // self.hop_length + 1
        print(
            f"Pitch - Padded length: {waveform_padded.size(-1)}, Actual frames: {actual_frames}"
        )
        if actual_frames != expected_frames:
            print(
                f"  Warning: Frame mismatch! Expected {expected_frames}, got {actual_frames}"
            )

        # Unfold into frames [channels, n_frames, frame_length]
        frames = waveform_padded.unfold(-1, self.n_fft, self.hop_length)

        # Compute energy for voicing detection
        energy = torch.norm(frames, dim=-1)  # [channels, n_frames]
        energy_threshold = 0.1 * torch.max(energy, dim=-1, keepdim=True)[0]

        # Center the signal by removing the mean (DC component)
        frames = frames - frames.mean(dim=-1, keepdim=True)

        # Compute autocorrelation using FFT
        fft_frames = torch.fft.rfft(
            frames, dim=-1, n=self.n_fft * 2
        )  # [channels, n_frames, n_fft+1]
        power_spectrum = fft_frames.real.pow(2) + fft_frames.imag.pow(2)
        autocorr = torch.fft.irfft(
            power_spectrum, dim=-1, n=self.n_fft * 2
        )  # [channels, n_frames, n_fft*2]

        # Normalize autocorrelation
        autocorr = autocorr / (torch.norm(frames, dim=-1, keepdim=True) ** 2 + 1e-10)

        # Find peaks in autocorrelation (skip first sample to avoid DC)
        # Look for peaks in the range corresponding to 80-1000 Hz
        min_period = self.sample_rate // 1000  # 1000 Hz
        max_period = self.sample_rate // 80  # 80 Hz

        # Extract relevant portion of autocorrelation
        autocorr = autocorr[
            ..., min_period : max_period + 1
        ]  # [channels, n_frames, max_period-min_period+1]

        # Find peaks
        max_vals, max_idxs = torch.max(autocorr, dim=-1)  # [channels, n_frames]

        # Convert to frequency
        pitch = self.sample_rate / (
            max_idxs.float() + min_period + 1e-10
        )  # [channels, n_frames]

        # Apply voicing detection
        pitch[max_vals < 0.3] = 0.0  # Low correlation means unvoiced
        pitch[energy < energy_threshold] = 0.0  # Low energy means silence

        # Average across channels if multi-channel
        if pitch.dim() > 1 and pitch.size(0) > 1:
            pitch = pitch.mean(dim=0)  # [n_frames]
        else:
            pitch = pitch.squeeze(0)  # Remove channel dimension

        # Smooth the pitch contour
        if pitch.size(0) > 1:
            pitch = torch.nn.functional.avg_pool1d(
                pitch.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                kernel_size=3,
                stride=1,
                padding=1,
            ).squeeze()  # Remove added dims

        return pitch

    def _extract_energy(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract energy from waveform.

        Args:
            waveform: Input tensor of shape [channels, time]

        Returns:
            torch.Tensor: Energy values of shape [time_frames]
        """
        # Ensure waveform is 2D [channel, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, time]

        # Add center padding to match STFT behavior
        pad = self.n_fft // 2
        waveform = F.pad(waveform, (pad, pad), mode="reflect")

        # Log frame info
        seq_len = waveform.size(-1) - 2 * pad  # Original length
        expected_frames = ((seq_len + 2 * pad - self.win_length) // self.hop_length) + 1
        print(
            f"Energy - Input length: {seq_len}, Padded length: {waveform.size(-1)}, Expected frames: {expected_frames}"
        )

        assert waveform.dim() == 2, f"Expected 2D input, got {waveform.dim()}D"

        # Pad the waveform to ensure complete frames
        pad_len = (
            self.hop_length - (waveform.size(-1) % self.hop_length)
        ) % self.hop_length
        if pad_len > 0:
            waveform = F.pad(waveform, (0, pad_len))

        # Unfold into frames [channel, n_frames, frame_length]
        frames = waveform.unfold(-1, self.win_length, self.hop_length)

        # Compute energy (L2 norm) for each frame
        energy = torch.norm(frames, dim=-1)  # [channel, n_frames]

        # Average over channels if multi-channel
        if energy.size(0) > 1:
            energy = energy.mean(dim=0)  # [n_frames]
        else:
            energy = energy.squeeze(0)  # Remove channel dim

        return energy

        # Convert to mono if needed
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extract features
        mel_spec = self._extract_mel_spectrogram(waveform)
        f0 = self._extract_pitch(waveform)
        energy = self._extract_energy(waveform)

        # Combine features
        features = torch.cat([mel_spec, f0.unsqueeze(-1), energy.unsqueeze(-1)], dim=-1)

        return features

    def get_feature_dim(self) -> int:
        """Get the dimension of the output features."""
        return self.n_mels + 2  # +2 for F0 and energy
