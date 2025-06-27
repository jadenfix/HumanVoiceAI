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

class FeatureExtractor:
    """
    Feature extractor for audio processing.
    Extracts mel-spectrogram, pitch (F0), and energy features.
    """
    
    Args:
        sample_rate: Audio sample rate in Hz
        n_mels: Number of mel filterbanks
        n_fft: FFT window size
        hop_length: Hop length between STFT windows
        win_length: Window length for STFT (default: n_fft)
        f_min: Minimum frequency for mel filterbanks
        f_max: Maximum frequency for mel filterbanks (default: sample_rate // 2)
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
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.f_min = f_min
        self.f_max = f_max or (sample_rate // 2)
        
        # Register buffer for the window function
        self.register_buffer('window', torch.hann_window(self.win_length), persistent=False)
        
        # Mel scale will be initialized in the first forward pass to handle device placement
        self.mel_scale = None
    
    def _init_mel_scale(self, device):
        """Initialize mel scale on the correct device."""
        if self.mel_scale is None or self.mel_scale.device != device:
            self.mel_scale = torchaudio.transforms.MelScale(
                n_mels=self.n_mels,
                sample_rate=self.sample_rate,
                f_min=self.f_min,
                f_max=self.f_max,
                n_stft=self.n_fft // 2 + 1
            ).to(device)
    
    def to(self, *args, **kwargs):
        """Move the module to the specified device."""
        self = super().to(*args, **kwargs)
        # Re-initialize mel scale on the new device if it exists
        if self.mel_scale is not None:
            self._init_mel_scale(next(self.parameters()).device)
        return self
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform."""
        return self.extract_from_waveform(waveform, self.sample_rate)
    
    def extract(self, audio_path: str) -> torch.Tensor:
        """Extract features from an audio file."""
        # Load audio file using soundfile which is more reliable
        import soundfile as sf
        waveform, sample_rate = sf.read(audio_path, dtype='float32')
        
        # Convert to tensor and ensure correct shape [1, T] for single channel
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        waveform = waveform.to(device)
        
        return self.extract_from_waveform(waveform, sample_rate)
    
    def extract_from_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Extract features from a waveform tensor."""
        # Ensure waveform is on the same device as the model
        device = next(self.parameters()).device
        waveform = waveform.to(device)
        
        # Initialize mel scale if needed
        self._init_mel_scale(device)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.sample_rate
            ).to(device)
            waveform = resampler(waveform)
        
        # Extract mel spectrogram
        mel_spec = self._extract_mel_spectrogram(waveform)
        
        # Extract pitch (F0)
        f0 = self._extract_pitch(waveform)
        
        # Extract energy
        energy = self._extract_energy(waveform)
        
        # Stack features
        features = torch.cat([mel_spec, f0.unsqueeze(-1), energy.unsqueeze(-1)], dim=-1)
        
        return features
    
    def _extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from waveform."""
        # Ensure waveform is 2D [channel, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Get device
        device = waveform.device
        
        # Ensure window is on the correct device
        window = self.window.to(device)
        
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Convert to power spectrogram
        spec = torch.view_as_real(stft).pow(2).sum(-1)  # [channel, freq, time, 2] -> [channel, freq, time]
        
        # Apply mel filterbank
        mel_spec = self.mel_scale(spec)
        
        # Convert to log scale with a small offset to avoid log(0)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
        
        # Average over channels if multi-channel
        if mel_spec.dim() == 3 and mel_spec.size(0) > 1:
            mel_spec = mel_spec.mean(dim=0, keepdim=True)
        
        # Remove channel dimension
        mel_spec = mel_spec.squeeze(0)
        
        # Transpose to [time, n_mels]
        mel_spec = mel_spec.transpose(0, 1)
        
        return mel_spec
    
    def _extract_pitch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract pitch (F0) using autocorrelation."""
        device = waveform.device
        
        # Pad the waveform
        pad = self.n_fft // 2
        waveform_padded = F.pad(waveform, (pad, pad), mode='reflect')
        
        # Unfold into frames
        frames = waveform_padded.unfold(-1, self.n_fft, self.hop_length)
        
        # Ensure window is on the correct device
        window = self.window.unsqueeze(0).to(device)
        
        # Apply window function
        frames = frames * window
        
        # Compute autocorrelation using FFT
        fft_frames = torch.fft.rfft(frames, dim=-1)
        power_spectrum = fft_frames.real.pow(2) + fft_frames.imag.pow(2)
        autocorr = torch.fft.irfft(power_spectrum, dim=-1)
        
        # Find peaks in autocorrelation (skip first sample to avoid DC)
        max_vals, max_idxs = torch.max(autocorr[..., 1:], dim=-1)
        
        # Convert to frequency
        pitch = self.sample_rate / (max_idxs.float() + 1e-10)
        
        # Apply simple voicing detection (energy-based)
        energy = torch.norm(frames, dim=-1)
        energy_threshold = 0.1 * torch.max(energy)
        pitch[energy < energy_threshold] = 0.0
        
        return pitch
    
    def _extract_energy(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract energy from waveform."""
        # Ensure waveform is 2D [channel, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Pad if needed to make sure we can extract complete frames
        seq_len = waveform.size(-1)
        pad_len = (self.hop_length - (seq_len - self.win_length) % self.hop_length) % self.hop_length
        
        if pad_len > 0:
            waveform = F.pad(waveform, (0, pad_len))
            
        # Unfold into frames
        frames = waveform.unfold(-1, self.win_length, self.hop_length)
        
        # Compute energy per frame (L2 norm)
        energy = torch.norm(frames, dim=-1)
        
        # Average over channels if multi-channel
        if energy.dim() > 1 and energy.size(0) > 1:
            energy = energy.mean(dim=0)
            
        return energy
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = F.resample(waveform, sample_rate, self.sample_rate)
        
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
