"""
Audio processing utilities for the VoiceAI project.
Includes functions for loading, saving, and processing audio data.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


def load_audio(
    filepath: Union[str, Path],
    sample_rate: int = 22050,
    mono: bool = True,
    normalize: bool = True,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Load an audio file and return it as a PyTorch tensor.
    
    Args:
        filepath: Path to the audio file
        sample_rate: Target sample rate (will resample if different from file)
        mono: If True, convert to mono by averaging channels
        normalize: If True, normalize audio to [-1, 1]
        device: Device to load the tensor onto
        
    Returns:
        torch.Tensor: Audio waveform with shape (channels, samples)
    """
    # Load audio file
    try:
        waveform, orig_sample_rate = torchaudio.load(filepath)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file {filepath}: {e}")
    
    # Move to target device
    waveform = waveform.to(device)
    
    # Convert to mono if needed
    if mono and waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if orig_sample_rate != sample_rate:
        resampler = T.Resample(
            orig_freq=orig_sample_rate,
            new_freq=sample_rate,
            dtype=waveform.dtype
        ).to(device)
        waveform = resampler(waveform)
    
    # Normalize to [-1, 1] if needed
    if normalize and waveform.abs().max() > 1.0:
        waveform = waveform / (waveform.abs().max() + 1e-8)
    
    return waveform


def save_audio(
    waveform: torch.Tensor,
    filepath: Union[str, Path],
    sample_rate: int = 22050,
    bits_per_sample: int = 16
) -> None:
    """
    Save a waveform to an audio file.
    
    Args:
        waveform: Audio tensor with shape (channels, samples)
        filepath: Path to save the audio file
        sample_rate: Sample rate of the audio
        bits_per_sample: Bit depth for saving (16 or 32)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure waveform is on CPU for saving
    waveform = waveform.detach().cpu()
    
    # Convert to appropriate format
    if bits_per_sample == 16:
        waveform = (waveform * (2**15 - 1)).to(torch.int16)
    elif bits_per_sample == 32:
        waveform = (waveform * (2**31 - 1)).to(torch.int32)
    else:
        raise ValueError("bits_per_sample must be 16 or 32")
    
    # Save the file
    torchaudio.save(
        filepath,
        waveform,
        sample_rate,
        bits_per_sample=bits_per_sample
    )


def resample_audio(
    waveform: torch.Tensor,
    orig_sample_rate: int,
    new_sample_rate: int,
    resampling_method: str = 'sinc_fast'
) -> torch.Tensor:
    """
    Resample an audio waveform to a new sample rate.
    
    Args:
        waveform: Input audio tensor with shape (channels, samples)
        orig_sample_rate: Original sample rate
        new_sample_rate: Target sample rate
        resampling_method: Resampling method ('sinc_fast', 'sinc_best', 'kaiser_window')
        
    Returns:
        Resampled audio tensor with shape (channels, new_samples)
    """
    if orig_sample_rate == new_sample_rate:
        return waveform
    
    # Convert resampling method to torchaudio enum
    methods = {
        'sinc_fast': torchaudio.transforms.ResamplingMethod.sinc_interp_fastest,
        'sinc_best': torchaudio.transforms.ResamplingMethod.sinc_interp_kaiser,
        'kaiser_window': torchaudio.transforms.ResamplingMethod.kaiser_window
    }
    
    if resampling_method not in methods:
        raise ValueError(f"Invalid resampling_method: {resampling_method}")
    
    # Create resampler
    resampler = T.Resample(
        orig_freq=orig_sample_rate,
        new_freq=new_sample_rate,
        resampling_method=methods[resampling_method]
    ).to(waveform.device)
    
    return resampler(waveform)


def mix_down_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert a multi-channel audio to mono by averaging channels.
    
    Args:
        waveform: Input audio tensor with shape (channels, samples)
        
    Returns:
        Mono audio tensor with shape (1, samples)
    """
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    return waveform.mean(dim=0, keepdim=True)


def normalize_audio(
    waveform: torch.Tensor,
    target_level_db: float = -16.0,
    max_gain_db: float = 30.0
) -> torch.Tensor:
    """
    Normalize audio to a target level in dBFS.
    
    Args:
        waveform: Input audio tensor
        target_level_db: Target level in dBFS
        max_gain_db: Maximum gain in dB to apply (prevents excessive amplification)
        
    Returns:
        Normalized audio tensor
    """
    if waveform.numel() == 0:
        return waveform
    
    # Calculate current RMS level in dBFS
    rms = torch.sqrt(torch.mean(waveform ** 2))
    if rms < 1e-6:
        return waveform  # Avoid division by zero for silence
    
    current_level_db = 20 * torch.log10(rms)
    gain = min(
        max_gain_db,
        target_level_db - current_level_db
    )
    
    return waveform * (10.0 ** (gain / 20.0))


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    top_db: float = 40.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> torch.Tensor:
    """
    Trim leading and trailing silence from an audio signal.
    
    Args:
        waveform: Input audio tensor with shape (channels, samples)
        sample_rate: Sample rate of the audio
        top_db: The threshold (in decibels) below reference to consider as silence
        frame_length: The number of samples per analysis frame
        hop_length: The number of samples to advance between frames
        
    Returns:
        Trimmed audio tensor
    """
    if waveform.numel() == 0:
        return waveform
    
    # Convert to mono for silence detection
    mono_audio = mix_down_to_mono(waveform)
    
    # Compute short-time energy
    energy = torch.stft(
        mono_audio,
        n_fft=frame_length,
        hop_length=hop_length,
        window=torch.hann_window(frame_length, device=waveform.device),
        return_complex=True
    ).abs()
    
    # Find frames above threshold
    db = 20 * torch.log10(energy + 1e-10)
    max_db = db.max()
    mask = db > (max_db - top_db)
    
    # Find first and last non-silent frame
    non_silent = torch.any(mask, dim=0)
    if not torch.any(non_silent):
        return waveform  # All silent, return as is
    
    first = torch.argmax(non_silent)
    last = len(non_silent) - torch.argmax(non_silent.flip(0)) - 1
    
    # Convert frame indices to sample indices
    start = first * hop_length
    end = min((last + 1) * hop_length, waveform.size(-1))
    
    return waveform[..., start:end]


def add_reverb(
    waveform: torch.Tensor,
    sample_rate: int,
    reverberance: float = 50.0,
    hf_damping: float = 50.0,
    room_scale: float = 50.0,
    stereo_depth: float = 50.0,
    pre_delay: float = 0.0,
    wet_gain: float = -3.0
) -> torch.Tensor:
    """
    Add reverb to an audio signal using a simple impulse response convolution.
    
    Args:
        waveform: Input audio tensor with shape (channels, samples)
        sample_rate: Sample rate of the audio
        reverberance: Amount of reverb (0-100)
        hf_damping: High-frequency damping (0-100)
        room_scale: Room size (0-100)
        stereo_depth: Stereo width (0-100)
        pre_delay: Pre-delay in milliseconds
        wet_gain: Wet gain in dB
        
    Returns:
        Audio with reverb applied
    """
    # Generate a simple impulse response
    impulse_length = int(2.0 * sample_rate)  # 2 seconds
    impulse = torch.zeros(impulse_length, device=waveform.device)
    impulse[0] = 1.0
    
    # Apply pre-delay
    delay_samples = int(pre_delay * sample_rate / 1000)
    if delay_samples > 0:
        impulse = torch.roll(impulse, delay_samples)
        impulse[:delay_samples] = 0.0
    
    # Simple exponential decay
    t = torch.linspace(0, 1.0, impulse_length, device=waveform.device)
    decay = torch.exp(-t * (100.0 / (reverberance + 1e-6)))
    
    # Apply high-frequency damping
    hf_damping = 1.0 - (hf_damping / 100.0)
    t = torch.linspace(0, 1.0, impulse_length, device=waveform.device)
    hf_filter = torch.exp(-t * (1.0 - hf_damping) * 10.0)
    
    # Combine effects
    impulse = impulse * decay * hf_filter
    
    # Normalize
    impulse = impulse / (torch.norm(impulse) + 1e-6)
    
    # Convert to stereo if needed
    if waveform.size(0) == 2:
        # Create stereo IR with slight delay between channels
        ir_left = torch.roll(impulse, -10)
        ir_right = torch.roll(impulse, 10)
        impulse = torch.stack([ir_left, ir_right])
    else:
        impulse = impulse.unsqueeze(0)
    
    # Apply wet gain
    impulse = impulse * (10.0 ** (wet_gain / 20.0))
    
    # Apply convolution
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        
    # Pad input for valid convolution
    padding = impulse.size(-1) - 1
    padded = torch.nn.functional.pad(waveform, (0, padding))
    
    # Convert to frequency domain for faster convolution
    fft_size = 1 << (2 * padded.size(-1) - 1).bit_length()
    fft_waveform = torch.fft.rfft(padded, n=fft_size)
    fft_ir = torch.fft.rfft(impulse, n=fft_size)
    
    # Multiply in frequency domain and convert back
    result = torch.fft.irfft(fft_waveform * fft_ir, n=fft_size)
    
    # Trim to original length
    result = result[..., :waveform.size(-1)]
    
    # Mix with dry signal
    dry_gain = 10.0 ** (-wet_gain / 20.0)  # Inverse of wet gain
    return dry_gain * waveform + result[..., :waveform.size(-1)]


def time_stretch(
    waveform: torch.Tensor,
    rate: float,
    n_fft: int = 1024,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None
) -> torch.Tensor:
    """
    Time-stretch an audio signal using phase vocoder.
    
    Args:
        waveform: Input audio tensor with shape (channels, samples)
        rate: Stretch factor (>1.0 is slower, <1.0 is faster)
        n_fft: Size of FFT
        hop_length: Number of samples between successive frames
        win_length: Window size
        
    Returns:
        Time-stretched audio tensor
    """
    if rate == 1.0:
        return waveform
        
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    
    # Compute STFT
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=waveform.device),
        return_complex=True
    )
    
    # Phase vocoder time stretching
    phase_advance = torch.linspace(
        0, 
        torch.pi * hop_length, 
        stft.shape[-2],
        device=stft.device
    )
    
    # Phase accumulation
    phase = torch.angle(stft[..., 0])
    
    # For each channel
    result = []
    for i in range(waveform.size(0)):
        # Get magnitude and phase
        mag = torch.abs(stft[i])
        phase = torch.angle(stft[i])
        
        # Phase accumulation
        delta_phase = phase[..., 1:] - phase[..., :-1] - phase_advance[:-1]
        delta_phase = delta_phase - 2 * torch.pi * torch.round(delta_phase / (2 * torch.pi))
        phase_advance_per_bin = 2 * torch.pi * hop_length / (n_fft // 2 + 1)
        delta_phase = delta_phase + phase_advance_per_bin
        
        # Compute new phase
        phase = torch.cat([phase[..., :1], phase[..., 1:] + delta_phase.cumsum(-1)], dim=-1)
        
        # Reconstruct complex STFT
        stft_stretched = mag * torch.exp(1j * phase)
        
        # Inverse STFT
        y = torch.istft(
            stft_stretched,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=waveform.device)
        )
        
        result.append(y)
    
    return torch.stack(result)
