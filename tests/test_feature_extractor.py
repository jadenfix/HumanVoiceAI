"""
Tests for the audio feature extractor.
"""

import os
import pytest
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.human_voice_ai.audio.feature_extractor import FeatureExtractor

# Test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
# Create test audio file if it doesn't exist
TEST_AUDIO_PATH = os.path.join(TEST_DATA_DIR, "test_audio.wav")

# Set device for testing
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@pytest.fixture(scope="module")
def setup_test_audio():
    """Create a test audio file if it doesn't exist."""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Generate a simple sine wave if test file doesn't exist
    if not os.path.exists(TEST_AUDIO_PATH):
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = 0.5 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(
            0
        )  # 440 Hz sine wave

        # Save using soundfile which is more reliable than torchaudio for file saving
        import soundfile as sf

        sf.write(TEST_AUDIO_PATH, waveform.numpy().T, sample_rate)

    return TEST_AUDIO_PATH


@pytest.fixture
def feature_extractor():
    """Create a feature extractor instance for testing."""
    return FeatureExtractor(
        sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160, device=DEVICE
    )


def test_feature_extractor_init(feature_extractor):
    """Test that the feature extractor initializes correctly."""
    assert feature_extractor.sample_rate == 16000
    assert feature_extractor.n_mels == 80
    assert feature_extractor.n_fft == 1024
    assert feature_extractor.hop_length == 160


def test_extract_features_from_waveform(feature_extractor, setup_test_audio):
    """Test feature extraction from a waveform tensor."""
    # Load test audio using soundfile
    import soundfile as sf

    audio_path = setup_test_audio
    waveform, sample_rate = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float().to(DEVICE)
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension if needed

    # Print debug info
    n_fft = feature_extractor.n_fft
    hop_length = feature_extractor.hop_length
    win_length = feature_extractor.win_length
    print(f"\nTest Configuration:")
    print(f"- Sample rate: {sample_rate}")
    print(f"- n_fft: {n_fft}")
    print(f"- hop_length: {hop_length}")
    print(f"- win_length: {win_length}")
    print(f"- Waveform length: {waveform.shape[-1]} samples")

    # Extract features
    features = feature_extractor.extract_from_waveform(waveform, sample_rate)

    # Print actual output shape
    print(f"\nOutput shape: {features.shape}")

    # Move to CPU for assertions if needed
    if str(DEVICE) != "cpu":
        features = features.cpu()
    features = features.detach().numpy()

    # Calculate expected frames with center padding
    n_samples = waveform.shape[-1]
    pad = n_fft // 2
    expected_frames = ((n_samples + 2 * pad - n_fft) // hop_length) + 1

    print(f"\nExpected frames calculation:")
    print(f"- Input samples: {n_samples}")
    print(f"- Padding (n_fft//2): {pad}")
    print(f"- Padded length: {n_samples + 2*pad}")
    print(
        f"- Expected frames: (({n_samples} + 2*{pad} - {n_fft}) // {hop_length}) + 1 = {expected_frames}"
    )

    # Check the shape of the output
    assert features.shape == (
        expected_frames,
        82,
    ), f"Expected shape {(expected_frames, 82)}, got {features.shape}"

    # Check that features are finite
    assert np.isfinite(features).all(), "Features contain NaN or infinite values"


def test_extract_features_from_file(feature_extractor, setup_test_audio):
    """Test feature extraction from an audio file."""
    audio_path = setup_test_audio

    # Load audio to get actual length
    import soundfile as sf

    waveform, sample_rate = sf.read(audio_path)
    waveform = torch.from_numpy(waveform).float().to(DEVICE)
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension if needed

    # Calculate expected frames with center padding
    n_samples = waveform.shape[-1]
    n_fft = feature_extractor.n_fft
    hop_length = feature_extractor.hop_length
    win_length = feature_extractor.win_length
    pad = n_fft // 2
    expected_frames = ((n_samples + 2 * pad - win_length) // hop_length) + 1

    # Extract features
    features = feature_extractor.extract(audio_path)

    # Move to CPU for assertions if needed
    if str(DEVICE) != "cpu":
        features = features.cpu()
    features = features.detach().numpy()

    # Check the shape of the output
    assert features.shape == (
        expected_frames,
        82,
    ), f"Expected shape {(expected_frames, 82)}, got {features.shape}"

    # Check that features are finite
    assert np.isfinite(features).all(), "Features contain NaN or infinite values"


def test_feature_dimension(feature_extractor):
    """Test that the feature dimension is correct."""
    assert feature_extractor.get_feature_dim() == 82  # 80 mels + F0 + energy


def test_pitch_extraction(feature_extractor):
    """Test that pitch extraction works on a simple sine wave."""
    # Create a simple 440 Hz sine wave on the correct device
    sample_rate = 16000
    duration = 0.1  # 100 ms
    t = torch.linspace(0, duration, int(sample_rate * duration), device=DEVICE)
    waveform = 0.5 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)  # 440 Hz sine wave

    # Extract pitch
    f0 = feature_extractor._extract_pitch(waveform)

    # Move to CPU for assertions if needed
    if str(DEVICE) != "cpu":
        f0 = f0.cpu()
    f0_mean = f0.mean().item()

    # Check that we got a reasonable pitch estimate
    # For a 440 Hz sine wave, we expect something close to 440 Hz
    # Allow for some error due to windowing effects
    assert (
        400 < f0_mean < 480
    ), f"Expected pitch between 400 and 480 Hz, got {f0_mean} Hz"


def test_energy_extraction(feature_extractor):
    """Test that energy extraction works."""
    # Create a simple sine wave on the correct device
    sample_rate = 16000
    duration = 0.1  # 100 ms
    t = torch.linspace(0, duration, int(sample_rate * duration), device=DEVICE)
    waveform = 0.5 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

    # Extract energy
    energy = feature_extractor._extract_energy(waveform)

    # Move to CPU for assertions if needed
    if str(DEVICE) != "cpu":
        energy = energy.cpu()

    # Energy should be positive and finite
    assert torch.all(energy > 0), "Energy values should be positive"
    assert torch.isfinite(energy).all(), "Energy values should be finite"


if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", __file__])
