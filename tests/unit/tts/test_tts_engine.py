"""Unit tests for the TTS Engine."""

import os
import tempfile
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from human_voice_ai.tts.tts_engine import TtsEngine, TTSConfig

# Mock audio data for testing
TEST_AUDIO = np.random.rand(1, 16000).astype(np.float32)
TEST_SAMPLE_RATE = 16000


# Mock torchaudio.load to return test audio
def mock_torch_load(*args, **kwargs):
    return torch.from_numpy(TEST_AUDIO), TEST_SAMPLE_RATE


# Mock soundfile.write for testing
def mock_sf_write(file_path, data, sample_rate, **kwargs):
    # Just verify the file can be created
    with open(file_path, "wb") as f:
        f.write(b"test_wav_data")


class TestTtsEngine:
    """Test suite for the TTS Engine."""

    @pytest.fixture
    def tts_config(self):
        """Create a test configuration for the TTS engine."""
        return TTSConfig(
            model_name="test_model",
            vocoder_name=None,
            device="cpu",
            sample_rate=TEST_SAMPLE_RATE,
        )

    @pytest.fixture
    def tts_engine(self, tts_config):
        """Create a TTS engine instance for testing."""
        return TtsEngine(config=tts_config)

    @pytest.fixture
    @patch("torchaudio.load", side_effect=mock_torch_load)
    @patch("soundfile.write", side_effect=mock_sf_write)
    @patch("builtins.open", new_callable=mock_open)
    def test_initialization(self, mock_file, mock_sf, mock_ta, tts_config):
        """Test TTS engine initialization."""
        tts_engine = TtsEngine(config=tts_config)

        assert tts_engine.config == tts_config
        assert tts_engine.device == torch.device("cpu")
        assert tts_engine.num_emotions == 5  # Default number of emotions

    @patch("torchaudio.load", side_effect=mock_torch_load)
    @patch("soundfile.write", side_effect=mock_sf_write)
    def test_synthesize(self, mock_sf, mock_ta, tts_engine):
        """Test basic speech synthesis."""
        text = "This is a test."

        audio, sample_rate = tts_engine.synthesize(text)

        assert isinstance(audio, torch.Tensor)
        assert audio.dim() == 2  # [channels, samples]
        assert sample_rate == TEST_SAMPLE_RATE
        assert audio.size(1) > 0  # Non-empty audio

    @patch("torchaudio.load", side_effect=mock_torch_load)
    @patch("soundfile.write", side_effect=mock_sf_write)
    def test_emotion_control(self, mock_sf, mock_ta, tts_engine):
        """Test emotion control functionality."""
        text = "This is a test with emotion."

        # Test with different emotions
        for emotion_id in range(5):
            audio, _ = tts_engine.synthesize(text, emotion_id=emotion_id)
            assert isinstance(audio, torch.Tensor)
            assert audio.size(1) > 0

        # Test with invalid emotion ID (should default to neutral)
        audio, _ = tts_engine.synthesize(text, emotion_id=999)
        assert isinstance(audio, torch.Tensor)

    def test_save_load_weights(self, tts_engine, tmp_path):
        """Test saving and loading model weights."""
        # Create a temporary file
        weights_path = tmp_path / "model_weights.pth"

        # Save weights
        tts_engine.save_weights(str(weights_path))
        assert weights_path.exists()

        # Create a new instance and load weights
        with patch(
            "torch.load",
            return_value={
                "emotion_embedding.weight": tts_engine.emotion_embedding.weight
            },
        ):
            new_tts = TtsEngine()
            new_tts._load_weights(str(weights_path))

        # Verify the weights were loaded (just check the shape for the test)
        assert (
            tts_engine.emotion_embedding.weight.shape
            == new_tts.emotion_embedding.weight.shape
        )

    def test_available_emotions(self, tts_engine):
        """Test getting available emotions."""
        emotions = tts_engine.get_available_emotions()
        assert isinstance(emotions, dict)
        assert len(emotions) == 5  # Should have 5 default emotions
        assert all(isinstance(k, str) for k in emotions.keys())
        assert all(isinstance(v, int) for v in emotions.values())

        # Verify default emotion names
        expected_emotions = ["happy", "sad", "angry", "neutral", "surprise"]
        for emo in expected_emotions:
            assert emo in emotions
            assert isinstance(emotions[emo], int)
        assert emotions["neutral"] == 3  # Default emotion


if __name__ == "__main__":
    pytest.main(["-v", "test_tts_engine.py"])
