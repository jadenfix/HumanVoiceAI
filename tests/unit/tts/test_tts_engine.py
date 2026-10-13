"""Actual synthetic WAV and embedding persistence tests; no speech-quality claim."""
import pytest
import torch
from unittest.mock import patch
from human_voice_ai.tts.tts_engine import TtsEngine, TTSConfig


class TestTtsEngine:
    @pytest.fixture
    def tts_config(self):
        return TTSConfig(model_name="test_model",allow_mock=True,device="cpu",sample_rate=16000)

    @pytest.fixture
    def tts_engine(self,tts_config):
        # Provider absence is explicit; mock mode writes an actual synthetic WAV.
        with patch("human_voice_ai.tts.tts_engine._tts_provider",side_effect=ImportError("fixture has no provider")):
            return TtsEngine(tts_config)

    def test_initialization(self,tts_engine,tts_config):
        assert tts_engine.config==tts_config
        assert tts_engine.device==torch.device("cpu")
        assert tts_engine.num_emotions==5

    def test_synthesize(self,tts_engine):
        audio,sample_rate=tts_engine.synthesize("Synthetic WAV plumbing")
        assert sample_rate==16000 and audio.shape[0]==1 and audio.shape[1]>0
        assert torch.isfinite(audio).all() and torch.count_nonzero(audio)>0

    def test_emotion_control(self,tts_engine):
        for emotion_id in [0,1,2,3,4,999]:
            audio,_=tts_engine.synthesize("Synthetic fixture",emotion_id=emotion_id)
            assert audio.shape[1]>0 and torch.isfinite(audio).all()

    def test_save_load_weights(self,tts_engine,tmp_path):
        path=tmp_path/'embeddings.pt';tts_engine.save_weights(str(path))
        with patch("human_voice_ai.tts.tts_engine._tts_provider",side_effect=ImportError("fixture has no provider")):
            restored=TtsEngine(TTSConfig(allow_mock=True,device='cpu'))
        restored._load_weights(str(path))
        torch.testing.assert_close(restored.emotion_embedding.weight,tts_engine.emotion_embedding.weight,rtol=0,atol=0)

    def test_available_emotions(self,tts_engine):
        emotions=tts_engine.get_available_emotions()
        assert set(emotions)=={'happy','sad','angry','neutral','surprise'}
        assert emotions['neutral']==3


def test_real_backend_failure_never_returns_synthetic_speech():
    with patch("human_voice_ai.tts.tts_engine._tts_provider",side_effect=ImportError("not installed")):
        with pytest.raises(RuntimeError,match="Real TTS initialization failed"):
            TtsEngine(TTSConfig(device="cpu"))


def test_synthesis_failure_never_returns_silence_as_success():
    with patch("human_voice_ai.tts.tts_engine._tts_provider",side_effect=ImportError("not installed")):
        engine=TtsEngine(TTSConfig(allow_mock=True,device="cpu"))
    with patch.object(engine.tts,"tts_to_file",side_effect=RuntimeError("backend failed")):
        with pytest.raises(RuntimeError,match="no speech audio"):
            engine.synthesize("test")
