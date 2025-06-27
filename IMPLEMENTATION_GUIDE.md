# Implementation Guide  
*Step-by-step tasks to get from empty scaffold → local demo*

---

## 1. Project Scaffolding

```bash
voice-agent/
├── data/
│   └── samples/
├── src/
│   ├── audio/      # FeatureExtractor
│   ├── emotion/    # SerModel + policy
│   ├── tts/        # TtsEngine
│   └── vocoder/    # HiFi-GAN wrapper
├── tests/          # pytest suites
├── demo.py         # E2E entrypoint
├── Makefile
├── pyproject.toml
└── README.md
```

## 2. FeatureExtractor (src/audio/feature_extractor.py)

```python
import torch
import torchaudio
import torchaudio.functional as F

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Mel spectrogram transform
        self.mel_scale = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
    
    def extract(self, wav_path: str) -> torch.Tensor:
        # Load audio
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            waveform = F.resample(waveform, sample_rate, self.sample_rate)
        
        # Convert to mono
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract features
        melspec = self.mel_scale(waveform)
        
        # Log scale
        melspec = torch.log(torch.clamp(melspec, min=1e-5))
        
        return melspec.squeeze(0).T  # [T, n_mels]
```

## 3. SerModel (src/emotion/ser_model.py)

```python
import torch
import torch.nn as nn

class SerModel(nn.Module):
    def __init__(self, num_emotions=6, input_dim=80, hidden_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, 
                           batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim, num_emotions)
    
    def forward(self, x):
        # x: [batch, time, features]
        x = x.transpose(1, 2)  # [batch, features, time]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [batch, time, features]
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classifier
        logits = self.classifier(x)
        return logits
```

## 4. TTS Engine (src/tts/tts_engine.py)

```python
from TTS.api import TTS
import numpy as np

class TtsEngine:
    def __init__(self, model_name="tts_models/en/xtts_v2_8bit"):
        self.tts = TTS(model_name=model_name, progress_bar=False)
        
    def synthesize(self, text: str, emotion: str = "neutral"):
        """Convert text to speech with specified emotion"""
        # Simple emotion mapping
        emotion_to_speaker = {
            "happy": "happy",
            "sad": "sad",
            "angry": "angry",
            "fearful": "fearful",
            "neutral": "neutral",
            "calm": "calm"
        }
        
        # Use a neutral voice for synthesis
        speaker_wav = "data/samples/neutral.wav"  # Pre-recorded neutral voice
        
        # Synthesize
        wav = self.tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            emotion=emotion_to_speaker.get(emotion.lower(), "neutral"),
            language="en"
        )
        
        return np.array(wav)
```

## 5. Vocoder (src/vocoder/hifigan.py)

```python
import torch
from hifi_gan import load_model
import numpy as np

class Vocoder:
    def __init__(self, model_path=None):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = load_model(model_path).to(self.device)
        self.model.eval()
    
    def vocode(self, mel_spectrogram):
        """Convert mel spectrogram to waveform"""
        with torch.no_grad():
            mel = torch.FloatTensor(mel_spectrogram).unsqueeze(0).to(self.device)
            audio = self.model(mel).cpu().numpy()
        return audio.squeeze()
```

## 6. Main Demo (demo.py)

```python
import time
import argparse
import sounddevice as sd
import numpy as np
from pathlib import Path

from src.audio.feature_extractor import FeatureExtractor
from src.emotion.ser_model import SerModel
from src.tts.tts_engine import TtsEngine
from src.vocoder.hifigan import Vocoder

def main(text, sample_rate=22050):
    print("Initializing components...")
    fe = FeatureExtractor()
    model = SerModel()
    tts = TtsEngine()
    vocoder = Vocoder()
    
    # Dummy audio capture (replace with real audio input)
    test_wav = "data/samples/test.wav"
    
    print("Processing audio...")
    # 1. Extract features
    feats = fe.extract(test_wav)
    
    # 2. Predict emotion
    with torch.no_grad():
        logits = model(feats.unsqueeze(0))
        emotion_id = logits.argmax().item()
    
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful"]
    detected_emotion = emotions[emotion_id]
    
    print(f"Detected emotion: {detected_emotion}")
    print(f"Generating speech for: {text}")
    
    # 3. Generate speech with detected emotion
    audio = tts.synthesize(text, emotion=detected_emotion)
    
    # 4. Play audio
    print("Playing...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Voice Agent")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the emotion voice agent.")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    args = parser.parse_args()
    
    main(args.text, args.sr)
```

## 7. Testing

Create test files in `tests/`:
- `test_feature_extractor.py`
- `test_ser_model.py`
- `test_tts_engine.py`

Example test:

```python
# tests/test_feature_extractor.py
def test_feature_extractor():
    fe = FeatureExtractor()
    test_wav = "data/samples/test.wav"
    feats = fe.extract(test_wav)
    assert feats.dim() == 2  # [time, n_mels]
    assert feats.shape[1] == 80  # 80 mel bands
```

## 8. Running the Demo

```bash
# Install in development mode
pip install -e .

# Run demo with default text
python demo.py

# Or specify custom text
python demo.py --text "This is a test of the emotion voice agent."
```
