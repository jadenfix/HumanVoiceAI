# Environment Setup (Mac M2 + 8 GB RAM Only)

> All steps assume macOS 14+ on an Apple M2 with 8 GB unified memory.  
> We use only local resources—no cloud GPU.  

---

## 1. Install Homebrew (if you don't have it)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 2. Python & Virtual Environment

### Install Python 3.11
```bash
brew install pyenv
pyenv install 3.11.8
pyenv global 3.11.8
```

### Create & Activate venv
```bash
python -m venv .venv
source .venv/bin/activate
```

### Upgrade pip & install wheels
```bash
pip install --upgrade pip setuptools wheel
```

## 3. Install MPS-Accelerated PyTorch & Torchaudio

Apple's metal-backend gives ~3× CPU speed for small CNNs/UNets.

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

Verify:

```python
python - <<EOF
import torch
print("MPS available:", torch.backends.mps.is_available())
print("Torch version:", torch.__version__)
EOF
```

## 4. Audio I/O & Feature Tools

```bash
brew install portaudio
pip install sounddevice  # Python wrapper for PortAudio
```

You'll also need ffmpeg for any conversions:

```bash
brew install ffmpeg
```

## 5. Install TTS & Vocoder

We'll use Coqui's XTTS-v2-small (8-bit) and HiFi-GAN-tiny:

```bash
pip install TTS==0.22.0  # XTTS-v2-small + dependencies
pip install git+https://github.com/jik876/hifi-gan.git@v0.2.3  # for tiny vocoder
```

## 6. Install ML Utilities & Testing

```bash
pip install hydra-core==1.3.2  # config management
pip install wandb==0.15.0      # optional for local logging
pip install pytest==7.4.0
pip install black isort flake8  # linting
```

## 7. Clone & Install Your Repo

```bash
git clone https://github.com/jadenfix/HumanVoiceAI.git
cd HumanVoiceAI
pip install -e .
```

## 8. Verify Everything

Run a quick smoke script:

```python
python - <<EOF
from src.audio import FeatureExtractor
from src.emotion import SerModel
from src.tts import TtsEngine

fe = FeatureExtractor()
feats = fe("data/samples/test.wav")
model = SerModel()  # random stub
logits = model(feats)
tts = TtsEngine()
wav = tts.synthesize("Hello world", emotion="happy")
print("Feats:", feats.shape, "Logits:", logits.shape, "TTS WAV:", type(wav))
EOF
```

If you see shapes and no errors, your environment is green.
