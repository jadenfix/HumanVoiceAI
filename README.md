# Cont-Time Emotion Voice Agent — Lean-Mac M2 Edition

## Project Overview
A real-time, emotion-adaptive voice agent optimized for MacBook M2 (8GB RAM) with minimal cloud GPU usage for fine-tuning. This project implements a lightweight yet powerful voice agent that can understand and respond with appropriate emotional tone.

## Features
- Real-time emotion detection from speech
- Adaptive emotional response generation
- Optimized for Apple Silicon M2 with < 1GB memory footprint
- End-to-end latency < 120ms
- Minimal cloud GPU usage (~$20 for fine-tuning)

## Quick Start
```bash
# Clone the repository
git clone https://github.com/jadenfix/HumanVoiceAI.git
cd HumanVoiceAI

# Set up the environment
make setup

# Run the demo
make demo
```

## System Requirements
- Apple Silicon M2 with 8GB+ unified memory
- macOS 14+
- Python 3.11
- Xcode Command Line Tools

## Installation
1. Install Homebrew if not already installed:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install system dependencies:
   ```bash
   brew install portaudio
   ```

3. Set up Python environment:
   ```bash
   pyenv install 3.11.0
   pyenv local 3.11.0
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
HumanVoiceAI/
├── data/                   # Datasets and voice samples
├── models/                 # Pretrained and fine-tuned models
├── src/                    # Source code
│   ├── audio/              # Audio processing
│   ├── emotion/            # Emotion detection
│   ├── tts/                # Text-to-speech
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── .gitignore
├── LICENSE
├── Makefile
├── pyproject.toml
└── README.md
```

## Training
For fine-tuning models on cloud GPU (costs ~$20):
```bash
make train-cloud
```

## Evaluation
Run the test suite:
```bash
make test
```

## License
MIT License

## Author
Jaden Fix
- GitHub: [jadenfix](https://github.com/jadenfix)
- LinkedIn: [jadenfix](https://linkedin.com/in/jadenfix)
- Email: jadenfix123@gmail.com

## Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Acknowledgments
- RAVDESS and CREMA-D datasets
- Coqui TTS and HiFi-GAN teams
- PyTorch and MPS team for Apple Silicon support
