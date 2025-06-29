# Cont-Time Emotion Voice Agent — Lean-Mac M2 Edition

## Project Overview
A real-time emotion-aware voice agent that adapts responses based on speech emotion recognition and reinforcement learning.

## Features

- Real-time speech emotion recognition
- Emotion-adaptive responses using reinforcement learning
- Interactive web interface with Streamlit
- Model interpretability with SHAP
- Support for Apple Silicon (M1/M2) acceleration

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/jadenfix/HumanVoiceAI.git
   cd HumanVoiceAI
   ```

2. Set up the environment:
   ```bash
   # Install system dependencies
   ./scripts/setup.sh
   
   # Activate virtual environment
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   
   # Install in development mode
   pip install -e .
   ```

3. Run the application:
   ```bash
   make demo
   # or
   python -m human_voice_ai
   ```

## Project Structure

```
voiceAI/
├── .github/                   # GitHub workflows
├── configs/                   # Configuration files
│   ├── app_config.yaml        # Application configuration
│   └── ser_config.yaml        # SER model configuration
├── data/                      # Data files
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── samples/               # Sample audio files
├── docs/                      # Documentation
├── logs/                      # Log files
├── models/                    # Trained models
│   └── checkpoints/           # Model checkpoints
├── notebooks/                 # Jupyter notebooks
├── scripts/                   # Utility and training scripts
│   ├── setup.sh               # Setup script
│   └── train/                 # Training scripts
├── src/                       # Source code
│   └── human_voice_ai/        # Main package
│       ├── __init__.py
│       ├── __main__.py        # CLI entry point
│       ├── audio/             # Audio processing
│       ├── emotion/           # Emotion recognition
│       ├── interpretability/  # Model interpretation
│       ├── policy/            # RL policies
│       ├── tts/               # Text-to-speech
│       └── utils/             # Utility functions
├── tests/                     # Tests
│   ├── integration/           # Integration tests
│   └── unit/                  # Unit tests
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Development

### Setting Up Development Environment

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

2. Run tests:
   ```bash
   make test
   ```

3. Run linter:
   ```bash
   make lint
   ```

4. Format code:
   ```bash
   make format
   ```

## Training Models

### Speech Emotion Recognition (SER)

```bash
python scripts/train/train_emotion.py --config configs/ser_config.yaml
```

### Text-to-Speech (TTS)

```bash
python scripts/train/train_tts.py --config configs/tts_config.yaml
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Jaden Fix
- GitHub: [jadenfix](https://github.com/jadenfix)
- LinkedIn: [jadenfix](https://www.linkedin.com/in/jadenfix)
- Email: jadenfix123@gmail.com

## Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

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

## Evaluation
Run the test suite:
```bash
make test
```

## Acknowledgments
- RAVDESS and CREMA-D datasets
- Coqui TTS and HiFi-GAN teams
- PyTorch and MPS team for Apple Silicon support
# HumanVoiceAI
