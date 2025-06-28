.PHONY: all setup clean test train-cloud demo lint format requirements help install-hooks

# Project variables
PYTHON = python3.11
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest
PYTHON_SRC = src/
TESTS = tests/
SCRIPTS = scripts/
CONFIGS = configs/

# Default target
all: help

# Set up the development environment
setup:
	@echo "🚀 Setting up development environment..."
	$(PYTHON) -m venv venv
	. venv/bin/activate && \
	$(PIP) install --upgrade pip setuptools wheel && \
	$(PIP) install -e ".[dev]"
	@echo "\n✅ Setup complete! Activate the virtual environment with 'source venv/bin/activate'"

# Install pre-commit hooks
install-hooks:
	@echo "🔧 Installing pre-commit hooks..."
	. venv/bin/activate && pre-commit install

# Clean up build artifacts
clean:
	@echo "🧹 Cleaning up..."
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/ .benchmarks/
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

# Run tests
test:
	@echo "🧪 Running tests..."
	. venv/bin/activate && $(PYTEST) -v --cov=$(PYTHON_SRC) --cov-report=term-missing $(TESTS)

# Train models
train: train-emotion train-tts

# Train emotion recognition model
train-emotion:
	@echo "🤖 Training emotion recognition model..."
	. venv/bin/activate && \
	python $(SCRIPTS)train/train_emotion.py --config $(CONFIGS)ser_config.yaml

# Train TTS model
train-tts:
	@echo "🔊 Training TTS model..."
	. venv/bin/activate && \
	python $(SCRIPTS)train/train_tts.py --config $(CONFIGS)tts_config.yaml

# Run the demo application
demo:
	@echo "🎤 Starting voice agent demo..."
	. venv/bin/activate && \
	python -m human_voice_ai

# Lint the code
lint:
	@echo "✨ Running linters..."
	. venv/bin/activate && \
	black --check $(PYTHON_SRC) $(TESTS) $(SCRIPTS) && \
	isort --check-only $(PYTHON_SRC) $(TESTS) $(SCRIPTS) && \
	flake8 $(PYTHON_SRC) $(TESTS) $(SCRIPTS) && \
	mypy $(PYTHON_SRC) $(SCRIPTS)

# Format the code
format:
	@echo "🎨 Formatting code..."
	. venv/bin/activate && \
	black $(PYTHON_SRC) $(TESTS) $(SCRIPTS) && \
	isort $(PYTHON_SRC) $(TESTS) $(SCRIPTS)

# Generate requirements.txt
requirements:
	@echo "📋 Generating requirements.txt..."
	. venv/bin/activate && \
	pip freeze | grep -v "human-voice-ai" > requirements.txt

# Help target
help:
	@echo "\n🤖 Human Voice AI - Available targets:\n"
	@echo "  setup          - Set up the development environment"
	@echo "  install-hooks  - Install pre-commit hooks"
	@echo "  clean          - Remove build artifacts and caches"
	@echo "  test           - Run tests with coverage"
	@echo "  train          - Train all models (emotion + TTS)"
	@echo "  train-emotion  - Train emotion recognition model"
	@echo "  train-tts      - Train text-to-speech model"
	@echo "  demo           - Run the voice agent demo"
	@echo "  lint           - Check code style and type hints"
	@echo "  format         - Format the code"
	@echo "  requirements   - Generate requirements.txt"
	@echo "  help           - Show this help message\n"
	@echo "📝 For more information, see the README.md file.\n"
