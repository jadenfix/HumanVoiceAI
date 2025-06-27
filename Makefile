.PHONY: setup clean test train-cloud demo lint format

# Project variables
PYTHON = python3.11
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest
PYTHON_SRC = src/
TESTS = tests/

# Default target
all: setup

# Set up the development environment
setup:
	@echo "Setting up development environment..."
	$(PYTHON) -m venv venv
	. venv/bin/activate && \
	$(PIP) install --upgrade pip setuptools wheel && \
	$(PIP) install -e ".[dev]"
	@echo "\nSetup complete! Activate the virtual environment with 'source venv/bin/activate'"

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

# Run tests
test:
	@echo "Running tests..."
	. venv/bin/activate && $(PYTEST) -v --cov=$(PYTHON_SRC) --cov-report=term-missing $(TESTS)

# Train models on cloud GPU
train-cloud:
	@echo "Starting cloud training..."
	@echo "Note: This will incur cloud costs (~$20)"
	. venv/bin/activate && python -m src.train.cloud_train

# Run the demo
demo:
	@echo "Starting voice agent demo..."
	. venv/bin/activate && python -m src.demo

# Lint the code
lint:
	@echo "Running linters..."
	. venv/bin/activate && \
	black --check $(PYTHON_SRC) && \
	isort --check-only $(PYTHON_SRC) && \
	flake8 $(PYTHON_SRC) && \
	mypy $(PYTHON_SRC)

# Format the code
format:
	@echo "Formatting code..."
	. venv/bin/activate && \
	black $(PYTHON_SRC) && \
	isort $(PYTHON_SRC)

# Create requirements.txt
requirements:
	. venv/bin/activate && \
	pip freeze > requirements.txt

# Help target
help:
	@echo "Available targets:"
	@echo "  setup     - Set up the development environment"
	@echo "  clean     - Remove build artifacts and caches"
	@echo "  test      - Run tests with coverage"
	@echo "  train-cloud - Train models on cloud GPU (~$20)"
	@echo "  demo      - Run the voice agent demo"
	@echo "  lint      - Check code style and type hints"
	@echo "  format    - Format the code"
	@echo "  requirements - Generate requirements.txt"
