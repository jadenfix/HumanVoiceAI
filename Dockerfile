# Use Python 3.11 slim as the base image
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.5.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e .

# Copy the rest of the application
COPY . .

# Create a non-root user
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
ENTRYPOINT ["python", "-m", "human_voice_ai"]
