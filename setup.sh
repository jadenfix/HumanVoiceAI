#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up Emotion-Aware Voice Agent..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "🔄 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
brew install portaudio python@3.11

# Create and activate virtual environment
echo "🐍 Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install Python dependencies
echo "🔧 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with MPS (Metal Performance Shaders) support
# This is optimized for Apple Silicon
echo "⚙️ Installing PyTorch with MPS support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Create necessary directories
echo "📂 Creating necessary directories..."
mkdir -p models data/processed data/raw logs

# Download sample data (if available)
# echo "📥 Downloading sample data..."
# curl -L -o data/raw/sample_audio.wav https://example.com/sample_audio.wav

echo "✨ Setup complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To start the application, run: streamlit run app.py"

# Make the setup script executable
chmod +x setup.sh
