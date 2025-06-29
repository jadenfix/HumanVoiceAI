#!/bin/bash

# Voice AI Stable Deployment Script
# Optimized for Mac M2 with 8GB RAM

set -e

echo "🚀 Voice AI Stable Deployment"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    print_warning "Shutting down services..."
    if [ ! -z "$STREAMLIT_PID" ] && ps -p $STREAMLIT_PID > /dev/null; then
        kill $STREAMLIT_PID 2>/dev/null
        print_status "Streamlit backend stopped"
    fi
    if [ ! -z "$NEXTJS_PID" ] && ps -p $NEXTJS_PID > /dev/null; then
        kill $NEXTJS_PID 2>/dev/null
        print_status "Next.js frontend stopped"
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Check system requirements
print_status "Checking system requirements..."

# Python check
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

# Node.js check
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed"
    exit 1
fi

# Check available memory (Mac specific)
AVAILABLE_MEMORY=$(sysctl -n hw.memsize)
AVAILABLE_MEMORY_GB=$((AVAILABLE_MEMORY / 1024 / 1024 / 1024))
print_status "Available RAM: ${AVAILABLE_MEMORY_GB}GB"

if [ $AVAILABLE_MEMORY_GB -lt 8 ]; then
    print_warning "Less than 8GB RAM available. Performance may be affected."
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt > /dev/null 2>&1
    print_success "Python dependencies installed"
else
    print_warning "requirements.txt not found"
fi

# Install Node.js dependencies
print_status "Installing Node.js dependencies..."
cd human-voice-ai-frontend
if [ -f "package.json" ]; then
    npm install > /dev/null 2>&1
    print_success "Node.js dependencies installed"
else
    print_error "package.json not found in human-voice-ai-frontend"
    exit 1
fi
cd ..

# Check ports
print_status "Checking ports..."
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null ; then
    print_warning "Port 8501 is already in use"
    lsof -ti:8501 | xargs kill -9 2>/dev/null || true
    print_status "Cleared port 8501"
fi

if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null ; then
    print_warning "Port 3000 is already in use"
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
    print_status "Cleared port 3000"
fi

# Create logs directory
mkdir -p logs

# Start Streamlit backend with stable app
print_status "Starting Streamlit backend (Stable Voice AI)..."
streamlit run src/stable_voice_ai.py \
    --server.port 8501 \
    --server.headless true \
    --server.address localhost \
    --server.fileWatcherType none \
    --server.runOnSave false \
    > logs/streamlit.log 2>&1 &

STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 5

if ps -p $STREAMLIT_PID > /dev/null; then
    print_success "Streamlit backend running on http://localhost:8501 (PID: $STREAMLIT_PID)"
else
    print_error "Failed to start Streamlit backend"
    cat logs/streamlit.log
    exit 1
fi

# Start Next.js frontend
print_status "Starting Next.js frontend..."
cd human-voice-ai-frontend
npm run dev > ../logs/nextjs.log 2>&1 &
NEXTJS_PID=$!
cd ..

# Wait for Next.js to start
sleep 10

if ps -p $NEXTJS_PID > /dev/null; then
    print_success "Next.js frontend running on http://localhost:3000 (PID: $NEXTJS_PID)"
else
    print_error "Failed to start Next.js frontend"
    cat logs/nextjs.log
    cleanup
    exit 1
fi

# Display success message
echo ""
echo "🎉 Voice AI Stable Deployment Complete!"
echo "========================================"
echo ""
echo "🔗 Services:"
echo "   • Stable Voice AI (Streamlit): http://localhost:8501"
echo "   • Modern Frontend (Next.js):   http://localhost:3000"
echo ""
echo "📊 System Status:"
echo "   • Memory Usage: Optimized for 8GB RAM"
echo "   • Audio Processing: File-based (stable)"
echo "   • Emotion Detection: Pre-trained models"
echo ""
echo "💡 Tips:"
echo "   • Use the Streamlit app for stable voice analysis"
echo "   • The Next.js frontend provides a modern interface"
echo "   • Press Ctrl+C to stop all services"
echo ""
echo "🎤 Ready for voice emotion detection!"

# Keep script running
print_status "Services running. Press Ctrl+C to stop..."
wait 