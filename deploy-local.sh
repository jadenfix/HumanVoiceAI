#!/bin/bash

# Voice AI Local Deployment Script
# This script runs both the Streamlit backend and Next.js frontend

echo "🚀 Starting Voice AI Local Deployment"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [[ ! -z "$STREAMLIT_PID" ]]; then
        kill $STREAMLIT_PID 2>/dev/null
        echo -e "${GREEN}Streamlit backend stopped${NC}"
    fi
    if [[ ! -z "$NEXTJS_PID" ]]; then
        kill $NEXTJS_PID 2>/dev/null
        echo -e "${GREEN}Next.js frontend stopped${NC}"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if Python dependencies are installed
echo -e "${BLUE}Checking Python environment...${NC}"
if ! python -c "import streamlit" 2>/dev/null; then
    echo -e "${RED}Streamlit not found. Please install dependencies:${NC}"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Check if Node.js dependencies are installed
echo -e "${BLUE}Checking Node.js environment...${NC}"
if [ ! -d "human-voice-ai-frontend/node_modules" ]; then
    echo -e "${RED}Node.js dependencies not found. Please install them:${NC}"
    echo "cd human-voice-ai-frontend && npm install"
    exit 1
fi

# Check ports
echo -e "${BLUE}Checking ports...${NC}"
if ! check_port 8501; then
    echo "Please stop the service using port 8501 or use: pkill -f streamlit"
    exit 1
fi

if ! check_port 3000; then
    echo "Please stop the service using port 3000 or use: pkill -f next"
    exit 1
fi

# Start Streamlit backend
echo -e "${BLUE}Starting Streamlit backend...${NC}"
cd "$(dirname "$0")"
python -m streamlit run src/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

# Wait a moment for Streamlit to start
sleep 3

# Check if Streamlit started successfully
if ! ps -p $STREAMLIT_PID > /dev/null; then
    echo -e "${RED}Failed to start Streamlit backend${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Streamlit backend running on http://localhost:8501${NC}"

# Start Next.js frontend
echo -e "${BLUE}Starting Next.js frontend...${NC}"
cd human-voice-ai-frontend
npm run dev &
NEXTJS_PID=$!

# Wait a moment for Next.js to start
sleep 5

# Check if Next.js started successfully
if ! ps -p $NEXTJS_PID > /dev/null; then
    echo -e "${RED}Failed to start Next.js frontend${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}✓ Next.js frontend running on http://localhost:3000${NC}"

# Print status
echo ""
echo -e "${GREEN}🎉 Voice AI is now running locally!${NC}"
echo "=================================="
echo -e "📱 Frontend (Next.js):  ${BLUE}http://localhost:3000${NC}"
echo -e "🖥️  Backend (Streamlit): ${BLUE}http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and wait for user interrupt
wait 