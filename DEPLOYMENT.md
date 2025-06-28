# Voice AI Local Deployment Guide

This guide will help you deploy the Voice AI application locally with both the Streamlit backend and Next.js frontend.

## Prerequisites

- Python 3.11+ installed
- Node.js 18+ installed
- npm or yarn package manager

## Quick Start

### Option 1: Full Deployment (Recommended)

Run both the Streamlit backend and Next.js frontend:

```bash
./deploy-local.sh
```

This will start:
- **Streamlit Backend**: http://localhost:8501
- **Next.js Frontend**: http://localhost:3000

### Option 2: Individual Services

#### Backend Only (Streamlit)

```bash
streamlit run src/app.py --server.port 8501
```

#### Frontend Only (Next.js)

```bash
cd human-voice-ai-frontend
npm run dev
```

## Manual Setup

If you encounter issues with the deployment script, follow these manual steps:

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Install Node.js Dependencies

```bash
cd human-voice-ai-frontend
npm install
```

### 3. Start Services

Open two terminal windows:

**Terminal 1 - Backend:**
```bash
streamlit run src/app.py --server.port 8501
```

**Terminal 2 - Frontend:**
```bash
cd human-voice-ai-frontend
npm run dev
```

## Features

### Streamlit Interface (Port 8501)
- Real-time emotion detection
- Audio streaming capabilities
- Interactive voice analysis
- Emotion distribution charts
- Action buttons for different responses

### Next.js Frontend (Port 3000)
- Modern web interface
- Voice recording functionality
- Real-time emotion processing
- Chat-like interaction
- Responsive design

## Troubleshooting

### Port Already in Use
If you get a "port already in use" error:

```bash
# Kill processes on port 8501 (Streamlit)
pkill -f streamlit

# Kill processes on port 3000 (Next.js)
pkill -f next
```

### Permission Denied
Make sure the deployment script is executable:
```bash
chmod +x deploy-local.sh
```

### Missing Dependencies
If you get import errors, make sure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install -e .
cd human-voice-ai-frontend && npm install
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   Next.js App   │    │  Streamlit App  │
│   (Frontend)    │    │   (Backend)     │
│   Port: 3000    │    │   Port: 8501    │
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────────────────┘
              API Communication
```

## Development

- The Next.js app includes hot-reloading for development
- The Streamlit app will automatically reload when you change Python files
- Both services run concurrently and can be stopped with Ctrl+C

## Production Notes

This setup is for local development and testing. For production deployment, you would need to:

1. Build the Next.js app for production: `npm run build`
2. Configure proper environment variables
3. Set up reverse proxy (nginx)
4. Use a process manager (PM2, systemd)
5. Configure SSL/HTTPS

## Support

If you encounter any issues:
1. Check the terminal output for error messages
2. Verify all dependencies are installed
3. Ensure ports 3000 and 8501 are available
4. Check the logs in the terminal for debugging information 