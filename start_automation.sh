#!/bin/bash

# Automation Controller Startup Script
# This script starts the Flask server and opens the web interface

echo "🚀 Starting Automation Controller..."

# Change to the project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Kill any existing processes on port 8080
echo "🧹 Cleaning up any existing processes on port 8080..."
lsof -ti:8080 | xargs kill -9 2>/dev/null || true

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
if ! python -c "import flask, mss, cv2" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Start the server in background
echo "🌐 Starting Flask server..."
PYTHONPATH=src python -m automation.web_app &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8080 >/dev/null 2>&1; then
        echo "✅ Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Server failed to start within 30 seconds"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Open the web page
echo "🌍 Opening web interface..."
if command -v open >/dev/null 2>&1; then
    # macOS
    open http://localhost:8080
elif command -v xdg-open >/dev/null 2>&1; then
    # Linux
    xdg-open http://localhost:8080
elif command -v start >/dev/null 2>&1; then
    # Windows (Git Bash)
    start http://localhost:8080
else
    echo "🔗 Please open http://localhost:8080 in your browser"
fi

echo ""
echo "🎮 Automation Controller is running!"
echo "📍 Web Interface: http://localhost:8080"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Wait for the server process and handle Ctrl+C
trap 'echo ""; echo "🛑 Stopping server..."; kill $SERVER_PID 2>/dev/null; exit 0' INT

wait $SERVER_PID
