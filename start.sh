#!/bin/bash

# Arcana Startup Script
# Starts both backend API server and frontend web server

set -e

echo "🚀 Starting Arcana LLM Observability Platform..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import flask" &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Kill any existing processes on ports 5000 and 8000
echo "🧹 Cleaning up existing processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

# Start backend API server
echo "🔧 Starting backend API server on http://localhost:5000..."
nohup python3 backend/api.py > /tmp/arcana-backend.log 2>&1 &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 2

# Check if backend is running
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Backend API server is running"
else
    echo "❌ Backend failed to start. Check /tmp/arcana-backend.log"
    exit 1
fi

# Start frontend web server
echo "🌐 Starting frontend server on http://localhost:8000..."
cd frontend
nohup python3 -m http.server 8000 > /tmp/arcana-frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "   Frontend PID: $FRONTEND_PID"

sleep 1

# Check if frontend is running
if curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo "✅ Frontend server is running"
else
    echo "❌ Frontend failed to start. Check /tmp/arcana-frontend.log"
    exit 1
fi

echo ""
echo "✨ Arcana is ready!"
echo ""
echo "📊 Dashboard: http://localhost:8000"
echo "🔌 API:       http://localhost:5000"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f /tmp/arcana-backend.log"
echo "   Frontend: tail -f /tmp/arcana-frontend.log"
echo ""
echo "🛑 To stop: ./stop.sh"
echo ""

# Save PIDs for stop script
echo "$BACKEND_PID" > /tmp/arcana-backend.pid
echo "$FRONTEND_PID" > /tmp/arcana-frontend.pid

# Open browser
if command -v open &> /dev/null; then
    echo "🌍 Opening browser..."
    sleep 1
    open http://localhost:8000
elif command -v xdg-open &> /dev/null; then
    echo "🌍 Opening browser..."
    sleep 1
    xdg-open http://localhost:8000
fi

echo ""
echo "Press Ctrl+C to view this message again, or run ./stop.sh to stop all servers"
