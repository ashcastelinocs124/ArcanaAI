#!/bin/bash

# Arcana Stop Script
# Stops both backend API server and frontend web server

echo "🛑 Stopping Arcana servers..."

# Kill backend
if [ -f /tmp/arcana-backend.pid ]; then
    BACKEND_PID=$(cat /tmp/arcana-backend.pid)
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "✅ Backend server stopped (PID: $BACKEND_PID)"
    fi
    rm /tmp/arcana-backend.pid
fi

# Kill frontend
if [ -f /tmp/arcana-frontend.pid ]; then
    FRONTEND_PID=$(cat /tmp/arcana-frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "✅ Frontend server stopped (PID: $FRONTEND_PID)"
    fi
    rm /tmp/arcana-frontend.pid
fi

# Cleanup any remaining processes on the ports
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo ""
echo "✨ All Arcana servers stopped"
