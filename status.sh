#!/bin/bash

# Arcana Status Check Script

echo "🔍 Arcana System Status"
echo "======================="
echo ""

# Check backend
echo "Backend API (http://localhost:5000):"
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:5000/health | jq -r '.status' 2>/dev/null || echo "unknown")
    echo "  ✅ Running (status: $HEALTH)"
    if [ -f /tmp/arcana-backend.pid ]; then
        PID=$(cat /tmp/arcana-backend.pid)
        echo "     PID: $PID"
    fi
else
    echo "  ❌ Not running"
fi

echo ""

# Check frontend
echo "Frontend Server (http://localhost:8000):"
if curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo "  ✅ Running"
    if [ -f /tmp/arcana-frontend.pid ]; then
        PID=$(cat /tmp/arcana-frontend.pid)
        echo "     PID: $PID"
    fi
else
    echo "  ❌ Not running"
fi

echo ""

# Check environment
echo "Environment:"
if [ -f .env ]; then
    if grep -q "OPENAI_API_KEY" .env 2>/dev/null; then
        echo "  ✅ .env file exists with API key"
    else
        echo "  ⚠️  .env file exists but no OPENAI_API_KEY found"
    fi
else
    echo "  ⚠️  No .env file (create one for LLM features)"
fi

echo ""

# Check Python packages
echo "Python Dependencies:"
if python3 -c "import flask" &> /dev/null; then
    echo "  ✅ Flask installed"
else
    echo "  ❌ Flask not installed (run: pip install -r requirements.txt)"
fi

if python3 -c "import litellm" &> /dev/null; then
    echo "  ✅ LiteLLM installed"
else
    echo "  ❌ LiteLLM not installed (run: pip install -r requirements.txt)"
fi

echo ""

# Log files
echo "Logs:"
if [ -f /tmp/arcana-backend.log ]; then
    BACKEND_LINES=$(wc -l < /tmp/arcana-backend.log)
    echo "  📄 Backend:  /tmp/arcana-backend.log ($BACKEND_LINES lines)"
else
    echo "  📄 Backend:  No log file"
fi

if [ -f /tmp/arcana-frontend.log ]; then
    FRONTEND_LINES=$(wc -l < /tmp/arcana-frontend.log)
    echo "  📄 Frontend: /tmp/arcana-frontend.log ($FRONTEND_LINES lines)"
else
    echo "  📄 Frontend: No log file"
fi

echo ""
echo "Quick Actions:"
echo "  Start:  ./start.sh"
echo "  Stop:   ./stop.sh"
echo "  Status: ./status.sh"
echo ""
