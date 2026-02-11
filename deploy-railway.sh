#!/bin/bash
# Deploy ArcanaAI to Railway
# Run: bash deploy-railway.sh
set -e

echo "=== ArcanaAI Railway Deployment ==="
echo ""

# Check Railway CLI
if ! command -v railway &> /dev/null; then
    echo "ERROR: Railway CLI not installed. Run: brew install railway"
    exit 1
fi

# Check login
if ! railway whoami 2>/dev/null; then
    echo "Not logged in. Opening browser..."
    railway login
fi

echo ""
echo "Step 1: Creating Railway project..."
railway init --name arcana-ai

echo ""
echo "Step 2: Adding PostgreSQL..."
railway add --database postgres
echo "Waiting for PostgreSQL to provision..."
sleep 10

echo ""
echo "Step 3: Adding Redis..."
railway add --database redis
echo "Waiting for Redis to provision..."
sleep 10

echo ""
echo "Step 4: Linking to web service and deploying..."
# Railway auto-detects the Dockerfile and deploys
railway up --detach

echo ""
echo "Step 5: Setting environment variables..."
# Railway auto-injects DATABASE_URL and REDIS_URL from the addons
# We just need the OpenAI key
if [ -f .env ]; then
    OPENAI_KEY=$(grep OPENAI_API_KEY .env | cut -d'=' -f2)
    if [ -n "$OPENAI_KEY" ]; then
        railway variables set OPENAI_API_KEY="$OPENAI_KEY"
        echo "OPENAI_API_KEY set from .env"
    fi
fi

echo ""
echo "Step 6: Getting public URL..."
railway domain
echo ""

echo "=== Deployment complete! ==="
echo ""
echo "Useful commands:"
echo "  railway logs          # View logs"
echo "  railway status        # Check status"
echo "  railway open          # Open in browser"
echo "  railway variables     # View env vars"
