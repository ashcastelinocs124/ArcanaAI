# Arcana Localhost Configuration

## Architecture Overview

Arcana runs as two separate servers on localhost:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Browser (http://localhost:8000)               │
│  ┌───────────────────────────────────────────┐ │
│  │  Frontend Dashboard (HTML/CSS/JS)         │ │
│  │  - Single-page application                │ │
│  │  - All UI rendering                       │ │
│  │  - Makes API calls to backend             │ │
│  └───────────────────────────────────────────┘ │
│           ▲                                     │
│           │ CORS Requests                      │
│           ▼                                     │
│  ┌───────────────────────────────────────────┐ │
│  │  Backend API (http://localhost:5000)      │ │
│  │  - Flask REST API                         │ │
│  │  - Semantic pipeline analysis             │ │
│  │  - Prompt optimization                    │ │
│  │  - LLM integration (OpenAI)               │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Server Configuration

### Frontend Server (Port 8000)

- **Technology**: Python built-in HTTP server
- **Purpose**: Serves static HTML/CSS/JS files
- **URL**: http://localhost:8000
- **Files**: frontend/index.html (single file)
- **Start**: `cd frontend && python3 -m http.server 8000`

### Backend Server (Port 5000)

- **Technology**: Flask with Flask-CORS
- **Purpose**: REST API for data processing
- **URL**: http://localhost:5000
- **Files**: backend/api.py
- **Start**: `python3 backend/api.py`

## API Endpoints

All backend endpoints are accessible at `http://localhost:5000/api/*`

### 1. Health Check
```
GET http://localhost:5000/health
```

### 2. Pipeline Analysis
```
POST http://localhost:5000/api/pipeline/analyze
Content-Type: application/json

{
  "traces": [...],
  "threshold": 0.8
}
```

### 3. Prompt Optimization
```
POST http://localhost:5000/api/optimizer/run
Content-Type: multipart/form-data

file: <Excel/CSV file>
prompt_template: "Your prompt with {input}"
target_score: 0.85
max_iters: 3
```

### 4. Custom Evaluations
```
POST http://localhost:5000/api/evaluations/run
Content-Type: application/json

{
  "traces": [...],
  "eval_type": "accuracy",
  "eval_prompt": "..."
}
```

### 5. Workflow Execution
```
POST http://localhost:5000/api/workflow/run
Content-Type: application/json

{
  "goal": "Book a flight...",
  "workflow_type": "trip_booking"
}
```

## CORS Configuration

The backend has CORS enabled to allow requests from the frontend:

```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Allows all origins
```

This enables the frontend (localhost:8000) to make requests to the backend (localhost:5000).

## Network Flow

1. **User opens browser** → http://localhost:8000
2. **Browser loads** → frontend/index.html from frontend server
3. **JavaScript executes** → Makes fetch() calls to http://localhost:5000/api/*
4. **Flask receives** → Processes request, runs Python code
5. **Flask responds** → JSON data back to frontend
6. **Frontend updates** → Renders data in UI

## Testing Connectivity

### Check Frontend
```bash
curl http://localhost:8000
# Should return HTML content
```

### Check Backend
```bash
curl http://localhost:5000/health
# Should return: {"status":"ok","service":"arcana-api","version":"1.0.0"}
```

### Test Full Stack
```bash
# In browser console (http://localhost:8000)
fetch('http://localhost:5000/health')
  .then(r => r.json())
  .then(console.log)

# Should print: {status: "ok", service: "arcana-api", version: "1.0.0"}
```

## Troubleshooting

### Port Already in Use

```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### CORS Errors

If you see CORS errors in browser console:

1. Make sure Flask-CORS is installed: `pip install flask-cors`
2. Restart backend server
3. Clear browser cache
4. Access frontend via http://localhost:8000 (not file://)

### Connection Refused

If frontend can't reach backend:

1. Verify backend is running: `curl http://localhost:5000/health`
2. Check backend logs: `tail -f /tmp/arcana-backend.log`
3. Restart backend: `./stop.sh && ./start.sh`

### File Not Found (404)

If frontend returns 404:

1. Make sure you're accessing http://localhost:8000 (with http://)
2. Check frontend is serving from correct directory
3. Verify frontend/index.html exists

## Security Notes

⚠️ **Development Only**: This setup is for local development only.

- Both servers run in debug mode
- No authentication or authorization
- CORS allows all origins
- Runs on HTTP (not HTTPS)
- Not suitable for production

For production deployment:
- Use a production WSGI server (gunicorn, uwsgi)
- Configure specific CORS origins
- Add authentication (JWT, OAuth)
- Use HTTPS with SSL certificates
- Add rate limiting and API key management

## File Locations

```
/Users/ash/Desktop/interview-prep/
├── frontend/
│   └── index.html          # Frontend SPA (served on :8000)
├── backend/
│   └── api.py             # Backend API (served on :5000)
├── start.sh               # Start both servers
├── stop.sh                # Stop both servers
├── status.sh              # Check server status
└── .env                   # API keys (OPENAI_API_KEY)
```

## Environment Variables

The backend reads from `.env`:

```bash
# .env file
OPENAI_API_KEY=sk-...
```

Used by:
- Prompt Optimizer (for LLM calls)
- Task Progress Monitor (for checkpoint evaluation)
- Custom Evaluations (optional)

## Logs

Server logs are written to:
- Backend: `/tmp/arcana-backend.log`
- Frontend: `/tmp/arcana-frontend.log`

View in real-time:
```bash
tail -f /tmp/arcana-backend.log
tail -f /tmp/arcana-frontend.log
```

## Quick Commands

```bash
# Start everything
./start.sh

# Stop everything
./stop.sh

# Check status
./status.sh

# View backend logs
tail -f /tmp/arcana-backend.log

# View frontend logs
tail -f /tmp/arcana-frontend.log

# Test backend health
curl http://localhost:5000/health

# Test frontend
curl http://localhost:8000 | head -20
```
