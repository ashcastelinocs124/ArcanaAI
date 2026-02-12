# ArcanaAI — Deployment Guide

## Production URL

**Live:** https://backend-production-9992.up.railway.app

## Architecture (Production)

```
Railway Project: arcana-ai
├── backend (Django + WhiteNoise)  ← serves both API and frontend
├── PostgreSQL                      ← persistent data
└── Redis                           ← Celery broker + result backend
```

Single service serves everything:
- **Frontend** — WhiteNoise serves `frontend/` at root URL (`/`)
- **API v0** — `/api/*` (legacy endpoints: traces, workflow, optimizer, pipeline)
- **API v1** — `/api/v1/*` (traces CRUD, batch, evaluation, proxy)
- **Health** — `/health`

## Environment Variables (Railway)

| Variable | Source | Required |
|----------|--------|----------|
| `DATABASE_URL` | Railway PostgreSQL addon | Auto-injected |
| `REDIS_URL` | Railway Redis addon | Auto-injected |
| `CELERY_BROKER_URL` | Set manually | Yes (same as REDIS_URL) |
| `CELERY_RESULT_BACKEND` | Set manually | Yes (same as REDIS_URL) |
| `DJANGO_SECRET_KEY` | Set manually | Yes |
| `RAILWAY_ENVIRONMENT` | Railway | Auto-injected |

**Not stored on Railway** (user-provided at runtime via X-API-Key header):
- `OPENAI_API_KEY` — users provide their own key through the dashboard modal

## Deploy Commands

### Standard Deploy (from project root)

```bash
# 1. Run pre-flight checks
python manage.py test tests -v 0             # Backend tests
python -m py_compile backend/api/views_v1.py  # Compile check
node --check <(python3 -c "import re; open('/tmp/c.js','w').write('\n'.join(re.findall(r'<script[^>]*>(.*?)</script>',open('frontend/index.html').read(),re.DOTALL)))")  # JS syntax

# 2. Commit and push
git add <files>
git commit -m "description"
git push origin main

# 3. Deploy to Railway
railway up --detach

# 4. Verify (wait ~60s for build)
curl -s https://backend-production-9992.up.railway.app/health
curl -s https://backend-production-9992.up.railway.app/ | grep -q 'ArcanaAI' && echo "Frontend OK"
```

### Database Migrations

```bash
# Run migrations on Railway (if new migrations were added)
railway run python backend/manage.py migrate

# Check migration status
railway run python backend/manage.py showmigrations
```

### View Logs

```bash
railway logs              # Tail recent logs
railway logs | grep ERROR # Filter for errors
```

### Rollback

```bash
# Option 1: Revert commit and redeploy
git revert HEAD
git push origin main
railway up --detach

# Option 2: Use Railway dashboard to redeploy previous build
# https://railway.com/project/e49a3d10-3bca-4a1a-b2f6-e8f3593d9644
```

## Railway Project Details

| Field | Value |
|-------|-------|
| Project ID | `e49a3d10-3bca-4a1a-b2f6-e8f3593d9644` |
| Service ID | `04f78e00-170d-45b9-ac7c-a8e46dcf18fa` |
| Service Name | `backend` |
| Public Domain | `backend-production-9992.up.railway.app` |
| GitHub Repo | `https://github.com/ashcastelinocs124/ArcanaAI.git` |

## Build Configuration

Railway uses **Nixpacks** auto-detection:
- Detects Python from `requirements.txt`
- Start command: `cd backend && python manage.py runserver 0.0.0.0:$PORT`
- No Dockerfile or Procfile needed

### Key Settings in `backend/arcana/settings.py`

```python
DEBUG = False                              # Required for SSE streaming
CORS_ALLOW_ALL_ORIGINS = True              # API is public
CORS_ALLOW_HEADERS = [..., 'x-api-key']    # Custom auth header
WHITENOISE_ROOT = PROJECT_ROOT / 'frontend' # Serves frontend at /
```

## Post-Deploy Verification Script

```bash
python3 -c "
import urllib.request, json

BASE = 'https://backend-production-9992.up.railway.app'

# Health
r = urllib.request.urlopen(f'{BASE}/health')
h = json.loads(r.read())
print('Health:', 'PASS' if h['status'] == 'ok' else 'FAIL')

# Frontend
r = urllib.request.urlopen(f'{BASE}/')
html = r.read().decode()
print('Frontend:', 'PASS' if 'ArcanaAI' in html else 'FAIL')

# API v1
r = urllib.request.urlopen(f'{BASE}/api/v1/traces')
d = json.loads(r.read())
print('API v1:', 'PASS' if d['success'] else 'FAIL', f'({d[\"meta\"][\"total\"]} traces)')

# CORS
req = urllib.request.Request(f'{BASE}/api/v1/traces', method='OPTIONS')
req.add_header('Origin', 'http://localhost')
req.add_header('Access-Control-Request-Method', 'GET')
req.add_header('Access-Control-Request-Headers', 'x-api-key')
r = urllib.request.urlopen(req)
print('CORS:', 'PASS' if 'x-api-key' in r.headers.get('Access-Control-Allow-Headers','') else 'FAIL')
"
```

## Local Development vs Production

| Aspect | Local | Production |
|--------|-------|------------|
| Frontend URL | `localhost:5000` (Django) or `localhost:8000` (static) | `backend-production-9992.up.railway.app` |
| Backend URL | `localhost:5000` | Same domain (single service) |
| Database | SQLite (`db.sqlite3`) | PostgreSQL (Railway addon) |
| Redis | `localhost:6379` | Railway Redis addon |
| Static files | Django dev server | WhiteNoise |
| Debug mode | Can be True | Must be False (SSE breaks) |
| API keys | `.env` file | User-provided via dashboard |
| HTTPS | No | Yes (Railway enforced) |

## Troubleshooting

### "Failed to fetch" on API page
- **Cause:** CORS blocking `X-API-Key` header
- **Fix:** Ensure `CORS_ALLOW_HEADERS` includes `x-api-key` in settings.py

### SSE streaming hangs
- **Cause:** `DEBUG=True` — Werkzeug/Django middleware buffers response
- **Fix:** Ensure `DEBUG=False` in production settings

### Static files not updating
- **Cause:** WhiteNoise caches aggressively
- **Fix:** Clear browser cache, or add cache-busting query params

### Migrations not applied
- **Cause:** Railway doesn't auto-run migrations
- **Fix:** `railway run python backend/manage.py migrate`

### Build fails on Railway
- **Cause:** Usually missing dependency in requirements.txt
- **Fix:** Check build logs at the URL printed by `railway up`
